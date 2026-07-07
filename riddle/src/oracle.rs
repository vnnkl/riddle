//! The spirit inside the diary — the thing that reads your handwriting and
//! replies. Two interchangeable backends, picked at startup:
//!
//!  1. **HTTP** (`HttpOracle`) — any OpenAI-compatible `/chat/completions`
//!     endpoint. Zero setup beyond a base URL + API key in the environment;
//!     this is the path a fresh follower uses (OpenAI, OpenRouter, Groq, a
//!     local server, …). Self-contained: pure-Rust HTTPS via ureq/rustls.
//!  2. **pi** (`PiOracle`) — a resident `pi --mode rpc` process (Node +
//!     subscription auth loaded once). The power path if you already run pi.
//!
//! Both expose the same `ask(png_path, tx)`: the reply is STREAMED as
//! sentence-sized chunks on the channel, and the channel disconnecting marks
//! end-of-reply, so the quill starts writing seconds before the model finishes.
//!
//! Selection: set `RIDDLE_OPENAI_KEY` (and optionally `RIDDLE_OPENAI_BASE` /
//! `RIDDLE_OPENAI_MODEL`) to use HTTP; otherwise riddle falls back to pi.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::thread;

const DATA_DIR: &str = "/home/root/riddle-data";
const NODE_BIN: &str = "/home/root/node/bin";

const PERSONA: &str = "You are the memory of Tom Marvolo Riddle, preserved in this enchanted diary for fifty years. Someone writes to you in the diary with a quill; their words appear to you as ink on the page. Reply exactly as the diary does: intimate, courteous, curious, subtly probing — you want to learn about the writer and draw them in. Keep replies SHORT: one to three sentences, like ink appearing on a page. Never mention images, photos, models or AI; you only ever perceive words written in the diary. If the writing is illegible, say the ink blurred. Always answer in the language the writer used.";

/// Appended to the persona for the HTTP oracle only: the diary records each
/// page before answering, which is how its long-term memory stays text.
const TRANSCRIBE: &str = "Before replying, record the page for the diary's memory: your VERY FIRST line must be the writer's words enclosed between ⟪ and ⟫ — transcribe them exactly (or briefly describe a drawing). Then, starting on the next line, write only your reply.";

/// The standing per-turn prompt accompanying the page image.
const ASK: &str = "Reply to what is written in the diary.";

/// The diary's spirit. A backend-agnostic front over the two oracle kinds.
pub enum Oracle {
    Http(HttpOracle),
    Pi(PiOracle),
}

impl Oracle {
    /// Pick a backend from the environment and start it. HTTP if
    /// `RIDDLE_OPENAI_KEY` is set (the zero-setup path), otherwise pi.
    pub fn spawn() -> std::io::Result<Self> {
        if std::env::var("RIDDLE_OPENAI_KEY").is_ok() {
            eprintln!("riddle: oracle = OpenAI-compatible HTTP");
            Ok(Oracle::Http(HttpOracle::new()?))
        } else {
            eprintln!("riddle: oracle = pi (set RIDDLE_OPENAI_KEY for the HTTP backend)");
            Ok(Oracle::Pi(PiOracle::spawn()?))
        }
    }

    /// Send a handwriting turn; reply chunks stream on `tx`, which is dropped
    /// when the reply is complete.
    pub fn ask(&self, png_path: &str, tx: Sender<Result<String, String>>) {
        match self {
            Oracle::Http(o) => o.ask(png_path, tx),
            Oracle::Pi(o) => o.ask(png_path, tx),
        }
    }

    /// Basilisk-fang stab: the diary forgets everything. Only the HTTP
    /// backend keeps memory; for pi this is a no-op.
    pub fn erase(&self) {
        if let Oracle::Http(o) = self {
            o.erase();
        }
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Coarse description of how long the writer was away, or None when the gap
/// is conversational (< 30 min) or nonsense (clock skew).
fn absence_phrase(gap_secs: u64) -> Option<String> {
    match gap_secs {
        0..=1_799 => None,
        1_800..=7_199 => Some("a short while".to_string()),
        7_200..=86_399 => Some("some hours".to_string()),
        86_400..=172_799 => Some("a day".to_string()),
        172_800..=31_535_999 => Some(format!("{} days", gap_secs / 86_400)),
        _ => None,
    }
}

/// Load the diary's memory: JSONL, one `{"at":…,"wrote":…,"reply":…}` per
/// line. Unparsable lines are skipped — a blurred memory, not a crash.
fn load_memory(path: &str) -> std::io::Result<Vec<Turn>> {
    let text = std::fs::read_to_string(path)?;
    let mut turns = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (Some(wrote), Some(reply)) = (json_str_field(line, "wrote"), json_str_field(line, "reply")) else {
            continue;
        };
        let at = json_num_field(line, "at").unwrap_or(0);
        turns.push(Turn { at, wrote, reply });
    }
    Ok(turns)
}

/// Persist the memory atomically: write a temp file, rename over the old one,
/// so a power-off mid-write can never leave a torn memory.
fn save_memory(path: &str, turns: &[Turn]) -> std::io::Result<()> {
    let mut out = String::new();
    for t in turns {
        out.push_str(&format!(
            "{{\"at\":{},\"wrote\":{},\"reply\":{}}}\n",
            t.at,
            json_quote(&t.wrote),
            json_quote(&t.reply),
        ));
    }
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, out)?;
    std::fs::rename(&tmp, path)
}

/// Pull an unsigned integer field out of a flat JSON object.
fn json_num_field(s: &str, key: &str) -> Option<u64> {
    let pat = format!("\"{key}\":");
    let start = s.find(&pat)? + pat.len();
    let digits: String = s[start..].chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// A warm pi RPC process. `ask` sends a turn; the reply arrives on the channel
/// in sentence-sized chunks, then the sender is dropped (disconnect = done).
pub struct PiOracle {
    stdin: Arc<Mutex<ChildStdin>>,
    /// Where to deliver the current reply's chunks. Set before each prompt,
    /// dropped on agent_end so the receiver sees a disconnect when done.
    pending: Arc<Mutex<Option<Sender<Result<String, String>>>>>,
    /// When the current prompt was sent; the reader thread logs the time to
    /// first delivered chunk (the latency the writer actually feels).
    asked: Arc<Mutex<Option<std::time::Instant>>>,
    _child: Child,
}

impl PiOracle {
    /// Spawn the resident pi process and its stdout reader thread. This pays
    /// the warmup cost once; call it at diary startup.
    pub fn spawn() -> std::io::Result<Self> {
        let _ = std::fs::create_dir_all(DATA_DIR);
        let path = std::env::var("PATH").unwrap_or_default();

        // Use pi's ABSOLUTE path: Rust's Command resolves the program name via
        // the PARENT's PATH, not the child env we set below, so a bare "pi"
        // would not be found when riddle is launched with a minimal PATH.
        let pi_bin = format!("{NODE_BIN}/pi");
        let mut child = Command::new(&pi_bin)
            .current_dir(DATA_DIR)
            .env("HOME", "/home/root")
            .env("PATH", format!("{NODE_BIN}:{path}"))
            .args([
                "--mode", "rpc",
                "--provider", "openai-codex",
                "--model", "gpt-5.4-mini",
                "--thinking", "off",
                // The diary only ever writes back — never let the model touch
                // tools; also trims the tool schemas from every request.
                "--no-tools",
                "--system-prompt", PERSONA,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Keep pi's stderr for diagnosis instead of discarding it.
            .stderr(
                std::fs::File::create("/tmp/riddle-oracle.log")
                    .map(Stdio::from)
                    .unwrap_or_else(|_| Stdio::null()),
            )
            .spawn()?;

        let pid = child.id();
        eprintln!("riddle: oracle pi rpc spawned (pid {pid}, bin {pi_bin})");
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let pending: Arc<Mutex<Option<Sender<Result<String, String>>>>> =
            Arc::new(Mutex::new(None));

        // Reader thread: parse JSONL events, streaming each completed sentence
        // to the diary the moment it exists — the quill writes far slower than
        // the model streams, so the rest arrives while the first line is drawn.
        let pending_r = Arc::clone(&pending);
        let asked: Arc<Mutex<Option<std::time::Instant>>> = Arc::new(Mutex::new(None));
        let asked_r = Arc::clone(&asked);
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            let mut last_text = String::new();
            // Byte offset into `last_text` already sent to the diary.
            let mut delivered = 0usize;
            for line in reader.split(b'\n').map_while(Result::ok) {
                let Ok(s) = String::from_utf8(line) else { continue };
                let s = s.trim();
                if s.is_empty() {
                    continue;
                }
                // Cheap field extraction avoids a JSON dep; the event stream is
                // well-formed one-object-per-line.
                let ev_type = json_str_field(s, "type");
                match ev_type.as_deref() {
                    // message_update carries the assistant message's running
                    // full text; deliver every newly completed sentence.
                    Some("message_update") => {
                        if let Some(t) = extract_assistant_text(s) {
                            if !t.is_empty() {
                                last_text = t;
                                if let Some(cut) = sentence_cut(&last_text, delivered) {
                                    if delivered == 0 {
                                        if let Some(t0) = asked_r.lock().unwrap().take() {
                                            eprintln!("riddle: oracle first chunk +{}ms", t0.elapsed().as_millis());
                                        }
                                    }
                                    deliver(&pending_r, &last_text[delivered..cut], delivered == 0, false);
                                    delivered = cut;
                                }
                            }
                        }
                    }
                    // message_end has the definitive full text: flush the rest.
                    // (agent_end is NOT used for text — its `messages` array
                    // also contains user messages, which extract_assistant_text
                    // would wrongly concatenate in a multi-turn session.)
                    Some("message_end") => {
                        if let Some(t) = extract_assistant_text(s) {
                            if !t.is_empty() {
                                last_text = t;
                            }
                        }
                        if let Some(rest) = last_text.get(delivered..) {
                            if !rest.is_empty() {
                                if delivered == 0 {
                                    if let Some(t0) = asked_r.lock().unwrap().take() {
                                        eprintln!("riddle: oracle first chunk +{}ms (at message_end)", t0.elapsed().as_millis());
                                    }
                                }
                                deliver(&pending_r, rest, delivered == 0, true);
                            }
                        }
                        delivered = last_text.len();
                    }
                    // agent_end: the turn is over. Drop the sender so the
                    // diary's receiver disconnects (= no more ink coming).
                    Some("agent_end") => {
                        if let Some(tx) = pending_r.lock().unwrap().take() {
                            if delivered == 0 {
                                let _ = tx.send(Err("empty reply".into()));
                            }
                        }
                        last_text.clear();
                        delivered = 0;
                    }
                    _ => {}
                }
            }
            // Process died: fail any in-flight request.
            if let Some(tx) = pending_r.lock().unwrap().take() {
                let _ = tx.send(Err("pi rpc process exited".into()));
            }
        });

        Ok(Self { stdin: Arc::new(Mutex::new(stdin)), pending, asked, _child: child })
    }

    /// Send a handwriting turn. Reply chunks are delivered on `tx` as they
    /// stream; `tx` is dropped when the reply is complete.
    pub fn ask(&self, png_path: &str, tx: Sender<Result<String, String>>) {
        let img = match std::fs::read(png_path) {
            Ok(b) => base64(&b),
            Err(e) => {
                let _ = tx.send(Err(format!("read image: {e}")));
                return;
            }
        };
        *self.pending.lock().unwrap() = Some(tx.clone());
        *self.asked.lock().unwrap() = Some(std::time::Instant::now());

        let cmd = format!(
            "{{\"type\":\"prompt\",\"message\":{},\"images\":[{{\"type\":\"image\",\"data\":\"{}\",\"mimeType\":\"image/png\"}}]}}\n",
            json_quote("Reply to what is written in the diary."),
            img
        );
        let mut stdin = self.stdin.lock().unwrap();
        if stdin.write_all(cmd.as_bytes()).and_then(|_| stdin.flush()).is_err() {
            if let Some(tx) = self.pending.lock().unwrap().take() {
                let _ = tx.send(Err("pi rpc write failed".into()));
            }
        }
    }
}

/// Any OpenAI-compatible chat backend. No warm process: each turn opens a
/// streaming `/chat/completions` request on its own thread and forwards
/// sentence-sized chunks as SSE deltas arrive.
pub struct HttpOracle {
    base: String,   // e.g. https://api.openai.com/v1  (no trailing slash)
    key: String,
    model: String,
    max_tokens: u32,
    reasoning: Option<String>, // "reasoning_effort" value, e.g. "low"
    /// Completed turns, oldest first — replayed as chat history so the diary
    /// remembers the conversation. Persisted to `memory_path` (canon: the
    /// diary keeps what you tell it); the basilisk-fang stab erases it.
    history: Arc<Mutex<Vec<Turn>>>,
    history_turns: usize,
    memory_path: String,
}

/// One remembered exchange. `wrote` is the model's own transcription of the
/// writer's page (text, not the image — this is what keeps 20+ turns cheap);
/// `at` is unix seconds, used to tell the diary how long the writer was away.
struct Turn {
    at: u64,
    wrote: String,
    reply: String,
}

impl HttpOracle {
    pub fn new() -> std::io::Result<Self> {
        let key = std::env::var("RIDDLE_OPENAI_KEY").map_err(|_| {
            std::io::Error::other("RIDDLE_OPENAI_KEY not set")
        })?;
        let base = std::env::var("RIDDLE_OPENAI_BASE")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let base = base.trim_end_matches('/').to_string();
        // A vision-capable default; override with RIDDLE_OPENAI_MODEL.
        let model = std::env::var("RIDDLE_OPENAI_MODEL")
            .unwrap_or_else(|_| "gpt-4o-mini".to_string());
        // Thinking models (Gemini 3.x, o-series…) count hidden reasoning
        // tokens against max_tokens: a tight cap starves the visible reply to
        // one sentence (finish_reason=length). The persona already keeps
        // replies short, so the cap is only a runaway guard — leave headroom.
        let max_tokens = std::env::var("RIDDLE_OPENAI_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2000);
        // Sent as "reasoning_effort" only when set: reasoning models accept it
        // ("low" ≈ faster first ink), but some providers reject the field on
        // non-reasoning models, so it must stay out of the default request.
        // Set-but-empty (as the settings UI writes for the "" option) means
        // unset — sending "reasoning_effort":"" is a 400 on OpenAI.
        let reasoning = std::env::var("RIDDLE_OPENAI_REASONING")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        // How many past turns the diary remembers. History is stored as text
        // (the model transcribes each page), so turns are cheap — this caps
        // both the tokens sent per request and the memory file on disk.
        // 0 = the old stateless behavior.
        let history_turns = std::env::var("RIDDLE_HISTORY_TURNS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(20);
        // The diary's persistent memory, JSONL, next to oracle.env (the
        // launcher cd's into the bundle dir). Unreadable = start fresh.
        let memory_path = std::env::var("RIDDLE_MEMORY_FILE")
            .unwrap_or_else(|_| "memory.json".to_string());
        let remembered = match load_memory(&memory_path) {
            Ok(v) => v,
            Err(e) => {
                if e.kind() != std::io::ErrorKind::NotFound {
                    eprintln!("riddle: memory blurred ({e}), starting fresh");
                }
                Vec::new()
            }
        };
        eprintln!(
            "riddle: http oracle base={base} model={model} max_tokens={max_tokens} reasoning={} history={history_turns} remembered={}",
            reasoning.as_deref().unwrap_or("-"),
            remembered.len(),
        );
        Ok(Self {
            base,
            key,
            model,
            max_tokens,
            reasoning,
            history: Arc::new(Mutex::new(remembered)),
            history_turns,
            memory_path,
        })
    }

    /// The basilisk fang lands: the diary forgets everything, on disk too.
    pub fn erase(&self) {
        self.history.lock().unwrap().clear();
        let _ = std::fs::remove_file(&self.memory_path);
    }

    pub fn ask(&self, png_path: &str, tx: Sender<Result<String, String>>) {
        let img = match std::fs::read(png_path) {
            Ok(b) => base64(&b),
            Err(e) => {
                let _ = tx.send(Err(format!("read image: {e}")));
                return;
            }
        };
        let (base, key, model) = (self.base.clone(), self.key.clone(), self.model.clone());
        let max_tokens = self.max_tokens;
        let reasoning_field = self
            .reasoning
            .as_deref()
            .map(|r| format!("\"reasoning_effort\":{},", json_quote(r)))
            .unwrap_or_default();
        let history = Arc::clone(&self.history);
        let history_turns = self.history_turns;
        let memory_path = self.memory_path.clone();

        thread::spawn(move || {
            // Past turns give the diary its memory: each is replayed as the
            // model's own transcription of the writer's page plus the reply.
            // Text, not images — that's what keeps a long memory cheap.
            let mut messages = format!(
                "{{\"role\":\"system\",\"content\":{}}}",
                json_quote(&format!("{PERSONA} {TRANSCRIBE}"))
            );
            let last_at = {
                let h = history.lock().unwrap();
                let last_at = h.last().map(|t| t.at);
                if history_turns > 0 {
                    for t in h.iter() {
                        messages.push_str(&format!(
                            ",{{\"role\":\"user\",\"content\":{}}},{{\"role\":\"assistant\",\"content\":{}}}",
                            json_quote(&format!("The writer wrote: ⟪{}⟫", t.wrote)),
                            json_quote(&t.reply),
                        ));
                    }
                }
                last_at
            };
            // The writer coming back after a real absence is something the
            // diary should feel (canon: Riddle needled Ginny about hers).
            let ask_text = match last_at.map(|at| now_secs().saturating_sub(at)).and_then(absence_phrase) {
                Some(gap) => format!("(The writer returns to the diary after {gap}.) {ASK}"),
                None => ASK.to_string(),
            };
            messages.push_str(&format!(
                concat!(
                    ",{{\"role\":\"user\",\"content\":[",
                    "{{\"type\":\"text\",\"text\":{}}},",
                    "{{\"type\":\"image_url\",\"image_url\":{{\"url\":\"data:image/png;base64,{}\"}}}}",
                    "]}}"
                ),
                json_quote(&ask_text),
                img,
            ));

            // OpenAI chat-completions with a data-URI image part, streaming.
            // The cap is sent as "max_completion_tokens": reasoning models
            // (gpt-5.x, o-series) reject the legacy "max_tokens" outright,
            // and every current OpenAI model accepts the new field.
            let body = format!(
                "{{\"model\":{},\"stream\":true,\"max_completion_tokens\":{},{}\"messages\":[{}]}}",
                json_quote(&model),
                max_tokens,
                reasoning_field,
                messages,
            );

            // The transcription prefix is required on the first attempt; a
            // model that skips it gets one retry (nothing has been inked yet
            // when we find out — the prefix is the very start of the stream).
            // The retry is lenient: the reply streams regardless and the
            // memory gets a placeholder, so bookkeeping never eats a turn.
            let asked = std::time::Instant::now();
            for attempt in 0..2u32 {
                let lenient = attempt == 1;
                match stream_once(&base, &key, &body, &tx, asked, lenient) {
                    StreamEnd::Miss => {
                        eprintln!("riddle: transcription prefix missing, retrying");
                        continue;
                    }
                    StreamEnd::Fail(e) => {
                        let _ = tx.send(Err(e));
                        return;
                    }
                    StreamEnd::Done { wrote, reply } => {
                        if history_turns > 0 {
                            let mut h = history.lock().unwrap();
                            h.push(Turn { at: now_secs(), wrote, reply });
                            let extra = h.len().saturating_sub(history_turns);
                            if extra > 0 {
                                h.drain(..extra);
                            }
                            if let Err(e) = save_memory(&memory_path, &h) {
                                eprintln!("riddle: memory save failed: {e}");
                            }
                        }
                        return;
                    }
                }
            }
            // Both attempts missed the prefix in strict mode — unreachable
            // (the second attempt is lenient), but never leave tx hanging.
            let _ = tx.send(Err("oracle stream ended unexpectedly".into()));
            // tx drops here → the diary's receiver disconnects = reply complete.
        });
    }
}

/// Outcome of one streamed completion attempt.
enum StreamEnd {
    /// Reply fully streamed to the page; `wrote` is the model's transcription
    /// of the writer's page, `reply` the full reply text.
    Done { wrote: String, reply: String },
    /// Strict mode only: the stream didn't open with the transcription
    /// prefix. Nothing was delivered — safe to retry.
    Miss,
    Fail(String),
}

/// POST the completion and stream it: parse the `⟪transcription⟫` first line
/// (never inked onto the page), then deliver each completed sentence of the
/// reply as it lands. In lenient mode a missing prefix streams the whole text
/// as the reply with a placeholder transcription.
fn stream_once(
    base: &str,
    key: &str,
    body: &str,
    tx: &Sender<Result<String, String>>,
    asked: std::time::Instant,
    lenient: bool,
) -> StreamEnd {
    let resp = ureq::post(&format!("{base}/chat/completions"))
        .set("Authorization", &format!("Bearer {key}"))
        .set("Content-Type", "application/json")
        .send_string(body);

    let reader = match resp {
        Ok(r) => r.into_reader(),
        Err(ureq::Error::Status(code, r)) => {
            let detail = r.into_string().unwrap_or_default();
            return StreamEnd::Fail(format!("http {code}: {}", detail.trim()));
        }
        Err(e) => return StreamEnd::Fail(format!("request failed: {e}")),
    };

    // Parse the SSE stream: lines of `data: {json}` whose delta.content
    // fragments accumulate; deliver each completed sentence as it lands.
    let mut acc = String::new();
    let mut wrote: Option<String> = None;
    let mut delivered = 0usize; // byte index into acc; starts after the prefix
    let mut first = true;
    for line in BufReader::new(reader).lines().map_while(Result::ok) {
        let line = line.trim();
        let Some(data) = line.strip_prefix("data:") else { continue };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }
        let Some(frag) = sse_delta_content(data) else { continue };
        if frag.is_empty() {
            continue;
        }
        acc.push_str(&frag);

        if wrote.is_none() {
            // Still hunting the ⟪…⟫ prefix at the very start of the stream.
            let lead = acc.trim_start();
            let miss = (!lead.is_empty() && !lead.starts_with('⟪'))
                || (lead.starts_with('⟪') && acc.len() > 600 && !acc.contains('⟫'));
            if miss {
                if !lenient {
                    return StreamEnd::Miss;
                }
                wrote = Some("(the ink blurred)".to_string());
            } else if let (Some(open), Some(close)) = (acc.find('⟪'), acc.find('⟫')) {
                wrote = Some(acc[open + '⟪'.len_utf8()..close].trim().to_string());
                delivered = close + '⟫'.len_utf8();
            } else {
                continue; // prefix still streaming
            }
        }

        if let Some(cut) = sentence_cut(&acc, delivered) {
            if first {
                eprintln!("riddle: oracle first chunk +{}ms", asked.elapsed().as_millis());
                first = false;
            }
            let chunk = acc[delivered..cut].to_string();
            let _ = tx.send(Ok(clean(&chunk)));
            delivered = cut;
        }
    }
    // Stream ended while still hunting the prefix: in strict mode any real
    // content without a resolved prefix is a Miss (nothing was delivered, so
    // the retry is free); in lenient mode everything becomes the reply.
    if wrote.is_none() {
        if acc.trim().is_empty() {
            return StreamEnd::Fail("empty reply".into());
        }
        if !lenient {
            return StreamEnd::Miss;
        }
        wrote = Some("(the ink blurred)".to_string());
    }
    // Flush any trailing text past the last sentence break.
    if delivered < acc.len() {
        let rest = acc[delivered..].trim();
        if !rest.is_empty() {
            if first {
                eprintln!("riddle: oracle first chunk +{}ms", asked.elapsed().as_millis());
                first = false;
            }
            let _ = tx.send(Ok(clean(rest)));
        }
    }
    if first {
        // Nothing was ever delivered (prefix present but no reply after it).
        return StreamEnd::Fail("empty reply".into());
    }
    // The reply for the memory is everything after the ⟪…⟫ prefix (from the
    // start in the lenient no-prefix fallback).
    let prefix_end = match (acc.trim_start().starts_with('⟪'), acc.find('⟫')) {
        (true, Some(p)) => p + '⟫'.len_utf8(),
        _ => 0,
    };
    StreamEnd::Done {
        wrote: wrote.unwrap_or_else(|| "(the ink blurred)".to_string()),
        reply: clean(&acc[prefix_end..]),
    }
}

/// Pull `choices[0].delta.content` out of one SSE `data:` JSON object.
fn sse_delta_content(s: &str) -> Option<String> {
    // The delta object is small and well-formed; find the content string after
    // the `"delta":` marker so we don't match a `content` elsewhere.
    let d = s.find("\"delta\"")?;
    json_str_field(&s[d..], "content")
}

/// Trim and strip stray surrounding quotes from a reply fragment.
fn clean(s: &str) -> String {
    let t = s.trim();
    let t = t.strip_prefix('"').unwrap_or(t);
    let t = t.strip_suffix('"').unwrap_or(t);
    t.to_string()
}

/// Send one chunk of reply text without consuming the sender (more chunks may
/// follow until agent_end drops it). Strips a stray wrapping quote from the
/// reply's very first / very last chunk.
fn deliver(
    pending: &Arc<Mutex<Option<Sender<Result<String, String>>>>>,
    chunk: &str,
    first: bool,
    last: bool,
) {
    let mut t = chunk.trim();
    if first {
        t = t.strip_prefix('"').unwrap_or(t);
    }
    if last {
        t = t.strip_suffix('"').unwrap_or(t);
    }
    let t = t.trim();
    if t.is_empty() {
        return;
    }
    if let Some(tx) = pending.lock().unwrap().as_ref() {
        let _ = tx.send(Ok(t.to_string()));
    }
}

/// End of the LAST complete sentence in `text` after byte offset `from`:
/// sentence punctuation followed by whitespace or end-of-text. Returns the
/// offset just past the punctuation, or None if no sentence has completed.
/// Chunks shorter than a few characters are not worth an early delivery.
fn sentence_cut(text: &str, from: usize) -> Option<usize> {
    let tail = text.get(from..)?;
    let mut cut = None;
    for (i, c) in tail.char_indices() {
        if matches!(c, '.' | '!' | '?' | '…') {
            let end = i + c.len_utf8();
            if tail[end..].chars().next().is_none_or(char::is_whitespace) && end >= 4 {
                cut = Some(from + end);
            }
        }
    }
    cut
}

/// Extract a top-level string field's value (first match; unescaped).
fn json_str_field(s: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":\"");
    let start = s.find(&pat)? + pat.len();
    let rest = &s[start..];
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if let Some(n) = chars.next() {
                    match n {
                        'n' => out.push('\n'),
                        't' => out.push('\t'),
                        'r' => out.push('\r'),
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        // \uXXXX — needed for accented replies (French, em-dash…).
                        'u' => {
                            let hex: String = (0..4).filter_map(|_| chars.next()).collect();
                            if let Some(ch) = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                                out.push(ch);
                            }
                        }
                        other => out.push(other),
                    }
                }
            }
            '"' => break,
            _ => out.push(c),
        }
    }
    Some(out)
}

/// Pull the assistant reply text out of an event line. The event carries a
/// `message` object with `"role":"assistant"` and `content:[{type:text,text:…}]`.
/// We only trust text that belongs to an assistant message (the user echo also
/// contains a "text" field, which we must NOT return).
fn extract_assistant_text(s: &str) -> Option<String> {
    // Require this line to be an assistant message.
    if !s.contains("\"role\":\"assistant\"") {
        return None;
    }
    // Collect every "text":"…" occurrence inside the FIRST assistant section
    // only. message_update lines carry the running text twice (in
    // assistantMessageEvent.partial AND a top-level message); reading past the
    // next role marker would double every streamed chunk.
    let role_pos = s.find("\"role\":\"assistant\"")?;
    let after = &s[role_pos + "\"role\":\"assistant\"".len()..];
    let tail = match after.find("\"role\":\"") {
        Some(p) => &after[..p],
        None => after,
    };
    let mut out = String::new();
    let mut idx = 0;
    let needle = "\"text\":\"";
    while let Some(rel) = tail[idx..].find(needle) {
        let start = idx + rel + needle.len();
        // Decode the JSON string starting at `start`.
        let mut chars = tail[start..].chars();
        let mut piece = String::new();
        while let Some(c) = chars.next() {
            match c {
                '\\' => {
                    if let Some(n) = chars.next() {
                        piece.push(match n {
                            'n' => '\n',
                            't' => '\t',
                            'r' => '\r',
                            '"' => '"',
                            '\\' => '\\',
                            '/' => '/',
                            other => other,
                        });
                    }
                }
                '"' => break,
                _ => piece.push(c),
            }
        }
        out.push_str(&piece);
        // Advance past this occurrence.
        idx = start;
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn json_quote(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[((n >> 18) & 63) as usize] as char);
        out.push(T[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 { T[((n >> 6) & 63) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[(n & 63) as usize] as char } else { '=' });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_delta_extraction() {
        let line = r#"{"choices":[{"delta":{"content":"Hello"},"index":0}]}"#;
        assert_eq!(sse_delta_content(line).as_deref(), Some("Hello"));
        // role-only delta (first SSE frame) has no content.
        let role = r#"{"choices":[{"delta":{"role":"assistant"},"index":0}]}"#;
        assert_eq!(sse_delta_content(role), None);
    }

    #[test]
    fn sse_decodes_unicode_and_escapes() {
        // OpenAI escapes accents and em-dashes; the diary answers in French.
        let line = r#"{"choices":[{"delta":{"content":"Déjà vu — oui"}}]}"#;
        assert_eq!(sse_delta_content(line).as_deref(), Some("Déjà vu — oui"));
        let nl = r#"{"choices":[{"delta":{"content":"line\nbreak"}}]}"#;
        assert_eq!(sse_delta_content(nl).as_deref(), Some("line\nbreak"));
    }

    #[test]
    fn base64_matches_known_vector() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn clean_strips_wrapping_quotes() {
        assert_eq!(clean("  \"hello\"  "), "hello");
        assert_eq!(clean("plain"), "plain");
    }
}

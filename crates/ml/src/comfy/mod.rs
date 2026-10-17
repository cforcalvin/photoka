//! The ComfyUI backend Photoka installs, launches and drives (see the engine's `ai_comfy_cmds`).
//!
//! - [`install`]: the managed install (pinned `uv` and ComfyUI, a Python venv with PyTorch).
//! - [`server`]: the ComfyUI process on 127.0.0.1, reading Photoka's models folder.
//! - [`workflow`]: API-format workflow JSON for one generation request.
//! - [`Client`]: uploads the frame, queues the workflow, follows progress and previews over the
//!   websocket (falling back to polling), cancels with `/interrupt` and fetches the result.

pub mod install;
pub mod server;
pub mod workflow;

use std::net::{SocketAddr, TcpStream};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tungstenite::{Message, WebSocket};

pub use install::Install;
pub use server::Server;

use crate::catalog::{Family, ModelKind};
use crate::image::{Mask, Rgba8};
use crate::params::{GenerateOutput, GenerateRequest};
use crate::store::ModelStore;
use crate::{MlError, Progress, Result};

const USER_AGENT: &str = "Photoka-dev";
/// Largest response read from the server (result images, history).
const MAX_RESPONSE: u64 = 256 << 20;
/// Longest a single generation may take before Photoka gives up on it.
const GENERATE_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const WS_POLL: Duration = Duration::from_millis(250);
const HISTORY_POLL: Duration = Duration::from_secs(2);

/// A ComfyUI server on the loopback interface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Client {
    pub port: u16,
}

fn net(what: &str, e: impl std::fmt::Display) -> MlError {
    MlError::Comfy(format!("{what}: {e} (is ComfyUI still running?)"))
}

/// Percent-encode a query value.
fn enc(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

fn new_client_id() -> String {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    format!("photoka-{:x}-{:x}", std::process::id(), nanos)
}

/// The frame as uploaded: its pixels, with the mask as transparency (transparent = regenerate),
/// which ComfyUI's `LoadImage` turns into its MASK output.
pub fn frame_png(init: &Rgba8, mask: Option<&Mask>) -> Result<Vec<u8>> {
    let mut img = init.clone();
    for (i, px) in img.data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let m = mask.and_then(|m| m.data.get(i).copied()).unwrap_or(0);
        if let Some(a) = px.get_mut(3) {
            *a = 255 - m;
        }
    }
    img.to_png()
}

/// A readable message from a `/prompt` rejection.
fn prompt_error(body: &Value) -> String {
    let mut msg = body.pointer("/error/message").and_then(Value::as_str).unwrap_or("the workflow was rejected").to_string();
    if let Some(nodes) = body.get("node_errors").and_then(Value::as_object) {
        for (_, n) in nodes.iter().take(2) {
            let class = n.get("class_type").and_then(Value::as_str).unwrap_or("node");
            for e in n.get("errors").and_then(Value::as_array).into_iter().flatten().take(2) {
                let m = e.get("message").and_then(Value::as_str).unwrap_or_default();
                let d = e.get("details").and_then(Value::as_str).unwrap_or_default();
                msg.push_str(&format!("; {class}: {m} {}", install::truncate(d, 200)));
            }
        }
    }
    msg
}

impl Client {
    pub fn new(port: u16) -> Client {
        Client { port }
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    fn agent(&self) -> ureq::Agent {
        ureq::Agent::config_builder()
            .user_agent(USER_AGENT)
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(5)))
            .timeout_recv_response(Some(Duration::from_secs(120)))
            .build()
            .into()
    }

    fn read(resp: ureq::http::Response<ureq::Body>, what: &str) -> Result<(u16, Vec<u8>)> {
        let status = resp.status().as_u16();
        let body = resp.into_body().into_with_config().limit(MAX_RESPONSE).read_to_vec().map_err(|e| net(what, e))?;
        Ok((status, body))
    }

    fn json(what: &str, status: u16, body: &[u8]) -> Result<Value> {
        if status != 200 {
            return Err(MlError::Comfy(format!("{what}: HTTP {status} {}", install::truncate(&String::from_utf8_lossy(body), 300))));
        }
        serde_json::from_slice(body).map_err(|e| MlError::Comfy(format!("{what}: unexpected answer ({e})")))
    }

    /// `GET /system_stats`: answers once the server is up.
    pub fn system_stats(&self) -> Result<Value> {
        let resp = self.agent().get(format!("{}/system_stats", self.base_url())).call().map_err(|e| net("system_stats", e))?;
        let (status, body) = Self::read(resp, "system_stats")?;
        Self::json("system_stats", status, &body)
    }

    /// Upload a PNG to ComfyUI's input folder; returns the name to load it by.
    pub fn upload_image(&self, name: &str, png: &[u8]) -> Result<String> {
        let boundary = format!("----photoka{}", new_client_id());
        let mut body = Vec::with_capacity(png.len() + 512);
        let safe: String = name.chars().filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')).collect();
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{safe}\"\r\nContent-Type: image/png\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(png);
        body.extend_from_slice(format!("\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue").as_bytes());
        body.extend_from_slice(format!("\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n--{boundary}--\r\n").as_bytes());
        let resp = self
            .agent()
            .post(format!("{}/upload/image", self.base_url()))
            .header("Content-Type", format!("multipart/form-data; boundary={boundary}"))
            .send(&body[..])
            .map_err(|e| net("upload", e))?;
        let (status, bytes) = Self::read(resp, "upload")?;
        let v = Self::json("upload", status, &bytes)?;
        let name = v.get("name").and_then(Value::as_str).ok_or_else(|| MlError::Comfy("upload: no file name in the answer".into()))?;
        Ok(match v.get("subfolder").and_then(Value::as_str).filter(|s| !s.is_empty()) {
            Some(sub) => format!("{sub}/{name}"),
            None => name.to_string(),
        })
    }

    /// Queue `workflow`; returns the prompt id.
    pub fn queue_prompt(&self, workflow: &Value, client_id: &str) -> Result<String> {
        let body = json!({"prompt": workflow, "client_id": client_id}).to_string();
        let resp = self
            .agent()
            .post(format!("{}/prompt", self.base_url()))
            .header("Content-Type", "application/json")
            .send(body.as_bytes())
            .map_err(|e| net("prompt", e))?;
        let (status, bytes) = Self::read(resp, "prompt")?;
        if status == 400 {
            let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
            return Err(MlError::Comfy(prompt_error(&v)));
        }
        let v = Self::json("prompt", status, &bytes)?;
        v.get("prompt_id").and_then(Value::as_str).map(str::to_string).ok_or_else(|| MlError::Comfy("prompt: no prompt id in the answer".into()))
    }

    /// The finished prompt's history entry, or `None` while it's still running.
    pub fn history(&self, prompt_id: &str) -> Result<Option<Value>> {
        let resp = self.agent().get(format!("{}/history/{}", self.base_url(), enc(prompt_id))).call().map_err(|e| net("history", e))?;
        let (status, bytes) = Self::read(resp, "history")?;
        let v = Self::json("history", status, &bytes)?;
        Ok(v.get(prompt_id).cloned())
    }

    /// `GET /view` for an output image reference (`{filename, subfolder, type}`).
    pub fn view(&self, image: &Value) -> Result<Vec<u8>> {
        let s = |k: &str| image.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
        let filename = s("filename");
        if filename.is_empty() {
            return Err(MlError::Comfy("the result has no image".into()));
        }
        let url = format!("{}/view?filename={}&subfolder={}&type={}", self.base_url(), enc(&filename), enc(&s("subfolder")), enc(&s("type")));
        let resp = self.agent().get(url).call().map_err(|e| net("view", e))?;
        let (status, bytes) = Self::read(resp, "view")?;
        if status != 200 {
            return Err(MlError::Comfy(format!("view: HTTP {status}")));
        }
        Ok(bytes)
    }

    /// Stop the running prompt.
    pub fn interrupt(&self) -> Result<()> {
        let resp = self
            .agent()
            .post(format!("{}/interrupt", self.base_url()))
            .header("Content-Type", "application/json")
            .send(&b"{}"[..])
            .map_err(|e| net("interrupt", e))?;
        Self::read(resp, "interrupt").map(|_| ())
    }

    fn connect_ws(&self, client_id: &str) -> Result<WebSocket<TcpStream>> {
        let addr = SocketAddr::from(([127, 0, 0, 1], self.port));
        let stream = TcpStream::connect_timeout(&addr, Duration::from_secs(5)).map_err(|e| net("websocket", e))?;
        stream.set_read_timeout(Some(Duration::from_secs(5))).map_err(|e| net("websocket", e))?;
        let url = format!("ws://127.0.0.1:{}/ws?clientId={}", self.port, enc(client_id));
        let (ws, _) = tungstenite::client(url, stream).map_err(|e| net("websocket", e))?;
        ws.get_ref().set_read_timeout(Some(WS_POLL)).map_err(|e| net("websocket", e))?;
        Ok(ws)
    }

    /// Run one generation: upload the frame, queue the workflow, follow it, fetch the image.
    /// Run Real-ESRGAN (or another file in `upscale_models/`) on `image` via ComfyUI.
    pub fn upscale(&self, store: &ModelStore, model: &str, image: &Rgba8, progress: &dyn Progress) -> Result<Rgba8> {
        let _ = store.existing(ModelKind::Upscale, model)?;
        if image.width == 0 || image.height == 0 {
            return Err(MlError::BadParams("nothing to upscale".into()));
        }
        let client_id = new_client_id();
        progress.report(0.01, "Sending the image to ComfyUI");
        let name = self.upload_image(&format!("{client_id}-upscale.png"), &image.to_png()?)?;
        let wf = workflow::build_upscale(model, &name);
        let mut ws = match self.connect_ws(&client_id) {
            Ok(ws) => Some(ws),
            Err(e) => {
                log::info!("ComfyUI websocket unavailable ({e}); polling instead");
                None
            }
        };
        let prompt_id = self.queue_prompt(&wf, &client_id)?;
        progress.report(0.05, "Upscaling in ComfyUI");
        let waited = self.wait(&mut ws, &prompt_id, progress);
        if let Some(mut ws) = ws {
            let _ = ws.close(None);
        }
        let entry = waited?;
        let img = entry
            .pointer(&format!("/outputs/{}/images/0", workflow::UPSCALE_OUTPUT_NODE))
            .ok_or_else(|| MlError::Comfy("ComfyUI finished without an upscaled image".into()))?;
        progress.report(0.98, "Fetching the result");
        Rgba8::decode_upscaled(&self.view(img)?)
    }

    pub fn generate(&self, store: &ModelStore, req: &GenerateRequest, progress: &dyn Progress) -> Result<GenerateOutput> {
        req.validate()?;
        let started = Instant::now();
        let family = store
            .existing(ModelKind::Checkpoint, &req.checkpoint)
            .ok()
            .and_then(|p| crate::store::detect(&p, Some(ModelKind::Checkpoint)).ok())
            .map_or(Family::Sd15, |d| d.family);
        let client_id = new_client_id();
        progress.report(0.01, "Sending the frame to ComfyUI");
        let image = match &req.init {
            Some(init) if req.mask.is_some() || req.strength < 1.0 => {
                Some(self.upload_image(&format!("{client_id}.png"), &frame_png(init, req.mask.as_ref())?)?)
            }
            _ => None,
        };
        let wf = workflow::build(req, family, image.as_deref());
        let mut ws = match self.connect_ws(&client_id) {
            Ok(ws) => Some(ws),
            Err(e) => {
                log::info!("ComfyUI websocket unavailable ({e}); polling instead");
                None
            }
        };
        let prompt_id = self.queue_prompt(&wf, &client_id)?;
        progress.report(0.03, "Queued in ComfyUI");
        let waited = self.wait(&mut ws, &prompt_id, progress);
        if let Some(mut ws) = ws {
            let _ = ws.close(None);
        }
        let entry = waited?;
        let img =
            entry.pointer(&format!("/outputs/{}/images/0", workflow::OUTPUT_NODE)).ok_or_else(|| MlError::Comfy("ComfyUI finished without an image".into()))?;
        progress.report(0.98, "Fetching the result");
        let mut image = Rgba8::decode(&self.view(img)?)?;
        if (image.width, image.height) != (req.width, req.height) {
            image = image.resized(req.width, req.height)?;
        }
        Ok(GenerateOutput { image, seed: req.seed, elapsed_ms: started.elapsed().as_millis() as u64 })
    }

    /// Follow `prompt_id` until it finishes; returns its history entry.
    fn wait(&self, ws: &mut Option<WebSocket<TcpStream>>, prompt_id: &str, progress: &dyn Progress) -> Result<Value> {
        let started = Instant::now();
        let mut polled = Instant::now();
        let mut done = false;
        loop {
            if progress.cancelled() {
                let _ = self.interrupt();
                return Err(MlError::Cancelled);
            }
            if started.elapsed() > GENERATE_TIMEOUT {
                let _ = self.interrupt();
                return Err(MlError::Comfy(format!("no result after {} minutes; it was stopped", GENERATE_TIMEOUT.as_secs() / 60)));
            }
            match ws.as_mut().map(|s| s.read()) {
                Some(Ok(Message::Text(t))) => done |= handle_event(t.as_str(), prompt_id, progress)?,
                Some(Ok(Message::Binary(b))) => show_preview(&b, progress),
                Some(Ok(_)) => {}
                Some(Err(tungstenite::Error::Io(e))) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {}
                Some(Err(e)) => {
                    log::info!("ComfyUI websocket closed ({e}); polling instead");
                    *ws = None;
                }
                None => std::thread::sleep(WS_POLL),
            }
            if done || polled.elapsed() >= HISTORY_POLL {
                polled = Instant::now();
                if let Some(entry) = self.history(prompt_id)? {
                    if entry.pointer("/status/status_str").and_then(Value::as_str) == Some("error") {
                        return Err(MlError::Comfy(history_error(&entry)));
                    }
                    return Ok(entry);
                }
                if done {
                    // Finished but not in the history yet: give it a moment.
                    std::thread::sleep(Duration::from_millis(200));
                }
            }
        }
    }
}

/// One websocket event. True once the prompt has finished.
fn handle_event(text: &str, prompt_id: &str, progress: &dyn Progress) -> Result<bool> {
    let Ok(v) = serde_json::from_str::<Value>(text) else { return Ok(false) };
    let data = v.get("data").cloned().unwrap_or(Value::Null);
    let ours = data.get("prompt_id").and_then(Value::as_str).is_none_or(|p| p == prompt_id);
    if !ours {
        return Ok(false);
    }
    match v.get("type").and_then(Value::as_str).unwrap_or_default() {
        "progress" => {
            let value = data.get("value").and_then(Value::as_f64).unwrap_or(0.0);
            let max = data.get("max").and_then(Value::as_f64).unwrap_or(1.0).max(1.0);
            progress.report(0.05 + 0.9 * (value / max).clamp(0.0, 1.0) as f32, &format!("Step {value} of {max}"));
            Ok(false)
        }
        "executing" => Ok(data.get("node").is_some_and(Value::is_null)),
        "execution_error" => {
            let node = data.get("node_type").and_then(Value::as_str).unwrap_or("a node");
            let msg = data.get("exception_message").and_then(Value::as_str).unwrap_or("failed");
            Err(MlError::Comfy(format!("{node}: {}", install::truncate(msg.trim(), 500))))
        }
        "execution_interrupted" => Err(MlError::Cancelled),
        _ => Ok(false),
    }
}

/// A binary preview frame: `u32` event (1 = preview image), `u32` format, then the image bytes.
fn show_preview(b: &[u8], progress: &dyn Progress) {
    let event = b.get(..4).and_then(|s| <[u8; 4]>::try_from(s).ok()).map(u32::from_be_bytes);
    if event != Some(1) {
        return;
    }
    if let Some(img) = b.get(8..).and_then(|bytes| Rgba8::decode(bytes).ok()) {
        progress.preview(&img);
    }
}

fn history_error(entry: &Value) -> String {
    for m in entry.pointer("/status/messages").and_then(Value::as_array).into_iter().flatten() {
        if m.get(0).and_then(Value::as_str) == Some("execution_error") {
            let msg = m.pointer("/1/exception_message").and_then(Value::as_str).unwrap_or("failed");
            return install::truncate(msg.trim(), 500);
        }
    }
    "the generation failed; see ComfyUI's log".into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct Seen {
        prompts: Vec<Value>,
        uploads: usize,
        interrupts: usize,
    }

    /// A minimal ComfyUI: `/system_stats`, `/upload/image`, `/prompt`, `/history`, `/view`,
    /// `/interrupt` and (when `ws`) a websocket that reports two steps, a preview and the end.
    fn mock(ws: bool, reject: bool, hang: bool) -> (u16, Arc<Mutex<Seen>>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let seen = Arc::new(Mutex::new(Seen::default()));
        let s2 = Arc::clone(&seen);
        std::thread::spawn(move || {
            for stream in l.incoming().flatten() {
                let seen = Arc::clone(&s2);
                std::thread::spawn(move || serve(stream, &seen, ws, reject, hang));
            }
        });
        (port, seen)
    }

    fn png(w: u32, h: u32) -> Vec<u8> {
        let mut img = Rgba8::new(w, h).unwrap();
        img.data.as_chunks_mut::<4>().0.iter_mut().for_each(|p| *p = [200, 30, 30, 255]);
        img.to_png().unwrap()
    }

    fn serve(mut stream: TcpStream, seen: &Mutex<Seen>, ws: bool, reject: bool, hang: bool) {
        let mut head = Vec::new();
        let mut byte = [0u8; 1];
        let mut buf = [0u8; 4096];
        // Peek the request line without consuming it (the websocket handshake needs it).
        let n = stream.peek(&mut buf).unwrap_or(0);
        let first = String::from_utf8_lossy(&buf[..n]).to_string();
        if first.starts_with("GET /ws") {
            if !ws {
                let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                return;
            }
            let Ok(mut sock) = tungstenite::accept(stream) else { return };
            while seen.lock().unwrap().prompts.is_empty() {
                std::thread::sleep(Duration::from_millis(20));
            }
            if hang {
                std::thread::sleep(Duration::from_secs(10));
                return;
            }
            for v in [1, 2] {
                let _ = sock.send(Message::text(json!({"type": "progress", "data": {"value": v, "max": 2, "prompt_id": "p1"}}).to_string()));
            }
            let mut frame = vec![0, 0, 0, 1, 0, 0, 0, 2];
            frame.extend(png(8, 8));
            let _ = sock.send(Message::binary(frame));
            let _ = sock.send(Message::text(json!({"type": "executing", "data": {"node": null, "prompt_id": "p1"}}).to_string()));
            std::thread::sleep(Duration::from_millis(500));
            return;
        }
        while !head.ends_with(b"\r\n\r\n") {
            if stream.read(&mut byte).unwrap_or(0) == 0 {
                return;
            }
            head.push(byte[0]);
        }
        let head = String::from_utf8_lossy(&head).to_string();
        let len: usize = head.lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap_or(0))).unwrap_or(0);
        let mut body = vec![0u8; len];
        let _ = stream.read_exact(&mut body);
        let path = head.split_whitespace().nth(1).unwrap_or("/").to_string();
        let (code, ctype, out): (u16, &str, Vec<u8>) = if path == "/system_stats" {
            (200, "application/json", br#"{"system":{}}"#.to_vec())
        } else if path == "/upload/image" {
            seen.lock().unwrap().uploads += 1;
            let text = String::from_utf8_lossy(&body);
            let name = text.split("filename=\"").nth(1).and_then(|s| s.split('"').next()).unwrap_or("x.png").to_string();
            (200, "application/json", json!({"name": name, "subfolder": "", "type": "input"}).to_string().into_bytes())
        } else if path == "/prompt" {
            let v: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
            if reject {
                let e = json!({"error": {"message": "Prompt outputs failed validation"}, "node_errors": {"1": {"class_type": "CheckpointLoaderSimple", "errors": [{"message": "Value not in list", "details": "ckpt_name: 'missing.safetensors' not in []"}]}}});
                (400, "application/json", e.to_string().into_bytes())
            } else {
                seen.lock().unwrap().prompts.push(v);
                (200, "application/json", br#"{"prompt_id":"p1","number":0,"node_errors":{}}"#.to_vec())
            }
        } else if path == "/history/p1" {
            let ready = !hang && !seen.lock().unwrap().prompts.is_empty();
            let v = if ready {
                json!({"p1": {"outputs": {"22": {"images": [{"filename": "out.png", "subfolder": "", "type": "temp"}]}}, "status": {"status_str": "success", "completed": true}}})
            } else {
                json!({})
            };
            (200, "application/json", v.to_string().into_bytes())
        } else if path.starts_with("/view?filename=out.png") {
            (200, "image/png", png(32, 32))
        } else if path == "/interrupt" {
            seen.lock().unwrap().interrupts += 1;
            (200, "application/json", b"{}".to_vec())
        } else {
            (404, "text/plain", b"not found".to_vec())
        };
        let _ =
            stream.write_all(format!("HTTP/1.1 {code} X\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", out.len()).as_bytes());
        let _ = stream.write_all(&out);
    }

    fn req() -> GenerateRequest {
        GenerateRequest {
            checkpoint: "v1-5.safetensors".into(),
            vae: None,
            loras: vec![],
            prompt: "a lighthouse".into(),
            negative: String::new(),
            steps: 2,
            cfg: 7.0,
            seed: 7,
            sampler: crate::params::Sampler::Euler,
            strength: 1.0,
            width: 64,
            height: 64,
            init: None,
            mask: None,
        }
    }

    #[derive(Default)]
    struct Record {
        reports: Mutex<Vec<String>>,
        previews: Mutex<usize>,
        cancel: bool,
    }

    impl Progress for Record {
        fn report(&self, _: f32, m: &str) {
            self.reports.lock().unwrap().push(m.to_string());
        }
        fn cancelled(&self) -> bool {
            self.cancel
        }
        fn preview(&self, _: &Rgba8) {
            *self.previews.lock().unwrap() += 1;
        }
    }

    fn store() -> ModelStore {
        ModelStore::new(std::env::temp_dir().join(format!("photoka-comfy-store-{}", std::process::id())))
    }

    #[test]
    fn generates_over_the_websocket_with_progress_and_previews() {
        let (port, seen) = mock(true, false, false);
        let c = Client::new(port);
        assert!(c.system_stats().is_ok());
        let rec = Record::default();
        let out = c.generate(&store(), &req(), &rec).unwrap();
        assert_eq!((out.image.width, out.image.height), (64, 64), "resized to the frame");
        assert_eq!(&out.image.data[..4], &[200, 30, 30, 255]);
        assert!(rec.reports.lock().unwrap().iter().any(|m| m == "Step 2 of 2"));
        assert_eq!(*rec.previews.lock().unwrap(), 1);
        let s = seen.lock().unwrap();
        assert_eq!(s.uploads, 0, "text to image uploads nothing");
        assert!(s.prompts[0]["client_id"].as_str().unwrap().starts_with("photoka-"));
        assert_eq!(s.prompts[0]["prompt"]["22"]["class_type"], "PreviewImage");
    }

    #[test]
    fn inpaints_by_polling_when_the_websocket_is_missing() {
        let (port, seen) = mock(false, false, false);
        let mut r = req();
        r.init = Some(Rgba8::new(64, 64).unwrap());
        let mut m = Mask::new(64, 64).unwrap();
        m.data[0] = 255;
        r.mask = Some(m);
        let out = Client::new(port).generate(&store(), &r, &crate::NoProgress).unwrap();
        assert_eq!(out.seed, 7);
        let s = seen.lock().unwrap();
        assert_eq!(s.uploads, 1);
        let name = s.prompts[0]["prompt"]["7"]["inputs"]["image"].as_str().unwrap();
        assert!(name.starts_with("photoka-") && name.ends_with(".png"), "{name}");
        assert_eq!(s.prompts[0]["prompt"]["9"]["class_type"], "SetLatentNoiseMask");
    }

    #[test]
    fn rejections_cancels_and_dead_servers_are_errors() {
        let (port, _) = mock(true, true, false);
        let e = Client::new(port).generate(&store(), &req(), &crate::NoProgress).unwrap_err().to_string();
        assert!(e.contains("ckpt_name") && e.contains("failed validation"), "{e}");

        let (port, seen) = mock(true, false, true);
        let rec = Record { cancel: true, ..Default::default() };
        assert!(matches!(Client::new(port).generate(&store(), &req(), &rec), Err(MlError::Cancelled)));
        assert_eq!(seen.lock().unwrap().interrupts, 1);

        let dead = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let t = Instant::now();
        assert!(Client::new(dead).generate(&store(), &req(), &crate::NoProgress).is_err());
        assert!(Client::new(dead).system_stats().is_err());
        assert!(t.elapsed() < Duration::from_secs(15));
        let mut bad = req();
        bad.steps = 0;
        assert!(Client::new(dead).generate(&store(), &bad, &crate::NoProgress).is_err());
    }

    /// Installs ComfyUI into `PHOTOKA_COMFY_DIR` (network, ~4 GB), starts it on the models in
    /// `PHOTOKA_MODELS_DIR` and runs text to image and an inpaint; writes `target/comfy-e2e-*.png`.
    #[test]
    #[ignore = "installs ComfyUI and needs real models (PHOTOKA_COMFY_DIR, PHOTOKA_MODELS_DIR)"]
    fn real_install_start_and_generate() {
        struct Print;
        impl Progress for Print {
            fn report(&self, f: f32, m: &str) {
                eprintln!("{:5.1}% {m}", f * 100.0);
            }
            fn cancelled(&self) -> bool {
                false
            }
        }
        let inst = Install::new(std::env::var_os("PHOTOKA_COMFY_DIR").expect("set PHOTOKA_COMFY_DIR"));
        let models = std::path::PathBuf::from(std::env::var_os("PHOTOKA_MODELS_DIR").expect("set PHOTOKA_MODELS_DIR"));
        let t = Instant::now();
        inst.install(&Print).unwrap();
        eprintln!("install: {:?}, {} on disk", t.elapsed(), crate::download::human(inst.size_on_disk()));
        let t = Instant::now();
        let server = Server::start(&inst, &models, &Print).unwrap();
        eprintln!("start: {:?} (port {})", t.elapsed(), server.port());
        let store = ModelStore::new(&models);
        let mut r = req();
        r.checkpoint = "v1-5-pruned-emaonly-fp16.safetensors".into();
        r.prompt = "a watercolor painting of a lighthouse on a cliff at sunset".into();
        r.negative = "blurry, low quality".into();
        r.width = 512;
        r.height = 512;
        r.steps = 20;
        r.sampler = crate::params::Sampler::Dpmpp2mKarras;
        let rec = Record::default();
        let out = server.client().generate(&store, &r, &rec).unwrap();
        eprintln!("txt2img: {} ms, {} previews", out.elapsed_ms, rec.previews.lock().unwrap());
        let target = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target");
        std::fs::write(target.join("comfy-e2e-txt2img.png"), out.image.to_png().unwrap()).unwrap();
        let mut mask = Mask::new(512, 512).unwrap();
        for y in 0..512usize {
            for x in 0..256usize {
                mask.data[y * 512 + x] = 255;
            }
        }
        let mut r2 = r.clone();
        r2.prompt = "a red sailing boat on the sea".into();
        r2.init = Some(out.image.clone());
        r2.mask = Some(mask);
        let out2 = server.client().generate(&store, &r2, &crate::NoProgress).unwrap();
        eprintln!("inpaint: {} ms", out2.elapsed_ms);
        std::fs::write(target.join("comfy-e2e-inpaint.png"), out2.image.to_png().unwrap()).unwrap();
        let same_right = out
            .image
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .zip(out2.image.data.as_chunks::<4>().0)
            .enumerate()
            .filter(|(i, (a, b))| i % 512 > 300 && a[..3] == b[..3])
            .count();
        eprintln!("unchanged right-side pixels: {same_right}");
    }

    #[test]
    fn frames_carry_the_mask_as_transparency_and_events_parse() {
        let mut init = Rgba8::new(2, 1).unwrap();
        init.data.copy_from_slice(&[1, 2, 3, 255, 4, 5, 6, 255]);
        let mask = Mask::from_raw(2, 1, vec![255, 0]).unwrap();
        let back = Rgba8::decode(&frame_png(&init, Some(&mask)).unwrap()).unwrap();
        assert_eq!(back.data[3], 0);
        assert_eq!(back.data[7], 255);
        let p = crate::NoProgress;
        assert!(handle_event(r#"{"type":"executing","data":{"node":null,"prompt_id":"p1"}}"#, "p1", &p).unwrap());
        assert!(!handle_event(r#"{"type":"executing","data":{"node":null,"prompt_id":"other"}}"#, "p1", &p).unwrap());
        assert!(!handle_event("not json", "p1", &p).unwrap());
        let e =
            handle_event(r#"{"type":"execution_error","data":{"prompt_id":"p1","node_type":"KSampler","exception_message":"MPS out of memory"}}"#, "p1", &p);
        assert!(e.unwrap_err().to_string().contains("KSampler: MPS out of memory"));
        assert!(matches!(handle_event(r#"{"type":"execution_interrupted","data":{"prompt_id":"p1"}}"#, "p1", &p), Err(MlError::Cancelled)));
        show_preview(&[0, 0], &p);
        show_preview(&[0, 0, 0, 1, 0, 0, 0, 1, 9, 9], &p);
        assert_eq!(enc("a b/ü"), "a%20b%2F%C3%BC");
    }
}

//! The ComfyUI backend Photoka installs and runs itself.
//!
//! - `ai.comfy.status`: installed? running? on which port? (cheap; the UI polls it)
//! - `ai.comfy.install {accept: true}`: background job; downloads the pinned `uv` and ComfyUI and
//!   sets up Python with PyTorch (about 4 GB). ComfyUI is GPL-3.0 and runs as a separate program,
//!   so the user accepts that explicitly.
//! - `ai.comfy.uninstall`: stops it and removes the install folder.
//! - `ai.comfy.start` / `ai.comfy.stop`: the process on 127.0.0.1, reading Photoka's models folder.
//!   It also stops when the session ends.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use photoka_ml::comfy::{Install, Server, install};
use serde_json::{Value, json};

use crate::ai_cmds::{JobProgress, ml_err};
use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

/// The managed install and the running process.
#[derive(Debug)]
pub struct ComfyState {
    /// The install folder. Defaults to `PHOTOKA_COMFY_DIR` or
    /// `~/Library/Application Support/Photoka/comfyui`.
    pub dir: PathBuf,
    server: Arc<Mutex<Option<Server>>>,
    /// What a running job is doing to the install ("installing", "starting", "removing").
    busy: Arc<Mutex<Option<&'static str>>>,
    /// Why the process last stopped on its own.
    last_exit: Option<String>,
}

impl Default for ComfyState {
    fn default() -> Self {
        let dir = std::env::var_os("PHOTOKA_COMFY_DIR").map(PathBuf::from).unwrap_or_else(Install::default_dir);
        ComfyState { dir, server: Arc::default(), busy: Arc::default(), last_exit: None }
    }
}

impl ComfyState {
    pub fn install(&self) -> Install {
        Install::new(&self.dir)
    }

    /// The port ComfyUI answers on, when it's running.
    pub fn running_port(&mut self) -> Option<u16> {
        let mut slot = self.server.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(srv) = slot.as_mut()
            && !srv.is_alive()
        {
            self.last_exit = srv.exit_report();
            *slot = None;
        }
        slot.as_ref().map(Server::port)
    }

    pub fn busy(&self) -> Option<&'static str> {
        *self.busy.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Stop the process (if any). True when one was running.
    pub fn stop(&self) -> bool {
        self.server.lock().unwrap_or_else(PoisonError::into_inner).take().is_some()
    }

    fn claim(&self, what: &'static str) -> Result<BusyGuard> {
        let mut b = self.busy.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(doing) = *b {
            return Err(EngineError::Other(format!("ComfyUI is busy ({doing}); wait for it to finish")));
        }
        *b = Some(what);
        Ok(BusyGuard(Arc::clone(&self.busy)))
    }
}

/// Clears the busy flag however the job ends.
struct BusyGuard(Arc<Mutex<Option<&'static str>>>);

impl Drop for BusyGuard {
    fn drop(&mut self) {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = None;
    }
}

/// A client for `ai.generate` with `engine: "comfyui"`.
pub(crate) fn generation_client(s: &mut Session) -> Result<photoka_ml::comfy::Client> {
    match s.ai.comfy.running_port() {
        Some(port) => Ok(photoka_ml::comfy::Client::new(port)),
        None => match s.ai.comfy.last_exit.clone() {
            Some(why) => Err(EngineError::Other(format!("{why}\nStart it again from the Generate panel."))),
            None => Err(EngineError::Other("ComfyUI isn't running; start it from the Generate panel (ai.comfy.start)".into())),
        },
    }
}

fn status(s: &mut Session, p: &Value) -> Result<Value> {
    let port = s.ai.comfy.running_port();
    let inst = s.ai.comfy.install();
    let info = inst.info();
    let size = (p.get("size").and_then(Value::as_bool) == Some(true)).then(|| inst.size_on_disk());
    Ok(json!({
        "supported": photoka_ml::system::NATIVE_SUPPORTED,
        "installed": info.is_some(),
        "version": info.map(|i| i.comfyui),
        "pinned": {"comfyui": install::COMFY_VERSION, "uv": install::UV_VERSION},
        "running": port.is_some(),
        "port": port,
        "busy": s.ai.comfy.busy(),
        "dir": s.ai.comfy.dir.display().to_string(),
        "logPath": photoka_ml::comfy::server::log_path().display().to_string(),
        "license": install::COMFY_LICENSE,
        "page": install::COMFY_PAGE,
        "lastExit": s.ai.comfy.last_exit,
        "sizeBytes": size,
    }))
}

fn install_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ai.comfy.install";
    let inst = s.ai.comfy.install();
    if inst.is_installed() {
        return Ok(json!({"installed": true, "already": true}));
    }
    if !photoka_ml::system::NATIVE_SUPPORTED {
        return Err(EngineError::Other("the managed ComfyUI needs a Mac with Apple silicon".into()));
    }
    if p.get("accept").and_then(Value::as_bool) != Some(true) {
        return Err(crate::ai_cmds::bad(
            CMD,
            format!(
                "ComfyUI is {} software ({}). Photoka downloads it with Python and PyTorch (about 4 GB) into {} and runs it as a separate program; pass `accept: true` to install",
                install::COMFY_LICENSE,
                install::COMFY_PAGE,
                inst.dir().display()
            ),
        ));
    }
    let guard = s.ai.comfy.claim("installing")?;
    crate::jobs::run(
        s,
        "Installing ComfyUI",
        false,
        move |ctx| {
            let _guard = guard;
            inst.install(&JobProgress(ctx)).map_err(ml_err)
        },
        |_, ()| Ok(json!({"installed": true, "version": install::COMFY_VERSION})),
    )
}

fn uninstall(s: &mut Session, _p: &Value) -> Result<Value> {
    let guard = s.ai.comfy.claim("removing")?;
    let was_running = s.ai.comfy.stop();
    s.ai.comfy.last_exit = None;
    let inst = s.ai.comfy.install();
    crate::jobs::run(
        s,
        "Removing ComfyUI",
        false,
        move |ctx| {
            let _guard = guard;
            ctx.progress(0.0, "Removing ComfyUI");
            inst.uninstall().map_err(ml_err)
        },
        move |_, ()| Ok(json!({"installed": false, "stopped": was_running})),
    )
}

fn start(s: &mut Session, _p: &Value) -> Result<Value> {
    if let Some(port) = s.ai.comfy.running_port() {
        return Ok(json!({"running": true, "port": port, "already": true}));
    }
    let inst = s.ai.comfy.install();
    if !inst.is_installed() {
        return Err(EngineError::Other("ComfyUI isn't installed; install it from the Generate panel (ai.comfy.install)".into()));
    }
    let guard = s.ai.comfy.claim("starting")?;
    s.ai.comfy.last_exit = None;
    let models = s.ai.models_dir.clone();
    s.ai.store().ensure_layout().map_err(ml_err)?;
    let slot = Arc::clone(&s.ai.comfy.server);
    crate::jobs::run(
        s,
        "Starting ComfyUI",
        false,
        move |ctx| {
            let _guard = guard;
            Server::start(&inst, &models, &JobProgress(ctx)).map_err(ml_err)
        },
        move |_, server| {
            let port = server.port();
            *slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(server);
            Ok(json!({"running": true, "port": port}))
        },
    )
}

fn stop(s: &mut Session, _p: &Value) -> Result<Value> {
    s.ai.comfy.last_exit = None;
    Ok(json!({"stopped": s.ai.comfy.stop()}))
}

pub fn specs() -> Vec<CommandSpec> {
    fn always(_: &Session) -> std::result::Result<(), String> {
        Ok(())
    }
    vec![
        CommandSpec {
            id: "ai.comfy.status",
            label: "ComfyUI Status",
            menu: &[],
            shortcut: None,
            params: "{size:bool?} → {supported,installed,version,pinned,running,port,busy,dir,logPath,license,page,lastExit,sizeBytes}",
            enabled: always,
            run: status,
            journal: false,
        },
        CommandSpec {
            id: "ai.comfy.install",
            label: "Install ComfyUI",
            menu: &[],
            shortcut: None,
            params: "{accept:true} (ComfyUI is GPL-3.0; about 4 GB with Python and PyTorch) → background job → {installed,version}",
            enabled: always,
            run: install_cmd,
            journal: false,
        },
        CommandSpec {
            id: "ai.comfy.uninstall",
            label: "Remove ComfyUI",
            menu: &[],
            shortcut: None,
            params: "{} → stops it and removes the install folder (background job) → {installed:false,stopped}",
            enabled: always,
            run: uninstall,
            journal: false,
        },
        CommandSpec {
            id: "ai.comfy.start",
            label: "Start ComfyUI",
            menu: &[],
            shortcut: None,
            params: "{} → background job; ComfyUI on 127.0.0.1 reading the models folder → {running,port}",
            enabled: always,
            run: start,
            journal: false,
        },
        CommandSpec {
            id: "ai.comfy.stop",
            label: "Stop ComfyUI",
            menu: &[],
            shortcut: None,
            params: "{} → {stopped}",
            enabled: always,
            run: stop,
            journal: false,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(tag: &str) -> (Session, PathBuf) {
        let dir = std::env::temp_dir().join(format!("photoka-comfy-cmds-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut s = Session::new();
        s.ai.comfy.dir = dir.join("comfyui");
        s.ai.models_dir = dir.join("models");
        (s, dir)
    }

    #[test]
    fn nothing_installs_or_starts_by_itself() {
        let (mut s, dir) = session("idle");
        let st = s.execute("ai.comfy.status", json!({})).unwrap();
        assert_eq!(st["installed"], false);
        assert_eq!(st["running"], false);
        assert_eq!(st["pinned"]["comfyui"], install::COMFY_VERSION);
        assert!(!s.ai.comfy.dir.exists(), "status creates nothing");
        let e = s.execute("ai.comfy.install", json!({})).unwrap_err().to_string();
        if photoka_ml::system::NATIVE_SUPPORTED {
            assert!(e.contains("GPL-3.0") && e.contains("accept"), "{e}");
        }
        assert!(s.execute("ai.comfy.install", json!({"accept": "yes"})).is_err());
        assert!(s.execute("ai.comfy.start", json!({})).unwrap_err().to_string().contains("isn't installed"));
        assert_eq!(s.execute("ai.comfy.stop", json!({})).unwrap()["stopped"], false);
        assert!(generation_client(&mut s).unwrap_err().to_string().contains("isn't running"));
        s.execute("ai.comfy.uninstall", json!({})).unwrap();
        assert!(s.ai.comfy.busy().is_none(), "the busy flag clears");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn busy_installs_refuse_a_second_job() {
        let (s, dir) = session("busy");
        let g = s.ai.comfy.claim("installing").unwrap();
        assert!(s.ai.comfy.claim("starting").is_err());
        drop(g);
        assert!(s.ai.comfy.claim("starting").is_ok());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn comfy_generation_without_a_server_is_an_error() {
        let (mut s, dir) = session("gen");
        s.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        let e = s.execute("ai.generate", json!({"model": "x.safetensors", "prompt": "a cat", "engine": "comfyui"})).unwrap_err().to_string();
        assert!(e.contains("ComfyUI isn't running"), "{e}");
        let _ = std::fs::remove_dir_all(dir);
    }
}

//! The ComfyUI process: started on demand on the loopback interface with a free port, pointed at
//! Photoka's models folder, and killed when dropped (so it ends when Photoka quits).

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use super::Client;
use super::install::{Install, truncate};
use crate::{MlError, Progress, Result};

/// How long a first start may take (PyTorch's first import is slow).
const START_TIMEOUT: Duration = Duration::from_secs(300);

/// A running ComfyUI.
#[derive(Debug)]
pub struct Server {
    child: Child,
    port: u16,
    log: PathBuf,
}

/// `~/Library/Logs/Photoka/comfyui.log`.
pub fn log_path() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    home.join("Library").join("Logs").join("Photoka").join("comfyui.log")
}

/// ComfyUI's `extra_model_paths.yaml` for the shared models folder.
pub fn model_paths_yaml(models_dir: &Path) -> String {
    let base = models_dir.display().to_string().replace('\'', "''");
    format!(
        "# Written by Photoka: ComfyUI reads models from Photoka's models folder.\nphotoka:\n    base_path: '{base}'\n    is_default: true\n    checkpoints: checkpoints/\n    loras: loras/\n    vae: vae/\n    embeddings: embeddings/\n    upscale_models: upscale_models/\n"
    )
}

fn free_port() -> Result<u16> {
    let l = std::net::TcpListener::bind(("127.0.0.1", 0)).map_err(|e| MlError::Comfy(format!("no free local port: {e}")))?;
    l.local_addr().map(|a| a.port()).map_err(|e| MlError::Comfy(format!("no free local port: {e}")))
}

fn log_tail(log: &Path) -> String {
    let text = std::fs::read_to_string(log).unwrap_or_default();
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let tail = lines.get(lines.len().saturating_sub(12)..).unwrap_or_default().join("\n");
    truncate(&tail, 1500)
}

impl Server {
    /// Start ComfyUI from `install`, reading models from `models_dir`, and wait until it answers.
    pub fn start(install: &Install, models_dir: &Path, progress: &dyn Progress) -> Result<Server> {
        if !install.is_installed() {
            return Err(MlError::Comfy("ComfyUI isn't installed; install it from the Generate panel first".into()));
        }
        let config = install.dir().join("extra_model_paths.yaml");
        std::fs::write(&config, model_paths_yaml(models_dir)).map_err(|e| MlError::io(config.display(), e))?;
        let log = log_path();
        if let Some(dir) = log.parent() {
            std::fs::create_dir_all(dir).map_err(|e| MlError::io(dir.display(), e))?;
        }
        let out = std::fs::File::create(&log).map_err(|e| MlError::io(log.display(), e))?;
        let err = out.try_clone().map_err(|e| MlError::io(log.display(), e))?;
        let port = free_port()?;
        let mut cmd = Command::new(install.python());
        cmd.arg("main.py")
            .args(["--listen", "127.0.0.1", "--port", &port.to_string()])
            .arg("--extra-model-paths-config")
            .arg(&config)
            .args(["--disable-auto-launch", "--preview-method", "latent2rgb"])
            .arg("--input-directory")
            .arg(install.dir().join("input"))
            .arg("--output-directory")
            .arg(install.dir().join("output"))
            .arg("--temp-directory")
            .arg(install.dir().join("temp"))
            .arg("--user-directory")
            .arg(install.dir().join("user"))
            .env("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            .env("PYTHONUNBUFFERED", "1")
            .current_dir(install.source())
            .stdin(Stdio::null())
            .stdout(Stdio::from(out))
            .stderr(Stdio::from(err));
        for d in ["input", "output", "temp", "user"] {
            let p = install.dir().join(d);
            std::fs::create_dir_all(&p).map_err(|e| MlError::io(p.display(), e))?;
        }
        let child = cmd.spawn().map_err(|e| MlError::Comfy(format!("could not start ComfyUI: {e}")))?;
        let mut server = Server { child, port, log };
        server.wait_ready(progress)?;
        Ok(server)
    }

    fn wait_ready(&mut self, progress: &dyn Progress) -> Result<()> {
        let started = Instant::now();
        let client = self.client();
        loop {
            if progress.cancelled() {
                self.kill();
                return Err(MlError::Cancelled);
            }
            if let Ok(Some(status)) = self.child.try_wait() {
                return Err(MlError::Comfy(format!("ComfyUI stopped while starting ({status}); see {}:\n{}", self.log.display(), log_tail(&self.log))));
            }
            if client.system_stats().is_ok() {
                progress.report(1.0, &format!("ComfyUI is running on port {}", self.port));
                return Ok(());
            }
            let t = started.elapsed();
            if t > START_TIMEOUT {
                self.kill();
                return Err(MlError::Comfy(format!("ComfyUI didn't answer within {} s; see {}", START_TIMEOUT.as_secs(), self.log.display())));
            }
            let frac = 1.0 - (-3.0 * t.as_secs_f32() / 60.0).exp();
            progress.report(frac * 0.95, "Starting ComfyUI (the first start takes a minute)");
            std::thread::sleep(Duration::from_millis(500));
        }
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn log(&self) -> &Path {
        &self.log
    }

    pub fn client(&self) -> Client {
        Client::new(self.port)
    }

    /// False once the process has exited.
    pub fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// Why the process ended, with the end of its log.
    pub fn exit_report(&mut self) -> Option<String> {
        match self.child.try_wait() {
            Ok(Some(status)) => Some(format!("ComfyUI stopped ({status}); see {}:\n{}", self.log.display(), log_tail(&self.log))),
            _ => None,
        }
    }

    fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_paths_point_at_the_shared_folder() {
        let y = model_paths_yaml(Path::new("/Users/x/It's Models"));
        assert!(y.contains("base_path: '/Users/x/It''s Models'"));
        for k in ["checkpoints: checkpoints/", "loras: loras/", "vae: vae/", "embeddings: embeddings/"] {
            assert!(y.contains(k), "{k}");
        }
        assert!(free_port().unwrap() > 0);
        assert!(log_path().ends_with("Logs/Photoka/comfyui.log"));
    }

    #[test]
    fn starting_without_an_install_is_an_error() {
        let d = std::env::temp_dir().join(format!("photoka-comfy-noinst-{}", std::process::id()));
        let e = Server::start(&Install::new(&d), &d, &crate::NoProgress).unwrap_err();
        assert!(e.to_string().contains("isn't installed"), "{e}");
    }
}

//! The ComfyUI install Photoka manages: a pinned `uv`, a pinned ComfyUI source release and a
//! Python 3.12 virtual environment with ComfyUI's requirements (PyTorch with Metal/MPS).
//!
//! Everything lives in one folder (`~/Library/Application Support/Photoka/comfyui/` by default):
//! `bin/uv`, `ComfyUI/`, `python/` (the interpreter uv fetched), `venv/`, `cache/` and
//! `installed.json`, so uninstalling is removing that folder. ComfyUI (GPL-3.0) is downloaded at
//! the user's request and run as a separate process; Photoka neither bundles nor links it.

use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::download::{DownloadRequest, download};
use crate::{MlError, Progress, Result};

/// The `uv` release Photoka installs ComfyUI's Python environment with.
pub const UV_VERSION: &str = "0.13.0";
const UV_URL: &str = "https://github.com/astral-sh/uv/releases/download/0.13.0/uv-aarch64-apple-darwin.tar.gz";
const UV_SIZE: u64 = 14_692_549;
const UV_SHA256: &str = "a9c1b29002cf3c83f07fa9cd8a887a3be0107d90e23189721221e7257db8e3d6";

/// The ComfyUI release Photoka installs.
pub const COMFY_VERSION: &str = "0.39.0";
const COMFY_URL: &str = "https://github.com/Comfy-Org/ComfyUI/archive/refs/tags/v0.39.0.tar.gz";
const COMFY_SIZE: u64 = 12_627_617;
const COMFY_SHA256: &str = "095d95805bdf36e73bbd15d6d2ee15fa627708679a2e67b5f467db4c57e7ff0a";
pub const COMFY_LICENSE: &str = "GPL-3.0";
pub const COMFY_PAGE: &str = "https://github.com/Comfy-Org/ComfyUI";
const PYTHON_VERSION: &str = "3.12";
/// Free space an install needs (PyTorch and friends come to about 4 GB).
const INSTALL_SPACE: u64 = 8 << 30;
/// Largest single file accepted from an archive.
const MAX_ENTRY: u64 = 256 << 20;

/// What `installed.json` records.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstallInfo {
    pub comfyui: String,
    pub uv: String,
}

/// One managed install folder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Install {
    dir: PathBuf,
}

impl Install {
    pub fn new(dir: impl Into<PathBuf>) -> Install {
        Install { dir: dir.into() }
    }

    /// `~/Library/Application Support/Photoka/comfyui`.
    pub fn default_dir() -> PathBuf {
        crate::system::app_support_dir().join("comfyui")
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn uv(&self) -> PathBuf {
        self.dir.join("bin").join("uv")
    }

    pub fn source(&self) -> PathBuf {
        self.dir.join("ComfyUI")
    }

    pub fn python(&self) -> PathBuf {
        self.dir.join("venv").join("bin").join("python")
    }

    fn marker(&self) -> PathBuf {
        self.dir.join("installed.json")
    }

    /// The finished install, if there is one.
    pub fn info(&self) -> Option<InstallInfo> {
        let text = std::fs::read_to_string(self.marker()).ok()?;
        let info: InstallInfo = serde_json::from_str(&text).ok()?;
        (self.python().is_file() && self.source().join("main.py").is_file()).then_some(info)
    }

    pub fn is_installed(&self) -> bool {
        self.info().is_some()
    }

    /// Install (or finish a partial install). Needs the network once; cancelling keeps what is
    /// done, so the next call continues.
    pub fn install(&self, progress: &dyn Progress) -> Result<()> {
        if !crate::system::NATIVE_SUPPORTED {
            return Err(MlError::Unsupported("the managed ComfyUI needs a Mac with Apple silicon".into()));
        }
        if self.is_installed() {
            progress.report(1.0, "ComfyUI is installed");
            return Ok(());
        }
        std::fs::create_dir_all(&self.dir).map_err(|e| MlError::io(self.dir.display(), e))?;
        if let Some(free) = crate::system::available_space(&self.dir)
            && free < INSTALL_SPACE
        {
            return Err(MlError::DiskFull(format!(
                "installing ComfyUI needs about {} free; {} is available",
                crate::download::human(INSTALL_SPACE),
                crate::download::human(free)
            )));
        }
        let downloads = self.dir.join("downloads");
        if !self.uv().is_file() {
            let archive = downloads.join(format!("uv-{UV_VERSION}.tar.gz"));
            let req = DownloadRequest { url: UV_URL.into(), dest: archive.clone(), size: UV_SIZE, sha256: UV_SHA256.into(), bearer: None };
            download(&req, &Scaled { inner: progress, start: 0.0, span: 0.04 })?;
            extract_file(&archive, "uv", &self.uv())?;
        }
        if !self.source().join("main.py").is_file() {
            let archive = downloads.join(format!("ComfyUI-{COMFY_VERSION}.tar.gz"));
            let req = DownloadRequest { url: COMFY_URL.into(), dest: archive.clone(), size: COMFY_SIZE, sha256: COMFY_SHA256.into(), bearer: None };
            download(&req, &Scaled { inner: progress, start: 0.04, span: 0.04 })?;
            progress.report(0.08, "Unpacking ComfyUI");
            extract_tree(&archive, &self.source())?;
        }
        progress.check()?;
        let log = self.dir.join("install.log");
        progress.report(0.09, "Setting up Python");
        let mut venv = self.uv_command();
        venv.arg("venv").arg("--python").arg(PYTHON_VERSION).arg("--allow-existing").arg(self.dir.join("venv"));
        run_logged(venv, &log, &Scaled { inner: progress, start: 0.09, span: 0.06 }, Duration::from_secs(60), "Setting up Python")?;
        let mut pip = self.uv_command();
        pip.arg("pip").arg("install").arg("--python").arg(self.python()).arg("-r").arg(self.source().join("requirements.txt"));
        run_logged(pip, &log, &Scaled { inner: progress, start: 0.15, span: 0.83 }, Duration::from_secs(600), "Installing PyTorch and ComfyUI's packages")?;
        let info = InstallInfo { comfyui: COMFY_VERSION.into(), uv: UV_VERSION.into() };
        let text = serde_json::to_string_pretty(&info).map_err(|e| MlError::Other(e.to_string()))?;
        std::fs::write(self.marker(), text).map_err(|e| MlError::io(self.marker().display(), e))?;
        let _ = std::fs::remove_dir_all(&downloads);
        progress.report(1.0, "ComfyUI is installed");
        Ok(())
    }

    /// Remove the whole install folder.
    pub fn uninstall(&self) -> Result<()> {
        if !self.dir.exists() {
            return Ok(());
        }
        // Refuse anything that doesn't look like ours, so a bad path can't erase other files.
        let ours = self.marker().exists() || self.uv().exists() || self.source().exists() || self.dir.join("downloads").exists();
        if !ours {
            return Err(MlError::BadParams(format!("{} is not a Photoka ComfyUI folder", self.dir.display())));
        }
        std::fs::remove_dir_all(&self.dir).map_err(|e| MlError::io(self.dir.display(), e))
    }

    /// Bytes the install takes on disk.
    pub fn size_on_disk(&self) -> u64 {
        dir_size(&self.dir, 0)
    }

    fn uv_command(&self) -> Command {
        let mut c = Command::new(self.uv());
        c.env("UV_PYTHON_INSTALL_DIR", self.dir.join("python"))
            .env("UV_CACHE_DIR", self.dir.join("cache"))
            .env("UV_NO_CONFIG", "1")
            .env("UV_PYTHON_PREFERENCE", "only-managed")
            .current_dir(&self.dir);
        c
    }
}

fn dir_size(p: &Path, depth: u32) -> u64 {
    if depth > 64 {
        return 0;
    }
    let Ok(rd) = std::fs::read_dir(p) else { return 0 };
    rd.flatten()
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => dir_size(&e.path(), depth + 1),
            Ok(t) if t.is_file() => e.metadata().map_or(0, |m| m.len()),
            _ => 0,
        })
        .fold(0u64, u64::saturating_add)
}

/// Reports a sub-range of an outer progress.
pub(crate) struct Scaled<'a> {
    pub inner: &'a dyn Progress,
    pub start: f32,
    pub span: f32,
}

impl Progress for Scaled<'_> {
    fn report(&self, fraction: f32, message: &str) {
        self.inner.report(self.start + self.span * fraction.clamp(0.0, 1.0), message);
    }
    fn cancelled(&self) -> bool {
        self.inner.cancelled()
    }
}

/// Paths inside an archive, without the top folder, rejecting anything that could escape.
fn inner_path(p: &Path) -> Option<PathBuf> {
    let mut out = PathBuf::new();
    for (i, c) in p.components().enumerate() {
        match c {
            std::path::Component::Normal(s) if i > 0 => out.push(s),
            std::path::Component::Normal(_) => {}
            std::path::Component::CurDir => {}
            _ => return None,
        }
    }
    (!out.as_os_str().is_empty()).then_some(out)
}

/// Extract the file named `name` (anywhere in the archive) to `dest`, executable.
fn extract_file(archive: &Path, name: &str, dest: &Path) -> Result<()> {
    let f = std::fs::File::open(archive).map_err(|e| MlError::io(archive.display(), e))?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(f));
    let entries = tar.entries().map_err(|e| MlError::io(archive.display(), e))?;
    for entry in entries {
        let mut entry = entry.map_err(|e| MlError::io(archive.display(), e))?;
        let path = entry.path().map_err(|e| MlError::io(archive.display(), e))?.into_owned();
        if path.file_name().is_none_or(|n| n != name) || !entry.header().entry_type().is_file() {
            continue;
        }
        if entry.size() > MAX_ENTRY {
            return Err(MlError::BadModel(format!("{name} in {} is unexpectedly large", archive.display())));
        }
        let dir = dest.parent().ok_or_else(|| MlError::BadParams("bad install path".into()))?;
        std::fs::create_dir_all(dir).map_err(|e| MlError::io(dir.display(), e))?;
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).map_err(|e| MlError::io(archive.display(), e))?;
        std::fs::write(dest, bytes).map_err(|e| MlError::io(dest.display(), e))?;
        set_executable(dest)?;
        return Ok(());
    }
    Err(MlError::BadModel(format!("{} has no `{name}`", archive.display())))
}

#[cfg(unix)]
fn set_executable(p: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o755)).map_err(|e| MlError::io(p.display(), e))
}

#[cfg(not(unix))]
fn set_executable(_p: &Path) -> Result<()> {
    Ok(())
}

/// Extract an archive's single top folder into `dest` (replacing it), through a staging folder
/// so a failed extract leaves nothing half-written.
fn extract_tree(archive: &Path, dest: &Path) -> Result<()> {
    let staging = dest.with_extension("staging");
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(|e| MlError::io(staging.display(), e))?;
    let f = std::fs::File::open(archive).map_err(|e| MlError::io(archive.display(), e))?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(f));
    let entries = tar.entries().map_err(|e| MlError::io(archive.display(), e))?;
    for entry in entries {
        let mut entry = entry.map_err(|e| MlError::io(archive.display(), e))?;
        let kind = entry.header().entry_type();
        let path = entry.path().map_err(|e| MlError::io(archive.display(), e))?.into_owned();
        let Some(rel) = inner_path(&path) else { continue };
        let out = staging.join(&rel);
        if kind.is_dir() {
            std::fs::create_dir_all(&out).map_err(|e| MlError::io(out.display(), e))?;
        } else if kind.is_file() {
            if entry.size() > MAX_ENTRY {
                return Err(MlError::BadModel(format!("{} in {} is unexpectedly large", rel.display(), archive.display())));
            }
            if let Some(dir) = out.parent() {
                std::fs::create_dir_all(dir).map_err(|e| MlError::io(dir.display(), e))?;
            }
            let mut file = std::fs::File::create(&out).map_err(|e| MlError::io(out.display(), e))?;
            std::io::copy(&mut entry, &mut file).map_err(|e| MlError::io(out.display(), e))?;
        }
        // Links and special files aren't needed and could point outside the folder.
    }
    let _ = std::fs::remove_dir_all(dest);
    std::fs::rename(&staging, dest).map_err(|e| MlError::io(dest.display(), e))
}

/// Run `cmd`, appending its output to `log` and showing its latest line as the status, until it
/// exits or the user cancels (which kills it). The bar creeps towards the end over `typical`.
pub(crate) fn run_logged(mut cmd: Command, log: &Path, progress: &dyn Progress, typical: Duration, what: &str) -> Result<()> {
    use std::io::Write;
    let file = std::fs::OpenOptions::new().create(true).append(true).open(log).map_err(|e| MlError::io(log.display(), e))?;
    let shared = Arc::new(Mutex::new((file, String::new(), Vec::<String>::new())));
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| MlError::Comfy(format!("{what}: could not start {}: {e}", cmd.get_program().to_string_lossy())))?;
    let mut readers = Vec::new();
    let streams: [Option<Box<dyn Read + Send>>; 2] =
        [child.stdout.take().map(|s| Box::new(s) as Box<dyn Read + Send>), child.stderr.take().map(|s| Box::new(s) as Box<dyn Read + Send>)];
    for stream in streams.into_iter().flatten() {
        let shared = Arc::clone(&shared);
        readers.push(std::thread::spawn(move || {
            for line in std::io::BufReader::new(stream).lines().map_while(std::result::Result::ok) {
                let mut g = shared.lock().unwrap_or_else(PoisonError::into_inner);
                let _ = writeln!(g.0, "{line}");
                let trimmed = line.trim().to_string();
                if !trimmed.is_empty() {
                    g.2.push(trimmed.clone());
                    if g.2.len() > 20 {
                        g.2.remove(0);
                    }
                    g.1 = trimmed;
                }
            }
        }));
    }
    let started = Instant::now();
    let status = loop {
        if progress.cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(MlError::Cancelled);
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(e) => return Err(MlError::Comfy(format!("{what}: {e}"))),
        }
        let t = started.elapsed().as_secs_f32() / typical.as_secs_f32().max(1.0);
        let frac = 1.0 - (-2.0 * t).exp();
        let last = shared.lock().unwrap_or_else(PoisonError::into_inner).1.clone();
        let msg = if last.is_empty() { what.to_string() } else { format!("{what}: {}", truncate(&last, 80)) };
        progress.report(frac * 0.98, &msg);
        std::thread::sleep(Duration::from_millis(200));
    };
    for r in readers {
        let _ = r.join();
    }
    if !status.success() {
        let tail = shared.lock().unwrap_or_else(PoisonError::into_inner).2.join("\n");
        return Err(MlError::Comfy(format!("{what} failed ({status}); see {}:\n{}", log.display(), truncate(&tail, 1500))));
    }
    Ok(())
}

pub(crate) fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("photoka-comfy-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn tar_gz(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut b = tar::Builder::new(flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast()));
        for (path, data) in entries {
            let mut h = tar::Header::new_gnu();
            h.set_size(data.len() as u64);
            h.set_mode(0o644);
            h.set_entry_type(tar::EntryType::Regular);
            // Write the name bytes directly so hostile paths can be tested.
            let name = h.as_old_mut().name.as_mut();
            name.iter_mut().for_each(|c| *c = 0);
            name[..path.len()].copy_from_slice(path.as_bytes());
            h.set_cksum();
            b.append(&h, *data).unwrap();
        }
        b.into_inner().unwrap().finish().unwrap()
    }

    #[test]
    fn archives_extract_without_escaping() {
        let d = tmp("tar");
        let archive = d.join("a.tar.gz");
        std::fs::write(
            &archive,
            tar_gz(&[("ComfyUI-0.39.0/main.py", b"print(1)"), ("ComfyUI-0.39.0/comfy/x.py", b"x"), ("ComfyUI-0.39.0/../../evil", b"no"), ("/abs", b"no")]),
        )
        .unwrap();
        let dest = d.join("ComfyUI");
        extract_tree(&archive, &dest).unwrap();
        assert_eq!(std::fs::read(dest.join("main.py")).unwrap(), b"print(1)");
        assert!(dest.join("comfy/x.py").is_file());
        assert!(!d.join("evil").exists() && !dest.join("evil").exists());
        assert!(!dest.with_extension("staging").exists());

        let uv = d.join("uvpkg.tar.gz");
        std::fs::write(&uv, tar_gz(&[("uv-aarch64-apple-darwin/uvx", b"x"), ("uv-aarch64-apple-darwin/uv", b"#!bin")])).unwrap();
        extract_file(&uv, "uv", &d.join("bin/uv")).unwrap();
        assert_eq!(std::fs::read(d.join("bin/uv")).unwrap(), b"#!bin");
        assert!(extract_file(&uv, "python", &d.join("bin/python")).is_err());
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn status_and_uninstall_are_safe() {
        let d = tmp("status");
        let inst = Install::new(d.join("comfyui"));
        assert!(!inst.is_installed());
        assert!(inst.uninstall().is_ok(), "nothing to remove");
        std::fs::create_dir_all(inst.dir().join("stranger")).unwrap();
        assert!(inst.uninstall().is_err(), "a folder that isn't ours is kept");
        std::fs::create_dir_all(inst.source()).unwrap();
        std::fs::write(inst.source().join("main.py"), "").unwrap();
        std::fs::create_dir_all(inst.python().parent().unwrap()).unwrap();
        std::fs::write(inst.python(), "").unwrap();
        std::fs::write(inst.dir().join("installed.json"), r#"{"comfyui":"0.39.0","uv":"0.13.0"}"#).unwrap();
        assert_eq!(inst.info().unwrap().comfyui, "0.39.0");
        assert!(inst.size_on_disk() > 0);
        inst.uninstall().unwrap();
        assert!(!inst.dir().exists());
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn child_processes_log_fail_and_cancel() {
        let d = tmp("run");
        let log = d.join("log.txt");
        let mut ok = Command::new("/bin/sh");
        ok.args(["-c", "echo hello; echo oops >&2"]);
        run_logged(ok, &log, &crate::NoProgress, Duration::from_secs(1), "test").unwrap();
        let text = std::fs::read_to_string(&log).unwrap();
        assert!(text.contains("hello") && text.contains("oops"));
        let mut bad = Command::new("/bin/sh");
        bad.args(["-c", "echo broken; exit 3"]);
        let e = run_logged(bad, &log, &crate::NoProgress, Duration::from_secs(1), "test").unwrap_err().to_string();
        assert!(e.contains("broken"), "{e}");
        struct Cancelled;
        impl Progress for Cancelled {
            fn report(&self, _: f32, _: &str) {}
            fn cancelled(&self) -> bool {
                true
            }
        }
        let mut slow = Command::new("/bin/sleep");
        slow.arg("30");
        let t = Instant::now();
        assert!(matches!(run_logged(slow, &log, &Cancelled, Duration::from_secs(1), "test"), Err(MlError::Cancelled)));
        assert!(t.elapsed() < Duration::from_secs(5));
        assert!(run_logged(Command::new("/nonexistent/tool"), &log, &crate::NoProgress, Duration::from_secs(1), "test").is_err());
        let _ = std::fs::remove_dir_all(d);
    }
}

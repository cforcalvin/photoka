//! What the machine can do: Apple silicon, memory, free disk space, default folders.

use std::path::{Path, PathBuf};

/// The native engine runs on Apple silicon (Metal) only.
pub const NATIVE_SUPPORTED: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

/// Why the native engine is unavailable on this build, if it is.
pub fn native_unavailable_reason() -> Option<&'static str> {
    (!NATIVE_SUPPORTED).then_some("AI generation needs a Mac with Apple silicon (M1 or later); this Mac has an Intel processor")
}

/// Physical memory in bytes (`hw.memsize`), if it can be read.
pub fn physical_memory() -> Option<u64> {
    let out = std::process::Command::new("/usr/sbin/sysctl").args(["-n", "hw.memsize"]).output().ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

/// Free bytes on the volume holding `path` (or its nearest existing ancestor), from `df -k`.
pub fn available_space(path: &Path) -> Option<u64> {
    let mut p = path;
    while !p.exists() {
        p = p.parent()?;
    }
    let out = std::process::Command::new("/bin/df").arg("-k").arg(p).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().nth(1)?;
    let avail_kb: u64 = line.split_whitespace().nth(3)?.parse().ok()?;
    avail_kb.checked_mul(1024)
}

/// `~/Library/Application Support/Photoka`.
pub fn app_support_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    home.join("Library").join("Application Support").join("Photoka")
}

/// Default models folder.
pub fn default_models_dir() -> PathBuf {
    app_support_dir().join("models")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probes_answer_on_this_machine() {
        assert!(physical_memory().unwrap_or(0) > 0);
        assert!(available_space(&std::env::temp_dir().join("does/not/exist")).is_some());
        assert!(default_models_dir().ends_with("Photoka/models"));
        assert_eq!(native_unavailable_reason().is_none(), NATIVE_SUPPORTED);
    }
}

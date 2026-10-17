//! What the machine can do: Apple silicon, memory, free disk space, default folders.

use std::path::{Path, PathBuf};

use serde::Serialize;

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

/// Chip / model label for UI (`hw.model`, else `machdep.cpu.brand_string`), if readable.
pub fn chip_label() -> Option<String> {
    for key in ["hw.model", "machdep.cpu.brand_string"] {
        let out = std::process::Command::new("/usr/sbin/sysctl").args(["-n", key]).output().ok()?;
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !s.is_empty() {
            return Some(s);
        }
    }
    None
}

/// Snapshot used to rank catalog models and pick generation defaults.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareProfile {
    pub ram_bytes: Option<u64>,
    pub disk_free_bytes: Option<u64>,
    pub apple_silicon: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chip: Option<String>,
}

impl HardwareProfile {
    /// Probe this machine for `models_dir`'s volume.
    pub fn probe(models_dir: &Path) -> Self {
        HardwareProfile {
            ram_bytes: physical_memory(),
            disk_free_bytes: available_space(models_dir),
            apple_silicon: NATIVE_SUPPORTED,
            chip: chip_label(),
        }
    }

    pub fn ram_gb(&self) -> Option<u64> {
        self.ram_bytes.map(|b| b >> 30)
    }

    pub fn disk_free_gb(&self) -> Option<u64> {
        self.disk_free_bytes.map(|b| b >> 30)
    }

    /// JSON block for `ai.catalog.list` / UI summaries.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "ramGb": self.ram_gb(),
            "diskFreeGb": self.disk_free_gb(),
            "ramBytes": self.ram_bytes,
            "diskFreeBytes": self.disk_free_bytes,
            "appleSilicon": self.apple_silicon,
            "chip": self.chip,
        })
    }

    /// Default AI upscale scale factor for this Mac.
    pub fn default_upscale_scale(&self) -> u32 {
        match self.ram_bytes {
            Some(b) if b >= (16 << 30) => 4,
            _ => 2,
        }
    }

    /// Default Real-ESRGAN / ComfyUI tile size (pixels) for peak memory.
    pub fn default_upscale_tile(&self) -> u32 {
        match self.ram_bytes {
            Some(b) if b >= (16 << 30) => 512,
            Some(b) if b >= (12 << 30) => 384,
            _ => 256,
        }
    }

    /// Fewer diffusion steps when memory is tight.
    pub fn prefer_fast_generate(&self) -> bool {
        self.ram_bytes.is_some_and(|b| b <= (12 << 30))
    }
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
        let hw = HardwareProfile::probe(&std::env::temp_dir());
        assert!(hw.ram_bytes.unwrap_or(0) > 0);
        assert!(hw.default_upscale_scale() == 2 || hw.default_upscale_scale() == 4);
        assert!(matches!(hw.default_upscale_tile(), 256 | 384 | 512));
    }
}

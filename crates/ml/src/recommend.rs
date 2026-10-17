//! Rank catalog models for this Mac (RAM, free disk, Apple Silicon) and build a starter pack.

use crate::catalog::{CatalogEntry, Family, ModelKind};
use crate::system::HardwareProfile;

/// Spare bytes kept free on the models volume after a download.
pub const DISK_SPARE: u64 = 512 << 20;

/// Minimum total RAM to comfortably run a family (aligned with native `check_memory` + headroom).
pub fn min_ram_bytes(family: Family, kind: ModelKind) -> u64 {
    match kind {
        ModelKind::Upscale => 4 << 30,
        ModelKind::Tokenizer | ModelKind::Embedding => 0,
        ModelKind::Vae | ModelKind::Lora => 4 << 30,
        ModelKind::Checkpoint => match family {
            Family::SdTurbo => 8 << 30,
            Family::Sd15 | Family::Sd15Inpaint | Family::Any => 8 << 30,
            Family::Upscale => 4 << 30,
        },
    }
}

/// Whether `entry` fits this Mac's RAM for recommendation purposes.
pub fn fits_ram(entry: &CatalogEntry, hw: &HardwareProfile) -> bool {
    let need = min_ram_bytes(entry.family, entry.kind);
    if need == 0 {
        return true;
    }
    match hw.ram_bytes {
        None => true,
        Some(ram) => ram >= need,
    }
}

/// Whether there is enough free disk for this file (+ spare).
pub fn fits_disk(entry: &CatalogEntry, hw: &HardwareProfile) -> bool {
    let need = entry.size.saturating_add(DISK_SPARE);
    match hw.disk_free_bytes {
        None => true,
        Some(free) => free >= need,
    }
}

fn has_tag(entry: &CatalogEntry, tag: &str) -> bool {
    entry.tags.iter().any(|t| t == tag)
}

fn is_upscale_x2(entry: &CatalogEntry) -> bool {
    entry.kind == ModelKind::Upscale && (has_tag(entry, "x2") || entry.id.contains("x2") || entry.file_name.contains("x2"))
}

fn is_upscale_x4(entry: &CatalogEntry) -> bool {
    entry.kind == ModelKind::Upscale && (has_tag(entry, "x4") || entry.id.contains("x4") || entry.file_name.contains("x4"))
}

/// Score for sorting Suggested (higher = better for this Mac).
pub fn score(entry: &CatalogEntry, hw: &HardwareProfile, installed: bool) -> i32 {
    let mut s: i32 = 0;
    let ram_ok = fits_ram(entry, hw);
    let disk_ok = fits_disk(entry, hw);
    if !disk_ok {
        s -= 1000;
    }
    if !ram_ok {
        s -= 500;
    }
    if installed {
        s -= 200;
    }
    if entry.kind == ModelKind::Tokenizer {
        // Keep tokenizers after checkpoints in the ranked list (UI already splits them out).
        s -= 50;
    }
    if ram_ok && disk_ok {
        let ram = hw.ram_bytes.unwrap_or(u64::MAX);
        // On roomy Macs, catalog `recommended` quality wins; on tight RAM, Turbo wins for speed.
        if has_tag(entry, "recommended") && ram >= (16 << 30) {
            s += 100;
        } else if has_tag(entry, "recommended") {
            s += 40;
        }
        if entry.family == Family::SdTurbo && ram <= (12 << 30) {
            s += 120;
        }
        if matches!(entry.family, Family::Sd15 | Family::Sd15Inpaint) && has_tag(entry, "recommended") && ram >= (16 << 30) {
            s += 40;
        }
        if is_upscale_x2(entry) && ram <= (12 << 30) {
            s += 90;
        }
        if is_upscale_x4(entry) && ram >= (16 << 30) {
            s += 70;
        }
        if is_upscale_x2(entry) && ram >= (16 << 30) {
            s += 20; // still useful, below x4
        }
        // Prefer smaller downloads when disk is tight (< 8 GB free after this file would leave little).
        if let Some(free) = hw.disk_free_bytes {
            let after = free.saturating_sub(entry.size);
            if after < (8 << 30) {
                // Smaller files score higher (cap contribution).
                let gb = (entry.size >> 20).min(8000) as i32; // MiB
                s += 40 - gb / 200;
            }
        }
    }
    s
}

/// Short reason shown under a Best card.
pub fn recommend_reason(entry: &CatalogEntry, hw: &HardwareProfile) -> String {
    if !fits_disk(entry, hw) {
        let need = (entry.size.saturating_add(DISK_SPARE) + ((1 << 30) - 1)) >> 30;
        return format!("Needs about {need} GB free");
    }
    if !fits_ram(entry, hw) {
        let need = min_ram_bytes(entry.family, entry.kind) >> 30;
        return format!("May be too large for this Mac’s memory (needs about {need} GB)");
    }
    let ram_gb = hw.ram_gb().unwrap_or(0);
    if entry.kind == ModelKind::Upscale {
        if is_upscale_x2(entry) {
            return format!("Fits your {ram_gb} GB Mac · lighter upscale");
        }
        if is_upscale_x4(entry) {
            return format!("Best quality upscale for {ram_gb} GB+");
        }
        return format!("Fits your {ram_gb} GB Mac");
    }
    if entry.family == Family::SdTurbo {
        return format!("Fits your {ram_gb} GB Mac · fastest");
    }
    if has_tag(entry, "recommended") {
        if ram_gb >= 16 {
            return format!("Best quality for {ram_gb} GB+");
        }
        return format!("Fits your {ram_gb} GB Mac · recommended");
    }
    if has_tag(entry, "inpainting") {
        return format!("Fits your {ram_gb} GB Mac · for removing objects");
    }
    format!("Fits your {ram_gb} GB Mac")
}

/// Stable sort: score desc, then original index.
pub fn rank_indices(entries: &[CatalogEntry], hw: &HardwareProfile, installed: &dyn Fn(&CatalogEntry) -> bool) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..entries.len()).collect();
    idx.sort_by(|&a, &b| {
        let sa = score(&entries[a], hw, installed(&entries[a]));
        let sb = score(&entries[b], hw, installed(&entries[b]));
        sb.cmp(&sa).then_with(|| a.cmp(&b))
    });
    idx
}

/// Top checkpoint picks that fit, for `recommended_for_hw` badges (max `limit`).
pub fn best_checkpoint_ids(entries: &[CatalogEntry], hw: &HardwareProfile, installed: &dyn Fn(&CatalogEntry) -> bool, limit: usize) -> Vec<String> {
    rank_indices(entries, hw, installed)
        .into_iter()
        .filter(|&i| {
            let e = &entries[i];
            e.kind == ModelKind::Checkpoint && fits_ram(e, hw) && fits_disk(e, hw)
        })
        .take(limit)
        .map(|i| entries[i].id.clone())
        .collect()
}

/// Mark upscalers that fit and match the preferred scale for this Mac.
pub fn best_upscale_ids(entries: &[CatalogEntry], hw: &HardwareProfile, limit: usize) -> Vec<String> {
    let prefer_x2 = hw.default_upscale_scale() == 2;
    let mut ids = Vec::new();
    for e in entries {
        if e.kind != ModelKind::Upscale || !fits_ram(e, hw) || !fits_disk(e, hw) {
            continue;
        }
        let match_scale = if prefer_x2 { is_upscale_x2(e) } else { is_upscale_x4(e) };
        if match_scale {
            ids.push(e.id.clone());
        }
        if ids.len() >= limit {
            break;
        }
    }
    if ids.is_empty() {
        for e in entries {
            if e.kind == ModelKind::Upscale && fits_ram(e, hw) && fits_disk(e, hw) {
                ids.push(e.id.clone());
                if ids.len() >= limit {
                    break;
                }
            }
        }
    }
    ids
}

/// Checkpoint + missing companions (tokenizer, recommended VAE) whose sizes sum to fit disk.
#[derive(Clone, Debug, PartialEq)]
pub struct StarterPack {
    pub ids: Vec<String>,
    pub total_bytes: u64,
}

pub fn starter_pack(entries: &[CatalogEntry], hw: &HardwareProfile, installed: &dyn Fn(&CatalogEntry) -> bool) -> Option<StarterPack> {
    let ranked = rank_indices(entries, hw, installed);
    let ckpt = ranked.into_iter().find(|&i| {
        let e = &entries[i];
        e.kind == ModelKind::Checkpoint && fits_ram(e, hw) && !installed(e)
    })?;
    let mut ids = vec![entries[ckpt].id.clone()];
    let mut total = if fits_disk(&entries[ckpt], hw) { entries[ckpt].size } else { return None };

    // Tokenizer if missing.
    if let Some(tok) = entries.iter().find(|e| e.kind == ModelKind::Tokenizer)
        && !installed(tok)
    {
        let next = total.saturating_add(tok.size).saturating_add(DISK_SPARE);
        if hw.disk_free_bytes.is_none_or(|f| f >= next) {
            ids.push(tok.id.clone());
            total = total.saturating_add(tok.size);
        }
    }

    // Optional VAE tagged quality / recommended for the checkpoint family.
    if let Some(vae) = entries.iter().find(|e| {
        e.kind == ModelKind::Vae && e.family.compatible_with(entries[ckpt].family) && (has_tag(e, "quality") || has_tag(e, "recommended")) && !installed(e)
    }) {
        let next = total.saturating_add(vae.size).saturating_add(DISK_SPARE);
        if hw.disk_free_bytes.is_none_or(|f| f >= next) {
            ids.push(vae.id.clone());
            total = total.saturating_add(vae.size);
        }
    }

    Some(StarterPack { ids, total_bytes: total })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{Defaults, Source};

    fn entry(id: &str, kind: ModelKind, family: Family, size: u64, tags: &[&str]) -> CatalogEntry {
        CatalogEntry {
            id: id.into(),
            name: id.into(),
            description: String::new(),
            kind,
            family,
            tags: tags.iter().map(|s| (*s).into()).collect(),
            source: Source::Huggingface,
            url: "https://example.com/a".into(),
            page: "https://example.com/a".into(),
            file_name: if kind == ModelKind::Tokenizer {
                "t.json".into()
            } else if kind == ModelKind::Upscale {
                format!("{id}.pth")
            } else {
                format!("{id}.safetensors")
            },
            size,
            sha256: "a".repeat(64),
            license: "MIT".into(),
            license_url: "https://example.com/l".into(),
            requires_key: false,
            trigger_words: vec![],
            defaults: Defaults::default(),
        }
    }

    fn hw(ram_gb: u64, disk_gb: u64) -> HardwareProfile {
        HardwareProfile {
            ram_bytes: Some(ram_gb << 30),
            disk_free_bytes: Some(disk_gb << 30),
            apple_silicon: true,
            chip: Some("Test".into()),
        }
    }

    #[test]
    fn low_ram_prefers_turbo_over_quality() {
        let models = vec![
            entry("quality", ModelKind::Checkpoint, Family::Sd15, 2 << 30, &["recommended"]),
            entry("turbo", ModelKind::Checkpoint, Family::SdTurbo, 5 << 30, &[]),
        ];
        let h = hw(8, 100);
        let none = |_: &CatalogEntry| false;
        let order = rank_indices(&models, &h, &none);
        assert_eq!(models[order[0]].id, "turbo");
    }

    #[test]
    fn high_ram_prefers_recommended_quality() {
        let models = vec![
            entry("turbo", ModelKind::Checkpoint, Family::SdTurbo, 5 << 30, &[]),
            entry("quality", ModelKind::Checkpoint, Family::Sd15, 2 << 30, &["recommended"]),
        ];
        let h = hw(16, 100);
        let none = |_: &CatalogEntry| false;
        let order = rank_indices(&models, &h, &none);
        assert_eq!(models[order[0]].id, "quality");
    }

    #[test]
    fn tiny_disk_demotes_large_files() {
        let models = [entry("huge", ModelKind::Checkpoint, Family::Sd15, 20 << 30, &["recommended"])];
        let h = hw(16, 2);
        assert!(!fits_disk(&models[0], &h));
        assert!(score(&models[0], &h, false) < -500);
    }

    #[test]
    fn starter_pack_sums_checkpoint_and_tokenizer() {
        let models = vec![
            entry("ckpt", ModelKind::Checkpoint, Family::Sd15, 2 << 30, &["recommended"]),
            entry("tok", ModelKind::Tokenizer, Family::Any, 2 << 20, &["required"]),
            entry("vae", ModelKind::Vae, Family::Sd15, 300 << 20, &["quality"]),
        ];
        let h = hw(16, 50);
        let none = |_: &CatalogEntry| false;
        let pack = starter_pack(&models, &h, &none).unwrap();
        assert_eq!(pack.ids, vec!["ckpt".to_string(), "tok".to_string(), "vae".to_string()]);
        assert_eq!(pack.total_bytes, models[0].size + models[1].size + models[2].size);
    }

    #[test]
    fn upscale_x2_wins_on_8gb() {
        let models = vec![
            entry("realesrgan-x4plus", ModelKind::Upscale, Family::Upscale, 67_000_000, &["x4", "upscale"]),
            entry("realesrgan-x2plus", ModelKind::Upscale, Family::Upscale, 67_000_000, &["x2", "upscale"]),
        ];
        let h = hw(8, 50);
        let ids = best_upscale_ids(&models, &h, 1);
        assert_eq!(ids, vec!["realesrgan-x2plus".to_string()]);
    }
}

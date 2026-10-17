//! The models folder, laid out like ComfyUI's `models/` so both engines share one copy of every
//! file: `checkpoints/`, `loras/`, `vae/`, `embeddings/`, plus `tokenizers/` for the native
//! engine's CLIP tokenizer.
//!
//! Only `.safetensors` model files are accepted: pickle files (`.ckpt`, `.pt`, `.bin`) can run
//! arbitrary code when loaded.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::catalog::{Catalog, Family, ModelKind};
use crate::safetensors_header::Header;
use crate::{MlError, Result};

const LICENSES_FILE: &str = "accepted-licenses.json";
const MAX_TOKENIZER: u64 = 32 << 20;

/// A bare file name: no directories, no hidden files, no `..`, nothing odd.
pub fn is_plain_file_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 255
        && !name.starts_with('.')
        && !name.contains(['/', '\\', ':'])
        && !name.chars().any(char::is_control)
        && name.trim() == name
}

/// What a model file turned out to be.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Detected {
    pub kind: ModelKind,
    pub family: Family,
}

/// One file in the models folder.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InstalledModel {
    pub kind: ModelKind,
    pub file_name: String,
    pub path: PathBuf,
    pub size: u64,
    /// `None` when the file isn't a model the engines can run (the reason is in `problem`).
    pub family: Option<Family>,
    pub problem: Option<String>,
    /// The catalog entry this file was downloaded from, if any.
    pub catalog_id: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ModelStore {
    root: PathBuf,
}

impl Default for ModelStore {
    fn default() -> Self {
        ModelStore::new(crate::system::default_models_dir())
    }
}

impl ModelStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        ModelStore { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn folder(&self, kind: ModelKind) -> PathBuf {
        self.root.join(kind.folder())
    }

    /// Create the folders if they're missing.
    pub fn ensure_layout(&self) -> Result<()> {
        for k in ModelKind::ALL {
            let dir = self.folder(k);
            std::fs::create_dir_all(&dir).map_err(|e| MlError::io(dir.display(), e))?;
        }
        Ok(())
    }

    /// Path of `file_name` in the `kind` folder, refusing anything that would escape it.
    pub fn path_for(&self, kind: ModelKind, file_name: &str) -> Result<PathBuf> {
        if !is_plain_file_name(file_name) {
            return Err(MlError::BadParams(format!("`{file_name}` is not a plain file name")));
        }
        Ok(self.folder(kind).join(file_name))
    }

    /// Path of an installed file, or a "not found" error naming it.
    pub fn existing(&self, kind: ModelKind, file_name: &str) -> Result<PathBuf> {
        let p = self.path_for(kind, file_name)?;
        if p.is_file() { Ok(p) } else { Err(MlError::NotFound(format!("{} `{file_name}` is not installed", kind.as_str()))) }
    }

    pub fn is_installed(&self, kind: ModelKind, file_name: &str) -> bool {
        self.existing(kind, file_name).is_ok()
    }

    /// Every model file, sorted by kind then name. Unreadable or unsupported files are listed
    /// with a `problem` so the user can see and delete them.
    pub fn scan(&self, catalog: Option<&Catalog>) -> Vec<InstalledModel> {
        let mut out = Vec::new();
        for kind in ModelKind::ALL {
            let Ok(rd) = std::fs::read_dir(self.folder(kind)) else { continue };
            let mut names: Vec<(String, PathBuf, u64)> = rd
                .flatten()
                .filter_map(|e| {
                    let name = e.file_name().to_str()?.to_string();
                    let meta = e.metadata().ok()?;
                    (meta.is_file() && is_plain_file_name(&name) && has_model_extension(kind, &name)).then(|| (name, e.path(), meta.len()))
                })
                .collect();
            names.sort();
            for (file_name, path, size) in names {
                let (family, problem) = match detect(&path, Some(kind)) {
                    Ok(d) => (Some(d.family), None),
                    Err(e) => (None, Some(e.to_string())),
                };
                let catalog_id = catalog.and_then(|c| c.by_file(kind, &file_name)).map(|e| e.id.clone());
                out.push(InstalledModel { kind, file_name, path, size, family, problem, catalog_id });
            }
        }
        out
    }

    /// Copy a model file into the right folder after checking it's a safetensors model the
    /// engines can run. `kind` is detected from the file when `None`.
    pub fn import(&self, src: &Path, kind: Option<ModelKind>) -> Result<InstalledModel> {
        let name = src.file_name().and_then(|n| n.to_str()).ok_or_else(|| MlError::BadParams(format!("{}: not a file name", src.display())))?;
        let lower = name.to_ascii_lowercase();
        // Upscale weights for ComfyUI are typically `.pth`; everything else must be safetensors.
        let upscale_pth = kind == Some(ModelKind::Upscale) && lower.ends_with(".pth");
        if !upscale_pth && [".ckpt", ".pt", ".pth", ".bin", ".pkl"].iter().any(|e| lower.ends_with(e)) {
            return Err(MlError::Unsupported(format!(
                "{name}: pickle model files can run code when loaded, so Photoka only imports .safetensors (or Real-ESRGAN .pth via Model Manager); download the .safetensors version of this model"
            )));
        }
        if !src.is_file() {
            return Err(MlError::NotFound(src.display().to_string()));
        }
        let detected = detect(src, kind)?;
        if !is_plain_file_name(name) {
            return Err(MlError::BadParams(format!("`{name}` is not a plain file name")));
        }
        self.ensure_layout()?;
        let dest = self.path_for(detected.kind, name)?;
        if dest.exists() {
            return Err(MlError::BadParams(format!("a {} named `{name}` is already installed", detected.kind.as_str())));
        }
        let part = dest.with_file_name(format!("{name}.part"));
        std::fs::copy(src, &part).map_err(|e| {
            let _ = std::fs::remove_file(&part);
            MlError::io(format!("copying {name}"), e)
        })?;
        std::fs::rename(&part, &dest).map_err(|e| MlError::io(dest.display(), e))?;
        let size = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
        Ok(InstalledModel {
            kind: detected.kind,
            file_name: name.to_string(),
            path: dest,
            size,
            family: Some(detected.family),
            problem: None,
            catalog_id: None,
        })
    }

    /// Remove an installed file (and any partial download of it).
    pub fn delete(&self, kind: ModelKind, file_name: &str) -> Result<()> {
        let p = self.existing(kind, file_name)?;
        std::fs::remove_file(&p).map_err(|e| MlError::io(p.display(), e))?;
        let _ = std::fs::remove_file(p.with_file_name(format!("{file_name}.part")));
        Ok(())
    }

    fn licenses_path(&self) -> PathBuf {
        self.root.join(LICENSES_FILE)
    }

    /// Catalog ids whose license the user has accepted.
    pub fn accepted_licenses(&self) -> BTreeSet<String> {
        std::fs::read(self.licenses_path()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
    }

    pub fn license_accepted(&self, id: &str) -> bool {
        self.accepted_licenses().contains(id)
    }

    pub fn accept_license(&self, id: &str) -> Result<()> {
        let mut set = self.accepted_licenses();
        if !set.insert(id.to_string()) {
            return Ok(());
        }
        std::fs::create_dir_all(&self.root).map_err(|e| MlError::io(self.root.display(), e))?;
        let json = serde_json::to_vec_pretty(&set).map_err(|e| MlError::Other(e.to_string()))?;
        let p = self.licenses_path();
        let tmp = p.with_extension("json.tmp");
        std::fs::write(&tmp, json).map_err(|e| MlError::io(tmp.display(), e))?;
        std::fs::rename(&tmp, &p).map_err(|e| MlError::io(p.display(), e))
    }
}

fn has_model_extension(kind: ModelKind, name: &str) -> bool {
    match kind {
        ModelKind::Tokenizer => name.ends_with(".json"),
        ModelKind::Upscale => name.ends_with(".pth") || name.ends_with(".safetensors"),
        _ => name.ends_with(".safetensors"),
    }
}

/// Identify a model file from its header. `expected` (when given) must match what's inside.
pub fn detect(path: &Path, expected: Option<ModelKind>) -> Result<Detected> {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if expected == Some(ModelKind::Tokenizer) || (expected.is_none() && name.ends_with(".json")) {
        return detect_tokenizer(path, &name);
    }
    if (expected == Some(ModelKind::Upscale)
        || (expected.is_none() && (name.ends_with(".pth") || name.to_ascii_lowercase().contains("realesrgan"))))
        && (name.ends_with(".pth") || name.ends_with(".safetensors"))
    {
        let len = std::fs::metadata(path).map_err(|e| MlError::io(path.display(), e))?.len();
        if len < 1_000_000 {
            return Err(MlError::BadModel(format!("{name}: too small for an upscale model")));
        }
        return Ok(Detected { kind: ModelKind::Upscale, family: Family::Upscale });
    }
    if !name.ends_with(".safetensors") {
        return Err(MlError::Unsupported(format!("{name}: only .safetensors model files are supported (upscale models may be .pth)")));
    }
    let h = Header::read(path)?;
    let found = detect_header(&h).map_err(|msg| MlError::Unsupported(format!("{name}: {msg}")))?;
    if let Some(k) = expected
        && k != found.kind
    {
        return Err(MlError::BadModel(format!("{name}: this is a {}, not a {}", found.kind.as_str(), k.as_str())));
    }
    Ok(found)
}

fn detect_tokenizer(path: &Path, name: &str) -> Result<Detected> {
    let len = std::fs::metadata(path).map_err(|e| MlError::io(path.display(), e))?.len();
    if len > MAX_TOKENIZER {
        return Err(MlError::BadModel(format!("{name}: too large for a tokenizer")));
    }
    let bytes = std::fs::read(path).map_err(|e| MlError::io(path.display(), e))?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| MlError::BadModel(format!("{name}: not a tokenizer.json ({e})")))?;
    if v.get("model").and_then(|m| m.get("vocab")).is_none() {
        return Err(MlError::BadModel(format!("{name}: not a tokenizer.json (no model vocabulary)")));
    }
    Ok(Detected { kind: ModelKind::Tokenizer, family: Family::Any })
}

/// Work out kind and family from tensor names and shapes.
pub fn detect_header(h: &Header) -> std::result::Result<Detected, String> {
    const UNET_IN: &str = "model.diffusion_model.input_blocks.0.0.weight";
    if h.has(UNET_IN) || h.any_key(|k| k.starts_with("model.diffusion_model.")) {
        // SDXL has two text encoders (`conditioner.embedders.0` and `.1`) and a label embedding;
        // SD-Turbo's SGM-style files use `conditioner.embedders.0.model` for its one OpenCLIP.
        if h.has("model.diffusion_model.label_emb.0.0.weight") || h.any_key(|k| k.starts_with("conditioner.embedders.1.")) {
            return Err("SDXL checkpoints are not supported yet (SD 1.5 and SD-Turbo are)".into());
        }
        if h.any_key(|k| k.starts_with("model.diffusion_model.joint_blocks.") || k.starts_with("model.diffusion_model.double_blocks.")) {
            return Err("SD3 and Flux checkpoints are not supported (SD 1.5 and SD-Turbo are)".into());
        }
        let in_ch = h.shape(UNET_IN).and_then(|s| s.get(1).copied()).ok_or("not a Stable Diffusion checkpoint (no UNet input layer)")?;
        let sd2 = h.any_key(|k| k.starts_with("cond_stage_model.model.") || k.starts_with("conditioner.embedders.0.model."));
        let sd1 = h.any_key(|k| k.starts_with("cond_stage_model.transformer."));
        if !h.any_key(|k| k.starts_with("first_stage_model.")) {
            return Err("checkpoint has no VAE; only full single-file checkpoints are supported".into());
        }
        return match (in_ch, sd1, sd2) {
            (4, true, false) => Ok(Detected { kind: ModelKind::Checkpoint, family: Family::Sd15 }),
            (9, true, false) => Ok(Detected { kind: ModelKind::Checkpoint, family: Family::Sd15Inpaint }),
            (4, false, true) => Ok(Detected { kind: ModelKind::Checkpoint, family: Family::SdTurbo }),
            (9, false, true) => Err("SD 2 inpainting checkpoints are not supported".into()),
            (_, false, false) => Err("checkpoint has no text encoder; only full single-file checkpoints are supported".into()),
            _ => Err(format!("unrecognised checkpoint layout (UNet takes {in_ch} channels)")),
        };
    }
    let lora_like = |k: &str| {
        k.contains("lora_down") || k.contains("lora_up") || k.contains("lora_A") || k.contains("lora_B") || k.contains(".lora.down") || k.starts_with("lora_")
    };
    if h.any_key(lora_like) {
        if h.any_key(|k| k.starts_with("lora_te1_") || k.starts_with("lora_te2_") || k.contains("text_encoder_2")) {
            return Err("SDXL LoRAs are not supported yet".into());
        }
        let cross = h.tensors.iter().find_map(|(k, t)| {
            let is_k = k.contains("attn2_to_k") || k.contains("attn2.to_k");
            let is_down = k.contains("lora_down") || k.contains("lora_A") || k.contains("lora.down");
            (is_k && is_down).then(|| t.shape.get(1).copied()).flatten()
        });
        return match cross {
            Some(768) | None => Ok(Detected { kind: ModelKind::Lora, family: Family::Sd15 }),
            Some(1024) => Ok(Detected { kind: ModelKind::Lora, family: Family::SdTurbo }),
            Some(2048) => Err("SDXL LoRAs are not supported yet".into()),
            Some(n) => Err(format!("unrecognised LoRA (cross-attention width {n})")),
        };
    }
    for (key, fam) in [("emb_params", None), ("clip_l", Some("SDXL")), ("clip_g", Some("SDXL"))] {
        if let Some(t) = h.tensors.get(key) {
            if let Some(f) = fam {
                return Err(format!("{f} embeddings are not supported yet"));
            }
            return match t.shape.last() {
                Some(768) => Ok(Detected { kind: ModelKind::Embedding, family: Family::Sd15 }),
                Some(1024) => Ok(Detected { kind: ModelKind::Embedding, family: Family::SdTurbo }),
                _ => Err("unrecognised textual-inversion embedding".into()),
            };
        }
    }
    for prefix in ["", "first_stage_model.", "vae."] {
        if let Some(s) = h.shape(&format!("{prefix}decoder.conv_in.weight")) {
            return match s.get(1) {
                Some(4) => Ok(Detected { kind: ModelKind::Vae, family: Family::Sd15 }),
                Some(n) => Err(format!("VAEs with {n} latent channels (SD3/Flux) are not supported")),
                None => Err("unrecognised VAE".into()),
            };
        }
    }
    Err("not a recognised Stable Diffusion checkpoint, LoRA, VAE or embedding".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors_header::write_test_file;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("photoka-ml-store-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    const SD1: &[(&str, &[u64])] = &[
        ("model.diffusion_model.input_blocks.0.0.weight", &[320, 4, 3, 3]),
        ("cond_stage_model.transformer.text_model.final_layer_norm.weight", &[768]),
        ("first_stage_model.decoder.conv_in.weight", &[512, 4, 3, 3]),
    ];

    #[test]
    fn plain_file_names() {
        assert!(is_plain_file_name("model v2.safetensors"));
        for bad in ["", "../x", "a/b", ".hidden", "a\\b", "x\n", " lead", "c:d"] {
            assert!(!is_plain_file_name(bad), "{bad:?}");
        }
        assert!(!is_plain_file_name(&"a".repeat(300)));
    }

    #[test]
    fn detects_families_and_rejects_unsupported() {
        let d = tmp("detect");
        let check = |name: &str, t: &[(&str, &[u64])]| {
            let p = d.join(name);
            write_test_file(&p, t);
            detect(&p, None)
        };
        assert_eq!(check("a.safetensors", SD1).unwrap(), Detected { kind: ModelKind::Checkpoint, family: Family::Sd15 });
        let inpaint = [("model.diffusion_model.input_blocks.0.0.weight", &[320u64, 9, 3, 3][..]), SD1[1], SD1[2]];
        assert_eq!(check("b.safetensors", &inpaint).unwrap().family, Family::Sd15Inpaint);
        let turbo = [SD1[0], ("cond_stage_model.model.ln_final.weight", &[1024u64][..]), SD1[2]];
        assert_eq!(check("c.safetensors", &turbo).unwrap().family, Family::SdTurbo);
        let sgm_turbo = [SD1[0], ("conditioner.embedders.0.model.ln_final.weight", &[1024u64][..]), SD1[2]];
        assert_eq!(check("c2.safetensors", &sgm_turbo).unwrap().family, Family::SdTurbo);
        let xl = [
            SD1[0],
            ("conditioner.embedders.0.transformer.text_model.final_layer_norm.weight", &[768u64][..]),
            ("conditioner.embedders.1.model.ln_final.weight", &[1280u64][..]),
            SD1[2],
        ];
        assert!(check("d.safetensors", &xl).unwrap_err().to_string().contains("SDXL"));
        let xl2 = [SD1[0], ("model.diffusion_model.label_emb.0.0.weight", &[1280u64, 2816][..]), SD1[1], SD1[2]];
        assert!(check("d2.safetensors", &xl2).unwrap_err().to_string().contains("SDXL"));
        let lora = [("lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight", &[8u64, 768][..])];
        assert_eq!(check("e.safetensors", &lora).unwrap(), Detected { kind: ModelKind::Lora, family: Family::Sd15 });
        let xl_lora = [("lora_unet_x_attn2_to_k.lora_down.weight", &[8u64, 2048][..])];
        assert!(check("f.safetensors", &xl_lora).is_err());
        assert_eq!(check("g.safetensors", &[("decoder.conv_in.weight", &[512, 4, 3, 3])]).unwrap().kind, ModelKind::Vae);
        assert!(check("h.safetensors", &[("decoder.conv_in.weight", &[512, 16, 3, 3])]).is_err());
        assert_eq!(check("i.safetensors", &[("emb_params", &[2, 768])]).unwrap().kind, ModelKind::Embedding);
        assert!(check("j.safetensors", &[("whatever", &[1])]).is_err());
        let p = d.join("a.safetensors");
        assert!(detect(&p, Some(ModelKind::Lora)).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn import_scan_delete_and_licenses() {
        let d = tmp("import");
        let store = ModelStore::new(d.join("models"));
        let src = d.join("My Model.safetensors");
        write_test_file(&src, SD1);
        let m = store.import(&src, None).unwrap();
        assert_eq!((m.kind, m.family), (ModelKind::Checkpoint, Some(Family::Sd15)));
        assert!(store.import(&src, None).is_err(), "duplicate import");
        let ckpt = d.join("old.ckpt");
        std::fs::write(&ckpt, b"pickle").unwrap();
        assert!(store.import(&ckpt, None).unwrap_err().to_string().contains("safetensors"));
        assert!(store.import(&d.join("missing.safetensors"), None).is_err());
        std::fs::write(store.folder(ModelKind::Checkpoint).join("junk.safetensors"), b"nope").unwrap();
        std::fs::write(store.folder(ModelKind::Checkpoint).join("x.safetensors.part"), b"partial").unwrap();
        let list = store.scan(None);
        assert_eq!(list.len(), 2);
        assert!(list.iter().any(|m| m.file_name == "junk.safetensors" && m.problem.is_some()));
        assert!(store.delete(ModelKind::Checkpoint, "../../etc").is_err());
        assert!(store.delete(ModelKind::Lora, "My Model.safetensors").is_err());
        store.delete(ModelKind::Checkpoint, "My Model.safetensors").unwrap();
        assert!(!store.is_installed(ModelKind::Checkpoint, "My Model.safetensors"));
        assert!(!store.license_accepted("sd15"));
        store.accept_license("sd15").unwrap();
        store.accept_license("sd15").unwrap();
        assert!(store.license_accepted("sd15"));
        std::fs::write(store.root().join(LICENSES_FILE), b"{corrupt").unwrap();
        assert!(!store.license_accepted("sd15"));
        let _ = std::fs::remove_dir_all(&d);
    }
}

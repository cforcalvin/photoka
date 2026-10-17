//! The bundled catalog of suggested models (`crates/ml/catalog.toml`).
//!
//! Versioned with the app, so it works offline. Each entry pins the exact file (size and sha256)
//! and names its license, which the user accepts once before downloading.

use serde::{Deserialize, Serialize};

use crate::{MlError, Result};

const BUNDLED: &str = include_str!("../catalog.toml");

/// What a model file is, which also decides its folder in the models directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    Checkpoint,
    Lora,
    Vae,
    Embedding,
    Tokenizer,
    /// Real-ESRGAN and similar ComfyUI `upscale_models/` weights (often `.pth`).
    Upscale,
}

impl ModelKind {
    pub const ALL: [ModelKind; 6] =
        [ModelKind::Checkpoint, ModelKind::Lora, ModelKind::Vae, ModelKind::Embedding, ModelKind::Tokenizer, ModelKind::Upscale];

    /// Subfolder of the models directory (ComfyUI's names, plus `tokenizers`).
    pub fn folder(self) -> &'static str {
        match self {
            ModelKind::Checkpoint => "checkpoints",
            ModelKind::Lora => "loras",
            ModelKind::Vae => "vae",
            ModelKind::Embedding => "embeddings",
            ModelKind::Tokenizer => "tokenizers",
            ModelKind::Upscale => "upscale_models",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ModelKind::Checkpoint => "checkpoint",
            ModelKind::Lora => "lora",
            ModelKind::Vae => "vae",
            ModelKind::Embedding => "embedding",
            ModelKind::Tokenizer => "tokenizer",
            ModelKind::Upscale => "upscale",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        ModelKind::ALL.into_iter().find(|k| k.as_str() == s || k.folder() == s)
    }
}

/// Model architecture family. The native engine runs SD 1.5 (and its inpainting variant) and
/// SD-Turbo; anything else is reported, not guessed at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Family {
    #[serde(rename = "sd15")]
    Sd15,
    #[serde(rename = "sd15-inpaint")]
    Sd15Inpaint,
    #[serde(rename = "sd-turbo")]
    SdTurbo,
    /// Real-ESRGAN and other dedicated upscalers.
    #[serde(rename = "upscale")]
    Upscale,
    /// Support files usable with every family (tokenizer).
    #[serde(rename = "any")]
    Any,
}

impl Family {
    pub fn as_str(self) -> &'static str {
        match self {
            Family::Sd15 => "sd15",
            Family::Sd15Inpaint => "sd15-inpaint",
            Family::SdTurbo => "sd-turbo",
            Family::Upscale => "upscale",
            Family::Any => "any",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Family::Sd15 => "SD 1.5",
            Family::Sd15Inpaint => "SD 1.5 Inpainting",
            Family::SdTurbo => "SD-Turbo",
            Family::Upscale => "Upscale",
            Family::Any => "Any",
        }
    }

    /// Whether a LoRA or VAE of family `self` works with a checkpoint of family `ckpt`.
    pub fn compatible_with(self, ckpt: Family) -> bool {
        match self {
            Family::Any => true,
            Family::Upscale => ckpt == Family::Upscale,
            Family::Sd15 | Family::Sd15Inpaint => matches!(ckpt, Family::Sd15 | Family::Sd15Inpaint),
            Family::SdTurbo => ckpt == Family::SdTurbo,
        }
    }
}

/// Where an entry is downloaded from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Huggingface,
    Civitai,
}

/// Recommended generation settings for a model.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Defaults {
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub cfg: Option<f32>,
    #[serde(default)]
    pub sampler: Option<String>,
    #[serde(default)]
    pub lora_weight: Option<f32>,
}

/// One suggested model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub id: String,
    pub name: String,
    pub description: String,
    pub kind: ModelKind,
    pub family: Family,
    #[serde(default)]
    pub tags: Vec<String>,
    pub source: Source,
    pub url: String,
    pub page: String,
    /// File name in the models folder.
    pub file_name: String,
    /// Exact size in bytes.
    pub size: u64,
    /// Lowercase hex sha256 of the file.
    pub sha256: String,
    pub license: String,
    pub license_url: String,
    /// Civitai: the user's API key is needed to download.
    #[serde(default)]
    pub requires_key: bool,
    #[serde(default)]
    pub trigger_words: Vec<String>,
    #[serde(default)]
    pub defaults: Defaults,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Catalog {
    pub version: u32,
    #[serde(rename = "model")]
    pub models: Vec<CatalogEntry>,
}

impl Catalog {
    /// The catalog compiled into the app.
    pub fn bundled() -> Result<Catalog> {
        Self::parse(BUNDLED)
    }

    /// Parse and validate a catalog file.
    pub fn parse(text: &str) -> Result<Catalog> {
        let c: Catalog = toml::from_str(text).map_err(|e| MlError::BadModel(format!("model catalog: {e}")))?;
        c.validate()?;
        Ok(c)
    }

    pub fn get(&self, id: &str) -> Option<&CatalogEntry> {
        self.models.iter().find(|m| m.id == id)
    }

    /// The entry whose file is `file_name` in a `kind` folder.
    pub fn by_file(&self, kind: ModelKind, file_name: &str) -> Option<&CatalogEntry> {
        self.models.iter().find(|m| m.kind == kind && m.file_name == file_name)
    }

    /// Every entry is complete and safe to act on: unique ids, https URLs, a plain file name,
    /// a size, a 64-digit lowercase hex sha256 and a license.
    pub fn validate(&self) -> Result<()> {
        let mut seen = std::collections::HashSet::new();
        for m in &self.models {
            let fail = |msg: &str| Err(MlError::BadModel(format!("model catalog entry `{}`: {msg}", m.id)));
            if m.id.is_empty() || !m.id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
                return fail("id must be non-empty ASCII letters, digits and dashes");
            }
            if !seen.insert(m.id.as_str()) {
                return fail("duplicate id");
            }
            if !m.url.starts_with("https://") || !m.page.starts_with("https://") || !m.license_url.starts_with("https://") {
                return fail("URLs must be https");
            }
            if !crate::store::is_plain_file_name(&m.file_name) {
                return fail("file_name must be a plain file name");
            }
            let ext_ok = match m.kind {
                ModelKind::Tokenizer => m.file_name.ends_with(".json"),
                // ComfyUI upscale weights are typically PyTorch `.pth` (loaded only in ComfyUI).
                ModelKind::Upscale => m.file_name.ends_with(".pth") || m.file_name.ends_with(".safetensors"),
                _ => m.file_name.ends_with(".safetensors"),
            };
            if !ext_ok {
                return fail("file_name has the wrong extension for its kind");
            }
            if m.size == 0 {
                return fail("size must be set");
            }
            if m.sha256.len() != 64 || !m.sha256.chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)) {
                return fail("sha256 must be 64 lowercase hex digits");
            }
            if m.license.trim().is_empty() || m.name.trim().is_empty() {
                return fail("name and license are required");
            }
            if m.source == Source::Civitai && !m.requires_key {
                return fail("Civitai downloads need an API key (requires_key = true)");
            }
            if m.family == Family::Any && m.kind != ModelKind::Tokenizer {
                return fail("only support files may have family `any`");
            }
            if m.kind == ModelKind::Upscale && m.family != Family::Upscale {
                return fail("upscale models must have family `upscale`");
            }
            if m.family == Family::Upscale && m.kind != ModelKind::Upscale {
                return fail("family `upscale` is only for upscale models");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_catalog_is_valid_and_has_the_suggestions() {
        let c = Catalog::bundled().unwrap();
        assert_eq!(c.version, 1);
        assert_eq!(c.models.first().map(|m| m.id.as_str()), Some("perfectdeliberate-v5"), "the suggested default comes first");
        for id in [
            "perfectdeliberate-v5",
            "sd15",
            "cyberrealistic-revamp",
            "dreamshaper8",
            "dreamshaper8-inpainting",
            "comic-book-lora",
            "pixel-art-lora",
            "lcm-lora-sd15",
            "sd-turbo",
            "realesrgan-x2plus",
            "realesrgan-x4plus",
            "clip-tokenizer",
        ] {
            assert!(c.get(id).is_some(), "catalog lacks {id}");
        }
        assert!(c.get("realesrgan-x2plus").is_some_and(|m| m.kind == ModelKind::Upscale));
        assert!(c.models.iter().any(|m| m.tags.iter().any(|t| t == "pixel-art")));
        assert!(c.models.iter().any(|m| m.tags.iter().any(|t| t == "comic")));
        assert!(c.models.iter().any(|m| m.family == Family::Sd15Inpaint));
        assert_eq!(c.by_file(ModelKind::Tokenizer, "clip-vit-large-patch14.json").map(|m| m.id.as_str()), Some("clip-tokenizer"));
    }

    #[test]
    fn validation_rejects_incomplete_entries() {
        let good = Catalog::bundled().unwrap();
        let entry = good.get("sd15").unwrap().clone();
        let check = |f: &dyn Fn(&mut CatalogEntry)| {
            let mut e = entry.clone();
            f(&mut e);
            Catalog { version: 1, models: vec![e] }.validate()
        };
        assert!(check(&|_| {}).is_ok());
        assert!(check(&|e| e.url = "http://example.com/x.safetensors".into()).is_err());
        assert!(check(&|e| e.sha256 = "ABC".into()).is_err());
        assert!(check(&|e| e.sha256 = e.sha256.to_uppercase()).is_err());
        assert!(check(&|e| e.size = 0).is_err());
        assert!(check(&|e| e.file_name = "../evil.safetensors".into()).is_err());
        assert!(check(&|e| e.file_name = "model.ckpt".into()).is_err());
        assert!(check(&|e| e.license = " ".into()).is_err());
        assert!(check(&|e| e.source = Source::Civitai).is_err());
        let dup = Catalog { version: 1, models: vec![entry.clone(), entry.clone()] };
        assert!(dup.validate().is_err());
        assert!(Catalog::parse("version = 1\n[[model]]\nid = 3").is_err());
    }

    #[test]
    fn family_compatibility() {
        assert!(Family::Sd15.compatible_with(Family::Sd15Inpaint));
        assert!(!Family::Sd15.compatible_with(Family::SdTurbo));
        assert!(Family::Any.compatible_with(Family::SdTurbo));
        assert_eq!(ModelKind::parse("loras"), Some(ModelKind::Lora));
        assert_eq!(ModelKind::parse("vae"), Some(ModelKind::Vae));
        assert_eq!(ModelKind::parse("x"), None);
    }
}

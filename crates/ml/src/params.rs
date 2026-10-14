//! A generation request, shared by both engines and parsed from command params.

use serde::{Deserialize, Serialize};

use crate::image::{Mask, Rgba8};
use crate::{MlError, Result};

/// Which engine runs a generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineKind {
    /// The built-in candle engine (Apple silicon).
    #[default]
    Native,
    /// The ComfyUI that Photoka installs and runs.
    Comfyui,
}

impl EngineKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "native" | "builtin" => Some(EngineKind::Native),
            "comfyui" | "comfy" => Some(EngineKind::Comfyui),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            EngineKind::Native => "native",
            EngineKind::Comfyui => "comfyui",
        }
    }
}

/// Samplers both engines offer (names match ComfyUI's where they exist).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Sampler {
    #[serde(rename = "ddim")]
    Ddim,
    #[serde(rename = "euler")]
    Euler,
    #[serde(rename = "euler_a")]
    EulerA,
    #[serde(rename = "dpmpp_2m")]
    Dpmpp2m,
    #[default]
    #[serde(rename = "dpmpp_2m_karras")]
    Dpmpp2mKarras,
    /// Latent consistency (use with an LCM-LoRA).
    #[serde(rename = "lcm")]
    Lcm,
}

impl Sampler {
    pub const ALL: [Sampler; 6] = [Sampler::Ddim, Sampler::Euler, Sampler::EulerA, Sampler::Dpmpp2m, Sampler::Dpmpp2mKarras, Sampler::Lcm];

    pub fn as_str(self) -> &'static str {
        match self {
            Sampler::Ddim => "ddim",
            Sampler::Euler => "euler",
            Sampler::EulerA => "euler_a",
            Sampler::Dpmpp2m => "dpmpp_2m",
            Sampler::Dpmpp2mKarras => "dpmpp_2m_karras",
            Sampler::Lcm => "lcm",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Sampler::Ddim => "DDIM",
            Sampler::Euler => "Euler",
            Sampler::EulerA => "Euler Ancestral",
            Sampler::Dpmpp2m => "DPM++ 2M",
            Sampler::Dpmpp2mKarras => "DPM++ 2M Karras",
            Sampler::Lcm => "LCM",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Sampler::ALL.into_iter().find(|x| x.as_str() == s)
    }
}

/// A LoRA applied at `weight` (typically 0..=1, negative inverts it).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoraRef {
    /// File name in the `loras` folder.
    pub file: String,
    pub weight: f32,
}

/// Most LoRAs the native engine stacks (ComfyUI takes more).
pub const MAX_NATIVE_LORAS: usize = 4;
pub const MAX_LORAS: usize = 8;
pub const MAX_PROMPT_CHARS: usize = 4000;

/// One generation.
#[derive(Clone, Debug)]
pub struct GenerateRequest {
    /// File name in the `checkpoints` folder.
    pub checkpoint: String,
    /// File name in the `vae` folder, or the checkpoint's own VAE.
    pub vae: Option<String>,
    pub loras: Vec<LoraRef>,
    pub prompt: String,
    pub negative: String,
    pub steps: u32,
    pub cfg: f32,
    pub seed: u64,
    pub sampler: Sampler,
    /// How much of the init image is replaced: 1 = everything (text-to-image when there is no
    /// init image or the mask covers it).
    pub strength: f32,
    /// Output size; multiples of 8.
    pub width: u32,
    pub height: u32,
    /// The frame's current pixels (img2img / inpaint). Same size as the output.
    pub init: Option<Rgba8>,
    /// 255 = regenerate. Same size as the output.
    pub mask: Option<Mask>,
}

impl GenerateRequest {
    /// Reject what no engine can run, with messages that say what to change.
    pub fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(MlError::BadParams(m));
        if self.checkpoint.trim().is_empty() {
            return bad("choose a model (checkpoint) first; Model Manager lists what's installed".into());
        }
        if !(1..=150).contains(&self.steps) {
            return bad(format!("steps must be 1…150 (got {})", self.steps));
        }
        if !self.cfg.is_finite() || !(0.0..=30.0).contains(&self.cfg) {
            return bad(format!("CFG scale must be 0…30 (got {})", self.cfg));
        }
        if !self.strength.is_finite() || !(0.0..=1.0).contains(&self.strength) {
            return bad(format!("strength must be 0…1 (got {})", self.strength));
        }
        for (side, v) in [("width", self.width), ("height", self.height)] {
            if !(64..=crate::image::MAX_SIDE).contains(&v) || v % 8 != 0 {
                return bad(format!("{side} must be a multiple of 8 in 64…{} (got {v})", crate::image::MAX_SIDE));
            }
        }
        if self.prompt.chars().count() > MAX_PROMPT_CHARS || self.negative.chars().count() > MAX_PROMPT_CHARS {
            return bad(format!("prompts are limited to {MAX_PROMPT_CHARS} characters"));
        }
        if self.loras.len() > MAX_LORAS {
            return bad(format!("at most {MAX_LORAS} LoRAs"));
        }
        for l in &self.loras {
            if !l.weight.is_finite() || !(-4.0..=4.0).contains(&l.weight) {
                return bad(format!("LoRA weight for {} must be -4…4", l.file));
            }
        }
        if let Some(i) = &self.init
            && (i.width, i.height) != (self.width, self.height)
        {
            return bad("init image size differs from the output size".into());
        }
        if let Some(m) = &self.mask
            && (m.width, m.height) != (self.width, self.height)
        {
            return bad("mask size differs from the output size".into());
        }
        Ok(())
    }
}

/// What a generation returns.
#[derive(Clone, Debug)]
pub struct GenerateOutput {
    pub image: Rgba8,
    pub seed: u64,
    pub elapsed_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn request() -> GenerateRequest {
        GenerateRequest {
            checkpoint: "a.safetensors".into(),
            vae: None,
            loras: vec![],
            prompt: "a cat".into(),
            negative: String::new(),
            steps: 20,
            cfg: 7.0,
            seed: 1,
            sampler: Sampler::EulerA,
            strength: 1.0,
            width: 512,
            height: 512,
            init: None,
            mask: None,
        }
    }

    #[test]
    fn validate_rejects_bad_values() {
        assert!(request().validate().is_ok());
        type Edit = Box<dyn Fn(&mut GenerateRequest)>;
        let cases: Vec<Edit> = vec![
            Box::new(|r| r.checkpoint.clear()),
            Box::new(|r| r.steps = 0),
            Box::new(|r| r.steps = 151),
            Box::new(|r| r.cfg = f32::NAN),
            Box::new(|r| r.cfg = 31.0),
            Box::new(|r| r.strength = 1.5),
            Box::new(|r| r.width = 500),
            Box::new(|r| r.height = 32),
            Box::new(|r| r.width = 4096),
            Box::new(|r| r.prompt = "x".repeat(MAX_PROMPT_CHARS + 1)),
            Box::new(|r| r.loras = vec![LoraRef { file: "l".into(), weight: f32::INFINITY }]),
            Box::new(|r| r.init = Some(Rgba8::new(64, 64).unwrap())),
            Box::new(|r| r.mask = Some(Mask::new(64, 64).unwrap())),
        ];
        for (i, f) in cases.iter().enumerate() {
            let mut r = request();
            f(&mut r);
            assert!(r.validate().is_err(), "case {i} passed");
        }
    }

    #[test]
    fn names_round_trip() {
        for s in Sampler::ALL {
            assert_eq!(Sampler::parse(s.as_str()), Some(s));
        }
        assert_eq!(EngineKind::parse("comfy"), Some(EngineKind::Comfyui));
        assert_eq!(EngineKind::parse("x"), None);
    }
}

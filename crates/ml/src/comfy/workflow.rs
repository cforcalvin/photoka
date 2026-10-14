//! ComfyUI API-format workflows ("prompt" JSON) for one generation request.
//!
//! The graph is built from ComfyUI's core nodes only (no custom nodes):
//! `CheckpointLoaderSimple` → `LoraLoader`… → `CLIPTextEncode` ×2 → latent → `KSampler` →
//! `VAEDecode` → `PreviewImage`. The latent is an `EmptyLatentImage` (text to image), a
//! `VAEEncode` of the uploaded frame (image to image), `VAEEncodeForInpaint` (inpainting
//! checkpoints at full strength; `InpaintModelConditioning` below it) or `VAEEncode` +
//! `SetLatentNoiseMask` (inpainting with a regular checkpoint).
//! The uploaded frame carries the mask in its alpha (transparent = regenerate), which
//! `LoadImage` turns into its MASK output.

use serde_json::{Value, json};

use crate::catalog::Family;
use crate::params::{GenerateRequest, Sampler};

/// The node whose output holds the result image.
pub const OUTPUT_NODE: &str = "22";
/// Pixels `VAEEncodeForInpaint` grows the mask by, for a seamless edge.
const INPAINT_GROW: u32 = 6;

/// ComfyUI's `sampler_name` and `scheduler` for one of our samplers.
pub fn sampler_names(sampler: Sampler, family: Family) -> (&'static str, &'static str) {
    let turbo = family == Family::SdTurbo;
    match sampler {
        Sampler::Ddim => ("ddim", "ddim_uniform"),
        Sampler::Euler => ("euler", if turbo { "sgm_uniform" } else { "normal" }),
        Sampler::EulerA => ("euler_ancestral", if turbo { "sgm_uniform" } else { "normal" }),
        Sampler::Dpmpp2m => ("dpmpp_2m", "normal"),
        Sampler::Dpmpp2mKarras => ("dpmpp_2m", "karras"),
        Sampler::Lcm => ("lcm", "sgm_uniform"),
    }
}

/// The workflow for `req`. `image` is the uploaded frame's name in ComfyUI's input folder, when
/// the request has an init image; `family` is the checkpoint's.
pub fn build(req: &GenerateRequest, family: Family, image: Option<&str>) -> Value {
    let mut nodes = serde_json::Map::new();
    nodes.insert("1".into(), json!({"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": req.checkpoint}}));
    let mut model = json!(["1", 0]);
    let mut clip = json!(["1", 1]);
    let mut vae = json!(["1", 2]);
    for (i, l) in req.loras.iter().filter(|l| l.weight != 0.0).enumerate() {
        let id = (100 + i).to_string();
        nodes.insert(
            id.clone(),
            json!({"class_type": "LoraLoader", "inputs": {"model": model, "clip": clip, "lora_name": l.file, "strength_model": l.weight, "strength_clip": l.weight}}),
        );
        model = json!([id, 0]);
        clip = json!([id, 1]);
    }
    if let Some(v) = req.vae.as_deref().filter(|v| !v.trim().is_empty()) {
        nodes.insert("3".into(), json!({"class_type": "VAELoader", "inputs": {"vae_name": v}}));
        vae = json!(["3", 0]);
    }
    nodes.insert("4".into(), json!({"class_type": "CLIPTextEncode", "inputs": {"text": req.prompt, "clip": clip}}));
    nodes.insert("5".into(), json!({"class_type": "CLIPTextEncode", "inputs": {"text": req.negative, "clip": clip}}));

    let masked = req.mask.is_some();
    let (mut positive, mut negative) = (json!(["4", 0]), json!(["5", 0]));
    let (latent, denoise) = match image {
        Some(name) if masked || req.strength < 1.0 => {
            nodes.insert("7".into(), json!({"class_type": "LoadImage", "inputs": {"image": name}}));
            if masked && family == Family::Sd15Inpaint && req.strength >= 1.0 {
                nodes.insert(
                    "8".into(),
                    json!({"class_type": "VAEEncodeForInpaint", "inputs": {"pixels": ["7", 0], "vae": vae, "mask": ["7", 1], "grow_mask_by": INPAINT_GROW}}),
                );
                (json!(["8", 0]), 1.0)
            } else if masked && family == Family::Sd15Inpaint {
                // `VAEEncodeForInpaint` greys out the masked pixels, so it needs full denoise;
                // this node keeps them, which partial strength needs.
                nodes.insert(
                    "10".into(),
                    json!({"class_type": "InpaintModelConditioning", "inputs": {
                        "positive": positive, "negative": negative, "vae": vae, "pixels": ["7", 0], "mask": ["7", 1], "noise_mask": true,
                    }}),
                );
                positive = json!(["10", 0]);
                negative = json!(["10", 1]);
                (json!(["10", 2]), req.strength)
            } else if masked {
                nodes.insert("8".into(), json!({"class_type": "VAEEncode", "inputs": {"pixels": ["7", 0], "vae": vae}}));
                nodes.insert("9".into(), json!({"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["8", 0], "mask": ["7", 1]}}));
                (json!(["9", 0]), req.strength)
            } else {
                nodes.insert("8".into(), json!({"class_type": "VAEEncode", "inputs": {"pixels": ["7", 0], "vae": vae}}));
                (json!(["8", 0]), req.strength)
            }
        }
        _ => {
            nodes.insert("6".into(), json!({"class_type": "EmptyLatentImage", "inputs": {"width": req.width, "height": req.height, "batch_size": 1}}));
            (json!(["6", 0]), 1.0)
        }
    };
    let (sampler_name, scheduler) = sampler_names(req.sampler, family);
    nodes.insert(
        "20".into(),
        json!({"class_type": "KSampler", "inputs": {
            "seed": req.seed, "steps": req.steps, "cfg": req.cfg, "sampler_name": sampler_name, "scheduler": scheduler,
            "denoise": denoise, "model": model, "positive": positive, "negative": negative, "latent_image": latent,
        }}),
    );
    nodes.insert("21".into(), json!({"class_type": "VAEDecode", "inputs": {"samples": ["20", 0], "vae": vae}}));
    nodes.insert(OUTPUT_NODE.into(), json!({"class_type": "PreviewImage", "inputs": {"images": ["21", 0]}}));
    Value::Object(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Mask, Rgba8};
    use crate::params::LoraRef;

    fn req() -> GenerateRequest {
        GenerateRequest {
            checkpoint: "v1-5.safetensors".into(),
            vae: None,
            loras: vec![],
            prompt: "a lighthouse".into(),
            negative: "blurry".into(),
            steps: 20,
            cfg: 7.0,
            seed: 42,
            sampler: Sampler::Dpmpp2mKarras,
            strength: 1.0,
            width: 512,
            height: 384,
            init: None,
            mask: None,
        }
    }

    #[test]
    fn text_to_image_golden() {
        let w = build(&req(), Family::Sd15, None);
        let expected = json!({
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5.safetensors"}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "a lighthouse", "clip": ["1", 1]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["1", 1]}},
            "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 384, "batch_size": 1}},
            "20": {"class_type": "KSampler", "inputs": {
                "seed": 42, "steps": 20, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
                "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
            "21": {"class_type": "VAEDecode", "inputs": {"samples": ["20", 0], "vae": ["1", 2]}},
            "22": {"class_type": "PreviewImage", "inputs": {"images": ["21", 0]}},
        });
        assert_eq!(w, expected);
    }

    #[test]
    fn loras_chain_and_a_separate_vae() {
        let mut r = req();
        r.vae = Some("vae-ft-mse.safetensors".into());
        r.loras = vec![
            LoraRef { file: "comic.safetensors".into(), weight: 0.8 },
            LoraRef { file: "off.safetensors".into(), weight: 0.0 },
            LoraRef { file: "lcm.safetensors".into(), weight: 1.0 },
        ];
        let w = build(&r, Family::Sd15, None);
        assert_eq!(w["100"]["inputs"]["model"], json!(["1", 0]));
        assert_eq!(w["100"]["inputs"]["lora_name"], "comic.safetensors");
        assert_eq!(w["101"]["inputs"]["model"], json!(["100", 0]));
        assert_eq!(w["101"]["inputs"]["clip"], json!(["100", 1]));
        assert_eq!(w["101"]["inputs"]["lora_name"], "lcm.safetensors");
        assert!(w.get("102").is_none(), "a zero-weight LoRA is skipped");
        assert_eq!(w["20"]["inputs"]["model"], json!(["101", 0]));
        assert_eq!(w["4"]["inputs"]["clip"], json!(["101", 1]));
        assert_eq!(w["3"]["inputs"]["vae_name"], "vae-ft-mse.safetensors");
        assert_eq!(w["21"]["inputs"]["vae"], json!(["3", 0]));
    }

    #[test]
    fn inpainting_picks_the_encoder_by_checkpoint() {
        let mut r = req();
        r.init = Some(Rgba8::new(512, 384).unwrap());
        r.mask = Some(Mask::new(512, 384).unwrap());
        let w = build(&r, Family::Sd15Inpaint, Some("frame.png"));
        assert_eq!(w["7"], json!({"class_type": "LoadImage", "inputs": {"image": "frame.png"}}));
        assert_eq!(w["8"]["class_type"], "VAEEncodeForInpaint");
        assert_eq!(w["8"]["inputs"]["mask"], json!(["7", 1]));
        assert_eq!(w["20"]["inputs"]["latent_image"], json!(["8", 0]));
        assert_eq!(w["20"]["inputs"]["denoise"], 1.0);
        assert_eq!(w["20"]["inputs"]["positive"], json!(["4", 0]));

        // Below full strength an inpainting checkpoint keeps the masked pixels and the strength.
        r.strength = 0.3;
        let w = build(&r, Family::Sd15Inpaint, Some("frame.png"));
        assert!(w.get("8").is_none());
        assert_eq!(
            w["10"],
            json!({"class_type": "InpaintModelConditioning", "inputs": {
                "positive": ["4", 0], "negative": ["5", 0], "vae": ["1", 2], "pixels": ["7", 0], "mask": ["7", 1], "noise_mask": true,
            }})
        );
        assert_eq!(w["20"]["inputs"]["positive"], json!(["10", 0]));
        assert_eq!(w["20"]["inputs"]["negative"], json!(["10", 1]));
        assert_eq!(w["20"]["inputs"]["latent_image"], json!(["10", 2]));
        assert!((w["20"]["inputs"]["denoise"].as_f64().unwrap() - 0.3).abs() < 1e-6);

        r.strength = 0.9;
        let w = build(&r, Family::Sd15, Some("frame.png"));
        assert_eq!(w["8"]["class_type"], "VAEEncode");
        assert_eq!(w["9"], json!({"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["8", 0], "mask": ["7", 1]}}));
        assert_eq!(w["20"]["inputs"]["latent_image"], json!(["9", 0]));
        assert!((w["20"]["inputs"]["denoise"].as_f64().unwrap() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn image_to_image_and_full_strength() {
        let mut r = req();
        r.init = Some(Rgba8::new(512, 384).unwrap());
        r.strength = 0.5;
        let w = build(&r, Family::Sd15, Some("frame.png"));
        assert_eq!(w["20"]["inputs"]["latent_image"], json!(["8", 0]));
        assert!(w.get("9").is_none());
        // Strength 1 without a mask replaces everything: no need to encode the frame.
        r.strength = 1.0;
        let w = build(&r, Family::Sd15, Some("frame.png"));
        assert!(w.get("7").is_none() && w.get("6").is_some());
    }

    #[test]
    fn samplers_map_to_comfy_names() {
        assert_eq!(sampler_names(Sampler::Lcm, Family::Sd15), ("lcm", "sgm_uniform"));
        assert_eq!(sampler_names(Sampler::EulerA, Family::SdTurbo), ("euler_ancestral", "sgm_uniform"));
        assert_eq!(sampler_names(Sampler::EulerA, Family::Sd15), ("euler_ancestral", "normal"));
        assert_eq!(sampler_names(Sampler::Ddim, Family::Sd15), ("ddim", "ddim_uniform"));
    }
}

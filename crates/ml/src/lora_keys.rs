//! LoRA files → the weights they modify.
//!
//! A LoRA stores, per modified layer, a low-rank pair (`down` `[r, in]`, `up` `[out, r]`) and
//! optionally `alpha`; merging adds `weight · alpha/r · up @ down` to the layer's weight. Files name
//! layers in one of three ways, all handled here:
//!
//! - kohya, on diffusers layer names: `lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight`
//! - kohya, on original (LDM) layer names: `lora_unet_input_blocks_1_1_proj_in.lora_down.weight`
//! - diffusers/PEFT: `unet.down_blocks.0.attentions.0.proj_in.lora_A.weight`
//!
//! Text-encoder layers use `lora_te_` / `text_encoder.`. LyCORIS variants (LoHa, LoKr) are reported
//! as unsupported rather than half-applied.

use std::collections::{BTreeMap, HashMap};

use crate::keymap::Component;
use crate::safetensors_header::Header;
use crate::{MlError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Part {
    Down,
    Up,
    Alpha,
}

fn split(key: &str) -> Option<(&str, Part)> {
    const SUFFIXES: [(&str, Part); 7] = [
        (".lora_down.weight", Part::Down),
        (".lora_up.weight", Part::Up),
        (".lora_A.weight", Part::Down),
        (".lora_B.weight", Part::Up),
        (".lora.down.weight", Part::Down),
        (".lora.up.weight", Part::Up),
        (".alpha", Part::Alpha),
    ];
    SUFFIXES.iter().find_map(|(s, p)| key.strip_suffix(s).map(|b| (b, *p)))
}

/// Every name a LoRA may use for each weight of the loaded model.
#[derive(Clone, Debug, Default)]
pub struct LoraIndex {
    map: HashMap<String, (Component, String)>,
}

impl LoraIndex {
    /// Register a model weight by its diffusers name (`….to_q.weight`).
    pub fn add(&mut self, component: Component, name: &str) {
        let Some(base) = name.strip_suffix(".weight") else { return };
        let target = (component, name.to_string());
        match component {
            Component::Unet => {
                self.map.insert(format!("lora_unet_{}", base.replace('.', "_")), target.clone());
                self.map.insert(format!("unet.{base}"), target);
            }
            Component::Clip => {
                if let Some(rest) = base.strip_prefix("text_model.") {
                    self.map.insert(format!("lora_te_text_model_{}", rest.replace('.', "_")), target.clone());
                    self.map.insert(format!("text_encoder.text_model.{rest}"), target);
                }
            }
            Component::Vae => {}
        }
    }

    /// Register the original (LDM) name of a UNet weight too, for kohya files trained on it.
    pub fn add_ldm(&mut self, ldm_key: &str, component: Component, name: &str) {
        if component != Component::Unet {
            return;
        }
        if let Some(base) = ldm_key.strip_prefix("model.diffusion_model.").and_then(|k| k.strip_suffix(".weight")) {
            self.map.insert(format!("lora_unet_{}", base.replace('.', "_")), (component, name.to_string()));
        }
    }

    pub fn target(&self, lora_base: &str) -> Option<&(Component, String)> {
        self.map.get(lora_base)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// One layer a LoRA modifies.
#[derive(Clone, Debug, PartialEq)]
pub struct LoraPair {
    pub component: Component,
    /// The model weight (diffusers name).
    pub target: String,
    pub down: String,
    pub up: String,
    pub alpha: Option<String>,
}

/// What a LoRA file will change.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoraPlan {
    pub pairs: Vec<LoraPair>,
    /// Layers in the file that match nothing in the model.
    pub unmatched: usize,
}

/// Match a LoRA file's layers against the model.
pub fn plan(file_name: &str, h: &Header, index: &LoraIndex) -> Result<LoraPlan> {
    if h.any_key(|k| k.contains(".hada_w1") || k.contains(".lokr_w1")) {
        return Err(MlError::Unsupported(format!("{file_name}: LoHa/LoKr (LyCORIS) LoRAs are not supported by the built-in engine; use the ComfyUI engine")));
    }
    let mut groups: BTreeMap<&str, [Option<&str>; 3]> = BTreeMap::new();
    for k in h.tensors.keys() {
        if let Some((base, part)) = split(k) {
            let slot = groups.entry(base).or_default();
            let i = match part {
                Part::Down => 0,
                Part::Up => 1,
                Part::Alpha => 2,
            };
            if let Some(s) = slot.get_mut(i) {
                *s = Some(k.as_str());
            }
        }
    }
    let mut out = LoraPlan::default();
    for (base, [down, up, alpha]) in groups {
        let (Some(down), Some(up)) = (down, up) else { continue };
        match index.target(base) {
            Some((component, target)) => {
                out.pairs.push(LoraPair { component: *component, target: target.clone(), down: down.into(), up: up.into(), alpha: alpha.map(Into::into) })
            }
            None => out.unmatched += 1,
        }
    }
    if out.pairs.is_empty() {
        return Err(MlError::BadModel(format!("{file_name}: none of its layers match this model (is it a LoRA for another model family?)")));
    }
    Ok(out)
}

/// The scale applied to `up @ down`: `weight · alpha / rank` (alpha defaults to the rank).
pub fn scale(weight: f32, alpha: Option<f32>, rank: usize) -> f32 {
    if rank == 0 {
        return 0.0;
    }
    let alpha = alpha.filter(|a| a.is_finite() && *a > 0.0).unwrap_or(rank as f32);
    weight * alpha / rank as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors_header::TensorInfo;

    fn header(keys: &[&str]) -> Header {
        let mut h = Header::default();
        for k in keys {
            h.tensors.insert((*k).into(), TensorInfo { dtype: "F16".into(), shape: vec![1], offsets: [0, 2] });
        }
        h
    }

    fn index() -> LoraIndex {
        let mut i = LoraIndex::default();
        i.add(Component::Unet, "down_blocks.0.attentions.0.proj_in.weight");
        i.add(Component::Unet, "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.weight");
        i.add_ldm("model.diffusion_model.input_blocks.1.1.proj_in.weight", Component::Unet, "down_blocks.0.attentions.0.proj_in.weight");
        i.add(Component::Clip, "text_model.encoder.layers.0.self_attn.q_proj.weight");
        i.add(Component::Vae, "decoder.conv_in.weight");
        i
    }

    #[test]
    fn all_three_naming_styles_match() {
        let i = index();
        let h = header(&[
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_down.weight",
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_up.weight",
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_out_0.alpha",
            "lora_unet_input_blocks_1_1_proj_in.lora_down.weight",
            "lora_unet_input_blocks_1_1_proj_in.lora_up.weight",
            "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight",
            "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_B.weight",
            "lora_unet_something_else.lora_down.weight",
            "lora_unet_something_else.lora_up.weight",
            "lora_unet_lonely.lora_down.weight",
        ]);
        let p = plan("x", &h, &i).unwrap();
        assert_eq!(p.pairs.len(), 3);
        assert_eq!(p.unmatched, 1);
        let to_out = p.pairs.iter().find(|x| x.target.contains("to_out")).unwrap();
        assert!(to_out.alpha.is_some());
        assert!(p.pairs.iter().any(|x| x.component == Component::Clip && x.target == "text_model.encoder.layers.0.self_attn.q_proj.weight"));
        assert!(p.pairs.iter().any(|x| x.target == "down_blocks.0.attentions.0.proj_in.weight"));
    }

    #[test]
    fn rejects_lycoris_and_mismatched_files() {
        let i = index();
        assert!(plan("x", &header(&["lora_unet_a.hada_w1_a", "lora_unet_a.hada_w1_b"]), &i).is_err());
        assert!(plan("x", &header(&["lora_unet_zzz.lora_down.weight", "lora_unet_zzz.lora_up.weight"]), &i).is_err());
        assert!(plan("x", &header(&[]), &i).is_err());
    }

    #[test]
    fn scale_defaults_alpha_to_rank() {
        assert_eq!(scale(0.8, None, 64), 0.8);
        assert_eq!(scale(1.0, Some(32.0), 64), 0.5);
        assert_eq!(scale(1.0, Some(f32::NAN), 8), 1.0);
        assert_eq!(scale(1.0, Some(4.0), 0), 0.0);
    }
}

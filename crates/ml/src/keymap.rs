//! Weight names: single-file (original LDM) Stable Diffusion checkpoints → the diffusers layout
//! the native engine's UNet, VAE and CLIP modules load.
//!
//! The mapping follows the published architecture descriptions of both layouts (CompVis
//! latent-diffusion and Hugging Face diffusers); it's a pure function on names so it can be tested
//! without loading weights.

/// Which network a weight belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Component {
    Unet,
    Vae,
    Clip,
}

/// How the tensor has to be reshaped on the way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transform {
    Same,
    /// A 1×1 convolution weight `[o, i, 1, 1]` used as a linear layer `[o, i]`.
    ConvToLinear,
    /// Chunk `index` of `of` equal chunks along dimension 0 (OpenCLIP's fused q/k/v).
    Chunk {
        index: usize,
        of: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target {
    pub component: Component,
    pub name: String,
    pub transform: Transform,
}

impl Target {
    fn new(component: Component, name: String) -> Self {
        Target { component, name, transform: Transform::Same }
    }
}

/// The diffusers name(s) for one checkpoint tensor; empty for tensors the engine doesn't use
/// (EMA copies, position-id buffers, CLIP projection heads, unknown extras).
pub fn map(key: &str) -> Vec<Target> {
    if let Some(rest) = key.strip_prefix("model.diffusion_model.") {
        return unet(rest).map(|n| vec![Target::new(Component::Unet, n)]).unwrap_or_default();
    }
    if let Some(rest) = key.strip_prefix("first_stage_model.") {
        return vae(rest).into_iter().collect();
    }
    if let Some(rest) = key.strip_prefix("cond_stage_model.transformer.") {
        if rest.ends_with("position_ids") {
            return Vec::new();
        }
        let rest = rest.strip_prefix("text_model.").unwrap_or(rest);
        return vec![Target::new(Component::Clip, format!("text_model.{rest}"))];
    }
    if let Some(rest) = key.strip_prefix("cond_stage_model.model.").or_else(|| key.strip_prefix("conditioner.embedders.0.model.")) {
        return open_clip(rest);
    }
    Vec::new()
}

/// Split `"12.0.rest"` into `(12, 0, "rest")`.
fn two_indices(s: &str) -> Option<(usize, usize, &str)> {
    let (a, s) = s.split_once('.')?;
    let (b, rest) = s.split_once('.')?;
    Some((a.parse().ok()?, b.parse().ok()?, rest))
}

fn unet_resnet(rest: &str) -> Option<String> {
    const MAP: [(&str, &str); 6] = [
        ("in_layers.0.", "norm1."),
        ("in_layers.2.", "conv1."),
        ("emb_layers.1.", "time_emb_proj."),
        ("out_layers.0.", "norm2."),
        ("out_layers.3.", "conv2."),
        ("skip_connection.", "conv_shortcut."),
    ];
    MAP.iter().find_map(|(from, to)| rest.strip_prefix(from).map(|r| format!("{to}{r}")))
}

fn unet(key: &str) -> Option<String> {
    const DIRECT: [(&str, &str); 5] = [
        ("time_embed.0.", "time_embedding.linear_1."),
        ("time_embed.2.", "time_embedding.linear_2."),
        ("input_blocks.0.0.", "conv_in."),
        ("out.0.", "conv_norm_out."),
        ("out.2.", "conv_out."),
    ];
    if let Some(n) = DIRECT.iter().find_map(|(from, to)| key.strip_prefix(from).map(|r| format!("{to}{r}"))) {
        return Some(n);
    }
    if let Some(s) = key.strip_prefix("input_blocks.") {
        let (i, j, rest) = two_indices(s)?;
        if i == 0 {
            return None;
        }
        let (b, l) = ((i - 1) / 3, (i - 1) % 3);
        return match j {
            0 if rest.starts_with("op.") => Some(format!("down_blocks.{b}.downsamplers.0.conv.{}", rest.strip_prefix("op.")?)),
            0 => Some(format!("down_blocks.{b}.resnets.{l}.{}", unet_resnet(rest)?)),
            1 => Some(format!("down_blocks.{b}.attentions.{l}.{rest}")),
            _ => None,
        };
    }
    if let Some(s) = key.strip_prefix("middle_block.") {
        let (j, rest) = s.split_once('.')?;
        return match j {
            "0" => Some(format!("mid_block.resnets.0.{}", unet_resnet(rest)?)),
            "1" => Some(format!("mid_block.attentions.0.{rest}")),
            "2" => Some(format!("mid_block.resnets.1.{}", unet_resnet(rest)?)),
            _ => None,
        };
    }
    if let Some(s) = key.strip_prefix("output_blocks.") {
        let (i, j, rest) = two_indices(s)?;
        let (b, l) = (i / 3, i % 3);
        if let Some(conv) = rest.strip_prefix("conv.")
            && j > 0
        {
            return Some(format!("up_blocks.{b}.upsamplers.0.conv.{conv}"));
        }
        return match j {
            0 => Some(format!("up_blocks.{b}.resnets.{l}.{}", unet_resnet(rest)?)),
            1 => Some(format!("up_blocks.{b}.attentions.{l}.{rest}")),
            _ => None,
        };
    }
    None
}

fn vae_resnet(rest: &str) -> String {
    match rest.strip_prefix("nin_shortcut.") {
        Some(r) => format!("conv_shortcut.{r}"),
        None => rest.to_string(),
    }
}

fn vae_mid(rest: &str) -> Option<Target> {
    if let Some(r) = rest.strip_prefix("block_1.") {
        return Some(Target::new(Component::Vae, format!("mid_block.resnets.0.{}", vae_resnet(r))));
    }
    if let Some(r) = rest.strip_prefix("block_2.") {
        return Some(Target::new(Component::Vae, format!("mid_block.resnets.1.{}", vae_resnet(r))));
    }
    let r = rest.strip_prefix("attn_1.")?;
    let (part, tail) = r.split_once('.')?;
    let (name, linear) = match part {
        "norm" => ("group_norm", false),
        "q" => ("to_q", true),
        "k" => ("to_k", true),
        "v" => ("to_v", true),
        "proj_out" => ("to_out.0", true),
        _ => return None,
    };
    let transform = if linear && tail == "weight" { Transform::ConvToLinear } else { Transform::Same };
    Some(Target { component: Component::Vae, name: format!("mid_block.attentions.0.{name}.{tail}"), transform })
}

/// Number of VAE resolution levels (SD 1.x/2.x).
const VAE_LEVELS: usize = 4;

fn vae(key: &str) -> Option<Target> {
    if key.starts_with("quant_conv.") || key.starts_with("post_quant_conv.") {
        return Some(Target::new(Component::Vae, key.to_string()));
    }
    let (side, rest) = key.split_once('.')?;
    if side != "encoder" && side != "decoder" {
        return None;
    }
    let mapped = |n: String| Some(Target::new(Component::Vae, format!("{side}.{n}")));
    if rest.starts_with("conv_in.") || rest.starts_with("conv_out.") {
        return mapped(rest.to_string());
    }
    if let Some(r) = rest.strip_prefix("norm_out.") {
        return mapped(format!("conv_norm_out.{r}"));
    }
    if let Some(r) = rest.strip_prefix("mid.") {
        let mut t = vae_mid(r)?;
        t.name = format!("{side}.{}", t.name);
        return Some(t);
    }
    let (level_key, blocks) = if side == "encoder" { ("down.", "down_blocks") } else { ("up.", "up_blocks") };
    let r = rest.strip_prefix(level_key)?;
    let (i, r) = r.split_once('.')?;
    let i: usize = i.parse().ok()?;
    if i >= VAE_LEVELS {
        return None;
    }
    let level = if side == "decoder" { VAE_LEVELS - 1 - i } else { i };
    if let Some(r) = r.strip_prefix("block.") {
        let (j, tail) = r.split_once('.')?;
        let j: usize = j.parse().ok()?;
        return mapped(format!("{blocks}.{level}.resnets.{j}.{}", vae_resnet(tail)));
    }
    let sampler = if side == "encoder" { "downsample.conv." } else { "upsample.conv." };
    let plural = if side == "encoder" { "downsamplers" } else { "upsamplers" };
    let tail = r.strip_prefix(sampler)?;
    mapped(format!("{blocks}.{level}.{plural}.0.conv.{tail}"))
}

fn open_clip(key: &str) -> Vec<Target> {
    let clip = |n: String| vec![Target::new(Component::Clip, format!("text_model.{n}"))];
    if key == "token_embedding.weight" {
        return clip("embeddings.token_embedding.weight".into());
    }
    if key == "positional_embedding" {
        return clip("embeddings.position_embedding.weight".into());
    }
    if let Some(r) = key.strip_prefix("ln_final.") {
        return clip(format!("final_layer_norm.{r}"));
    }
    let Some(r) = key.strip_prefix("transformer.resblocks.") else { return Vec::new() };
    let Some((i, r)) = r.split_once('.') else { return Vec::new() };
    let Ok(i) = i.parse::<usize>() else { return Vec::new() };
    let layer = format!("encoder.layers.{i}");
    let fused = match r {
        "attn.in_proj_weight" => Some("weight"),
        "attn.in_proj_bias" => Some("bias"),
        _ => None,
    };
    if let Some(suffix) = fused {
        return ["q_proj", "k_proj", "v_proj"]
            .iter()
            .enumerate()
            .map(|(index, p)| Target {
                component: Component::Clip,
                name: format!("text_model.{layer}.self_attn.{p}.{suffix}"),
                transform: Transform::Chunk { index, of: 3 },
            })
            .collect();
    }
    const MAP: [(&str, &str); 5] = [
        ("ln_1.", "layer_norm1."),
        ("ln_2.", "layer_norm2."),
        ("attn.out_proj.", "self_attn.out_proj."),
        ("mlp.c_fc.", "mlp.fc1."),
        ("mlp.c_proj.", "mlp.fc2."),
    ];
    MAP.iter().find_map(|(from, to)| r.strip_prefix(from).map(|t| clip(format!("{layer}.{to}{t}")))).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(key: &str) -> String {
        let t = map(key);
        assert_eq!(t.len(), 1, "{key} -> {t:?}");
        t[0].name.clone()
    }

    #[test]
    fn unet_names() {
        let p = "model.diffusion_model.";
        assert_eq!(one(&format!("{p}time_embed.0.weight")), "time_embedding.linear_1.weight");
        assert_eq!(one(&format!("{p}input_blocks.0.0.bias")), "conv_in.bias");
        assert_eq!(one(&format!("{p}input_blocks.1.0.in_layers.0.weight")), "down_blocks.0.resnets.0.norm1.weight");
        assert_eq!(one(&format!("{p}input_blocks.2.1.proj_in.weight")), "down_blocks.0.attentions.1.proj_in.weight");
        assert_eq!(one(&format!("{p}input_blocks.3.0.op.weight")), "down_blocks.0.downsamplers.0.conv.weight");
        assert_eq!(one(&format!("{p}input_blocks.4.0.skip_connection.weight")), "down_blocks.1.resnets.0.conv_shortcut.weight");
        assert_eq!(one(&format!("{p}input_blocks.11.0.out_layers.3.weight")), "down_blocks.3.resnets.1.conv2.weight");
        assert_eq!(one(&format!("{p}middle_block.1.transformer_blocks.0.attn2.to_k.weight")), "mid_block.attentions.0.transformer_blocks.0.attn2.to_k.weight");
        assert_eq!(one(&format!("{p}middle_block.2.emb_layers.1.bias")), "mid_block.resnets.1.time_emb_proj.bias");
        assert_eq!(one(&format!("{p}output_blocks.2.1.conv.weight")), "up_blocks.0.upsamplers.0.conv.weight");
        assert_eq!(one(&format!("{p}output_blocks.5.2.conv.bias")), "up_blocks.1.upsamplers.0.conv.bias");
        assert_eq!(one(&format!("{p}output_blocks.7.1.norm.weight")), "up_blocks.2.attentions.1.norm.weight");
        assert_eq!(one(&format!("{p}out.2.weight")), "conv_out.weight");
        assert!(map(&format!("{p}input_blocks.1.7.x")).is_empty());
        assert!(map("model_ema.decay").is_empty());
    }

    #[test]
    fn vae_names() {
        let p = "first_stage_model.";
        assert_eq!(one(&format!("{p}encoder.down.0.block.1.nin_shortcut.weight")), "encoder.down_blocks.0.resnets.1.conv_shortcut.weight");
        assert_eq!(one(&format!("{p}encoder.down.2.downsample.conv.weight")), "encoder.down_blocks.2.downsamplers.0.conv.weight");
        assert_eq!(one(&format!("{p}decoder.up.3.block.0.conv1.weight")), "decoder.up_blocks.0.resnets.0.conv1.weight");
        assert_eq!(one(&format!("{p}decoder.up.1.upsample.conv.bias")), "decoder.up_blocks.2.upsamplers.0.conv.bias");
        assert_eq!(one(&format!("{p}decoder.norm_out.weight")), "decoder.conv_norm_out.weight");
        assert_eq!(one(&format!("{p}post_quant_conv.weight")), "post_quant_conv.weight");
        let q = map(&format!("{p}encoder.mid.attn_1.q.weight"));
        assert_eq!(q[0].name, "encoder.mid_block.attentions.0.to_q.weight");
        assert_eq!(q[0].transform, Transform::ConvToLinear);
        assert_eq!(map(&format!("{p}encoder.mid.attn_1.q.bias"))[0].transform, Transform::Same);
        assert_eq!(one(&format!("{p}decoder.mid.attn_1.norm.bias")), "decoder.mid_block.attentions.0.group_norm.bias");
        assert_eq!(one(&format!("{p}decoder.mid.block_2.norm1.weight")), "decoder.mid_block.resnets.1.norm1.weight");
        assert!(map(&format!("{p}decoder.up.9.block.0.conv1.weight")).is_empty());
    }

    #[test]
    fn clip_names() {
        assert_eq!(one("cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc1.weight"), "text_model.encoder.layers.3.mlp.fc1.weight");
        assert_eq!(one("cond_stage_model.transformer.encoder.layers.0.self_attn.q_proj.bias"), "text_model.encoder.layers.0.self_attn.q_proj.bias");
        assert!(map("cond_stage_model.transformer.text_model.embeddings.position_ids").is_empty());
        let qkv = map("cond_stage_model.model.transformer.resblocks.5.attn.in_proj_weight");
        assert_eq!(qkv.len(), 3);
        assert_eq!(qkv[2].name, "text_model.encoder.layers.5.self_attn.v_proj.weight");
        assert_eq!(qkv[2].transform, Transform::Chunk { index: 2, of: 3 });
        assert_eq!(one("cond_stage_model.model.positional_embedding"), "text_model.embeddings.position_embedding.weight");
        assert_eq!(one("cond_stage_model.model.transformer.resblocks.0.mlp.c_proj.bias"), "text_model.encoder.layers.0.mlp.fc2.bias");
        assert_eq!(one("cond_stage_model.model.ln_final.weight"), "text_model.final_layer_norm.weight");
        assert!(map("cond_stage_model.model.text_projection").is_empty());
        assert_eq!(one("conditioner.embedders.0.model.ln_final.weight"), "text_model.final_layer_norm.weight");
    }

    /// Set `PHOTOKA_MODELS_DIR` to a models folder holding the SD 1.5 checkpoint to check that
    /// every tensor in a real file maps to a unique name.
    #[test]
    fn real_checkpoint_maps_completely() {
        let Some(dir) = std::env::var_os("PHOTOKA_MODELS_DIR") else { return };
        let p = std::path::Path::new(&dir).join("checkpoints/v1-5-pruned-emaonly-fp16.safetensors");
        let h = crate::safetensors_header::Header::read(&p).unwrap();
        let mut seen = std::collections::HashSet::new();
        let mut unmapped = Vec::new();
        for k in h.tensors.keys() {
            let t = map(k);
            let network = ["model.diffusion_model.", "first_stage_model.", "cond_stage_model."].iter().any(|p| k.starts_with(p));
            if t.is_empty() && network && !k.ends_with("position_ids") {
                unmapped.push(k.clone());
            }
            for t in t {
                assert!(seen.insert((t.component, t.name.clone())), "duplicate {:?}", t.name);
            }
        }
        assert!(unmapped.is_empty(), "unmapped: {unmapped:?}");
        assert_eq!(seen.iter().filter(|(c, _)| *c == Component::Unet).count(), 686);
    }
}

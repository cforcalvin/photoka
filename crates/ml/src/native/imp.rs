//! The candle (Metal) implementation of [`super::NativeEngine`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::{clip, unet_2d, vae};

use super::ResolvedModels;
use crate::catalog::Family;
use crate::image::{Mask, Rgba8};
use crate::keymap::{self, Component, Transform};
use crate::lora_keys::{self, LoraIndex};
use crate::params::GenerateRequest;
use crate::safetensors_header::Header;
use crate::sampling::{self, Schedule, ScheduleKind};
use crate::{MlError, Progress, Result};

const LATENT_SCALE: f64 = 0.18215;
const UNET_DTYPE: DType = DType::F16;
/// CLIP and the VAE run in f32: f16 text encoders and VAEs can overflow.
const CLIP_DTYPE: DType = DType::F32;
const VAE_DTYPE: DType = DType::F32;
/// VAE tile side and overlap, in latent pixels. candle's Metal convolutions build an im2col
/// matrix of about 1.2 GB per full-resolution layer at 512×512, so whole-frame VAE passes run
/// out of GPU memory (silently, as garbage pixels) on 16 GB Macs.
const VAE_TILE: usize = 32;
const VAE_OVERLAP: usize = 8;

fn ce(e: candle_core::Error) -> MlError {
    let msg = e.to_string();
    if msg.to_ascii_lowercase().contains("out of memory") || msg.contains("OutOfMemory") {
        log::warn!("generation: {msg}");
        MlError::OutOfMemory(format!("the GPU ran out of memory; make the frame smaller, close other apps, or use fewer LoRAs ({msg})"))
    } else {
        MlError::Other(format!("generation failed: {msg}"))
    }
}

fn dtype_of(s: &str) -> Option<DType> {
    Some(match s {
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "F32" => DType::F32,
        "F64" => DType::F64,
        _ => return None,
    })
}

/// Read tensor `name` from an open safetensors file onto `device` as `dtype`.
fn read_tensor(h: &Header, f: &mut std::fs::File, name: &str, dtype: DType, device: &Device) -> Result<Tensor> {
    let info = h.tensors.get(name).ok_or_else(|| MlError::BadModel(format!("tensor {name} is missing")))?;
    let src = dtype_of(&info.dtype).ok_or_else(|| MlError::BadModel(format!("tensor {name} has unsupported type {}", info.dtype)))?;
    let shape: Vec<usize> = info.shape.iter().map(|&d| usize::try_from(d).unwrap_or(usize::MAX)).collect();
    let bytes = h.read_bytes(f, name)?;
    let t = Tensor::from_raw_buffer(&bytes, src, &shape, &Device::Cpu).map_err(ce)?;
    t.to_dtype(dtype).and_then(|t| t.to_device(device)).map_err(ce)
}

fn apply_transform(t: &Tensor, tr: Transform) -> Result<Tensor> {
    match tr {
        Transform::Same => Ok(t.clone()),
        Transform::ConvToLinear => {
            let d = t.dims();
            match d {
                [o, i, 1, 1] => t.reshape((*o, *i)).map_err(ce),
                [_, _] => Ok(t.clone()),
                _ => Err(MlError::BadModel(format!("expected a 1×1 convolution weight, got shape {d:?}"))),
            }
        }
        Transform::Chunk { index, of } => {
            let parts = t.chunk(of, 0).map_err(ce)?;
            parts.get(index).cloned().ok_or_else(|| MlError::BadModel("fused q/k/v weight has the wrong size".into()))
        }
    }
}

fn component_dtype(c: Component) -> DType {
    match c {
        Component::Unet => UNET_DTYPE,
        Component::Clip => CLIP_DTYPE,
        Component::Vae => VAE_DTYPE,
    }
}

fn device() -> Device {
    match Device::new_metal(0) {
        Ok(d) => d,
        Err(e) => {
            log::warn!("Metal is unavailable ({e}); generating on the CPU, which is much slower");
            Device::Cpu
        }
    }
}

fn unet_config(family: Family, sliced: Option<usize>) -> unet_2d::UNet2DConditionModelConfig {
    let bc = |out_channels, use_cross_attn, attention_head_dim| unet_2d::BlockConfig { out_channels, use_cross_attn, attention_head_dim };
    let sd2 = family == Family::SdTurbo;
    let heads = if sd2 { [5, 10, 20, 20] } else { [8, 8, 8, 8] };
    unet_2d::UNet2DConditionModelConfig {
        blocks: vec![bc(320, Some(1), heads[0]), bc(640, Some(1), heads[1]), bc(1280, Some(1), heads[2]), bc(1280, None, heads[3])],
        center_input_sample: false,
        cross_attention_dim: if sd2 { 1024 } else { 768 },
        downsample_padding: 1,
        flip_sin_to_cos: true,
        freq_shift: 0.,
        layers_per_block: 2,
        mid_block_scale_factor: 1.,
        norm_eps: 1e-5,
        norm_num_groups: 32,
        sliced_attention_size: sliced,
        use_linear_projection: sd2,
    }
}

fn clip_config(family: Family) -> clip::Config {
    if family == Family::SdTurbo { clip::Config::v2_1() } else { clip::Config::v1_5() }
}

fn vae_config() -> vae::AutoEncoderKLConfig {
    vae::AutoEncoderKLConfig {
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        latent_channels: 4,
        norm_num_groups: 32,
        use_quant_conv: true,
        use_post_quant_conv: true,
    }
}

/// A loaded checkpoint: base weights on the GPU, the LoRA-merged copies in use, and the modules.
pub(super) struct Loaded {
    checkpoint: PathBuf,
    vae_file: Option<PathBuf>,
    tokenizer_file: PathBuf,
    family: Family,
    device: Device,
    in_channels: usize,
    base_unet: HashMap<String, Tensor>,
    base_clip: HashMap<String, Tensor>,
    index: LoraIndex,
    /// The weights the UNet is built from (base, or base with LoRAs merged).
    unet_weights: HashMap<String, Tensor>,
    clip_model: clip::ClipTextTransformer,
    vae: vae::AutoEncoderKL,
    tokenizer: tokenizers::Tokenizer,
    loras: Vec<(PathBuf, f32)>,
    /// Low-rank factors of the LoRAs in use, read once per file, so a weight change re-merges
    /// from memory.
    factors: HashMap<PathBuf, LoraFactors>,
}

/// One LoRA's factors, matched to this checkpoint's layers.
struct LoraFactors {
    /// The file's size and modification time when read: a replaced file is read again.
    stamp: (u64, Option<std::time::SystemTime>),
    layers: Vec<LoraLayer>,
}

/// `delta = up @ down * scale(weight, alpha, rank)` for one weight.
struct LoraLayer {
    component: Component,
    target: String,
    /// `(out, rank)`.
    up: Tensor,
    /// `(rank, in)`.
    down: Tensor,
    alpha: Option<f32>,
    rank: usize,
}

fn file_stamp(path: &Path) -> (u64, Option<std::time::SystemTime>) {
    std::fs::metadata(path).map_or((0, None), |m| (m.len(), m.modified().ok()))
}

impl Loaded {
    pub(super) fn checkpoint(&self) -> &Path {
        &self.checkpoint
    }

    pub(super) fn matches(&self, m: &ResolvedModels) -> bool {
        self.checkpoint == m.checkpoint && self.vae_file == m.vae && self.tokenizer_file == m.tokenizer
    }

    pub(super) fn load(m: &ResolvedModels, progress: &dyn Progress) -> Result<Loaded> {
        let name = m.checkpoint.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        progress.report(0.0, &format!("Loading {name}"));
        let h = Header::read(&m.checkpoint)?;
        check_memory(m.family)?;
        let device = device();
        let mut f = std::fs::File::open(&m.checkpoint).map_err(|e| MlError::io(m.checkpoint.display(), e))?;
        let mut keys: Vec<(&String, u64)> = h.tensors.iter().map(|(k, t)| (k, t.offsets[0])).collect();
        keys.sort_by_key(|(_, o)| *o);
        let total = keys.len().max(1);
        let mut maps: HashMap<Component, HashMap<String, Tensor>> = HashMap::new();
        let mut index = LoraIndex::default();
        for (n, (key, _)) in keys.into_iter().enumerate() {
            let targets = keymap::map(key);
            let Some(first) = targets.first() else { continue };
            if first.component == Component::Vae && m.vae.is_some() {
                continue;
            }
            let t = read_tensor(&h, &mut f, key, component_dtype(first.component), &Device::Cpu)?;
            for target in &targets {
                let tt = apply_transform(&t, target.transform)?.to_device(&device).map_err(ce)?;
                index.add(target.component, &target.name);
                index.add_ldm(key, target.component, &target.name);
                maps.entry(target.component).or_default().insert(target.name.clone(), tt);
            }
            if n % 64 == 0 {
                progress.check()?;
                progress.report(0.10 * n as f32 / total as f32, &format!("Loading {name}"));
            }
        }
        drop(f);
        if let Some(vae_path) = &m.vae {
            maps.insert(Component::Vae, read_vae_file(vae_path, &device)?);
        }
        let base_unet = maps.remove(&Component::Unet).unwrap_or_default();
        let base_clip = maps.remove(&Component::Clip).unwrap_or_default();
        let vae_w = maps.remove(&Component::Vae).unwrap_or_default();
        let in_channels =
            base_unet.get("conv_in.weight").and_then(|t| t.dims().get(1).copied()).ok_or_else(|| MlError::BadModel(format!("{name}: no UNet found")))?;
        let vae = vae::AutoEncoderKL::new(VarBuilder::from_tensors(vae_w, VAE_DTYPE, &device), 3, 3, vae_config())
            .map_err(|e| MlError::BadModel(format!("{name}: VAE: {e}")))?;
        let clip_model = build_clip(&base_clip, m.family, &device).map_err(|e| MlError::BadModel(format!("{name}: text encoder: {e}")))?;
        // Build once to check every UNet weight is present and shaped right.
        unet_2d::UNet2DConditionModel::new(
            VarBuilder::from_tensors(base_unet.clone(), UNET_DTYPE, &device),
            in_channels,
            4,
            false,
            unet_config(m.family, None),
        )
        .map_err(|e| MlError::BadModel(format!("{name}: UNet: {e}")))?;
        let tokenizer = tokenizers::Tokenizer::from_file(&m.tokenizer).map_err(|e| MlError::BadModel(format!("tokenizer: {e}")))?;
        Ok(Loaded {
            checkpoint: m.checkpoint.clone(),
            vae_file: m.vae.clone(),
            tokenizer_file: m.tokenizer.clone(),
            family: m.family,
            device,
            in_channels,
            unet_weights: base_unet.clone(),
            base_unet,
            base_clip,
            index,
            clip_model,
            vae,
            tokenizer,
            loras: Vec::new(),
            factors: HashMap::new(),
        })
    }

    /// Merge `loras` into the UNet and text encoder, always from the base weights. Each file's
    /// factors are read once and cached, so changing a weight only re-runs the merge.
    pub(super) fn set_loras(&mut self, loras: &[(PathBuf, f32)], progress: &dyn Progress) -> Result<()> {
        if self.loras == loras {
            return Ok(());
        }
        let started = std::time::Instant::now();
        self.factors.retain(|path, f| loras.iter().any(|(p, _)| p == path) && f.stamp == file_stamp(path));
        for (i, (path, _)) in loras.iter().enumerate() {
            if self.factors.contains_key(path) {
                continue;
            }
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            progress.report(0.10 + 0.02 * i as f32, &format!("Reading {name}"));
            progress.check()?;
            let factors = self.read_factors(path, &name)?;
            self.factors.insert(path.clone(), factors);
        }
        let read_ms = started.elapsed().as_millis();
        progress.report(0.18, "Applying LoRAs");
        // Group the layers by the weight they change, then merge one weight at a time: the
        // expanded deltas are as large as the weights, so they can't all be held at once.
        type Target<'a> = (bool, &'a str);
        let mut by_target: Vec<(Target, Vec<(&LoraLayer, f32)>)> = Vec::new();
        let mut slot: HashMap<Target, usize> = HashMap::new();
        for (path, weight) in loras {
            let Some(f) = self.factors.get(path) else { continue };
            for l in &f.layers {
                let key = (l.component == Component::Unet, l.target.as_str());
                let i = *slot.entry(key).or_insert_with(|| {
                    by_target.push((key, Vec::new()));
                    by_target.len() - 1
                });
                if let Some((_, v)) = by_target.get_mut(i) {
                    v.push((l, *weight));
                }
            }
        }
        let mut unet = self.base_unet.clone();
        let mut clip_w = self.base_clip.clone();
        for (n, ((is_unet, target), layers)) in by_target.iter().enumerate() {
            if n % 32 == 0 {
                progress.check()?;
                self.device.synchronize().map_err(ce)?;
            }
            let map = if *is_unet { &mut unet } else { &mut clip_w };
            let Some(w0) = map.get(*target) else { continue };
            let mut w = w0.to_dtype(DType::F32).map_err(ce)?;
            for (l, weight) in layers {
                let s = f64::from(lora_keys::scale(*weight, l.alpha, l.rank));
                if s != 0.0 {
                    let d = l.up.matmul(&l.down).and_then(|d| d.affine(s, 0.0)).and_then(|d| d.reshape(w0.shape())).map_err(ce)?;
                    w = w.add(&d).map_err(ce)?;
                }
            }
            let merged = w.to_dtype(w0.dtype()).map_err(ce)?;
            map.insert((*target).to_string(), merged);
        }
        self.clip_model = build_clip(&clip_w, self.family, &self.device).map_err(ce)?;
        self.device.synchronize().map_err(ce)?;
        self.unet_weights = unet;
        self.loras = loras.to_vec();
        log::info!("LoRAs: read in {read_ms} ms, merged in {} ms", started.elapsed().as_millis().saturating_sub(read_ms));
        Ok(())
    }

    /// Read the factors of the LoRA at `path` that match this checkpoint.
    fn read_factors(&self, path: &Path, name: &str) -> Result<LoraFactors> {
        let stamp = file_stamp(path);
        let h = Header::read(path)?;
        let plan = lora_keys::plan(name, &h, &self.index)?;
        if plan.unmatched > 0 {
            log::info!("{name}: {} layers don't match this model and were skipped", plan.unmatched);
        }
        let mut f = std::fs::File::open(path).map_err(|e| MlError::io(path.display(), e))?;
        let mut layers = Vec::with_capacity(plan.pairs.len());
        for pair in &plan.pairs {
            let base = match pair.component {
                Component::Unet => &self.base_unet,
                Component::Clip => &self.base_clip,
                Component::Vae => continue,
            };
            let Some(w0) = base.get(&pair.target) else { continue };
            let down = read_tensor(&h, &mut f, &pair.down, DType::F32, &self.device)?;
            let up = read_tensor(&h, &mut f, &pair.up, DType::F32, &self.device)?;
            let alpha = match &pair.alpha {
                Some(a) => read_tensor(&h, &mut f, a, DType::F32, &Device::Cpu)?.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(ce)?.first().copied(),
                None => None,
            };
            let rank = down.dims().first().copied().unwrap_or(0);
            let out = up.dims().first().copied().unwrap_or(0);
            if rank == 0 || out == 0 || up.elem_count() != out * rank || down.elem_count() % rank != 0 {
                continue;
            }
            let cols = down.elem_count() / rank;
            if out.checked_mul(cols) != Some(w0.elem_count()) {
                log::info!("{name}: {} has a different shape; skipped", pair.target);
                continue;
            }
            let up = up.reshape((out, rank)).map_err(ce)?;
            let down = down.reshape((rank, cols)).map_err(ce)?;
            layers.push(LoraLayer { component: pair.component, target: pair.target.clone(), up, down, alpha, rank });
        }
        Ok(LoraFactors { stamp, layers })
    }

    fn encode_prompt(&self, text: &str) -> Result<Tensor> {
        let cfg = clip_config(self.family);
        let max = cfg.max_position_embeddings;
        let eos = self.tokenizer.token_to_id("<|endoftext|>").unwrap_or(49407);
        let pad = cfg.pad_with.as_deref().and_then(|p| self.tokenizer.token_to_id(p)).unwrap_or(eos);
        let enc = self.tokenizer.encode(text, true).map_err(|e| MlError::Other(format!("tokenizer: {e}")))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        if ids.len() > max {
            log::info!("prompt is longer than {max} tokens; the end is ignored");
            ids.truncate(max - 1);
            ids.push(eos);
        }
        ids.resize(max, pad);
        let tokens = Tensor::new(ids.as_slice(), &self.device).and_then(|t| t.unsqueeze(0)).map_err(ce)?;
        self.clip_model.forward(&tokens).and_then(|t| t.to_dtype(UNET_DTYPE)).map_err(ce)
    }

    fn image_tensor(&self, img: &Rgba8) -> Result<Tensor> {
        let (w, h) = (img.width as usize, img.height as usize);
        let mut planes = vec![0f32; 3 * w * h];
        for (i, px) in img.data.chunks_exact(4).enumerate() {
            for c in 0..3 {
                if let (Some(dst), Some(&v)) = (planes.get_mut(c * w * h + i), px.get(c)) {
                    *dst = f32::from(v) / 127.5 - 1.0;
                }
            }
        }
        Tensor::from_vec(planes, (1, 3, h, w), &self.device).and_then(|t| t.to_dtype(VAE_DTYPE)).map_err(ce)
    }

    /// VAE-encode an image tensor (`[1, 3, h, w]`, −1…1) to scaled latents, in overlapping tiles
    /// so large frames stay within GPU memory.
    fn encode_image(&self, img: &Tensor, seed: u64) -> Result<Tensor> {
        let (_, _, h, w) = img.dims4().map_err(ce)?;
        let (lh, lw) = (h / 8, w / 8);
        let mut acc = Blend::new(4, lh, lw);
        for (n, (y, x, th, tw)) in tiles2(lh, lw, VAE_TILE, VAE_OVERLAP).into_iter().enumerate() {
            let part = img.narrow(2, y * 8, th * 8).and_then(|t| t.narrow(3, x * 8, tw * 8)).map_err(ce)?;
            // The VAE samples its latent distribution: seed it so a seed gives the same image.
            let _ = self.device.set_seed(seed.wrapping_add(n as u64));
            let lat = self.vae.encode(&part).and_then(|d| d.sample()).map_err(ce)?;
            acc.add(&to_cpu_vec(&lat)?, y, x, th, tw, VAE_OVERLAP);
            self.device.synchronize().map_err(ce)?;
        }
        Tensor::from_vec(acc.finish(), (1, 4, lh, lw), &self.device).and_then(|t| t.affine(LATENT_SCALE, 0.0)).map_err(ce)
    }

    /// Decode scaled latents to pixels, in overlapping tiles (see [`Self::encode_image`]).
    fn decode(&self, latents: &Tensor) -> Result<Rgba8> {
        let (_, _, lh, lw) = latents.dims4().map_err(ce)?;
        let (h, w) = (lh * 8, lw * 8);
        let latents = latents.to_dtype(VAE_DTYPE).and_then(|l| l.affine(1.0 / LATENT_SCALE, 0.0)).map_err(ce)?;
        self.device.synchronize().map_err(ce)?;
        let mut acc = Blend::new(3, h, w);
        for (y, x, th, tw) in tiles2(lh, lw, VAE_TILE, VAE_OVERLAP) {
            let part = latents.narrow(2, y, th).and_then(|t| t.narrow(3, x, tw)).map_err(ce)?;
            let img = self.vae.decode(&part).map_err(ce)?;
            acc.add(&to_cpu_vec(&img)?, y * 8, x * 8, th * 8, tw * 8, VAE_OVERLAP * 8);
            self.device.synchronize().map_err(ce)?;
        }
        let v = acc.finish();
        let mut out = vec![255u8; w * h * 4];
        for (i, px) in out.chunks_exact_mut(4).enumerate() {
            for c in 0..3 {
                if let (Some(d), Some(s)) = (px.get_mut(c), v.get(c * w * h + i)) {
                    *d = ((s * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        Rgba8::from_raw(w as u32, h as u32, out)
    }

    fn mask_tensor(&self, m: &Mask, w: usize, h: usize, binary: bool) -> Result<Tensor> {
        let small = m.resized(w as u32, h as u32)?;
        let v: Vec<f32> = small.data.iter().map(|&x| if binary { if x >= 128 { 1.0 } else { 0.0 } } else { f32::from(x) / 255.0 }).collect();
        Tensor::from_vec(v, (1, 1, h, w), &self.device).map_err(ce)
    }

    fn noise(&self, seed: u64, shape: (usize, usize, usize, usize)) -> Result<Tensor> {
        let n = shape.0 * shape.1 * shape.2 * shape.3;
        Tensor::from_vec(sampling::gaussian(seed, n), shape, &self.device).map_err(ce)
    }

    pub(super) fn generate(&mut self, req: &GenerateRequest, progress: &dyn Progress) -> Result<Rgba8> {
        let (lw, lh) = ((req.width / 8) as usize, (req.height / 8) as usize);
        let turbo = self.family == Family::SdTurbo;
        let inpaint_model = self.in_channels == 9;
        let use_cfg = req.cfg > 1.0;
        let unet = unet_2d::UNet2DConditionModel::new(
            VarBuilder::from_tensors(self.unet_weights.clone(), UNET_DTYPE, &self.device),
            self.in_channels,
            4,
            false,
            unet_config(self.family, None),
        )
        .map_err(ce)?;
        progress.report(0.12, "Reading the prompt");
        let cond = self.encode_prompt(&req.prompt)?;
        let uncond = if use_cfg { Some(self.encode_prompt(&req.negative)?) } else { None };

        let schedule = Schedule::sd();
        let all = schedule.sigmas(ScheduleKind::for_sampler(req.sampler, turbo), req.steps as usize);
        let init = req.init.as_ref().filter(|_| req.strength < 1.0 || req.mask.is_some() || inpaint_model);
        let i0 = if init.is_some() && req.mask.is_none() { sampling::img2img_start(all.len() - 1, req.strength) } else { 0 };
        let sigmas: Vec<f64> = all.get(i0..).map(<[f64]>::to_vec).unwrap_or_else(|| all.clone());
        let shape = (1, 4, lh, lw);
        let noise = self.noise(req.seed, shape)?;
        let sigma0 = sigmas.first().copied().unwrap_or(1.0);

        let init_t = init.map(|i| self.image_tensor(i)).transpose()?;
        let init_latent = init_t.as_ref().map(|t| self.encode_image(t, req.seed)).transpose()?;
        let mask_soft = req.mask.as_ref().map(|m| self.mask_tensor(m, lw, lh, false)).transpose()?;
        // 9-channel inpainting: the binary mask and the masked image's latent ride along.
        let extra = if inpaint_model {
            let m = match &req.mask {
                Some(m) => self.mask_tensor(m, lw, lh, true)?,
                None => Tensor::ones((1, 1, lh, lw), DType::F32, &self.device).map_err(ce)?,
            };
            let masked = match (&init_t, &req.mask) {
                (Some(img), Some(mask)) => {
                    let full = self.mask_tensor(mask, req.width as usize, req.height as usize, true)?;
                    let keep = full.affine(-1.0, 1.0).map_err(ce)?;
                    let masked_img = img.broadcast_mul(&keep).map_err(ce)?;
                    self.encode_image(&masked_img, req.seed)?
                }
                (Some(img), None) => self.encode_image(&img.zeros_like().map_err(ce)?, req.seed)?,
                (None, _) => {
                    self.encode_image(&Tensor::zeros((1, 3, req.height as usize, req.width as usize), VAE_DTYPE, &self.device).map_err(ce)?, req.seed)?
                }
            };
            Some(Tensor::cat(&[&m, &masked.to_dtype(DType::F32).map_err(ce)?], 1).map_err(ce)?)
        } else {
            None
        };
        // Latent blending: only for ordinary checkpoints with an init image and a mask.
        let blend = match (&init_latent, &mask_soft) {
            (Some(l), Some(m)) if !inpaint_model => Some((l.to_dtype(DType::F32).map_err(ce)?, m.clone())),
            _ => None,
        };
        let mut x = match (&init_latent, inpaint_model) {
            (Some(l), false) => l.to_dtype(DType::F32).and_then(|l| l.add(&noise.affine(sigma0, 0.0)?)).map_err(ce)?,
            _ => noise.affine(sigma0, 0.0).map_err(ce)?,
        };
        let steps = sigmas.len().saturating_sub(1);
        // Previews cost a small VAE decode each: at most three, and none for quick runs.
        let preview_every = if steps >= 8 { (steps / 4).max(3) } else { usize::MAX };
        let mut old: Option<Tensor> = None;
        for i in 0..steps {
            progress.check()?;
            let sigma = sigmas.get(i).copied().unwrap_or(0.0);
            let t = schedule.timestep(sigma);
            let mut inp = x.affine(sampling::c_in(sigma), 0.0).map_err(ce)?;
            if let Some(e) = &extra {
                inp = Tensor::cat(&[&inp, e], 1).map_err(ce)?;
            }
            let inp = inp.to_dtype(UNET_DTYPE).map_err(ce)?;
            // Two batch-1 calls instead of one batch-2 call: half the attention memory.
            let cond_eps = unet.forward(&inp, t, &cond).and_then(|e| e.to_dtype(DType::F32)).map_err(ce)?;
            let eps = match &uncond {
                Some(un) => {
                    self.device.synchronize().map_err(ce)?;
                    let u = unet.forward(&inp, t, un).and_then(|e| e.to_dtype(DType::F32)).map_err(ce)?;
                    u.add(&cond_eps.sub(&u).map_err(ce)?.affine(f64::from(req.cfg), 0.0).map_err(ce)?).map_err(ce)?
                }
                None => cond_eps,
            };
            let denoised = x.sub(&eps.affine(sigma, 0.0).map_err(ce)?).map_err(ce)?;
            let c = sampling::step(req.sampler, &sigmas, i);
            let mut next = x.affine(c.x, 0.0).and_then(|a| a.add(&denoised.affine(c.denoised, 0.0)?)).map_err(ce)?;
            if c.old != 0.0
                && let Some(o) = &old
            {
                next = next.add(&o.affine(c.old, 0.0).map_err(ce)?).map_err(ce)?;
            }
            if c.noise > 0.0 {
                next = next.add(&self.noise(sampling::step_seed(req.seed, i), shape)?.affine(c.noise, 0.0).map_err(ce)?).map_err(ce)?;
            }
            if let Some((lat, m)) = &blend {
                let sn = sigmas.get(i + 1).copied().unwrap_or(0.0);
                let reference = lat.add(&noise.affine(sn, 0.0).map_err(ce)?).map_err(ce)?;
                let keep = m.affine(-1.0, 1.0).map_err(ce)?;
                next = next.broadcast_mul(m).and_then(|a| a.add(&reference.broadcast_mul(&keep)?)).map_err(ce)?;
            }
            old = Some(denoised.clone());
            x = next;
            self.device.synchronize().map_err(ce)?;
            progress.report(0.15 + 0.8 * (i + 1) as f32 / steps.max(1) as f32, &format!("Step {} of {steps}", i + 1));
            if (i + 1) % preview_every == 0 && i + 1 < steps {
                // A small preview of what the model currently predicts.
                let k = (lh.max(lw) / 16).max(1);
                if let Ok(small) = denoised.interpolate2d((lh / k).max(4), (lw / k).max(4))
                    && let Ok(img) = self.decode(&small)
                {
                    progress.preview(&img);
                }
            }
        }
        progress.report(0.96, "Decoding");
        self.decode(&x)
    }
}

fn to_cpu_vec(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32).and_then(|t| t.to_device(&Device::Cpu)).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>()).map_err(ce)
}

/// Tile origins along one axis: `tile`-long windows overlapping by at least `overlap`.
fn tile_starts(len: usize, tile: usize, overlap: usize) -> Vec<usize> {
    if len <= tile {
        return vec![0];
    }
    let step = tile.saturating_sub(overlap).max(1);
    let mut out: Vec<usize> = (0..).map(|i| i * step).take_while(|&s| s + tile < len).collect();
    out.push(len - tile);
    out
}

/// `(y, x, height, width)` of every tile covering a `h × w` grid.
fn tiles2(h: usize, w: usize, tile: usize, overlap: usize) -> Vec<(usize, usize, usize, usize)> {
    let (th, tw) = (h.min(tile), w.min(tile));
    tile_starts(h, tile, overlap).into_iter().flat_map(|y| tile_starts(w, tile, overlap).into_iter().map(move |x| (y, x, th, tw))).collect()
}

/// Feathered accumulation of planar (`c × h × w`) tiles into one image.
struct Blend {
    c: usize,
    h: usize,
    w: usize,
    sum: Vec<f32>,
    weight: Vec<f32>,
}

impl Blend {
    fn new(c: usize, h: usize, w: usize) -> Blend {
        Blend { c, h, w, sum: vec![0.0; c * h * w], weight: vec![0.0; h * w] }
    }

    /// Ramp weights over `ramp` pixels at tile edges that lie inside the image.
    fn add(&mut self, tile: &[f32], y0: usize, x0: usize, th: usize, tw: usize, ramp: usize) {
        let edge = |i: usize, len: usize, at_start: bool, at_end: bool| -> f32 {
            let r = ramp.max(1) as f32;
            let a = if at_start { 1.0 } else { ((i as f32 + 0.5) / r).min(1.0) };
            let b = if at_end { 1.0 } else { ((len - i) as f32 - 0.5) / r };
            a.min(b.min(1.0)).max(1e-3)
        };
        for ty in 0..th {
            let wy = edge(ty, th, y0 == 0, y0 + th >= self.h);
            for tx in 0..tw {
                let wgt = wy * edge(tx, tw, x0 == 0, x0 + tw >= self.w);
                let (y, x) = (y0 + ty, x0 + tx);
                if y >= self.h || x >= self.w {
                    continue;
                }
                if let Some(acc) = self.weight.get_mut(y * self.w + x) {
                    *acc += wgt;
                }
                for c in 0..self.c {
                    if let (Some(d), Some(v)) = (self.sum.get_mut((c * self.h + y) * self.w + x), tile.get((c * th + ty) * tw + tx)) {
                        *d += wgt * v;
                    }
                }
            }
        }
    }

    fn finish(mut self) -> Vec<f32> {
        let plane = self.h * self.w;
        for (i, v) in self.sum.iter_mut().enumerate() {
            let wgt = self.weight.get(i % plane.max(1)).copied().unwrap_or(0.0);
            if wgt > 0.0 {
                *v /= wgt;
            }
        }
        self.sum
    }
}

/// A standalone VAE file: original (LDM) names, or already in the diffusers layout.
fn read_vae_file(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let h = Header::read(path)?;
    let mut f = std::fs::File::open(path).map_err(|e| MlError::io(path.display(), e))?;
    let mut out = HashMap::new();
    let names: Vec<String> = h.tensors.keys().cloned().collect();
    for key in names {
        let diffusers = key.contains("down_blocks") || key.contains("up_blocks") || key.contains("mid_block");
        let targets = if diffusers {
            vec![keymap::Target { component: Component::Vae, name: key.clone(), transform: Transform::Same }]
        } else {
            let k = key.strip_prefix("first_stage_model.").unwrap_or(&key);
            keymap::map(&format!("first_stage_model.{k}"))
        };
        if targets.is_empty() {
            continue;
        }
        let t = read_tensor(&h, &mut f, &key, VAE_DTYPE, &Device::Cpu)?;
        for target in targets {
            out.insert(target.name, apply_transform(&t, target.transform)?.to_device(device).map_err(ce)?);
        }
    }
    if out.is_empty() {
        return Err(MlError::BadModel(format!("{}: not a VAE", path.display())));
    }
    Ok(out)
}

fn build_clip(weights: &HashMap<String, Tensor>, family: Family, device: &Device) -> candle_core::Result<clip::ClipTextTransformer> {
    clip::ClipTextTransformer::new(VarBuilder::from_tensors(weights.clone(), CLIP_DTYPE, device), &clip_config(family))
}

/// Refuse to load a model that can't fit, instead of letting macOS swap to a crawl.
fn check_memory(family: Family) -> Result<()> {
    let Some(total) = crate::system::physical_memory() else { return Ok(()) };
    // UNet in f16, text encoder and VAE in f32, plus working memory for a 512×512 frame.
    let need: u64 = match family {
        Family::SdTurbo => 6 << 30,
        _ => 4 << 30,
    };
    if total < need + (2 << 30) {
        return Err(MlError::OutOfMemory(format!("this model needs about {} GB of memory and this Mac has {} GB", (need + (2 << 30)) >> 30, total >> 30)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_cover_and_blend_seamlessly() {
        assert_eq!(tile_starts(64, 32, 8), vec![0, 24, 32]);
        assert_eq!(tile_starts(20, 32, 8), vec![0]);
        assert_eq!(tile_starts(128, 32, 8), vec![0, 24, 48, 72, 96]);
        // A constant image stays constant through the feathered blend.
        let (h, w) = (40, 72);
        let mut b = Blend::new(2, h, w);
        for (y, x, th, tw) in tiles2(h, w, 32, 8) {
            b.add(&vec![0.25; 2 * th * tw], y, x, th, tw, 8);
        }
        assert!(b.finish().iter().all(|v| (v - 0.25).abs() < 1e-5));
    }

    /// The UNet's denoised prediction lands near a real image's latent at every noise level.
    /// A weight change re-merges from the cached factors without reading the file again, and the
    /// empty stack restores the base weights.
    #[test]
    #[ignore = "needs real model files (PHOTOKA_MODELS_DIR)"]
    fn lora_weight_changes_reuse_cached_factors() {
        let dir = PathBuf::from(std::env::var_os("PHOTOKA_MODELS_DIR").unwrap());
        let store = crate::store::ModelStore::new(dir.clone());
        let req = crate::params::GenerateRequest {
            checkpoint: "v1-5-pruned-emaonly-fp16.safetensors".into(),
            vae: None,
            loras: vec![],
            prompt: String::new(),
            negative: String::new(),
            steps: 4,
            cfg: 7.0,
            seed: 1,
            sampler: crate::params::Sampler::Euler,
            strength: 1.0,
            width: 512,
            height: 512,
            init: None,
            mask: None,
        };
        let mut l = Loaded::load(&super::super::resolve(&store, &req).unwrap(), &crate::NoProgress).unwrap();
        let lora = dir.join("loras/lcm-lora-sdv1-5.safetensors");
        let t = std::time::Instant::now();
        l.set_loras(&[(lora.clone(), 1.0)], &crate::NoProgress).unwrap();
        let first = t.elapsed();
        let layers = l.factors.get(&lora).map_or(0, |f| f.layers.len());
        assert!(layers > 100, "only {layers} LCM-LoRA layers matched");
        let t = std::time::Instant::now();
        l.set_loras(&[(lora.clone(), 0.5)], &crate::NoProgress).unwrap();
        let second = t.elapsed();
        eprintln!("LoRA merge: first {first:?}, weight change {second:?}, {layers} layers");
        let changed = l.unet_weights.iter().filter(|(k, w)| l.base_unet.get(*k).is_some_and(|b| b.id() != w.id())).count();
        assert!(changed > 100, "only {changed} weights changed");
        l.set_loras(&[], &crate::NoProgress).unwrap();
        assert!(l.factors.is_empty());
        let diff = l.unet_weights.iter().filter(|(k, w)| l.base_unet.get(*k).is_some_and(|b| b.id() != w.id())).count();
        assert_eq!(diff, 0, "the empty stack must restore the base weights");
    }

    #[test]
    #[ignore = "needs real model files (PHOTOKA_MODELS_DIR)"]
    fn unet_denoises_real_latents() {
        let dir = std::env::var_os("PHOTOKA_MODELS_DIR").unwrap();
        let store = crate::store::ModelStore::new(PathBuf::from(dir));
        let req = crate::params::GenerateRequest {
            checkpoint: "v1-5-pruned-emaonly-fp16.safetensors".into(),
            vae: None,
            loras: vec![],
            prompt: String::new(),
            negative: String::new(),
            steps: 4,
            cfg: 7.0,
            seed: 1,
            sampler: crate::params::Sampler::Euler,
            strength: 1.0,
            width: 512,
            height: 512,
            init: None,
            mask: None,
        };
        let l = Loaded::load(&super::super::resolve(&store, &req).unwrap(), &crate::NoProgress).unwrap();
        let mut img = Rgba8::new(512, 512).unwrap();
        for (i, px) in img.data.chunks_exact_mut(4).enumerate() {
            let (x, y) = (i % 512, i / 512);
            let sky = y < 300;
            px.copy_from_slice(&[
                if sky { 90 + (y / 4) as u8 } else { 40 + (x / 8) as u8 },
                if sky { 140 + (y / 6) as u8 } else { 120 },
                if sky { 230 } else { 50 },
                255,
            ]);
        }
        let lat = l.encode_image(&l.image_tensor(&img).unwrap(), 1).unwrap();
        // VAE round trip (RGB only).
        let back = l.decode(&lat).unwrap();
        let err = back
            .data
            .chunks_exact(4)
            .zip(img.data.chunks_exact(4))
            .map(|(a, b)| (0..3).map(|c| (f64::from(a[c]) - f64::from(b[c])).abs()).sum::<f64>())
            .sum::<f64>()
            / (512.0 * 512.0 * 3.0);
        assert!(err < 6.0, "VAE round trip error {err}");
        let emb = l.encode_prompt("a photo of a field under a blue sky").unwrap();
        let unet = unet_2d::UNet2DConditionModel::new(
            VarBuilder::from_tensors(l.unet_weights.clone(), UNET_DTYPE, &l.device),
            4,
            4,
            false,
            unet_config(Family::Sd15, None),
        )
        .unwrap();
        let n = l.noise(7, (1, 4, 64, 64)).unwrap();
        let sch = Schedule::sd();
        let zero_guess = lat.abs().unwrap().mean_all().unwrap().to_vec0::<f32>().unwrap();
        for sigma in [14.0, 2.0, 0.5] {
            let x = lat.add(&n.affine(sigma, 0.0).unwrap()).unwrap();
            let eps = unet
                .forward(&x.affine(sampling::c_in(sigma), 0.0).unwrap().to_dtype(UNET_DTYPE).unwrap(), sch.timestep(sigma), &emb)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let den = x.sub(&eps.affine(sigma, 0.0).unwrap()).unwrap();
            let err = den.sub(&lat).unwrap().abs().unwrap().mean_all().unwrap().to_vec0::<f32>().unwrap();
            assert!(err < zero_guess / 2.0, "sigma {sigma}: denoised error {err} vs {zero_guess}");
        }
    }
}

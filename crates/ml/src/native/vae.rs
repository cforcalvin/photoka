//! The Stable Diffusion VAE (AutoencoderKL) of the built-in engine.
//!
//! The same layers and weight names as candle's `stable_diffusion::vae`, but with its own
//! convolutions and group norms, which are several times faster on Metal, its own attention, in
//! chunks of queries so large frames don't need a gigabyte-sized score matrix, and
//! a frame always goes through in one pass, so the group norms see the whole image (tiles each
//! get their own statistics and show as patches of different colour). Memory stays bounded
//! because the 3×3 convolutions run in bands of rows and the pass waits for the GPU after every
//! block: candle only recycles a buffer once the work queued on it has finished, so one opaque
//! pass piles up every layer's buffers (about 11 GB for a 512×512 frame in f32, which silently
//! returns garbage on a 16 GB Mac).

use candle_core::{D, DType, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};

/// The shape of an SD VAE.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct VaeConfig {
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub latent_channels: usize,
    pub groups: usize,
}

impl VaeConfig {
    /// SD 1.x and 2.x.
    pub(super) fn sd() -> VaeConfig {
        VaeConfig { block_out_channels: vec![128, 256, 512, 512], layers_per_block: 2, latent_channels: 4, groups: 32 }
    }
}

const EPS: f64 = 1e-6;
/// Largest column matrix a 3×3 convolution builds at once; larger frames go in bands of rows.
const COL_BYTES: usize = 256 << 20;
/// Largest input a 3×3 convolution pads whole, which is fastest. Bigger inputs are padded a band
/// at a time: slower, but without a second whole copy (which the buffer pool rounds up to a power
/// of two). At 1024×1024 that keeps the peak at 7.5 GB instead of 8.6 GB.
const PAD_BYTES: usize = 256 << 20;
/// Largest f32 copy a group norm makes at once to sum its statistics.
const NORM_BYTES: usize = 128 << 20;
/// Largest f32 attention score matrix computed at once.
const SCORE_BYTES: usize = 256 << 20;

/// Group normalization of `[c, l]` activations, with the statistics in f32 from per-channel
/// sums. (candle's version broadcasts over the whole frame several times and is ~3× slower.)
struct Norm {
    /// `[c, 1]`, f32.
    gamma: Tensor,
    beta: Tensor,
    groups: usize,
}

impl Norm {
    fn new(vb: VarBuilder, c: usize, groups: usize) -> Result<Norm> {
        if groups == 0 || !c.is_multiple_of(groups) {
            candle_core::bail!("{c} channels don't split into {groups} groups");
        }
        let get = |name| vb.get(c, name).and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.reshape((c, 1)));
        Ok(Norm { gamma: get("weight")?, beta: get("bias")?, groups })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (c, l) = x.dims2()?;
        let per = c / self.groups;
        self.forward_in(x, (NORM_BYTES / (per * l * 4).max(1)).clamp(1, self.groups))
    }

    /// Sums the statistics `groups_at_once` groups at a time.
    fn forward_in(&self, x: &Tensor, groups_at_once: usize) -> Result<Tensor> {
        let (c, l) = x.dims2()?;
        let per = c / self.groups;
        let n = (per * l).max(1) as f64;
        // Per-channel sums in f32, a few groups at a time.
        let chunk = groups_at_once.clamp(1, self.groups) * per;
        let (mut sums, mut squares) = (Vec::new(), Vec::new());
        for c0 in (0..c).step_by(chunk) {
            let part = x.narrow(0, c0, chunk.min(c - c0))?.to_dtype(DType::F32)?;
            sums.push(part.sum_keepdim(1)?);
            squares.push(part.sqr()?.sum_keepdim(1)?);
        }
        let group_sum = |v: &[Tensor]| Tensor::cat(v, 0)?.reshape((self.groups, per))?.sum_keepdim(1);
        let mean = (group_sum(&sums)? / n)?;
        let var = ((group_sum(&squares)? / n)? - mean.sqr()?)?.relu()?;
        let inv = (var + EPS)?.sqrt()?.recip()?;
        let per_channel = |t: &Tensor| t.broadcast_as((self.groups, per))?.contiguous()?.reshape((c, 1));
        let scale = per_channel(&inv)?.mul(&self.gamma)?;
        let shift = self.beta.sub(&per_channel(&mean)?.mul(&scale)?)?;
        x.broadcast_mul(&scale.to_dtype(x.dtype())?)?.broadcast_add(&shift.to_dtype(x.dtype())?)
    }
}

/// A 3×3 convolution (stride 1, padding 1) of `[c, h, w]` activations as a matmul per band of
/// rows. Each band's rows, with a row above and below, are zero-padded and flattened so that the
/// column matrix is nine shifted slices, each a plain 2-D copy; the bias rides along as a row of
/// ones. (candle's Metal im2col does 64-bit divisions per element and is ~4× slower than this
/// whole convolution, and padding the whole frame at once would cost two frame-sized copies.)
struct Conv3 {
    /// `[cout, 9 · cin + 1]`: the taps in (row, column, channel) order, then the bias.
    w: Tensor,
    cout: usize,
}

impl Conv3 {
    /// `extra_bias` is added to the convolution's own (a 1×1 shortcut's, folded in).
    fn new(vb: VarBuilder, cin: usize, cout: usize, extra_bias: Option<Tensor>) -> Result<Conv3> {
        let w = vb.get((cout, cin, 3, 3), "weight")?.permute((0, 2, 3, 1))?.contiguous()?.reshape((cout, 9 * cin))?;
        let mut b = vb.get(cout, "bias")?;
        if let Some(extra) = extra_bias {
            b = b.add(&extra.to_dtype(b.dtype())?)?;
        }
        Ok(Conv3 { w: Tensor::cat(&[&w, &b.reshape((cout, 1))?], 1)?, cout })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (cin, _, w) = x.dims3()?;
        let pad_whole = x.elem_count().saturating_mul(x.dtype().size_in_bytes()) <= PAD_BYTES;
        self.forward_in(x, COL_BYTES / ((9 * cin + 1) * (w + 2) * x.dtype().size_in_bytes()).max(1), pad_whole)
    }

    /// Runs `band` rows at a time, padding the whole input once or each band's rows.
    fn forward_in(&self, x: &Tensor, band: usize, pad_whole: bool) -> Result<Tensor> {
        let (cin, h, w) = x.dims3()?;
        let wp = w + 2;
        let band = band.clamp(1, h.max(1));
        let padded = if pad_whole { Some(x.pad_with_zeros(1, 1, 1)?.pad_with_zeros(2, 1, 1)?) } else { None };
        // Bands are written into one output; a single band is the output.
        let whole = if band < h { Some(Tensor::zeros((self.cout, h, w), x.dtype(), x.device())?) } else { None };
        let mut y0 = 0;
        while y0 < h {
            let rows = band.min(h - y0);
            // Rows y0 − 1 … y0 + rows, zero outside the image, padded left and right, flattened,
            // plus two spare zeros for the last slice. Output pixel (y, x) is column y·wp + x;
            // columns x ≥ w are dropped.
            let slab = match &padded {
                Some(padded) => padded.narrow(1, y0, rows + 2)?,
                None => {
                    let zero_row = || Tensor::zeros((cin, 1, w), x.dtype(), x.device());
                    let (top, bottom) = (y0.saturating_sub(1), (y0 + rows + 1).min(h));
                    let mut slab = Vec::with_capacity(3);
                    if y0 == 0 {
                        slab.push(zero_row()?);
                    }
                    slab.push(x.narrow(1, top, bottom - top)?);
                    if y0 + rows == h {
                        slab.push(zero_row()?);
                    }
                    Tensor::cat(&slab, 1)?.pad_with_zeros(2, 1, 1)?
                }
            };
            let xp = slab.reshape((cin, (rows + 2) * wp))?.pad_with_zeros(1, 0, 2)?;
            let n = rows * wp;
            let mut parts = Vec::with_capacity(10);
            for ky in 0..3 {
                for kx in 0..3 {
                    parts.push(xp.narrow(1, ky * wp + kx, n)?);
                }
            }
            parts.push(Tensor::ones((1, n), x.dtype(), x.device())?);
            let out = self.w.matmul(&Tensor::cat(&parts, 0)?)?;
            let out = out.reshape((self.cout * rows, wp))?.narrow(1, 0, w)?.contiguous()?.reshape((self.cout, rows, w))?;
            match &whole {
                Some(whole) => whole.slice_set(&out, 1, y0)?,
                None => return Ok(out),
            }
            y0 += rows;
        }
        whole.ok_or_else(|| candle_core::Error::Msg("the convolution's input has no rows".into()))
    }
}

/// A ResNet block without a time embedding, as in candle's `ResnetBlock2D`.
struct Resnet {
    norm1: Norm,
    conv1: Conv3,
    norm2: Norm,
    /// Its bias includes the shortcut's.
    conv2: Conv3,
    /// The 1×1 shortcut's weights, `[cout, cin]`, when the channel count changes.
    shortcut: Option<Tensor>,
}

impl Resnet {
    fn new(vb: VarBuilder, cin: usize, cout: usize, groups: usize) -> Result<Resnet> {
        let (shortcut, extra) = if cin == cout {
            (None, None)
        } else {
            let s = vb.pp("conv_shortcut");
            (Some(s.get((cout, cin, 1, 1), "weight")?.reshape((cout, cin))?), Some(s.get(cout, "bias")?))
        };
        Ok(Resnet {
            norm1: Norm::new(vb.pp("norm1"), cin, groups)?,
            conv1: Conv3::new(vb.pp("conv1"), cin, cout, None)?,
            norm2: Norm::new(vb.pp("norm2"), cout, groups)?,
            conv2: Conv3::new(vb.pp("conv2"), cout, cout, extra)?,
            shortcut,
        })
    }

    /// Waits for the GPU after each layer, so its temporary buffers are freed before the next.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (c, h, w) = x.dims3()?;
        let device = x.device();
        let flat = x.reshape((c, h * w))?;
        let y = candle_nn::ops::silu(&self.norm1.forward(&flat)?)?.reshape((c, h, w))?;
        device.synchronize()?;
        let y = self.conv1.forward(&y)?;
        let cout = y.dim(0)?;
        let y = candle_nn::ops::silu(&self.norm2.forward(&y.reshape((cout, h * w))?)?)?.reshape((cout, h, w))?;
        device.synchronize()?;
        let y = self.conv2.forward(&y)?;
        match &self.shortcut {
            Some(s) => y.add(&s.matmul(&flat)?.reshape((cout, h, w))?),
            None => y.add(x),
        }
    }
}

/// A `[c, c]` linear layer as a weight and a `[c, 1]` bias; the weight may be stored as a 1×1
/// convolution.
fn linear(vb: VarBuilder, c: usize) -> Result<(Tensor, Tensor)> {
    let w = match vb.get((c, c), "weight") {
        Ok(w) => w,
        Err(_) => vb.get((c, c, 1, 1), "weight")?.reshape((c, c))?,
    };
    Ok((w, vb.get(c, "bias")?.reshape((c, 1))?))
}

/// The mid block's single-head self-attention over every pixel, as in candle's
/// `AttentionBlock`, with the scores in f32, a chunk of queries at a time.
struct Attention {
    norm: Norm,
    q: (Tensor, Tensor),
    k: (Tensor, Tensor),
    v: (Tensor, Tensor),
    out: (Tensor, Tensor),
}

impl Attention {
    fn new(vb: VarBuilder, c: usize, groups: usize) -> Result<Attention> {
        let names = if vb.contains_tensor("to_q.weight") { ["to_q", "to_k", "to_v", "to_out.0"] } else { ["query", "key", "value", "proj_attn"] };
        let [q, k, v, out] = names.map(|n| linear(vb.pp(n), c));
        Ok(Attention { norm: Norm::new(vb.pp("group_norm"), c, groups)?, q: q?, k: k?, v: v?, out: out? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, h, w) = x.dims3()?;
        self.forward_in(x, (SCORE_BYTES / (h * w * 4).max(1)).max(1))
    }

    /// Runs `chunk` queries at a time.
    fn forward_in(&self, x: &Tensor, chunk: usize) -> Result<Tensor> {
        let (c, h, w) = x.dims3()?;
        let t = h * w;
        let n = self.norm.forward(&x.reshape((c, t))?)?;
        let project = |(wt, b): &(Tensor, Tensor)| wt.matmul(&n)?.broadcast_add(b)?.to_dtype(DType::F32);
        let scale = (c.max(1) as f64).powf(-0.5);
        let qt = (project(&self.q)? * scale)?.t()?.contiguous()?;
        let k = project(&self.k)?;
        let vt = project(&self.v)?.t()?.contiguous()?;
        let mut rows = Vec::new();
        for s in (0..t).step_by(chunk.max(1)) {
            let scores = qt.narrow(0, s, chunk.min(t - s))?.matmul(&k)?;
            rows.push(candle_nn::ops::softmax_last_dim(&scores)?.matmul(&vt)?);
            x.device().synchronize()?;
        }
        let o = Tensor::cat(&rows, 0)?.to_dtype(x.dtype())?;
        let (wo, bo) = &self.out;
        wo.matmul(&o.t()?)?.broadcast_add(bo)?.reshape((c, h, w))?.add(x)
    }
}

/// One block of an encoder or decoder (`[c, h, w]` in and out), run between two GPU waits.
enum Block {
    Resnet(Resnet),
    /// Pad right and bottom by one, then a stride-2 3×3 convolution.
    Down(Conv2d),
    /// Nearest-neighbour 2× upsampling, then a 3×3 convolution.
    Up(Conv3),
    Attention(Attention),
}

impl Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Block::Resnet(r) => r.forward(xs),
            Block::Down(conv) => conv.forward(&xs.pad_with_zeros(D::Minus1, 0, 1)?.pad_with_zeros(D::Minus2, 0, 1)?.unsqueeze(0)?)?.squeeze(0),
            Block::Up(conv) => {
                let (_, h, w) = xs.dims3()?;
                let up = xs.unsqueeze(0)?.upsample_nearest2d(2 * h, 2 * w)?.squeeze(0)?;
                xs.device().synchronize()?;
                conv.forward(&up)
            }
            Block::Attention(a) => a.forward(xs),
        }
    }
}

/// An encoder or a decoder: `conv_in`, the blocks, then norm, SiLU and `conv_out`.
struct Coder {
    conv_in: Conv3,
    blocks: Vec<Block>,
    norm_out: Norm,
    conv_out: Conv3,
}

impl Coder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let device = xs.device().clone();
        let mut xs = self.conv_in.forward(xs)?;
        device.synchronize()?;
        for block in &self.blocks {
            xs = block.forward(&xs)?;
            device.synchronize()?;
        }
        let (c, h, w) = xs.dims3()?;
        let ys = candle_nn::ops::silu(&self.norm_out.forward(&xs.reshape((c, h * w))?)?)?.reshape((c, h, w))?;
        self.conv_out.forward(&ys)
    }
}

/// The mid block: ResNet, attention, ResNet.
fn mid(blocks: &mut Vec<Block>, vb: VarBuilder, channels: usize, groups: usize) -> Result<()> {
    blocks.push(Block::Resnet(Resnet::new(vb.pp("resnets").pp("0"), channels, channels, groups)?));
    blocks.push(Block::Attention(Attention::new(vb.pp("attentions").pp("0"), channels, groups)?));
    blocks.push(Block::Resnet(Resnet::new(vb.pp("resnets").pp("1"), channels, channels, groups)?));
    Ok(())
}

fn encoder(vb: VarBuilder, cfg: &VaeConfig) -> Result<Coder> {
    let ch = &cfg.block_out_channels;
    let (first, last) = match (ch.first(), ch.last()) {
        (Some(&f), Some(&l)) => (f, l),
        _ => candle_core::bail!("the VAE has no blocks"),
    };
    let mut blocks = Vec::new();
    let mut cin = first;
    for (i, &cout) in ch.iter().enumerate() {
        let b = vb.pp("down_blocks").pp(i.to_string());
        for j in 0..cfg.layers_per_block {
            blocks.push(Block::Resnet(Resnet::new(b.pp("resnets").pp(j.to_string()), cin, cout, cfg.groups)?));
            cin = cout;
        }
        if i + 1 < ch.len() {
            let conv = candle_nn::conv2d(cout, cout, 3, Conv2dConfig { stride: 2, ..Default::default() }, b.pp("downsamplers").pp("0").pp("conv"))?;
            blocks.push(Block::Down(conv));
        }
    }
    mid(&mut blocks, vb.pp("mid_block"), last, cfg.groups)?;
    Ok(Coder {
        conv_in: Conv3::new(vb.pp("conv_in"), 3, first, None)?,
        blocks,
        norm_out: Norm::new(vb.pp("conv_norm_out"), last, cfg.groups)?,
        conv_out: Conv3::new(vb.pp("conv_out"), last, 2 * cfg.latent_channels, None)?,
    })
}

fn decoder(vb: VarBuilder, cfg: &VaeConfig) -> Result<Coder> {
    let ch: Vec<usize> = cfg.block_out_channels.iter().rev().copied().collect();
    let (first, last) = match (ch.first(), ch.last()) {
        (Some(&f), Some(&l)) => (f, l),
        _ => candle_core::bail!("the VAE has no blocks"),
    };
    let mut blocks = Vec::new();
    mid(&mut blocks, vb.pp("mid_block"), first, cfg.groups)?;
    let mut cin = first;
    for (i, &cout) in ch.iter().enumerate() {
        let b = vb.pp("up_blocks").pp(i.to_string());
        for j in 0..=cfg.layers_per_block {
            blocks.push(Block::Resnet(Resnet::new(b.pp("resnets").pp(j.to_string()), cin, cout, cfg.groups)?));
            cin = cout;
        }
        if i + 1 < ch.len() {
            blocks.push(Block::Up(Conv3::new(b.pp("upsamplers").pp("0").pp("conv"), cout, cout, None)?));
        }
    }
    Ok(Coder {
        conv_in: Conv3::new(vb.pp("conv_in"), cfg.latent_channels, first, None)?,
        blocks,
        norm_out: Norm::new(vb.pp("conv_norm_out"), last, cfg.groups)?,
        conv_out: Conv3::new(vb.pp("conv_out"), last, 3, None)?,
    })
}

/// The VAE, in the dtype of the `VarBuilder` it was built from.
pub(super) struct Vae {
    encoder: Coder,
    decoder: Coder,
    quant_conv: Conv2d,
    post_quant_conv: Conv2d,
    dtype: DType,
}

impl Vae {
    pub(super) fn new(vb: VarBuilder, cfg: &VaeConfig) -> Result<Vae> {
        let z = 2 * cfg.latent_channels;
        Ok(Vae {
            encoder: encoder(vb.pp("encoder"), cfg)?,
            decoder: decoder(vb.pp("decoder"), cfg)?,
            quant_conv: candle_nn::conv2d(z, z, 1, Conv2dConfig::default(), vb.pp("quant_conv"))?,
            post_quant_conv: candle_nn::conv2d(cfg.latent_channels, cfg.latent_channels, 1, Conv2dConfig::default(), vb.pp("post_quant_conv"))?,
            dtype: vb.dtype(),
        })
    }

    /// The latent distribution of an image (`[1, 3, h, w]`, −1…1): mean and log variance
    /// concatenated on the channel axis, in f32.
    pub(super) fn encode_params(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.encoder.forward(&single(xs)?.to_dtype(self.dtype)?)?;
        self.quant_conv.forward(&xs.unsqueeze(0)?)?.to_dtype(DType::F32)
    }

    /// Unscaled latents of an image: the distribution's mean plus `noise` (standard normal, the
    /// shape of the latents) times its deviation. Passing the noise in keeps a seed's latents the
    /// same on every device and across tiles.
    pub(super) fn encode(&self, xs: &Tensor, noise: &Tensor) -> Result<Tensor> {
        let params = self.encode_params(xs)?;
        let parts = params.chunk(2, 1)?;
        let (Some(mean), Some(logvar)) = (parts.first(), parts.get(1)) else { candle_core::bail!("the VAE encoder returned no channels") };
        let std = logvar.clamp(-30f32, 20f32)?.affine(0.5, 0.0)?.exp()?;
        mean.add(&std.mul(&noise.to_dtype(DType::F32)?)?)
    }

    /// Pixels (`[1, 3, h, w]`, about −1…1, f32) of unscaled latents.
    pub(super) fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let z = self.post_quant_conv.forward(&z.to_dtype(self.dtype)?)?;
        self.decoder.forward(&single(&z)?)?.unsqueeze(0)?.to_dtype(DType::F32)
    }
}

/// The one image of a batch of one, as `[c, h, w]`.
fn single(xs: &Tensor) -> Result<Tensor> {
    let (n, _, _, _) = xs.dims4()?;
    if n != 1 {
        candle_core::bail!("the VAE takes one image at a time, not {n}");
    }
    xs.squeeze(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;
    use candle_transformers::models::stable_diffusion::vae;

    fn tiny() -> VaeConfig {
        VaeConfig { block_out_channels: vec![32, 64], layers_per_block: 1, latent_channels: 4, groups: 8 }
    }

    fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
        a.sub(b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_vec0::<f32>().unwrap()
    }

    /// Matches candle's AutoEncoderKL on the same (random) weights, so the weight names, the
    /// block order and the padding are all the same.
    #[test]
    fn matches_candles_autoencoder() {
        let dev = Device::Cpu;
        let cfg = tiny();
        let vm = VarMap::new();
        let theirs = vae::AutoEncoderKL::new(
            VarBuilder::from_varmap(&vm, DType::F32, &dev),
            3,
            3,
            vae::AutoEncoderKLConfig {
                block_out_channels: cfg.block_out_channels.clone(),
                layers_per_block: cfg.layers_per_block,
                latent_channels: cfg.latent_channels,
                norm_num_groups: cfg.groups,
                use_quant_conv: true,
                use_post_quant_conv: true,
            },
        )
        .unwrap();
        // A near-zero deviation, so candle's randomly sampled latents are its mean.
        let mut w = Tensor::randn(0f32, 0.1, (8, 8, 1, 1), &dev).unwrap();
        w = Tensor::cat(&[&w.narrow(0, 0, 4).unwrap(), &Tensor::zeros((4, 8, 1, 1), DType::F32, &dev).unwrap()], 0).unwrap();
        let mut vm = vm;
        vm.set_one("quant_conv.weight", &w).unwrap();
        vm.set_one("quant_conv.bias", Tensor::new(&[0.1f32, -0.2, 0.3, 0.0, -28.0, -28.0, -28.0, -28.0], &dev).unwrap()).unwrap();
        let ours = Vae::new(VarBuilder::from_varmap(&vm, DType::F32, &dev), &cfg).unwrap();

        let img = Tensor::randn(0f32, 0.5, (1, 3, 32, 48), &dev).unwrap();
        let zero = Tensor::zeros((1, 4, 16, 24), DType::F32, &dev).unwrap();
        let a = ours.encode(&img, &zero).unwrap();
        let b = theirs.encode(&img).unwrap().sample().unwrap();
        assert_eq!(a.dims(), b.dims());
        assert!(max_diff(&a, &b) < 1e-4, "encode differs by {}", max_diff(&a, &b));

        let z = Tensor::randn(0f32, 1.0, (1, 4, 8, 12), &dev).unwrap();
        let a = ours.decode(&z).unwrap();
        let b = theirs.decode(&z).unwrap();
        assert_eq!(a.dims(), &[1, 3, 16, 24]);
        assert!(max_diff(&a, &b) < 1e-4, "decode differs by {}", max_diff(&a, &b));
    }

    #[test]
    fn noise_moves_latents_by_the_deviation_and_f16_follows_f32() {
        let dev = Device::Cpu;
        let cfg = tiny();
        let vm = VarMap::new();
        let v32 = Vae::new(VarBuilder::from_varmap(&vm, DType::F32, &dev), &cfg).unwrap();
        let weights: std::collections::HashMap<String, Tensor> = vm.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.as_tensor().clone())).collect();
        let v16 = Vae::new(VarBuilder::from_tensors(weights, DType::F16, &dev), &cfg).unwrap();
        assert_eq!((v32.dtype, v16.dtype), (DType::F32, DType::F16));
        let img = Tensor::randn(0f32, 0.5, (1, 3, 32, 32), &dev).unwrap();
        let zero = Tensor::zeros((1, 4, 16, 16), DType::F32, &dev).unwrap();
        let one = Tensor::ones((1, 4, 16, 16), DType::F32, &dev).unwrap();
        let params = v32.encode_params(&img).unwrap();
        let std = params.narrow(1, 4, 4).unwrap().affine(0.5, 0.0).unwrap().exp().unwrap();
        let moved = v32.encode(&img, &one).unwrap().sub(&v32.encode(&img, &zero).unwrap()).unwrap();
        assert!(max_diff(&moved, &std) < 1e-4);
        let z = Tensor::randn(0f32, 1.0, (1, 4, 8, 8), &dev).unwrap();
        let (a, b) = (v32.decode(&z).unwrap(), v16.decode(&z).unwrap());
        assert_eq!(b.dtype(), DType::F32);
        let scale = a.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_vec0::<f32>().unwrap();
        assert!(max_diff(&a, &b) < 0.02 * scale.max(1.0), "f16 drifted {} (scale {scale})", max_diff(&a, &b));
    }

    /// Convolutions in bands of rows and norms in chunks of groups match candle's, whole.
    #[test]
    fn bands_and_chunks_match_candles_layers() {
        let dev = Device::Cpu;
        let mut vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let theirs = candle_nn::conv2d(16, 8, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("c")).unwrap();
        let ours = Conv3::new(vb.pp("c"), 16, 8, None).unwrap();
        let x = Tensor::randn(0f32, 1.0, (16, 13, 21), &dev).unwrap();
        let want = theirs.forward(&x.unsqueeze(0).unwrap()).unwrap().squeeze(0).unwrap();
        for band in [1, 4, 13, 100] {
            for pad_whole in [true, false] {
                assert!(max_diff(&ours.forward_in(&x, band, pad_whole).unwrap(), &want) < 1e-4, "band {band}, {pad_whole}");
            }
        }
        assert!(max_diff(&ours.forward(&x).unwrap(), &want) < 1e-4);
        for pad_whole in [true, false] {
            assert!(ours.forward_in(&x.narrow(1, 0, 0).unwrap(), 4, pad_whole).is_err(), "no rows is an error, not a panic");
        }

        let theirs = candle_nn::group_norm(4, 16, EPS, vb.pp("n")).unwrap();
        vm.set_one("n.weight", Tensor::randn(1f32, 0.2, 16, &dev).unwrap()).unwrap();
        vm.set_one("n.bias", Tensor::randn(0f32, 0.2, 16, &dev).unwrap()).unwrap();
        let ours = Norm::new(vb.pp("n"), 16, 4).unwrap();
        let x = (Tensor::randn(0f32, 2.0, (16, 13 * 21), &dev).unwrap() + 3.0).unwrap();
        let want = theirs.forward(&x.unsqueeze(0).unwrap()).unwrap().squeeze(0).unwrap();
        for groups in [1, 3, 4] {
            assert!(max_diff(&ours.forward_in(&x, groups).unwrap(), &want) < 1e-4, "{groups} groups at once");
        }
        assert!(Norm::new(vb.pp("n"), 16, 5).is_err(), "16 channels don't split into 5 groups");

        use candle_transformers::models::stable_diffusion::attention::{AttentionBlock, AttentionBlockConfig};
        let cfg = AttentionBlockConfig { num_head_channels: None, num_groups: 4, rescale_output_factor: 1., eps: EPS };
        let theirs = AttentionBlock::new(vb.pp("a"), 16, cfg).unwrap();
        let ours = Attention::new(vb.pp("a"), 16, 4).unwrap();
        let x = Tensor::randn(0f32, 1.0, (16, 5, 7), &dev).unwrap();
        let want = theirs.forward(&x.unsqueeze(0).unwrap()).unwrap().squeeze(0).unwrap();
        for chunk in [1, 8, 35, 100] {
            assert!(max_diff(&ours.forward_in(&x, chunk).unwrap(), &want) < 1e-4, "{chunk} queries at once");
        }
    }

    #[test]
    fn missing_or_misshapen_weights_are_errors() {
        let dev = Device::Cpu;
        assert!(Vae::new(VarBuilder::from_tensors(Default::default(), DType::F32, &dev), &VaeConfig::sd()).is_err());
        let empty = VaeConfig { block_out_channels: vec![], ..tiny() };
        assert!(Vae::new(VarBuilder::from_varmap(&VarMap::new(), DType::F32, &dev), &empty).is_err());
    }
}

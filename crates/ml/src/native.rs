//! The built-in Stable Diffusion engine: candle on Metal (Apple silicon only).
//!
//! Runs SD 1.5 (and its 9-channel inpainting variant) and SD-Turbo (the SD 2.1 architecture)
//! from single-file `.safetensors` checkpoints, with LoRAs merged at load time. The loaded model
//! stays warm between generations; a different checkpoint, VAE or LoRA set reloads or re-merges
//! only what changed. On other machines every call returns an "unsupported" error.
//!
//! - text-to-image when there is no init image;
//! - image-to-image at `strength` (the frame's pixels, VAE-encoded and noised);
//! - inpainting: a 9-channel inpainting UNet gets the mask and the masked image's latent; any
//!   other checkpoint resets the unmasked latents each step from the noised original.

use std::path::{Path, PathBuf};

use crate::catalog::{Family, ModelKind};
use crate::params::{GenerateOutput, GenerateRequest, MAX_NATIVE_LORAS};
use crate::store::ModelStore;
use crate::{MlError, Progress, Result};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod imp;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod vae;

/// Largest frame side the built-in engine generates (SD 1.5 and SD-Turbo degrade well before).
pub const MAX_NATIVE_SIDE: u32 = 1024;

/// The tokenizer file the native engine looks for first (the catalog's `clip-tokenizer`).
pub const TOKENIZER_FILE: &str = "clip-vit-large-patch14.json";

/// Everything a generation needs, resolved to files.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedModels {
    pub checkpoint: PathBuf,
    pub family: Family,
    pub vae: Option<PathBuf>,
    pub tokenizer: PathBuf,
    pub loras: Vec<(PathBuf, f32)>,
}

/// Find the request's files in the store and check they fit together.
pub fn resolve(store: &ModelStore, req: &GenerateRequest) -> Result<ResolvedModels> {
    let checkpoint = store.existing(ModelKind::Checkpoint, &req.checkpoint)?;
    let family = crate::store::detect(&checkpoint, Some(ModelKind::Checkpoint))?.family;
    let vae = req.vae.as_deref().filter(|v| !v.trim().is_empty()).map(|v| store.existing(ModelKind::Vae, v)).transpose()?;
    if req.loras.len() > MAX_NATIVE_LORAS {
        return Err(MlError::BadParams(format!("the built-in engine stacks at most {MAX_NATIVE_LORAS} LoRAs; use the ComfyUI engine for more")));
    }
    let mut loras = Vec::new();
    for l in req.loras.iter().filter(|l| l.weight != 0.0) {
        let p = store.existing(ModelKind::Lora, &l.file)?;
        let lf = crate::store::detect(&p, Some(ModelKind::Lora))?.family;
        if !lf.compatible_with(family) {
            return Err(MlError::BadModel(format!("{} is a {} LoRA; it doesn't work with {} ({})", l.file, lf.label(), req.checkpoint, family.label())));
        }
        loras.push((p, l.weight));
    }
    let tokenizer = tokenizer_path(store)?;
    Ok(ResolvedModels { checkpoint, family, vae, tokenizer, loras })
}

/// The CLIP tokenizer in the store's `tokenizers` folder.
pub fn tokenizer_path(store: &ModelStore) -> Result<PathBuf> {
    if let Ok(p) = store.existing(ModelKind::Tokenizer, TOKENIZER_FILE) {
        return Ok(p);
    }
    let folder = store.folder(ModelKind::Tokenizer);
    let mut others: Vec<PathBuf> = std::fs::read_dir(&folder)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "json") && p.file_name().and_then(|n| n.to_str()).is_some_and(crate::store::is_plain_file_name))
        .collect();
    others.sort();
    others
        .into_iter()
        .next()
        .ok_or_else(|| MlError::NotFound("the CLIP tokenizer (2 MB) isn't installed; download it from Model Manager › Suggested › Support files".into()))
}

/// Check what only the native engine limits, before anything loads.
pub fn check_request(req: &GenerateRequest) -> Result<()> {
    req.validate()?;
    if req.width > MAX_NATIVE_SIDE || req.height > MAX_NATIVE_SIDE {
        return Err(MlError::BadParams(format!(
            "the built-in engine generates up to {MAX_NATIVE_SIDE}×{MAX_NATIVE_SIDE}; make the frame smaller or use the ComfyUI engine"
        )));
    }
    if let Some(reason) = crate::system::native_unavailable_reason() {
        return Err(MlError::Unsupported(reason.into()));
    }
    Ok(())
}

/// The engine with its warm model.
#[derive(Default)]
pub struct NativeEngine {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    loaded: Option<imp::Loaded>,
}

impl std::fmt::Debug for NativeEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeEngine").field("loaded", &self.loaded_checkpoint()).finish()
    }
}

impl NativeEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// The checkpoint kept in memory, if any.
    pub fn loaded_checkpoint(&self) -> Option<&Path> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.loaded.as_ref().map(|l| l.checkpoint())
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            None
        }
    }

    /// Free the model's memory.
    pub fn unload(&mut self) {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            self.loaded = None;
        }
    }

    /// Run one generation. Loads (or re-merges) the model first when the request needs it.
    pub fn generate(&mut self, store: &ModelStore, req: &GenerateRequest, progress: &dyn Progress) -> Result<GenerateOutput> {
        check_request(req)?;
        let models = resolve(store, req)?;
        self.generate_resolved(&models, req, progress)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn generate_resolved(&mut self, models: &ResolvedModels, req: &GenerateRequest, progress: &dyn Progress) -> Result<GenerateOutput> {
        let started = std::time::Instant::now();
        let reuse = self.loaded.as_ref().is_some_and(|l| l.matches(models));
        if !reuse {
            // Free the old model before loading the next one.
            self.loaded = None;
            self.loaded = Some(imp::Loaded::load(models, progress)?);
        }
        let loaded = self.loaded.as_mut().ok_or_else(|| MlError::Other("the model failed to load".into()))?;
        loaded.set_loras(&models.loras, progress)?;
        let image = loaded.generate(req, progress)?;
        Ok(GenerateOutput { image, seed: req.seed, elapsed_ms: started.elapsed().as_millis() as u64 })
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    fn generate_resolved(&mut self, _models: &ResolvedModels, _req: &GenerateRequest, _progress: &dyn Progress) -> Result<GenerateOutput> {
        Err(MlError::Unsupported(crate::system::native_unavailable_reason().unwrap_or("the built-in engine needs Apple silicon").into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{LoraRef, Sampler};

    fn req(ckpt: &str) -> GenerateRequest {
        GenerateRequest {
            checkpoint: ckpt.into(),
            vae: None,
            loras: vec![],
            prompt: "a lighthouse".into(),
            negative: String::new(),
            steps: 4,
            cfg: 7.0,
            seed: 1,
            sampler: Sampler::Euler,
            strength: 1.0,
            width: 512,
            height: 512,
            init: None,
            mask: None,
        }
    }

    /// The last generation's stage timings, for the real-model tests' output.
    fn stages(e: &NativeEngine) -> String {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            format!("{:?}", e.loaded.as_ref().map(|l| l.timings()))
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = e;
            String::new()
        }
    }

    #[test]
    fn missing_files_and_limits_are_errors() {
        let dir = std::env::temp_dir().join(format!("photoka-native-{}", std::process::id()));
        let store = ModelStore::new(&dir);
        let mut e = NativeEngine::new();
        assert!(e.generate(&store, &req("nope.safetensors"), &crate::NoProgress).is_err());
        let mut big = req("x.safetensors");
        big.width = 1536;
        assert!(check_request(&big).unwrap_err().to_string().contains("1024"));
        let mut many = req("x.safetensors");
        many.loras = (0..5).map(|i| LoraRef { file: format!("l{i}.safetensors"), weight: 1.0 }).collect();
        assert!(e.generate(&store, &many, &crate::NoProgress).is_err());
        assert!(tokenizer_path(&store).unwrap_err().to_string().contains("tokenizer"));
        assert!(e.loaded_checkpoint().is_none());
    }

    /// Real SD 1.5 end to end. Needs `PHOTOKA_MODELS_DIR` with the SD 1.5 checkpoint and the CLIP
    /// tokenizer; writes `target/native-e2e-*.png`.
    #[test]
    #[ignore = "needs real model files (PHOTOKA_MODELS_DIR)"]
    fn real_sd15_end_to_end() {
        let dir = std::env::var_os("PHOTOKA_MODELS_DIR").expect("set PHOTOKA_MODELS_DIR");
        let store = ModelStore::new(PathBuf::from(dir));
        let mut e = NativeEngine::new();
        struct Print;
        impl Progress for Print {
            fn report(&self, f: f32, m: &str) {
                eprintln!("{:5.1}% {m}", f * 100.0);
            }
            fn cancelled(&self) -> bool {
                false
            }
        }
        let mut r = req("v1-5-pruned-emaonly-fp16.safetensors");
        r.prompt = "a watercolor painting of a lighthouse on a cliff at sunset".into();
        r.negative = "blurry, low quality".into();
        r.steps = 20;
        r.sampler = Sampler::Dpmpp2mKarras;
        let out = e.generate(&store, &r, &Print).unwrap();
        eprintln!("txt2img: {} ms {}", out.elapsed_ms, stages(&e));
        let target = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target");
        std::fs::write(target.join("native-e2e-txt2img.png"), out.image.to_png().unwrap()).unwrap();
        // Inpaint the left half of that image with latent blending.
        let mut mask = crate::image::Mask::new(512, 512).unwrap();
        for y in 0..512usize {
            for x in 0..256usize {
                mask.data[y * 512 + x] = 255;
            }
        }
        let mut r2 = r.clone();
        r2.prompt = "a red sailing boat on the sea".into();
        r2.init = Some(out.image.clone());
        r2.mask = Some(mask);
        let out2 = e.generate(&store, &r2, &Print).unwrap();
        eprintln!("inpaint: {} ms {}", out2.elapsed_ms, stages(&e));
        std::fs::write(target.join("native-e2e-inpaint.png"), out2.image.to_png().unwrap()).unwrap();
        // LCM-LoRA, 6 steps.
        if store.is_installed(ModelKind::Lora, "lcm-lora-sdv1-5.safetensors") {
            let mut r3 = r.clone();
            r3.loras = vec![LoraRef { file: "lcm-lora-sdv1-5.safetensors".into(), weight: 1.0 }];
            r3.sampler = Sampler::Lcm;
            r3.steps = 6;
            r3.cfg = 1.5;
            let out3 = e.generate(&store, &r3, &Print).unwrap();
            eprintln!("lcm: {} ms {}", out3.elapsed_ms, stages(&e));
            std::fs::write(target.join("native-e2e-lcm.png"), out3.image.to_png().unwrap()).unwrap();
        }
    }

    /// SD-Turbo (SD 2.1 architecture, OpenCLIP-H text encoder): 2 steps, no CFG, trailing
    /// timesteps. Writes `target/native-e2e-turbo.png`.
    #[test]
    #[ignore = "needs real model files (PHOTOKA_MODELS_DIR with sd_turbo.safetensors)"]
    fn real_sd_turbo_end_to_end() {
        let dir = std::env::var_os("PHOTOKA_MODELS_DIR").expect("set PHOTOKA_MODELS_DIR");
        let store = ModelStore::new(PathBuf::from(dir));
        let mut e = NativeEngine::new();
        let mut r = req("sd_turbo.safetensors");
        r.prompt = "a photo of a red fox in a snowy forest, detailed".into();
        r.steps = 2;
        r.cfg = 0.0;
        r.sampler = Sampler::EulerA;
        let out = e.generate(&store, &r, &crate::NoProgress).unwrap();
        eprintln!("turbo: {} ms {}", out.elapsed_ms, stages(&e));
        let target = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target");
        std::fs::write(target.join("native-e2e-turbo.png"), out.image.to_png().unwrap()).unwrap();
        let mean = out.image.data.as_chunks::<4>().0.iter().map(|p| f64::from(p[0]) + f64::from(p[1]) + f64::from(p[2])).sum::<f64>() / (512.0 * 512.0 * 3.0);
        assert!(mean > 10.0 && mean < 245.0, "a blank image (mean {mean})");
        let again = e.generate(&store, &r, &crate::NoProgress).unwrap();
        eprintln!("turbo, loaded: {} ms {}", again.elapsed_ms, stages(&e));
    }

    /// Low strength keeps the frame, with and without a mask; full strength replaces it.
    /// Writes `target/native-e2e-strength-*.png`.
    #[test]
    #[ignore = "needs real model files (PHOTOKA_MODELS_DIR with sd_turbo.safetensors)"]
    fn real_low_strength_keeps_the_frame() {
        let dir = std::env::var_os("PHOTOKA_MODELS_DIR").expect("set PHOTOKA_MODELS_DIR");
        let store = ModelStore::new(PathBuf::from(dir));
        let mut e = NativeEngine::new();
        let mut r = req("sd_turbo.safetensors");
        r.prompt = "a photo of a red fox in a snowy forest, detailed".into();
        r.steps = 2;
        r.cfg = 0.0;
        r.sampler = Sampler::EulerA;
        let base = e.generate(&store, &r, &crate::NoProgress).unwrap().image;
        let target = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target");
        let mut left = crate::image::Mask::new(512, 512).unwrap();
        for y in 0..512usize {
            left.data[y * 512..y * 512 + 256].fill(255);
        }
        // Mean absolute RGB difference from `base` over the left half.
        let diff = |img: &crate::image::Rgba8| {
            let (a, b) = (base.data.as_chunks::<4>().0, img.data.as_chunks::<4>().0);
            let mut sum = 0.0;
            for (i, (p, q)) in a.iter().zip(b).enumerate() {
                if i % 512 < 256 {
                    sum += (0..3).map(|c| (f64::from(p[c]) - f64::from(q[c])).abs()).sum::<f64>();
                }
            }
            sum / (512.0 * 256.0 * 3.0)
        };
        let mut r2 = r.clone();
        r2.prompt = "an oil painting of a blue whale".into();
        r2.seed = 7;
        r2.init = Some(base.clone());
        let mut results = Vec::new();
        for (name, strength, mask) in [("img2img-19", 0.19, None), ("inpaint-19", 0.19, Some(left.clone())), ("inpaint-100", 1.0, Some(left.clone()))] {
            r2.strength = strength;
            r2.mask = mask;
            let out = e.generate(&store, &r2, &crate::NoProgress).unwrap().image;
            std::fs::write(target.join(format!("native-e2e-strength-{name}.png")), out.to_png().unwrap()).unwrap();
            let d = diff(&out);
            eprintln!("{name}: mean difference {d:.1}");
            results.push(d);
        }
        assert!(results[0] < 20.0, "19% image to image drifted too far: {}", results[0]);
        assert!(results[1] < 20.0, "19% inpainting drifted too far: {}", results[1]);
        assert!(results[2] > results[1] * 2.0, "full strength should replace the masked area: {results:?}");
    }
}

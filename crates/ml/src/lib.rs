//! Offline image generation for Photoka (Draw Things-style generation frame).
//!
//! - [`catalog`]: the bundled list of suggested models (nothing is downloaded by default).
//! - [`store`]: the models folder, laid out like ComfyUI's (`checkpoints/`, `loras/`, `vae/`,
//!   `embeddings/`, plus `tokenizers/`), shared by both engines; import, detection and deletion.
//! - [`download`]: resumable, sha256-verified downloads from Hugging Face and Civitai.
//! - [`keymap`] and [`lora_keys`]: weight-name mapping from single-file (LDM) checkpoints and LoRA
//!   files onto the layout the native engine loads.
//! - [`sampling`]: noise schedules and sampler coefficients (pure math, shared by the engines).
//! - [`native`]: the built-in Stable Diffusion engine (candle on Metal, Apple silicon only).
//! - [`comfy`]: the ComfyUI backend Photoka installs, launches and drives over HTTP.
//!
//! The crate knows nothing about documents: images go in and out as RGBA8 buffers, and the
//! engine (`photoka-engine`, `ai_cmds`) turns results into layers.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod catalog;
pub mod comfy;
pub mod download;
mod error;
pub mod image;
pub mod keychain;
pub mod keymap;
pub mod lora_keys;
pub mod native;
pub mod params;
pub mod safetensors_header;
pub mod sampling;
pub mod store;
pub mod system;

pub use error::{MlError, Result};

/// Progress and cancellation for long operations (downloads, installs, generation). The engine
/// adapts its job context to this, so this crate has no engine dependency.
pub trait Progress: Sync {
    /// `fraction` in 0..=1 and a short status line.
    fn report(&self, fraction: f32, message: &str);
    /// True once the user cancelled; long loops check it and return [`MlError::Cancelled`].
    fn cancelled(&self) -> bool;
    /// An intermediate preview (RGBA8, `width`×`height`) while generating. Optional.
    fn preview(&self, _image: &image::Rgba8) {}
    /// [`MlError::Cancelled`] when cancelled.
    fn check(&self) -> Result<()> {
        if self.cancelled() { Err(MlError::Cancelled) } else { Ok(()) }
    }
}

/// A [`Progress`] that ignores reports and is never cancelled (tests, CLI).
pub struct NoProgress;

impl Progress for NoProgress {
    fn report(&self, _: f32, _: &str) {}
    fn cancelled(&self) -> bool {
        false
    }
}

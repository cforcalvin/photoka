//! Offline image generation: the generation frame, its inpaint mask and `ai.generate`
//! (Photoka extension, Draw Things-style).
//!
//! Each document has a frame (512×512 by default, centred; sides snap to 64 px) and an 8-bit
//! inpaint mask in document coordinates. Both are session state, not document data: nothing is
//! saved in files and changing them isn't an undo step.
//!
//! `ai.generate` reads the composite under the frame. An empty frame is text-to-image; one with
//! pixels is image-to-image at `strength`; a mask (painted, or the frame's transparent pixels
//! for outpainting) inpaints. The result lands as a new layer at the frame, with a layer mask
//! from the feathered inpaint mask, in one undo step. The model stays loaded between runs.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use photoka_color::PixelFormat;
use photoka_doc::{DocId, Document, Layer, LayerMask};
use photoka_geom::Rect;
use photoka_ml::image::{Mask, Rgba8};
use photoka_ml::native::NativeEngine;
use photoka_ml::params::{EngineKind, GenerateRequest, LoraRef, MAX_LORAS, Sampler};
use photoka_raster::Surface;
use serde_json::{Value, json};

use crate::ai_cmds::{bad, ml_err};
use crate::commands::CommandSpec;
use crate::jobs::JobCtx;
use crate::{EngineError, Result, Session};

/// The default frame side (SD 1.5's native resolution).
pub const DEFAULT_SIDE: u32 = 512;
/// Frame sides snap to this (the latent is 1/8 of the frame, and UNets halve it three times).
pub const SNAP: u32 = 64;
pub const MIN_SIDE: u32 = 64;
pub const MAX_SIDE: u32 = photoka_ml::image::MAX_SIDE;
/// Coordinates further out than this are rejected.
const MAX_COORD: f64 = 1_000_000.0;
/// Largest mask plane, in pixels.
const MAX_MASK_PIXELS: u64 = 64 << 20;
const MAX_BRUSH: f64 = 2000.0;
const MAX_POINTS: usize = 100_000;
/// How far the result's layer mask feathers past the inpaint mask, in pixels.
const FEATHER: u32 = 12;

/// The inpaint mask: 255 = regenerate, in document coordinates over `rect`.
#[derive(Clone, Debug, PartialEq)]
pub struct MaskPlane {
    rect: Rect,
    data: Vec<u8>,
}

impl MaskPlane {
    pub fn rect(&self) -> Rect {
        self.rect
    }

    /// Row-major values over [`Self::rect`].
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn value(&self, x: i32, y: i32) -> u8 {
        if !self.rect.contains(x, y) {
            return 0;
        }
        let i = (y - self.rect.y0) as usize * self.rect.width() as usize + (x - self.rect.x0) as usize;
        self.data.get(i).copied().unwrap_or(0)
    }

    fn blank(rect: Rect) -> Result<MaskPlane> {
        let n = u64::from(rect.width()) * u64::from(rect.height());
        if n > MAX_MASK_PIXELS {
            return Err(EngineError::Other(format!("the mask would cover {n} pixels (at most {MAX_MASK_PIXELS})")));
        }
        Ok(MaskPlane { rect, data: vec![0; n as usize] })
    }

    /// A copy grown to cover `r` too.
    fn grown(&self, r: Rect) -> Result<MaskPlane> {
        let want = self.rect.union(&r);
        if want == self.rect {
            return Ok(self.clone());
        }
        let mut out = MaskPlane::blank(want)?;
        let (w, ow) = (self.rect.width() as usize, want.width() as usize);
        for (row, src) in self.data.chunks_exact(w.max(1)).enumerate() {
            let y = self.rect.y0 + row as i32;
            let start = (y - want.y0) as usize * ow + (self.rect.x0 - want.x0) as usize;
            if let Some(dst) = out.data.get_mut(start..start + w) {
                dst.copy_from_slice(src);
            }
        }
        Ok(out)
    }

    fn get_mut(&mut self, x: i32, y: i32) -> Option<&mut u8> {
        if !self.rect.contains(x, y) {
            return None;
        }
        let i = (y - self.rect.y0) as usize * self.rect.width() as usize + (x - self.rect.x0) as usize;
        self.data.get_mut(i)
    }

    /// The mask over `frame` (frame-sized; `None` when nothing there is masked).
    fn crop(&self, frame: Rect) -> Option<Mask> {
        let (w, h) = (frame.width(), frame.height());
        let mut data = vec![0u8; w as usize * h as usize];
        let mut any = false;
        let inside = self.rect.intersect(&frame);
        for y in inside.y0..inside.y1 {
            for x in inside.x0..inside.x1 {
                let v = self.value(x, y);
                if v > 0
                    && let Some(d) = data.get_mut((y - frame.y0) as usize * w as usize + (x - frame.x0) as usize)
                {
                    *d = v;
                    any = true;
                }
            }
        }
        if !any {
            return None;
        }
        Mask::from_raw(w, h, data).ok()
    }
}

/// The frame and mask of one document.
#[derive(Clone, Debug, Default)]
struct DocGen {
    frame: Option<Rect>,
    mask: Option<MaskPlane>,
}

/// The latest preview of a running generation.
#[derive(Clone, Debug)]
pub struct Preview {
    pub doc: DocId,
    pub frame: Rect,
    pub image: Rgba8,
    /// Increments with every new preview (for UI texture caches).
    pub serial: u64,
}

/// Frames, masks and the warm native model.
#[derive(Debug, Default)]
pub struct GenState {
    docs: HashMap<DocId, DocGen>,
    native: Arc<Mutex<NativeEngine>>,
    preview: Arc<Mutex<Option<Preview>>>,
    /// Increments whenever a frame or mask changes (for UI caches).
    revision: u64,
}

impl GenState {
    /// Drop the loaded model if it was loaded from `path` (it's being deleted).
    pub fn forget_model(&mut self, path: &Path) {
        if let Ok(mut e) = self.native.try_lock()
            && e.loaded_checkpoint() == Some(path)
        {
            e.unload();
        }
    }

    /// Drop the loaded model (the models folder changed, or memory is needed). A running
    /// generation keeps its model until it finishes.
    pub fn unload(&mut self) {
        if let Ok(mut e) = self.native.try_lock() {
            e.unload();
        }
    }

    /// Whether a native generation is running.
    pub fn busy(&self) -> bool {
        self.native.try_lock().is_err()
    }

    /// The checkpoint the native engine keeps loaded.
    pub fn loaded_model(&self) -> Option<String> {
        let e = self.native.try_lock().ok()?;
        e.loaded_checkpoint().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned())
    }

    /// The document's frame (the default when none was set).
    pub fn frame_for(&self, doc: &Document) -> Rect {
        self.docs.get(&doc.id).and_then(|g| g.frame).unwrap_or_else(|| default_frame(doc))
    }

    pub fn mask_for(&self, doc: DocId) -> Option<&MaskPlane> {
        self.docs.get(&doc).and_then(|g| g.mask.as_ref())
    }

    pub fn preview(&self) -> Option<Preview> {
        self.preview.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    fn doc_mut(&mut self, id: DocId) -> &mut DocGen {
        self.revision = self.revision.wrapping_add(1);
        self.docs.entry(id).or_default()
    }
}

/// 512×512 (or smaller, for a small canvas) centred on the canvas.
fn default_frame(doc: &Document) -> Rect {
    let b = doc.bounds();
    let side = |v: u32| (v / SNAP * SNAP).clamp(MIN_SIDE, DEFAULT_SIDE);
    let (w, h) = (side(b.width()), side(b.height()));
    let x = b.x0 + (b.width() as i32 - w as i32) / 2;
    let y = b.y0 + (b.height() as i32 - h as i32) / 2;
    Rect::from_xywh(x, y, w, h)
}

fn snap_side(v: f64) -> u32 {
    let snapped = (v / f64::from(SNAP)).round() * f64::from(SNAP);
    snapped.clamp(f64::from(MIN_SIDE), f64::from(MAX_SIDE)) as u32
}

fn num(cmd: &str, p: &Value, key: &str) -> Result<Option<f64>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => {
            let f = v.as_f64().filter(|f| f.is_finite()).ok_or_else(|| bad(cmd, format!("`{key}` must be a number")))?;
            if f.abs() > MAX_COORD {
                return Err(bad(cmd, format!("`{key}` is out of range")));
            }
            Ok(Some(f))
        }
    }
}

fn active_doc(s: &Session) -> Result<Arc<Document>> {
    s.active().map(|d| d.doc.clone()).ok_or(EngineError::NoDocument)
}

fn frame_json(s: &Session, doc: &Document) -> Value {
    let f = s.ai.generate.frame_for(doc);
    let mask = s.ai.generate.mask_for(doc.id).and_then(|m| m.crop(f));
    let coverage = mask.as_ref().map_or(0.0, |m| m.data.iter().filter(|&&v| v >= 128).count() as f64 / m.data.len().max(1) as f64);
    json!({"x": f.x0, "y": f.y0, "width": f.width(), "height": f.height(), "masked": mask.is_some(), "maskCoverage": coverage})
}

fn frame_get(s: &mut Session, _: &Value) -> Result<Value> {
    let doc = active_doc(s)?;
    Ok(frame_json(s, &doc))
}

fn frame_set(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ai.frame.set";
    let doc = active_doc(s)?;
    let cur = s.ai.generate.frame_for(&doc);
    let x = num(CMD, p, "x")?.map_or(cur.x0, |v| v.round() as i32);
    let y = num(CMD, p, "y")?.map_or(cur.y0, |v| v.round() as i32);
    let w = num(CMD, p, "width")?.map_or(cur.width(), snap_side);
    let h = num(CMD, p, "height")?.map_or(cur.height(), snap_side);
    s.ai.generate.doc_mut(doc.id).frame = Some(Rect::from_xywh(x, y, w, h));
    Ok(frame_json(s, &doc))
}

fn frame_reset(s: &mut Session, _: &Value) -> Result<Value> {
    let doc = active_doc(s)?;
    s.ai.generate.doc_mut(doc.id).frame = None;
    Ok(frame_json(s, &doc))
}

fn parse_points(cmd: &str, p: &Value) -> Result<Vec<(f64, f64)>> {
    let list = p.get("points").and_then(Value::as_array).ok_or_else(|| bad(cmd, "`points` (a list of [x, y]) is required"))?;
    if list.is_empty() || list.len() > MAX_POINTS {
        return Err(bad(cmd, format!("`points` must have 1…{MAX_POINTS} points")));
    }
    let coord = |v: Option<&Value>| v.and_then(Value::as_f64).filter(|f| f.is_finite() && f.abs() <= MAX_COORD);
    list.iter()
        .map(|pt| {
            let (x, y) = match pt {
                Value::Array(a) => (coord(a.first()), coord(a.get(1))),
                Value::Object(_) => (coord(pt.get("x")), coord(pt.get("y"))),
                _ => (None, None),
            };
            x.zip(y).ok_or_else(|| bad(cmd, "each point is [x, y] or {\"x\", \"y\"} with finite coordinates"))
        })
        .collect()
}

/// Paint (or erase) a soft round brush along `points`.
fn mask_stroke(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ai.mask.stroke";
    let doc = active_doc(s)?;
    let points = parse_points(CMD, p)?;
    let size = num(CMD, p, "size")?.unwrap_or(48.0);
    if !(1.0..=MAX_BRUSH).contains(&size) {
        return Err(bad(CMD, format!("`size` must be 1…{MAX_BRUSH}")));
    }
    let hardness = num(CMD, p, "hardness")?.unwrap_or(0.8).clamp(0.0, 1.0);
    let erase = p.get("erase").and_then(Value::as_bool).unwrap_or(false);
    let r = size / 2.0;
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for &(x, y) in &points {
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
    }
    let bounds = Rect::new((x0 - r).floor() as i32, (y0 - r).floor() as i32, (x1 + r).ceil() as i32 + 1, (y1 + r).ceil() as i32 + 1);
    // Masks only matter where a frame can go: the canvas and the frame (which may stick out).
    let frame = s.ai.generate.frame_for(&doc);
    let limit = doc.bounds().union(&frame).inflate(MAX_SIDE as i32);
    let area = bounds.intersect(&limit);
    if area.is_empty() {
        return Ok(frame_json(s, &doc));
    }
    let spacing = (size * 0.15).max(0.5);
    let length: f64 = points.windows(2).map(|w| ((w[1].0 - w[0].0).powi(2) + (w[1].1 - w[0].1).powi(2)).sqrt()).sum();
    let work = (length / spacing + points.len() as f64) * size * size;
    if work > 1e9 {
        return Err(bad(CMD, "the stroke is too long for its brush size; split it into shorter strokes"));
    }
    let g = s.ai.generate.doc_mut(doc.id);
    let mut plane = match &g.mask {
        Some(m) => m.grown(area)?,
        None => MaskPlane::blank(area)?,
    };
    let mut dab = |cx: f64, cy: f64| {
        let (ax0, ay0) = ((cx - r).floor() as i32, (cy - r).floor() as i32);
        let (ax1, ay1) = ((cx + r).ceil() as i32, (cy + r).ceil() as i32);
        for py in ay0.max(area.y0)..=ay1.min(area.y1 - 1) {
            for px in ax0.max(area.x0)..=ax1.min(area.x1 - 1) {
                let d = ((f64::from(px) + 0.5 - cx).powi(2) + (f64::from(py) + 0.5 - cy).powi(2)).sqrt() / r;
                if d >= 1.0 {
                    continue;
                }
                let inner = hardness.min(0.999);
                let a = if d <= inner { 1.0 } else { 1.0 - (d - inner) / (1.0 - inner) };
                let v = (a * 255.0).round().clamp(0.0, 255.0) as u8;
                if let Some(m) = plane.get_mut(px, py) {
                    *m = if erase { (*m).min(255 - v) } else { (*m).max(v) };
                }
            }
        }
    };
    let mut prev: Option<(f64, f64)> = None;
    for &(x, y) in &points {
        match prev {
            None => dab(x, y),
            Some((px, py)) => {
                let len = ((x - px).powi(2) + (y - py).powi(2)).sqrt();
                let n = (len / spacing).ceil().min(1e6) as usize;
                for i in 1..=n.max(1) {
                    let t = i as f64 / n.max(1) as f64;
                    dab(px + (x - px) * t, py + (y - py) * t);
                }
            }
        }
        prev = Some((x, y));
    }
    g.mask = (!plane.data.iter().all(|&v| v == 0)).then_some(plane);
    Ok(frame_json(s, &doc))
}

fn mask_fill(s: &mut Session, _: &Value) -> Result<Value> {
    let doc = active_doc(s)?;
    let frame = s.ai.generate.frame_for(&doc);
    let g = s.ai.generate.doc_mut(doc.id);
    let mut plane = match &g.mask {
        Some(m) => m.grown(frame)?,
        None => MaskPlane::blank(frame)?,
    };
    for y in frame.y0..frame.y1 {
        for x in frame.x0..frame.x1 {
            if let Some(m) = plane.get_mut(x, y) {
                *m = 255;
            }
        }
    }
    g.mask = Some(plane);
    Ok(frame_json(s, &doc))
}

fn mask_clear(s: &mut Session, _: &Value) -> Result<Value> {
    let doc = active_doc(s)?;
    s.ai.generate.doc_mut(doc.id).mask = None;
    Ok(frame_json(s, &doc))
}

/// Invert the mask inside the frame.
fn mask_invert(s: &mut Session, _: &Value) -> Result<Value> {
    let doc = active_doc(s)?;
    let frame = s.ai.generate.frame_for(&doc);
    let g = s.ai.generate.doc_mut(doc.id);
    let mut plane = match &g.mask {
        Some(m) => m.grown(frame)?,
        None => MaskPlane::blank(frame)?,
    };
    for y in frame.y0..frame.y1 {
        for x in frame.x0..frame.x1 {
            if let Some(m) = plane.get_mut(x, y) {
                *m = 255 - *m;
            }
        }
    }
    g.mask = (!plane.data.iter().all(|&v| v == 0)).then_some(plane);
    Ok(frame_json(s, &doc))
}

/// Whether the document's composite is linear light (32-bit documents): generation works on
/// sRGB-encoded pixels.
fn composite_is_linear(doc: &Document) -> bool {
    crate::display_color::is_linear_rgb(&crate::color_cmds::composite_profile(doc))
}

/// The composite under `frame` as 8-bit RGBA (transparent outside the canvas).
fn frame_pixels(doc: &Document, frame: Rect, linear: bool) -> Result<Rgba8> {
    let (w, h) = (frame.width(), frame.height());
    let mut out = vec![0u8; w as usize * h as usize * 4];
    let inside = frame.intersect(&doc.bounds());
    if !inside.is_empty() {
        let buf = photoka_compose::render(doc, inside);
        let iw = inside.width() as usize;
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        for (i, px) in buf.px.iter().enumerate() {
            let (x, y) = (inside.x0 + (i % iw.max(1)) as i32, inside.y0 + (i / iw.max(1)) as i32);
            let j = ((y - frame.y0) as usize * w as usize + (x - frame.x0) as usize) * 4;
            let rgb = px.map(|v| if linear { photoka_color::convert::linear_to_srgb(v) } else { v });
            if let Some(d) = out.get_mut(j..j + 4) {
                d.copy_from_slice(&[q(rgb[0]), q(rgb[1]), q(rgb[2]), q(px[3])]);
            }
        }
    }
    Rgba8::from_raw(w, h, out).map_err(ml_err)
}

/// The generated pixels as a new layer at `frame`, masked by the feathered inpaint mask.
fn result_layer(doc: &Document, name: String, frame: Rect, img: &Rgba8, mask: Option<&Mask>, linear: bool) -> Result<Layer> {
    if (img.width, img.height) != (frame.width(), frame.height()) {
        return Err(EngineError::Other("internal error: the result doesn't match the frame".into()));
    }
    let base = doc.pixel_format();
    let fmt = PixelFormat::new(base.mode, base.sample, true);
    let n = fmt.channels();
    let mut data = vec![0.0f32; img.data.len() / 4 * n];
    let mut enc = [0.0f32; 8];
    for (px, dst) in img.data.chunks_exact(4).zip(data.chunks_exact_mut(n)) {
        let c = |v: u8| {
            let f = f32::from(v) / 255.0;
            if linear { photoka_color::convert::srgb_to_linear(f) } else { f }
        };
        let k = photoka_raster::from_rgba_into(&fmt, [c(px[0]), c(px[1]), c(px[2]), 1.0], &mut enc);
        for (d, e) in dst.iter_mut().zip(enc.iter().take(k)) {
            *d = *e;
        }
    }
    let mut l = Layer::raster(name, fmt);
    crate::pixels_mut(&mut l)?.write_region(frame, &data);
    if let Some(m) = mask {
        let soft = m.dilated(FEATHER / 2).feathered(FEATHER);
        let mut surface = Surface::with_default(PixelFormat::GRAY8, &[0.0]);
        let vals: Vec<f32> = soft.data.iter().map(|&v| f32::from(v) / 255.0).collect();
        surface.write_region(frame, &vals);
        let format = surface.format().with_sample(doc.depth);
        if surface.format() != format {
            surface = surface.try_convert(format)?;
        }
        l.mask = Some(LayerMask { surface, ..LayerMask::reveal_all() });
    }
    Ok(l)
}

/// `ai.generate` params → a request (without the frame's pixels).
fn parse_request(p: &Value, frame: Rect) -> Result<GenerateRequest> {
    const CMD: &str = "ai.generate";
    let checkpoint = p
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .ok_or_else(|| bad(CMD, "`model` (an installed checkpoint's file name) is required; ai.models.list lists them"))?
        .to_string();
    let text = |k: &str| -> Result<String> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(String::new()),
            Some(Value::String(t)) => Ok(t.clone()),
            Some(_) => Err(bad(CMD, format!("`{k}` must be a string"))),
        }
    };
    let int = |k: &str, d: u64| -> Result<u64> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(d),
            Some(v) => v.as_u64().ok_or_else(|| bad(CMD, format!("`{k}` must be a whole number ≥ 0"))),
        }
    };
    let float = |k: &str, d: f64| -> Result<f64> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(d),
            Some(v) => v.as_f64().filter(|f| f.is_finite()).ok_or_else(|| bad(CMD, format!("`{k}` must be a number"))),
        }
    };
    let sampler = match p.get("sampler") {
        None | Some(Value::Null) => Sampler::Dpmpp2mKarras,
        Some(v) => {
            let name = v.as_str().unwrap_or_default();
            Sampler::parse(name)
                .ok_or_else(|| bad(CMD, format!("unknown sampler `{name}` ({})", Sampler::ALL.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))))?
        }
    };
    let seed = match p.get("seed") {
        None | Some(Value::Null) => random_seed(),
        Some(v) if v.as_i64() == Some(-1) => random_seed(),
        Some(v) => v.as_u64().ok_or_else(|| bad(CMD, "`seed` must be a whole number ≥ 0 (or -1 for random)"))?,
    };
    let loras = match p.get("loras") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) if a.len() <= MAX_LORAS => a
            .iter()
            .map(|l| {
                let file = l.get("file").and_then(Value::as_str).filter(|f| !f.trim().is_empty());
                let weight = l.get("weight").map_or(Some(1.0), Value::as_f64).filter(|w| w.is_finite());
                match (file, weight) {
                    (Some(f), Some(w)) => Ok(LoraRef { file: f.trim().to_string(), weight: w as f32 }),
                    _ => Err(bad(CMD, "each LoRA is {\"file\": name, \"weight\": number}")),
                }
            })
            .collect::<Result<Vec<_>>>()?,
        Some(_) => return Err(bad(CMD, format!("`loras` must be a list of at most {MAX_LORAS}"))),
    };
    let steps = u32::try_from(int("steps", 20)?).map_err(|_| bad(CMD, "`steps` is too large"))?;
    let req = GenerateRequest {
        checkpoint,
        vae: p.get("vae").and_then(Value::as_str).map(str::trim).filter(|v| !v.is_empty()).map(str::to_string),
        loras,
        prompt: text("prompt")?,
        negative: text("negative")?,
        steps,
        cfg: float("cfg", 7.0)? as f32,
        seed,
        sampler,
        strength: float("strength", 1.0)? as f32,
        width: frame.width(),
        height: frame.height(),
        init: None,
        mask: None,
    };
    req.validate().map_err(|e| bad(CMD, e.to_string()))?;
    Ok(req)
}

fn random_seed() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    let c = COUNTER.fetch_add(1, Ordering::Relaxed);
    // SplitMix64 of time and a counter; seeds stay within 2^32 so they're easy to type.
    let mut z = t ^ c.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (z ^ (z >> 31)) & 0xFFFF_FFFF
}

/// Job progress that also publishes previews.
struct GenProgress<'a> {
    ctx: &'a JobCtx,
    slot: Arc<Mutex<Option<Preview>>>,
    doc: DocId,
    frame: Rect,
}

impl photoka_ml::Progress for GenProgress<'_> {
    fn report(&self, fraction: f32, message: &str) {
        self.ctx.progress(fraction, message);
    }
    fn cancelled(&self) -> bool {
        self.ctx.cancelled()
    }
    fn preview(&self, image: &Rgba8) {
        let mut slot = self.slot.lock().unwrap_or_else(PoisonError::into_inner);
        let serial = slot.as_ref().map_or(0, |p| p.serial.wrapping_add(1));
        *slot = Some(Preview { doc: self.doc, frame: self.frame, image: image.clone(), serial });
    }
}

/// Clears the preview however the job ends.
struct PreviewGuard(Arc<Mutex<Option<Preview>>>);

impl Drop for PreviewGuard {
    fn drop(&mut self) {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = None;
    }
}

fn layer_name(prompt: &str) -> String {
    let p: String = prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    let short: String = p.chars().take(40).collect();
    match (short.is_empty(), short.len() < p.len()) {
        (true, _) => "Generated".into(),
        (false, true) => format!("Generated: {short}…"),
        (false, false) => format!("Generated: {short}"),
    }
}

fn generate(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ai.generate";
    let engine = match p.get("engine").and_then(Value::as_str) {
        None => EngineKind::Native,
        Some(e) => EngineKind::parse(e).ok_or_else(|| bad(CMD, format!("unknown engine `{e}` (native, comfyui)")))?,
    };
    let doc = active_doc(s)?;
    let frame = s.ai.generate.frame_for(&doc);
    let mut req = parse_request(p, frame)?;
    let store = s.ai.store();
    if engine == EngineKind::Native {
        photoka_ml::native::check_request(&req).map_err(ml_err)?;
        photoka_ml::native::resolve(&store, &req).map_err(ml_err)?;
        if s.ai.generate.busy() {
            return Err(EngineError::Other("a generation is already running; wait for it or cancel it".into()));
        }
    }
    let comfy = match engine {
        EngineKind::Comfyui => Some(crate::ai_comfy_cmds::generation_client(s)?),
        EngineKind::Native => None,
    };
    let user_mask = s.ai.generate.mask_for(doc.id).and_then(|m| m.crop(frame));
    let linear = composite_is_linear(&doc);
    let native = Arc::clone(&s.ai.generate.native);
    let slot = Arc::clone(&s.ai.generate.preview);
    let doc_id = doc.id;
    let label = "Generate Image";
    let name = layer_name(&req.prompt);
    crate::jobs::run(
        s,
        label,
        false,
        move |ctx| {
            let _clear = PreviewGuard(Arc::clone(&slot));
            ctx.progress(0.0, "Reading the frame");
            let init = frame_pixels(&doc, frame, linear)?;
            if init.data.chunks_exact(4).any(|px| px[3] > 0) {
                req.mask = photoka_ml::image::effective_mask(&init, user_mask.as_ref()).map_err(ml_err)?;
                req.init = Some(init);
            }
            let progress = GenProgress { ctx, slot: Arc::clone(&slot), doc: doc_id, frame };
            let out = match comfy {
                Some(client) => client.generate(&store, &req, &progress).map_err(ml_err)?,
                None => {
                    let mut eng = native.lock().unwrap_or_else(PoisonError::into_inner);
                    eng.generate(&store, &req, &progress).map_err(ml_err)?
                }
            };
            Ok((out, req.mask, doc))
        },
        move |s, (out, mask, doc)| {
            if s.active().map(|d| d.doc.id) != Some(doc_id) {
                return Err(EngineError::Other("the document changed while generating (it was closed or another one is active); generate again".into()));
            }
            let layer = result_layer(&doc, name, frame, &out.image, mask.as_ref(), linear)?;
            let nid = s.edit(label, move |d, active| {
                let id = d.insert_above(*active, layer);
                *active = Some(id);
                Ok(id)
            })?;
            if let Some(st) = s.active_mut() {
                st.last_damage = Some(frame);
            }
            Ok(json!({
                "layer": nid.0,
                "seed": out.seed,
                "elapsedMs": out.elapsed_ms,
                "frame": {"x": frame.x0, "y": frame.y0, "width": frame.width(), "height": frame.height()},
                "masked": mask.is_some(),
            }))
        },
    )
}

pub fn specs() -> Vec<CommandSpec> {
    use crate::vector_cmds::has_doc;
    vec![
        CommandSpec {
            id: "ai.frame.get",
            label: "Get Generation Frame",
            menu: &[],
            shortcut: None,
            params: "{} → {x,y,width,height,masked,maskCoverage}",
            enabled: has_doc,
            run: frame_get,
            journal: false,
        },
        CommandSpec {
            id: "ai.frame.set",
            label: "Set Generation Frame",
            menu: &[],
            shortcut: None,
            params: r#"{"x":px?,"y":px?,"width":px?,"height":px?} (sides snap to 64, 64…2048; missing values keep the current frame)"#,
            enabled: has_doc,
            run: frame_set,
            journal: false,
        },
        CommandSpec {
            id: "ai.frame.reset",
            label: "Reset Generation Frame",
            menu: &[],
            shortcut: None,
            params: "{} (512×512 centred on the canvas)",
            enabled: has_doc,
            run: frame_reset,
            journal: false,
        },
        CommandSpec {
            id: "ai.mask.stroke",
            label: "Paint Inpaint Mask",
            menu: &[],
            shortcut: None,
            params: r#"{"points":[[x,y],…],"size":px=48,"hardness":0…1=0.8,"erase":bool=false}"#,
            enabled: has_doc,
            run: mask_stroke,
            journal: false,
        },
        CommandSpec {
            id: "ai.mask.fill",
            label: "Mask Whole Frame",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: has_doc,
            run: mask_fill,
            journal: false,
        },
        CommandSpec {
            id: "ai.mask.clear",
            label: "Clear Inpaint Mask",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: has_doc,
            run: mask_clear,
            journal: false,
        },
        CommandSpec {
            id: "ai.mask.invert",
            label: "Invert Inpaint Mask",
            menu: &[],
            shortcut: None,
            params: "{} (inside the frame)",
            enabled: has_doc,
            run: mask_invert,
            journal: false,
        },
        CommandSpec {
            id: "ai.generate",
            label: "Generate Image",
            menu: &[],
            shortcut: None,
            params: r#"{"model":checkpoint file,"prompt":str,"negative":str?,"steps":1…150=20,"cfg":0…30=7,"seed":int|-1=random,"sampler":"ddim|euler|euler_a|dpmpp_2m|dpmpp_2m_karras|lcm"="dpmpp_2m_karras","strength":0…1=1,"vae":file?,"loras":[{"file","weight"}]?,"engine":"native|comfyui"="native"} → background job; a new layer at the frame (masked when inpainting) → {layer,seed,elapsedMs,frame,masked}"#,
            enabled: has_doc,
            run: generate,
            journal: true,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> Session {
        let mut s = Session::default();
        s.execute("file.new", json!({"width": 800, "height": 600})).unwrap();
        s
    }

    #[test]
    fn frame_defaults_snaps_and_resets() {
        let mut s = session();
        let f = s.execute("ai.frame.get", json!({})).unwrap();
        assert_eq!((f["x"].as_i64(), f["y"].as_i64(), f["width"].as_u64(), f["height"].as_u64()), (Some(144), Some(44), Some(512), Some(512)));
        let f = s.execute("ai.frame.set", json!({"x": 10, "y": -20, "width": 700, "height": 100})).unwrap();
        assert_eq!((f["x"].as_i64(), f["y"].as_i64(), f["width"].as_u64(), f["height"].as_u64()), (Some(10), Some(-20), Some(704), Some(128)));
        let f = s.execute("ai.frame.set", json!({"width": 99999})).unwrap();
        assert_eq!(f["width"].as_u64(), Some(u64::from(MAX_SIDE)));
        assert!(s.execute("ai.frame.set", json!({"x": "left"})).is_err());
        assert!(s.execute("ai.frame.set", json!({"x": 1e300})).is_err());
        let f = s.execute("ai.frame.reset", json!({})).unwrap();
        assert_eq!(f["width"].as_u64(), Some(512));
        // A small canvas gets a smaller frame.
        s.execute("file.new", json!({"width": 200, "height": 100})).unwrap();
        let f = s.execute("ai.frame.get", json!({})).unwrap();
        assert_eq!((f["width"].as_u64(), f["height"].as_u64()), (Some(192), Some(64)));
    }

    #[test]
    fn mask_strokes_fill_invert_and_clear() {
        let mut s = session();
        let f = s.execute("ai.mask.stroke", json!({"points": [[300, 300], [400, 300]], "size": 40})).unwrap();
        assert_eq!(f["masked"], json!(true));
        let doc = s.active().unwrap().doc.clone();
        let m = s.ai.generate.mask_for(doc.id).unwrap();
        assert_eq!(m.value(350, 300), 255);
        assert_eq!(m.value(350, 360), 0);
        // Erasing the same stroke empties it.
        let f = s.execute("ai.mask.stroke", json!({"points": [{"x": 300, "y": 300}, {"x": 400, "y": 300}], "size": 60, "erase": true, "hardness": 1})).unwrap();
        assert_eq!(f["masked"], json!(false));
        let f = s.execute("ai.mask.fill", json!({})).unwrap();
        assert!((f["maskCoverage"].as_f64().unwrap() - 1.0).abs() < 1e-9);
        let f = s.execute("ai.mask.invert", json!({})).unwrap();
        assert_eq!(f["masked"], json!(false));
        s.execute("ai.mask.invert", json!({})).unwrap();
        let f = s.execute("ai.mask.clear", json!({})).unwrap();
        assert_eq!(f["masked"], json!(false));
        // Bad input is an error, never a panic.
        for p in [
            json!({}),
            json!({"points": []}),
            json!({"points": [[1]]}),
            json!({"points": [[0, 0]], "size": 0}),
            json!({"points": [["a", 1]]}),
            json!({"points": [[1e300, 0]]}),
        ] {
            assert!(s.execute("ai.mask.stroke", p.clone()).is_err(), "{p}");
        }
        // A stroke far off the canvas is ignored.
        let f = s.execute("ai.mask.stroke", json!({"points": [[900000, 900000]], "size": 10})).unwrap();
        assert_eq!(f["masked"], json!(false));
    }

    #[test]
    fn generate_checks_its_params_before_starting() {
        let mut s = session();
        assert!(s.execute("ai.generate", json!({})).unwrap_err().to_string().contains("model"));
        for p in [
            json!({"model": "x.safetensors", "steps": 0}),
            json!({"model": "x.safetensors", "cfg": "high"}),
            json!({"model": "x.safetensors", "sampler": "turbo"}),
            json!({"model": "x.safetensors", "seed": -5}),
            json!({"model": "x.safetensors", "loras": [{"weight": 1}]}),
            json!({"model": "x.safetensors", "engine": "cloud"}),
            json!({"model": "x.safetensors"}),
        ] {
            assert!(s.execute("ai.generate", p.clone()).is_err(), "{p}");
        }
        let doc = s.active().unwrap().doc.clone();
        assert!(s.active().unwrap().doc.layers.len() == doc.layers.len());
    }

    #[test]
    fn frame_pixels_and_result_layer_round_trip() {
        let mut s = session();
        s.execute("edit.fill", json!({"color": [1.0, 0.0, 0.0]})).ok();
        let doc = s.active().unwrap().doc.clone();
        // Half the frame hangs off the right edge: transparent there.
        let frame = Rect::from_xywh(600, 100, 256, 64);
        let img = frame_pixels(&doc, frame, false).unwrap();
        assert_eq!((img.width, img.height), (256, 64));
        assert_eq!(img.data[3], 255);
        assert_eq!(img.data[(255 * 4) + 3], 0);
        let mask = photoka_ml::image::effective_mask(&img, None).unwrap().unwrap();
        let l = result_layer(&doc, layer_name("  a  red\nbarn "), frame, &img, Some(&mask), false).unwrap();
        assert_eq!(l.name, "Generated: a red barn");
        let m = l.mask.as_ref().unwrap();
        assert!(m.value(frame.x0 + 250, frame.y0 + 30) > 0.9);
        assert!(m.value(frame.x0 + 5, frame.y0 + 30) < 0.1);
        assert!(m.value(0, 0) < 0.01);
        assert_eq!(layer_name(&"x".repeat(60)), format!("Generated: {}…", "x".repeat(40)));
        assert_eq!(layer_name(""), "Generated");
    }

    #[test]
    fn seeds_are_random_unless_given() {
        let f = Rect::from_xywh(0, 0, 512, 512);
        let a = parse_request(&json!({"model": "m"}), f).unwrap().seed;
        let b = parse_request(&json!({"model": "m", "seed": -1}), f).unwrap().seed;
        assert_ne!(a, b);
        assert_eq!(parse_request(&json!({"model": "m", "seed": 42}), f).unwrap().seed, 42);
    }

    /// Inpaint a frame through the command, with the real SD 1.5 checkpoint.
    #[test]
    #[ignore = "needs real model files (PHOTOKA_MODELS_DIR)"]
    fn real_generate_adds_a_masked_layer() {
        assert!(std::env::var_os("PHOTOKA_MODELS_DIR").is_some(), "set PHOTOKA_MODELS_DIR");
        let mut s = session();
        s.execute("ai.frame.set", json!({"x": 100, "y": 100, "width": 256, "height": 256})).unwrap();
        s.execute("ai.mask.stroke", json!({"points": [[180, 228], [276, 228]], "size": 80})).unwrap();
        let layers = s.active().unwrap().doc.layers.len();
        let r = s
            .execute(
                "ai.generate",
                json!({"model": "v1-5-pruned-emaonly-fp16.safetensors", "prompt": "a red apple", "steps": 8, "seed": 7, "sampler": "euler"}),
            )
            .unwrap();
        assert_eq!(r["masked"], json!(true));
        assert_eq!(r["seed"], json!(7));
        let doc = s.active().unwrap().doc.clone();
        assert_eq!(doc.layers.len(), layers + 1);
        let l = doc.layers.iter().find(|l| l.name.starts_with("Generated")).unwrap();
        let m = l.mask.as_ref().unwrap();
        assert!(m.value(228, 228) > 0.9 && m.value(110, 340) < 0.1);
        let png = doc_png(&doc);
        std::fs::write(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/ai-generate-e2e.png"), png).unwrap();
        // Undo removes it in one step.
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(s.active().unwrap().doc.layers.len(), layers);
    }

    fn doc_png(doc: &Document) -> Vec<u8> {
        let b = doc.bounds();
        let buf = photoka_compose::render(doc, b);
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let data: Vec<u8> = buf.px.iter().flat_map(|p| p.map(q)).collect();
        Rgba8::from_raw(b.width(), b.height(), data).unwrap().to_png().unwrap()
    }
}

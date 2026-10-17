//! AI Upscale (Real-ESRGAN via ComfyUI): `ai.upscale`.
//!
//! Reads the document composite (or selection), runs an installed upscale model through the
//! managed ComfyUI, and inserts the result as a new layer. Hardware probe supplies default scale
//! and tile hints (tile is advisory; ComfyUI's core ImageUpscaleWithModel is whole-image).

use std::sync::Arc;

use photoka_color::PixelFormat;
use photoka_doc::{Document, Layer};
use photoka_geom::Rect;
use photoka_ml::Progress;
use photoka_ml::catalog::ModelKind;
use photoka_ml::image::{MAX_SIDE, Rgba8};
use serde_json::{Value, json};

use crate::ai_cmds::{JobProgress, bad, ml_err};
use crate::ai_comfy_cmds;
use crate::commands::CommandSpec;
use crate::vector_cmds::has_doc;
use crate::{EngineError, Result, Session};

fn active_doc(s: &Session) -> Result<Arc<Document>> {
    s.active().map(|d| d.doc.clone()).ok_or(EngineError::NoDocument)
}

fn pick_upscale_model(store: &photoka_ml::store::ModelStore, prefer_scale: u32) -> Result<String> {
    let models = store.scan(None);
    let ups: Vec<_> = models.iter().filter(|m| m.kind == ModelKind::Upscale && m.problem.is_none()).collect();
    if ups.is_empty() {
        return Err(EngineError::Other(
            "no upscale model is installed; download Real-ESRGAN from Model Manager › Suggested".into(),
        ));
    }
    let prefer_x2 = prefer_scale <= 2;
    let score = |name: &str| -> i32 {
        let n = name.to_ascii_lowercase();
        let x2 = n.contains("x2");
        let x4 = n.contains("x4");
        match (prefer_x2, x2, x4) {
            (true, true, _) => 2,
            (false, _, true) => 2,
            (_, true, _) | (_, _, true) => 1,
            _ => 0,
        }
    };
    ups.into_iter()
        .max_by_key(|m| score(&m.file_name))
        .map(|m| m.file_name.clone())
        .ok_or_else(|| EngineError::Other("no upscale model is installed".into()))
}

fn composite_is_linear(doc: &Document) -> bool {
    crate::display_color::is_linear_rgb(&crate::color_cmds::composite_profile(doc))
}

/// Composite (or selection) as 8-bit sRGB RGBA.
fn source_rgba(doc: &Document) -> Result<(Rgba8, Rect)> {
    let canvas = doc.bounds();
    let region = doc
        .selection
        .as_ref()
        .map(|sel| sel.content_bounds())
        .filter(|b| !b.is_empty())
        .unwrap_or(canvas)
        .intersect(&canvas);
    if region.is_empty() {
        return Err(bad("ai.upscale", "nothing to upscale (empty selection)"));
    }
    let (rw, rh) = (region.width(), region.height());
    if rw > MAX_SIDE || rh > MAX_SIDE {
        return Err(bad(
            "ai.upscale",
            format!("the region is {rw}×{rh}; AI Upscale accepts at most {MAX_SIDE}×{MAX_SIDE} (crop or resize first)"),
        ));
    }
    let linear = composite_is_linear(doc);
    let mut out = vec![0u8; rw as usize * rh as usize * 4];
    let buf = photoka_compose::render(doc, region);
    let iw = region.width() as usize;
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    for (i, px) in buf.px.iter().enumerate() {
        let (x, y) = (region.x0 + (i % iw.max(1)) as i32, region.y0 + (i / iw.max(1)) as i32);
        let j = ((y - region.y0) as usize * rw as usize + (x - region.x0) as usize) * 4;
        let rgb = px.map(|v| if linear { photoka_color::convert::linear_to_srgb(v) } else { v });
        if let Some(d) = out.get_mut(j..j + 4) {
            d.copy_from_slice(&[q(rgb[0]), q(rgb[1]), q(rgb[2]), q(px[3])]);
        }
    }
    Ok((Rgba8::from_raw(rw, rh, out).map_err(ml_err)?, region))
}

fn raster_at(doc: &Document, name: String, frame: Rect, img: &Rgba8, linear: bool) -> Result<Layer> {
    if (img.width, img.height) != (frame.width(), frame.height()) {
        return Err(EngineError::Other("internal error: the upscale result doesn't match its frame".into()));
    }
    let base = doc.pixel_format();
    let fmt = PixelFormat::new(base.mode, base.sample, true);
    let n = fmt.channels();
    let mut data = vec![0.0f32; img.data.len() / 4 * n];
    let mut enc = [0.0f32; 8];
    for (px, dst) in img.data.as_chunks::<4>().0.iter().zip(data.chunks_exact_mut(n)) {
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
    Ok(l)
}

fn upscale(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ai.upscale";
    let doc = active_doc(s)?;
    let hw = photoka_ml::system::HardwareProfile::probe(&s.ai.models_dir);
    let scale = match p.get("scale") {
        None | Some(Value::Null) => hw.default_upscale_scale(),
        Some(v) => {
            let n = v.as_u64().ok_or_else(|| bad(CMD, "`scale` must be 2 or 4"))?;
            if n != 2 && n != 4 {
                return Err(bad(CMD, "`scale` must be 2 or 4"));
            }
            n as u32
        }
    };
    let store = s.ai.store();
    let model = match p.get("model").and_then(Value::as_str).map(str::trim).filter(|m| !m.is_empty()) {
        Some(m) => {
            store.existing(ModelKind::Upscale, m).map_err(ml_err)?;
            m.to_string()
        }
        None => pick_upscale_model(&store, scale)?,
    };
    let tile = match p.get("tile") {
        None | Some(Value::Null) => hw.default_upscale_tile(),
        Some(v) => {
            let n = v.as_u64().ok_or_else(|| bad(CMD, "`tile` must be a whole number"))? as u32;
            if !(64..=2048).contains(&n) {
                return Err(bad(CMD, "`tile` must be between 64 and 2048"));
            }
            n
        }
    };
    let client = ai_comfy_cmds::generation_client(s).map_err(|e| {
        EngineError::Other(format!(
            "{e}\nAI Upscale needs ComfyUI running (Install / Start from the Generate panel) and a Real-ESRGAN model from Model Manager."
        ))
    })?;
    let (src, region) = source_rgba(&doc)?;
    let linear = composite_is_linear(&doc);
    let store = s.ai.store();
    let doc_id = doc.id;
    let label = format!("AI Upscale {scale}×");
    crate::jobs::run(
        s,
        &label,
        false,
        move |ctx| {
            let progress = JobProgress(ctx);
            progress.report(0.0, "Upscaling");
            let out = client.upscale(&store, &model, &src, &progress).map_err(ml_err)?;
            let want_w = src.width.saturating_mul(scale).min(photoka_ml::image::MAX_UPSCALE_SIDE);
            let want_h = src.height.saturating_mul(scale).min(photoka_ml::image::MAX_UPSCALE_SIDE);
            let image = if out.width == want_w && out.height == want_h {
                out
            } else {
                out.resized(want_w, want_h).map_err(ml_err)?
            };
            Ok((image, region, scale, tile, model, linear))
        },
        move |s, (image, region, scale, tile, model, linear)| {
            if s.active().map(|d| d.doc.id) != Some(doc_id) {
                return Err(EngineError::Other("the document changed while upscaling; try again".into()));
            }
            let dest = Rect::from_xywh(region.x0, region.y0, image.width, image.height);
            let name = format!("Upscaled {scale}×");
            let layer = raster_at(&doc, name, dest, &image, linear)?;
            let nid = s.edit("AI Upscale", move |d, active| {
                let id = d.insert_above(*active, layer);
                *active = Some(id);
                Ok(id)
            })?;
            Ok(json!({
                "layer": nid.0,
                "scale": scale,
                "tile": tile,
                "model": model,
                "width": image.width,
                "height": image.height,
                "source": {"x": region.x0, "y": region.y0, "width": region.width(), "height": region.height()},
                "dest": {"x": dest.x0, "y": dest.y0, "width": dest.width(), "height": dest.height()},
            }))
        },
    )
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: "ai.upscale",
        label: "AI Upscale…",
        menu: &["Image"],
        shortcut: None,
        params: r#"{"scale":2|4?,"model":upscale file?,"tile":px?} → background job via ComfyUI Real-ESRGAN; new layer → {layer,scale,model,width,height,source,dest}. Defaults from this Mac's RAM."#,
        enabled: has_doc,
        run: upscale,
        journal: true,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn upscale_rejects_bad_scale_and_missing_comfy() {
        let mut s = Session::default();
        s.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        let e = s.execute("ai.upscale", json!({"scale": 3})).unwrap_err().to_string();
        assert!(e.contains("2 or 4"), "{e}");
        let e = s.execute("ai.upscale", json!({"scale": 2})).unwrap_err().to_string();
        assert!(e.contains("ComfyUI") || e.contains("upscale model") || e.contains("isn't running"), "{e}");
    }
}

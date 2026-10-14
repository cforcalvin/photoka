//! Offline image generation in the shell (Photoka extension, Draw Things-style): the Generate
//! Frame tool (move and resize the frame; paint or erase the inpaint mask), the frame, mask and
//! live-preview overlay on the canvas, and Window › Generate, the panel with the prompt and
//! settings. Every change is an engine `ai.*` command, so agents drive the same thing; the
//! generation itself is a background job (`ai.generate`) that adds a layer.

use egui::{Color32, Pos2, RichText, Stroke, TextureHandle, vec2};
use photoka_doc::DocId;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::PhotokaApp;
use crate::canvas::{ToolEvent, ViewXform};
use crate::state::Tool;
use crate::theme::Tokens;

pub const MENU_PANEL: &str = "window.generate";
pub const MENU_RUN: &str = "edit.generateInFrame";

/// What the Generate Frame tool's drags do.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenMode {
    /// Move and resize the frame.
    #[default]
    Frame,
    /// Paint the inpaint mask.
    Mask,
    /// Erase the inpaint mask.
    Erase,
}

/// The engine that generates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenEngine {
    #[default]
    Native,
    Comfyui,
}

impl GenEngine {
    pub fn id(self) -> &'static str {
        match self {
            GenEngine::Native => "native",
            GenEngine::Comfyui => "comfyui",
        }
    }
}

/// One LoRA in the panel's stack.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoraSlot {
    pub file: String,
    pub weight: f32,
}

/// A frame gesture in progress (document coordinates).
#[derive(Clone, Debug, PartialEq)]
pub enum GenDrag {
    /// Moving (`handle` = (0, 0)) or resizing by an edge or corner (−1/+1 per axis).
    Frame {
        start: [f64; 2],
        cur: [f64; 2],
        rect: photoka_geom::Rect,
        handle: (i8, i8),
    },
    /// Drawing a new frame from a corner.
    New {
        start: [f64; 2],
        cur: [f64; 2],
    },
    Stroke {
        points: Vec<[f64; 2]>,
    },
}

/// Generate panel and tool state (serde: readable and drivable through the control channel).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GenerateUi {
    pub open: bool,
    pub engine: GenEngine,
    pub mode: GenMode,
    pub model: String,
    pub prompt: String,
    pub negative: String,
    pub sampler: String,
    pub steps: u32,
    pub cfg: f32,
    pub strength: f32,
    pub seed: u64,
    /// A new seed for every generation (the dice); off keeps [`Self::seed`].
    pub random_seed: bool,
    pub loras: Vec<LoraSlot>,
    /// Mask brush size and hardness (document pixels, 0…1).
    pub brush_size: f32,
    pub brush_hardness: f32,
    #[serde(skip)]
    pub drag: Option<GenDrag>,
    /// `ai.models.list`, refreshed about once a second while the panel is open.
    #[serde(skip)]
    pub installed: Value,
    #[serde(skip)]
    pub refreshed_at: Option<f64>,
    /// `ai.comfy.status`, refreshed with `installed`.
    #[serde(skip)]
    pub comfy: Value,
    /// The install button was pressed: show the license and size before installing.
    #[serde(skip)]
    pub comfy_confirm: bool,
    #[serde(skip)]
    pub textures: GenTextures,
}

impl Default for GenerateUi {
    fn default() -> Self {
        GenerateUi {
            open: false,
            engine: GenEngine::Native,
            mode: GenMode::Frame,
            model: String::new(),
            prompt: String::new(),
            negative: String::new(),
            sampler: "dpmpp_2m_karras".into(),
            steps: 20,
            cfg: 7.0,
            strength: 1.0,
            seed: 0,
            random_seed: true,
            loras: Vec::new(),
            brush_size: 48.0,
            brush_hardness: 0.8,
            drag: None,
            installed: Value::Null,
            refreshed_at: None,
            comfy: Value::Null,
            comfy_confirm: false,
            textures: GenTextures::default(),
        }
    }
}

/// Cached overlay textures.
#[derive(Clone, Default)]
pub struct GenTextures {
    /// The mask tint: (document, mask revision, texture, the plane's rect).
    mask: Option<(DocId, u64, TextureHandle, photoka_geom::Rect)>,
    /// The live preview: (preview serial, texture).
    preview: Option<(u64, TextureHandle)>,
}

impl std::fmt::Debug for GenTextures {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GenTextures")
    }
}

impl PartialEq for GenTextures {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

pub fn handles(id: &str) -> bool {
    id == MENU_PANEL || id == MENU_RUN
}

pub fn checked(app: &PhotokaApp, id: &str) -> Option<bool> {
    (id == MENU_PANEL).then_some(app.ui.generate.open)
}

pub fn enabled(app: &PhotokaApp, id: &str) -> Option<bool> {
    match id {
        MENU_PANEL => Some(true),
        MENU_RUN => Some(app.session.active().is_some() && !app.session.ai.generate.busy()),
        _ => None,
    }
}

/// Window › Generate toggles the panel (`{"open": bool}` sets it); Edit › Generate in Frame…
/// generates with the panel's settings, or opens the panel when no model is chosen yet.
pub fn menu(app: &mut PhotokaApp, id: &str, params: &Value) -> Option<Result<Value, String>> {
    match id {
        MENU_PANEL => {
            let engine = match params.get("engine").and_then(Value::as_str) {
                None => None,
                Some("native") => Some(GenEngine::Native),
                Some("comfyui") => Some(GenEngine::Comfyui),
                Some(e) => return Some(Err(format!("unknown engine `{e}` (native, comfyui)"))),
            };
            let g = &mut app.ui.generate;
            if let Some(e) = engine {
                g.engine = e;
            }
            g.open = params.get("open").and_then(Value::as_bool).unwrap_or(engine.is_some() || !g.open);
            g.refreshed_at = None;
            if g.open && app.session.active().is_some() {
                app.ui.tool = Tool::GenFrame;
            }
            Some(Ok(json!({"open": app.ui.generate.open})))
        }
        MENU_RUN => {
            if app.ui.generate.model.is_empty() {
                app.ui.generate.open = true;
                app.ui.generate.refreshed_at = None;
                return Some(Ok(json!({"open": true, "started": false})));
            }
            Some(start(app))
        }
        _ => None,
    }
}

fn random_seed() -> u64 {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    let mut z = t.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (z ^ (z >> 31)) & 0xFFFF_FFFF
}

/// The `ai.generate` params for the panel's settings.
pub fn params(g: &GenerateUi) -> Value {
    let loras: Vec<Value> = g.loras.iter().filter(|l| !l.file.is_empty()).map(|l| json!({"file": l.file, "weight": l.weight})).collect();
    json!({
        "model": g.model,
        "prompt": g.prompt,
        "negative": g.negative,
        "sampler": g.sampler,
        "steps": g.steps,
        "cfg": g.cfg,
        "strength": g.strength,
        "seed": g.seed,
        "loras": loras,
        "engine": g.engine.id(),
    })
}

/// Start a generation with the panel's settings.
pub fn start(app: &mut PhotokaApp) -> Result<Value, String> {
    if app.ui.generate.random_seed {
        app.ui.generate.seed = random_seed();
    }
    let p = params(&app.ui.generate);
    app.run("ai.generate", p)
}

/// The running generation, if any.
fn running_job(app: &PhotokaApp) -> Option<photoka_engine::jobs::JobInfo> {
    app.session.jobs().into_iter().find(|j| j.command == "ai.generate")
}

fn comfy_job(app: &PhotokaApp) -> Option<photoka_engine::jobs::JobInfo> {
    app.session.jobs().into_iter().find(|j| j.command.starts_with("ai.comfy."))
}

/// The ComfyUI engine's install / start / stop rows.
fn comfy_section(ui: &mut egui::Ui, t: &Tokens, g: &mut GenerateUi, job: Option<&photoka_engine::jobs::JobInfo>, acts: &mut Vec<Act>) {
    let st = &g.comfy;
    let flag = |k: &str| st.get(k).and_then(Value::as_bool).unwrap_or(false);
    let dim = |s: &str| RichText::new(s.to_string()).color(t.text_dim).size(11.0);
    if let Some(j) = job {
        let (r, _) = ui.allocate_exact_size(vec2(ui.available_width(), 6.0), egui::Sense::hover());
        crate::jobs_ui::bar(ui, r, (j.progress > 0.0).then_some(j.progress), t);
        ui.horizontal(|ui| {
            ui.add(egui::Label::new(dim(&j.message)).truncate());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button(tl!("Cancel")).clicked() {
                    acts.push(Act::Cancel(j.id));
                }
            });
        });
        return;
    }
    if !flag("supported") {
        ui.label(RichText::new(tl!("The managed ComfyUI needs a Mac with Apple silicon.")).color(t.warning).size(11.5));
        return;
    }
    if !flag("installed") {
        if g.comfy_confirm {
            ui.label(dim(tl!("ComfyUI is open-source software under the GPL-3.0 license. Photoka downloads it with Python and PyTorch (about 4 GB) and runs it as a separate program on this Mac; nothing leaves your computer.")));
            ui.hyperlink_to(RichText::new("github.com/Comfy-Org/ComfyUI").size(11.0), photoka_ml::comfy::install::COMFY_PAGE);
            ui.horizontal(|ui| {
                if ui.button(tl!("Accept and Install")).clicked() {
                    acts.push(Act::Run("ai.comfy.install", json!({"accept": true})));
                    g.comfy_confirm = false;
                }
                if ui.button(tl!("Cancel")).clicked() {
                    g.comfy_confirm = false;
                }
            });
        } else {
            ui.horizontal(|ui| {
                ui.label(dim(tl!("ComfyUI isn't installed.")));
                if ui.small_button(tl!("Install ComfyUI…")).clicked() {
                    g.comfy_confirm = true;
                }
            });
        }
        return;
    }
    let version = st.get("version").and_then(Value::as_str).unwrap_or_default().to_string();
    match st.get("port").and_then(Value::as_u64) {
        Some(port) => {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(crate::i18n::fmt(tl!("ComfyUI {version} is running on port {port}"), &[("version", &version), ("port", &port.to_string())]))
                        .color(t.accent_text)
                        .size(11.0),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button(tl!("Stop")).clicked() {
                        acts.push(Act::Run("ai.comfy.stop", json!({})));
                    }
                });
            });
        }
        None => {
            ui.horizontal(|ui| {
                ui.label(dim(&crate::i18n::fmt(tl!("ComfyUI {version} is installed"), &[("version", &version)])));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button(tl!("Remove")).on_hover_text(tl!("Delete ComfyUI, its Python and PyTorch (your models stay)")).clicked() {
                        acts.push(Act::Run("ai.comfy.uninstall", json!({})));
                    }
                    if ui.small_button(tl!("Start")).clicked() {
                        acts.push(Act::Run("ai.comfy.start", json!({})));
                    }
                });
            });
            if let Some(why) = st.get("lastExit").and_then(Value::as_str) {
                ui.add(egui::Label::new(RichText::new(why.lines().next().unwrap_or_default()).color(t.warning).size(10.5)).truncate());
            }
        }
    }
    if let Some(log) = st.get("logPath").and_then(Value::as_str) {
        ui.add(egui::Label::new(RichText::new(format!("{} {log}", tl!("Log:"))).color(t.text_faint).size(10.0)).truncate());
    }
}

fn doc_rect(a: [f64; 2], b: [f64; 2]) -> photoka_geom::Rect {
    let (x0, x1) = (a[0].min(b[0]).round() as i32, a[0].max(b[0]).round() as i32);
    let (y0, y1) = (a[1].min(b[1]).round() as i32, a[1].max(b[1]).round() as i32);
    photoka_geom::Rect::new(x0, y0, x1, y1)
}

fn snap(v: i32) -> i32 {
    let s = photoka_engine::ai_gen_cmds::SNAP as i32;
    ((v as f32 / s as f32).round() as i32 * s).clamp(photoka_engine::ai_gen_cmds::MIN_SIDE as i32, photoka_engine::ai_gen_cmds::MAX_SIDE as i32)
}

/// The frame a drag ends at: sides snapped to 64 px, the fixed edges kept in place.
pub fn dragged_frame(d: &GenDrag) -> Option<photoka_geom::Rect> {
    match d {
        GenDrag::Frame { start, cur, rect, handle } => {
            let (dx, dy) = ((cur[0] - start[0]).round() as i32, (cur[1] - start[1]).round() as i32);
            if *handle == (0, 0) {
                return Some(rect.translate(dx, dy));
            }
            let (mut x0, mut y0, mut x1, mut y1) = (rect.x0, rect.y0, rect.x1, rect.y1);
            match handle.0 {
                -1 => x0 += dx,
                1 => x1 += dx,
                _ => {}
            }
            match handle.1 {
                -1 => y0 += dy,
                1 => y1 += dy,
                _ => {}
            }
            let (w, h) = (snap((x1 - x0).abs()), snap((y1 - y0).abs()));
            let (x0, x1) = if handle.0 == -1 { (x1.max(x0) - w, x1.max(x0)) } else { (x0.min(x1), x0.min(x1) + w) };
            let (y0, y1) = if handle.1 == -1 { (y1.max(y0) - h, y1.max(y0)) } else { (y0.min(y1), y0.min(y1) + h) };
            Some(photoka_geom::Rect::new(x0, y0, x1, y1))
        }
        GenDrag::New { start, cur } => {
            let r = doc_rect(*start, *cur);
            if r.width() < 8 && r.height() < 8 {
                return None;
            }
            let (w, h) = (snap(r.width() as i32), snap(r.height() as i32));
            let x0 = if cur[0] < start[0] { start[0].round() as i32 - w } else { start[0].round() as i32 };
            let y0 = if cur[1] < start[1] { start[1].round() as i32 - h } else { start[1].round() as i32 };
            Some(photoka_geom::Rect::from_xywh(x0, y0, w as u32, h as u32))
        }
        GenDrag::Stroke { .. } => None,
    }
}

/// Which handle of `r` is under `(x, y)`, within `tol` document pixels: (0, 0) inside.
fn handle_at(r: photoka_geom::Rect, x: f64, y: f64, tol: f64) -> Option<(i8, i8)> {
    let (x0, y0, x1, y1) = (f64::from(r.x0), f64::from(r.y0), f64::from(r.x1), f64::from(r.y1));
    if x < x0 - tol || x > x1 + tol || y < y0 - tol || y > y1 + tol {
        return None;
    }
    let hx = if (x - x0).abs() <= tol {
        -1
    } else if (x - x1).abs() <= tol {
        1
    } else {
        0
    };
    let hy = if (y - y0).abs() <= tol {
        -1
    } else if (y - y1).abs() <= tol {
        1
    } else {
        0
    };
    Some((hx, hy))
}

/// Pointer input for the Generate Frame tool. Returns true when the event was its.
pub fn pointer(app: &mut PhotokaApp, ev: ToolEvent, _mods: egui::Modifiers) -> bool {
    if app.ui.tool != Tool::GenFrame {
        return false;
    }
    let Some(doc) = app.session.active().map(|d| d.doc.clone()) else { return true };
    let frame = app.session.ai.generate.frame_for(&doc);
    let zoom = ViewXform::active(app).map_or(1.0, |xf| f64::from(xf.zoom.max(1e-3)));
    let mode = app.ui.generate.mode;
    match ev {
        ToolEvent::Down { x, y, .. } => {
            app.ui.generate.drag = Some(match mode {
                GenMode::Frame => match handle_at(frame, x, y, 8.0 / zoom) {
                    Some(handle) => GenDrag::Frame { start: [x, y], cur: [x, y], rect: frame, handle },
                    None => GenDrag::New { start: [x, y], cur: [x, y] },
                },
                GenMode::Mask | GenMode::Erase => GenDrag::Stroke { points: vec![[x, y]] },
            });
        }
        ToolEvent::Move { x, y, .. } => match &mut app.ui.generate.drag {
            Some(GenDrag::Frame { cur, .. }) | Some(GenDrag::New { cur, .. }) => *cur = [x, y],
            Some(GenDrag::Stroke { points }) if points.last().is_none_or(|p| (p[0] - x).hypot(p[1] - y) >= 1.0) && points.len() < 50_000 => {
                points.push([x, y]);
            }
            Some(GenDrag::Stroke { .. }) | None => {}
        },
        ToolEvent::Up { x, y } => {
            let drag = app.ui.generate.drag.take();
            let r = match drag {
                Some(GenDrag::Stroke { mut points }) => {
                    points.push([x, y]);
                    let g = &app.ui.generate;
                    app.run("ai.mask.stroke", json!({"points": points, "size": g.brush_size, "hardness": g.brush_hardness, "erase": mode == GenMode::Erase}))
                }
                Some(mut d) => {
                    if let GenDrag::Frame { cur, .. } | GenDrag::New { cur, .. } = &mut d {
                        *cur = [x, y];
                    }
                    match dragged_frame(&d) {
                        Some(f) if f != frame => app.run("ai.frame.set", json!({"x": f.x0, "y": f.y0, "width": f.width(), "height": f.height()})),
                        _ => Ok(Value::Null),
                    }
                }
                None => Ok(Value::Null),
            };
            if let Err(e) = r {
                crate::notices::error(app, e);
            }
        }
    }
    true
}

/// The Generate Frame tool's options bar.
pub fn options_bar(app: &mut PhotokaApp, ui: &mut egui::Ui, tool: Tool) -> bool {
    if tool != Tool::GenFrame {
        return false;
    }
    let g = &mut app.ui.generate;
    for (mode, icon, tip) in [
        (GenMode::Frame, "frame-placeholder", tl!("Move and resize the frame")),
        (GenMode::Mask, "brush", tl!("Paint what to regenerate")),
        (GenMode::Erase, "eraser", tl!("Erase the mask")),
    ] {
        if crate::icons::button(ui, icon, 24.0, g.mode == mode, tip).clicked() {
            g.mode = mode;
        }
    }
    crate::widgets::vline(ui, 22.0);
    if g.mode != GenMode::Frame {
        ui.label(tl!("Size"));
        ui.add(egui::DragValue::new(&mut g.brush_size).range(1.0..=2000.0).speed(1.0).suffix(" px"));
        ui.label(tl!("Hardness"));
        let mut hard = g.brush_hardness * 100.0;
        if ui.add(egui::DragValue::new(&mut hard).range(0.0..=100.0).speed(1.0).suffix("%")).changed() {
            g.brush_hardness = hard / 100.0;
        }
        crate::widgets::vline(ui, 22.0);
    }
    let mut run: Option<&str> = None;
    if ui.small_button(tl!("Mask Frame")).on_hover_text(tl!("Regenerate the whole frame")).clicked() {
        run = Some("ai.mask.fill");
    }
    if ui.small_button(tl!("Invert Mask")).clicked() {
        run = Some("ai.mask.invert");
    }
    if ui.small_button(tl!("Clear Mask")).clicked() {
        run = Some("ai.mask.clear");
    }
    if ui.small_button(tl!("Reset Frame")).clicked() {
        run = Some("ai.frame.reset");
    }
    crate::widgets::vline(ui, 22.0);
    if ui.small_button(tl!("Generate…")).clicked() {
        app.ui.generate.open = true;
        app.ui.generate.refreshed_at = None;
    }
    if let Some(id) = run
        && let Err(e) = app.run(id, json!({}))
    {
        crate::notices::error(app, e);
    }
    true
}

/// A textured quad over a document rectangle (follows the view's rotation and flip).
fn quad(painter: &egui::Painter, xf: &ViewXform, r: photoka_geom::Rect, tex: egui::TextureId, tint: Color32) {
    let (x0, y0, x1, y1) = (r.x0 as f32, r.y0 as f32, r.x1 as f32, r.y1 as f32);
    let mut mesh = egui::Mesh::with_texture(tex);
    for (p, uv) in [((x0, y0), (0.0, 0.0)), ((x1, y0), (1.0, 0.0)), ((x1, y1), (1.0, 1.0)), ((x0, y1), (0.0, 1.0))] {
        mesh.vertices.push(egui::epaint::Vertex { pos: xf.to_screen(p.0, p.1), uv: Pos2::new(uv.0, uv.1), color: tint });
    }
    mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
    painter.add(egui::Shape::mesh(mesh));
}

fn outline(painter: &egui::Painter, xf: &ViewXform, r: photoka_geom::Rect, stroke: Stroke) {
    let (x0, y0, x1, y1) = (r.x0 as f32, r.y0 as f32, r.x1 as f32, r.y1 as f32);
    let pts: Vec<Pos2> = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)].into_iter().map(|(x, y)| xf.to_screen(x, y)).collect();
    painter.add(egui::Shape::line(pts, stroke));
}

/// The mask as a tinted texture (downsampled so a side stays within 2048 texels).
fn mask_texture(ctx: &egui::Context, plane: &photoka_engine::ai_gen_cmds::MaskPlane, tint: Color32) -> TextureHandle {
    let r = plane.rect();
    let (w, h) = (r.width() as usize, r.height() as usize);
    let k = w.max(h).div_ceil(2048).max(1);
    let (tw, th) = (w.div_ceil(k).max(1), h.div_ceil(k).max(1));
    let data = plane.data();
    let mut px = Vec::with_capacity(tw * th);
    for ty in 0..th {
        for tx in 0..tw {
            let v = data.get((ty * k).min(h.saturating_sub(1)) * w + (tx * k).min(w.saturating_sub(1))).copied().unwrap_or(0);
            let a = (f32::from(v) / 255.0 * 0.45 * 255.0) as u8;
            px.push(Color32::from_rgba_unmultiplied(tint.r(), tint.g(), tint.b(), a));
        }
    }
    let img = egui::ColorImage::new([tw, th], px);
    ctx.load_texture("gen-mask", img, egui::TextureOptions::NEAREST)
}

/// The frame, the mask tint and the live preview (while the tool or the panel is up).
pub fn draw_overlay(app: &mut PhotokaApp, painter: &egui::Painter, xf: &ViewXform) {
    let shown = app.ui.tool == Tool::GenFrame || app.ui.generate.open;
    let Some(doc) = app.session.active().map(|d| d.doc.clone()) else { return };
    let gen_state = &app.session.ai.generate;
    let preview = gen_state.preview().filter(|p| p.doc == doc.id);
    if !shown && preview.is_none() {
        return;
    }
    let t = Tokens::get(painter.ctx());
    let ctx = painter.ctx().clone();
    // The mask.
    if shown && let Some(plane) = gen_state.mask_for(doc.id) {
        let rev = gen_state.revision();
        let stale = app.ui.generate.textures.mask.as_ref().is_none_or(|(d, r, _, _)| *d != doc.id || *r != rev);
        if stale {
            let tex = mask_texture(&ctx, plane, t.danger);
            app.ui.generate.textures.mask = Some((doc.id, rev, tex, plane.rect()));
        }
        if let Some((_, _, tex, rect)) = &app.ui.generate.textures.mask {
            quad(painter, xf, *rect, tex.id(), Color32::WHITE);
        }
    }
    // The live preview, stretched over the frame it's for.
    if let Some(p) = &preview {
        let stale = app.ui.generate.textures.preview.as_ref().is_none_or(|(s, _)| *s != p.serial);
        if stale {
            let img = egui::ColorImage::from_rgba_unmultiplied([p.image.width as usize, p.image.height as usize], &p.image.data);
            app.ui.generate.textures.preview = Some((p.serial, ctx.load_texture("gen-preview", img, egui::TextureOptions::LINEAR)));
        }
        if let Some((_, tex)) = &app.ui.generate.textures.preview {
            quad(painter, xf, p.frame, tex.id(), Color32::WHITE);
        }
    }
    if !shown {
        return;
    }
    // A stroke being painted.
    if let Some(GenDrag::Stroke { points }) = &app.ui.generate.drag {
        let w = (app.ui.generate.brush_size * xf.zoom).max(1.0);
        let c = if app.ui.generate.mode == GenMode::Erase { t.text_faint } else { t.danger.gamma_multiply(0.45) };
        let pts: Vec<Pos2> = points.iter().map(|p| xf.to_screen(p[0] as f32, p[1] as f32)).collect();
        if pts.len() == 1 {
            painter.circle_filled(pts[0], w / 2.0, c);
        }
        painter.add(egui::Shape::line(pts, Stroke::new(w, c)));
    }
    // The frame: where the drag puts it, else where it is.
    let frame = app.ui.generate.drag.as_ref().and_then(dragged_frame).unwrap_or_else(|| app.session.ai.generate.frame_for(&doc));
    outline(painter, xf, frame, Stroke::new(3.0, t.shadow));
    outline(painter, xf, frame, Stroke::new(1.5, t.accent));
    if app.ui.tool == Tool::GenFrame && app.ui.generate.mode == GenMode::Frame {
        for (hx, hy) in [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0), (0.0, 1.0), (0.0, 0.5)] {
            let p = xf.to_screen(frame.x0 as f32 + frame.width() as f32 * hx, frame.y0 as f32 + frame.height() as f32 * hy);
            let r = egui::Rect::from_center_size(p, vec2(7.0, 7.0));
            painter.rect_filled(r, 1.0, t.card);
            painter.rect_stroke(r, 1.0, Stroke::new(1.0, t.accent), egui::StrokeKind::Inside);
        }
    }
    let label = format!("{} × {}", frame.width(), frame.height());
    let at = xf.to_screen(frame.x0 as f32, frame.y0 as f32) + vec2(0.0, -16.0);
    let galley = painter.layout_no_wrap(label, egui::FontId::proportional(11.0), t.primary_text);
    let bg = egui::Rect::from_min_size(at, galley.size() + vec2(8.0, 2.0));
    painter.rect_filled(bg, 3.0, t.accent);
    painter.galley(at + vec2(4.0, 1.0), galley, t.primary_text);
    // The brush outline under the pointer.
    if app.ui.tool == Tool::GenFrame
        && app.ui.generate.mode != GenMode::Frame
        && let Some(pos) = ctx.pointer_hover_pos().filter(|p| xf.rect.contains(*p))
    {
        painter.circle_stroke(pos, (app.ui.generate.brush_size * xf.zoom / 2.0).max(2.0), Stroke::new(1.0, t.text));
    }
}

fn refresh(app: &mut PhotokaApp, now: f64) {
    if app.ui.generate.refreshed_at.is_some_and(|t| now - t < 1.0) {
        return;
    }
    let installed = app.session.execute("ai.models.list", json!({})).unwrap_or(Value::Null);
    let comfy = app.session.execute("ai.comfy.status", json!({})).unwrap_or(Value::Null);
    let g = &mut app.ui.generate;
    g.installed = installed;
    g.comfy = comfy;
    g.refreshed_at = Some(now);
}

/// Installed files of `kind` that the built-in engine can use, with their family.
fn installed(g: &GenerateUi, kind: &str) -> Vec<(String, String)> {
    g.installed
        .get("models")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter(|m| m.get("kind").and_then(Value::as_str) == Some(kind) && m.get("problem").is_none_or(Value::is_null))
                .filter_map(|m| Some((m.get("file_name")?.as_str()?.to_string(), m.get("family").and_then(Value::as_str).unwrap_or_default().to_string())))
                .collect()
        })
        .unwrap_or_default()
}

/// What the panel asked for this frame.
enum Act {
    Run(&'static str, Value),
    Generate,
    Cancel(photoka_engine::jobs::JobId),
    ModelManager,
    Close,
}

pub fn windows(app: &mut PhotokaApp, ctx: &egui::Context) {
    if !app.ui.generate.open {
        return;
    }
    let now = ctx.input(|i| i.time);
    refresh(app, now);
    let job = running_job(app);
    let cjob = comfy_job(app);
    if job.is_some() || cjob.is_some() {
        ctx.request_repaint_after(std::time::Duration::from_millis(200));
    }
    let t = Tokens::get(ctx);
    let mut g = std::mem::take(&mut app.ui.generate);
    let doc = app.session.active().map(|d| d.doc.clone());
    let frame = doc.as_ref().map(|d| app.session.ai.generate.frame_for(d));
    let masked = doc.as_ref().and_then(|d| app.session.ai.generate.mask_for(d.id)).is_some();
    let native_ok = photoka_ml::system::NATIVE_SUPPORTED;
    let checkpoints = installed(&g, "checkpoint");
    let loras = installed(&g, "lora");
    let loaded = app.session.ai.generate.loaded_model();
    let mut acts: Vec<Act> = Vec::new();
    crate::analysis_ui::panel_window(app, ctx, "generate-panel", tl!("Generate"), vec2(-40.0, 40.0), 340.0, |ui| {
        if crate::analysis_ui::title_row(ui, tl!("Generate")) {
            acts.push(Act::Close);
        }
        ui.horizontal(|ui| {
            ui.label(RichText::new(tl!("Engine")).color(t.text_dim).size(11.5));
            for (e, label) in [(GenEngine::Native, tl!("Built-in")), (GenEngine::Comfyui, "ComfyUI")] {
                if ui.selectable_label(g.engine == e, label).clicked() {
                    g.engine = e;
                }
            }
        });
        if g.engine == GenEngine::Native && !native_ok {
            ui.label(RichText::new(tl!("The built-in engine needs a Mac with Apple silicon (M1 or later); use ComfyUI instead.")).color(t.warning).size(11.5));
        }
        if g.engine == GenEngine::Comfyui {
            comfy_section(ui, &t, &mut g, cjob.as_ref(), &mut acts);
        }
        ui.add_space(2.0);
        // Model.
        ui.horizontal(|ui| {
            ui.label(RichText::new(tl!("Model")).color(t.text_dim).size(11.5));
            let shown = if g.model.is_empty() { tl!("Choose a model").to_string() } else { g.model.clone() };
            egui::ComboBox::from_id_salt("gen-model").width(230.0).selected_text(shown).show_ui(ui, |ui| {
                for (file, family) in &checkpoints {
                    let label = if family.is_empty() { file.clone() } else { format!("{file}  ·  {family}") };
                    ui.selectable_value(&mut g.model, file.clone(), label);
                }
            });
        });
        if checkpoints.is_empty() {
            ui.horizontal(|ui| {
                ui.label(RichText::new(tl!("No models installed yet.")).color(t.text_faint).size(11.5));
                if ui.small_button(tl!("Model Manager…")).clicked() {
                    acts.push(Act::ModelManager);
                }
            });
        } else if loaded.as_deref() == Some(g.model.as_str()) {
            ui.label(RichText::new(tl!("Loaded and ready")).color(t.text_faint).size(10.5));
        }
        ui.add_space(2.0);
        ui.label(RichText::new(tl!("Prompt")).color(t.text_dim).size(11.5));
        ui.add(egui::TextEdit::multiline(&mut g.prompt).desired_rows(3).desired_width(f32::INFINITY).hint_text(tl!("Describe the image…")));
        ui.label(RichText::new(tl!("Negative prompt")).color(t.text_dim).size(11.5));
        ui.add(egui::TextEdit::multiline(&mut g.negative).desired_rows(2).desired_width(f32::INFINITY).hint_text(tl!("What to avoid…")));
        ui.add_space(2.0);
        // Sampling.
        ui.horizontal(|ui| {
            ui.label(RichText::new(crate::i18n::tr_ctx(crate::i18n::current(), "ai", "Sampler")).color(t.text_dim).size(11.5));
            let current = photoka_ml::params::Sampler::parse(&g.sampler).map_or(g.sampler.clone(), |s| s.label().to_string());
            egui::ComboBox::from_id_salt("gen-sampler").width(160.0).selected_text(current).show_ui(ui, |ui| {
                for s in photoka_ml::params::Sampler::ALL {
                    ui.selectable_value(&mut g.sampler, s.as_str().to_string(), s.label());
                }
            });
        });
        let mut steps = g.steps as f32;
        if crate::widgets::slider_row(ui, tl!("Steps"), &mut steps, 1.0..=60.0, "", None).changed() {
            g.steps = steps.round().clamp(1.0, 150.0) as u32;
        }
        crate::widgets::slider_row(ui, tl!("Guidance (CFG)"), &mut g.cfg, 0.0..=20.0, "", None);
        let mut strength = g.strength * 100.0;
        if crate::widgets::slider_row(ui, tl!("Strength"), &mut strength, 0.0..=100.0, "%", None)
            .on_hover_text(tl!("How much of the frame's pixels to replace: 100% ignores them; lower keeps more of the original."))
            .changed()
        {
            g.strength = (strength / 100.0).clamp(0.0, 1.0);
        }
        ui.horizontal(|ui| {
            ui.label(RichText::new(tl!("Seed")).color(t.text_dim).size(11.5));
            ui.add_enabled(!g.random_seed, egui::DragValue::new(&mut g.seed).range(0..=u64::from(u32::MAX)));
            if crate::icons::button(ui, "sparkles", 22.0, g.random_seed, tl!("New seed for every image")).clicked() {
                g.random_seed = !g.random_seed;
            }
        });
        // LoRAs.
        crate::widgets::hairline(ui);
        ui.horizontal(|ui| {
            ui.label(RichText::new(tl!("LoRAs")).color(t.text_dim).size(11.5));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let max = if g.engine == GenEngine::Native { photoka_ml::params::MAX_NATIVE_LORAS } else { photoka_ml::params::MAX_LORAS };
                if ui.add_enabled(!loras.is_empty() && g.loras.len() < max, egui::Button::new(tl!("Add")).small()).clicked()
                    && let Some((file, _)) = loras.first()
                {
                    g.loras.push(LoraSlot { file: file.clone(), weight: 1.0 });
                }
                if ui.small_button(tl!("Fast (LCM)")).on_hover_text(tl!("Add the LCM LoRA and switch to 6 steps, CFG 1.5 and the LCM sampler")).clicked() {
                    acts.push(Act::Run("gen.fast", json!({})));
                }
            });
        });
        let mut remove = None;
        for (i, slot) in g.loras.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt(("gen-lora", i)).width(170.0).selected_text(slot.file.clone()).show_ui(ui, |ui| {
                    for (file, _) in &loras {
                        ui.selectable_value(&mut slot.file, file.clone(), file);
                    }
                });
                ui.add(egui::DragValue::new(&mut slot.weight).range(-2.0..=2.0).speed(0.01).fixed_decimals(2));
                if crate::icons::button(ui, "x", 18.0, false, tl!("Remove")).clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            g.loras.remove(i);
        }
        if loras.is_empty() && g.loras.is_empty() {
            ui.label(RichText::new(tl!("No LoRAs installed.")).color(t.text_faint).size(10.5));
        }
        // Frame and mask.
        crate::widgets::hairline(ui);
        if let Some(f) = frame {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{} {} × {}", tl!("Frame"), f.width(), f.height())).color(t.text_dim).size(11.5));
                ui.label(RichText::new(if masked { tl!("· inpainting the masked area") } else { tl!("· whole frame") }).color(t.text_faint).size(11.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if masked && ui.small_button(tl!("Clear Mask")).clicked() {
                        acts.push(Act::Run("ai.mask.clear", json!({})));
                    }
                });
            });
        } else {
            ui.label(RichText::new(tl!("Open or create a document to generate into it.")).color(t.text_faint).size(11.5));
        }
        ui.add_space(4.0);
        // Generate / progress.
        match &job {
            Some(j) => {
                let (r, _) = ui.allocate_exact_size(vec2(ui.available_width(), 6.0), egui::Sense::hover());
                crate::jobs_ui::bar(ui, r, (j.progress > 0.0).then_some(j.progress), &t);
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&j.message).color(t.text_dim).size(11.0));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button(tl!("Cancel")).clicked() {
                            acts.push(Act::Cancel(j.id));
                        }
                    });
                });
            }
            None => {
                let comfy_up = g.comfy.get("running").and_then(Value::as_bool) == Some(true);
                let ready = doc.is_some() && !g.model.is_empty() && if g.engine == GenEngine::Comfyui { comfy_up } else { native_ok };
                ui.add_enabled_ui(ready, |ui| {
                    if crate::widgets::primary_button(ui, tl!("Generate"), ui.available_width()).clicked() {
                        acts.push(Act::Generate);
                    }
                });
                if !g.random_seed || g.seed != 0 {
                    ui.label(RichText::new(format!("{} {}", tl!("Seed"), g.seed)).color(t.text_faint).size(10.5));
                }
            }
        }
    });
    app.ui.generate = g;
    if !acts.is_empty() {
        app.ui.generate.refreshed_at = None;
    }
    for act in acts {
        let r = match act {
            Act::Run("gen.fast", _) => {
                fast_preset(app);
                Ok(Value::Null)
            }
            Act::Run(id, p) => app.run(id, p),
            Act::Generate => start(app),
            Act::Cancel(job) => {
                crate::jobs_ui::cancel(app, job);
                Ok(Value::Null)
            }
            Act::ModelManager => crate::model_manager_ui::menu(app, crate::model_manager_ui::MENU_ID, &json!({"tab": "suggested"})).unwrap_or(Ok(Value::Null)),
            Act::Close => {
                app.ui.generate.open = false;
                Ok(Value::Null)
            }
        };
        if let Err(e) = r {
            crate::notices::error(app, e);
        }
    }
}

/// Fast (LCM): the LCM LoRA (when installed), 6 steps, CFG 1.5 and the LCM sampler.
pub fn fast_preset(app: &mut PhotokaApp) {
    const LCM: &str = "lcm-lora-sdv1-5.safetensors";
    let has = installed(&app.ui.generate, "lora").iter().any(|(f, _)| f == LCM);
    let g = &mut app.ui.generate;
    if has && !g.loras.iter().any(|l| l.file == LCM) {
        g.loras.push(LoraSlot { file: LCM.into(), weight: 1.0 });
    }
    g.sampler = "lcm".into();
    g.steps = 6;
    g.cfg = 1.5;
    if !has {
        crate::notices::error(app, tl!("Fast mode needs the LCM LoRA: download it from Model Manager › Suggested.").to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::tool_event;
    use photoka_geom::Rect as DocRect;

    fn drag(app: &mut PhotokaApp, from: [f64; 2], to: [f64; 2]) {
        let m = egui::Modifiers::default();
        tool_event(app, ToolEvent::Down { x: from[0], y: from[1], pressure: 1.0 }, m);
        tool_event(app, ToolEvent::Move { x: to[0], y: to[1], pressure: 1.0 }, m);
        tool_event(app, ToolEvent::Up { x: to[0], y: to[1] }, m);
    }

    fn app() -> PhotokaApp {
        let mut app = PhotokaApp::new(photoka_engine::Session::new(), crate::Services::default());
        app.session.ai.models_dir = std::env::temp_dir().join(format!("photoka-gen-ui-{}", std::process::id()));
        app.session.ai.comfy.dir = std::env::temp_dir().join(format!("photoka-gen-ui-comfy-{}", std::process::id()));
        app.run("file.new", json!({"width": 1024, "height": 768})).unwrap();
        app
    }

    fn frame(app: &PhotokaApp) -> DocRect {
        let doc = app.session.active().map(|d| d.doc.clone()).unwrap();
        app.session.ai.generate.frame_for(&doc)
    }

    #[test]
    fn frame_drags_snap_to_64_and_keep_the_fixed_edge() {
        let rect = DocRect::new(256, 128, 768, 640);
        let moved = dragged_frame(&GenDrag::Frame { start: [300.0, 300.0], cur: [364.0, 332.0], rect, handle: (0, 0) });
        assert_eq!(moved, Some(DocRect::new(320, 160, 832, 672)));
        let right = dragged_frame(&GenDrag::Frame { start: [768.0, 400.0], cur: [900.0, 400.0], rect, handle: (1, 0) });
        assert_eq!(right, Some(DocRect::new(256, 128, 896, 640)));
        let left = dragged_frame(&GenDrag::Frame { start: [256.0, 400.0], cur: [200.0, 400.0], rect, handle: (-1, 0) }).unwrap();
        assert_eq!((left.x1, left.width()), (768, 576));
        let tiny = dragged_frame(&GenDrag::Frame { start: [768.0, 640.0], cur: [300.0, 200.0], rect, handle: (1, 1) }).unwrap();
        assert_eq!((tiny.width(), tiny.height()), (64, 64));
        let new = dragged_frame(&GenDrag::New { start: [10.0, 10.0], cur: [300.0, 200.0] }).unwrap();
        assert_eq!(new, DocRect::from_xywh(10, 10, 320, 192));
        assert_eq!(dragged_frame(&GenDrag::New { start: [10.0, 10.0], cur: [12.0, 13.0] }), None);
        assert_eq!(handle_at(rect, 256.0, 128.0, 4.0), Some((-1, -1)));
        assert_eq!(handle_at(rect, 500.0, 400.0, 4.0), Some((0, 0)));
        assert_eq!(handle_at(rect, 10.0, 10.0, 4.0), None);
    }

    #[test]
    fn the_tool_moves_resizes_and_masks() {
        let mut app = app();
        let ctx = egui::Context::default();
        crate::menus::invoke(&mut app, &ctx, MENU_PANEL, json!({})).unwrap();
        assert!(app.ui.generate.open && app.ui.tool == Tool::GenFrame);
        crate::menus::invoke(&mut app, &ctx, MENU_PANEL, json!({"engine": "comfyui"})).unwrap();
        assert!(app.ui.generate.open && app.ui.generate.engine == GenEngine::Comfyui);
        assert!(crate::menus::invoke(&mut app, &ctx, MENU_PANEL, json!({"engine": "dall-e"})).is_err());
        assert!(crate::menus::is_live(MENU_PANEL) && crate::menus::is_live(MENU_RUN));
        assert_eq!(frame(&app), DocRect::new(256, 128, 768, 640));
        drag(&mut app, [400.0, 400.0], [336.0, 336.0]);
        assert_eq!(frame(&app), DocRect::new(192, 64, 704, 576));
        drag(&mut app, [704.0, 300.0], [900.0, 300.0]);
        assert_eq!(frame(&app).width(), 704);
        app.ui.generate.mode = GenMode::Mask;
        drag(&mut app, [300.0, 300.0], [500.0, 300.0]);
        let id = app.session.active().map(|d| d.doc.id).unwrap();
        let mask = app.session.ai.generate.mask_for(id).expect("the stroke painted a mask");
        assert!(mask.value(400, 300) > 0 && mask.value(400, 500) == 0);
        app.ui.generate.mode = GenMode::Erase;
        drag(&mut app, [250.0, 300.0], [550.0, 300.0]);
        let erased = app.session.ai.generate.mask_for(id).map_or(0, |m| m.value(400, 300));
        assert_eq!(erased, 0);
        // Without a model, Generate in Frame opens the panel rather than failing.
        app.ui.generate.open = false;
        let r = crate::menus::invoke(&mut app, &ctx, MENU_RUN, json!({})).unwrap();
        assert_eq!(r["started"], false);
        assert!(app.ui.generate.open);
    }

    #[test]
    fn the_panel_and_overlay_draw_without_models() {
        let mut app = app();
        let ctx = egui::Context::default();
        app.ui.generate.open = true;
        app.ui.tool = Tool::GenFrame;
        for engine in [GenEngine::Native, GenEngine::Comfyui] {
            app.ui.generate.engine = engine;
            crate::analysis_ui::tests::render(&mut app, &ctx, windows);
        }
        assert!(app.ui.generate.comfy.get("installed").is_some(), "the panel polls ComfyUI's status");
        app.ui.generate.comfy_confirm = true;
        crate::analysis_ui::tests::render(&mut app, &ctx, windows);
        app.ui.generate.model = "missing.safetensors".into();
        assert!(start(&mut app).is_err() || running_job(&app).is_some());
        let p = params(&app.ui.generate);
        assert_eq!(p["model"], "missing.safetensors");
        assert!(p["steps"].as_u64().is_some_and(|s| s > 0));
    }
}

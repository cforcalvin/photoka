//! Offline image generation in the shell (Photoka extension, Draw Things-style): the Generate
//! Frame tool (move and resize the frame; paint or erase the inpaint mask), the frame, mask and
//! live-preview overlay on the canvas, and Window › Generate, the panel with the prompt and
//! settings. Every change is an engine `ai.*` command, so agents drive the same thing; the
//! generation itself is a background job (`ai.generate`) that adds a layer.

use egui::{Color32, CornerRadius, Pos2, RichText, Stroke, TextureHandle, vec2};
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
    pub engine: GenEngine,
    pub mode: GenMode,
    pub model: String,
    pub prompt: String,
    pub negative: String,
    pub sampler: String,
    pub steps: u32,
    pub cfg: f32,
    pub strength: f32,
    /// The strength the frame's own prompt bar generates with (it keeps more of the image).
    pub bar_strength: f32,
    pub seed: u64,
    /// A new seed for every generation (the dice); off keeps [`Self::seed`].
    pub random_seed: bool,
    pub loras: Vec<LoraSlot>,
    /// Mask brush size and hardness (document pixels, 0…1).
    pub brush_size: f32,
    pub brush_hardness: f32,
    /// Lock frame to view (Draw Things style): the frame stays put on screen while the image
    /// zooms and pans under it, and always generates at [`Self::gen_size`].
    pub lock_to_view: bool,
    /// The locked frame's generation size (sides are multiples of 64).
    pub gen_size: [u32; 2],
    #[serde(skip)]
    pub drag: Option<GenDrag>,
    /// The locked frame last computed, and since when (it's sent once the view settles).
    #[serde(skip)]
    pub settling: Option<(photoka_geom::Rect, [u32; 2], f64)>,
    /// The egui pass the Generate tab was last drawn in (the frame shows while it's on screen).
    #[serde(skip)]
    pub drawn_pass: Option<u64>,
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
            engine: GenEngine::Native,
            mode: GenMode::Frame,
            model: String::new(),
            prompt: String::new(),
            negative: String::new(),
            sampler: "dpmpp_2m_karras".into(),
            steps: 12,
            cfg: 7.0,
            strength: 1.0,
            bar_strength: 0.5,
            seed: 0,
            random_seed: true,
            loras: Vec::new(),
            brush_size: 48.0,
            brush_hardness: 0.8,
            lock_to_view: true,
            gen_size: [512, 512],
            drag: None,
            settling: None,
            drawn_pass: None,
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
    (id == MENU_PANEL).then(|| tab_open(app))
}

/// The Generate tab's index in the Layers | Channels | Paths | Generate group.
pub const TAB: usize = 3;
/// The Generate tab's panel id in a customised workspace.
const PANEL_ID: &str = "generate";

/// Whether the Generate tab is the one showing in its dock group (or, in a customised
/// workspace, is open anywhere).
pub fn tab_open(app: &PhotokaApp) -> bool {
    if let Some(layout) = &app.ui.docking {
        return layout.contains(&PANEL_ID.to_string());
    }
    let pro = app.ui.theme.is_pro();
    app.ui.panels.layers && app.ui.dock_tabs.layers == TAB && app.ui.dock.visible_tabs(crate::dock::Group::Layers, pro).iter().any(|(i, _)| *i == TAB)
}

/// Bring the Generate tab up, or (`false`) put it away: back to the Layers tab, or closed in a
/// customised workspace.
pub fn show_tab(app: &mut PhotokaApp, show: bool) -> Result<(), String> {
    app.ui.generate.refreshed_at = None;
    if app.ui.docking.is_some() {
        let id = format!("window.panel.{PANEL_ID}");
        return crate::panel_docking::command(app, &id, &json!({"show": show})).unwrap_or(Ok(Value::Null)).map(|_| ());
    }
    if show {
        app.ui.dock_tabs.layers = TAB;
        crate::dock::reveal(app, crate::dock::Group::Layers);
    } else if app.ui.dock_tabs.layers == TAB {
        app.ui.dock_tabs.layers = 0;
    }
    Ok(())
}

/// Whether the Generate tab was drawn in this pass or the last one.
fn tab_drawn(g: &GenerateUi, ctx: &egui::Context) -> bool {
    g.drawn_pass.is_some_and(|p| p.saturating_add(1) >= ctx.cumulative_pass_nr())
}

pub fn enabled(app: &PhotokaApp, id: &str) -> Option<bool> {
    match id {
        MENU_PANEL => Some(true),
        MENU_RUN => Some(app.session.active().is_some() && !app.session.ai.generate.busy()),
        _ => None,
    }
}

/// Window › Generate toggles the Generate tab (`{"open": bool}` sets it); Edit › Generate in
/// Frame… generates with the panel's settings, or opens the tab when no model is chosen yet.
pub fn menu(app: &mut PhotokaApp, id: &str, params: &Value) -> Option<Result<Value, String>> {
    match id {
        MENU_PANEL => {
            let engine = match params.get("engine").and_then(Value::as_str) {
                None => None,
                Some("native") => Some(GenEngine::Native),
                Some("comfyui") => Some(GenEngine::Comfyui),
                Some(e) => return Some(Err(format!("unknown engine `{e}` (native, comfyui)"))),
            };
            if let Some(e) = engine {
                app.ui.generate.engine = e;
            }
            let open = match params.get("open") {
                None => engine.is_some() || !tab_open(app),
                Some(v) => match v.as_bool() {
                    Some(b) => b,
                    None => return Some(Err("open must be a boolean".into())),
                },
            };
            if let Err(e) = show_tab(app, open) {
                return Some(Err(e));
            }
            if open && app.session.active().is_some() {
                app.ui.tool = Tool::GenFrame;
            }
            Some(Ok(json!({"open": open})))
        }
        MENU_RUN => {
            if app.ui.generate.model.is_empty() {
                return Some(show_tab(app, true).map(|()| json!({"open": true, "started": false})));
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
    start_with(app, None)
}

/// Start a generation with the panel's settings, at `strength` (0…1) when given.
fn start_with(app: &mut PhotokaApp, strength: Option<f32>) -> Result<Value, String> {
    sync_locked(app)?;
    if app.ui.generate.random_seed {
        app.ui.generate.seed = random_seed();
    }
    let mut p = params(&app.ui.generate);
    if let Some(s) = strength {
        p["strength"] = json!(s.clamp(0.0, 1.0));
    }
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

/// The largest share of the canvas the locked frame takes on screen.
const LOCKED_FIT: f32 = 0.85;
/// How long the view must rest before a moved locked frame is sent to the engine (seconds).
const SETTLE: f64 = 0.15;
/// The generation sizes on offer for a locked frame.
pub const GEN_SIZES: [[u32; 2]; 7] = [[512, 512], [640, 640], [768, 768], [1024, 1024], [512, 768], [768, 512], [576, 1024]];

/// A locked frame: its document rect, and how far it's off the document sides it can have
/// (−1: zoomed in too far to fit [`MIN_FRAME_SIDE`], +1: zoomed out past [`MAX_FRAME_SIDE`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Locked {
    pub rect: photoka_geom::Rect,
    pub limit: i8,
}

/// With Lock frame to view: the frame centred on the canvas, as many screen points as the
/// generation size has pixels (scaled down to fit the canvas), in document pixels.
pub fn locked_frame(g: &GenerateUi, xf: &ViewXform) -> Option<Locked> {
    use photoka_engine::ai_gen_cmds::{MAX_FRAME_SIDE, MIN_FRAME_SIDE};
    let [gw, gh] = g.gen_size.map(|v| v.max(1) as f32);
    let (cw, ch) = (xf.rect.width(), xf.rect.height());
    let (zx, zy) = (xf.zoom_x(), xf.zoom);
    if !(cw > 1.0 && ch > 1.0 && zx.is_finite() && zy.is_finite() && zx > 0.0 && zy > 0.0) {
        return None;
    }
    let k = (LOCKED_FIT * cw / gw).min(LOCKED_FIT * ch / gh).min(1.0);
    let (w, h) = (f64::from(gw * k / zx).round(), f64::from(gh * k / zy).round());
    let (min, max) = (f64::from(MIN_FRAME_SIDE), f64::from(MAX_FRAME_SIDE));
    let limit = if w.min(h) < min {
        -1
    } else if w.max(h) > max {
        1
    } else {
        0
    };
    let (w, h) = (w.clamp(min, max), h.clamp(min, max));
    let (cx, cy) = (f64::from(xf.center[0]), f64::from(xf.center[1]));
    if !(cx.is_finite() && cy.is_finite() && cx.abs() < 1e6 && cy.abs() < 1e6) {
        return None;
    }
    let rect = photoka_geom::Rect::from_xywh((cx - w / 2.0).round() as i32, (cy - h / 2.0).round() as i32, w as u32, h as u32);
    Some(Locked { rect, limit })
}

/// The locked frame the active view wants (`None` when unlocked, or with no canvas yet).
fn wanted(app: &PhotokaApp) -> Option<Locked> {
    if !app.ui.generate.lock_to_view {
        return None;
    }
    locked_frame(&app.ui.generate, &ViewXform::active(app)?)
}

/// Send the locked frame to the engine now if it moved (before generating or filling the mask).
pub fn sync_locked(app: &mut PhotokaApp) -> Result<(), String> {
    match wanted(app) {
        Some(l) => send_locked(app, l.rect),
        None => Ok(()),
    }
}

fn send_locked(app: &mut PhotokaApp, rect: photoka_geom::Rect) -> Result<(), String> {
    let Some(doc) = app.session.active().map(|d| d.doc.clone()) else { return Ok(()) };
    let [gw, gh] = app.ui.generate.gen_size;
    let st = &app.session.ai.generate;
    if st.frame_for(&doc) == rect && st.gen_size_for(&doc) == (gw, gh) {
        return Ok(());
    }
    app.run("ai.frame.set", json!({"x": rect.x0, "y": rect.y0, "width": rect.width(), "height": rect.height(), "genWidth": gw, "genHeight": gh})).map(|_| ())
}

/// Turn Lock frame to view on or off. Off leaves a frame of the generation size on the document,
/// centred where the locked one was.
pub fn set_lock(app: &mut PhotokaApp, on: bool) -> Result<(), String> {
    if on {
        app.ui.generate.lock_to_view = true;
        return sync_locked(app);
    }
    let was = wanted(app);
    app.ui.generate.lock_to_view = false;
    app.ui.generate.settling = None;
    let Some(l) = was else { return Ok(()) };
    let [gw, gh] = app.ui.generate.gen_size;
    let (cx, cy) = (l.rect.x0 + l.rect.width() as i32 / 2, l.rect.y0 + l.rect.height() as i32 / 2);
    app.run("ai.frame.set", json!({"x": cx - gw as i32 / 2, "y": cy - gh as i32 / 2, "width": gw, "height": gh})).map(|_| ())
}

/// Whether a drag on the canvas pans the view (the locked frame stays put; the image moves).
pub fn pans_view(app: &PhotokaApp) -> bool {
    app.ui.tool == Tool::GenFrame && app.ui.generate.lock_to_view && app.ui.generate.mode == GenMode::Frame && app.session.active().is_some()
}

/// The Frame, Mask and Erase-mask-paint buttons (the options bar and the bar above the frame).
fn mode_buttons(ui: &mut egui::Ui, g: &mut GenerateUi, size: f32) {
    for (mode, icon, tip) in [
        (GenMode::Frame, "frame-placeholder", if g.lock_to_view { tl!("Zoom and pan the image under the frame") } else { tl!("Move and resize the frame") }),
        (GenMode::Mask, "brush", tl!("Paint what to regenerate")),
        (GenMode::Erase, "brush-cleaning", tl!("Erase mask paint")),
    ] {
        if crate::icons::button(ui, icon, size, g.mode == mode, tip).clicked() {
            g.mode = mode;
        }
    }
}

/// Pointer input for the Generate Frame tool. Returns true when the event was its.
pub fn pointer(app: &mut PhotokaApp, ev: ToolEvent, _mods: egui::Modifiers) -> bool {
    if app.ui.tool != Tool::GenFrame {
        return false;
    }
    // A locked frame doesn't move: the canvas pans instead (`pans_view`).
    if pans_view(app) {
        return true;
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
    mode_buttons(ui, g, 24.0);
    crate::widgets::vline(ui, 22.0);
    let mut lock = g.lock_to_view;
    let lock_changed = ui
        .checkbox(&mut lock, tl!("Lock frame to view"))
        .on_hover_text(tl!("The frame stays put on screen while you zoom and pan the image under it, and always generates at its size"))
        .changed();
    if g.lock_to_view {
        let label = |s: [u32; 2]| format!("{} × {}", s[0], s[1]);
        egui::ComboBox::from_id_salt("gen-size").selected_text(label(g.gen_size)).width(96.0).show_ui(ui, |ui| {
            for s in GEN_SIZES {
                ui.selectable_value(&mut g.gen_size, s, label(s));
            }
        });
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
    if crate::icons::button(ui, "eraser", 24.0, false, tl!("Remove the masked object")).clicked() {
        run = Some("ai.removeMasked");
    }
    if crate::icons::button(ui, "x", 24.0, false, tl!("Clear Mask")).clicked() {
        run = Some("ai.mask.clear");
    }
    if !g.lock_to_view && ui.small_button(tl!("Reset Frame")).clicked() {
        run = Some("ai.frame.reset");
    }
    crate::widgets::vline(ui, 22.0);
    let show = ui.small_button(tl!("Generate…")).clicked();
    let mut r = if lock_changed { set_lock(app, lock) } else { Ok(()) };
    if show {
        r = r.and_then(|()| show_tab(app, true));
    }
    if let Some(id) = run {
        r = r.and_then(|()| sync_locked(app)).and_then(|()| app.run(id, json!({})).map(|_| ()));
    }
    if let Err(e) = r {
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

/// Dab centres for a mask stroke preview, matching `ai.mask.stroke` (15% of size, at least 0.5 px).
fn stroke_dabs(points: &[[f64; 2]], size: f64) -> Vec<[f64; 2]> {
    let spacing = (size * 0.15).max(0.5);
    let mut out = Vec::new();
    let mut prev: Option<[f64; 2]> = None;
    for &p in points {
        match prev {
            None => out.push(p),
            Some(q) => {
                let len = (p[0] - q[0]).hypot(p[1] - q[1]);
                let n = (len / spacing).ceil().min(1e6) as usize;
                let n = n.max(1);
                for i in 1..=n {
                    let t = i as f64 / n as f64;
                    out.push([q[0] + (p[0] - q[0]) * t, q[1] + (p[1] - q[1]) * t]);
                }
            }
        }
        prev = Some(p);
    }
    out
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
    let shown = app.ui.tool == Tool::GenFrame || tab_drawn(&app.ui.generate, painter.ctx());
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
        let rev = gen_state.mask_revision();
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
    // A stroke being painted: round dabs, same centres as `ai.mask.stroke` (not a fat line).
    if let Some(GenDrag::Stroke { points }) = &app.ui.generate.drag {
        let radius = (app.ui.generate.brush_size * xf.zoom / 2.0).max(0.5);
        let c = if app.ui.generate.mode == GenMode::Erase { t.text_faint } else { t.danger.gamma_multiply(0.45) };
        for [x, y] in stroke_dabs(points, f64::from(app.ui.generate.brush_size)) {
            painter.circle_filled(xf.to_screen(x as f32, y as f32), radius, c);
        }
    }
    // The frame: locked to the view, else where the drag puts it, else where it is.
    let locked = if app.ui.generate.lock_to_view { locked_frame(&app.ui.generate, xf) } else { None };
    if let Some(l) = locked {
        settle(app, &ctx, l.rect);
    }
    let frame = match locked {
        Some(l) => l.rect,
        None => app.ui.generate.drag.as_ref().and_then(dragged_frame).unwrap_or_else(|| app.session.ai.generate.frame_for(&doc)),
    };
    outline(painter, xf, frame, Stroke::new(3.0, t.shadow));
    outline(painter, xf, frame, Stroke::new(1.5, t.accent));
    if app.ui.tool == Tool::GenFrame && app.ui.generate.mode == GenMode::Frame && locked.is_none() {
        for (hx, hy) in [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0), (0.0, 1.0), (0.0, 0.5)] {
            let p = xf.to_screen(frame.x0 as f32 + frame.width() as f32 * hx, frame.y0 as f32 + frame.height() as f32 * hy);
            let r = egui::Rect::from_center_size(p, vec2(7.0, 7.0));
            painter.rect_filled(r, 1.0, t.card);
            painter.rect_stroke(r, 1.0, Stroke::new(1.0, t.accent), egui::StrokeKind::Inside);
        }
    }
    let [gw, gh] = app.ui.generate.gen_size;
    let label = match locked {
        Some(l) if l.limit < 0 => tl!("Zoom out to fit the frame").to_string(),
        Some(l) if l.limit > 0 => tl!("Zoom in to fit the frame").to_string(),
        Some(_) if (gw, gh) != (frame.width(), frame.height()) => crate::i18n::fmt(
            tl!("{w} × {h} from {fw} × {fh} px"),
            &[("w", &gw.to_string()), ("h", &gh.to_string()), ("fw", &frame.width().to_string()), ("fh", &frame.height().to_string())],
        ),
        _ => format!("{} × {}", frame.width(), frame.height()),
    };
    let screen = xf.doc_rect(frame);
    let at = screen.left_top() + vec2(0.0, -18.0);
    let galley = painter.layout_no_wrap(label, egui::FontId::proportional(11.0), t.primary_text);
    let bg = egui::Rect::from_min_size(at, galley.size() + vec2(8.0, 2.0));
    painter.rect_filled(bg, 3.0, t.accent);
    painter.galley(at + vec2(4.0, 1.0), galley, t.primary_text);
    if app.ui.tool == Tool::GenFrame {
        frame_bar(app, &ctx, &t, xf.rect, screen);
    }
    // The brush outline under the pointer.
    if app.ui.tool == Tool::GenFrame
        && app.ui.generate.mode != GenMode::Frame
        && let Some(pos) = ctx.pointer_hover_pos().filter(|p| xf.rect.contains(*p))
    {
        painter.circle_stroke(pos, (app.ui.generate.brush_size * xf.zoom / 2.0).max(2.0), Stroke::new(1.0, t.text));
    }
}

/// Send a moved locked frame to the engine once the view has rested for [`SETTLE`] seconds, so
/// zooming doesn't run a command every frame.
fn settle(app: &mut PhotokaApp, ctx: &egui::Context, rect: photoka_geom::Rect) {
    let now = ctx.input(|i| i.time);
    let size = app.ui.generate.gen_size;
    match app.ui.generate.settling {
        Some((r, s, since)) if r == rect && s == size => {
            if now - since >= SETTLE {
                if let Err(e) = send_locked(app, rect) {
                    log::warn!("locked generation frame: {e}");
                }
            } else {
                ctx.request_repaint_after(std::time::Duration::from_secs_f64(SETTLE));
            }
        }
        _ => {
            app.ui.generate.settling = Some((rect, size, now));
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(SETTLE));
        }
    }
}

/// How strongly the frame bar pulses while a generation is running (0…1 from wall time).
fn bar_pulse(time: f64) -> f32 {
    ((time * std::f64::consts::TAU * 1.25).sin() * 0.5 + 0.5) as f32
}

fn mix_rgb(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    Color32::from_rgba_unmultiplied(
        (f32::from(a.r()) + (f32::from(b.r()) - f32::from(a.r())) * t).round() as u8,
        (f32::from(a.g()) + (f32::from(b.g()) - f32::from(a.g())) * t).round() as u8,
        (f32::from(a.b()) + (f32::from(b.b()) - f32::from(a.b())) * t).round() as u8,
        a.a(),
    )
}

/// The prompt, its strength and Generate, then the Frame, Mask and Erase buttons and Clear Mask,
/// just above the frame's top-right corner (kept inside the canvas when the frame's top is off
/// screen). Generate here uses [`GenerateUi::bar_strength`]; the rest are the panel's settings.
fn frame_bar(app: &mut PhotokaApp, ctx: &egui::Context, t: &Tokens, canvas: egui::Rect, frame: egui::Rect) {
    let (mut clear, mut generate, mut remove_obj) = (false, false, false);
    let busy = app.session.ai.generate.busy();
    let pulse = if busy {
        let time = ctx.input(|i| i.time);
        ctx.request_repaint();
        bar_pulse(time)
    } else {
        0.0
    };
    let fill = if busy { mix_rgb(t.card, t.accent.gamma_multiply(0.35), pulse * 0.55) } else { t.card };
    let border = if busy { mix_rgb(t.card_border, t.accent, 0.35 + pulse * 0.65) } else { t.card_border };
    let stroke_w = if busy { 1.0 + pulse } else { 1.0 };
    // Above the size label on the frame's top edge.
    let pos = egui::pos2(frame.right(), frame.top() - 22.0);
    egui::Area::new(egui::Id::new("gen-frame-bar")).order(egui::Order::Foreground).pivot(egui::Align2::RIGHT_BOTTOM).fixed_pos(pos).constrain_to(canvas).show(
        ctx,
        |ui| {
            // Same height as the strength DragValue (theme interact size).
            let row_h = ui.spacing().interact_size.y;
            // Pill ends: radius = half the bar height (inner row + margins).
            let pill = ((row_h * 0.5) + 3.0).round().clamp(10.0, 24.0) as u8;
            let cap = pill.saturating_sub(3).max(8);
            let left_cap = CornerRadius { nw: cap, ne: 0, sw: cap, se: 0 };
            let right_cap = CornerRadius { nw: 0, ne: cap, sw: 0, se: cap };
            egui::Frame::new().fill(fill).stroke(Stroke::new(stroke_w, border)).corner_radius(CornerRadius::same(pill)).inner_margin(3.0).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 2.0;
                    let g = &mut app.ui.generate;
                    let prompt = with_corners(ui, left_cap, |ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut g.prompt)
                                .desired_width(220.0)
                                .min_size(vec2(0.0, row_h))
                                .margin(egui::Margin::symmetric(6, 2))
                                .vertical_align(egui::Align::Center)
                                .hint_text(tl!("Describe the image…")),
                        )
                    });
                    let entered = prompt.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    let mut strength = g.bar_strength * 100.0;
                    if ui
                        .add(egui::DragValue::new(&mut strength).range(0.0..=100.0).speed(1.0).suffix("%"))
                        .on_hover_text(tl!("How much of the frame's pixels to replace: 100% ignores them; lower keeps more of the original."))
                        .changed()
                    {
                        g.bar_strength = (strength / 100.0).clamp(0.0, 1.0);
                    }
                    let go = ui.add_enabled_ui(!busy, |ui| crate::icons::button(ui, "sparkles", 22.0, false, tl!("Generate"))).inner.clicked();
                    generate = (go || entered) && !busy;
                    crate::widgets::vline(ui, 18.0);
                    mode_buttons(ui, g, 22.0);
                    crate::widgets::vline(ui, 18.0);
                    remove_obj = ui.add_enabled_ui(!busy, |ui| crate::icons::button(ui, "eraser", 22.0, false, tl!("Remove the masked object"))).inner.clicked();
                    clear = with_corners(ui, right_cap, |ui| ui.small_button(tl!("Clear Mask"))).clicked();
                });
            });
        },
    );
    let r = if clear {
        app.run("ai.mask.clear", json!({})).map(|_| ())
    } else if remove_obj && !busy {
        app.run("ai.removeMasked", json!({})).map(|_| ())
    } else if generate && app.ui.generate.model.is_empty() {
        show_tab(app, true).and_then(|()| Err(tl!("Choose a model").to_string()))
    } else if generate {
        let s = app.ui.generate.bar_strength;
        start_with(app, Some(s)).map(|_| ())
    } else {
        Ok(())
    };
    if let Err(e) = r {
        crate::notices::error(app, e);
    }
}

/// Runs `add` with every widget state drawn at corner radius `cr`, leaving sizes untouched.
fn with_corners<R>(ui: &mut egui::Ui, cr: CornerRadius, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    ui.scope(|ui| {
        let w = &mut ui.visuals_mut().widgets;
        for v in [&mut w.noninteractive, &mut w.inactive, &mut w.hovered, &mut w.active, &mut w.open] {
            v.corner_radius = cr;
        }
        add(ui)
    })
    .inner
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

/// Installed files of `kind`, with their family, ordered for this Mac (Turbo first when RAM is tight).
fn installed(g: &GenerateUi, kind: &str) -> Vec<(String, String)> {
    let prefer_fast = photoka_ml::system::physical_memory().is_some_and(|b| b <= (12 << 30));
    let mut list: Vec<(String, String)> = g
        .installed
        .get("models")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter(|m| m.get("kind").and_then(Value::as_str) == Some(kind) && m.get("problem").is_none_or(Value::is_null))
                .filter_map(|m| Some((m.get("file_name")?.as_str()?.to_string(), m.get("family").and_then(Value::as_str).unwrap_or_default().to_string())))
                .collect()
        })
        .unwrap_or_default();
    list.sort_by(|a, b| {
        let rank = |fam: &str, file: &str| -> i32 {
            if kind == "upscale" {
                let n = file.to_ascii_lowercase();
                return if prefer_fast {
                    if n.contains("x2") { 0 } else if n.contains("x4") { 1 } else { 2 }
                } else if n.contains("x4") {
                    0
                } else if n.contains("x2") {
                    1
                } else {
                    2
                };
            }
            if prefer_fast {
                if fam.contains("turbo") { 0 } else { 1 }
            } else if fam == "sd15" || fam == "sd15-inpaint" {
                0
            } else {
                1
            }
        };
        rank(&a.1, &a.0).cmp(&rank(&b.1, &b.0)).then_with(|| a.0.cmp(&b.0))
    });
    list
}

/// What the panel asked for this frame.
enum Act {
    Run(&'static str, Value),
    Generate,
    Cancel(photoka_engine::jobs::JobId),
    ModelManager,
}

/// The Generate tab (Layers | Channels | Paths | Generate): engine, model, prompt, settings,
/// LoRAs, the frame and Generate.
pub fn panel(app: &mut PhotokaApp, ui: &mut egui::Ui) {
    let ctx = ui.ctx().clone();
    app.ui.generate.drawn_pass = Some(ctx.cumulative_pass_nr());
    let now = ctx.input(|i| i.time);
    refresh(app, now);
    let job = running_job(app);
    let cjob = comfy_job(app);
    if job.is_some() || cjob.is_some() {
        ctx.request_repaint_after(std::time::Duration::from_millis(200));
    }
    let t = Tokens::get(&ctx);
    let mut g = std::mem::take(&mut app.ui.generate);
    let doc = app.session.active().map(|d| d.doc.clone());
    let frame = doc.as_ref().map(|d| app.session.ai.generate.frame_for(d));
    let masked = doc.as_ref().and_then(|d| app.session.ai.generate.mask_for(d.id)).is_some();
    let native_ok = photoka_ml::system::NATIVE_SUPPORTED;
    let checkpoints = installed(&g, "checkpoint");
    let loras = installed(&g, "lora");
    let loaded = app.session.ai.generate.loaded_model();
    let mut acts: Vec<Act> = Vec::new();
    {
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
            let width = (ui.available_width() - 8.0).max(80.0);
            egui::ComboBox::from_id_salt("gen-model").width(width).selected_text(shown).truncate().show_ui(ui, |ui| {
                for (file, family) in &checkpoints {
                    let label = if family.is_empty() { file.clone() } else { format!("{file}  ·  {family}") };
                    ui.selectable_value(&mut g.model, file.clone(), label);
                }
            });
        });
        ui.horizontal(|ui| {
            if checkpoints.is_empty() {
                ui.label(RichText::new(tl!("No models installed yet.")).color(t.text_faint).size(11.5));
                if ui.small_button(tl!("Get a model for this Mac…")).clicked() {
                    acts.push(Act::ModelManager);
                }
            } else if loaded.as_deref() == Some(g.model.as_str()) {
                ui.label(RichText::new(tl!("Loaded and ready")).color(t.text_faint).size(10.5));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button(tl!("Model Manager…")).clicked() {
                    acts.push(Act::ModelManager);
                }
            });
        });
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
                let width = (ui.available_width() - 90.0).max(80.0);
                egui::ComboBox::from_id_salt(("gen-lora", i)).width(width).selected_text(slot.file.clone()).truncate().show_ui(ui, |ui| {
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
        // AI Upscale (ComfyUI + Real-ESRGAN).
        crate::widgets::hairline(ui);
        let upscalers = installed(&g, "upscale");
        let hw = photoka_ml::system::HardwareProfile::probe(std::path::Path::new(
            g.installed.get("modelsDir").and_then(Value::as_str).unwrap_or(""),
        ));
        ui.label(RichText::new(tl!("AI Upscale")).color(t.text_dim).size(11.5));
        ui.label(
            RichText::new(format!(
                "{}: {}× · {} px tiles",
                tl!("Recommended for this Mac"),
                hw.default_upscale_scale(),
                hw.default_upscale_tile()
            ))
            .color(t.text_faint)
            .size(10.5),
        );
        if upscalers.is_empty() {
            ui.horizontal(|ui| {
                ui.label(RichText::new(tl!("No upscale model installed.")).color(t.text_faint).size(11.5));
                if ui.small_button(tl!("Get a model for this Mac…")).clicked() {
                    acts.push(Act::ModelManager);
                }
            });
        } else {
            ui.horizontal(|ui| {
                let comfy_up = g.comfy.get("running").and_then(Value::as_bool) == Some(true);
                if ui
                    .add_enabled(doc.is_some() && comfy_up, egui::Button::new(tl!("Upscale…")))
                    .on_hover_text(tl!("Image › AI Upscale… — needs ComfyUI running"))
                    .clicked()
                {
                    acts.push(Act::Run(
                        "ai.upscale",
                        json!({
                            "scale": hw.default_upscale_scale(),
                            "model": upscalers.first().map(|(f, _)| f.clone()),
                            "tile": hw.default_upscale_tile(),
                        }),
                    ));
                }
                if !comfy_up {
                    ui.label(RichText::new(tl!("Start ComfyUI to upscale")).color(t.text_faint).size(10.5));
                }
            });
        }
    }
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
    fn a_stroke_preview_stamps_round_dabs_along_the_path() {
        // One point → one dab; a short hop under the spacing still gets the end dab.
        assert_eq!(stroke_dabs(&[[10.0, 10.0]], 40.0), vec![[10.0, 10.0]]);
        let short = stroke_dabs(&[[0.0, 0.0], [3.0, 0.0]], 40.0);
        assert_eq!(short, vec![[0.0, 0.0], [3.0, 0.0]]);
        // 100 px at size 40 → spacing 6 → 17 steps between ends (plus the start).
        let long = stroke_dabs(&[[0.0, 0.0], [100.0, 0.0]], 40.0);
        assert_eq!(long.first(), Some(&[0.0, 0.0]));
        assert_eq!(long.last(), Some(&[100.0, 0.0]));
        assert_eq!(long.len(), 1 + (100.0_f64 / 6.0).ceil() as usize);
        assert!(long.windows(2).all(|w| (w[1][0] - w[0][0]).abs() <= 6.0 + 1e-9));
    }

    #[test]
    fn the_tool_moves_resizes_and_masks() {
        let mut app = app();
        let ctx = egui::Context::default();
        crate::menus::invoke(&mut app, &ctx, MENU_PANEL, json!({})).unwrap();
        assert!(tab_open(&app) && app.ui.tool == Tool::GenFrame);
        assert_eq!(app.ui.dock_tabs.layers, TAB);
        assert_eq!(checked(&app, MENU_PANEL), Some(true));
        // Choosing it again puts it away, back to the Layers tab.
        crate::menus::invoke(&mut app, &ctx, MENU_PANEL, json!({})).unwrap();
        assert!(!tab_open(&app) && app.ui.dock_tabs.layers == 0);
        crate::menus::invoke(&mut app, &ctx, MENU_PANEL, json!({"engine": "comfyui"})).unwrap();
        assert!(tab_open(&app) && app.ui.generate.engine == GenEngine::Comfyui);
        assert!(crate::menus::invoke(&mut app, &ctx, MENU_PANEL, json!({"engine": "dall-e"})).is_err());
        assert!(crate::menus::invoke(&mut app, &ctx, MENU_PANEL, json!({"open": "yes"})).is_err());
        assert!(crate::menus::is_live(MENU_PANEL) && crate::menus::is_live(MENU_RUN));
        app.ui.generate.lock_to_view = false;
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
        // Without a model, Generate in Frame opens the tab rather than failing.
        show_tab(&mut app, false).unwrap();
        assert!(!tab_open(&app));
        let r = crate::menus::invoke(&mut app, &ctx, MENU_RUN, json!({})).unwrap();
        assert_eq!(r["started"], false);
        assert!(tab_open(&app));
    }

    #[test]
    fn the_generate_tab_lives_with_layers_channels_and_paths() {
        let mut app = app();
        let ctx = egui::Context::default();
        let names: Vec<&str> = app.ui.dock.visible_tabs(crate::dock::Group::Layers, true).into_iter().map(|(_, n)| n).collect();
        assert_eq!(names, ["Layers", "Channels", "Paths", "Generate"]);
        // A customised workspace opens and closes it as its own panel.
        crate::menus::invoke(&mut app, &ctx, "window.panel.activate", json!({"panel": "Generate"})).unwrap();
        assert!(app.ui.docking.is_some() && tab_open(&app));
        show_tab(&mut app, false).unwrap();
        assert!(!tab_open(&app));
        show_tab(&mut app, true).unwrap();
        assert!(tab_open(&app));
    }

    #[test]
    fn defaults_and_the_frame_bar_strength() {
        let g = GenerateUi::default();
        assert_eq!((g.sampler.as_str(), g.steps, g.cfg, g.strength, g.bar_strength), ("dpmpp_2m_karras", 12, 7.0, 1.0, 0.5));
        let mut app = app();
        app.ui.generate.model = "missing.safetensors".into();
        // The bar's strength overrides the panel's for its own Generate.
        let r = start_with(&mut app, Some(0.5));
        assert!(r.is_err() || running_job(&app).is_some());
        assert_eq!(params(&app.ui.generate)["strength"], 1.0);
    }

    #[test]
    fn the_frame_bar_pulse_oscillates() {
        let a = bar_pulse(0.0);
        let b = bar_pulse(0.2);
        let c = bar_pulse(0.4);
        assert!((0.0..=1.0).contains(&a) && (0.0..=1.0).contains(&b) && (0.0..=1.0).contains(&c));
        assert!((a - b).abs() > 0.05 || (b - c).abs() > 0.05, "pulse should move over time");
        assert_eq!(mix_rgb(Color32::BLACK, Color32::WHITE, 0.0), Color32::BLACK);
        assert_eq!(mix_rgb(Color32::BLACK, Color32::WHITE, 1.0), Color32::WHITE);
    }

    #[test]
    fn a_locked_frame_follows_the_view() {
        let mut app = app();
        app.ui.tool = Tool::GenFrame;
        app.last_canvas_rect = egui::Rect::from_min_size(Pos2::ZERO, vec2(1400.0, 1000.0));
        let id = app.session.active().map(|d| d.doc.id).unwrap();
        let set_view = |app: &mut PhotokaApp, zoom: f32, center: [f32; 2]| {
            let ppp = app.canvas_ppp();
            let v = &mut app.ui.views[0];
            (v.zoom, v.center, v.rotation, v.fit_pending, v.fill_pending) = (zoom * ppp, center, 0.0, false, false);
        };
        // 100%: 512 document pixels for 512 generated ones, centred on the view.
        set_view(&mut app, 1.0, [400.0, 300.0]);
        sync_locked(&mut app).unwrap();
        let f = frame(&app);
        assert_eq!((f.width(), f.height()), (512, 512));
        assert_eq!((f.x0 + 256, f.y0 + 256), (400, 300));
        // 400%: the same frame on screen covers 128 document pixels, still generating 512.
        set_view(&mut app, 4.0, [200.0, 150.0]);
        sync_locked(&mut app).unwrap();
        let f = frame(&app);
        let doc = app.session.active().map(|d| d.doc.clone()).unwrap();
        assert_eq!((f.width(), f.height(), f.x0, f.y0), (128, 128, 136, 86));
        assert_eq!(app.session.ai.generate.gen_size_for(&doc), (512, 512));
        // 25% at 768: 3072 document pixels.
        app.ui.generate.gen_size = [768, 768];
        set_view(&mut app, 0.25, [500.0, 400.0]);
        sync_locked(&mut app).unwrap();
        assert_eq!(frame(&app).width(), 3072);
        // A drag in Frame mode pans the view; mask strokes still paint.
        assert!(pans_view(&app));
        drag(&mut app, [400.0, 400.0], [200.0, 200.0]);
        assert_eq!(frame(&app).width(), 3072, "the frame didn't move or resize");
        app.ui.generate.mode = GenMode::Mask;
        assert!(!pans_view(&app));
        drag(&mut app, [450.0, 400.0], [550.0, 400.0]);
        assert!(app.session.ai.generate.mask_for(id).is_some());
        // Unlocking leaves a 768 frame on the document, centred where the locked one was.
        set_view(&mut app, 4.0, [400.0, 300.0]);
        set_lock(&mut app, false).unwrap();
        let f = frame(&app);
        assert_eq!((f.width(), f.height(), f.x0 + 384, f.y0 + 384), (768, 768, 400, 300));
        assert_eq!(app.session.ai.generate.gen_size_for(&doc), (768, 768));
        set_lock(&mut app, true).unwrap();
        assert_eq!(frame(&app).width(), 192);
        // Zoomed in past 8 document pixels: clamped, and the label says so.
        set_view(&mut app, 200.0, [400.0, 300.0]);
        let xf = ViewXform::active(&app).unwrap();
        let l = locked_frame(&app.ui.generate, &xf).unwrap();
        assert_eq!((l.rect.width(), l.limit), (8, -1));
        // No canvas yet: nothing to lock to.
        app.last_canvas_rect = egui::Rect::NOTHING;
        assert!(wanted(&app).is_none());
    }

    #[test]
    fn the_panel_and_overlay_draw_without_models() {
        let mut app = app();
        let ctx = egui::Context::default();
        crate::PhotokaApp::setup_context(&ctx, Default::default());
        let render = |app: &mut PhotokaApp| {
            for _ in 0..3 {
                let mut o = ctx.run_ui(Default::default(), |ui| panel(app, ui));
                o.textures_delta.clear();
            }
        };
        app.ui.tool = Tool::GenFrame;
        for engine in [GenEngine::Native, GenEngine::Comfyui] {
            app.ui.generate.engine = engine;
            render(&mut app);
        }
        assert!(app.ui.generate.comfy.get("installed").is_some(), "the panel polls ComfyUI's status");
        assert!(app.ui.generate.drawn_pass.is_some());
        app.ui.generate.comfy_confirm = true;
        render(&mut app);
        app.ui.generate.model = "missing.safetensors".into();
        assert!(start(&mut app).is_err() || running_job(&app).is_some());
        let p = params(&app.ui.generate);
        assert_eq!(p["model"], "missing.safetensors");
        assert!(p["steps"].as_u64().is_some_and(|s| s > 0));
    }
}

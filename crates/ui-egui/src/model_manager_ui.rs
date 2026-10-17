//! Window › Model Manager (Photoka extension): suggested models to download, installed models,
//! and importing your own. Every action runs an engine `ai.*` command, so agents drive the same
//! thing through the control channel; downloads and imports are background jobs.

use egui::{RichText, Sense, Stroke, vec2};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::PhotokaApp;
use crate::theme::Tokens;

/// Model Manager tabs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ManagerTab {
    #[default]
    Suggested,
    Installed,
    Import,
}

/// Model Manager view state.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelManagerUi {
    #[serde(default)]
    pub open: bool,
    #[serde(default)]
    pub tab: ManagerTab,
    /// A models folder the user picked (applied with `ai.models.setFolder` at startup).
    #[serde(default)]
    pub models_dir: Option<String>,
    /// The catalog entry whose license is being shown before its download.
    #[serde(default, skip)]
    pub license_prompt: Option<String>,
    /// The starter pack's licences are being shown before its downloads.
    #[serde(default, skip)]
    pub starter_prompt: bool,
    /// The installed file a Delete click is waiting to be confirmed for.
    #[serde(default, skip)]
    pub confirm_delete: Option<String>,
    #[serde(default, skip)]
    pub key_input: String,
    /// `ai.catalog.list` and `ai.models.list`, refreshed about once a second while shown.
    #[serde(default, skip)]
    pub catalog: Value,
    #[serde(default, skip)]
    pub installed: Value,
    #[serde(default, skip)]
    pub refreshed_at: Option<f64>,
    #[serde(default, skip)]
    pub folder_applied: bool,
}

pub const MENU_ID: &str = "window.modelManager";

pub fn handles(id: &str) -> bool {
    id == MENU_ID
}

pub fn checked(app: &PhotokaApp, id: &str) -> Option<bool> {
    (id == MENU_ID).then_some(app.ui.model_manager.open)
}

/// Window › Model Manager toggles the window; `{"tab": "suggested"|"installed"|"import"}` opens it
/// on a tab.
pub fn menu(app: &mut PhotokaApp, id: &str, params: &Value) -> Option<Result<Value, String>> {
    if id != MENU_ID {
        return None;
    }
    let tab = match params.get("tab").and_then(Value::as_str) {
        Some("suggested") => Some(ManagerTab::Suggested),
        Some("installed") => Some(ManagerTab::Installed),
        Some("import") => Some(ManagerTab::Import),
        Some(other) => return Some(Err(format!("unknown Model Manager tab `{other}` (suggested, installed, import)"))),
        None => None,
    };
    let m = &mut app.ui.model_manager;
    match tab {
        Some(t) => {
            m.open = true;
            m.tab = t;
        }
        None => m.open = !m.open,
    }
    m.refreshed_at = None;
    Some(Ok(json!({"open": m.open})))
}

/// Apply the saved models folder once per launch.
pub fn apply_saved_folder(app: &mut PhotokaApp) {
    if app.ui.model_manager.folder_applied {
        return;
    }
    app.ui.model_manager.folder_applied = true;
    if let Some(dir) = app.ui.model_manager.models_dir.clone()
        && let Err(e) = app.session.execute("ai.models.setFolder", json!({"path": dir}))
    {
        crate::notices::error(app, format!("{}: {e}", tl!("Models folder")));
    }
}

/// `.safetensors` files dropped on the window are imported as models; the rest are returned.
pub fn take_model_drops(app: &mut PhotokaApp, dropped: Vec<egui::DroppedFileHandle>) -> Vec<egui::DroppedFileHandle> {
    let (models, rest): (Vec<_>, Vec<_>) =
        dropped.into_iter().partition(|f| f.path().is_absolute() && f.path().extension().is_some_and(|e| e.eq_ignore_ascii_case("safetensors")));
    for f in models {
        import_path(app, &f.path().to_string_lossy());
    }
    rest
}

fn import_path(app: &mut PhotokaApp, path: &str) {
    app.ui.model_manager.open = true;
    app.ui.model_manager.tab = ManagerTab::Installed;
    app.ui.model_manager.refreshed_at = None;
    if let Err(e) = app.run("ai.models.import", json!({"path": path})) {
        crate::notices::error(app, e);
    }
}

fn refresh(app: &mut PhotokaApp, now: f64) {
    let m = &app.ui.model_manager;
    if m.refreshed_at.is_some_and(|t| now - t < 1.0) {
        return;
    }
    let catalog = app.session.execute("ai.catalog.list", json!({})).unwrap_or(Value::Null);
    let installed = app.session.execute("ai.models.list", json!({})).unwrap_or(Value::Null);
    let m = &mut app.ui.model_manager;
    m.catalog = catalog;
    m.installed = installed;
    m.refreshed_at = Some(now);
}

fn human(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e12 {
        format!("{:.1} TB", b / 1e12)
    } else if b >= 1e9 {
        format!("{:.1} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.0} MB", b / 1e6)
    } else {
        format!("{:.0} KB", b / 1e3)
    }
}

/// What a card or row asked for this frame.
enum Act {
    Run(&'static str, Value),
    PromptLicense(String),
    CloseLicense,
    StarterPrompt(bool),
    AcceptStarter,
    ConfirmDelete(Option<String>),
    Cancel(photoka_engine::jobs::JobId),
    PickFile,
    PickFolder,
    Tab(ManagerTab),
    Close,
}

pub fn windows(app: &mut PhotokaApp, ctx: &egui::Context) {
    apply_saved_folder(app);
    if !app.ui.model_manager.open {
        return;
    }
    let now = ctx.input(|i| i.time);
    refresh(app, now);
    if app.session.has_jobs() {
        ctx.request_repaint_after(std::time::Duration::from_millis(250));
    }
    let t = Tokens::get(ctx);
    let m = app.ui.model_manager.clone();
    let jobs = app.session.jobs();
    let mut acts: Vec<Act> = Vec::new();
    let mut key_input = m.key_input.clone();
    crate::analysis_ui::panel_window(app, ctx, "model-manager", tl!("Model Manager"), vec2(-260.0, 40.0), 600.0, |ui| {
        if crate::analysis_ui::title_row(ui, tl!("Model Manager")) {
            acts.push(Act::Close);
        }
        ui.horizontal(|ui| {
            for (tab, label) in [(ManagerTab::Suggested, tl!("Suggested")), (ManagerTab::Installed, tl!("Installed")), (ManagerTab::Import, tl!("Import"))] {
                if ui.selectable_label(m.tab == tab, label).clicked() {
                    acts.push(Act::Tab(tab));
                }
            }
        });
        ui.add_space(4.0);
        if m.installed.get("nativeSupported") == Some(&json!(false)) {
            ui.label(RichText::new(tl!("AI generation needs a Mac with Apple silicon (M1 or later).")).color(t.warning).size(12.0));
        }
        let dir = m.installed.get("modelsDir").and_then(Value::as_str).unwrap_or_default().to_string();
        ui.horizontal(|ui| {
            ui.label(RichText::new(tl!("Models folder")).color(t.text_dim).size(11.5));
            ui.add(egui::Label::new(RichText::new(&dir).color(t.text_faint).size(11.5)).truncate()).on_hover_text(&dir);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if crate::icons::button(ui, "folder-open", 20.0, false, tl!("Reveal in Finder")).clicked() {
                    acts.push(Act::Run("ai.models.reveal", json!({"root": true})));
                }
                if ui.small_button(tl!("Change…")).clicked() {
                    acts.push(Act::PickFolder);
                }
            });
        });
        crate::widgets::hairline(ui);
        ui.add_space(4.0);
        match m.tab {
            ManagerTab::Suggested => suggested(ui, &t, &m, &jobs, &mut key_input, &mut acts),
            ManagerTab::Installed => installed(ui, &t, &m, &mut acts),
            ManagerTab::Import => import_tab(ui, &t, &mut acts),
        }
    });
    app.ui.model_manager.key_input = key_input;
    for act in acts {
        let refresh_after = !matches!(act, Act::Tab(_) | Act::PromptLicense(_) | Act::CloseLicense | Act::StarterPrompt(_) | Act::ConfirmDelete(_));
        match act {
            Act::Run(id, params) => {
                let accept_then_download = id == "ai.models.acceptLicense";
                let result = app.run(id, params.clone());
                match result {
                    Ok(_) if accept_then_download => {
                        app.ui.model_manager.license_prompt = None;
                        if let Err(e) = app.run("ai.models.download", params) {
                            crate::notices::error(app, e);
                        }
                    }
                    Ok(_) => {
                        if id == "ai.civitai.setKey" {
                            app.ui.model_manager.key_input.clear();
                        }
                    }
                    Err(e) => crate::notices::error(app, e),
                }
            }
            Act::PromptLicense(id) => app.ui.model_manager.license_prompt = Some(id),
            Act::CloseLicense => app.ui.model_manager.license_prompt = None,
            Act::StarterPrompt(on) => app.ui.model_manager.starter_prompt = on,
            Act::AcceptStarter => {
                app.ui.model_manager.starter_prompt = false;
                match app.run("ai.models.acceptStarterPack", json!({"accept": true})) {
                    Ok(r) => {
                        let ids: Vec<String> = r["toDownload"].as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_string).collect();
                        for id in ids {
                            if let Err(e) = app.run("ai.models.download", json!({"id": id})) {
                                crate::notices::error(app, e);
                            }
                        }
                    }
                    Err(e) => crate::notices::error(app, e),
                }
            }
            Act::ConfirmDelete(f) => app.ui.model_manager.confirm_delete = f,
            Act::Cancel(job) => crate::jobs_ui::cancel(app, job),
            Act::PickFile => {
                let request = crate::file_dialog::FileDialogRequest::Open { multiple: true, initial_dir: None, extensions: Some(vec!["safetensors".into()]) };
                let r = app.ask_file(request, |app, answer| match answer {
                    crate::file_dialog::FileDialogAnswer::Paths(paths) => {
                        for p in paths {
                            import_path(app, &p);
                        }
                        Ok(Value::Null)
                    }
                    _ => Err(tl!("Choose a model file on this Mac").into()),
                });
                if let Err(e) = r
                    && e != crate::file_dialog::CANCELLED
                {
                    crate::notices::error(app, e);
                }
            }
            Act::PickFolder => {
                let r = app.ask_folder(|app, path| {
                    app.session.execute("ai.models.setFolder", json!({"path": path})).map_err(|e| e.to_string())?;
                    app.ui.model_manager.models_dir = Some(path);
                    app.ui.model_manager.refreshed_at = None;
                    Ok(Value::Null)
                });
                if let Err(e) = r
                    && e != crate::file_dialog::CANCELLED
                {
                    crate::notices::error(app, e);
                }
            }
            Act::Tab(tab) => app.ui.model_manager.tab = tab,
            Act::Close => app.ui.model_manager.open = false,
        }
        if refresh_after {
            app.ui.model_manager.refreshed_at = None;
        }
    }
}

fn badge(ui: &mut egui::Ui, t: &Tokens, text: &str) {
    let galley = ui.painter().layout_no_wrap(text.to_string(), egui::FontId::proportional(10.5), t.text_dim);
    let (r, _) = ui.allocate_exact_size(galley.size() + vec2(10.0, 4.0), Sense::hover());
    ui.painter().rect(r, 4.0, t.field, Stroke::new(1.0, t.field_border), egui::StrokeKind::Inside);
    ui.painter().galley(r.min + vec2(5.0, 2.0), galley, t.text_dim);
}

fn kind_label(kind: &str) -> &'static str {
    match kind {
        "checkpoint" => tl!("Checkpoint"),
        "lora" => tl!("LoRA"),
        "vae" => tl!("VAE"),
        "embedding" => tl!("Embedding"),
        "tokenizer" => tl!("Tokenizer"),
        "upscale" => tl!("Upscale"),
        _ => tl!("Model"),
    }
}

fn family_label(family: &str) -> &'static str {
    match family {
        "sd15" => "SD 1.5",
        "sd15-inpaint" => tl!("SD 1.5 Inpainting"),
        "sd-turbo" => "SD-Turbo",
        "upscale" => tl!("Upscale"),
        _ => "",
    }
}

fn hardware_summary(catalog: &Value) -> String {
    let hw = catalog.get("hardware");
    let ram = hw.and_then(|h| h.get("ramGb")).and_then(Value::as_u64);
    let disk = hw.and_then(|h| h.get("diskFreeGb")).and_then(Value::as_u64);
    let silicon = hw.and_then(|h| h.get("appleSilicon")).and_then(Value::as_bool).unwrap_or(false);
    let chip = hw.and_then(|h| h.get("chip")).and_then(Value::as_str).unwrap_or("");
    let mut parts = Vec::new();
    if let Some(g) = ram {
        parts.push(format!("{g} GB"));
    }
    if let Some(g) = disk {
        parts.push(format!("{g} GB free"));
    }
    if silicon {
        parts.push("Apple Silicon".into());
    }
    if !chip.is_empty() && parts.len() < 3 {
        parts.push(chip.into());
    }
    if parts.is_empty() {
        tl!("Models that fit this Mac are listed first.").to_string()
    } else {
        format!("{} — {}", parts.join(" · "), tl!("models that fit are listed first"))
    }
}

fn suggested(ui: &mut egui::Ui, t: &Tokens, m: &ModelManagerUi, jobs: &[photoka_engine::jobs::JobInfo], key_input: &mut String, acts: &mut Vec<Act>) {
    let has_key = m.catalog.get("civitaiKey").and_then(Value::as_bool).unwrap_or(false);
    ui.horizontal(|ui| {
        ui.label(RichText::new(tl!("Civitai API key")).color(t.text_dim).size(11.5));
        if has_key {
            ui.label(RichText::new(tl!("Saved in the Keychain")).color(t.text_faint).size(11.5));
            if ui.small_button(tl!("Remove")).clicked() {
                acts.push(Act::Run("ai.civitai.setKey", json!({"key": ""})));
            }
        } else {
            ui.add(egui::TextEdit::singleline(key_input).password(true).desired_width(220.0).hint_text(tl!("Optional, for Civitai downloads")));
            if ui.small_button(tl!("Save")).clicked() && !key_input.trim().is_empty() {
                acts.push(Act::Run("ai.civitai.setKey", json!({"key": key_input.trim()})));
            }
        }
    });
    ui.add_space(2.0);
    ui.label(RichText::new(format!("{}: {}", tl!("This Mac"), hardware_summary(&m.catalog))).color(t.text_faint).size(11.5));
    if let Some(pack) = m.catalog.get("starterPack") {
        let ids = pack.get("ids").and_then(Value::as_array).cloned().unwrap_or_default();
        let need = ids.iter().filter_map(|id| id.as_str()).any(|id| {
            m.catalog
                .get("models")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .any(|e| e.get("id").and_then(Value::as_str) == Some(id) && e["installed"] != json!(true))
        });
        if need && !ids.is_empty() {
            ui.horizontal(|ui| {
                ui.label(RichText::new(tl!("Starter pack for this Mac")).color(t.text_dim).size(11.5));
                if crate::widgets::primary_button(ui, tl!("Download starter pack"), 150.0).clicked() {
                    acts.push(Act::StarterPrompt(true));
                }
                let total = pack.get("totalBytes").and_then(Value::as_u64).unwrap_or(0);
                if total > 0 {
                    ui.label(RichText::new(human(total)).color(t.text_faint).size(11.5));
                }
            });
            if m.starter_prompt {
                let models = m.catalog.get("models").and_then(Value::as_array);
                ui.label(RichText::new(tl!("Read each licence before you download the starter pack:")).color(t.text_dim).size(11.5));
                for id in ids.iter().filter_map(Value::as_str) {
                    let Some(e) = models.into_iter().flatten().find(|e| e.get("id").and_then(Value::as_str) == Some(id)) else { continue };
                    let s = |k: &str| e.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(s("name")).color(t.text).size(11.5));
                        ui.hyperlink_to(RichText::new(s("license")).size(11.5), s("license_url"));
                    });
                }
                ui.horizontal(|ui| {
                    if crate::widgets::primary_button(ui, tl!("Accept and download"), 150.0).clicked() {
                        acts.push(Act::AcceptStarter);
                    }
                    if ui.button(tl!("Cancel")).clicked() {
                        acts.push(Act::StarterPrompt(false));
                    }
                });
            }
        }
    }
    ui.add_space(4.0);
    let empty = Vec::new();
    let models = m.catalog.get("models").and_then(Value::as_array).unwrap_or(&empty);
    egui::ScrollArea::vertical().max_height(460.0).min_scrolled_height(460.0).auto_shrink([false, true]).show(ui, |ui| {
        for e in models.iter().filter(|e| e["kind"] != "tokenizer") {
            card(ui, t, m, e, jobs, has_key, acts);
            ui.add_space(6.0);
        }
        if let Some(tok) = models.iter().find(|e| e["kind"] == "tokenizer") {
            ui.label(RichText::new(tl!("Support files")).color(t.text_dim).size(11.5));
            card(ui, t, m, tok, jobs, has_key, acts);
        }
    });
}

fn card(ui: &mut egui::Ui, t: &Tokens, m: &ModelManagerUi, e: &Value, jobs: &[photoka_engine::jobs::JobInfo], has_key: bool, acts: &mut Vec<Act>) {
    let s = |k: &str| e.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
    let (id, name) = (s("id"), s("name"));
    let installed = e["installed"] == json!(true);
    let needs_key = e["requires_key"] == json!(true) && !has_key;
    let fits_disk = e.get("fitsDisk").and_then(Value::as_bool).unwrap_or(true);
    let fits_ram = e.get("fitsRam").and_then(Value::as_bool).unwrap_or(true);
    let best = e.get("recommendedForHw").and_then(Value::as_bool).unwrap_or(false);
    let reason = s("recommendReason");
    let job = jobs.iter().find(|j| j.command == "ai.models.download" && j.label.ends_with(&name));
    egui::Frame::NONE.fill(t.field).stroke(Stroke::new(1.0, t.field_border)).corner_radius(t.radius_sm).inner_margin(8.0).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.label(RichText::new(&name).strong().color(t.text));
            badge(ui, t, kind_label(&s("kind")));
            let fam = family_label(&s("family"));
            if !fam.is_empty() {
                badge(ui, t, fam);
            }
            if best {
                badge(ui, t, tl!("Best for this Mac"));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if installed {
                    ui.label(RichText::new(tl!("Installed")).color(t.accent_text));
                    let (r, _) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
                    crate::icons::paint(ui, r, "check", 14.0, t.accent_text);
                } else if let Some(j) = job {
                    if crate::icons::button(ui, "x", 20.0, false, tl!("Cancel")).clicked() {
                        acts.push(Act::Cancel(j.id));
                    }
                    let (r, _) = ui.allocate_exact_size(vec2(120.0, 6.0), Sense::hover());
                    crate::jobs_ui::bar(ui, r, (j.progress > 0.0).then_some(j.progress), t);
                } else if needs_key {
                    ui.add_enabled(false, egui::Button::new(tl!("Needs a Civitai key")));
                } else if !fits_disk {
                    ui.add_enabled(false, egui::Button::new(tl!("Not enough free space")));
                } else {
                    let label = if e["partialBytes"].is_u64() { tl!("Resume") } else { tl!("Download") };
                    if crate::widgets::primary_button(ui, label, 90.0).clicked() {
                        if e["licenseAccepted"] == json!(true) {
                            acts.push(Act::Run("ai.models.download", json!({"id": id})));
                        } else {
                            acts.push(Act::PromptLicense(id.clone()));
                        }
                    }
                }
                ui.label(RichText::new(human(e["size"].as_u64().unwrap_or(0))).color(t.text_faint).size(11.5));
            });
        });
        ui.label(RichText::new(s("description")).color(t.text_dim).size(12.0));
        if !reason.is_empty() {
            ui.label(RichText::new(&reason).color(if fits_ram { t.text_faint } else { t.danger }).size(11.0));
        } else if fits_disk && !fits_ram {
            ui.label(RichText::new(tl!("May be too large for this Mac’s memory")).color(t.danger).size(11.0));
        }
        ui.horizontal_wrapped(|ui| {
            if let Some(tags) = e["tags"].as_array() {
                for tag in tags.iter().filter_map(Value::as_str) {
                    badge(ui, t, tag);
                }
            }
            ui.hyperlink_to(RichText::new(s("license")).size(11.5), s("license_url"));
        });
        if let Some(j) = job {
            ui.label(RichText::new(&j.message).color(t.text_faint).size(11.0));
        }
        if m.license_prompt.as_deref() == Some(id.as_str()) {
            crate::widgets::hairline(ui);
            ui.label(
                RichText::new(crate::i18n::fmt(
                    tl!("{name} is licensed under {license}. Read the license before you download it."),
                    &[("name", &name), ("license", &s("license"))],
                ))
                .color(t.text)
                .size(12.0),
            );
            ui.horizontal(|ui| {
                ui.hyperlink_to(tl!("Read the license"), s("license_url"));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if crate::widgets::primary_button(ui, tl!("Accept and Download"), 150.0).clicked() {
                        acts.push(Act::Run("ai.models.acceptLicense", json!({"id": id})));
                    }
                    if crate::widgets::secondary_button(ui, tl!("Cancel"), 70.0).clicked() {
                        acts.push(Act::CloseLicense);
                    }
                });
            });
        }
    });
}

fn installed(ui: &mut egui::Ui, t: &Tokens, m: &ModelManagerUi, acts: &mut Vec<Act>) {
    let empty = Vec::new();
    let models = m.installed.get("models").and_then(Value::as_array).unwrap_or(&empty);
    if models.is_empty() {
        ui.add_space(12.0);
        ui.vertical_centered(|ui| {
            ui.label(RichText::new(tl!("No models yet. Download one from Suggested, or import your own.")).color(t.text_dim));
        });
        ui.add_space(12.0);
    }
    egui::ScrollArea::vertical().max_height(420.0).auto_shrink([false, true]).show(ui, |ui| {
        for e in models {
            let s = |k: &str| e.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
            let (file, kind) = (s("file_name"), s("kind"));
            let key = format!("{kind}/{file}");
            ui.horizontal(|ui| {
                ui.add(egui::Label::new(RichText::new(&file).color(t.text)).truncate()).on_hover_text(s("path"));
                badge(ui, t, kind_label(&kind));
                match e.get("family").and_then(Value::as_str) {
                    Some(f) if !family_label(f).is_empty() => badge(ui, t, family_label(f)),
                    _ => {}
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if m.confirm_delete.as_deref() == Some(key.as_str()) {
                        if ui.button(RichText::new(tl!("Delete")).color(t.danger)).clicked() {
                            acts.push(Act::Run("ai.models.delete", json!({"kind": kind, "file": file})));
                            acts.push(Act::ConfirmDelete(None));
                        }
                        if ui.small_button(tl!("Cancel")).clicked() {
                            acts.push(Act::ConfirmDelete(None));
                        }
                    } else if crate::icons::button(ui, "trash", 20.0, false, tl!("Delete")).clicked() {
                        acts.push(Act::ConfirmDelete(Some(key.clone())));
                    }
                    if crate::icons::button(ui, "folder-open", 20.0, false, tl!("Reveal in Finder")).clicked() {
                        acts.push(Act::Run("ai.models.reveal", json!({"kind": kind, "file": file})));
                    }
                    ui.label(RichText::new(human(e["size"].as_u64().unwrap_or(0))).color(t.text_faint).size(11.5));
                });
            });
            if let Some(p) = e.get("problem").and_then(Value::as_str) {
                ui.label(RichText::new(p).color(t.warning).size(11.0));
            }
        }
    });
    crate::widgets::hairline(ui);
    let total = m.installed["totalBytes"].as_u64().unwrap_or(0);
    let free = m.installed["freeBytes"].as_u64();
    let line = match free {
        Some(f) => crate::i18n::fmt(tl!("{used} used, {free} free"), &[("used", &human(total)), ("free", &human(f))]),
        None => human(total),
    };
    ui.label(RichText::new(line).color(t.text_faint).size(11.5));
}

fn import_tab(ui: &mut egui::Ui, t: &Tokens, acts: &mut Vec<Act>) {
    let (r, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 150.0), Sense::click());
    let hovering = ui.ctx().input(|i| !i.raw.hovered_files.is_empty());
    let stroke = Stroke::new(1.5, if hovering || resp.hovered() { t.accent } else { t.field_border });
    ui.painter().rect(r, t.radius_lg, t.field, stroke, egui::StrokeKind::Inside);
    crate::icons::paint(ui, egui::Rect::from_center_size(r.center() - vec2(0.0, 28.0), vec2(28.0, 28.0)), "file-plus", 26.0, t.text_dim);
    ui.painter().text(r.center() + vec2(0.0, 6.0), egui::Align2::CENTER_CENTER, tl!("Drop .safetensors files here"), egui::FontId::proportional(14.0), t.text);
    ui.painter().text(
        r.center() + vec2(0.0, 28.0),
        egui::Align2::CENTER_CENTER,
        tl!("or click to choose a file"),
        egui::FontId::proportional(12.0),
        t.text_dim,
    );
    if resp.clicked() {
        acts.push(Act::PickFile);
    }
    ui.add_space(8.0);
    ui.label(
        RichText::new(tl!(
            "Checkpoints, LoRAs, VAEs and embeddings for SD 1.5 and SD-Turbo. Pickle files (.ckpt, .pt) aren't accepted because they can run code when loaded."
        ))
        .color(t.text_faint)
        .size(11.5),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app(tag: &str) -> (PhotokaApp, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("photoka-mm-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut s = photoka_engine::Session::new();
        s.ai.models_dir = dir.join("models");
        (PhotokaApp::new(s, crate::Services::default()), dir)
    }

    #[test]
    fn menu_toggles_and_opens_tabs() {
        let (mut app, dir) = app("menu");
        let ctx = egui::Context::default();
        crate::menus::invoke(&mut app, &ctx, MENU_ID, json!({})).unwrap();
        assert!(app.ui.model_manager.open);
        crate::menus::invoke(&mut app, &ctx, MENU_ID, json!({"tab": "import"})).unwrap();
        assert!(app.ui.model_manager.open && app.ui.model_manager.tab == ManagerTab::Import);
        assert!(crate::menus::invoke(&mut app, &ctx, MENU_ID, json!({"tab": "nope"})).is_err());
        crate::menus::invoke(&mut app, &ctx, MENU_ID, json!({})).unwrap();
        assert!(!app.ui.model_manager.open);
        assert!(crate::menus::is_live(MENU_ID));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn every_tab_draws_with_an_empty_folder() {
        let (mut app, dir) = app("draw");
        let ctx = egui::Context::default();
        for tab in [ManagerTab::Suggested, ManagerTab::Installed, ManagerTab::Import] {
            app.ui.model_manager.open = true;
            app.ui.model_manager.tab = tab;
            app.ui.model_manager.license_prompt = Some("sd15".into());
            crate::analysis_ui::tests::render(&mut app, &ctx, windows);
            assert!(app.ui.model_manager.catalog["models"].as_array().is_some_and(|m| m.len() > 5));
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn sizes_read_naturally() {
        assert_eq!(human(2_132_696_762), "2.1 GB");
        assert_eq!(human(37_864_776), "38 MB");
        assert_eq!(human(2_914_100_000_000), "2.9 TB");
    }
}

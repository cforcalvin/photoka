//! Offline image generation: the model manager (Photoka extension, Draw Things-style).
//!
//! - `ai.catalog.list`: the bundled suggestions, with install, download and license state.
//! - `ai.models.list`: what's in the models folder (ComfyUI layout, shared by both engines).
//! - `ai.models.acceptLicense`, then `ai.models.download` (a background job: resumable,
//!   sha256-verified). Nothing is ever downloaded unless asked for.
//! - `ai.models.import` (a job: copies a `.safetensors` file in after checking it), `ai.models.delete`,
//!   `ai.models.reveal`, `ai.models.setFolder`, `ai.civitai.setKey` (Keychain).
//!
//! The work itself lives in `photoka-ml`; this module adapts it to commands and jobs.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use photoka_ml::catalog::{Catalog, ModelKind};
use photoka_ml::store::ModelStore;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::jobs::JobCtx;
use crate::{EngineError, Result, Session};

/// Session state for AI generation (not document data; nothing here is saved in files).
#[derive(Debug)]
pub struct AiState {
    /// The models folder. Defaults to `PHOTOKA_MODELS_DIR` or
    /// `~/Library/Application Support/Photoka/models`; `ai.models.setFolder` changes it.
    pub models_dir: PathBuf,
    /// Catalog ids being downloaded right now.
    downloading: Arc<Mutex<HashSet<String>>>,
    /// Generation frames, masks and the warm native model (see `ai_gen_cmds`).
    pub generate: crate::ai_gen_cmds::GenState,
    /// The managed ComfyUI install and process (see `ai_comfy_cmds`).
    pub comfy: crate::ai_comfy_cmds::ComfyState,
}

impl Default for AiState {
    fn default() -> Self {
        let models_dir = std::env::var_os("PHOTOKA_MODELS_DIR").map(PathBuf::from).unwrap_or_else(photoka_ml::system::default_models_dir);
        AiState { models_dir, downloading: Arc::default(), generate: Default::default(), comfy: Default::default() }
    }
}

impl AiState {
    pub fn store(&self) -> ModelStore {
        ModelStore::new(&self.models_dir)
    }

    pub fn is_downloading(&self, id: &str) -> bool {
        self.downloading.lock().unwrap_or_else(PoisonError::into_inner).contains(id)
    }
}

/// Removes a catalog id from the downloading set however the job ends.
struct DownloadGuard(Arc<Mutex<HashSet<String>>>, String);

impl Drop for DownloadGuard {
    fn drop(&mut self) {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).remove(&self.1);
    }
}

/// A job's progress and cancellation, as `photoka-ml` sees them.
pub(crate) struct JobProgress<'a>(pub &'a JobCtx);

impl photoka_ml::Progress for JobProgress<'_> {
    fn report(&self, fraction: f32, message: &str) {
        self.0.progress(fraction, message);
    }
    fn cancelled(&self) -> bool {
        self.0.cancelled()
    }
}

pub(crate) fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

pub(crate) fn ml_err(e: photoka_ml::MlError) -> EngineError {
    match e {
        photoka_ml::MlError::Cancelled => EngineError::Cancelled,
        e => EngineError::Other(e.to_string()),
    }
}

fn always(_: &Session) -> std::result::Result<(), String> {
    Ok(())
}

pub(crate) fn catalog() -> Result<Catalog> {
    Catalog::bundled().map_err(ml_err)
}

pub(crate) fn str_param<'a>(cmd: &str, p: &'a Value, key: &str) -> Result<&'a str> {
    p.get(key).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).ok_or_else(|| bad(cmd, format!("`{key}` (a non-empty string) is required")))
}

fn kind_param(cmd: &str, p: &Value) -> Result<Option<ModelKind>> {
    match p.get("kind") {
        None | Some(Value::Null) => Ok(None),
        Some(v) => {
            let s = v.as_str().ok_or_else(|| bad(cmd, "`kind` must be a string"))?;
            ModelKind::parse(s).map(Some).ok_or_else(|| {
                bad(cmd, format!("unknown kind `{s}` (checkpoint, lora, vae, embedding, tokenizer, upscale)"))
            })
        }
    }
}

fn catalog_list(s: &mut Session, _: &Value) -> Result<Value> {
    let c = catalog()?;
    let store = s.ai.store();
    let accepted = store.accepted_licenses();
    let hw = photoka_ml::system::HardwareProfile::probe(&s.ai.models_dir);
    let installed = |e: &photoka_ml::catalog::CatalogEntry| store.is_installed(e.kind, &e.file_name);
    let best_ckpt = photoka_ml::recommend::best_checkpoint_ids(&c.models, &hw, &installed, 2);
    let best_up = photoka_ml::recommend::best_upscale_ids(&c.models, &hw, 1);
    let pack = photoka_ml::recommend::starter_pack(&c.models, &hw, &installed);
    let order = photoka_ml::recommend::rank_indices(&c.models, &hw, &installed);
    let models: Vec<Value> = order
        .into_iter()
        .map(|i| {
            let m = &c.models[i];
            let dest = store.path_for(m.kind, &m.file_name).ok();
            let fits_ram = photoka_ml::recommend::fits_ram(m, &hw);
            let fits_disk = photoka_ml::recommend::fits_disk(m, &hw);
            let recommended_for_hw = best_ckpt.iter().any(|id| id == &m.id) || best_up.iter().any(|id| id == &m.id);
            let mut v = serde_json::to_value(m).unwrap_or(Value::Null);
            if let Some(o) = v.as_object_mut() {
                o.insert("installed".into(), json!(installed(m)));
                o.insert("licenseAccepted".into(), json!(accepted.contains(&m.id)));
                o.insert("downloading".into(), json!(s.ai.is_downloading(&m.id)));
                o.insert("partialBytes".into(), json!(dest.as_deref().and_then(photoka_ml::download::partial_len)));
                o.insert("fitsRam".into(), json!(fits_ram));
                o.insert("fitsDisk".into(), json!(fits_disk));
                o.insert("recommendedForHw".into(), json!(recommended_for_hw));
                o.insert("recommendReason".into(), json!(photoka_ml::recommend::recommend_reason(m, &hw)));
            }
            v
        })
        .collect();
    Ok(json!({
        "version": c.version,
        "models": models,
        "civitaiKey": photoka_ml::keychain::civitai_key().is_some(),
        "modelsDir": s.ai.models_dir,
        "hardware": hw.to_json(),
        "starterPack": pack.as_ref().map(|p| json!({
            "ids": p.ids,
            "totalBytes": p.total_bytes,
        })),
        "upscaleDefaults": {
            "scale": hw.default_upscale_scale(),
            "tile": hw.default_upscale_tile(),
        },
    }))
}

/// The starter pack's licences. With `accept: true` they're accepted and `toDownload` lists the
/// ids still missing; the caller downloads each with `ai.models.download` (one job each).
fn accept_starter_pack(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ai.models.acceptStarterPack";
    let c = catalog()?;
    let store = s.ai.store();
    let hw = photoka_ml::system::HardwareProfile::probe(&s.ai.models_dir);
    let installed = |e: &photoka_ml::catalog::CatalogEntry| store.is_installed(e.kind, &e.file_name);
    let pack = photoka_ml::recommend::starter_pack(&c.models, &hw, &installed)
        .ok_or_else(|| bad(CMD, "no starter pack fits this Mac’s free disk; free some space or pick a model manually"))?;
    let members: Vec<&photoka_ml::catalog::CatalogEntry> = pack.ids.iter().filter_map(|id| c.get(id)).collect();
    if p.get("accept").and_then(Value::as_bool) != Some(true) {
        let list = members.iter().map(|m| format!("{} ({})", m.name, m.license)).collect::<Vec<_>>().join("; ");
        return Err(bad(CMD, format!("the starter pack is {list}; read the licences, then pass `accept: true`")));
    }
    let mut to_download = Vec::new();
    for m in &members {
        if store.is_installed(m.kind, &m.file_name) {
            continue;
        }
        store.accept_license(&m.id).map_err(ml_err)?;
        to_download.push(m.id.clone());
    }
    let licenses: Vec<Value> = members.iter().map(|m| json!({"id": m.id, "name": m.name, "license": m.license, "licenseUrl": m.license_url})).collect();
    Ok(json!({
        "ids": pack.ids,
        "totalBytes": pack.total_bytes,
        "licenses": licenses,
        "toDownload": to_download,
    }))
}

fn models_list(s: &mut Session, _: &Value) -> Result<Value> {
    let c = catalog().ok();
    let store = s.ai.store();
    let models = store.scan(c.as_ref());
    let total: u64 = models.iter().map(|m| m.size).sum();
    Ok(json!({
        "modelsDir": store.root(),
        "models": models,
        "totalBytes": total,
        "freeBytes": photoka_ml::system::available_space(store.root()),
        "nativeSupported": photoka_ml::system::NATIVE_SUPPORTED,
    }))
}

fn accept_license(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ai.models.acceptLicense";
    let id = str_param(CMD, p, "id")?;
    let c = catalog()?;
    let m = c.get(id).ok_or_else(|| bad(CMD, format!("no catalog model `{id}`")))?;
    s.ai.store().accept_license(id).map_err(ml_err)?;
    Ok(json!({"id": id, "license": m.license, "accepted": true}))
}

fn download(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ai.models.download";
    let id = str_param(CMD, p, "id")?.to_string();
    let c = catalog()?;
    let m = c.get(&id).ok_or_else(|| bad(CMD, format!("no catalog model `{id}`")))?.clone();
    let store = s.ai.store();
    if store.is_installed(m.kind, &m.file_name) {
        return Ok(json!({"id": id, "installed": true, "already": true}));
    }
    if !store.license_accepted(&id) {
        return Err(EngineError::Other(format!(
            "{} is licensed under {} ({}); accept the license first (ai.models.acceptLicense {{\"id\":\"{id}\"}})",
            m.name, m.license, m.license_url
        )));
    }
    let bearer = if m.requires_key {
        Some(photoka_ml::keychain::civitai_key().ok_or_else(|| {
            EngineError::Other(format!(
                "{} downloads from Civitai, which needs a free API key: add one in the Model Manager (civitai.com › Account settings › API Keys)",
                m.name
            ))
        })?)
    } else {
        None
    };
    {
        let mut set = s.ai.downloading.lock().unwrap_or_else(PoisonError::into_inner);
        if !set.insert(id.clone()) {
            return Err(EngineError::Other(format!("{} is already downloading", m.name)));
        }
    }
    let guard = DownloadGuard(Arc::clone(&s.ai.downloading), id.clone());
    store.ensure_layout().map_err(ml_err)?;
    let req = photoka_ml::download::DownloadRequest {
        url: m.url.clone(),
        dest: store.path_for(m.kind, &m.file_name).map_err(ml_err)?,
        size: m.size,
        sha256: m.sha256.clone(),
        bearer,
    };
    let label = format!("Downloading {}", m.name);
    let (id_a, file_a, kind_a) = (id.clone(), m.file_name.clone(), m.kind);
    crate::jobs::run(
        s,
        &label,
        false,
        move |ctx| {
            let _guard = guard;
            photoka_ml::download::download(&req, &JobProgress(ctx)).map_err(ml_err)
        },
        move |_, ()| Ok(json!({"id": id_a, "installed": true, "file": file_a, "kind": kind_a.as_str()})),
    )
}

fn import(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ai.models.import";
    let path = PathBuf::from(str_param(CMD, p, "path")?);
    let kind = kind_param(CMD, p)?;
    let store = s.ai.store();
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    crate::jobs::run(
        s,
        &format!("Importing {name}"),
        false,
        move |ctx| {
            ctx.progress(0.0, "Checking the model");
            store.import(&path, kind).map_err(ml_err)
        },
        |_, m| Ok(serde_json::to_value(&m).unwrap_or(Value::Null)),
    )
}

fn delete(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ai.models.delete";
    let kind = kind_param(CMD, p)?.ok_or_else(|| bad(CMD, "`kind` is required"))?;
    let file = str_param(CMD, p, "file")?;
    s.ai.generate.forget_model(&s.ai.store().path_for(kind, file).map_err(ml_err)?);
    s.ai.store().delete(kind, file).map_err(ml_err)?;
    Ok(json!({"deleted": file, "kind": kind.as_str()}))
}

fn reveal(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ai.models.reveal";
    let store = s.ai.store();
    let target = match (kind_param(CMD, p)?, p.get("file").and_then(Value::as_str)) {
        (Some(k), Some(f)) => store.existing(k, f).map_err(ml_err)?,
        (Some(k), None) => store.folder(k),
        (None, None) if p.get("root").and_then(Value::as_bool) == Some(true) => store.root().to_path_buf(),
        _ => return Err(bad(CMD, "give `kind` and `file`, `kind`, or `root: true`")),
    };
    let dry = p.get("dryRun").and_then(Value::as_bool).unwrap_or(false);
    if !dry {
        store.ensure_layout().map_err(ml_err)?;
    }
    crate::layer_menu_cmds::reveal(&target.to_string_lossy(), dry)
}

fn set_folder(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ai.models.setFolder";
    let path = PathBuf::from(str_param(CMD, p, "path")?);
    if !path.is_absolute() {
        return Err(bad(CMD, "`path` must be an absolute folder path"));
    }
    if path.is_file() {
        return Err(bad(CMD, format!("{} is a file, not a folder", path.display())));
    }
    ModelStore::new(&path).ensure_layout().map_err(ml_err)?;
    s.ai.generate.unload();
    s.ai.models_dir = path;
    Ok(json!({"modelsDir": s.ai.models_dir}))
}

fn set_civitai_key(_: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ai.civitai.setKey";
    let key = p.get("key").and_then(Value::as_str).ok_or_else(|| bad(CMD, "`key` (a string; empty removes it) is required"))?;
    photoka_ml::keychain::set_civitai_key(key).map_err(ml_err)?;
    Ok(json!({"stored": !key.trim().is_empty()}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "ai.catalog.list",
            label: "List Suggested AI Models",
            menu: &[],
            shortcut: None,
            params: "{} → {models:[{…,fitsRam,fitsDisk,recommendedForHw,recommendReason}],hardware,starterPack,upscaleDefaults,civitaiKey,modelsDir} (Suggested order is ranked for this Mac)",
            enabled: always,
            run: catalog_list,
            journal: false,
        },
        CommandSpec {
            id: "ai.models.list",
            label: "List Installed AI Models",
            menu: &[],
            shortcut: None,
            params: "{} → {modelsDir,models:[{kind,file_name,path,size,family,problem,catalog_id}],totalBytes,freeBytes,nativeSupported}",
            enabled: always,
            run: models_list,
            journal: false,
        },
        CommandSpec {
            id: "ai.models.acceptLicense",
            label: "Accept AI Model License",
            menu: &[],
            shortcut: None,
            params: r#"{"id":catalog id}"#,
            enabled: always,
            run: accept_license,
            journal: false,
        },
        CommandSpec {
            id: "ai.models.download",
            label: "Download AI Model",
            menu: &[],
            shortcut: None,
            params: r#"{"id":catalog id} → background job; resumes a partial download; the license must be accepted first"#,
            enabled: always,
            run: download,
            journal: false,
        },
        CommandSpec {
            id: "ai.models.acceptStarterPack",
            label: "Accept AI Starter Pack Licenses",
            menu: &[],
            shortcut: None,
            params: r#"{"accept":true} → accepts the licences of the hardware-sized starter pack → {ids,totalBytes,licenses,toDownload}; then ai.models.download each id in toDownload. Without accept: an error listing the licences"#,
            enabled: always,
            run: accept_starter_pack,
            journal: false,
        },
        CommandSpec {
            id: "ai.models.import",
            label: "Import AI Model",
            menu: &[],
            shortcut: None,
            params: r#"{"path":".safetensors|.pth file","kind":"checkpoint"|"lora"|"vae"|"embedding"|"upscale"? (detected when omitted)} → background job"#,
            enabled: always,
            run: import,
            journal: false,
        },
        CommandSpec {
            id: "ai.models.delete",
            label: "Delete AI Model",
            menu: &[],
            shortcut: None,
            params: r#"{"kind":"checkpoint"|"lora"|"vae"|"embedding"|"tokenizer"|"upscale","file":file name}"#,
            enabled: always,
            run: delete,
            journal: false,
        },
        CommandSpec {
            id: "ai.models.reveal",
            label: "Reveal AI Models in Finder",
            menu: &[],
            shortcut: None,
            params: r#"{"kind":str,"file":str?} | {"root":true}; "dryRun":bool?"#,
            enabled: always,
            run: reveal,
            journal: false,
        },
        CommandSpec {
            id: "ai.models.setFolder",
            label: "Set AI Models Folder",
            menu: &[],
            shortcut: None,
            params: r#"{"path":absolute folder} (created with the ComfyUI layout when missing)"#,
            enabled: always,
            run: set_folder,
            journal: false,
        },
        CommandSpec {
            id: "ai.civitai.setKey",
            label: "Set Civitai API Key",
            menu: &[],
            shortcut: None,
            params: r#"{"key":str} (stored in the macOS Keychain; empty removes it)"#,
            enabled: always,
            run: set_civitai_key,
            journal: false,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(tag: &str) -> (Session, PathBuf) {
        let dir = std::env::temp_dir().join(format!("photoka-ai-cmds-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut s = Session::new();
        s.ai.models_dir = dir.join("models");
        (s, dir)
    }

    #[test]
    fn catalog_and_empty_store() {
        let (mut s, dir) = session("list");
        let c = s.execute("ai.catalog.list", json!({})).unwrap();
        let models = c["models"].as_array().unwrap();
        assert!(models.len() >= 9);
        assert!(models.iter().all(|m| m["installed"] == json!(false) && m["licenseAccepted"] == json!(false)));
        assert!(c.get("hardware").is_some());
        assert!(c.get("starterPack").is_some());
        assert!(models.iter().any(|m| m["recommendedForHw"] == json!(true)));
        assert!(models.iter().all(|m| m.get("fitsRam").is_some() && m.get("recommendReason").is_some()));
        let l = s.execute("ai.models.list", json!({})).unwrap();
        assert_eq!(l["models"], json!([]));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn starter_pack_needs_explicit_acceptance_and_downloads_nothing() {
        let (mut s, dir) = session("starter");
        let e = s.execute("ai.models.acceptStarterPack", json!({})).unwrap_err().to_string();
        assert!(e.contains("accept") || e.contains("free"), "{e}");
        assert!(s.execute("ai.models.acceptStarterPack", json!({"accept": "yes"})).is_err());
        if let Ok(r) = s.execute("ai.models.acceptStarterPack", json!({"accept": true})) {
            let ids: Vec<&str> = r["toDownload"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
            assert!(!ids.is_empty(), "{r}");
            let accepted = s.ai.store().accepted_licenses();
            assert!(ids.iter().all(|id| accepted.contains(*id)));
            let l = s.execute("ai.models.list", json!({})).unwrap();
            assert_eq!(l["models"], json!([]), "accepting licences downloads nothing");
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn download_needs_an_accepted_license_and_nothing_downloads_by_default() {
        let (mut s, dir) = session("license");
        let e = s.execute("ai.models.download", json!({"id": "sd15"})).unwrap_err().to_string();
        assert!(e.contains("accept the license"), "{e}");
        assert!(!dir.join("models").exists(), "nothing written before the license is accepted");
        s.execute("ai.models.acceptLicense", json!({"id": "sd15"})).unwrap();
        let c = s.execute("ai.catalog.list", json!({})).unwrap();
        let sd15 = c["models"].as_array().unwrap().iter().find(|m| m["id"] == "sd15").unwrap().clone();
        assert_eq!(sd15["licenseAccepted"], json!(true));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn bad_params_are_errors() {
        let (mut s, dir) = session("bad");
        for (id, p) in [
            ("ai.models.download", json!({})),
            ("ai.models.download", json!({"id": "nope"})),
            ("ai.models.download", json!({"id": 5})),
            ("ai.models.acceptLicense", json!({"id": ""})),
            ("ai.models.import", json!({})),
            ("ai.models.import", json!({"path": "/nonexistent/x.safetensors"})),
            ("ai.models.import", json!({"path": "/tmp/x.ckpt"})),
            ("ai.models.import", json!({"path": "/tmp/x.safetensors", "kind": "sdxl"})),
            ("ai.models.delete", json!({})),
            ("ai.models.delete", json!({"kind": "lora", "file": "../../x"})),
            ("ai.models.delete", json!({"kind": "lora", "file": "missing.safetensors"})),
            ("ai.models.reveal", json!({})),
            ("ai.models.reveal", json!({"kind": "lora", "file": "missing.safetensors"})),
            ("ai.models.setFolder", json!({"path": "relative/dir"})),
            ("ai.civitai.setKey", json!({})),
            ("ai.civitai.setKey", json!({"key": "a\u{0}b"})),
        ] {
            assert!(s.execute(id, p.clone()).is_err(), "{id} {p}");
        }
        let r = s.execute("ai.models.reveal", json!({"kind": "lora", "dryRun": true})).unwrap();
        assert_eq!(r["program"], "open");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn import_list_delete() {
        let (mut s, dir) = session("import");
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("tiny.safetensors");
        // A minimal SD 1.5 LoRA header.
        let header = br#"{"lora_unet_x_attn2_to_k.lora_down.weight":{"dtype":"F16","shape":[1,768],"data_offsets":[0,1536]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.resize(bytes.len() + 1536, 0);
        std::fs::write(&src, bytes).unwrap();
        let m = s.execute("ai.models.import", json!({"path": src})).unwrap();
        assert_eq!(m["kind"], "lora");
        let l = s.execute("ai.models.list", json!({})).unwrap();
        assert_eq!(l["models"].as_array().unwrap().len(), 1);
        s.execute("ai.models.delete", json!({"kind": "lora", "file": "tiny.safetensors"})).unwrap();
        assert_eq!(s.execute("ai.models.list", json!({})).unwrap()["models"], json!([]));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn set_folder_creates_the_layout() {
        let (mut s, dir) = session("folder");
        let target = dir.join("elsewhere");
        s.execute("ai.models.setFolder", json!({"path": target})).unwrap();
        assert!(target.join("checkpoints").is_dir() && target.join("loras").is_dir());
        assert_eq!(s.ai.models_dir, target);
        let _ = std::fs::remove_dir_all(dir);
    }
}

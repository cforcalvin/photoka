# Offline AI generation

> **Last updated:** 2026-10-13 · **Status:** first version (SD 1.5, inpainting, LoRAs, SD-Turbo/LCM; native and ComfyUI engines)

Photoka can generate and inpaint images on your Mac, offline, in the style of Draw Things: a
movable frame (512 × 512 by default) sits over any document, you paint a mask inside it for
inpainting, and each result lands as a new layer. Nothing is downloaded until you ask for it.

Apple silicon only. On an Intel Mac, the Generate panel explains why it's off.

## Using it

1. **Window › Model Manager…**. Under **Suggested**, pick a model, read and accept its licence,
   and download it. Under **Import**, add your own `.safetensors` checkpoint, LoRA or VAE (the kind
   and family are detected from the file). **Installed** lists what's on disk.
2. **Window › Generate** opens the Generate panel and selects the **Generate Frame** tool.
3. Drag inside the frame to move it, or drag its edges to resize it (sides snap to multiples of
   64 px, 64–2048). Switch the tool to **Mask** to paint what should change; **Erase** removes
   mask. **Invert Mask** and **Clear Mask** are on the options bar.
4. Type a prompt, choose a model, and press **Generate** (or **Edit › Generate in Frame…**).
   - With no mask, the frame is generated from scratch (text to image); with **Strength** below 1,
     the frame's current pixels are the starting point (image to image).
   - With a mask, only the masked area changes (inpainting). Transparent pixels in the frame are
     added to the mask automatically, so dragging the frame past the canvas edge outpaints.
   - **Strength** works in both cases: 100% replaces the pixels, and lower values keep more of
     the original's shapes and colours (20–40% reworks it lightly). Every step you ask for
     runs, so low strength works with few-step models like SD-Turbo too.
5. The result is a new pixel layer named after the prompt, at the frame, with a layer mask when
   you inpainted. It's one undo step.

## Engines

| | Built-in (default) | ComfyUI (optional) |
|---|---|---|
| Runs on | Rust, [candle](https://github.com/huggingface/candle) on Metal, inside Photoka | A ComfyUI that Photoka installs and runs as a separate process on 127.0.0.1 |
| Setup | none | **Install ComfyUI…** in the Generate panel (about 4 GB: Python 3.12, PyTorch, ComfyUI) |
| LoRAs | up to 4 | up to 8 |
| Samplers | DDIM, Euler, Euler a, DPM++ 2M (Karras), LCM | the same names, mapped to ComfyUI's samplers and schedulers |

Both engines read the same models folder, so a model is downloaded once.

### Managed ComfyUI

ComfyUI is GPL-3.0. Photoka never bundles or links it: it is downloaded only when you click
**Install ComfyUI…** and accept its licence, and it runs as its own process.

- **Install** downloads a pinned [uv](https://github.com/astral-sh/uv) binary and a pinned ComfyUI
  release, verifies both by sha256, creates a Python 3.12 environment with uv, and installs
  ComfyUI's requirements. The pins are in `crates/ml/src/comfy/install.rs`.
- Everything lives in `~/Library/Application Support/Photoka/comfyui/`. **Remove** deletes it.
- **Start** launches ComfyUI on a free loopback port, pointed at Photoka's models folder through
  `--extra-model-paths-config`. Photoka stops it when it quits. Its output goes to
  `~/Library/Logs/Photoka/comfyui.log`.
- Generation uploads the frame (the mask as alpha), queues a workflow built in Rust (checkpoint,
  LoRA chain, `VAEEncodeForInpaint` for inpainting models or `SetLatentNoiseMask` otherwise,
  KSampler), follows progress over ComfyUI's websocket, and reads the result back. Cancel calls
  `/interrupt`.

## Models

The suggested catalog is bundled with the app (`crates/ml/catalog.toml`): each entry has a
pinned URL, sha256, size and licence. Downloads resume after an interruption and are verified
before they're used. Civitai downloads may need an API key; set it in the Model Manager (it is
stored in the macOS Keychain, never in the preferences file).

| Entry | Kind | Use |
|---|---|---|
| Stable Diffusion 1.5 | checkpoint | the general-purpose base |
| CyberRealistic Revamp, CyberRealistic (Final) | checkpoint | photoreal |
| DreamShaper 8, DreamShaper 8 Inpainting | checkpoint | digital art and illustration; the inpainting variant for masked edits |
| Comic Book Page Style | LoRA | comic-book look |
| 8bitdiffuser 64x | LoRA | pixel art for games |
| LCM-LoRA (SD 1.5) | LoRA | 4–8 step generation with any SD 1.5 checkpoint |
| SD-Turbo | checkpoint | 1–4 steps, no CFG |
| SD 1.5 VAE (ft-MSE), CLIP tokenizer | support files | |

Every licence is shown and accepted once, before its download. Some (SD-Turbo, several Civitai
models) restrict commercial use; the Model Manager shows the licence on each card.

The models folder defaults to `~/Library/Application Support/Photoka/models` and uses ComfyUI's
layout (`checkpoints/`, `loras/`, `vae/`, `embeddings/`, `tokenizers/`). You can move it.
Model files are never committed to this repository.

## Commands

Every action is a command, so the CLI, the control channel and MCP can drive generation. Params
are documented in each command's spec (`photoka-cli commands`).

| Area | Commands |
|---|---|
| Models | `ai.catalog.list`, `ai.models.list`, `ai.models.acceptLicense`, `ai.models.download`, `ai.models.import`, `ai.models.delete`, `ai.models.reveal`, `ai.models.setFolder`, `ai.civitai.setKey` |
| Frame and mask | `ai.frame.get`, `ai.frame.set`, `ai.frame.reset`, `ai.mask.stroke`, `ai.mask.fill`, `ai.mask.clear`, `ai.mask.invert` |
| Generation | `ai.generate` (`engine: "native"` or `"comfyui"`) |
| ComfyUI | `ai.comfy.status`, `ai.comfy.install` (`{"accept":true}`), `ai.comfy.uninstall`, `ai.comfy.start`, `ai.comfy.stop` |

Downloads, imports, generation and the ComfyUI install, start and removal run as background
jobs with progress and cancel. Shell-only entries: `window.generate` (`{"engine":"native"|"comfyui"}`),
`window.modelManager`, `edit.generateInFrame`.

```sh
photoka-cli run in.png \
  --cmd ai.frame.set --params '{"x":0,"y":0,"width":512,"height":512}' \
  --cmd ai.generate  --params '{"model":"sd_turbo.safetensors","prompt":"a red fox in snow","steps":2,"cfg":0,"sampler":"euler_a"}' \
  --out out.png
```

## Code map

- `crates/ml` (`photoka-ml`, L4): catalog, models folder (`store.rs`), downloader, request
  validation (`params.rs`), the native pipeline (`native/`: checkpoint key remap, CLIP/OpenCLIP,
  UNet, VAE, schedulers, img2img, inpainting, LoRA merge with a per-file factor cache), and
  `comfy/` (install, server process, HTTP/websocket client, workflow builder).
- `crates/engine/src/ai_cmds.rs`, `ai_gen_cmds.rs`, `ai_comfy_cmds.rs`: the commands and jobs.
- `crates/ui-egui/src/generate_ui.rs` (frame tool, mask overlay, Generate panel) and
  `model_manager_ui.rs`.

## Tests

CI never downloads models. Unit tests cover the catalog, downloader (local HTTP server: resume,
hash mismatch, cancel), key remapping, LoRA key parsing, schedulers, request validation,
ComfyUI workflow JSON (golden), and the ComfyUI client against a mock server.

Ignored end-to-end tests run real models when you point them at a models folder:

```sh
PHOTOKA_MODELS_DIR=/path/to/models cargo test -p photoka-ml --release -- --ignored real_
PHOTOKA_COMFY_DIR=/tmp/comfy PHOTOKA_MODELS_DIR=/path/to/models \
  cargo test -p photoka-ml --release -- --ignored real_install_start_and_generate
```

## Performance (M-series, models on an external exFAT drive)

- SD 1.5, 512 × 512: first load ~60 s from the external drive, then ~2.8 s per step with CFG.
- SD-Turbo, 2 steps: ~21 s per image once loaded.
- LoRA merge: ~5.6 s the first time, ~2 s when only a weight changes (factors are cached).
- ComfyUI text to image: ~100 s cold (process start and model load), faster once warm.

## Not yet

SDXL and Flux, ControlNet, textual-inversion embeddings in the native engine, upscaling,
Photoshop's Generative Fill/Expand UI, and ML Select Subject.

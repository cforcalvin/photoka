# Offline AI generation

> **Last updated:** 2026-10-16 · **Status:** SD 1.5 / Turbo / LoRAs, hardware-ranked Suggested, Remove Object, AI Upscale (ComfyUI)

Photoka can generate and inpaint images on your Mac, offline, in the style of Draw Things: a
movable frame (512 × 512 by default) sits over any document, you paint a mask inside it for
inpainting, and each result lands as a new layer. Nothing is downloaded until you ask for it.

Apple silicon only. On an Intel Mac, the Generate panel explains why it's off.

## Using it

1. **Window › Model Manager…** (or **Model Manager…** in the Generate tab). Under **Suggested**,
   models are ranked for this Mac (RAM, free disk on the models volume, Apple Silicon). Cards
   marked **Best for this Mac** fit your hardware; the line under each card explains why. Use
   **Download starter pack** for the best checkpoint plus missing companions (tokenizer / VAE)
   that still fit free disk. Read and accept each licence before download (the starter pack
   lists its members' licences first, then **Accept and download**). Under **Import**, add your own
   `.safetensors` checkpoint, LoRA or VAE, or a Real-ESRGAN `.pth` as kind `upscale`.
   **Installed** lists what's on disk.
2. **Window › Generate** opens the **Generate** tab, next to Layers, Channels and Paths, and
   selects the **Generate Frame** tool. Its settings start at DPM++ 2M Karras, 12 steps,
   guidance 7 and 100% strength.
3. **Lock frame to view** (on by default, on the options bar) works like Draw Things: the frame
   stays put in the middle of the canvas while you zoom (pinch, scroll, ⌘+/⌘−) and drag the
   image under it, and it always generates at its size (512 × 512, or another size from the
   menu next to the checkbox). Zoomed in 400% on a hand, the frame covers 128 × 128 document
   pixels: they're scaled up as the starting point, and the 512 × 512 result lands as a smart
   object scaled onto those 128 pixels, so its shape and size match while every generated pixel
   is kept (enlarge the image or rasterize later). The label above the frame reads
   "512 × 512 from 128 × 128 px". With the checkbox off, drag inside the frame to move it, or
   drag its edges to resize it (sides snap to multiples of 64 px, 64–2048).
   Switch the tool to **Mask** to paint what should change; **Erase mask paint** (brush-cleaning
   icon) removes mask strokes. The **eraser** icon runs **Remove Object** (inpaints the mask to
   clear the object); the **x** clears the mask. These controls also sit just above the frame;
   **Invert Mask** is on the options bar.
   The bar above the frame also has the prompt, a strength box (50% to start, so it reworks
   what's there) and a Generate button (or press Return in the prompt). It uses the Generate
   tab's model and other settings, with its own strength.
4. Type a prompt, choose a model, and press **Generate** (or **Edit › Generate in Frame…**).
   With no checkpoint installed, **Get a model for this Mac…** opens Model Manager on Suggested.
   - With no mask, the frame is generated from scratch (text to image); with **Strength** below 1,
     the frame's current pixels are the starting point (image to image).
   - With a mask, only the masked area changes (inpainting). Transparent pixels in the frame are
     added to the mask automatically, so dragging the frame past the canvas edge outpaints.
   - **Edit › Remove Masked Object…** (or the eraser on the frame bar) inpaints the mask at full
     strength with a cleanup prompt; prefers an installed SD 1.5 inpainting checkpoint.
   - **Strength** works in both cases: 100% replaces the pixels, and lower values keep more of
     the original's shapes and colours (20–40% reworks it lightly). Every step you ask for
     runs, so low strength works with few-step models like SD-Turbo too.
5. The result is a new layer named after the prompt, at the frame, with a layer mask when you
   inpainted: a pixel layer, or a smart object when it has more pixels than the frame covers
   (not in 32-bit documents). It's one undo step.
6. **Image › AI Upscale…** (or **Upscale…** in the Generate tab) runs Real-ESRGAN through
   ComfyUI. Download an upscaler from Suggested (`Real-ESRGAN x2plus` / `x4plus`); start ComfyUI
   first. Default scale and tile size come from this Mac's RAM (2× / smaller tiles on ≤ 12 GB).

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
| Models | `ai.catalog.list` (hardware ranking, `starterPack`, `fitsRam` / `recommendedForHw`), `ai.models.list`, `ai.models.acceptLicense`, `ai.models.download`, `ai.models.acceptStarterPack` (`{"accept":true}`, then `ai.models.download` each id in `toDownload`), `ai.models.import`, `ai.models.delete`, `ai.models.reveal`, `ai.models.setFolder`, `ai.civitai.setKey` |
| Frame and mask | `ai.frame.get`, `ai.frame.set`, `ai.frame.reset`, `ai.mask.stroke`, `ai.mask.fill`, `ai.mask.clear`, `ai.mask.invert` |
| Generation | `ai.generate` (`engine: "native"` or `"comfyui"`), `ai.removeMasked` |
| Upscale | `ai.upscale` (`scale` 2\|4, ComfyUI + Real-ESRGAN) |
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

## Performance (M4, 16 GB, models on an external exFAT drive)

- SD 1.5, 512 × 512: first load ~60 s from the external drive, including a short warm-up pass that
  compiles the GPU kernels, then ~2.8 s per step with CFG. Reading the prompt takes ~50 ms.
- The VAE (`native/vae.rs`) encodes and decodes the whole frame in one pass in f16, redoing a pass
  in f32 only if f16 overflows. Convolutions run in bands of rows and the GPU syncs between
  blocks, so memory stays bounded and there are no tile seams.

  | Frame | Encode | Decode |
  |---|---|---|
  | 512 × 512 | 1.6 s | 3.0 s |
  | 768 × 768 | 3.6 s | 7.6 s |
  | 1024 × 1024 | 6.1 s | 13.2 s |

  The old tiled f32 VAE took 10.0 s and 20.6 s at 512 × 512. Peak memory, model included, is
  7.5 GB at 1024 × 1024.
- SD-Turbo, 2 steps: ~5.9 s per image once loaded (was ~21 s).
- LoRA merge: ~5.6 s the first time, ~2 s when only a weight changes (factors are cached).
- ComfyUI text to image: ~100 s cold (process start and model load), faster once warm.

## Not yet

SDXL and Flux, ControlNet, textual-inversion embeddings in the native engine, native (non-ComfyUI)
Real-ESRGAN, Photoshop's Generative Fill/Expand UI, and ML Select Subject.

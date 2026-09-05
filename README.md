# Waterbody Segmentation Batch Pipeline

Batch inference pipeline that runs a pre-trained waterbody segmentation model
(`giswqs/s2-water-unetplusplus-efficientnet-b4` — UNet++ with an EfficientNet-B4
encoder) over large Sentinel-2 GeoTIFFs, producing a georeferenced binary water mask
aligned exactly to the input's CRS/transform/dimensions. Packaged as a self-contained
Docker container intended for AWS Batch-style managed batch compute. No training or
fine-tuning is involved — the model is fixed.

The image is processed in overlapping 512×512 tiles via `rasterio.windows.Window`
rather than loaded into memory whole, so it scales to inputs larger than available
RAM/VRAM.

There are **two backends**, both built and validated:

- **PyTorch** (`Dockerfile` / `docker-compose.yml`) — the primary, fully validated
  path. `waterbody-segmentation:latest`.
- **ONNX Runtime** (`Dockerfile.onnx` / `docker-compose.onnx.yml`) — a smaller,
  optional variant built after hitting a hard size ceiling on the PyTorch image (see
  "Why ONNX Runtime" below). `waterbody-segmentation:onnx`.

Both produce numerically equivalent output — see accuracy numbers below.

## Usage

**1. Place data.** Put the input GeoTIFF at `data/input.tif` (the `data/` directory
is mounted into the container and is not committed to the repo):

```
data/
└── input.tif
```

**2. Build the image:**

```bash
docker compose build                              # PyTorch backend
docker compose -f docker-compose.onnx.yml build    # ONNX Runtime backend (optional)
```

**3. Run it:**

```bash
docker compose run --rm waterseg          # PyTorch, writes data/output.tif
docker compose -f docker-compose.onnx.yml run --rm waterseg-onnx   # ONNX, writes data/output-onnx.tif
```

Both compose services are CPU-only (no GPU reservation declared) — that's the path
used for local development on a machine with no NVIDIA GPU. For a GPU run, use the
plain `docker run` form below directly.

The exact command the submission is evaluated against:

```bash
docker run --gpus all -v /path/to/data:/data waterbody-segmentation:latest \
  --input /data/input.tif --output /data/output.tif
```

The pipeline is device-agnostic: it uses the GPU automatically when `--gpus all` is
passed on a host with an NVIDIA GPU, and falls back to CPU otherwise — no flags or
code changes needed either way. Same is true of the ONNX backend, just via ONNX
Runtime's `CUDAExecutionProvider` → `CPUExecutionProvider` fallback instead of
`torch.cuda.is_available()`.

Both `docker-compose*.yml` files pin `platform: linux/amd64` — required because AWS
Batch GPU instances are x86_64, and building unpinned on a non-amd64 dev machine
(this project was built on Apple Silicon) either fails outright or silently resolves
the wrong wheels. Details below.

## Project structure

```
Dockerfile               PyTorch image: python:3.11-slim + pinned deps; bakes the
                          fixed model checkpoint in at build time (no network needed
                          at run time).
Dockerfile.onnx           ONNX Runtime image: multi-stage build. Builder stage
                          exports the pretrained model to ONNX using CPU-only
                          torch (never ships); runtime stage only installs
                          onnxruntime-gpu + rasterio, no torch at all.
docker-compose.yml        Builds + runs the PyTorch image against ./data.
docker-compose.onnx.yml   Builds + runs the ONNX image against ./data (separate
                          output file, so both can be run and diffed against
                          each other).
requirements.txt          Pinned Python dependencies (PyTorch backend).
src/waterseg/
├── tiling.py              Pure sliding-window tile-grid geometry — shared by both
│                          backends. Computes which windows to read (with context
│                          margin) and where each tile's cropped core prediction
│                          gets written, so cores tile the output with no gaps or
│                          double-writes.
├── model.py                PyTorch backend: downloads the fixed checkpoint from HF
│                          Hub, builds the UNet++/EfficientNet-B4 architecture,
│                          loads weights, places the model on the auto-detected
│                          device (GPU if available, else CPU).
├── inference.py             PyTorch backend: normalizes pixel values, zero-pads
│                          edge tiles, predicts, crops to the tile's core region.
├── pipeline.py               PyTorch backend: end-to-end driver — opens input
│                          once, opens output once with corrected georeferencing,
│                          streams tile-by-tile without holding the full image in
│                          memory.
├── cli.py                   PyTorch backend argparse entrypoint (--input, --output).
├── export_onnx.py            ONNX backend, build-time only: traces the same
│                          pretrained PyTorch model and exports it to model.onnx.
│                          Runs once inside the Dockerfile.onnx builder stage —
│                          never runs at inference time.
├── model_onnx.py             ONNX backend: picks CUDA vs CPU execution provider,
│                          loads the ONNX Runtime InferenceSession. Mirrors
│                          model.py's two responsibilities (pick device, load
│                          ready-to-run model), different backend.
├── inference_onnx.py         ONNX backend: same normalize → pad → predict → crop
│                          shape as inference.py, calling session.run() instead
│                          of a PyTorch forward pass.
├── pipeline_onnx.py          ONNX backend driver — same streaming read/predict/
│                          write shape as pipeline.py.
└── cli_onnx.py                ONNX backend argparse entrypoint.
```

## Architecture

```
                  ┌─────────────────────────────┐
                  │            run()              │
                  │    pipeline.py — orchestrator   │
                  └───────────────┬───────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
         tiling.py            model.py           inference.py
     generate_tiles()       load_model()        predict_tile()
    (tile-grid geometry:   (arch + weights,    (normalize → pad
     read/write windows,    placed on device)    → forward pass
     overlap-crop math)                           → crop core)
    called once, up front  called once, up front  called per tile
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  ▼
                          for each tile:
                    ┌──────────────────────────┐
                    │          rasterio           │
                    │  windowed read (input.tif)  │
                    │  windowed write (output.tif) │
                    └──────────────┬──────────────┘
                                  ▼
                              output.tif
```

`tiling.py` and `model.py` each run once, before the loop starts — the tile grid
doesn't depend on pixel values, and the model only needs to be loaded once. Only
`inference.py` and the rasterio read/write calls repeat per tile, which is what keeps
memory usage flat regardless of the input raster's size.

The ONNX backend (`pipeline_onnx.py` / `model_onnx.py` / `inference_onnx.py`) follows
the exact same shape — same orchestrator pattern, same `tiling.py` — just with an
ONNX Runtime `InferenceSession` in place of the PyTorch model object. One extra
build-time-only step precedes it:

```
model.pth (HF Hub) → export_onnx.py (traces model, CPU-only) → model.onnx
                                                                     │
                                                          baked into image at build
                                                          time, same as the PyTorch
                                                          checkpoint
```

## Why ONNX Runtime — upgrading the stack mid-project

The PyTorch image was the primary deliverable and is fully validated end to end. The
ONNX variant was added afterward, for one concrete reason: **image size**.

- Measured (not guessed) where the PyTorch image's size actually goes:
  `docker history` + `du -sh` inside a running container showed **~6.9GB on disk /
  ~3.9GB to pull**, and almost all of it (7.13GB of one layer) is CUDA support
  libraries pip installs alongside `torch` — 14 separate `nvidia-*-cu12` packages
  (cuBLAS, cuDNN, cuSOLVER, cuSPARSE, cuFFT, cuRAND, NCCL, cuSPARSELt, NVRTC,
  nvJitLink, CUPTI, cuFile, cuda_runtime, NVTX), not rasterio/GDAL as first assumed.
- Tried to strip the ones this model's forward pass never actually calls (no
  distributed training, no sparse ops, no FFT). **That assumption was wrong.**
  PyTorch's Linux wheel unconditionally preloads its *entire* declared CUDA
  dependency set into `torch._C` at import time — not lazy per-op loading. Verified
  this directly: removing `cusparselt`, `cufile`, or `nccl` each broke `import
  torch` outright, on CPU, before any model code ran, 3 times in a row. Only
  `triton` (542MB, `torch.compile()`-only) was genuinely safe to drop. That's the
  real ceiling for trimming this dependency list — any further reduction needs a
  different lever.
- **ONNX Runtime is that lever.** `onnxruntime-gpu`'s own dependency metadata (unlike
  torch's) only pulls in CUDA libraries via opt-in `[cuda,cudnn]` extras — no
  mandatory preload of the full CUDA toolkit's worth of packages. Exported the fixed
  model once (build-time only, via a CPU-only exporter stage that never ships) to
  `model.onnx`, and the runtime image only needs `onnxruntime-gpu` + `rasterio` —
  no `torch` in the final image at all.
- Result: **2.23GB total**, vs. ~3.99GB content / ~6.9GB on disk for the PyTorch
  image. Verified end-to-end (not just at export time) that this loses nothing:
  ONNX output vs. PyTorch logits matched to 7e-5 at export, and the full pipeline
  run against `expected_mask.tif` matched PyTorch's own accuracy numbers within
  rounding noise (see table below).
- Both images are kept — the PyTorch one is the proven, primary deliverable; the
  ONNX one is a from-scratch, fully separate build (new Dockerfile, new compose
  file, new Python modules with an `_onnx` suffix) so the working path was never at
  risk while exploring this. Nothing in `Dockerfile`/`docker-compose.yml`/the
  non-`_onnx` Python modules was touched to build it.

### Stack comparison

| | PyTorch backend | ONNX Runtime backend |
|---|---|---|
| Inference engine | `torch` + `segmentation-models-pytorch` | `onnxruntime-gpu` |
| Image size (total) | ~10.6GB / 3.99GB content | 2.23GB |
| CUDA dependency model | mandatory, preloads full toolkit set | opt-in extras only |
| GPU→CPU fallback | `torch.cuda.is_available()` | `ort.get_available_providers()` |
| Accuracy vs. ground truth | 99.941% acc / 0.9903 IoU | 99.9407% acc / 0.9903 IoU |
| Status | Primary, fully validated | Optional, size-optimized, CPU-fallback validated; real GPU execution not yet tested on hardware |

## Prerequisites

Key libraries this project depends on, and why:

- **rasterio** — windowed reads/writes of large GeoTIFFs via `rasterio.windows.Window`,
  and propagating CRS/affine-transform metadata from input to output. Used throughout
  `tiling.py` and both pipeline modules; this is what makes memory-bounded processing
  of an arbitrarily large raster possible at all.
- **torch** — runs the segmentation model, device-agnostically (CPU/GPU), in the
  PyTorch backend. Used in `model.py` and `inference.py`.
- **segmentation-models-pytorch** — provides the UNet++ architecture with an
  EfficientNet-B4 encoder that the pretrained checkpoint's weights are loaded into.
  Used in `model.py` and (CPU-only, build-time) `export_onnx.py`.
- **huggingface_hub** — downloads the fixed pretrained checkpoint and its config from
  the Hugging Face Hub. Used in `model.py`; the download is triggered once at Docker
  build time so the container needs no network access at inference time.
- **onnx** / **onnxruntime-gpu** — `onnx` (build-time only) serializes the traced
  model in `export_onnx.py`; `onnxruntime-gpu` runs it at inference time in the ONNX
  backend (`model_onnx.py`, `inference_onnx.py`). Pinned to `1.22.0` deliberately —
  verified via its actual wheel metadata that this version targets CUDA 12, matching
  the driver requirement already proven against; newer releases moved to CUDA 13.
- **numpy** — pixel-array manipulation (normalization, edge-tile zero-padding,
  cropping) between rasterio and the model, in both backends.
- **tqdm** — progress reporting over the tile loop, useful for visibility into a
  long-running batch job. Used in both pipeline modules.

## Findings & errors along the way

Real bugs and dead ends hit while building this, and how each was actually resolved
(full detail with root-cause analysis lives in `problems.md`/`learnings.md`):

- **A silent correctness bug in tile-batching, caught only by checking accuracy.**
  `preds[i][row][col]` instead of `preds[i][row, col]` — two sequential 1-D slices,
  not a 2-D index. Wrong shape, but `rasterio` doesn't validate a write's shape
  against its window, so the pipeline ran clean and produced a plausible-looking
  output — accuracy silently dropped from ~99.8% to 80.4%. "It ran without error" is
  not proof of correctness for this pipeline; every change gets re-checked against
  `expected_mask.tif`, no exceptions.
- **Pixel normalization wasn't documented anywhere authoritative.** The model card
  doesn't state it, and the obvious reference implementation's heuristic (`/255.0`)
  looks wrong for 16-bit Sentinel-2 data (values up to ~14,000). Tested 5 candidate
  scalings empirically against a real mixed tile instead of trusting intuition —
  `/255.0` was actually correct (99.41% acc), while the "physically sensible"
  `/10000.0` reflectance scaling degenerated to predicting water everywhere (51.8%
  acc). A default that looks naive is a hypothesis to test, not something to accept
  or reject on sight.
- **`docker-compose.yml` had never actually been run before it was "done."**
  Building it (rather than just the raw `docker buildx build` used up to that point)
  surfaced a real portability gap: no `platform` pin meant it defaulted to native
  `arm64` on this dev Mac, where `rasterio` has no prebuilt wheel at all (needs GDAL
  dev headers to compile from source — not installed, build fails outright). Fixed
  with an explicit `platform: linux/amd64` pin. The exact same class of bug recurred
  later for the ONNX image's `onnxruntime-gpu` dependency — same fix.
- **Guessed which CUDA libraries were "safe to strip" from the image — guessed
  wrong, 3 times in a row.** Assumed libraries an op-free forward pass never
  dispatches to (`cusparselt`, `cufile`, `nccl`) must be lazily loaded and therefore
  droppable. Each one broke `import torch` outright when removed, even on CPU with
  no GPU involved — proving torch's Linux wheel preloads its entire CUDA dependency
  set unconditionally, not per-op. This is what motivated moving to ONNX Runtime
  instead of continuing to guess at the strip list.
- **A missing system library only surfaced at `docker run`, not at build time.**
  `rasterio`'s bundled GDAL dynamically links `libexpat`, absent from
  `python:3.11-slim`. The build's own sanity check didn't catch it because that
  check only imported `model.py`, which never imports `rasterio`. Fixed by adding
  `libexpat1` and widening the build-time check to import the actual pipeline
  module, so this class of bug fails fast at build time going forward.
- **`docker images`' human-readable size column is not trustworthy on this setup** —
  it reported a ~7.6GB drop from removing one 542MB package. Real ground truth:
  `du -sh` inside a running container (actual unpacked size) and
  `docker save | wc -c` (actual transferable bytes). Even the original "~12GB
  image" framing this whole optimization effort started from turned out to be
  measured wrong by the same column.
- **A CUDA-torch reintroduction bug in the ONNX exporter stage.** The exporter
  installed CPU-only `torch` via `--index-url .../whl/cpu` in one `pip install`,
  then `segmentation-models-pytorch` in a second, unpinned one — whose transitive
  `torchvision` dependency got resolved from default PyPI, silently pulling the
  *default CUDA* `torch` back in and undoing the CPU-only pin. Fixed by pinning
  `torchvision` explicitly in the same `--index-url` command as torch.
- **`onnxruntime-gpu==1.22.0` has no `arm64` Linux wheel at all** (checked PyPI's
  file list directly) — building unpinned-platform on this arm64 Mac silently
  offered `1.29.0` instead, which targets CUDA 13, not the CUDA 12 version
  deliberately chosen to match the proven driver requirement. Same `platform:
  linux/amd64` compose pin fixed it.

## Validation results

Full-image comparison against the provided `expected_mask.tif` (165.3M pixels), both
backends, real sample data, exact required invocation shape:

| | Pixel accuracy | IoU | Precision | Recall |
|---|---|---|---|---|
| PyTorch (native + Docker) | 99.941% | 0.99033 | 0.99492 | 0.99536 |
| ONNX Runtime (Docker, CPU fallback) | 99.9407% | 0.9903 | 0.9949 | 0.9954 |

Both confirmed bit-identical/near-identical across native-vs-Docker and crop-vs-full-
scale runs — not just "the container ran without error."

**Not yet verified:** real GPU execution on actual NVIDIA hardware, for either
backend — both have only been confirmed to fall back to CPU correctly when no GPU is
present (true of every machine used to build this so far).

## AI usage disclosure

This project was built by me, with Claude Code used as a research assistant and
pair-programming mentor throughout — not as a code generator I copy-pasted from. For
each module (tiling, model loading, the inference loop, stitching, GeoTIFF I/O, the
Dockerfile, docker-compose, and later the ONNX Runtime variant), I had it first
explain the relevant concepts (e.g. what a rasterio `Window` is, why tile overlap
avoids boundary artifacts, how CRS/transform metadata has to propagate to the output
raster, what "device-agnostic" means for `torch.cuda.is_available()` vs. ONNX
Runtime's execution providers) and walk through the design trade-offs before any
code was written, so I could reason about and defend the decisions myself rather
than just accept output. It was also used to debug problems as they came up (e.g.
tile-boundary mismatches, a Docker GPU/base-image issue, the CUDA-library-stripping
dead end, the ONNX exporter's torchvision bug) by explaining root causes rather than
just supplying a fix. ChatGPT was used in a similar supporting capacity — mainly for
quick research and clarifying questions on rasterio/GeoTIFF and PyTorch concepts
while working through the assignment. I wrote and understand the resulting code end
to end, including both backends and why the second one exists.

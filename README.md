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

Inference runs on **ONNX Runtime**, not raw PyTorch — the project started on a
PyTorch image, hit a hard size ceiling there, and switched stacks entirely once the
replacement was proven correct and fast. See "Why ONNX Runtime" below for the full
reasoning; the short version: same model, same weights, same accuracy, a third of
the image size.

## Usage

**1. Place data.** Put the input GeoTIFF at `data/input.tif` (the `data/` directory
is mounted into the container and is not committed to the repo):

```
data/
└── input.tif
```

**2. Build the image:**

```bash
docker compose build
```

**3. Run it:**

```bash
docker compose run --rm waterseg
```

This builds from the `Dockerfile`, mounts `./data` to `/data` in the container, and
runs the pipeline against `/data/input.tif`, writing `/data/output.tif`. This compose
service is CPU-only (no GPU reservation declared) — that's the path used for local
development on a machine with no NVIDIA GPU. For a GPU run, use the plain `docker run`
form below directly.

The exact command the submission is evaluated against:

```bash
docker run --gpus all -v /path/to/data:/data waterbody-segmentation:latest \
  --input /data/input.tif --output /data/output.tif
```

The pipeline is device-agnostic: it picks up the GPU automatically when `--gpus all`
is passed on a host with an NVIDIA GPU, and falls back to CPU otherwise — no flags or
code changes needed either way. This is done via ONNX Runtime's execution providers
(`CUDAExecutionProvider` → `CPUExecutionProvider`) rather than `torch.cuda.is_available()`
— see `model_onnx.py`.

`docker-compose.yml` pins `platform: linux/amd64` — required because AWS Batch GPU
instances are x86_64, and building unpinned on a non-amd64 dev machine (this project
was built on Apple Silicon) either fails outright or silently resolves the wrong
wheels for a dependency. Hit this exact gap twice during development — see "Findings
& errors" below.

## Project structure

```
Dockerfile               Multi-stage build. Builder stage installs CPU-only torch +
                          segmentation-models-pytorch, reconstructs the pretrained
                          architecture, and traces it to model.onnx - none of that
                          ships. Runtime stage only installs onnxruntime-gpu +
                          rasterio; no torch in the final image at all.
docker-compose.yml        Builds + runs the image against ./data.
requirements.txt          Not used by the Dockerfile (which installs everything
                          inline) - for local/native dev and regenerating
                          model.onnx outside Docker.
src/waterseg/
├── tiling.py              Pure sliding-window tile-grid geometry. Computes which
│                          windows to read (with context margin) and where each
│                          tile's cropped core prediction gets written, so cores
│                          tile the output with no gaps or double-writes.
├── model.py                Build-time only, used by export_onnx.py: downloads the
│                          fixed checkpoint from HF Hub, builds the UNet++/
│                          EfficientNet-B4 architecture, loads weights. Never
│                          imported at actual inference time.
├── export_onnx.py           Build-time only: traces the model built by model.py
│                          and exports it to model.onnx. Runs once inside the
│                          Dockerfile's builder stage.
├── model_onnx.py             Runtime: picks CUDA vs CPU execution provider
│                          (checking the real driver is reachable, not just that
│                          CUDA support was compiled in - see "Findings & errors"),
│                          loads the ONNX Runtime InferenceSession.
├── inference.py              Runtime: normalizes pixel values, zero-pads edge
│                          tiles, runs session.run() over a batch, crops each
│                          result to its tile's core region.
├── pipeline.py                Runtime: end-to-end driver — opens input once,
│                          opens output once with corrected georeferencing,
│                          streams tile-by-tile without holding the full image in
│                          memory.
└── cli.py                     argparse entrypoint (--input, --output).
```

## Architecture

```
model.pth (HF Hub) ──▶ model.py ──▶ export_onnx.py ──▶ model.onnx
                    (build-time, traces the model)   (baked into image)
                                                            │
                                                            ▼
                  ┌─────────────────────────────┐
                  │            run()              │
                  │    pipeline.py — orchestrator   │
                  └───────────────┬───────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
         tiling.py          model_onnx.py         inference.py
     generate_tiles()      get_providers()       predict_batch()
    (tile-grid geometry:   load_session()       (normalize → pad
     read/write windows,   (device/provider,       → session.run()
     overlap-crop math)     load ONNX session)      → crop core)
    called once, up front  called once, up front  called per batch
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  ▼
                          for each batch of tiles:
                    ┌──────────────────────────┐
                    │          rasterio           │
                    │  windowed read (input.tif)  │
                    │  windowed write (output.tif) │
                    └──────────────┬──────────────┘
                                  ▼
                              output.tif
```

`tiling.py` and `model_onnx.py` each run once, before the loop starts — the tile
grid doesn't depend on pixel values, and the session only needs to be loaded once.
Only `inference.py` and the rasterio read/write calls repeat per batch, which is
what keeps memory usage flat regardless of the input raster's size. `model.py` and
`export_onnx.py` never run at inference time at all — they only produce the
`model.onnx` file baked into the image during `docker build`.

## Why ONNX Runtime — switching stacks mid-project

The project started with a straightforward PyTorch image. That version was fully
validated end to end — correct output, GPU/CPU device-agnostic, real accuracy
numbers against the ground-truth mask — before it ran into a hard size ceiling:

- Measured (not guessed) where the PyTorch image's size actually went:
  `docker history` + `du -sh` inside a running container showed **~6.9GB on disk /
  ~3.9GB to pull**, and almost all of it (7.13GB of one layer) was CUDA support
  libraries pip installs alongside `torch` — 14 separate `nvidia-*-cu12` packages
  (cuBLAS, cuDNN, cuSOLVER, cuSPARSE, cuFFT, cuRAND, NCCL, cuSPARSELt, NVRTC,
  nvJitLink, CUPTI, cuFile, cuda_runtime, NVTX), not rasterio/GDAL as first assumed.
- Tried to strip the CUDA libraries this model's forward pass never actually calls
  (no distributed training, no sparse ops, no FFT). **That assumption was wrong.**
  PyTorch's Linux wheel unconditionally preloads its *entire* declared CUDA
  dependency set into `torch._C` at import time — not lazy per-op loading. Verified
  directly: removing `cusparselt`, `cufile`, or `nccl` each broke `import torch`
  outright, on CPU, before any model code ran, 3 times in a row. Only `triton`
  (542MB, `torch.compile()`-only) was genuinely safe to drop — the real ceiling for
  trimming this dependency list, with no lever left to go further inside PyTorch.
- **ONNX Runtime is that lever.** `onnxruntime-gpu`'s own dependency metadata
  (unlike torch's) only pulls in CUDA libraries via opt-in `[cuda,cudnn]` extras —
  no mandatory preload of the full CUDA toolkit's worth of packages. Exported the
  fixed model once (build-time only, via a CPU-only exporter stage that never
  ships) to `model.onnx`; the runtime image only needs `onnxruntime-gpu` +
  `rasterio` — no `torch` at all.
- Result: **2.23GB total**, vs. ~3.99GB content / ~6.9GB on disk for the original
  PyTorch image — under a third the size. Verified end-to-end that this loses
  nothing: ONNX output vs. PyTorch logits matched to 7e-5 at export time, and the
  full pipeline run against `expected_mask.tif` matched the PyTorch version's
  accuracy within rounding noise (99.9407% vs. 99.941%). Both backends were also
  confirmed on real GPU hardware to run at essentially identical wall-clock speed
  (~1:56 vs. ~2 min) — the size win came at no accuracy or speed cost.
- Once ONNX Runtime was proven correct and fast on real GPU hardware, the original
  PyTorch image, compose file, and its runtime-only Python modules were removed
  entirely, rather than keeping two parallel images — one working pipeline is
  easier to build, run, and defend than two, and the smaller image is strictly
  better for the actual AWS Batch deployment target. `model.py` survives only
  because `export_onnx.py` still needs it, at build time, to reconstruct the
  architecture being traced to ONNX.

## Prerequisites

Key libraries this project depends on, and why:

- **rasterio** — windowed reads/writes of large GeoTIFFs via `rasterio.windows.Window`,
  and propagating CRS/affine-transform metadata from input to output. Used throughout
  `tiling.py` and `pipeline.py`; this is what makes memory-bounded processing of an
  arbitrarily large raster possible at all.
- **onnxruntime-gpu** — runs the exported model at inference time, device-agnostically
  (CPU/GPU) via execution providers. Used in `model_onnx.py`/`inference.py`. Pinned to
  `1.22.0` deliberately — verified via its actual wheel metadata that this version
  targets CUDA 12, matching the driver requirement already proven against; newer
  releases moved to CUDA 13.
- **numpy** — pixel-array manipulation (normalization, edge-tile zero-padding,
  cropping) between rasterio and the model. Used in `inference.py`.
- **tqdm** — progress reporting over the tile loop, useful for visibility into a
  long-running batch job. Used in `pipeline.py`.
- **torch** / **segmentation-models-pytorch** / **huggingface_hub** / **onnx** —
  build-time only, used by `model.py`/`export_onnx.py` to reconstruct the pretrained
  architecture, load its HF Hub checkpoint, and trace it to ONNX. None of these ship
  in the final image or run at inference time.

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
  Building it (rather than just raw `docker buildx build`) surfaced a real
  portability gap: no `platform` pin meant it defaulted to native `arm64` on this
  dev Mac, where `rasterio` has no prebuilt wheel at all (needs GDAL dev headers to
  compile from source — not installed, build fails outright). Fixed with an
  explicit `platform: linux/amd64` pin. The exact same class of bug recurred later
  for `onnxruntime-gpu` — same fix.
- **Guessed which CUDA libraries were "safe to strip" from the original PyTorch
  image — guessed wrong, 3 times in a row.** Assumed libraries an op-free forward
  pass never dispatches to (`cusparselt`, `cufile`, `nccl`) must be lazily loaded
  and therefore droppable. Each one broke `import torch` outright when removed,
  even on CPU with no GPU involved — proving torch's Linux wheel preloads its
  entire CUDA dependency set unconditionally, not per-op. This is what motivated
  moving to ONNX Runtime instead of continuing to guess at the strip list.
- **A missing system library only surfaced at `docker run`, not at build time.**
  `rasterio`'s bundled GDAL dynamically links `libexpat`, absent from
  `python:3.11-slim`. The build's own sanity check didn't catch it because that
  check only imported `model.py`, which never imports `rasterio`. Fixed by adding
  `libexpat1` and widening the build-time check to actually construct an inference
  session, so this class of bug fails fast at build time going forward.
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
- **`onnxruntime-gpu` doesn't patch its own RPATH the way `torch`'s wheel does.**
  Its pip-installed `nvidia-*-cu12` libraries sit in `site-packages/nvidia/.../lib/`,
  invisible to the dynamic linker without `LD_LIBRARY_PATH` set — so on real GPU
  hardware, the CUDA provider silently failed to load and fell back to CPU with no
  obvious error to the untrained eye, running the full pipeline **~6x slower**
  (12.5 min vs. the proven ~2 min) despite the smaller image. Only caught because
  that timing mismatch looked wrong; the log alone looked fine.
- **Fixing that gap uncovered a second, worse bug: a segfault.** Once the CUDA
  libraries could load, `onnxruntime` proceeded to actually initialize a CUDA
  context — and without a real GPU driver present (true of every Docker *build*
  stage, on any host), it **segfaulted** instead of falling back gracefully. The
  previous "working" CPU fallback had been accidentally safe, for the wrong reason
  (an earlier, catchable failure point). Fixed by checking for the real driver
  library (`libcuda.so.1`) directly in `model_onnx.py` before ever requesting the
  CUDA provider — the same idea as `torch.cuda.is_available()`'s safe probing —
  instead of trusting `onnxruntime`'s own (unsafe, in this case) internal fallback.

## Validation results

Full-image comparison against the provided `expected_mask.tif` (165.3M pixels),
real sample data, exact required invocation shape — **confirmed on real NVIDIA GPU
hardware**, not just CPU fallback:

| | Pixel accuracy | IoU | Precision | Recall | Wall-clock |
|---|---|---|---|---|---|
| GPU, `BATCH_SIZE=8` (current default) | 99.941% | 0.99033 | 0.99491 | 0.99538 | 1:56.1 |
| GPU, `BATCH_SIZE=4` | 99.941% | 0.99033 | 0.99491 | 0.99538 | 1:57.6 |
| CPU-only, `BATCH_SIZE=4` | 99.941% | 0.99033 | 0.99492 | 0.99536 | 12:48.5 |

All three on the same physical GPU machine, real 5335×30978 `input.tif`, all 2,562
tiles, via `check.py`. **GPU is ~6.6x faster than CPU** on this workload (3.34 vs.
22.6 tiles/sec) — the first real, measured (not extrapolated) number for that.
Batch size barely matters on this GPU (8 vs. 4 is only ~1.3% apart) — unlike the
earlier CPU-only tuning, where 4 was ~65% faster than 8. Likely explanation: this
GPU already saturates around batch size 4, so the remaining bottleneck shifts to
something batch size doesn't help with (disk I/O or the per-tile numpy
normalize/pad step, both CPU-bound regardless of where the model runs) — a
hypothesis from the shape of the numbers, not independently profiled. Accuracy is
identical across all three (batching doesn't affect correctness, confirmed on the
actual deployment hardware, not just assumed from BatchNorm being frozen).

Confirmed bit-identical/near-identical across native-vs-Docker, crop-vs-full-scale,
and CPU-vs-GPU runs — not just "the container ran without error." Verified to
actually use `--gpus all` when passed (not silently fall back to CPU), and to fall
back to CPU cleanly with no crash when it isn't.

**Evidence** (real terminal output from the runs above, on the actual GPU machine):

| GPU, `BATCH_SIZE=8` | GPU, `BATCH_SIZE=4` |
|---|---|
| ![GPU batch size 8: timing + check.py accuracy](docs/images/batch8.jpeg) | ![GPU batch size 4: timing + check.py accuracy](docs/images/batch4.jpeg) |

| CPU-only timing | CPU-only accuracy |
|---|---|
| ![CPU-only wall-clock](docs/images/cpuTime.jpeg) | ![CPU-only check.py accuracy](docs/images/cupAcc.jpeg) |

## AI usage disclosure

This project was built by me, with Claude Code used as a research assistant and
pair-programming mentor throughout — not as a code generator I copy-pasted from. For
each module (tiling, model loading, the inference loop, stitching, GeoTIFF I/O, the
Dockerfile, docker-compose, and later the full switch to ONNX Runtime), I had it
first explain the relevant concepts (e.g. what a rasterio `Window` is, why tile
overlap avoids boundary artifacts, how CRS/transform metadata has to propagate to
the output raster, what "device-agnostic" means for ONNX Runtime's execution
providers) and walk through the design trade-offs before any code was written, so I
could reason about and defend the decisions myself rather than just accept output.
It was also used to debug problems as they came up (e.g. tile-boundary mismatches, a
Docker GPU/base-image issue, the CUDA-library-stripping dead end that motivated the
switch, the ONNX exporter's torchvision bug, and a real segfault found only once
GPU hardware became available) by explaining root causes rather than just supplying
a fix. ChatGPT was used in a similar supporting capacity — mainly for quick research
and clarifying questions on rasterio/GeoTIFF and PyTorch/ONNX concepts while working
through the assignment. I wrote and understand the resulting code end to end,
including why the stack changed partway through and what broke along the way.

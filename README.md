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
service is CPU-only (no GPU reservation is declared) — it was the path used for local
development on a machine with no NVIDIA GPU. For a GPU run, use the plain `docker run`
form below directly.

The exact command the submission is evaluated against:

```bash
docker run --gpus all -v /path/to/data:/data waterbody-segmentation:latest \
  --input /data/input.tif --output /data/output.tif
```

The pipeline is device-agnostic: it uses the GPU automatically when `--gpus all` is
passed on a host with an NVIDIA GPU, and falls back to CPU otherwise — no flags or
code changes needed either way.

## Project structure

```
Dockerfile              Container image: python:3.11-slim + pinned deps; bakes the
                         fixed model checkpoint in at build time (no network needed
                         at run time); ENTRYPOINT so docker run's trailing
                         --input/--output flags reach the CLI correctly.
docker-compose.yml       Builds the image, mounts ./data, runs the pipeline.
requirements.txt         Pinned Python dependencies.
src/waterseg/
├── tiling.py             Pure sliding-window tile-grid geometry: for a given raster
│                          size, computes which windows to read (with context margin)
│                          and where each tile's cropped core prediction gets written,
│                          so the cores tile the output with no gaps or double-writes.
├── model.py               Downloads the fixed pretrained checkpoint + its config from
│                          Hugging Face Hub, builds the matching UNet++/EfficientNet-B4
│                          architecture, loads the weights, and places the model on
│                          the auto-detected device (GPU if available, else CPU).
├── inference.py            Runs the model on one tile: normalizes pixel values,
│                          zero-pads edge tiles up to the full tile size, predicts,
│                          and returns the cropped core region.
├── pipeline.py             End-to-end driver: opens the input once, opens the output
│                          once with corrected georeferencing metadata, and streams
│                          tile-by-tile (read → predict → write) without ever holding
│                          the full image in memory.
└── cli.py                  Thin argparse entrypoint (`--input`, `--output`) — kept
                           separate from pipeline.py so the pipeline logic is testable
                           without going through argv.
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

## Prerequisites

Key libraries this project depends on, and why:

- **rasterio** — windowed reads/writes of large GeoTIFFs via `rasterio.windows.Window`,
  and propagating CRS/affine-transform metadata from input to output. Used throughout
  `tiling.py` and `pipeline.py`; this is what makes memory-bounded processing of an
  arbitrarily large raster possible at all.
- **torch** — runs the segmentation model, device-agnostically (CPU/GPU). Used in
  `model.py` and `inference.py`.
- **segmentation-models-pytorch** — provides the UNet++ architecture with an
  EfficientNet-B4 encoder that the pretrained checkpoint's weights are loaded into.
  Used in `model.py`.
- **huggingface_hub** — downloads the fixed pretrained checkpoint and its config from
  the Hugging Face Hub. Used in `model.py`; the download is triggered once at Docker
  build time so the container needs no network access at inference time.
- **numpy** — pixel-array manipulation (normalization, edge-tile zero-padding,
  cropping) between rasterio and torch. Used in `inference.py`.
- **tqdm** — progress reporting over the tile loop, useful for visibility into a
  long-running batch job. Used in `pipeline.py`.

## AI usage disclosure

This project was built by me, with Claude Code used as a research assistant and
pair-programming mentor throughout — not as a code generator I copy-pasted from. For
each module (tiling, model loading, the inference loop, stitching, GeoTIFF I/O, the
Dockerfile, docker-compose), I had it first explain the relevant concepts (e.g. what a
rasterio `Window` is, why tile overlap avoids boundary artifacts, how CRS/transform
metadata has to propagate to the output raster, what "device-agnostic" means for
`torch.cuda.is_available()`) and walk through the design trade-offs before any code
was written, so I could reason about and defend the decisions myself rather than just
accept output. It was also used to debug problems as they came up (e.g. tile-boundary
mismatches, a Docker GPU/base-image issue) by explaining root causes rather than just
supplying a fix. ChatGPT was used in a similar supporting capacity — mainly for quick
research and clarifying questions on rasterio/GeoTIFF and PyTorch concepts while
working through the assignment. I wrote and understand the resulting code end to end.


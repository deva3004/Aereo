# CLAUDE.md — Waterbody Segmentation MLOps Pipeline

## Project Context

This is an MLOps internship assignment. The deadline is one week from receipt.
Point of contact: Shreyansh Agarwal (+91 7014100328), submissions to shreyansh@aereo.io.

### Task

Productionize a **pre-trained** waterbody segmentation model for satellite imagery into
an end-to-end batch inference pipeline, packaged as a Docker container deployable on
AWS Batch (or similar managed batch compute). No training or fine-tuning is involved —
the model is fixed and already trained.

### Model

- `giswqs/s2-water-unetplusplus-efficientnet-b4` (Hugging Face)
- UNet++ architecture with an EfficientNet-B4 encoder
- Performs semantic segmentation on Sentinel-2 multispectral imagery to identify surface water
- Trained on the Earth Surface Water Dataset
- Model card: https://huggingface.co/giswqs/s2-water-unetplusplus-efficientnet-b4
- Dataset card: https://huggingface.co/datasets/giswqs/s2-water-dataset

### I/O Contract

- **Input:** a single large GeoTIFF (externally mounted, satellite imagery)
- **Output:** a georeferenced GeoTIFF mask with merged predictions, aligned exactly to
  the input's CRS/transform/dimensions
- A sample input GeoTIFF and its expected output mask are provided for validation.

### Required Verification Command

The submission will be evaluated by running exactly this:

```
docker run --gpus all -v /path/to/data:/data <your-imagename> --input /data/input.tif --output /data/output.tif
```

The container MUST work correctly against this exact invocation.

## Scope

### In scope
- Python pipeline that reads/writes large GeoTIFFs
- Memory-efficient inference via tiling / sliding windows (`rasterio.windows.Window`)
  so images larger than available memory can be processed — the provided sample input
  is the benchmark for this
- Seamless stitching of tile-wise predictions into one cohesive output raster
  (watch tile-boundary artifacts — overlap + blending or careful cropping)
- Self-contained, production-ready Dockerfile
- GPU/CPU adaptability: CUDA-compatible base image, device-agnostic code
  (must use GPU when `--gpus all` is passed, must run gracefully CPU-only otherwise)
- CLI via `argparse` accepting `--input` and `--output`

### Optional (nice-to-have, not required)
- Inference performance optimization
- Modular architecture for onboarding new segmentation models

### Explicitly out of scope — do not build these
- Model training / re-training
- Any user-facing web app, dashboard, or visualization interface
- A real-time inference API or online serving endpoint
- Actual provisioning/configuration of AWS Batch, ECS, EC2, or Kubernetes
  (the container just needs to be *structured* to be deployable that way)
- Production-grade orchestration, monitoring, logging, or CI/CD beyond what's needed
  to run the containerized batch job

## Deliverables

1. **GitHub repository**
   - Clean, well-structured implementation
   - `README.md` with:
     - Usage instructions (placing data, building image, running via docker compose)
     - Project structure overview (purpose of key files/dirs)
     - Prerequisites: major libraries/deps with a brief reason each was chosen and
       where it's used (skip minor/incidental deps)
     - A mandatory one-paragraph AI usage disclosure — which AI tools were used and
       how (this project uses Claude Code; disclose it honestly and specifically)
   - `docker-compose.yml` that builds the image from the Dockerfile, mounts a local
     data directory into the container, and runs the pipeline with `--input`/`--output`
2. **Presentation video** — demo of the full pipeline on the sample dataset, optionally
   covering low-level design and key architectural decisions

## Evaluation Criteria (what's actually being graded)

1. Problem understanding & solution completeness
2. **Codebase understanding & ownership** — must be able to explain every component
3. Code quality & low-level design, and the reasoning behind decisions
4. Deployment readiness (containerization quality, ease of running as a batch workload)
5. Initiative & optimizations beyond the core requirement

**Critical:** if shortlisted, expect a review where the author must explain every line,
defend architectural trade-offs (tiling strategy, raster memory handling, etc.), and
adapt the code live to new constraints introduced on the spot. AI-assisted code that
the author can't explain or defend will fail this review. This is the single biggest
risk in this assignment.

## How Claude Code should work with me on this project

This is a learning exercise, not just a delivery. I am doing this myself with Claude
Code as a teaching assistant / pair programmer, and I need to be able to defend and
explain every part of this codebase in a live review afterward. Follow this workflow
for every module/component (e.g., tiling/windowed reading, model loading, inference
loop, stitching logic, GeoTIFF writing, CLI, Dockerfile, docker-compose):

1. **Teach first.** Before writing any code for a module, explain:
   - What the module needs to do and why it's needed
   - The relevant concepts (e.g., what a rasterio Window is, why tile overlap matters,
     how CRS/transform propagate to an output raster, why device-agnostic code checks
     `torch.cuda.is_available()`, etc.)
   - The design options/trade-offs available and which one we're picking and why
2. **Then write the code.** Keep it clean and modular, matching what was just taught —
   don't introduce unexplained magic.
3. **Flag important modules in advance.** Before starting a module that is central to
   the evaluation (tiling/windowed I/O, stitching, GPU/CPU device handling, Dockerfile
   GPU base image setup) or that is easy to get subtly wrong (edge-tile handling,
   georeferencing propagation, memory limits), say so explicitly up front — e.g.
   "this one matters a lot for the review, pay close attention" — before diving in.
4. **After each module**, briefly summarize what was built and why, in plain terms I
   could repeat back in an interview.
5. Prioritize that I can explain and defend the code over pure development speed.
   If a shortcut would make something harder to explain later, flag that trade-off
   before taking it.

## Running logs: learnings.md and problems.md

Two files at the repo root track the project as it goes, kept up to date throughout
— not written retroactively at the end:

- **`learnings.md`** — plain-terms lessons as they're understood, e.g. what a
  component is and why it works the way it does (what the input/output data actually
  represent, why a design choice was made, a concept once it's understood). This is
  interview-prep material: things I should be able to repeat back unprompted.
- **`problems.md`** — concrete problems hit and how they were resolved, e.g. a
  tile-boundary/ratio mismatch, a georeferencing bug, a Docker GPU build failure,
  a dependency conflict. Include what went wrong, the root cause, and the fix —
  not just "fixed it."

After any teaching explanation, module build, or debugging session that produces a
non-obvious lesson or a real problem+fix, append an entry to the relevant file
(append, don't rewrite prior entries) before moving on. Keep entries short and dated
relative to the module they belong to, not narrated blow-by-blow.

## Notes for later modules

- Preserve georeferencing by copying `meta`/`profile` from the input GeoTIFF via
  rasterio and applying it to the output raster.
- The provided sample input/output GeoTIFF pair is for validating correctness, not
  just for demo purposes — use it to check the pipeline before considering it done.
# ONNX Runtime backend - built after the original PyTorch-only image hit a hard
# size ceiling (torch's Linux wheel unconditionally preloads its entire CUDA
# dependency graph; see problems.md/learnings.md for the full saga and the
# accuracy/perf numbers that justified switching to this stack entirely).

# ---- Builder stage: exports the pretrained model to ONNX. Needs torch +
# segmentation-models-pytorch + huggingface_hub + onnx to construct and trace the
# model - none of this ships in the final image. CPU-only torch wheel here:
# tracing for export needs no CUDA at all, so even this discarded stage avoids
# the ~5GB CUDA dependency graph rather than needlessly downloading it.
FROM python:3.11-slim AS exporter

WORKDIR /export

# --index-url replaces the default PyPI index for the WHOLE command, not just the
# package named next to it - so this has to be a separate install from the
# PyPI-hosted packages below, or pip looks for all of them on download.pytorch.org.
# torchvision is pinned here too (matching torch 2.8.0's CPU wheel) because
# segmentation-models-pytorch below transitively depends on it - without pinning
# it here, its second pip install pulls torchvision from default PyPI, which
# drags the default CUDA torch build back in and undoes this CPU-only pin.
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        segmentation-models-pytorch==0.5.0 \
        huggingface_hub==1.8.0 \
        onnx==1.17.0

COPY src/ ./src/

RUN python -c "from src.waterseg.export_onnx import export; export('/export/model.onnx')"

# ---- Runtime stage: only what's needed to actually run inference.
# onnxruntime-gpu's own dependency metadata (unlike torch's - see problems.md)
# only pulls cuDNN/cuBLAS/cuFFT/cuRAND/nvrtc via opt-in [cuda,cudnn] extras, not
# an unconditional preload of the full CUDA toolkit's worth of libraries. Pinned
# to 1.22.0 deliberately: it's the version confirmed (by inspecting its actual
# pip metadata, not assumed) to target CUDA 12 - matching the driver requirement
# already proven to work, rather than newer releases that moved to CUDA 13.
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "onnxruntime-gpu[cuda,cudnn]==1.22.0" \
    rasterio==1.4.3 \
    numpy==2.0.2 \
    tqdm==4.70.0

# onnxruntime-gpu's CUDA execution provider dlopen()s its CUDA libraries at
# runtime, but - unlike torch's wheel, which patches its own RPATH at install
# time - pip just drops these nvidia-*-cu12 packages under site-packages/nvidia/
# without telling the dynamic linker to look there. Confirmed empirically (real
# GPU, real drivers, CUDAExecutionProvider still failed with "libcublasLt.so.12:
# cannot open shared object file" until this was added) rather than assumed.
ENV NV_LIB=/usr/local/lib/python3.11/site-packages/nvidia
ENV LD_LIBRARY_PATH="\
${NV_LIB}/cublas/lib:\
${NV_LIB}/cudnn/lib:\
${NV_LIB}/cufft/lib:\
${NV_LIB}/curand/lib:\
${NV_LIB}/cuda_nvrtc/lib:\
${NV_LIB}/cuda_runtime/lib:\
${NV_LIB}/nvjitlink/lib:\
${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

COPY --from=exporter /export/model.onnx ./model.onnx
COPY src/ ./src/

# Actually constructs an InferenceSession (not just imports the module) - this
# only proves the CPU-fallback path works, since no build stage ever gets GPU
# access (on any host, `docker build` never passes through `--gpus all`).
# It does NOT prove the CUDA path works - that segfaulted here once already
# (see problems.md) until model_onnx.py added a real driver-presence check
# instead of trusting onnxruntime's own provider-availability check. Real CUDA
# execution was separately verified with `docker run --gpus all` on actual GPU
# hardware - see learnings.md/README for those numbers.
RUN python -c "from src.waterseg.model_onnx import get_providers, load_session; load_session('model.onnx', get_providers())"

ENTRYPOINT ["python", "-m", "src.waterseg.cli"]

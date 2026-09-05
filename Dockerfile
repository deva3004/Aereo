FROM python:3.11-slim

WORKDIR /app  

COPY requirements.txt .

# Strip unused CUDA libraries in the SAME layer as the install: pip installing here
# and uninstalling in a later RUN would not shrink the image (Docker layers are
# additive diffs - a later layer's delete just adds a whiteout marker, the bytes
# still ship in the earlier layer).
#
# Tested one library at a time (see problems.md): torch's Linux wheel
# unconditionally preloads its *entire* declared nvidia-*-cu12 dependency list into
# torch._C at import time, regardless of whether this pipeline's forward pass ever
# dispatches an op through it. cusparselt, cufile, and nccl were each argued to be
# safely unreachable and each broke `import torch` outright - conclusive enough
# (3/3) to stop testing the rest of that list (cusolver/cusparse/cufft/curand/
# nvrtc/nvjitlink/cupti) individually; same failure mode is expected for all of
# them. Only `triton` is genuinely independent of that graph - a separate package,
# only touched by torch.compile(), which this pipeline never calls.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y triton \
    && find /usr/local/lib/python3.11/site-packages/torch -type d \( -name include -o -name test \) -exec rm -rf {} + \
    && find /usr/local/lib/python3.11/site-packages -type d -name __pycache__ -exec rm -rf {} +

# Container stdout/stderr aren't a real TTY, so Python block-buffers stdout and
# only flushes stderr on '\n' - tqdm redraws its bar with '\r', which never
# triggers that flush. Forces every write to flush immediately so progress is
# actually visible in `docker logs`/`docker compose up`.
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

COPY src/ ./src/

RUN python -c "from src.waterseg.pipeline import run; from src.waterseg.model import load_model, get_device; load_model(get_device())"

ENTRYPOINT ["python", "-m", "src.waterseg.cli"]

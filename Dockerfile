FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# rasterio's wheel bundles GDAL, which dynamically links against the system's
# libexpat (XML parsing) at runtime — not present on python:slim by default.
# Placed after the pip install layer so that expensive layer stays cached
# when this changes.
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

COPY src/ ./src/

# Bake the fixed, pre-trained checkpoint into the image at build time so
# `docker run` never needs network access — huggingface_hub caches to
# ~/.cache/huggingface/hub, which persists as an image layer. Importing
# pipeline (not just model) also exercises the rasterio import path, so a
# broken runtime dependency like the libexpat one above fails the build,
# not a later `docker run`.
RUN python -c "from src.waterseg.pipeline import run; from src.waterseg.model import load_model, get_device; load_model(get_device())"

ENTRYPOINT ["python", "-m", "src.waterseg.cli"]

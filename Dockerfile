FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Bake the fixed, pre-trained checkpoint into the image at build time so
# `docker run` never needs network access — huggingface_hub caches to
# ~/.cache/huggingface/hub, which persists as an image layer.
RUN python -c "from src.waterseg.model import load_model, get_device; load_model(get_device())"

ENTRYPOINT ["python", "-m", "src.waterseg.cli"]

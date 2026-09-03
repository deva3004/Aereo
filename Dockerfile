FROM python:3.11-slim

WORKDIR /app  

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

COPY src/ ./src/

RUN python -c "from src.waterseg.pipeline import run; from src.waterseg.model import load_model, get_device; load_model(get_device())"

ENTRYPOINT ["python", "-m", "src.waterseg.cli"]

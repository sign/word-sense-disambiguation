FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install Flash Attention
RUN pip install packaging ninja psutil && \
    MAX_JOBS=4 pip install flash_attn --no-build-isolation

# Rendering system deps (pango, cairo...)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Install package dependencies
RUN mkdir -p /app/welt/vision && \
    touch /app/README.md
COPY pyproject.toml /app/pyproject.toml
RUN pip install ".[train]"

# Copy requirements first for better Docker layer caching
COPY pyproject.toml .
RUN mkdir wsd && touch wsd/__init__.py

# Install Python dependencies & Accelerate spaCy with CUDA
RUN pip install --no-cache-dir ".[web]" spacy[cuda12x]

# Copy application code
COPY wsd/ ./wsd/

# Prime the dependency cache
RUN python -m wsd.prime

# Command to run the application
CMD python -m uvicorn wsd.server:app --host 0.0.0.0 --port $PORT

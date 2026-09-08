# Stage 1: install dependencies and download models.
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY pyproject.toml .
RUN mkdir wsd && touch wsd/__init__.py && touch /app/README.md

# Install Python dependencies. torch's pip wheel bundles its CUDA runtime, so no CUDA base image is
# needed; spaCy (en_core_web_lg) runs on the CPU. Bytecode caches and package tests are dropped.
RUN pip install --no-cache-dir ".[web]" && find /opt/venv -name "__pycache__" -type d -exec rm -rf {} + \
    && find /opt/venv -name "*.pyc" -delete && rm -rf /opt/venv/lib/python3.12/site-packages/*/tests

# Download the models before the code copy, so these heavy layers (and the
# venv layer, which the spaCy entity-linker KB is written into) stay identical
# across code-only changes and registries/Cloud Run can reuse them.
RUN python -c "import spacy; spacy.load('en_core_web_lg'); from spacy_entity_linker.DatabaseConnection import get_wikidata_instance; get_wikidata_instance()"
# The model name mirrors DEFAULT_MODEL in wsd/masked_language_model.py; it is
# repeated here so the download can run before the code copy (keep in sync).
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('sign/Ettin-150m-WSD')"

# Copy application code
COPY wsd/ ./wsd/

# Everything is already downloaded, so this is a fast end-to-end check that
# the model and pipeline actually run.
RUN python -m wsd.prime

# Stage 2: runtime image carrying only the venv, model caches, and app code.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    HF_HOME=/opt/hf-cache \
    PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home --uid 1000 app  # non-root; the home holds runtime caches

# Largest and most stable layers first, so code-only rebuilds reuse them.
COPY --from=builder --chown=app:app /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/hf-cache /opt/hf-cache

WORKDIR /app
COPY --chown=app:app wsd/ ./wsd/

USER app

# Command to run the application. exec makes uvicorn PID 1 so it receives
# SIGTERM and can shut down gracefully (Cloud Run sends SIGTERM, then SIGKILL).
# $PORT is provided by the runtime; Cloud Run sets it automatically.
CMD ["sh", "-c", "exec python -m uvicorn wsd.server:app --host 0.0.0.0 --port $PORT"]

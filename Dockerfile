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

# Install Python dependencies. torch arrives via accelerate/spacy-transformers
# with its CUDA runtime bundled as pip wheels, so no CUDA base image is needed.
# cupy-cuda13x matches torch's bundled CUDA 13 and accelerates spaCy on GPU.
RUN pip install --no-cache-dir ".[web]" cupy-cuda13x

# torch is unpinned, so a future release may bundle a different CUDA major and
# silently break the cupy pairing (spacy would fall back to CPU at runtime).
# Fail the build instead. Works without a GPU.
RUN python -c "import re, torch; from importlib.metadata import distributions; \
cupy = next(d.metadata['Name'] for d in distributions() if d.metadata['Name'].startswith('cupy-cuda')); \
tm = torch.version.cuda.split('.')[0]; cm = re.search('cuda([0-9]+)', cupy).group(1); \
assert tm == cm, f'CUDA major mismatch: torch {torch.version.cuda} vs {cupy}'"

# Download the models before the code copy, so these heavy layers (and the
# venv layer, which the spaCy entity-linker KB is written into) stay identical
# across code-only changes and registries/Cloud Run can reuse them.
RUN python -c "import spacy; nlp = spacy.load('en_core_web_trf'); nlp.add_pipe('entityLinker'); nlp('Apple is a technology company.')"
# The model name mirrors _DEFAULT_MODEL in wsd/masked_language_model.py; it is
# repeated here so the download can run before the code copy (keep in sync).
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('sign/ModernBERT-Large-Instruct-WSD')"

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

# Run as a non-root user; --create-home gives runtime caches (e.g. cupy's
# kernel cache in ~/.cupy) a writable location.
RUN useradd --create-home --uid 1000 app

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

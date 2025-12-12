FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY pyproject.toml .
RUN mkdir wsd && touch wsd/__init__.py && touch /app/README.md

# Install Python dependencies & Accelerate spaCy with CUDA
RUN pip install --no-cache-dir ".[web]" spacy[cuda12x]

# Copy application code
COPY wsd/ ./wsd/

# Prime the dependency cache
RUN python -m wsd.prime

# Command to run the application
CMD python -m uvicorn wsd.server:app --host 0.0.0.0 --port $PORT

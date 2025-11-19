FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for spaCy and transformers
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY pyproject.toml .

RUN mkdir wsd && touch wsd/__init__.py

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[web]"

# Install Flash Attention
## Install fused kernel packages
RUN pip install packaging ninja psutil
## Limit Jobs, due to memory issues
RUN MAX_JOBS=4 pip install flash_attn --no-build-isolation

# Copy application code
COPY wsd/ ./wsd/

# Prime the dependency cache
RUN python -m wsd.prime

# Command to run the application
CMD python -m uvicorn wsd.server:app --host 0.0.0.0 --port $PORT
# Benchmarking Guide

Install

```shell
pip install ".[benchmark]"
```

Start a wordnet server

```shell
docker run --platform=linux/amd64 -e PORT=8080 -p 8001:8080 ghcr.io/sign/wn:v0.1.0
```

Set `.env`:

```shell
WORDNET_URL=http://127.0.0.1:8001
```

Run the benchmark:

```shell
python wsd/benchmark.py
```

On an NVIDIA DGS Spark:

| Time  | Accuracy | Notes                         |
|-------|----------|-------------------------------|
| 19:06 | 53.8%    | ModernBERT model, initial run |

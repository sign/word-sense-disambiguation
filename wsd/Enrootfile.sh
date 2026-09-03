#!/usr/bin/env bash
# Cluster image for the batch pipeline: the published serve image (spaCy + cupy +
# the WSD model) plus the WordNet API and a `torchrun` on PATH so
# run_distributed.py can launch it. Run on a compute node:
#   enroot import -o /mnt/nfs-1/amit/wsd/wsd-serve.sqsh docker://ghcr.io#sign/word-sense-disambiguation:latest
#   srun -N1 --cpus-per-task 224 bash wsd/Enrootfile.sh /mnt/nfs-1/amit/wsd/wsd-serve.sqsh /mnt/nfs-1/amit/wsd/wsd-batch.sqsh
set -Eeuo pipefail
[[ $# -eq 2 && -f $1 ]] || { echo "usage: $0 BASE_IMAGE.sqsh OUTPUT_IMAGE.sqsh" >&2; exit 1; }
BASE=$1 OUT=$2 NAME="wsd-batch-build-$$"
cleanup() { enroot remove -f "$NAME" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
enroot create -n "$NAME" "$BASE"
enroot start --root --rw "$NAME" bash -c '
  set -Eeuo pipefail
  apt-get update -qq && apt-get install -y -qq gcc git && rm -rf /var/lib/apt/lists/*  # gcc: triton (torch.compile); git: clone the API
  # WordNet API, built like https://github.com/sign/wn/blob/main/Dockerfile (ghcr.io/sign/wn), so jobs can
  # start it locally when WORDNET_URL is unset (see wsd/env.py).
  git clone --depth 1 https://github.com/sign/wn /opt/wn
  /opt/venv/bin/pip install --no-cache-dir "/opt/wn[web]" uvicorn
  /opt/venv/bin/python -m wn download omw:1.4 cili
  /opt/venv/bin/python /opt/wn/extensions/wikidata-lexemes/merge_extension.py /opt/wn/extensions/wikidata-lexemes/output/*.xml
  /opt/venv/bin/python -c "from wn._db import connect; c = connect(); c.execute(\"ANALYZE\")"
  rm -rf ~/.wn_data/downloads
  # srun exports the host PATH over the image PATH, hiding /opt/venv/bin; give torchrun a fixed home.
  printf "#!/bin/sh\nexport PATH=/opt/venv/bin:\$PATH\nexec /opt/venv/bin/torchrun \"\$@\"\n" > /usr/local/bin/torchrun
  chmod +x /usr/local/bin/torchrun
  /opt/venv/bin/python -c "import wn.web, spacy, torch, transformers; print(\"ok\", transformers.__version__)"
'
rm -f "$OUT"
enroot export -o "$OUT" "$NAME"
echo "built $OUT"

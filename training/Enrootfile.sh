#!/usr/bin/env bash
# Build the training/benchmark image: NeMo base (torch, transformers, flash-attn)
# plus this repo's small pure-python deps and the WordNet API. Run on a compute node:
#   srun -N1 --cpus-per-task 224 bash training/Enrootfile.sh /mnt/nfs-1/nemo-26.06.sqsh /mnt/nfs-1/amit/wsd/wsd-train.sqsh
set -Eeuo pipefail
[[ $# -eq 2 && -f $1 ]] || { echo "usage: $0 BASE_IMAGE.sqsh OUTPUT_IMAGE.sqsh" >&2; exit 1; }
BASE=$1 OUT=$2 NAME="wsd-build-$$"
cleanup() { enroot remove -f "$NAME" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
enroot create -n "$NAME" "$BASE"
enroot start --root --rw "$NAME" bash -c '
  set -Eeuo pipefail
  pip install --no-cache-dir python-dotenv tqdm requests
  # WordNet API, built like https://github.com/sign/wn/blob/main/Dockerfile (ghcr.io/sign/wn), so jobs can
  # start it locally when WORDNET_URL is unset (see wsd/env.py).
  git clone --depth 1 https://github.com/sign/wn /opt/wn
  pip install --no-cache-dir "/opt/wn[web]" uvicorn
  python -m wn download omw:1.4 cili
  python /opt/wn/extensions/wikidata-lexemes/merge_extension.py /opt/wn/extensions/wikidata-lexemes/output/*.xml
  python -c "from wn._db import connect; c = connect(); c.execute(\"ANALYZE\")"
  rm -rf ~/.wn_data/downloads
  python -c "import wn.web, dotenv, tqdm, flash_attn, transformers; print(\"ok\", transformers.__version__)"
'
rm -f "$OUT"
enroot export -o "$OUT" "$NAME"
echo "built $OUT"

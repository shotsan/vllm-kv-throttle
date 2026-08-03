#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
if [[ -x .venv-proof/bin/python ]]; then
  PYTHON=.venv-proof/bin/python
fi

exec "$PYTHON" -c 'from vllm.entrypoints.cli.main import main; main()' serve \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype half \
  --max-model-len 2048 \
  --kv-cache-memory-bytes 46137344 \
  --max-num-seqs 16 \
  --port 8000

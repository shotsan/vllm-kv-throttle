#!/usr/bin/env bash
# Start the side-channel admission gateway (gateway.py) in front of a running vLLM.
# It observes the vLLM KV cache via /metrics and delays requests so the server is
# never over-subscribed. Start the vLLM server (run_server_preempt.sh) FIRST.
#
#   POLICY=kv-gate THRESHOLD=0.50 ./run_gateway.sh
#   POLICY=bypass ./run_gateway.sh            # pass-through (reproduce the livelock)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"

LISTEN_PORT="${LISTEN_PORT:-8001}"
UPSTREAM="${UPSTREAM:-http://127.0.0.1:8000}"
METRICS_URL="${METRICS_URL:-http://127.0.0.1:8000/metrics}"
POLICY="${POLICY:-kv-gate}"
THRESHOLD="${THRESHOLD:-0.50}"
POLL="${POLL:-0.1}"
SETTLE_TIMEOUT="${SETTLE_TIMEOUT:-8.0}"
LOG_DIR="${LOG_DIR:-$HERE/gateway_logs}"

STUDY_VENV="/home/ways_lab/Documents/LLM-Network-Study/.venv/bin/python"
PYTHON="${PYTHON:-}"
[ -z "$PYTHON" ] && { [ -x "$STUDY_VENV" ] && PYTHON="$STUDY_VENV" || PYTHON="python3"; }

echo "[gateway] python: $PYTHON  policy=$POLICY threshold=$THRESHOLD  listen :$LISTEN_PORT -> $UPSTREAM"
exec "$PYTHON" gateway.py \
  --listen-port "$LISTEN_PORT" \
  --upstream "$UPSTREAM" \
  --metrics-url "$METRICS_URL" \
  --policy "$POLICY" \
  --threshold "$THRESHOLD" \
  --poll "$POLL" \
  --settle-timeout "$SETTLE_TIMEOUT" \
  --log-dir "$LOG_DIR"

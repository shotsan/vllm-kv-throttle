#!/usr/bin/env bash
# Run the agentic-coding preemption load, but pointed at the GATEWAY instead of the
# raw vLLM server. The client's request URL is the gateway (:8001); the KV sampler
# still scrapes the REAL server's /metrics (:8000) for ground-truth KV/scheduler
# state. Start run_server_preempt.sh AND run_gateway.sh first (or use
# run_control_experiment.sh, which orchestrates all three).
#
#   WORKERS=8 DURATION=120 MAX_TOKENS=1500 SEED_TOKENS=6000 TIMEOUT=180 ./run_control_load.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"

MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
GATEWAY_HOSTPORT="${GATEWAY_HOSTPORT:-127.0.0.1:8001}"   # client -> gateway
SERVER_HOSTPORT="${SERVER_HOSTPORT:-127.0.0.1:8000}"     # sampler -> real vLLM
URL="http://$GATEWAY_HOSTPORT/v1/chat/completions"
METRICS_URL="http://$SERVER_HOSTPORT/metrics"
WORKERS="${WORKERS:-8}"
DURATION="${DURATION:-120}"
MAX_TOKENS="${MAX_TOKENS:-1500}"
SEED_TOKENS="${SEED_TOKENS:-6000}"   # ~56% of usable KV -> one in-flight pins KV > 50%
TIMEOUT="${TIMEOUT:-180}"

STUDY_VENV="/home/ways_lab/Documents/LLM-Network-Study/.venv/bin/python"
PYTHON="${PYTHON:-}"
[ -z "$PYTHON" ] && { [ -x "$STUDY_VENV" ] && PYTHON="$STUDY_VENV" || PYTHON="python3"; }
echo "[run] python: $PYTHON"

if [ -z "${VLLM_API_KEY:-}" ]; then
  ENV_FILE="/home/ways_lab/Documents/LLM-Network-Study/local_llm/.env"
  [ -f "$ENV_FILE" ] && { set -a; source "$ENV_FILE"; set +a; echo "[run] loaded VLLM_API_KEY"; }
fi
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

echo "[run] checking gateway at $URL ..."
auth=(); [ "$VLLM_API_KEY" != "EMPTY" ] && auth=(-H "Authorization: Bearer $VLLM_API_KEY")
curl -s -m 5 "${auth[@]}" "http://$GATEWAY_HOSTPORT/v1/models" | grep -q '"id"' || {
  echo "[run] ERROR: no gateway at http://$GATEWAY_HOSTPORT. Start ./run_gateway.sh first." >&2
  exit 1; }
echo "[run] gateway is up (proxying to real server)."

echo "[run] preparing realistic coding task (seed ~${SEED_TOKENS} tok) -> ./generated"
"$PYTHON" prepare_task.py --model "$MODEL" \
  --seed-tokens "$SEED_TOKENS" --output-dir "$HERE/generated"

echo "[run] launching $WORKERS workers x ${DURATION}s (max_tokens=$MAX_TOKENS) via gateway"
mkdir -p results
"$PYTHON" deadlock_run.py --model "$MODEL" --url "$URL" --metrics-url "$METRICS_URL" \
  --workers "$WORKERS" --duration "$DURATION" --max-tokens "$MAX_TOKENS" --timeout "$TIMEOUT" \
  | tee "results/control_$(date +%Y%m%d_%H%M%S).console.log"

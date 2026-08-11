#!/usr/bin/env bash
# One-shot driver for the KV-oversubscription / preemption-livelock experiment.
# Prepares the (realistic agentic-coding) task, checks the server, then runs the
# continuous load + KV sampler + watchdog. It does NOT start the model server --
# launch ./run_server_preempt.sh separately first.
#
#   WORKERS=8 DURATION=180 MAX_TOKENS=4096 ./run_preempt.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
HOSTPORT="${HOSTPORT:-127.0.0.1:8000}"
URL="http://$HOSTPORT/v1/chat/completions"
WORKERS="${WORKERS:-8}"
DURATION="${DURATION:-300}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
SEED_TOKENS="${SEED_TOKENS:-1800}"   # keep seed + max_tokens < server max-model-len
TIMEOUT="${TIMEOUT:-180}"            # per-request client patience (s); raise to let the oldest drain

STUDY_VENV="/home/ways_lab/Documents/LLM-Network-Study/.venv/bin/python"
PYTHON="${PYTHON:-}"
[ -z "$PYTHON" ] && { [ -x "$STUDY_VENV" ] && PYTHON="$STUDY_VENV" || PYTHON="python3"; }
echo "[run] python: $PYTHON"

if [ -z "${VLLM_API_KEY:-}" ]; then
  ENV_FILE="/home/ways_lab/Documents/LLM-Network-Study/local_llm/.env"
  [ -f "$ENV_FILE" ] && { set -a; source "$ENV_FILE"; set +a; echo "[run] loaded VLLM_API_KEY"; }
fi
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

echo "[run] checking server at $URL ..."
auth=(); [ "$VLLM_API_KEY" != "EMPTY" ] && auth=(-H "Authorization: Bearer $VLLM_API_KEY")
curl -s -m 5 "${auth[@]}" "http://$HOSTPORT/v1/models" | grep -q '"id"' || {
  echo "[run] ERROR: no server at http://$HOSTPORT. Start ./run_server_preempt.sh first." >&2
  exit 1; }
echo "[run] server is up."

echo "[run] preparing realistic coding task (seed ~${SEED_TOKENS} tok) -> ./generated"
"$PYTHON" ../agentic_coding/prepare_agent_task.py --model "$MODEL" \
  --seed-tokens "$SEED_TOKENS" --output-dir "$HERE/generated"

echo "[run] launching $WORKERS workers x ${DURATION}s (max_tokens=$MAX_TOKENS)"
"$PYTHON" deadlock_run.py --model "$MODEL" --url "$URL" \
  --workers "$WORKERS" --duration "$DURATION" --max-tokens "$MAX_TOKENS" --timeout "$TIMEOUT" \
  | tee "results/preempt_$(date +%Y%m%d_%H%M%S).console.log"

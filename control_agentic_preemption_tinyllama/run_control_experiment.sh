#!/usr/bin/env bash
# Run ONE controlled TinyLlama experiment end-to-end (the TinyLlama analogue of
# ../control_agentic_preemption/run_control_experiment.sh, but on the PARENT
# repo's framework: TinyLlama + LooGLE prompt + one barrier-synced wave):
#   1. instrument the venv vLLM (probe_patch, idempotent) for ground-truth preemptions
#   2. prepare the LooGLE prompt (parent's prepare_prompt.py) if missing
#   3. (re)start a fresh OVER-ADMIT TinyLlama server (RESERVE_ISL=0 by default)
#   4. start the side-channel admission GATEWAY (gateway.py) in front of it
#   5. fire one synchronized wave of requests at the GATEWAY + sample metrics
#   6. parse ground-truth probes + collect gateway logs; stop gateway + server
#
# Server knobs (env): RESERVE_ISL(0) NUM_GPU_BLOCKS(128) MAX_NUM_SEQS(16) GPU_UTIL(0.3)
# Gateway knobs(env): POLICY(kv-gate) THRESHOLD(0.50) SETTLE_TIMEOUT(8.0)
# Load knobs   (env): CONCURRENCY(8) MAX_TOKENS(1200) TIMEOUT(1200) PROMPT_TOKENS(500)
#
#   # negative control: over-admit server, gateway pass-through -> preemption thrash
#   POLICY=bypass ./run_control_experiment.sh
#   # control: kv-gate holds requests at the gateway -> clean network queueing
#   POLICY=kv-gate ./run_control_experiment.sh
#   # when finished with ALL experiments:
#   python3 probe_patch.py revert
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
PARENT="$(dirname "$HERE")"
PY="/home/ways_lab/Documents/LLM-Network-Study/.venv/bin/python"

RESERVE_ISL="${RESERVE_ISL:-0}"; NUM_GPU_BLOCKS="${NUM_GPU_BLOCKS:-128}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"; GPU_UTIL="${GPU_UTIL:-0.3}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
POLICY="${POLICY:-kv-gate}"; THRESHOLD="${THRESHOLD:-0.50}"; SETTLE_TIMEOUT="${SETTLE_TIMEOUT:-8.0}"
CONCURRENCY="${CONCURRENCY:-8}"; MAX_TOKENS="${MAX_TOKENS:-1200}"
TIMEOUT="${TIMEOUT:-1200}"; PROMPT_TOKENS="${PROMPT_TOKENS:-500}"
LISTEN_PORT="${LISTEN_PORT:-8001}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SERVER_LOG="$HERE/server_probe_$STAMP.log"
GATEWAY_LOG_DIR="$HERE/gateway_logs/gw_$STAMP"
OUT_DIR="$HERE/results/${POLICY}_$STAMP"
GATEWAY_PID=""

kill_port() {
  local port="$1" pid
  pid=$(ss -ltnp "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "$pid" ] && { kill "$pid" 2>/dev/null; echo "[ctl-tiny]   killed :$port pid $pid"; sleep 2; }
}
cleanup() {
  [ -n "$GATEWAY_PID" ] && kill "$GATEWAY_PID" 2>/dev/null
  kill_port "$LISTEN_PORT"
  kill_port 8000
}
trap cleanup EXIT

echo "[ctl-tiny] 1/6  ensure venv is instrumented (idempotent)"
python3 probe_patch.py apply

echo "[ctl-tiny] 2/6  prepare LooGLE prompt (~$PROMPT_TOKENS tok, parent framework)"
if [ ! -f generated/prompt.txt ] || [ "$(cat generated/prompt.tokens 2>/dev/null)" != "$PROMPT_TOKENS" ]; then
  (cd "$PARENT" && "$PY" prepare_prompt.py --target-tokens "$PROMPT_TOKENS" \
      --output "$HERE/generated/prompt.txt")
  echo "$PROMPT_TOKENS" > generated/prompt.tokens
fi

echo "[ctl-tiny] 3/6  (re)start a fresh OVER-ADMIT TinyLlama server  RESERVE_ISL=$RESERVE_ISL -> $SERVER_LOG"
kill_port 8000
: > "$SERVER_LOG"
RESERVE_ISL="$RESERVE_ISL" NUM_GPU_BLOCKS="$NUM_GPU_BLOCKS" MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  GPU_UTIL="$GPU_UTIL" MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
  nohup ./run_server_tinyllama.sh >>"$SERVER_LOG" 2>&1 &
echo "[ctl-tiny]   server pid $! — waiting for ready..."
ready=0
for i in $(seq 1 60); do
  if curl -s -m 3 http://127.0.0.1:8000/metrics 2>/dev/null | grep -q kv_cache_usage_perc; then ready=1; break; fi
  if grep -qiE "EngineCore failed|initialization failed|ValueError:" "$SERVER_LOG" 2>/dev/null; then
    echo "[ctl-tiny]   SERVER FAILED:"; grep -iE "ValueError:|RuntimeError:" "$SERVER_LOG" | tail -3; exit 1; fi
  sleep 5
done
[ "$ready" = 1 ] || { echo "[ctl-tiny]   server not ready after ~5 min; aborting"; exit 1; }
echo "[ctl-tiny]   server ready."

echo "[ctl-tiny] 4/6  start gateway  POLICY=$POLICY THRESHOLD=$THRESHOLD -> $GATEWAY_LOG_DIR"
kill_port "$LISTEN_PORT"
mkdir -p "$GATEWAY_LOG_DIR"
LISTEN_PORT="$LISTEN_PORT" POLICY="$POLICY" THRESHOLD="$THRESHOLD" SETTLE_TIMEOUT="$SETTLE_TIMEOUT" \
  LOG_DIR="$GATEWAY_LOG_DIR" nohup ./run_gateway.sh >>"$GATEWAY_LOG_DIR/gateway.stdout.log" 2>&1 &
GATEWAY_PID=$!
gw_ready=0
for i in $(seq 1 20); do
  if curl -s -m 3 "http://127.0.0.1:$LISTEN_PORT/v1/models" 2>/dev/null | grep -q '"id"'; then gw_ready=1; break; fi
  sleep 1
done
[ "$gw_ready" = 1 ] || { echo "[ctl-tiny]   gateway not ready; aborting"; cat "$GATEWAY_LOG_DIR/gateway.stdout.log"; exit 1; }
echo "[ctl-tiny]   gateway ready."

echo "[ctl-tiny] 5/6  fire wave  CONCURRENCY=$CONCURRENCY MAX_TOKENS=$MAX_TOKENS TIMEOUT=$TIMEOUT -> $OUT_DIR"
"$PY" tinyllama_load.py \
  --url "http://127.0.0.1:$LISTEN_PORT/v1/completions" \
  --metrics-url "http://127.0.0.1:8000/metrics" \
  --prompt generated/prompt.txt \
  --concurrency "$CONCURRENCY" --max-tokens "$MAX_TOKENS" --timeout "$TIMEOUT" \
  --out-dir "$OUT_DIR"

echo "[ctl-tiny] 6/6  parse ground-truth probes + collect gateway logs; stop everything"
cp "$SERVER_LOG" "$OUT_DIR/server_probe.log"
cp -r "$GATEWAY_LOG_DIR" "$OUT_DIR/gateway_logs"
python3 parse_probe.py "$SERVER_LOG" --out "$OUT_DIR"
cleanup; trap - EXIT

echo ""
echo "[ctl-tiny] DONE  policy=$POLICY -> $OUT_DIR"
echo "  client:  events.log, kv_metrics.jsonl, summary.json, requests.jsonl"
echo "  gateway: gateway_logs/gateway_events.log, gateway_requests.jsonl"
echo "  server:  server_probe.log, probe_events.jsonl, probe_summary.json (true preemptions)"
echo "  venv is STILL instrumented — run 'python3 probe_patch.py revert' after your LAST experiment."

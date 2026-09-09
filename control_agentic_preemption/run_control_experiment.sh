#!/usr/bin/env bash
# Run ONE controlled experiment end-to-end:
#   1. instrument the venv vLLM (probe_patch, idempotent) for ground-truth preemptions
#   2. (re)start a fresh OVER-ADMIT vLLM server  (RESERVE_ISL=0 by default -> the
#      SAME thrashing config that livelocks in ../agentic_preemption Recipe A)
#   3. start the side-channel admission GATEWAY (gateway.py) in front of it
#   4. run the agentic load pointed at the GATEWAY
#   5. parse ground-truth probes + collect gateway logs into the run's results dir
#   6. stop the gateway and the server
#
# The point: the SERVER admission mode is left in its livelock-prone state; the
# GATEWAY's KV observation is the only thing preventing over-subscription.
#
# Server knobs (env): RESERVE_ISL(0) MAX_MODEL_LEN(8192) NUM_GPU_BLOCKS(10)
#                     MAX_NUM_SEQS(10) GPU_UTIL(0.5)
# Gateway knobs(env): POLICY(kv-gate) THRESHOLD(0.50) SETTLE_TIMEOUT(8.0)
# Load knobs   (env): WORKERS(8) DURATION(120) MAX_TOKENS(1500) TIMEOUT(180) SEED_TOKENS(6000)
#
# Examples:
#   # controlled run: over-admit server + kv-gate -> expect clean queueing, ~0 preemptions
#   ./run_control_experiment.sh
#   # negative control: over-admit server + gateway in BYPASS -> expect the livelock back
#   POLICY=bypass ./run_control_experiment.sh
#   # when finished with ALL experiments:
#   python3 probe_patch.py revert
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"

RESERVE_ISL="${RESERVE_ISL:-0}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
NUM_GPU_BLOCKS="${NUM_GPU_BLOCKS:-10}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-10}"; GPU_UTIL="${GPU_UTIL:-0.5}"
POLICY="${POLICY:-kv-gate}"; THRESHOLD="${THRESHOLD:-0.50}"; SETTLE_TIMEOUT="${SETTLE_TIMEOUT:-8.0}"
WORKERS="${WORKERS:-8}"; DURATION="${DURATION:-120}"; MAX_TOKENS="${MAX_TOKENS:-1500}"
TIMEOUT="${TIMEOUT:-180}"; SEED_TOKENS="${SEED_TOKENS:-6000}"
LISTEN_PORT="${LISTEN_PORT:-8001}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SERVER_LOG="$HERE/server_probe_$STAMP.log"
GATEWAY_LOG_DIR="$HERE/gateway_logs/gw_$STAMP"
ENV_FILE="/home/ways_lab/Documents/LLM-Network-Study/local_llm/.env"
GATEWAY_PID=""

kill_port() {
  local port="$1" pid
  pid=$(ss -ltnp "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "$pid" ] && { kill "$pid" 2>/dev/null; echo "[ctl-exp]   killed :$port pid $pid"; sleep 2; }
}
cleanup() {
  [ -n "$GATEWAY_PID" ] && kill "$GATEWAY_PID" 2>/dev/null
  kill_port "$LISTEN_PORT"
  kill_port 8000
}
trap cleanup EXIT

echo "[ctl-exp] 1/6  ensure venv is instrumented (idempotent)"
python3 probe_patch.py apply

echo "[ctl-exp] 2/6  (re)start a fresh OVER-ADMIT server  RESERVE_ISL=$RESERVE_ISL -> $SERVER_LOG"
kill_port 8000
[ -f "$ENV_FILE" ] && { set -a; source "$ENV_FILE"; set +a; }
: > "$SERVER_LOG"
RESERVE_ISL="$RESERVE_ISL" MAX_MODEL_LEN="$MAX_MODEL_LEN" NUM_GPU_BLOCKS="$NUM_GPU_BLOCKS" \
  MAX_NUM_SEQS="$MAX_NUM_SEQS" GPU_UTIL="$GPU_UTIL" \
  nohup ./run_server_preempt.sh >>"$SERVER_LOG" 2>&1 &
echo "[ctl-exp]   server pid $! — waiting for ready (~5 min)..."
ready=0
for i in $(seq 1 40); do
  if curl -s -m 3 http://127.0.0.1:8000/metrics 2>/dev/null | grep -q kv_cache_size_tokens; then ready=1; break; fi
  if grep -qiE "EngineCore failed|initialization failed|ValueError:" "$SERVER_LOG" 2>/dev/null; then
    echo "[ctl-exp]   SERVER FAILED:"; grep -iE "ValueError:|RuntimeError:" "$SERVER_LOG" | tail -3; exit 1; fi
  sleep 15
done
[ "$ready" = 1 ] || { echo "[ctl-exp]   server not ready after ~10 min; aborting"; exit 1; }
echo "[ctl-exp]   server ready."

echo "[ctl-exp] 3/6  start gateway  POLICY=$POLICY THRESHOLD=$THRESHOLD -> $GATEWAY_LOG_DIR"
kill_port "$LISTEN_PORT"
mkdir -p "$GATEWAY_LOG_DIR"
LISTEN_PORT="$LISTEN_PORT" POLICY="$POLICY" THRESHOLD="$THRESHOLD" SETTLE_TIMEOUT="$SETTLE_TIMEOUT" \
  LOG_DIR="$GATEWAY_LOG_DIR" nohup ./run_gateway.sh >>"$GATEWAY_LOG_DIR/gateway.stdout.log" 2>&1 &
GATEWAY_PID=$!
echo "[ctl-exp]   gateway pid $GATEWAY_PID — waiting for ready..."
gw_ready=0
for i in $(seq 1 20); do
  gw_auth=(); [ -n "${VLLM_API_KEY:-}" ] && [ "${VLLM_API_KEY}" != "EMPTY" ] && gw_auth=(-H "Authorization: Bearer $VLLM_API_KEY")
  if curl -s -m 3 "${gw_auth[@]}" "http://127.0.0.1:$LISTEN_PORT/v1/models" 2>/dev/null | grep -q '"id"'; then gw_ready=1; break; fi
  sleep 1
done
[ "$gw_ready" = 1 ] || { echo "[ctl-exp]   gateway not ready; aborting"; cat "$GATEWAY_LOG_DIR/gateway.stdout.log"; exit 1; }
echo "[ctl-exp]   gateway ready."

echo "[ctl-exp] 4/6  run load through gateway  WORKERS=$WORKERS DURATION=$DURATION MAX_TOKENS=$MAX_TOKENS SEED_TOKENS=$SEED_TOKENS"
WORKERS="$WORKERS" DURATION="$DURATION" MAX_TOKENS="$MAX_TOKENS" TIMEOUT="$TIMEOUT" SEED_TOKENS="$SEED_TOKENS" \
  ./run_control_load.sh

echo "[ctl-exp] 5/6  parse ground-truth probes + collect gateway logs into results"
d=$(ls -td results/control_*/ 2>/dev/null | head -1)
if [ -z "$d" ]; then
  # deadlock_run wrote to results/preempt_* (its default out-dir); grab the newest run dir
  d=$(ls -td results/*/ | head -1)
fi
cp "$SERVER_LOG" "$d/server_probe.log"
cp -r "$GATEWAY_LOG_DIR" "$d/gateway_logs"
python3 parse_probe.py "$SERVER_LOG" --out "$d"

echo "[ctl-exp] 6/6  stop gateway + server (next call starts fresh)"
cleanup; trap - EXIT

echo ""
echo "[ctl-exp] DONE  policy=$POLICY -> $d"
echo "  client:  events.log, kv_metrics.jsonl, summary.json, requests.jsonl"
echo "  gateway: gateway_logs/gateway_events.log, gateway_requests.jsonl (admission ground truth)"
echo "  server:  server_probe.log, probe_events.jsonl, probe_summary.json (true preemptions)"
echo "  venv is STILL instrumented — run 'python3 probe_patch.py revert' after your LAST experiment."

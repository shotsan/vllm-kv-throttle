#!/usr/bin/env bash
# Run ONE experiment against its OWN freshly-started, probe-instrumented server,
# then parse the ground-truth probe events into that run's results folder.
#
# Call it once per experiment. EACH call restarts the server (a NEW probe server
# with a NEW log), so experiments never share/mix probe data. The venv patch is
# applied once (idempotent) and LEFT applied across calls -- run
#   python3 probe_patch.py revert
# after your LAST experiment to restore the venv.
#
# Server knobs (env): RESERVE_ISL(0) MAX_MODEL_LEN(8192) NUM_GPU_BLOCKS(10)
#                     MAX_NUM_SEQS(10) GPU_UTIL(0.5)
# Load knobs   (env): WORKERS(8) DURATION(120) MAX_TOKENS(4096) TIMEOUT(180) SEED_TOKENS(1800)
#
# Examples:
#   # experiment 1 — Recipe A (over-admit)
#   WORKERS=8 MAX_TOKENS=4096 TIMEOUT=180 ./run_probe_experiment.sh
#   # experiment 2 — Recipe D (clean-admission), gets a brand-new probe server
#   RESERVE_ISL=1 SEED_TOKENS=6000 MAX_TOKENS=1500 TIMEOUT=180 ./run_probe_experiment.sh
#   # when finished with ALL experiments:
#   python3 probe_patch.py revert
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"

RESERVE_ISL="${RESERVE_ISL:-0}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
NUM_GPU_BLOCKS="${NUM_GPU_BLOCKS:-10}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-10}"; GPU_UTIL="${GPU_UTIL:-0.5}"
WORKERS="${WORKERS:-8}"; DURATION="${DURATION:-120}"; MAX_TOKENS="${MAX_TOKENS:-4096}"
TIMEOUT="${TIMEOUT:-180}"; SEED_TOKENS="${SEED_TOKENS:-1800}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SERVER_LOG="$HERE/server_probe_$STAMP.log"
ENV_FILE="/home/ways_lab/Documents/LLM-Network-Study/local_llm/.env"

kill_server() {
  local pid; pid=$(ss -ltnp 'sport = :8000' 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "$pid" ] && { kill "$pid" 2>/dev/null; echo "[probe-exp]   killed server $pid"; sleep 5; }
}

echo "[probe-exp] 1/5  ensure venv is instrumented (idempotent)"
python3 probe_patch.py apply

echo "[probe-exp] 2/5  (re)start a fresh probe server  RESERVE_ISL=$RESERVE_ISL  -> $SERVER_LOG"
kill_server
[ -f "$ENV_FILE" ] && { set -a; source "$ENV_FILE"; set +a; }
: > "$SERVER_LOG"
RESERVE_ISL="$RESERVE_ISL" MAX_MODEL_LEN="$MAX_MODEL_LEN" NUM_GPU_BLOCKS="$NUM_GPU_BLOCKS" \
  MAX_NUM_SEQS="$MAX_NUM_SEQS" GPU_UTIL="$GPU_UTIL" \
  nohup ./run_server_preempt.sh >>"$SERVER_LOG" 2>&1 &
echo "[probe-exp]   server pid $! — waiting for ready (~5 min)..."
ready=0
for i in $(seq 1 40); do
  if curl -s -m 3 http://127.0.0.1:8000/metrics 2>/dev/null | grep -q kv_cache_size_tokens; then ready=1; break; fi
  if grep -qiE "EngineCore failed|initialization failed|ValueError:" "$SERVER_LOG" 2>/dev/null; then
    echo "[probe-exp]   SERVER FAILED:"; grep -iE "ValueError:|RuntimeError:" "$SERVER_LOG" | tail -3; exit 1; fi
  sleep 15
done
[ "$ready" = 1 ] || { echo "[probe-exp]   server not ready after ~10 min; aborting"; exit 1; }
echo "[probe-exp]   server ready."

echo "[probe-exp] 3/5  run load  WORKERS=$WORKERS DURATION=$DURATION MAX_TOKENS=$MAX_TOKENS TIMEOUT=$TIMEOUT SEED_TOKENS=$SEED_TOKENS"
WORKERS="$WORKERS" DURATION="$DURATION" MAX_TOKENS="$MAX_TOKENS" TIMEOUT="$TIMEOUT" SEED_TOKENS="$SEED_TOKENS" \
  ./run_preempt.sh

echo "[probe-exp] 4/5  parse ground-truth probes into the run's results folder"
d=$(ls -td results/preempt_*/ | head -1)
cp "$SERVER_LOG" "$d/server_probe.log"
python3 parse_probe.py "$SERVER_LOG" --out "$d"

echo "[probe-exp] 5/5  stop this server (the next call starts a brand-new one)"
kill_server

echo ""
echo "[probe-exp] DONE  -> $d"
echo "  client:  events.log (ADMIT/PREEMPT causes), kv_metrics.jsonl, summary.json"
echo "  server:  server_probe.log (raw PROBE), probe_events.jsonl, probe_summary.json (ground truth)"
echo "  venv is STILL instrumented — run 'python3 probe_patch.py revert' after your LAST experiment."

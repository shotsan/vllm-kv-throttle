#!/usr/bin/env bash
# Serve TinyLlama with the PARENT repo's KV geometry (../run_server.sh: one
# 2048-token sequence of KV) but with the scheduler in OVER-ADMIT mode, so that
# several concurrent growing sequences force real preemptions -- the TinyLlama
# analogue of ../control_agentic_preemption/run_server_preempt.sh.
#
# Parent framework kept: TinyLlama/TinyLlama-1.1B-Chat-v1.0, --dtype half,
# --max-model-len 2048, --max-num-seqs 16, KV sized to exactly 2048 tokens.
# (The parent pins KV via --kv-cache-memory-bytes; on this venv's vLLM 0.25.1 the
# equivalent block-precise knob is --num-gpu-blocks-override: TinyLlama block_size
# is 16 tokens -> 128 blocks = 2048 tokens.)
#
# Admission mode:
#   RESERVE_ISL=0 (default) -> --no-scheduler-reserve-full-isl: admit on current
#     footprint (prompt fits now), let growth blow past the tiny KV -> PREEMPT.
#   RESERVE_ISL=1 -> omit the flag: full-ISL guard ON, clean serial queueing
#     (this is what the parent's proof demonstrates).
#
#   NUM_GPU_BLOCKS=128 MAX_NUM_SEQS=16 ./run_server_tinyllama.sh
set -euo pipefail

MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
NUM_GPU_BLOCKS="${NUM_GPU_BLOCKS:-128}"   # 128 blocks x 16 tok = 2048 KV tokens
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_UTIL="${GPU_UTIL:-0.3}"
RESERVE_ISL="${RESERVE_ISL:-0}"
# Prefill chunk size (scheduler token budget per step). SMALL values let a huge
# prompt be admitted with only one small chunk's worth of free KV (over-admit),
# then keep grabbing blocks chunk-by-chunk -> mutual eviction during re-prefill.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"

VLLM_BIN="${VLLM_BIN:-/home/ways_lab/Documents/LLM-Network-Study/.venv/bin/vllm}"
if [ ! -x "$VLLM_BIN" ]; then VLLM_BIN="$(command -v vllm)"; fi

if [ "$RESERVE_ISL" = "1" ]; then
  admission_flags=()
  admission_desc="reserve-full-isl ON (clean admission / queue)"
else
  admission_flags=(--no-scheduler-reserve-full-isl)
  admission_desc="--no-scheduler-reserve-full-isl (over-admit / preempt)"
fi

echo "[serve] $([ "$RESERVE_ISL" = 1 ] && echo CLEAN-ADMISSION || echo OVERSUBSCRIPTION) mode" >&2
echo "[serve]   model=$MODEL  max_model_len=$MAX_MODEL_LEN" >&2
echo "[serve]   num_gpu_blocks_override=$NUM_GPU_BLOCKS (=$((NUM_GPU_BLOCKS*16)) KV tokens)  max_num_seqs=$MAX_NUM_SEQS" >&2
echo "[serve]   flags: $admission_desc --watermark 0.0 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS" >&2

exec env HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" "$VLLM_BIN" serve \
  "$MODEL" \
  --dtype half \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --num-gpu-blocks-override "$NUM_GPU_BLOCKS" \
  "${admission_flags[@]}" \
  --watermark 0.0 \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --host 0.0.0.0 --port "$PORT"

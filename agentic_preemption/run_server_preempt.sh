#!/usr/bin/env bash
# Serve Qwen with an ADVERSARIALLY TINY KV cache to force scheduler preemption.
#
# This is the opposite of ../agentic_coding/run_server_qwen.sh (which pins KV so
# exactly one sequence runs). Here we deliberately OVERSUBSCRIBE: admit several
# sequences whose prompts fit, then let each generate long (ignore_eos) so their
# combined KV blows past capacity and vLLM must PREEMPT running sequences
# (recompute-only in V1: it frees ALL of a victim's KV blocks and re-prefills it
# from scratch -- i.e. "throws away a run's context"). Under continuous load this
# produces a preemption livelock / goodput collapse.
#
# Key flags (all verified in vLLM 0.25.1):
#   --num-gpu-blocks-override N     tiny KV; help says literally "Used for testing
#                                   preemption." (Sidesteps this hybrid Qwen-MoE's
#                                   1072-token block geometry -- blocks, not bytes.)
#   --no-scheduler-reserve-full-isl allow OPTIMISTIC over-admission: admit on the
#                                   current footprint, not the full output length.
#                                   This is what lets agent 3 "begin running" before
#                                   the cache discovers it cannot sustain 1+2+3.
#   --watermark 0.0                 keep no free-block reserve -> maximize churn.
#   --max-num-seqs high             let many sequences be admitted at once.
# NB: NO --speculative-config here (the agentic_coding server used MTP; a clean
# scheduler is easier to reason about). max-model-len is kept small so vLLM's
# "one max-len sequence must fit" startup guarantee holds with very few blocks.
#
#   MAX_MODEL_LEN=8192 NUM_GPU_BLOCKS=10 MAX_NUM_SEQS=32 ./run_server_preempt.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"      # per-request context; one seq must fit
NUM_GPU_BLOCKS="${NUM_GPU_BLOCKS:-10}"      # TINY KV (block_size ~1072 tok on this model)
# HYBRID-MAMBA CONSTRAINT: this Qwen-MoE has Mamba layers; each concurrent decode
# sequence needs one Mamba cache block, and num-gpu-blocks-override caps those. So
# max-num-seqs must be <= NUM_GPU_BLOCKS or the engine refuses to start.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-10}"
# KV size is fixed by --num-gpu-blocks-override, so util only gates the startup
# memory check; keep it modest so it fits alongside other GPU users.
GPU_UTIL="${GPU_UTIL:-0.5}"
KV_DTYPE="${KV_DTYPE:-fp8}"
# Admission mode:
#   RESERVE_ISL=0 (default) -> pass --no-scheduler-reserve-full-isl: OVER-ADMIT and
#     thrash (Recipes A/B/C). A large prompt can be chunk-prefilled and admitted
#     before it's known to fit, then preempted.
#   RESERVE_ISL=1 -> omit the flag (vLLM default guard ON): the FULL prompt must fit
#     in KV before admission, so with large prompts only one fits and the rest queue
#     cleanly with NO preemption -- the parent's admission control (Recipe D).
RESERVE_ISL="${RESERVE_ISL:-0}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
# Load the API key from the study .env if not already exported, so the server key
# MATCHES what run_preempt.sh sends (that script sources the same .env). Otherwise
# the server ends up keyed EMPTY while the client sends the real key -> 401.
if [ -z "${VLLM_API_KEY:-}" ]; then
  ENV_FILE="/home/ways_lab/Documents/LLM-Network-Study/local_llm/.env"
  [ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }
fi
API_KEY="${VLLM_API_KEY:-EMPTY}"

# Prefer the study venv's vllm (it has the model + deps).
VLLM_BIN="${VLLM_BIN:-/home/ways_lab/Documents/LLM-Network-Study/.venv/bin/vllm}"
if [ ! -x "$VLLM_BIN" ]; then VLLM_BIN="$(command -v vllm)"; fi

if [ "$RESERVE_ISL" = "1" ]; then
  admission_flags=()                                 # guard ON (vLLM default): clean admission
  admission_desc="reserve-full-isl ON (clean admission / queue)"
else
  admission_flags=(--no-scheduler-reserve-full-isl)  # guard OFF: over-admit / thrash
  admission_desc="--no-scheduler-reserve-full-isl (over-admit / preempt)"
fi

echo "[serve] $([ "$RESERVE_ISL" = 1 ] && echo CLEAN-ADMISSION || echo OVERSUBSCRIPTION) mode" >&2
echo "[serve]   model=$MODEL  max_model_len=$MAX_MODEL_LEN" >&2
echo "[serve]   num_gpu_blocks_override=$NUM_GPU_BLOCKS  max_num_seqs=$MAX_NUM_SEQS" >&2
echo "[serve]   flags: $admission_desc --watermark 0.0 (no spec-decode)" >&2

exec env HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" "$VLLM_BIN" serve \
  "$MODEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --num-gpu-blocks-override "$NUM_GPU_BLOCKS" \
  "${admission_flags[@]}" \
  --watermark 0.0 \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --kv-cache-dtype "$KV_DTYPE" \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_PARSER" \
  --reasoning-parser "$REASONING_PARSER" \
  --trust-remote-code \
  --host 0.0.0.0 --port "$PORT" \
  --api-key "$API_KEY"

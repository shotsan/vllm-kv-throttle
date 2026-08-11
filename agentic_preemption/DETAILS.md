# Agentic preemption: KV-cache oversubscription → livelock

This is the **opposite** of the sibling [`agentic_coding/`](../agentic_coding/) throttle demo.
There, KV is pinned so exactly one sequence runs and the rest wait cleanly. Here we deliberately
**oversubscribe** the KV cache: the scheduler *optimistically admits* several agentic-coding
requests (their prompts fit), then — because each keeps generating (`ignore_eos`) past what the
cache can hold — vLLM is forced to **preempt** running sequences, **throwing away a whole run's
KV context** and re-doing it from scratch. Under continuous load the cache thrashes and useful
throughput collapses.

This reproduces the user's scenario: *"agents 1 & 2 running, agent 3 also begins; the budget
expects 1+2+3 to fit, but in reality the total produced exceeds the limit, so it throws away a
run's context — run continuously until nothing gets done."*

## It's a livelock, not a true deadlock (and why)

vLLM's V1 scheduler **structurally cannot hard-deadlock** — we verified this in the 0.25.1
source:
- Preemption is **newest-first** (`self.running.pop()`), so the **oldest request is never
  preempted** and always drains (`v1/core/sched/scheduler.py`).
- The engine **refuses to start** unless one max-length sequence fits, and an over-length prompt
  is **rejected at admission**, not hung (`kv_cache_utils.py`, `input_processor.py`).
- Preemption is **recompute-only** (no CPU swap in V1): `_preempt_request` frees **all** the
  victim's KV blocks and resets `num_computed_tokens = 0`, re-queuing it — the generated *text*
  is kept but its KV is discarded and must be **re-prefilled**. That is literally "lose all
  context of a run."

So the reachable end-state is a **livelock / goodput-collapse**: `kv_cache_usage_perc` pinned at
~100%, `num_preemptions_total` climbing, requests waiting on `reason="capacity"`, and
completions-per-minute cratering — while the oldest one trickles out very slowly. The watchdog
flags this as `LIVELOCK DETECTED`.

## How the oversubscription is created

`run_server_preempt.sh` serves the same Qwen model but with a deliberately tiny, over-admitting
KV cache (flags confirmed in vLLM 0.25.1):

| Flag | Why |
|---|---|
| `--num-gpu-blocks-override N` | Shrink KV to a handful of blocks. vLLM's help literally says *"Used for testing preemption."* |
| `--no-scheduler-reserve-full-isl` | **The key flag** — admit on the *current* footprint, not the full output length, so extra requests "begin running" before the cache discovers it can't sustain them. |
| `--watermark 0.0` | No reserved free blocks → maximize churn. |
| `--max-num-seqs ≤ N` | Allow several concurrent sequences. |

**Model-specific gotchas found on this box (`Qwen/Qwen3.6-35B-A3B-FP8`, a hybrid Mamba-MoE):**
- It has **Mamba layers**: each concurrent decode sequence needs one Mamba cache block, and
  `--num-gpu-blocks-override` caps those — so **`--max-num-seqs` must be ≤ `NUM_GPU_BLOCKS`** or
  the engine won't start.
- Attention **block size is 2096 tokens** (aligned to the Mamba page), so `kv_budget.py`'s dense
  16-token/40,960-byte math from `agentic_coding/` does **not** apply here — that's why we size
  by *blocks*, not bytes.
- With `NUM_GPU_BLOCKS=10`, `MAX_MODEL_LEN=8192`: vLLM reports **GPU KV cache size ≈ 11,702
  tokens, max concurrency ≈ 1.43× a full sequence** — i.e. barely more than one full run fits,
  so a handful of concurrent workers guarantees thrash.

## Files

| Path | Purpose |
|---|---|
| `run_server_preempt.sh` | Oversubscription vLLM server (tiny KV, over-admission, no spec-decode) |
| `deadlock_run.py` | Continuous load (reuses the real `agentic_coding` coding prompt) + `/metrics` sampler + livelock watchdog + summary |
| `run_preempt.sh` | One-shot driver: prepare task → check server → run the load |

## Run it

```bash
# 1) Start the oversubscription server (reloads the 35 GB model, ~5 min).
MAX_MODEL_LEN=8192 NUM_GPU_BLOCKS=10 MAX_NUM_SEQS=10 GPU_UTIL=0.5 ./run_server_preempt.sh
#    (max_num_seqs MUST be <= num_gpu_blocks for this hybrid model.)

# 2) Drive continuous load for a fixed duration.
WORKERS=6 DURATION=120 MAX_TOKENS=4096 SEED_TOKENS=1800 ./run_preempt.sh
```

Each worker fires single-shot `/v1/chat/completions` requests with the **real agentic-coding
prompt** (system + "refactor `metrics_pipeline.py`" + the generated `agent_seed.py`),
`ignore_eos:true`, and a large `max_tokens`, relaunching immediately so ≥ `WORKERS` are always
in flight. Each request **is "a run"** — a preemption discards its context.

## Outputs (`results/preempt_<ts>/`)

| File | Holds |
|---|---|
| `kv_metrics.jsonl` | per-sample `running`, `waiting`, `waiting_capacity`, `kv_cache_usage_perc`, `num_preemptions_total`, client completions — the KV/preemption analysis log |
| `events.log` | human-readable transition rows + `LIVELOCK DETECTED` lines |
| `requests.jsonl` | one line per finished/failed run (latency, `finish_reason`) |
| `transcripts/` | the actual code each completed run produced |
| `summary.json` | verdict + totals (preemptions, peak/avg KV, goodput, latency, livelock seconds) |
| `run_meta.json` | run configuration |

## What you should see

`kv_cache_usage_perc` pinned near 1.0, `num_requests_running` capped at ~1–2 while the rest sit
in `waiting`, at least one preemption (a run's context thrown away), and completions collapsing
to ~zero — the goodput-collapse livelock, flagged as `LIVELOCK DETECTED` in `events.log`.

### Measured result (8 workers × 90 s, `max_tokens=2048`, `NUM_GPU_BLOCKS=10`)

```text
KV cache size:        11,702 tokens  (max concurrency 1.43x an 8192-token seq)
requests started:     8
requests completed:   0        <-- nothing finished
requests "failed":    8        <-- all client-side 180 s TIMEOUTS, not errors:
                                    the server made ~no progress on them
goodput:              0.0 completions/min
KV usage:             peak 100%, avg 79%
running / waiting:    max 2 / max 7   (mean 1.5 / 6.4)
preemptions:          2  (context discarded + re-queued)
livelock detected:    41.2 s of the 90 s run
verdict:              LIVELOCK / goodput-collapse observed
```

This is the empirical form of "run continuously until nothing gets done": **eight concurrent
agentic-coding runs, and not one completed** — the oversubscribed KV cache lets only ~1.5 run at
a time and they make so little progress that every request hit the client's 180 s patience.

**Why it's a livelock, not a hard hang:** with a *longer* client timeout the single oldest
request would eventually drain (vLLM protects the oldest from preemption). Raise `--timeout` to
see the oldest trickle out while the rest still starve. The dominant collapse mechanism here was
KV-capacity **queueing** plus stalled decode; **preemption** (the literal "throw away a run's
context") also occurs — to make it climb faster, use more workers with smaller `max_tokens` so
requests turn over and the admit→evict cycle repeats.

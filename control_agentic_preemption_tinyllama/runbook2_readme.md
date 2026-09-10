# Runbook 2 — hunting a TinyLlama livelock: big prompts, small chunks, long decodes

Goal: find a config where the over-admit TinyLlama server (`runbook_readme.md`
setup: KV = 2048 tokens, `--no-scheduler-reserve-full-isl --watermark 0.0`)
produces **massive preemption churn and drastic latency inflation**, like the Qwen
livelock. New knob: `MAX_NUM_BATCHED_TOKENS` (prefill chunk size) — small chunks
let a big prompt be admitted with almost no free KV and keep grabbing blocks
chunk-by-chunk. All runs 2026-09-09, one wave, client `TIMEOUT=1200`.

```bash
# the search (bypass = over-admission reaches the engine):
POLICY=bypass PROMPT_TOKENS=1700 MAX_TOKENS=340  MAX_NUM_BATCHED_TOKENS=256 ./run_control_experiment.sh   # A: deadlock!
POLICY=bypass PROMPT_TOKENS=1000 MAX_TOKENS=900  MAX_NUM_BATCHED_TOKENS=256 ./run_control_experiment.sh   # B: mild
POLICY=bypass PROMPT_TOKENS=120  MAX_TOKENS=1800 MAX_NUM_BATCHED_TOKENS=256 CONCURRENCY=16 ./run_control_experiment.sh  # C: thrash
```

---

## Attempt A — "prompts as big as possible" → total DEADLOCK, not livelock

`results/bypass_20260909_231502/` — 1700-tok prompt + 340 output = 2040 tokens =
**128 blocks/request**. One request prefilled, grew to computed=2032 (127 blocks),
asked for block 128 → preempted. It can never resume: the pool's **null block**
leaves only **127 usable blocks**, and a resume needs all computed blocks up
front. FCFS head-of-line → the scheduler sat **idle for 20 minutes** (KV 0%,
running=0, 8 waiting, 127 blocks free):

```
   4.244s  running=1  waiting=7  KV=100.00%  done=0
   4.470s  running=0  waiting=8  KV=  0.00%  done=0   <- PREEMPT x1   # ...then silence
1200.162s  running=0  waiting=7  KV=  0.00%  done=0   # clients time out
```

**Result:** 2 preemptions, **0/8 completed**, all latencies = the 1200 s timeout.
Worse than livelock, but not the churn we wanted. Rule: keep per-request total
≤ ~126 blocks (2016 tokens).

## Attempt B — 1000 + 900 (93% KV each) → seniority saves it

`results/bypass_20260909_233739/` — only **7 preemptions**, every victim evicted
exactly once at computed=1008, 8/8 done in 90 s. Two things defuse the thrash:
vLLM preempts from the **tail** of the running list, so the senior request always
progresses; and a preempted request's **resume needs all its computed blocks at
once** (~64 here), so evicted juniors can't storm back in. Serial-with-waste, no
livelock.

## Attempt C — invert it: tiny prompts, huge decodes → sustained thrash ✅

`results/bypass_20260909_234051/` — 120-tok prompt (6% KV) + 1800 output → each
request *fits trivially at admission* but grows to **94% KV**; 16 at once. All 16
are admitted immediately (16×120 ≈ 94%), then all grow 1 tok/step, the pool
empties within seconds, tails get evicted — and because a young victim's resume is
*cheap* (few computed blocks), they keep storming back and getting evicted again:

**Result: 65 true preemptions** (15/16 requests evicted, up to 6× each; 81
admissions = 16 fresh + 65 re-admits), **~26,100 computed tokens thrown away**,
KV avg 79.5% / peak 100%, repeated livelock windows. Latency inflates ~6–10×: a
~20 s job takes a 21 → 214 s staircase (mean 116 s, p95 200 s). 16/16 complete —
TinyLlama's 16-token blocks + tail-eviction seniority always drain the senior
request, so goodput never hits literal zero (that outcome needs Attempt A's
geometry, which deadlocks instead).

---

## Control — kv-gate on config C: the 50% threshold FAILS, 5% works

```bash
POLICY=kv-gate THRESHOLD=0.50 PROMPT_TOKENS=120 MAX_TOKENS=1800 MAX_NUM_BATCHED_TOKENS=256 CONCURRENCY=16 ./run_control_experiment.sh
POLICY=kv-gate THRESHOLD=0.05 PROMPT_TOKENS=120 MAX_TOKENS=1800 MAX_NUM_BATCHED_TOKENS=256 CONCURRENCY=16 ./run_control_experiment.sh
```

- **thr 0.50** (`results/kv-gate_20260909_234522/`): **15 preemptions** — the gate
  admits on *current* KV (e.g. 38%), blind to the admitted request's future growth
  to 94%, so it co-admits pairs that cannot coexist. No better than bypass.
- **thr 0.05** (`results/kv-gate_20260909_235011/`): **0 preemptions**, 16/16, no
  livelock, peak KV 94.5%. Clean network queueing: one admit per ~22.8 s drain
  (22.9, 45.7, …, 365.3 s staircase; mean 194 s).

| config C (16 × 120+1800) | bypass | kv-gate 50% | kv-gate 5% |
|---|---|---|---|
| true preemptions | **65** (15/16 req) | 15 | **0** |
| computed tokens wasted | ~26,100 | ~10,700 | 0 |
| admissions (fresh + resume) | 16 + 65 | 16 + 15 | 16 + 0 |
| peak KV (avg) | 100% (79.5%) | 100% (73.0%) | 94.5% (50.4%) |
| completed | 16/16 | 16/16 | 16/16 |
| latency mean / max | 116 / 214 s | 125 / 230 s | 194 / 365 s |
| where requests wait | in-engine (evict/recompute) | both | at the gateway |

## Observations

- **The livelock lever is growth-to-admission-footprint ratio, not prompt size.**
  Big prompts (A, B) are self-limiting: expensive resumes keep evicted requests
  out, and the extreme case deadlocks outright. Maximum churn comes from requests
  that are *cheap to admit and cheap to resume but enormous when grown* — 65
  evictions from 6%-KV prompts growing 15×.
- **A 2048-KV server cannot host a 2040-token request twice.** The null block
  makes 127 of 128 blocks usable; eviction past 2016 tokens is unrecoverable →
  permanent stall (A). An admission guard sized to *completed* length would have
  rejected it up front.
- **KV-gating must budget for future growth.** Threshold 50% worked in runbook 1
  only because the *prompt* already exceeded it. Here current-KV ≤ 50% admits
  doomed pairs; the safe threshold is `1 − full_footprint` (~5%). An external
  controller needs the request's expected total footprint, not just live KV.
- **The price of safety is throughput on purpose:** the 5% gate fully serializes
  (mean 194 s vs 116 s bypass) but eliminates all recompute waste and bounds KV;
  bypass "wins" mean latency only by overlapping 2–3 requests between evictions
  and re-paying ~26k tokens of prefill.

**Cleanup** (the server run instruments the shared venv):
```bash
python3 probe_patch.py revert
```

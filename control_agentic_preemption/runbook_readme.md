# Runbook — side-channel KV controller vs. the preemption livelock

Same livelock-prone setup for both runs: an **over-admit vLLM server**
(`RESERVE_ISL=0`, tiny KV) + the **same load** (8 workers, `SEED_TOKENS=1800`,
`MAX_TOKENS=4096`, `DURATION=120`, `TIMEOUT=1200` — long client patience so queued
requests drain instead of timing out). Only the gateway policy changes.
`run_control_experiment.sh` boots the server + gateway, runs the load, and parses
ground-truth preemptions.

> Both runs below executed 2026-09-09 with identical client patience
> (`TIMEOUT=1200`). If the model isn't in the page cache, boot with
> `VLLM_ENGINE_READY_TIMEOUT_S=1800` — cold shard loading can exceed vLLM's
> default 600 s engine-ready timeout and kill the server mid-boot.

---

## Experiment 1 — baseline (bypass, no delaying)

```bash
cd control_agentic_preemption
POLICY=bypass SEED_TOKENS=1800 MAX_TOKENS=4096 DURATION=120 TIMEOUT=1200 ./run_control_experiment.sh
```

**Output:** `results/preempt_20260909_150736/`
(`events.log`, `probe_summary.json`, `summary.json`, `gateway_logs/`)

The server admits **two** requests, KV pins at 100%, and it preempts forever —
`events.log`:

```
   0.877s  running=2  waiting=6  KV= 88.89%  done=0   <- ADMIT x2
   5.185s  running=1  waiting=7  KV= 55.56%  done=0   <- PREEMPT x1
   5.406s  running=2  waiting=6  KV=100.00%  done=0   <- ADMIT x1     # thrash: preempt<->re-admit
   ...
  10.982s  *** LIVELOCK DETECTED *** KV=100.0%  running=2 waiting=6  completions/10s=0
```

Ground truth (`probe_summary.json`): **2620 true preemptions**, all absorbed by
just **3 distinct requests** (1048 / 965 / 607 evictions each) — 2623 admissions of
which only 4 were fresh; the other 2619 were re-admissions after eviction. Each
preemption hit at 2047 computed tokens, i.e. the engine repeatedly threw away a
nearly full sequence.

**Result:** true preemptions **2620**, livelock **56.8 s** (of the 120 s watch
window), peak KV **100%** (avg 78.5%). Even with 1200 s of client patience only
**2 / 8 completed** (3 client failures + 3 gateway 502s; the 4 requests still stuck
at the end hit the gateway's 1200 s upstream timeout). Goodput **1.0/min**, mean
latency 711 s, p95 938 s. Verdict: `LIVELOCK / goodput-collapse observed`.

*(vs. the earlier `TIMEOUT=180` baseline: longer patience rescues 2 stragglers
after arrivals stop, but the livelock itself — thrash, 0 completions while loaded —
is unchanged.)*

---

## Experiment 2 — control (kv-gate at 50%)

```bash
cd control_agentic_preemption
POLICY=kv-gate THRESHOLD=0.50 SEED_TOKENS=1800 MAX_TOKENS=4096 DURATION=120 TIMEOUT=1200 ./run_control_experiment.sh
```

**Output:** `results/preempt_20260909_155317/`
(`gateway_logs/gateway_events.log`, `events.log`, `probe_summary.json`, `summary.json`)

One request in flight drives KV to ~67% (> 50%), so the gateway **holds all others**
and releases them one at a time (~82 s apart) as each drains —
`gateway_logs/gateway_events.log`:

```
   6.339s  ADMIT req=g00001 wait=0.0s     kv=0.00%  (<= thr 0.50)  # admit #1, KV climbs to ~67%
  87.973s  ADMIT req=g00002 wait=77.623s  kv=0.00%                 # #1 drained -> KV fell -> next
 169.570s  ADMIT req=g00003 wait=77.781s  kv=0.00%
 ...       one admit every ~82s ...
 660.016s  ADMIT req=g00009 wait=78.05s   kv=0.00%
```

Server never oversubscribes — `events.log` stays at `running=1`, completing one at a time:

```
   0.694s  running=1  waiting=0  KV= 44.44%  done=0   <- ADMIT x1
  81.723s  running=0  waiting=0  KV=  0.00%  done=1   <- COMPLETE x1 + ARRIVE x1
  82.265s  running=1  waiting=0  KV= 44.44%  done=1   <- ADMIT x1
```

**Result:** true preemptions **0** (9 fresh admissions, all
`FINISHED_LENGTH_CAPPED`), **9 / 9 completed, 0 failed**, goodput **4.5/min**, peak
KV **66.7%** (avg 57.8%), **no livelock**. Serial latencies climb with queue
position (mean 399 s, p95 654 s); gate wait is a steady ~78 s per position.

---

## Baseline → control

| | bypass | kv-gate 50% |
|---|---|---|
| true preemptions (probe) | **2620** | **0** |
| distinct requests thrashed | 3 (1048/965/607 evictions) | 0 |
| livelock seconds | 56.8 | 0 |
| completed | 2 / 8 | **9 / 9** |
| failed | 6 (3 errors + 3 502s) | 0 |
| goodput | 1.0 / min | 4.5 / min |
| mean / p95 latency | 711 s / 938 s | 399 s / 654 s |
| peak KV (avg) | 100% (78.5%) | 66.7% (57.8%) |

The external KV controller turns in-engine thrash into out-of-engine queueing on an
unchanged server: **preemptions 2620 → 0, and every request completes (2/8 → 9/9)**
— with *lower* latency than the thrashing baseline, since work is never thrown
away. Cost is serialization — requests wait their turn at the gateway (~82 s per
queue position) rather than being admitted and evicted inside the engine.

> 50% works because one seed sequence alone exceeds 50% KV (~67% here). For smaller
> prompts, set `THRESHOLD` below one sequence's footprint (or size prompts so one
> exceeds it).

**Cleanup** (the server run instruments the shared venv):
```bash
python3 probe_patch.py revert
```

# Runbook — TinyLlama: preemption thrash vs. side-channel network queueing

Same design as `../control_agentic_preemption`, on the **parent repo's TinyLlama
framework** (`../run_server.sh`): TinyLlama-1.1B-Chat, `--max-model-len 2048`,
`--max-num-seqs 16`, KV pinned to **exactly 2048 tokens**
(`--num-gpu-blocks-override 128`), one barrier-synced wave of 8 identical LooGLE
`/v1/completions` requests (`temperature=0`, `ignore_eos`, client `TIMEOUT=1200`).

Twist vs. the parent's proof: the server runs **over-admit**
(`--no-scheduler-reserve-full-isl --watermark 0.0`) and each request *fits at
admission but not at full growth* — 500-token prompt (~24% KV) + `max_tokens=1200`
(~83% KV total). Only the gateway policy changes between runs. Both run 2026-09-09.

---

## Experiment 1 — baseline (bypass): preemption thrash

```bash
POLICY=bypass ./run_control_experiment.sh
```

**Output:** `results/bypass_20260909_225304/`

Three requests admitted together, KV saturates, the scheduler evicts mid-generation —
`events.log`:

```
   0.221s  running=3  waiting=5  KV= 75.59%  done=0   <- ADMIT x3
   ...     KV climbs as all three generate: 82% ... 92% ... 95% ...
  15.638s  *** LIVELOCK DETECTED ***  KV=100.0%  no completion for 10s with 7 outstanding
```

**Result:** true preemptions **12** (7 of 8 requests evicted; 20 admissions = 8
fresh + 12 re-admits), **~8,980 computed tokens thrown away** and recomputed
(evictions at 532–1008 computed tokens), peak KV **100%**, livelock episodes 1.6 s.
All 8 complete, but latency is churn-inflated: **14 → 82 s** (mean 47.3, p95 71.6)
for a ~13 s job. Verdict: `PREEMPTION THRASH / high latency`.

---

## Experiment 2 — control (kv-gate at 50%): network queueing

```bash
POLICY=kv-gate ./run_control_experiment.sh
```

**Output:** `results/kv-gate_20260909_225634/`

One in-flight request peaks at ~84% KV (> 50%), so the gateway holds the other
seven **in the network** and releases one per drain — `gateway_events.log`:

```
   1.170s  ADMIT req=g00001 wait=0.0s  kv=0.00% (<= thr 0.50)   # 7 others HELD
  16.296s  ADMIT req=g00002        # #1 drained -> KV fell -> next
  ...      one admit every ~15s ...
```

Server never queues or oversubscribes — `events.log` stays `running=1 waiting=0`.

**Result:** true preemptions **0** (8 fresh admits, all `FINISHED_LENGTH_CAPPED`),
**8/8 completed**, no livelock, zero recompute waste, peak KV 84.3%. Latency is a
clean serial staircase: 15.2, 30.3, …, 121.5 s — one ~15 s service time per queue
position.

---

## Baseline → control

| | bypass | kv-gate 50% |
|---|---|---|
| true preemptions (probe) | **12** (7/8 requests) | **0** |
| computed tokens wasted | ~8,980 | 0 |
| admissions (fresh + re-admit) | 8 + 12 | 8 + 0 |
| peak KV (avg) | 100% (73.5%) | 84.3% (53.8%) |
| livelock episodes | 1.6 s | 0 |
| completed | 8 / 8 | 8 / 8 |
| latency mean / p95 | 47.3 s / 71.6 s | 68.3 s / 106.2 s |
| where requests wait | in-engine (evict + recompute) | at the gateway (network queue) |

## Observations

- **Same conclusion as the Qwen study, on an unchanged over-admitting server:**
  the external KV controller converts in-engine eviction thrash into out-of-engine
  queueing — 12 → 0 preemptions, ~9k recomputed tokens → 0.
- **TinyLlama's thrash is lossy, not terminal.** Unlike the Qwen livelock (0
  completions under load), bypass here still finishes the wave — the damage is
  wasted KV-writes and a staircase of inflated latencies. Scale `MAX_TOKENS` /
  `CONCURRENCY` up to push it toward full livelock.
- **The trade is predictability, not raw speed.** At this scale bypass has lower
  mean latency (47 vs 68 s) because 2–3 requests overlap between evictions; the
  gate fully serializes. What kv-gate buys: bounded KV, no wasted work, and
  deterministic ~15 s-per-position queueing instead of eviction churn.
- **Why threshold 50% works:** a full request (~83% KV) exceeds it while its
  prompt (~24%) does not — the gate admits onto an idle server, then the admitted
  request's growth locks the gate until it drains. For other sizes, keep
  `THRESHOLD` below one request's full footprint.

**Cleanup** (the server run instruments the shared venv):
```bash
python3 probe_patch.py revert
```

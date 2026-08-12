# Agentic preemption — full design & findings

Deliberately **oversubscribe** a vLLM KV cache with parallel agentic-coding requests and watch
what happens. It's the opposite of a KV *throttle* demo (the former sibling `agentic_coding/`,
since removed, which pinned KV so exactly one sequence ran and the rest waited cleanly). Here the
scheduler *optimistically admits* several requests, then — because each keeps generating
(`ignore_eos`) past what the cache can hold — vLLM **preempts** running sequences, **throwing away
a whole run's KV context** and recomputing it. Under continuous load the cache thrashes and useful
throughput collapses.

Target scenario: *"agents 1 & 2 running, agent 3 also begins; the budget expects 1+2+3 to fit, but
in reality the total produced exceeds the limit, so it throws away a run's context — run
continuously until nothing gets done."*

---

## 1. It's a livelock, not a true deadlock

vLLM's V1 scheduler **structurally cannot hard-deadlock** (verified in the 0.25.1 source):

- **Newest-first preemption.** FCFS preempts `self.running.pop()` (the newest), so the **oldest
  request is never preempted** and always drains (`v1/core/sched/scheduler.py`).
- **Startup + admission guarantees.** The engine refuses to start unless one max-length sequence
  fits, and an over-length prompt is *rejected at admission*, not hung (`kv_cache_utils.py`,
  `input_processor.py`).
- **Recompute-only preemption** (no CPU swap in V1). `_preempt_request` frees **all** the victim's
  KV blocks, sets `num_computed_tokens = 0`, and **re-queues it to the *front* of waiting**
  (`self.waiting.prepend_request`). The generated *text* is kept, but its KV is discarded and must
  be re-prefilled — literally "lose all context of a run."

So the reachable end-state is a **livelock / goodput-collapse**: `kv_cache_usage_perc` pinned near
100%, requests waiting on `reason="capacity"`, completions-per-minute cratering — while the oldest
one trickles out very slowly. The watchdog flags this as `LIVELOCK DETECTED`.

---

## 2. ⚠️ The `num_preemptions_total` metric is broken here — don't trust it

This is the single most important finding. On this build, `vllm:num_preemptions_total` **massively
undercounts real preemptions** — it appears to count *distinct requests ever preempted*, not
*evictions*. We proved it with source-level instrumentation (§5b):

| | Recipe C (90 s) | Recipe A (~174 s) |
|---|---|---|
| `num_preemptions_total` (metric) | **1** | **1** |
| **true preemptions** (probe) | **243** | **368** |

All of one request. So the metric is off by ~200–370×. Consequently:

- We **removed `num_preemptions_total`** from `kv_metrics.jsonl` / `summary.json` — reporting it as
  if trustworthy is worse than omitting it.
- The **reliable tell of preemption thrash is the KV-usage oscillation** (`100% ↔ 56%` = a
  request's blocks freed on preemption, then re-allocated on re-admit), plus the client-inferred
  and probe counts below.
- An earlier analysis that concluded "A/C stall rather than preempt" was **wrong** — it trusted the
  metric. A, B, and C are *all* heavy preemption thrash.

---

## 3. Anatomy of the livelock (why "one request preempted" yet `running=2`)

Under Recipe A the aggregate looks like `running` oscillating 1↔2 and `waiting` 6↔7, but the probe
shows the true structure — **8 requests split into three roles**:

- **1 protected runner (the oldest).** FCFS never preempts the oldest, so whichever request grabs
  the first running slot holds it continuously and grinds forward (slowly). It never appears in the
  PREEMPT stream — only once at the end as an abort. With enough client patience it *would* finish
  (that's Recipe C's two completions at 535 s / 1034 s).
- **1 perpetual victim.** The second admitted request is the eviction target. Preempted →
  `num_computed_tokens=0` → **re-queued to the *front*** → re-admitted before anyone else →
  preempted again… **hundreds of times**, zero net progress, always evicted at `computed≈2047`
  (~one 2096-token block). Every `running 1→2` / `2→1` in `events.log` is this one request bouncing.
- **6 starved.** Because the victim keeps jumping to the front of the queue, the other six **never
  get admitted** — they wait the whole run and abort at timeout. They have zero preemptions (never
  ran) — not because there's no thrash, but because they never reached a running slot.

So `running=2` = *protected oldest + victim*, `running=1` = *oldest only (victim momentarily in
waiting)*. At the client timeout **all 8 abort**, 0 complete.

---

## 4. How the oversubscription is created (server)

`run_server_preempt.sh` serves the same Qwen model with a deliberately tiny, over-admitting KV
cache (flags confirmed in vLLM 0.25.1):

| Flag | Why |
|---|---|
| `--num-gpu-blocks-override N` | Shrink KV to a handful of blocks. vLLM's help literally says *"Used for testing preemption."* |
| `--no-scheduler-reserve-full-isl` | Admit on the *first prefill chunk* fitting, not the whole prompt — so extra requests "begin running" before the cache can sustain them. **Toggled by `RESERVE_ISL`** (0 = this flag on = over-admit; 1 = omit it = clean admission, Recipe D). |
| `--watermark 0.0` | No reserved free blocks → maximize churn. |
| `--max-num-seqs ≤ NUM_GPU_BLOCKS` | Allow several concurrent sequences (see Mamba note). |

**Model-specific gotchas (`Qwen/Qwen3.6-35B-A3B-FP8`, a hybrid Mamba-MoE):**
- **Mamba layers** → each concurrent decode sequence needs one Mamba cache block, which
  `--num-gpu-blocks-override` caps → **`--max-num-seqs` must be ≤ `NUM_GPU_BLOCKS`** or it won't
  start.
- **Attention block size = 2096 tokens** (aligned to the Mamba page). So byte-per-token math does
  not apply; we size by *blocks*. `kv_cache_usage_perc` is `1 − free_blocks/total` (allocated
  fraction), which is why it genuinely drops when a preemption frees blocks.
- With `NUM_GPU_BLOCKS=10, MAX_MODEL_LEN=8192`: vLLM reports **≈ 11,702 KV tokens, max concurrency
  ≈ 1.43×** a full sequence — barely more than one full run fits, so a few workers guarantee thrash.

---

## 5. The load, and how we make the behavior legible

### The workload
Each worker fires single-shot `/v1/chat/completions` requests with the **real agentic-coding
prompt** — system prompt + "refactor `metrics_pipeline.py`" instruction + a synthetic legacy module
(`prompts.py` supplies `SYSTEM_PROMPT`/`build_first_user`/`TASK_INSTRUCTION`; `prepare_task.py`
generates `generated/agent_seed.py`). `ignore_eos:true` + a large `max_tokens` force each request to
keep generating; workers relaunch immediately so ≥ `WORKERS` are always in flight. **Each request
is "a run"** — a preemption discards its context. (These two files are local because the former
sibling `agentic_coding/` was deleted; the folder is now self-contained.)

### 5a. Client-side inferred causes — always on, no setup
Because a request leaves `running` only by **preemption or completion** and enters only by
**admission**, the sampler in `deadlock_run.py` can label every transition from polled aggregates:

```
residual = Δrunning + completions      # = admissions − preemptions
  residual < 0 → PREEMPT   (a request left running, not via completion)
  residual > 0 → ADMIT     (a request entered running)
  Δcompletions → COMPLETE ;  new client requests → ARRIVE
```

Every `--sample-interval` (default **0.1 s**) it scrapes `/metrics` for
`num_requests_running / num_requests_waiting / kv_cache_usage_perc` (and the labeled
`num_requests_waiting_by_reason{reason="capacity"}`), reads its own `client_started/completed`
counters, and writes each row with a **`cause`** field plus cumulative `inferred_preemptions /
inferred_admissions`. `events.log` shows it inline:

```
6.108s  running=2  waiting=6(cap=6)  KV=100.00%  done=0   <- ADMIT x1
6.329s  running=1  waiting=7(cap=7)  KV= 55.56%  done=0   <- PREEMPT x1
```

This answers "is that KV drop a preemption?" directly. **Caveat:** it's polled, so a preempt+admit
pair that nets to zero between samples is missed → `inferred_preemptions` is a good **lower bound**
(e.g. 241 inferred vs 243/368 true). It's the correct *cause* per observed transition, not the
exact rate.

### 5b. Server-side probe mode — opt-in, exact
For ground truth we patch the scheduler to log each real event. `probe_patch.py apply|revert|status`
reversibly inserts six `logger.info("PROBE …")` lines (backs up the venv files to `*.bak`, always
revert after). The patched **server log** then records, and `parse_probe.py <server_log> --out <dir>`
turns it into `probe_events.jsonl` + `probe_summary.json`. Event types:

| PROBE event | Scheduler site | Means |
|---|---|---|
| `PREEMPT` | `_preempt_request` | an eviction (with `req`, `computed`) — the true preemption count |
| `ADMIT` | `self.running.append` | a request enters running; `kind=new` (fresh) or `kind=resume` (re-admit after preempt) |
| `FREEREQ` | `_free_request_blocks` | a request's KV released; `status=RUNNING` = preempt-free, `FINISHED_ABORTED` = abort/complete |
| `POOLFREE` | `block_pool.free_blocks` | blocks physically returned to the free pool — **this is what moves the KV gauge**; several per FREEREQ |
| `SKIPRUN` | running-loop `num_new_tokens==0` | a resident request not advanced this step (hybrid-Mamba block-aligned chunking) |
| `DRAIN` | `_drain_deferred_frees` | deferred blocks returned later (async scheduling) |

The causal chain per thrash cycle:
```
PREEMPT (evict victim) → FREEREQ (status=RUNNING) → POOLFREE ×~4 (blocks to pool → KV 100%→56%)
                                                  → ADMIT (kind=resume) → …repeat…
```
`probe_summary.json` reports `true_preemptions`, `preemptions_per_request`, `total_admissions /
re_admissions_after_preempt` (should track `true_preemptions`), and the FREEREQ status split.

`run_probe_experiment.sh` wraps one full cycle per experiment: ensure patched → **restart a fresh
probe server** (own log) → run the load → parse into the results folder → stop the server. Call it
once per experiment so each gets an isolated probe capture; `probe_patch.py revert` after the last.

---

## 6. Recipe D — the clean-queueing opposite (no preemption)

Flip two things and the thrash becomes the parent's clean throttle — they do **different jobs**:

1. **Large prompt** (`SEED_TOKENS≈6000` ≈ 3 of the ~5.5 blocks) is the *physical* reason two can't
   coexist (2×3 > 5.58), so **only one fits at a time**.
2. **Admission guard ON** (`RESERVE_ISL=1`, i.e. drop `--no-scheduler-reserve-full-isl`) makes the
   scheduler check the **whole** prompt fits before admitting, instead of just the first chunk — so
   with chunked prefill on it can't sneak a second request in and then preempt it.

Result (measured): **1 running, 7 waiting, 0 preemptions** (corroborated by KV steady ~74% with *no*
oscillation), requests drain **serially** (latencies 36/70/105/139/173 s), goodput 4/min. Requests
that "fail" here are honest **queue-timeouts** (never admitted before the client's patience), not
thrash. Neither large-prompt-alone nor guard-alone suffices: small prompts let many in (guard only
checks input length), and guard-off + chunked prefill re-opens over-admission.

---

## 7. Files

| Path | Purpose |
|---|---|
| `run_server_preempt.sh` | Oversubscription vLLM server; `RESERVE_ISL` toggles over-admit (0) vs clean-admission (1); tiny KV via `--num-gpu-blocks-override`; no spec-decode |
| `prompts.py` | `SYSTEM_PROMPT` + `build_first_user` + `TASK_INSTRUCTION` (the realistic coding task) |
| `prepare_task.py` | Generates `generated/agent_seed.py` + `agent_task.json` (the file to refactor) |
| `deadlock_run.py` | Continuous load + `/metrics` sampler with **inferred cause labels** + livelock watchdog + summary |
| `run_preempt.sh` | One-shot client driver: prepare task → check server → run the load |
| `probe_patch.py` | `apply`/`revert`/`status` the reversible scheduler PROBE instrumentation in the venv |
| `parse_probe.py` | Parse a PROBE server log → `probe_events.jsonl` + `probe_summary.json` (ground truth) |
| `run_probe_experiment.sh` | One experiment against its own fresh probe server (apply → restart → run → parse → stop) |

---

## 8. Outputs (`results/preempt_<ts>/`)

| File | Holds |
|---|---|
| `kv_metrics.jsonl` | per-sample `running/waiting/waiting_capacity/kv_cache_usage_perc`, `client_started/completions`, an inferred **`cause`**, and cumulative **`inferred_preemptions/admissions`**. (`num_preemptions_total` is deliberately **omitted** — see §2.) |
| `events.log` | human-readable transition rows tagged with the inferred cause (`… <- PREEMPT x1`) + `LIVELOCK DETECTED` |
| `summary.json` | verdict + peak/avg KV, goodput, latency, livelock seconds, `inferred_preemptions/admissions/completions` |
| `requests.jsonl` | one line per request (latency, `finish_reason` / timeout) |
| `transcripts/` | code each **completed** run produced (empty when nothing finishes — that *is* the collapse) |
| `probe_events.jsonl` / `probe_summary.json` / `server_probe.log` | **probe mode only**: exact per-event ground truth (`true_preemptions`, `preemptions_per_request`, …) |

**Data sources:** the four `kv_metrics` gauges come from the server's open `/metrics` (Prometheus);
`client_*` counters and the inferred fields are computed by the client; the `probe_*` files come
from the *patched scheduler's log*, not `/metrics`.

---

## 9. Recipes (each restarts its own fresh probe server)

| # | Detail | Command |
|---|---|---|
| **A** | Nothing completes (big outputs, no run finishes → 0 done) | `WORKERS=8 DURATION=120 MAX_TOKENS=4096 TIMEOUT=180 ./run_probe_experiment.sh` |
| **B** | Turnover: small outputs → fast admit→evict cycle → `inferred_preemptions` / probe `true_preemptions` climb (goodput > 0) | `WORKERS=8 DURATION=120 MAX_TOKENS=512 TIMEOUT=180 ./run_probe_experiment.sh` |
| **C** | Livelock, not a hang: long patience → the oldest eventually drains while the rest starve | `WORKERS=8 DURATION=120 MAX_TOKENS=4096 TIMEOUT=1200 ./run_probe_experiment.sh` |
| **D** | Clean queueing (guard ON + large prompt → only 1 fits, no preemption) | `RESERVE_ISL=1 SEED_TOKENS=6000 MAX_TOKENS=1500 WORKERS=8 DURATION=120 TIMEOUT=180 ./run_probe_experiment.sh` |

After the last experiment: `python3 probe_patch.py revert`. For client-only runs (no probe server
restart), use `./run_preempt.sh` with the same env against an already-running server.

---

## 10. What you should see (measured)

Over-admit recipes (A/B/C): `kv_cache_usage_perc` oscillating **100% ↔ 56%** (the preemption
signature), `running` capped at 1–2 with 6–7 waiting, and **hundreds of preemptions of a single
request** in the probe log while `num_preemptions_total` reads ~1. Recipe A: **8 started, 0
completed, all abort at the client timeout**, goodput 0 — the empirical "run continuously until
nothing gets done." Recipe C shows the same thrash but, given 20 min of patience, the protected
oldest drains (proving livelock, not deadlock). Recipe D: KV steady ~74%, **0 preemptions**, serial
completions — the clean counterpart.

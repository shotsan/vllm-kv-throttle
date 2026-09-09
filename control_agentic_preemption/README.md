# Side-channel KV-observing request controller

This folder builds on `../agentic_preemption`. There we showed that an
**over-admitting** vLLM server (`--no-scheduler-reserve-full-isl`, tiny KV) can be
driven into a **preemption livelock** (Recipes A–C): several agentic-coding requests
are admitted on their first prompt chunk, their combined KV blows past the tiny
cache, and the scheduler preempts running sequences forever — KV pinned ~100%,
goodput collapses. Recipe **D** avoided this by changing the **server**
(full-ISL admission guard → clean queueing).

Here we recover that clean queueing **without touching the server's admission
mode**, using an **external side-channel controller** that observes the KV cache and
delays requests before they ever reach vLLM.

```
   clients ──► gateway.py (:8001) ──► vLLM (:8000, OVER-ADMIT / thrashing config)
   (8 workers)     │  observes /metrics kv_cache_usage_perc
                   └─ admits a request only when observed KV ≤ threshold (50%)
```

## The control policy (`--policy kv-gate`)

- Each request's prompt is sized to **~56% of usable KV** (`SEED_TOKENS=6000`,
  matching Recipe D's footprint). One in-flight request drives KV to ≈56%.
- The gateway polls vLLM `/metrics` and admits a request **only when observed KV ≤
  `--threshold` (default 0.50)**. Because one admitted request already exceeds 50%,
  **every other request is held at the gateway** until it drains and KV falls back
  under the threshold.
- Admission is **serialized** by a single gate lock so requests can't all slip
  through at KV≈0 at once. After one is let through, the gate waits until KV actually
  rises above the threshold (the admit "landed") or `--settle-timeout` elapses,
  before evaluating the next contender. The upstream call runs on a helper thread so
  the settle overlaps generation.

Net effect: **at most one >56% request in flight → no oversubscription → no
preemption → clean serial queueing**, enforced entirely from outside the server.

`--policy bypass` is a plain pass-through (no throttling): use it as a **negative
control** to reproduce the livelock *through* the gateway and prove the gateway
itself isn't what fixes things.

## Files

| File | Role |
|---|---|
| `gateway.py` | The side-channel controller: threaded proxy + `KvMonitor` + `Controller` admission policy. Writes `gateway_events.log`, `gateway_requests.jsonl`, `gateway_meta.json`. |
| `run_gateway.sh` | Start the gateway in front of a running vLLM. |
| `run_control_load.sh` | Run the agentic load, but with the client URL pointed at the gateway (`:8001`); the KV sampler still scrapes the real server (`:8000`). |
| `run_control_experiment.sh` | End-to-end orchestrator: probe-patch → over-admit server → gateway → load → parse probes + collect gateway logs → stop. |
| `run_server_preempt.sh`, `deadlock_run.py`, `prepare_task.py`, `prompts.py`, `parse_probe.py`, `probe_patch.py` | Copied from `../agentic_preemption` so this folder is self-contained. |

## Run it

```bash
cd control_agentic_preemption

# Controlled run: over-admit server + kv-gate -> expect clean queueing, ~0 preemptions
./run_control_experiment.sh

# Negative control: same server, gateway in BYPASS -> the livelock returns
POLICY=bypass ./run_control_experiment.sh

# when finished with ALL experiments, restore the venv:
python3 probe_patch.py revert
```

Server / gateway / load knobs are env vars (defaults in parentheses):

- Server: `RESERVE_ISL(0)` `NUM_GPU_BLOCKS(10)` `MAX_NUM_SEQS(10)` `MAX_MODEL_LEN(8192)` `GPU_UTIL(0.5)`
- Gateway: `POLICY(kv-gate)` `THRESHOLD(0.50)` `SETTLE_TIMEOUT(8.0)` `LISTEN_PORT(8001)`
- Load: `WORKERS(8)` `DURATION(120)` `MAX_TOKENS(1500)` `SEED_TOKENS(6000)` `TIMEOUT(180)`

> `RESERVE_ISL` is left at **0** on purpose — the server stays in the livelock-prone
> over-admit mode. The gateway is the only thing preventing over-subscription.

## Outputs (per run, under `results/<run>/`)

- `events.log`, `kv_metrics.jsonl`, `summary.json`, `requests.jsonl` — client-side
  KV samples, inferred admissions/preemptions, latency, goodput (same schema as
  `../agentic_preemption`).
- `gateway_logs/gateway_events.log`, `gateway_requests.jsonl` — the **admission
  ground truth**: per-request gate wait time and observed KV at admit.
- `server_probe.log`, `probe_events.jsonl`, `probe_summary.json` — the **preemption
  ground truth** from the instrumented scheduler (`true_preemptions`).

## What to expect

| | Preemptions (probe) | Completions | Per-request latency | Behavior |
|---|---|---|---|---|
| Recipe A (no gateway) | hundreds, KV ~100% | 0 (livelock) | timeouts | goodput collapse |
| **kv-gate (this)** | **~0** | serial, like Recipe D | admit + queue wait at gateway | clean queueing |
| bypass (this) | back to hundreds | 0 (livelock) | timeouts | proves gateway is the cause |

The gateway adds a visible **queue wait** (`gateway_requests.jsonl.wait_s`) in place
of the server's destructive preempt/recompute churn: requests wait *outside* the
engine instead of being admitted and thrown away inside it.

## Verification

`gateway.py` was smoke-tested against a mock upstream whose reported KV rises 56% per
in-flight request. With `kv-gate` (threshold 0.50), 4 concurrent clients were
serialized (returns at ~3s/6s/9s/12s, `wait_s` ≈ 3s each); with `bypass` all 4 ran
concurrently (~3s each). See `gateway.py`'s module docstring for the mechanism.

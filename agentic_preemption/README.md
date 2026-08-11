# agentic_preemption — quick commands

Force a vLLM **KV-cache oversubscription → preemption livelock**: many agentic-coding
requests are admitted into a deliberately tiny KV cache, their context gets thrown away
(preempted), and goodput collapses to ~zero. Full explanation + measured results in
[`DETAILS.md`](./DETAILS.md).

Two steps: **(1)** start the oversubscription server, **(2)** run the load. The server is a
heavy, long-running process (loads a 35 GB model, ~5 min) — start it once, then rerun step 2.

---

## 0. Prerequisites

- vLLM env with the model cached: `/home/ways_lab/Documents/LLM-Network-Study/.venv` (has `vllm` + `transformers`).
- API key is read automatically from `…/local_llm/.env` (`VLLM_API_KEY`).
- Run all commands from this folder:
  ```bash
  cd /home/ways_lab/repos/vllm-kv-throttle/agentic_preemption
  ```

## 1. Start the oversubscription server (terminal 1)

```bash
MAX_MODEL_LEN=8192 NUM_GPU_BLOCKS=10 MAX_NUM_SEQS=10 GPU_UTIL=0.5 ./run_server_preempt.sh
```

Wait for `Application startup complete`. Confirm the tiny KV geometry via the **open `/metrics`**
endpoint (no API key needed — unlike `/v1/*`):

```bash
curl -s http://127.0.0.1:8000/metrics \
  | grep -oE 'block_size="[0-9]*"|kv_cache_size_tokens="[0-9]*"|kv_cache_max_concurrency="[0-9.]*"' | sort -u
# expect: block_size="2096"  kv_cache_size_tokens="11702"  kv_cache_max_concurrency="1.42857..."
```

`/v1/models` (which reports `max_model_len`) **requires the API key** — it 401s without it, and
the key must match how the server was started. Source `.env` first so your shell key matches the
server (both derive it from the same file), otherwise the command silently 401s → empty output:

```bash
source /home/ways_lab/Documents/LLM-Network-Study/local_llm/.env 2>/dev/null   # sets VLLM_API_KEY
curl -s -H "Authorization: Bearer ${VLLM_API_KEY:-EMPTY}" http://127.0.0.1:8000/v1/models \
  | grep -o '"max_model_len":[0-9]*'
```

> If it returns nothing, your key doesn't match the server's. Check the server's key with
> `ps -eo args | grep -oE '\-\-api-key [^ ]+'` and pass that exact value. The keyless `/metrics`
> check above never has this problem.

> ⚠️ `MAX_NUM_SEQS` must be **≤ `NUM_GPU_BLOCKS`** (hybrid-Mamba model: one Mamba block per
> concurrent sequence). Raise `GPU_UTIL` if startup complains about free memory.

## 2. Run the experiment (terminal 2)

```bash
WORKERS=8 DURATION=90 MAX_TOKENS=2048 SEED_TOKENS=1800 ./run_preempt.sh
```

This prepares the coding task, checks the server, launches the load + KV sampler + livelock
watchdog, and writes everything to `results/preempt_<timestamp>/`.

## 3. Read the results

```bash
d=$(ls -td results/preempt_*/ | head -1)          # newest run

grep -iE "LIVELOCK|cleared" "$d/events.log"        # when the collapse windows fired
cat "$d/summary.json"                              # verdict + aggregates
tail -f "$d/events.log"                            # live timeline (running/waiting/KV/preempt/done)
```

## 4. Stop the server

Kill whatever is listening on the port (robust — no self-match, no error if already down):

```bash
pid=$(ss -ltnp 'sport = :8000' 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
[ -n "$pid" ] && kill "$pid" && echo "killed $pid" || echo "nothing on :8000"
```

(Do **not** `pkill -f 'vllm serve'` — that pattern also matches its own shell command line and
kills the wrong process. Match by port, or `kill <pid>` explicitly.)

---

## Output files (`results/preempt_<timestamp>/`)

| File | What it is |
|---|---|
| `kv_metrics.jsonl` | per-sample timeline: `at_seconds, kv_cache_usage_perc, num_requests_running/waiting, waiting_capacity, num_preemptions_total, client_completions` |
| `events.log` | human-readable transitions + `LIVELOCK DETECTED` markers |
| `summary.json` | end-of-run verdict + totals (preemptions, peak/avg KV, goodput, livelock seconds) |
| `requests.jsonl` | one line per request (latency, finish_reason / timeout) |
| `transcripts/` | code each **completed** run produced (empty when nothing finishes — that's the collapse) |
| `run_meta.json` | run configuration |

## Knobs

**Server (`run_server_preempt.sh`)** — env vars:

| Var | Default | Effect |
|---|---|---|
| `NUM_GPU_BLOCKS` | 10 | KV size in blocks — smaller ⇒ more oversubscription |
| `MAX_MODEL_LEN` | 8192 | per-request context (must fit in the blocks) |
| `MAX_NUM_SEQS` | 10 | max concurrent sequences (**≤ `NUM_GPU_BLOCKS`**) |
| `GPU_UTIL` | 0.5 | startup memory fraction |
| `RESERVE_ISL` | 0 | `0` = over-admit/thrash (A/B/C); `1` = admission guard ON → clean queue (Recipe D) |

**Load (`run_preempt.sh`)** — env vars:

| Var | Default | Effect |
|---|---|---|
| `WORKERS` | 8 | concurrent agents hammering the server |
| `DURATION` | 300 | run length (seconds) |
| `MAX_TOKENS` | 4096 | output tokens/request (`ignore_eos`); larger ⇒ 0 completions, smaller ⇒ turnover + more preemptions |
| `SEED_TOKENS` | 1800 | prompt (coding-file) size — **the clean-queue-vs-preempt lever**: large (~6000) ⇒ only 1 fits ⇒ clean queue |
| `TIMEOUT` | 180 | per-request client patience (s); raise to let the oldest request drain |

## Recipes

A/B/C run on the **over-admitting** server and differ by **`MAX_TOKENS`** and **`TIMEOUT`**.
**D** is the opposite — clean queueing (the parent throttle) — and needs a **different server**
(guard ON) plus a **large prompt**:

```bash
# A) "Nothing ever completes" (strongest collapse): big outputs, so no run finishes -> 0 done
WORKERS=8 DURATION=120 MAX_TOKENS=4096 TIMEOUT=180 ./run_preempt.sh

# B) Maximize visible preemptions (context thrown away repeatedly): small outputs -> fast
#    turnover -> the admit→evict cycle repeats -> num_preemptions_total climbs (goodput > 0)
WORKERS=8 DURATION=120 MAX_TOKENS=512 TIMEOUT=180 ./run_preempt.sh

# C) Prove it's a livelock, not a hang: same big outputs as (A) but give clients long patience,
#    so the OLDEST request eventually drains while the rest still starve
WORKERS=8 DURATION=120 MAX_TOKENS=4096 TIMEOUT=1200 ./run_preempt.sh   # then check requests.jsonl

# D) CLEAN QUEUEING (parent behavior): admit request 2 only if the whole thing fits, no preemption.
#    Requires RESTARTING the server with the admission guard ON, and a LARGE prompt so only 1 fits:
RESERVE_ISL=1 MAX_MODEL_LEN=8192 NUM_GPU_BLOCKS=10 MAX_NUM_SEQS=10 GPU_UTIL=0.5 ./run_server_preempt.sh
SEED_TOKENS=6000 MAX_TOKENS=1500 WORKERS=8 DURATION=120 TIMEOUT=180 ./run_preempt.sh
```

| | server | `SEED_TOKENS` | `MAX_TOKENS` | `TIMEOUT` | Expected outcome |
|---|---|---|---|---|---|
| **A** | over-admit | 1800 | 4096 | 180 | 0 completions, all client-timeout — "nothing gets done" |
| **B** | over-admit | 1800 | 512 | 180 | some completions + **preemptions climb** (visible context-throwing) |
| **C** | over-admit | 1800 | 4096 | 1200 | a few oldest requests slowly drain — confirms livelock, not deadlock |
| **D** | **guard ON** | **6000** | 1500 | 180 | **1 running / 7 waiting, 0 preemptions**, requests drain serially (measured: latencies 36/70/105/139/173 s, goodput 4/min) |

> **Can Recipe D be done with just `WORKERS/DURATION/MAX_TOKENS/TIMEOUT`?** No. `MAX_TOKENS` is
> *output* and vLLM never reserves output KV at admission, so those four knobs can't gate
> admission. Clean queueing needs a **large prompt** (`SEED_TOKENS`, so only one fits) **and** the
> **admission guard ON** (`RESERVE_ISL=1`, a server flag) — i.e. a **server restart**. On the
> over-admitting server a large prompt can still be chunk-prefill-admitted and then preempted.

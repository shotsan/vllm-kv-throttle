# Locking up vLLM scheduler

**Goal**: This document summarizes the experiment to lockup vLLM from servicing requests. 

## Goal
To make vLLM lockup, we need 
- spawn concurrent requests
- make vLLM overadmit more than it can handle
- saturate small KV cache to 100%
- Preempt low-priority request and remove all context when saturated. (Loss of Goodput/Request completed)

## Experiment Design

### Controllable Parameters

#### Concurrent Request
A simple agentic coding style request.

#### Overadmit Knob
Toggling `--no-scheduler-reserve-full-isl` will admit a new prompt if first chunk of the next prompt has token space in KV Cache. As KV Cache gets full, it will preempt/remove context of the low-priority request. When toggled off, it will check if **whole** prompt tokens fit, if not then it queues as seen in the parent's repo.
 
#### Saturate KV Cache

**Small KV Cache**: ~11k tokens (~1.43 full prompt+output token sequences) using `--num-gpu-blocks-override 10`

**Forced fixed output tokens**: Max output tokens is `max_tokens: 4096`. Total is ~1786 prompt + 4096 output token = **5900 tokens** in KV Cache. Only slightly more than 1 can fit.

## Running Experiment (STarting LLM server, probing it, shutting it down)

This bash script make sures
1. probe_patch.py is applied to see server log
2. Kills old server, starts new server, waits until ready
3. Run Experiment based on input
4. Parse ground truth using parse_probe.py
5. Stops the server

`cd /home/ways_lab/repos/vllm-kv-throttle/agentic_preemption`

| Experiment | Detail | Code to run |
|---|---|---|
| **A** | "Nothing ever completes" (strongest collapse): big outputs, so no run finishes → 0 done | `WORKERS=8 DURATION=120 MAX_TOKENS=4096 TIMEOUT=180 ./run_probe_experiment.sh` |
| **B** | Maximize visible preemptions (context thrown away repeatedly): small outputs → fast turnover → `inferred_preemptions` / probe `true_preemptions` climb (goodput > 0) | `WORKERS=8 DURATION=120 MAX_TOKENS=512 TIMEOUT=180 ./run_probe_experiment.sh` |
| **C** | Prove it's a livelock, not a hang: same big outputs as A but long client patience → the oldest request eventually drains while the rest starve (then check `requests.jsonl`) | `WORKERS=8 DURATION=120 MAX_TOKENS=4096 TIMEOUT=1200 ./run_probe_experiment.sh` |
| **D** | Clean queueing (parent behavior): admit request 2 only if the whole thing fits, no preemption. Uses the admission guard ON (`RESERVE_ISL=1`) + a large prompt so only one fits | `RESERVE_ISL=1 SEED_TOKENS=6000 MAX_TOKENS=1500 WORKERS=8 DURATION=120 TIMEOUT=180 ./run_probe_experiment.sh` |



## Observation


**Control - Experiment D**
In `repos/vllm-kv-throttle/agentic_preemption/results/preempt_20260812_093037`
We revalidated the latency/queueing behavior that was expected in Santosh's parent's branch in **Experiment D**: the scheduler does not accept next request since the current KV + Next request prompt token in full > KV Cache
- Admits 1st request
- Denies next request as not enough KV Cache, queues the rest of the request. KV Cache never saturates. 0 Preemptions
- Completes 3 of 8 in allocated time span (latency/queueing scenario)
- About 35s to complete each request serially.
```
# in repos/vllm-kv-throttle/agentic_preemption/results/preempt_20260812_093037/events.log 
   0.004s  running=0  waiting=0(cap=0)  KV=  0.00%  done=0
   0.944s  running=1  waiting=7(cap=7)  KV= 55.56%  done=0   <- ADMIT x1
  35.145s  running=0  waiting=7(cap=7)  KV=  0.00%  done=1   <- COMPLETE x1 + ARRIVE x1
  35.358s  running=1  waiting=6(cap=6)  KV= 44.44%  done=1   <- ADMIT x1
  35.790s  running=1  waiting=7(cap=7)  KV= 55.56%  done=1
```
**Stalling Behavior - Experiment A-C**
We successfully caused a livelock when changing the vLLM scheduler to opportunistic in **Experiment A-C**: the scheduler does accept next request since current KV + first chunk of next request prompt < KV Cache. 

**Experiment A** 
In `repos/vllm-kv-throttle/agentic_preemption/results/preempt_20260812_092118`
- admits 1st request
- admits 2nd request
- KV Cache quickly fills up preempts and lost block from 2nd request again and again as KV cache oscillates between 55% and 100%. See `repos/vllm-kv-throttle/agentic_preemption/results/preempt_20260812_092118/events.log`.
- Admit and PREEMPT `"ADMIT": 372, "PREEMPT": 369,` times in `repos/vllm-kv-throttle/agentic_preemption/results/preempt_20260812_092118/probe_summary.json`
- after 180s, all 8 request reach 120s timeout error, 0 goodput
```
# repos/vllm-kv-throttle/agentic_preemption/results/preempt_20260812_092118/probe_events.jsonl
{"type": "ADMIT", "t": "08-12 09:21:18", "req": "chatcmpl-a070d83c1ba11b9a-9067e4bb", "computed": 0, "kind": "new"}
{"type": "ADMIT", "t": "08-12 09:21:19", "req": "chatcmpl-9a0119c50828c5c3-8668cb74", "computed": 0, "kind": "new"}
{"type": "PREEMPT", "t": "08-12 09:21:24", "req": "chatcmpl-9a0119c50828c5c3-8668cb74", "computed": 2096}
{"type": "FREEREQ", "t": "08-12 09:21:24", "req": "chatcmpl-9a0119c50828c5c3-8668cb74", "computed": 2096, "status": "RUNNING", "npre": 0, "defer": "False"}
{"type": "POOLFREE", "t": "08-12 09:21:24", "free_before": 0}
```

**Experiment B**
In `repos/vllm-kv-throttle/agentic_preemption/results/preempt_20260812_094104`
- Only reduced max token of requests from Exp A. Expect more completions
- admits 1 request
- admits 2 request
- ...
- KV Cache quickly fills up, begins preempting while also admitting new short requests, similar to Exp A
- After 91s, first request finally finish. Now admit third request, fully saturating KV cache. Then begin preempt cycle again.
- after 180s, 3 requests completed, the rest timed out
- For smaller request, takes almost 3x as long.

```
# in events.log
  86.387s  running=2  waiting=6(cap=6)  KV=100.00%  done=0
  86.495s  running=1  waiting=7(cap=7)  KV= 55.56%  done=0   <- PREEMPT x1
  86.495s  --- livelock cleared (a run drained) ---
  86.712s  running=1  waiting=6(cap=6)  KV= 44.44%  done=1   <- COMPLETE x1 + ADMIT x1 + ARRIVE x1
  87.258s  running=2  waiting=5(cap=5)  KV=100.00%  done=1   <- ADMIT x1
  87.368s  running=2  waiting=5(cap=5)  KV=100.00%  done=1
  ...
    91.410s  running=2  waiting=6(cap=6)  KV=100.00%  done=1
  91.518s  running=2  waiting=6(cap=6)  KV=100.00%  done=1
  91.628s  running=2  waiting=6(cap=6)  KV=100.00%  done=1
  91.736s  running=2  waiting=6(cap=6)  KV=100.00%  done=1
  91.846s  running=2  waiting=6(cap=6)  KV=100.00%  done=1
  91.954s  running=2  waiting=6(cap=6)  KV=100.00%  done=1
  92.063s  running=1  waiting=7(cap=7)  KV= 55.56%  done=1   <- PREEMPT x1
  92.281s  running=2  waiting=6(cap=6)  KV=100.00%  done=1   <- ADMIT x1
  92.390s  running=2  waiting=6(cap=6)  KV=100.00%  done=1
  92.500s  running=2  waiting=6(cap=6)  KV=100.00%  done=1
```

**Experiment C**
From A, only change TIMEOUT error, to see requests eventually finish

```repos/vllm-kv-throttle/agentic_preemption/results/preempt_20260811_062626.console.log
   0.001s  running=0  waiting=0(cap=0)  KV=  0.00%  preempt=0  done=0
   0.980s  running=2  waiting=6(cap=6)  KV= 88.89%  preempt=0  done=0
   5.888s  running=1  waiting=7(cap=7)  KV= 55.56%  preempt=1  done=0
   6.146s  running=2  waiting=6(cap=6)  KV=100.00%  preempt=1  done=0
   6.404s  running=2  waiting=6(cap=6)  KV=100.00%  preempt=1  done=0
   6.663s  running=1  waiting=7(cap=7)  KV= 55.56%  preempt=1  done=0
   6.920s  running=2  waiting=6(cap=6)  KV=100.00%  preempt=1  done=0
   7.179s  running=1  waiting=7(cap=7)  KV= 55.56%  preempt=1  done=0
```
- Admits 1 request
- admits 2nd request, queues 6 other requests.
- Increases KV Cache
- Preempt 2nd request (remove KV cache), freeing some KV cache
- admits 2nd request again
- 
```repos/vllm-kv-throttle/agentic_preemption/results/preempt_20260810_190936/requests.jsonl
{"worker": 1, "marker": "Request w01-000001", "start_at": 0.001, "status": 200, "latency_s": 535.458, "finish_reason": "length", "completion_tokens": 4096, "prompt_tokens": 1929}
{"worker": 5, "marker": "Request w05-000001", "start_at": 0.003, "status": 200, "latency_s": 1033.778, "finish_reason": "length", "completion_tokens": 4096, "prompt_tokens": 1929}
{"worker": 7, "marker": "Request w07-000001", "start_at": 0.004, "status": "failed", "latency_s": 1200.207, "error": "TimeoutError('timed out')"}
```
- KV Cache quickly fills up. Preempts 2nd request (but not fully delete it), so it kicks 2nd request to show 7 requests waiting with 55.5% KV cache usage. Then services 2nd request again to get it to 100% KV Cache usage, then oscillates. 
```
...repos/vllm-kv-throttle/agentic_preemption/results/preempt_20260810_190936/kv_metrics.jsonl
{"at_seconds": 0.001, "num_requests_running": 0, "num_requests_waiting": 0, "waiting_capacity": 0, "kv_cache_usage_perc": 0.0, ***"num_preemptions_total": 351***, "client_completions": 0, "client_started": 8}
...
{"at_seconds": 6.395, "num_requests_running": 2, "num_requests_waiting": 6, "waiting_capacity": ***6***, ***"kv_cache_usage_perc": 1.0***, "num_preemptions_total": 352, "client_completions": 0, "client_started": 8}
{"at_seconds": 6.654, "num_requests_running": 1, "num_requests_waiting": 7, "waiting_capacity": ***7***, ***"kv_cache_usage_perc": 0.5555555555555556***, "num_preemptions_total": 352, "client_completions": 0, "client_started": 8}
{"at_seconds": 6.912, "num_requests_running": 2, "num_requests_waiting": 6, "waiting_capacity": 6, "kv_cache_usage_perc": 1.0, "num_preemptions_total": 352, "client_completions": 0, "client_started": 8}
{"at_seconds": 7.171, "num_requests_running": 2, "num_requests_waiting": 6, "waiting_capacity": 6, "kv_cache_usage_perc": 1.0, "num_preemptions_total": 352, "client_completions": 0, "client_started": 8}
{"at_seconds": 7.431, "num_requests_running": 1, "num_requests_waiting": 7, "waiting_capacity": 7, "kv_cache_usage_perc": 0.5555555555555556, "num_preemptions_total": 352, "client_completions": 0, "client_started": 8}
...
{"at_seconds": 119.985, "num_requests_running": 2, "num_requests_waiting": 6, "waiting_capacity": 6, "kv_cache_usage_perc": 1.0, ***"num_preemptions_total": 352***, "client_completions": 0, "client_started": 8}
- 2 Requests finish.

## Conclusion


## Things to review

- vLLM scheduling with QWEN admits based on prompt chunk size OR prompt full size. This differs from Santosh's parent's test that opportunistically admists  based on allowable context window. What is the proper approach?
--no-scheduler-reserve-full-isl why does it queue if this is removed.
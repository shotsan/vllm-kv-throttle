#!/usr/bin/env python3
"""Continuous agentic-coding load that oversubscribes the KV cache to force a
vLLM preemption livelock, while sampling scheduler/KV metrics.

Workload: a pool of `--workers` threads, each in a loop for `--duration` seconds
POSTing a single-shot `/v1/chat/completions` request whose prompt is the SAME
realistic agentic-coding task the `agentic_coding/` agents run (system prompt +
"refactor this module" instruction + the generated `agent_seed.py`), with a large
`max_tokens` and `ignore_eos:true` so every request is forced to generate a long
output. Each request is "a run"; when the scheduler preempts it, vLLM (V1,
recompute-only) frees ALL its KV blocks and re-prefills from scratch -- literally
throwing away that run's context.

Against a server started with `run_server_preempt.sh` (tiny KV, over-admission on),
several requests get admitted but cannot all grow -> continuous preemption,
KV pinned ~100%, and collapsing goodput. A watchdog flags the livelock. It is a
livelock, NOT a hard hang: vLLM guarantees the oldest request still drains.

Outputs under `--out-dir` (default results/preempt_<ts>/):
  kv_metrics.jsonl  - per-sample scheduler/KV metrics + client completion count
  events.log        - human-readable transitions + "LIVELOCK DETECTED" lines
  requests.jsonl    - one line per finished/failed request (latency, finish_reason)
  transcripts/      - the actual code each completed run produced
  summary.json      - end-of-run verdict + aggregate numbers
  run_meta.json     - the run configuration
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread

HERE = Path(__file__).resolve().parent
# Realistic agentic-coding prompt, now local to this folder (was ../agentic_coding).
sys.path.insert(0, str(HERE))
from prompts import SYSTEM_PROMPT, build_first_user  # noqa: E402

METRICS_PATH = "/metrics"

# ---------------------------------------------------------------- metrics ----
# NOTE: vLLM's `num_preemptions_total` is deliberately NOT scraped or recorded --
# on this build it undercounts real preemptions ~200x (it counts distinct requests
# preempted, not evictions), so it is misleading. We record the client-inferred
# preemption/admission counts instead; use the probe mode for exact ground truth.
GAUGES = ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc")


def scrape(metrics_url: str) -> dict:
    """One /metrics scrape -> the gauges we care about, plus the labeled
    waiting-by-reason 'capacity' gauge and the summed request_success_total."""
    text = urllib.request.urlopen(metrics_url, timeout=5).read().decode()
    out = {}
    success_total = 0.0
    waiting_capacity = 0.0
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for key in GAUGES:
            if line.startswith(f"vllm:{key}{{"):
                out[key] = float(line.rsplit(" ", 1)[-1])
        if line.startswith("vllm:request_success_total{"):
            success_total += float(line.rsplit(" ", 1)[-1])
        if line.startswith("vllm:num_requests_waiting_by_reason{") and 'reason="capacity"' in line:
            waiting_capacity = float(line.rsplit(" ", 1)[-1])
    out["waiting_capacity"] = waiting_capacity
    out["request_success_total"] = success_total
    return out


# ---------------------------------------------------------------- shared -----
class State:
    def __init__(self):
        self.lock = Lock()
        self.started = 0
        self.completed = 0
        self.failed = 0
        self.records = []  # per-request dicts


def worker(wid: int, args, task, stop: Event, deadline: float, state: State, tdir: Path):
    seed = task["seed"]
    instruction = task["instruction"]
    counter = 0
    while not stop.is_set() and time.monotonic() < deadline:
        counter += 1
        marker = f"Request w{wid:02d}-{counter:06d}."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_first_user(marker, instruction, seed)},
        ]
        payload = json.dumps({
            "model": args.model, "messages": messages,
            "max_tokens": args.max_tokens, "ignore_eos": True, "temperature": 0,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if args.api_key and args.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {args.api_key}"
        req = urllib.request.Request(args.url, data=payload, headers=headers, method="POST")
        with state.lock:
            state.started += 1
        t0 = time.monotonic()
        rec = {"worker": wid, "marker": marker.strip("."), "start_at": round(t0 - args.t_origin, 3)}
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                body = json.load(resp)
            dt = round(time.monotonic() - t0, 3)
            msg = body["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            usage = body.get("usage", {})
            finish = body["choices"][0].get("finish_reason")
            rec.update({"status": 200, "latency_s": dt, "finish_reason": finish,
                        "completion_tokens": usage.get("completion_tokens"),
                        "prompt_tokens": usage.get("prompt_tokens")})
            with state.lock:
                state.completed += 1
                state.records.append(rec)
            # Save the actual produced code for this run.
            (tdir / f"{rec['marker']}.json").write_text(json.dumps({
                **rec, "reasoning_content": reasoning, "content": content,
            }, indent=2, ensure_ascii=False), encoding="utf-8")
        except urllib.error.HTTPError as e:
            dt = round(time.monotonic() - t0, 3)
            rec.update({"status": e.code, "latency_s": dt,
                        "error": e.read().decode(errors="replace")[:300]})
            with state.lock:
                state.failed += 1
                state.records.append(rec)
        except Exception as e:
            dt = round(time.monotonic() - t0, 3)
            rec.update({"status": "failed", "latency_s": dt, "error": repr(e)[:300]})
            with state.lock:
                state.failed += 1
                state.records.append(rec)


def sampler(args, stop: Event, state: State, out_dir: Path, summary: dict):
    kv_log = (out_dir / "kv_metrics.jsonl").open("w", encoding="utf-8")
    ev_log = (out_dir / "events.log").open("w", encoding="utf-8")

    def event(line: str):
        print(line, flush=True)
        ev_log.write(line + "\n"); ev_log.flush()

    event(f"# preemption run  workers={args.workers}  duration={args.duration}s  "
          f"max_tokens={args.max_tokens}  model={args.model}")
    win = max(1, int(args.window / max(args.sample_interval, 0.01)))
    hist = deque(maxlen=win + 1)  # (t, completed, preemptions)
    last_key = None
    peak_kv = 0.0
    kv_sum = 0.0
    kv_n = 0
    in_livelock = False
    livelock_seconds = 0.0
    livelock_started_at = None
    last_t = None
    prev = None  # (running, waiting, completed, started) for cause inference
    infer = {"PREEMPT": 0, "ADMIT": 0, "COMPLETE": 0, "ARRIVE": 0}
    while not stop.is_set():
        t = round(time.monotonic() - args.t_origin, 3)
        try:
            m = scrape(args.metrics_url)
        except Exception:
            time.sleep(args.sample_interval); continue
        with state.lock:
            completed = state.completed
            started = state.started
        running = int(m.get("num_requests_running", 0))
        waiting = int(m.get("num_requests_waiting", 0))
        cap = int(m.get("waiting_capacity", 0))
        kv = m.get("kv_cache_usage_perc", 0.0)
        peak_kv = max(peak_kv, kv); kv_sum += kv; kv_n += 1

        # Infer the CAUSE of each transition from aggregate deltas. A request
        # leaves `running` only by preemption or completion, and enters only by
        # admission; new client requests add to `waiting`. So:
        #   residual = Δrunning + completions = admissions - preemptions
        # This labels every observed transition (PREEMPT/ADMIT/COMPLETE/ARRIVE),
        # but because it's polled it can miss a preempt+admit pair that nets to
        # zero between samples -> it UNDERCOUNTS like the metric (use the
        # ground-truth probe mode for exact per-eviction counts). The direction is
        # corroborated by KV (preempt frees blocks -> KV drops; admit -> KV rises).
        cause = "steady"
        if prev is not None:
            d_run = running - prev[0]
            c = max(0, completed - prev[2])
            arrivals = max(0, started - prev[3])
            residual = d_run + c            # admissions - preemptions
            adm = residual if residual > 0 else 0
            pre = -residual if residual < 0 else 0
            parts = []
            if c:        parts.append(f"COMPLETE x{c}");  infer["COMPLETE"] += c
            if pre:      parts.append(f"PREEMPT x{pre}");  infer["PREEMPT"] += pre
            if adm:      parts.append(f"ADMIT x{adm}");    infer["ADMIT"] += adm
            if arrivals: parts.append(f"ARRIVE x{arrivals}"); infer["ARRIVE"] += arrivals
            if parts:    cause = " + ".join(parts)
        prev = (running, waiting, completed, started)

        kv_log.write(json.dumps({
            "at_seconds": t, "num_requests_running": running,
            "num_requests_waiting": waiting, "waiting_capacity": cap,
            "kv_cache_usage_perc": kv,
            "client_completions": completed, "client_started": started,
            "cause": cause,
            "inferred_preemptions": infer["PREEMPT"],   # cumulative, client-inferred
            "inferred_admissions": infer["ADMIT"],
        }) + "\n"); kv_log.flush()

        # rolling-window completion delta for the watchdog
        hist.append((t, completed))
        compl_win = None
        if len(hist) >= 2:
            compl_win = completed - hist[0][1]

        key = (running, waiting)
        if key != last_key or kv >= 0.999:
            tag = "" if cause == "steady" else f"   <- {cause}"
            event(f"{t:8.3f}s  running={running}  waiting={waiting}(cap={cap})  "
                  f"KV={kv*100:6.2f}%  done={completed}{tag}")
            last_key = key

        # livelock / goodput-collapse: KV saturated, work in flight, but ~zero
        # completions across the whole window (preemptions are a secondary signal;
        # once the admitted set already fills KV the scheduler may simply queue the
        # rest, so we do NOT require the preempt counter to keep rising).
        if compl_win is not None:
            work_in_flight = (running + waiting) > 0
            livelock_now = (kv >= args.kv_livelock and compl_win == 0 and work_in_flight
                            and len(hist) == hist.maxlen)
            if last_t is not None and in_livelock:
                livelock_seconds += (t - last_t)
            if livelock_now and not in_livelock:
                in_livelock = True; livelock_started_at = t
                event(f"{t:8.3f}s  *** LIVELOCK DETECTED *** KV={kv*100:.1f}%  "
                      f"running={running} waiting={waiting}  inferred_preemptions={infer['PREEMPT']}  "
                      f"completions/{args.window:.0f}s={compl_win} "
                      f"-> KV is saturated and no run has finished for {args.window:.0f}s: "
                      f"admitted runs' context is thrown away / re-queued faster than any drains.")
            elif not livelock_now and in_livelock:
                in_livelock = False
                event(f"{t:8.3f}s  --- livelock cleared (a run drained) ---")
        last_t = t
        time.sleep(args.sample_interval)

    kv_log.close()
    if in_livelock and livelock_started_at is not None:
        pass
    summary["peak_kv"] = round(peak_kv, 4)
    summary["avg_kv"] = round(kv_sum / kv_n, 4) if kv_n else 0.0
    summary["livelock_seconds"] = round(livelock_seconds, 1)
    # Client-inferred event tallies (undercount vs ground-truth probe mode).
    summary["inferred_preemptions"] = infer["PREEMPT"]
    summary["inferred_admissions"] = infer["ADMIT"]
    summary["inferred_completions"] = infer["COMPLETE"]
    ev_log.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Oversubscribe KV -> preemption livelock demonstrator")
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    p.add_argument("--task-dir", type=Path, default=HERE / "generated")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--duration", type=float, default=300.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--sample-interval", type=float, default=0.1,
                   help="metrics poll interval (s); finer catches more transitions")
    p.add_argument("--window", type=float, default=10.0, help="watchdog sliding window (s)")
    p.add_argument("--kv-livelock", type=float, default=0.95, help="KV frac to call it saturated")
    p.add_argument("--timeout", type=float, default=180,
                   help="per-request HTTP timeout; also bounds post-deadline drain")
    p.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    meta = json.loads((args.task_dir / "agent_task.json").read_text(encoding="utf-8"))
    seed = (args.task_dir / meta["seed_file"]).read_text(encoding="utf-8")
    task = {"seed": seed, "instruction": meta["task_instruction"]}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (HERE / "results" / f"preempt_{stamp}")
    (out_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta.json").write_text(json.dumps({
        "timestamp": stamp, "model": args.model, "url": args.url,
        "workers": args.workers, "duration": args.duration, "max_tokens": args.max_tokens,
        "seed_tokens_actual": meta.get("seed_tokens_actual"),
    }, indent=2), encoding="utf-8")

    args.t_origin = time.monotonic()
    deadline = args.t_origin + args.duration
    state = State()
    stop = Event()
    summary = {}

    samp = Thread(target=sampler, args=(args, stop, state, out_dir, summary), daemon=True)
    samp.start()
    workers = [Thread(target=worker, args=(i, args, task, stop, deadline, state,
                                           out_dir / "transcripts"), daemon=True)
               for i in range(args.workers)]
    for w in workers:
        w.start()

    # let the fixed-duration run proceed; workers stop themselves at the deadline
    try:
        while time.monotonic() < deadline:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    stop.set()
    for w in workers:
        w.join(timeout=args.timeout + 5)
    time.sleep(args.sample_interval * 3)
    stop.set()
    samp.join(timeout=5)

    with state.lock:
        records = list(state.records)
        started, completed, failed = state.started, state.completed, state.failed
    (out_dir / "requests.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    lat = sorted(r["latency_s"] for r in records if r.get("status") == 200)
    def pct(v, q):
        return round(v[min(len(v) - 1, int(q * len(v)))], 2) if v else None
    summary.update({
        "duration_s": args.duration, "workers": args.workers,
        "requests_started": started, "requests_completed": completed,
        "requests_failed": failed,
        "goodput_completions_per_min": round(completed / (args.duration / 60), 2),
        "mean_latency_s": round(sum(lat) / len(lat), 2) if lat else None,
        "p95_latency_s": pct(lat, 0.95),
        "finish_reasons": _count(records),
    })
    verdict = ("LIVELOCK / goodput-collapse observed"
               if summary.get("livelock_seconds", 0) > 0 else
               "no livelock detected (try more workers / fewer blocks)")
    summary["verdict"] = verdict
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_dir}")


def _count(records):
    out = {}
    for r in records:
        k = str(r.get("finish_reason") or r.get("status"))
        out[k] = out.get(k, 0) + 1
    return out


if __name__ == "__main__":
    main()

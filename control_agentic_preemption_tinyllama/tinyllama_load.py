#!/usr/bin/env python3
"""Parent-framework load driver for the TinyLlama control experiment.

Same shape as the parent repo's load_test.py / proof_test.py: N barrier-synced
workers fire ONE wave of identical LooGLE /v1/completions requests
(temperature=0, ignore_eos=True, per-request "Request NNNNNN." marker) -- but
pointed at the admission GATEWAY, while a sampler thread scrapes the REAL
server's /metrics for scheduler ground truth (running/waiting/KV/preemptions).

Outputs (into --out-dir, created):
  events.log       transition rows like the parent's proof_test + livelock marks
  kv_metrics.jsonl every raw metric sample that changed
  requests.jsonl   one line per request result (status, seconds, usage)
  summary.json     aggregates + verdict
"""
import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

KEYS = ("num_requests_running", "num_requests_waiting",
        "kv_cache_usage_perc", "num_preemptions_total")


def scrape(metrics_url: str) -> dict:
    text = urllib.request.urlopen(metrics_url, timeout=5).read().decode()
    out = {}
    for key in KEYS:
        for line in text.splitlines():
            if line.startswith(f"vllm:{key}{{"):
                out[key] = float(line.rsplit(" ", 1)[-1])
                break
    return out


def send(index: int, barrier: Barrier, args, prompt_template: str, state) -> dict:
    prompt = prompt_template.replace("Request 000000.", f"Request {index:06d}.", 1)
    payload = json.dumps({
        "model": args.model, "prompt": prompt, "max_tokens": args.max_tokens,
        "temperature": 0, "ignore_eos": True,
    }).encode()
    request = urllib.request.Request(
        args.url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    barrier.wait()
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = json.load(response)
        rec = {"request": index, "status": response.status,
               "seconds": round(time.monotonic() - started, 3),
               "finish_reason": (body.get("choices") or [{}])[0].get("finish_reason"),
               "usage": body.get("usage", {})}
    except urllib.error.HTTPError as error:
        rec = {"request": index, "status": error.code,
               "seconds": round(time.monotonic() - started, 3),
               "error": error.read().decode(errors="replace")[:300]}
    except Exception as error:  # noqa: BLE001
        rec = {"request": index, "status": "failed",
               "seconds": round(time.monotonic() - started, 3), "error": repr(error)}
    with state["lock"]:
        if rec["status"] == 200:
            state["done"] += 1
        state["records"].append(rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description="one synchronized wave through the gateway + metrics sampling")
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1/completions")
    ap.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--prompt", type=Path, default=Path("generated/prompt.txt"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    prompt = a.prompt.read_text(encoding="utf-8")
    state = {"lock": threading.Lock(), "done": 0, "records": []}
    barrier = Barrier(a.concurrency)
    stop = threading.Event()
    t0 = time.monotonic()

    ev_fh = (a.out_dir / "events.log").open("w", encoding="utf-8")
    kv_fh = (a.out_dir / "kv_metrics.jsonl").open("w", encoding="utf-8")
    stats = {"peak_kv": 0.0, "kv_sum": 0.0, "kv_n": 0, "livelock_s": 0.0,
             "admits": 0, "preempts": 0, "completes": 0, "livelocked": False}

    def ev(line: str) -> None:
        s = f"{time.monotonic() - t0:8.3f}s  {line}"
        print(s, flush=True)
        ev_fh.write(s + "\n"); ev_fh.flush()

    ev_fh.write(f"# tinyllama wave  concurrency={a.concurrency} max_tokens={a.max_tokens} "
                f"timeout={a.timeout} model={a.model}\n")

    def sampler() -> None:
        last = None
        prev = None
        prev_done = 0
        last_progress = time.monotonic()
        last_t = time.monotonic()
        while not stop.is_set():
            now = time.monotonic()
            try:
                m = scrape(a.metrics_url)
            except Exception:
                time.sleep(0.2); continue
            if not m:
                time.sleep(0.2); continue
            kv = m.get("kv_cache_usage_perc", 0.0)
            stats["peak_kv"] = max(stats["peak_kv"], kv)
            stats["kv_sum"] += kv; stats["kv_n"] += 1
            with state["lock"]:
                done = state["done"]
            snap = (m.get("num_requests_running"), m.get("num_requests_waiting"),
                    round(kv, 4), m.get("num_preemptions_total"), done)
            if snap != last:
                kv_fh.write(json.dumps({"t": round(now - t0, 3), **m, "done": done}) + "\n")
                kv_fh.flush()
                running = int(m.get("num_requests_running", 0))
                waiting = int(m.get("num_requests_waiting", 0))
                notes = []
                if prev is not None:
                    dr = running - int(prev.get("num_requests_running", 0))
                    dp = int(m.get("num_preemptions_total", 0)) - int(prev.get("num_preemptions_total", 0))
                    dc = done - prev_done
                    if dr > 0:
                        notes.append(f"<- ADMIT x{dr}"); stats["admits"] += dr
                    if dp > 0:
                        notes.append(f"<- PREEMPT x{dp} (metric)")
                    if dc > 0:
                        notes.append(f"<- COMPLETE x{dc}"); stats["completes"] += dc
                    if dr < 0 and dc == 0:
                        # running fell with no completion -> preemption (the metric undercounts)
                        stats["preempts"] += -dr
                        if dp <= 0:
                            notes.append(f"<- PREEMPT x{-dr} (inferred)")
                ev(f"running={running}  waiting={waiting}  KV={kv * 100:6.2f}%  done={done}"
                   + ("   " + " ".join(notes) if notes else ""))
                prev = m; prev_done = done; last = snap
            # livelock: KV saturated with work outstanding and nothing completing
            if done > prev_done:
                last_progress = now
            outstanding = a.concurrency - done
            if kv >= 0.99 and outstanding > 0 and now - last_progress >= 10.0:
                if not stats["livelocked"]:
                    ev(f"*** LIVELOCK DETECTED ***  KV={kv*100:.1f}%  no completion for 10s "
                       f"with {outstanding} outstanding")
                    stats["livelocked"] = True
                stats["livelock_s"] += now - last_t
            elif stats["livelocked"] and (kv < 0.99 or done > prev_done):
                ev("--- livelock cleared ---")
                stats["livelocked"] = False
            last_t = now
            time.sleep(0.05)

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futures = [pool.submit(send, i, barrier, a, prompt, state) for i in range(a.concurrency)]
        results = [f.result() for f in futures]
    wall = time.monotonic() - t0
    time.sleep(0.5)
    stop.set()
    sampler_thread.join(timeout=3)

    with (a.out_dir / "requests.jsonl").open("w", encoding="utf-8") as fh:
        for rec in sorted(results, key=lambda r: r["request"]):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    ok = [r for r in results if r["status"] == 200]
    lat = [r["seconds"] for r in ok]
    finish = {}
    for r in results:
        key = r.get("finish_reason") or str(r["status"])
        finish[key] = finish.get(key, 0) + 1
    summary = {
        "wall_seconds": round(wall, 3),
        "concurrency": a.concurrency,
        "max_tokens": a.max_tokens,
        "client_timeout_s": a.timeout,
        "requests_completed": len(ok),
        "requests_failed": len(results) - len(ok),
        "goodput_completions_per_min": round(len(ok) / wall * 60, 2) if wall else 0,
        "mean_latency_s": round(statistics.mean(lat), 3) if lat else None,
        "p95_latency_s": round(sorted(lat)[max(0, int(len(lat) * 0.95) - 1)], 3) if lat else None,
        "latencies_s": sorted(round(v, 3) for v in lat),
        "peak_kv": round(stats["peak_kv"], 4),
        "avg_kv": round(stats["kv_sum"] / stats["kv_n"], 4) if stats["kv_n"] else None,
        "livelock_seconds": round(stats["livelock_s"], 1),
        "inferred_admissions": stats["admits"],
        "inferred_preemptions": stats["preempts"],
        "inferred_completions": stats["completes"],
        "finish_reasons": finish,
        "verdict": ("PREEMPTION THRASH / high latency" if stats["preempts"] > 0 or stats["livelock_s"] > 0
                    else "clean queueing (no preemption observed)"),
    }
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ev_fh.close(); kv_fh.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

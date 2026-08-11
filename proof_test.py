#!/usr/bin/env python3
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

from load_test import send

URL = "http://127.0.0.1:8000/v1/completions"
KEYS = ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc", "num_preemptions_total")


def metrics() -> dict:
    text = urllib.request.urlopen("http://127.0.0.1:8000/metrics").read().decode()
    return {
        key: float(next(line.rsplit(" ", 1)[-1] for line in text.splitlines()
                        if line.startswith(f"vllm:{key}{{")))
        for key in KEYS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample vLLM scheduler metrics under concurrent load")
    parser.add_argument("--json", action="store_true",
                        help="stream raw JSON on every metric change (default: sanitized transition rows)")
    cli = parser.parse_args()

    concurrency = 4
    args = SimpleNamespace(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", url=URL,
                           max_tokens=282, timeout=600)
    prompt = Path("generated/prompt.txt").read_text(encoding="utf-8")
    barrier = Barrier(concurrency)
    started = time.monotonic()
    last = None
    last_key = None
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send, i, barrier, args, prompt) for i in range(concurrency)]
        while not all(future.done() for future in futures):
            current = metrics()
            state = tuple(current.values())
            if state != last:
                at = round(time.monotonic() - started, 3)
                if cli.json:
                    print(json.dumps({"at_seconds": at, **current}), flush=True)
                else:
                    running = int(current["num_requests_running"])
                    waiting = int(current["num_requests_waiting"])
                    kv = current["kv_cache_usage_perc"]
                    key = (running, waiting)
                    # only the rows that matter: an admission/release, or full saturation
                    if key != last_key or kv == 1.0:
                        print(f"{at:.3f}s  running={running}  waiting={waiting}  KV={kv * 100:.2f}%", flush=True)
                        last_key = key
                last = state
            time.sleep(0.05)
    print(json.dumps({"responses": sorted((future.result() for future in futures),
                                          key=lambda result: result["seconds"])}, indent=2))


if __name__ == "__main__":
    main()

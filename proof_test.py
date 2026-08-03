#!/usr/bin/env python3
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
    concurrency = 4
    args = SimpleNamespace(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", url=URL,
                           max_tokens=282, timeout=600)
    prompt = Path("generated/prompt.txt").read_text(encoding="utf-8")
    barrier = Barrier(concurrency)
    started = time.monotonic()
    last = None
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send, i, barrier, args, prompt) for i in range(concurrency)]
        while not all(future.done() for future in futures):
            current = metrics()
            state = tuple(current.values())
            if state != last:
                print(json.dumps({"at_seconds": round(time.monotonic() - started, 3), **current}), flush=True)
                last = state
            time.sleep(0.05)
    print(json.dumps({"responses": sorted((future.result() for future in futures),
                                          key=lambda result: result["seconds"])}, indent=2))


if __name__ == "__main__":
    main()

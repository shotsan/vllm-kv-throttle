#!/usr/bin/env python3
import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Barrier


def send(index: int, barrier: Barrier, args, prompt_template: str) -> dict:
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
        return {"request": index, "status": response.status,
                "seconds": round(time.monotonic() - started, 3), "usage": body.get("usage", {})}
    except urllib.error.HTTPError as error:
        return {"request": index, "status": error.code,
                "seconds": round(time.monotonic() - started, 3),
                "error": error.read().decode(errors="replace")}
    except Exception as error:
        return {"request": index, "status": "failed",
                "seconds": round(time.monotonic() - started, 3), "error": repr(error)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Saturate vLLM KV cache with concurrent prompts")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--prompt", type=Path, default=Path("generated/prompt.txt"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=282)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    prompt = args.prompt.read_text(encoding="utf-8")
    barrier = Barrier(args.concurrency)
    wall_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(send, i, barrier, args, prompt) for i in range(args.concurrency)]
        for future in as_completed(futures):
            print(json.dumps(future.result(), ensure_ascii=False), flush=True)
    print(json.dumps({"total_wall_seconds": round(time.monotonic() - wall_start, 3)}))


if __name__ == "__main__":
    main()

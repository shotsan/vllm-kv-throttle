#!/usr/bin/env python3
"""Side-channel request controller (admission gateway) in front of vLLM.

WHY
---
In `../agentic_preemption` (Recipes A-C) an OVER-ADMITTING vLLM server
(`--no-scheduler-reserve-full-isl`) livelocks: several agentic-coding requests are
admitted on their first prompt chunk, their combined KV blows past a tiny cache,
and the scheduler PREEMPTS running sequences forever -> KV pinned ~100%, goodput
collapses. Recipe D avoided this by changing the SERVER (full-ISL admission guard).

This gateway shows the SAME clean queueing behavior can be recovered WITHOUT
touching the server's admission mode -- by an EXTERNAL controller that observes the
KV cache and delays requests. The server stays in its over-admit (thrashing) config;
the gateway is what keeps it from over-subscribing.

POLICY (starter, `--policy kv-gate`)
------------------------------------
Each request's prompt is sized (by the load driver) to ~56% of usable KV. The
gateway continuously scrapes vLLM `/metrics` for `kv_cache_usage_perc` and admits a
request to the upstream ONLY when observed KV <= `--threshold` (default 0.50). Once
one request is admitted it drives KV to ~56% > 50%, so every other request is HELD
at the gateway (never reaches vLLM) until that one drains and KV falls back under
the threshold. Result: at most one over-56% request in flight -> no oversubscription
-> no preemption -> clean serial queueing, enforced from OUTSIDE the server.

Admission is serialized by a single gate lock so contenders cannot all slip through
at KV~0 simultaneously (thundering herd): after a request is let through, the gate
waits until KV actually rises above the threshold (the admit "landed" on the server)
-- or `--settle-timeout` elapses -- before evaluating the next contender. The
upstream call runs on a helper thread so this settle can overlap generation.

`--policy bypass` is a plain pass-through (no throttling) -- use it to reproduce the
livelock THROUGH the gateway and prove the gateway itself isn't what fixes things.

Outputs (under `--log-dir`):
  gateway_events.log     - human-readable ADMIT/HOLD/SCRAPE-FAIL transitions
  gateway_requests.jsonl - one line per proxied request: wait_s, kv_at_admit, status, latency_s
  gateway_meta.json      - the gateway configuration
Run standalone via run_gateway.sh, or end-to-end via run_control_experiment.sh.
"""
import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Paths we actively throttle (generation requests). Everything else (e.g.
# /v1/models health checks, /metrics) is proxied straight through.
THROTTLED_PATHS = ("/v1/chat/completions", "/v1/completions")


# ------------------------------------------------------------------ monitor ---
class KvMonitor(threading.Thread):
    """Background poller: keeps the latest observed KV-cache utilization fraction
    from the upstream vLLM `/metrics`, so admission decisions read a cached value
    instead of blocking on a scrape each time."""

    def __init__(self, metrics_url: str, poll: float, stop: threading.Event):
        super().__init__(daemon=True)
        self.metrics_url = metrics_url
        self.poll = poll
        self.stop = stop
        self._kv = None            # last good KV fraction, or None if never scraped
        self._ok = False
        self._lock = threading.Lock()

    def scrape_once(self):
        text = urllib.request.urlopen(self.metrics_url, timeout=5).read().decode()
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if line.startswith("vllm:kv_cache_usage_perc{"):
                return float(line.rsplit(" ", 1)[-1])
        return None

    def kv(self):
        with self._lock:
            return self._kv

    def run(self):
        while not self.stop.is_set():
            try:
                v = self.scrape_once()
                with self._lock:
                    self._kv = v if v is not None else self._kv
                    self._ok = True
            except Exception:
                with self._lock:
                    self._ok = False
            self.stop.wait(self.poll)


# --------------------------------------------------------------- controller ---
class Controller:
    """The admission policy. Serializes gate decisions and blocks requests while
    observed KV exceeds the threshold."""

    def __init__(self, monitor: KvMonitor, args, log, stop: threading.Event):
        self.mon = monitor
        self.threshold = args.threshold
        self.poll = args.poll
        self.settle_timeout = args.settle_timeout
        self.fail_open_after = args.fail_open_after
        self.policy = args.policy
        self.log = log
        self.stop = stop
        self._gate = threading.Lock()   # one admission decision at a time
        self._held = 0                  # requests currently waiting at the gate
        self._held_lock = threading.Lock()

    def _observe(self):
        """Best-effort current KV; None if metrics never seen yet."""
        return self.mon.kv()

    def wait_and_enter(self, req_id: str) -> dict:
        """Block until this request may proceed. Returns an admission ticket
        {wait_s, kv_at_admit, fail_open}. On return the gate lock is HELD and must
        be released via settle_and_release()."""
        if self.policy == "bypass":
            return {"wait_s": 0.0, "kv_at_admit": self._observe(), "fail_open": False, "gated": False}

        with self._held_lock:
            self._held += 1
            held_now = self._held
        self.log(f"ARRIVE req={req_id} waiting_at_gate={held_now} kv={_fmt(self._observe())}")

        self._gate.acquire()            # serialize admission; released in settle_and_release
        t0 = time.monotonic()
        no_metrics_since = None
        fail_open = False
        while not self.stop.is_set():
            kv = self._observe()
            if kv is None:
                # metrics not available yet -> after a grace, fail OPEN so the
                # gateway never deadlocks the whole load on a scrape outage.
                now = time.monotonic()
                no_metrics_since = no_metrics_since or now
                if now - no_metrics_since >= self.fail_open_after:
                    fail_open = True
                    self.log(f"SCRAPE-FAIL req={req_id} no KV for "
                             f"{self.fail_open_after:.0f}s -> FAIL-OPEN admit")
                    break
            else:
                no_metrics_since = None
                if kv <= self.threshold:
                    break
            time.sleep(self.poll)
        wait_s = round(time.monotonic() - t0, 3)
        kv_at = self._observe()
        with self._held_lock:
            self._held -= 1
        self.log(f"ADMIT  req={req_id} wait={wait_s}s kv={_fmt(kv_at)} "
                 f"(<= thr {self.threshold:.2f}){' FAIL-OPEN' if fail_open else ''}")
        return {"wait_s": wait_s, "kv_at_admit": kv_at, "fail_open": fail_open, "gated": True}

    def settle_and_release(self, ticket: dict, req_id: str) -> None:
        """After the upstream call has been kicked off, wait until KV rises above
        the threshold (the admit landed) or settle-timeout elapses, then release
        the gate so the next contender is evaluated against real, post-admit KV."""
        if not ticket.get("gated"):
            return
        deadline = time.monotonic() + self.settle_timeout
        while not self.stop.is_set() and time.monotonic() < deadline:
            kv = self._observe()
            if kv is not None and kv > self.threshold:
                break
            time.sleep(self.poll)
        try:
            self._gate.release()
        except RuntimeError:
            pass


# ------------------------------------------------------------------ handler ---
def make_handler(controller: Controller, args, state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # silence default stderr access log
            pass

        # -- helpers ------------------------------------------------------------
        def _read_body(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            return self.rfile.read(n) if n else b""

        def _fwd_headers(self):
            h = {}
            for k, v in self.headers.items():
                if k.lower() in ("host", "content-length", "connection", "accept-encoding"):
                    continue
                h[k] = v
            return h

        def _proxy(self, body: bytes):
            """Forward to upstream; return (status, headers_list, body_bytes)."""
            url = args.upstream.rstrip("/") + self.path
            req = urllib.request.Request(url, data=body if body else None,
                                         headers=self._fwd_headers(), method=self.command)
            try:
                with urllib.request.urlopen(req, timeout=args.upstream_timeout) as r:
                    return r.status, list(r.headers.items()), r.read()
            except urllib.error.HTTPError as e:
                return e.code, list(e.headers.items()), e.read()

        def _send(self, status, headers, body):
            self.send_response(status)
            sent_len = False
            for k, v in headers:
                if k.lower() in ("transfer-encoding", "connection", "content-length"):
                    continue
                self.send_header(k, v)
                if k.lower() == "content-length":
                    sent_len = True
            if not sent_len:
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        # -- verbs --------------------------------------------------------------
        def do_GET(self):
            status, headers, body = self._proxy(b"")
            self._send(status, headers, body)

        def do_POST(self):
            body = self._read_body()
            throttled = any(self.path.startswith(p) for p in THROTTLED_PATHS)
            if not throttled:
                status, headers, rbody = self._proxy(body)
                self._send(status, headers, rbody)
                return

            seq = state.next_id()
            req_id = f"g{seq:05d}"
            ticket = controller.wait_and_enter(req_id)

            # Kick the upstream call on a helper thread so the gate's settle wait
            # can overlap generation instead of blocking on the full response.
            result = {}
            done = threading.Event()

            def _do():
                try:
                    result["resp"] = self._proxy(body)
                except Exception as e:  # noqa: BLE001
                    result["err"] = repr(e)
                finally:
                    done.set()

            t0 = time.monotonic()
            worker = threading.Thread(target=_do, daemon=True)
            worker.start()
            controller.settle_and_release(ticket, req_id)
            done.wait()
            latency = round(time.monotonic() - t0, 3)

            if "err" in result:
                controller.log(f"UPSTREAM-ERR req={req_id} {result['err'][:160]}")
                self._send(502, [("Content-Type", "application/json")],
                           json.dumps({"error": result["err"]}).encode())
                status = 502
            else:
                status, headers, rbody = result["resp"]
                self._send(status, headers, rbody)

            state.record({
                "req": req_id, "path": self.path,
                "wait_s": ticket["wait_s"], "kv_at_admit": ticket["kv_at_admit"],
                "fail_open": ticket["fail_open"], "upstream_status": status,
                "latency_s": latency,
                "t": round(time.monotonic() - state.t_origin, 3),
            })

    return Handler


# -------------------------------------------------------------------- state ---
class State:
    def __init__(self, log_dir: Path, t_origin: float):
        self._lock = threading.Lock()
        self._id = 0
        self.records = []
        self.t_origin = t_origin
        self._rec_fh = (log_dir / "gateway_requests.jsonl").open("w", encoding="utf-8")

    def next_id(self):
        with self._lock:
            self._id += 1
            return self._id

    def record(self, rec):
        with self._lock:
            self.records.append(rec)
            self._rec_fh.write(json.dumps(rec) + "\n")
            self._rec_fh.flush()

    def close(self):
        self._rec_fh.close()


def _fmt(kv):
    return "n/a" if kv is None else f"{kv*100:.2f}%"


def main():
    p = argparse.ArgumentParser(description="Side-channel KV-observing admission gateway for vLLM")
    p.add_argument("--listen-host", default="0.0.0.0")
    p.add_argument("--listen-port", type=int, default=8001)
    p.add_argument("--upstream", default="http://127.0.0.1:8000",
                   help="the real vLLM base URL")
    p.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    p.add_argument("--policy", choices=["kv-gate", "bypass"], default="kv-gate")
    p.add_argument("--threshold", type=float, default=0.50,
                   help="admit only when observed KV frac <= this (kv-gate)")
    p.add_argument("--poll", type=float, default=0.1, help="KV scrape / gate poll interval (s)")
    p.add_argument("--settle-timeout", type=float, default=8.0,
                   help="max wait for an admit to register in KV before releasing the gate")
    p.add_argument("--fail-open-after", type=float, default=10.0,
                   help="if /metrics is unreachable this long, admit anyway (avoid deadlock)")
    p.add_argument("--upstream-timeout", type=float, default=1200.0)
    p.add_argument("--log-dir", type=Path, default=HERE / "gateway_logs")
    args = p.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    ev_fh = (args.log_dir / "gateway_events.log").open("w", encoding="utf-8")
    ev_lock = threading.Lock()
    t_origin = time.monotonic()

    def log(line: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        rel = time.monotonic() - t_origin
        s = f"{rel:8.3f}s [{stamp}] {line}"
        with ev_lock:
            print(s, flush=True)
            ev_fh.write(s + "\n"); ev_fh.flush()

    (args.log_dir / "gateway_meta.json").write_text(json.dumps({
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "listen": f"{args.listen_host}:{args.listen_port}",
        "upstream": args.upstream, "metrics_url": args.metrics_url,
        "policy": args.policy, "threshold": args.threshold, "poll": args.poll,
        "settle_timeout": args.settle_timeout,
    }, indent=2), encoding="utf-8")

    stop = threading.Event()
    monitor = KvMonitor(args.metrics_url, args.poll, stop)
    monitor.start()
    controller = Controller(monitor, args, log, stop)
    state = State(args.log_dir, t_origin)

    handler = make_handler(controller, args, state)
    httpd = ThreadingHTTPServer((args.listen_host, args.listen_port), handler)
    log(f"gateway up on {args.listen_host}:{args.listen_port} -> {args.upstream}  "
        f"policy={args.policy} threshold={args.threshold:.2f}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        state.close()
        ev_fh.close()


if __name__ == "__main__":
    main()

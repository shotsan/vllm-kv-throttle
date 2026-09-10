#!/usr/bin/env python3
"""Turn a PROBE-instrumented server log into ground-truth preemption data.

Reads a vLLM server log produced while `probe_patch.py apply` was active, extracts
every `PROBE ...` line, and writes per-event JSONL + a summary that tells you what
`num_preemptions_total` cannot: the true number of preemptions and which requests
were evicted (and how many times).

  python3 parse_probe.py <server_log> [--out <results_dir>] [--since "<marker>"]

Writes (to --out, else next to the log):
  probe_events.jsonl  - one line per PROBE event: {t, type, req, computed, status, ...}
  probe_summary.json  - totals, per-request preemption counts, FREEREQ status split
Also prints the summary. Event types: PREEMPT (an eviction), ADMIT (a request
enters `running`; kind=new for a fresh admit, kind=resume for a re-admit after
preemption), FREEREQ (any block free; status=RUNNING => preempt-free, FINISHED =>
completion), SKIPRUN (a resident request not advanced this step), DRAIN (deferred
blocks returned), POOLFREE (blocks returned to the pool -> the KV-usage drop).
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

# e.g. "... INFO 08-12 07:47:05 [scheduler.py:1150] PROBE PREEMPT req=abc computed=2096"
TS = re.compile(r"\b(\d\d-\d\d \d\d:\d\d:\d\d)\b")
PROBE = re.compile(r"PROBE (\w+)\s*(.*)")
KV = re.compile(r"(\w+)=(\S+)")


def parse(line: str):
    mp = PROBE.search(line)
    if not mp:
        return None
    ev = {"type": mp.group(1)}
    mt = TS.search(line)
    if mt:
        ev["t"] = mt.group(1)
    for k, v in KV.findall(mp.group(2)):
        if v.isdigit():
            v = int(v)
        ev[k] = v
    return ev


def main() -> None:
    ap = argparse.ArgumentParser(description="parse PROBE server log -> ground-truth preemptions")
    ap.add_argument("server_log", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="results dir to write into (default: log's dir)")
    ap.add_argument("--since", default=None, help="only parse lines after the first line containing this marker")
    a = ap.parse_args()

    out_dir = a.out or a.server_log.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    events = []
    started = a.since is None
    for line in a.server_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if not started:
            if a.since in line:
                started = True
            continue
        if "PROBE " not in line:
            continue
        ev = parse(line)
        if ev:
            events.append(ev)

    ev_path = out_dir / "probe_events.jsonl"
    ev_path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")

    by_type = Counter(e["type"] for e in events)
    preempt_by_req = Counter(e.get("req") for e in events if e["type"] == "PREEMPT")
    admit_by_req = Counter(e.get("req") for e in events if e["type"] == "ADMIT")
    admit_kind = Counter(e.get("kind") for e in events if e["type"] == "ADMIT")
    freereq_status = Counter(e.get("status") for e in events if e["type"] == "FREEREQ")
    # 'computed' at preemption: how far each eviction got before being thrown away
    computed_at_preempt = Counter(e.get("computed") for e in events if e["type"] == "PREEMPT")

    summary = {
        "server_log": str(a.server_log),
        "events_parsed": len(events),
        "by_type": dict(by_type),
        "true_preemptions": by_type.get("PREEMPT", 0),
        "distinct_requests_preempted": len(preempt_by_req),
        "preemptions_per_request": dict(preempt_by_req.most_common()),
        "total_admissions": by_type.get("ADMIT", 0),
        "fresh_admissions": admit_kind.get("new", 0),
        "re_admissions_after_preempt": admit_kind.get("resume", 0),
        "admissions_per_request": dict(admit_by_req.most_common()),
        "freereq_status_split": dict(freereq_status),
        "computed_tokens_at_preempt": dict(sorted(computed_at_preempt.items())),
        "note": ("true_preemptions is the ground truth; the vLLM num_preemptions_total "
                 "metric typically reports only ~distinct_requests_preempted (a large "
                 "undercount). re_admissions_after_preempt should track true_preemptions "
                 "(each eviction is followed by a resume)."),
    }
    (out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({k: summary[k] for k in
                      ("events_parsed", "by_type", "true_preemptions",
                       "distinct_requests_preempted", "preemptions_per_request",
                       "total_admissions", "fresh_admissions", "re_admissions_after_preempt",
                       "freereq_status_split")}, indent=2))
    print(f"\nWrote: {ev_path}\n       {out_dir / 'probe_summary.json'}")


if __name__ == "__main__":
    main()

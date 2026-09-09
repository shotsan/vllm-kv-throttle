#!/usr/bin/env python3
"""Apply / revert ground-truth scheduler instrumentation to the vLLM in the venv.

`num_preemptions_total` is unreliable on this build (it undercounts real
preemptions ~200x). To get the exact per-eviction truth, this patches a few
`logger.info("PROBE ...")` lines into the installed vLLM scheduler + block pool so
the SERVER LOG records every preemption / block-free / skip with the request id.

  python3 probe_patch.py apply     # back up the 3 files (*.bak) and insert probes
  python3 probe_patch.py revert     # restore from *.bak (removes all probes)
  python3 probe_patch.py status     # show whether probes are currently applied

Workflow: `apply` -> start the server (run_server_preempt.sh, logging to a file)
-> run the experiment -> `parse_probe.py <server_log>` -> `revert`.
The edits are minimal and reversible; ALWAYS `revert` when done (the venv is shared).
"""
import argparse
import sys
from pathlib import Path

VENV = Path("/home/ways_lab/Documents/LLM-Network-Study/.venv"
            "/lib/python3.12/site-packages/vllm/v1/core")
SCHED = VENV / "sched" / "scheduler.py"
BLOCKS = VENV / "block_pool.py"

# Each patch: (file, anchor_substring, inserted_line(s)). The inserted line is
# added immediately AFTER the anchor line. Idempotent: skipped if already present.
PATCHES = [
    # every request-level block free (called by preempt AND completion)
    (SCHED,
     '        """Free the request\'s KV blocks, deferring the return to the block\n'
     '        pool when an in-flight GPU step may still write them.\n'
     '        """\n',
     '        logger.info("PROBE FREEREQ req=%s computed=%d status=%s npre=%d defer=%s",\n'
     '                    request.request_id, request.num_computed_tokens, request.status,\n'
     '                    request.num_preemptions, self.defer_block_free)\n'),
    # the counted-preemption path
    (SCHED,
     '        assert request.status == RequestStatus.RUNNING, (\n'
     '            "Only running requests can be preempted"\n'
     '        )\n',
     '        logger.info("PROBE PREEMPT req=%s computed=%d", request.request_id, request.num_computed_tokens)\n'),
    # a waiting/preempted request admitted into the running set (kind=new|resume)
    (SCHED,
     '                self.running.append(request)\n',
     '                logger.info("PROBE ADMIT req=%s computed=%d kind=%s", request.request_id, '
     'request.num_computed_tokens, "resume" if request.status == RequestStatus.PREEMPTED else "new")\n'),
    # a running request skipped this step (e.g. mamba block-aligned chunk == 0)
    (SCHED,
     '            if num_new_tokens == 0:\n',
     '                logger.info("PROBE SKIPRUN req=%s computed=%d", request.request_id, request.num_computed_tokens)\n'),
    # deferred blocks actually returned to the pool (async scheduling)
    (SCHED,
     '            _, blocks = self.deferred_frees.popleft()\n',
     '            logger.info("PROBE DRAIN nblocks=%d", len(blocks))\n'),
    # every actual return of blocks to the pool (moves the KV-usage gauge)
    (BLOCKS,
     '        # Identify blocks with hash (LRU cache) and without it (will never match in APC)\n',
     '        logger.info("PROBE POOLFREE free_before=%d", self.get_num_free_blocks())\n'
     '        # Identify blocks with hash (LRU cache) and without it (will never match in APC)\n'),
]

FILES = [SCHED, BLOCKS]


def _bak(p: Path) -> Path:
    return p.with_suffix(p.suffix + ".bak")


def status() -> bool:
    applied = any("PROBE " in p.read_text(encoding="utf-8") for p in FILES if p.exists())
    for p in FILES:
        n = p.read_text(encoding="utf-8").count("PROBE ") if p.exists() else 0
        print(f"  {p.name}: {n} probe line(s)  (backup: {'yes' if _bak(p).exists() else 'no'})")
    print("STATUS:", "APPLIED" if applied else "clean")
    return applied


def apply() -> None:
    if any("PROBE " in p.read_text(encoding="utf-8") for p in FILES):
        print("Probes already present; revert first if you want a clean re-apply.")
        return
    # back up
    for p in FILES:
        _bak(p).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    # apply each patch (anchor found once; insert the replacement which is the
    # anchor block possibly wrapped/prefixed with the probe line)
    added = 0
    for path, anchor, insert in PATCHES:
        text = path.read_text(encoding="utf-8")
        if anchor not in text:
            print(f"!! anchor not found in {path.name}; reverting")
            revert(); sys.exit(1)
        if insert.endswith(anchor.splitlines(keepends=True)[-1]):
            # POOLFREE style: `insert` already includes the anchor line at its end
            replacement = insert
        else:
            replacement = anchor + insert
        text = text.replace(anchor, replacement, 1)
        path.write_text(text, encoding="utf-8")
        added += 1
    print(f"applied {added} probes to {', '.join(p.name for p in FILES)}")
    _verify_imports()


def revert() -> None:
    for p in FILES:
        b = _bak(p)
        if b.exists():
            p.write_text(b.read_text(encoding="utf-8"), encoding="utf-8")
            b.unlink()
            print(f"restored {p.name}")
        else:
            print(f"no backup for {p.name} (left as-is)")
    if any(p.exists() and "PROBE " in p.read_text(encoding="utf-8") for p in FILES):
        print("!! PROBE lines still present -- check manually")
    else:
        print("clean (no PROBE lines remain)")


def _verify_imports() -> None:
    import py_compile
    for p in FILES:
        py_compile.compile(str(p), doraise=True)
    print("py_compile OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="apply/revert vLLM scheduler probes")
    ap.add_argument("action", choices=["apply", "revert", "status"])
    a = ap.parse_args()
    {"apply": apply, "revert": revert, "status": status}[a.action]()


if __name__ == "__main__":
    main()

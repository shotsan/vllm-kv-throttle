#!/usr/bin/env python3
"""Runtime probe: dump vLLM's GROUND-TRUTH KVCacheConfig + live block accounting.

`kv_geometry.py` derives the page geometry statically (no GPU) and is already
verified against the logs. THIS probe closes the last gap -- the live per-step
KV% (the '44/56/89/100 = ninths' pattern) and the observed 2-running cap -- by
patching two dumps into the installed vLLM:

  1. KVCacheConfig at startup: every kv_cache_group's spec type
     (FullAttentionSpec vs MambaSpec), block_size, page_size_bytes, #layers,
     and num_blocks. This says EXACTLY how the 10 blocks split across the
     attention group and the mamba group.
  2. Per-schedule block-manager usage: free/used blocks PER GROUP each step,
     so you can see whether the '9 usable' denominator and the 2-seq ceiling
     come from the attention pool or the mamba pool.

Usage (same pattern as probe_patch.py -- shared venv, ALWAYS revert after):
    python3 probe_kvconfig.py apply
    ./run_server_preempt.sh > server_kvcfg.log 2>&1      # boot once (~5 min, 35GB)
    grep -E "KVCFG|GRPUSE" server_kvcfg.log | head -40
    python3 probe_kvconfig.py revert

The startup dump (KVCFG lines) appears within the first minute after
'Available KV cache memory'; you do NOT need to run the load test to get it.
"""
import argparse, sys
from pathlib import Path

VENV = Path("/home/ways_lab/Documents/LLM-Network-Study/.venv"
            "/lib/python3.12/site-packages/vllm/v1/core")
KVU = VENV / "kv_cache_utils.py"
KVCM = VENV / "kv_cache_manager.py"

# Dump #1: full KVCacheConfig right after vLLM logs "GPU KV cache size".
KVCFG_ANCHOR = '            logger.info_once("GPU KV cache size: %s tokens", f"{num_tokens:,}")\n'
KVCFG_INSERT = (
    '            for _gi, _g in enumerate(kv_cache_config.kv_cache_groups):\n'
    '                _s = _g.kv_cache_spec\n'
    '                logger.info("KVCFG group=%d type=%s block_size=%s page_bytes=%s nlayers=%d num_blocks=%s",\n'
    '                            _gi, type(_s).__name__, getattr(_s, "block_size", "?"),\n'
    '                            _s.page_size_bytes, len(_g.layer_names), kv_cache_config.num_blocks)\n'
)

# Dump #2: exact block accounting each step. usage() -> block_pool.get_usage() is
# the SAME number reported as "GPU KV cache usage: X%" (the ninths). We log the raw
# free/total block counts (the denominator) plus per-group manager block counts, so
# the '9 usable' and 2-seq cap are resolved from ground truth.
GRPUSE_ANCHOR = '        return self.block_pool.get_usage()\n'
GRPUSE_INSERT = (
    '        try:\n'
    '            _bp = self.block_pool\n'
    '            _free = _bp.get_num_free_blocks(); _tot = _bp.num_gpu_blocks\n'
    '            _mgrs = getattr(self.coordinator, "single_type_managers", None) or []\n'
    '            _per = [(type(m).__name__, len(getattr(m, "req_to_blocks", {}))) for m in _mgrs]\n'
    '            logger.info("GRPUSE free=%d total=%d used=%d groups=%s",\n'
    '                        _free, _tot, _tot - _free, _per)\n'
    '        except Exception as _e:\n'
    '            logger.info("GRPUSE err=%s", _e)\n'
)

PATCHES = [(KVU, KVCFG_ANCHOR, KVCFG_INSERT, "after"),
           (KVCM, GRPUSE_ANCHOR, GRPUSE_INSERT, "before")]  # before: anchor is a `return`
FILES = [KVU, KVCM]


def _bak(p): return p.with_suffix(p.suffix + ".bak2")

def status():
    for p in FILES:
        n = p.read_text().count("KVCFG") + p.read_text().count("GRPUSE") if p.exists() else 0
        print(f"  {p.name}: {n} probe line(s)  backup={'yes' if _bak(p).exists() else 'no'}")

def apply():
    if any(("KVCFG" in p.read_text() or "GRPUSE" in p.read_text()) for p in FILES if p.exists()):
        print("already applied; revert first"); return
    for p in FILES:
        if not p.exists(): print(f"!! missing {p}"); sys.exit(1)
        _bak(p).write_text(p.read_text())
    for path, anchor, insert, pos in PATCHES:
        txt = path.read_text()
        if anchor not in txt:
            print(f"!! anchor not found in {path.name}:\n    {anchor.strip()[:70]}")
            print("   (vLLM version drift -- inspect and adjust anchor); reverting")
            revert(); sys.exit(1)
        repl = (insert + anchor) if pos == "before" else (anchor + insert)
        path.write_text(txt.replace(anchor, repl, 1))
    import py_compile
    for p in FILES: py_compile.compile(str(p), doraise=True)
    print("applied KVCFG + GRPUSE probes; py_compile OK")

def revert():
    for p in FILES:
        b = _bak(p)
        if b.exists(): p.write_text(b.read_text()); b.unlink(); print(f"restored {p.name}")
        else: print(f"no backup for {p.name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["apply","revert","status"])
    {"apply":apply,"revert":revert,"status":status}[ap.parse_args().action]()

if __name__ == "__main__":
    main()

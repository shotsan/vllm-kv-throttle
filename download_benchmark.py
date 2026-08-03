#!/usr/bin/env python3
import hashlib
import urllib.request
from pathlib import Path

COMMIT = "6734382215bea3a63f055ffd7873b2967b6a2477"
URL = (
    "https://raw.githubusercontent.com/bigai-nlco/LooGLE/"
    f"{COMMIT}/LooGLE-testdata/longdep_qa.jsonl"
)
EXPECTED_SHA256 = "596af3daf28053ff8c29c7c88eb132b9053497d8021db8c4f84af446363a3927"
TARGET = Path("benchmark-source/LooGLE-testdata/longdep_qa.jsonl")


def main() -> None:
    print(f"Downloading pinned LooGLE data from {URL}")
    with urllib.request.urlopen(URL, timeout=120) as response:
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"SHA-256 mismatch: expected {EXPECTED_SHA256}, got {digest}")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(content)
    print(f"Wrote {len(content):,} bytes to {TARGET} (SHA-256 verified)")


if __name__ == "__main__":
    main()

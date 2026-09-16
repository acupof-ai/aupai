#!/usr/bin/env python3
"""Verify the vendored DeepSeek-V4.1 reference snapshot against its frozen hashes.

Run: python third_party/deepseek_v41_ref/verify_ref.py
Exits nonzero if any byte changed, so CI can fail on a silently edited oracle.
"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

WANT = {
    "model_ref.py.ref": "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
    "engram_ref.py.ref": "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
    "kernel_ref.py.ref": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
    "config.json": "2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809",
}


def main() -> int:
    bad = 0
    for name, want in WANT.items():
        p = HERE / name
        if not p.exists():
            print(f"MISSING {name}")
            bad += 1
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != want:
            print(f"DIFF {name}\n  got  {got}\n  want {want}")
            bad += 1
        else:
            print(f"ok {name} {got[:16]}")
    if bad:
        print(f"FAIL {bad} file(s) changed; the oracle is read-only")
        return 1
    print(f"verified {len(WANT)}/{len(WANT)} reference files")
    return 0


if __name__ == "__main__":
    sys.exit(main())

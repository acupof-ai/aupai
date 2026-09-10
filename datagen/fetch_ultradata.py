#!/usr/bin/env python3
"""Fetch openbmb/UltraData-Code python shards from hf-mirror, resume-capable.

    python datagen/fetch_ultradata.py --level L3 --first 1 --last 3

Shards land in data/raw/ultradata/. A shard with a valid parquet footer is
skipped; a partial file is resumed in place with curl -C -.
"""
import argparse
import os
import subprocess
import time

BASE = "https://hf-mirror.com/datasets/openbmb/UltraData-Code/resolve/main"
N_SHARDS = {"L2": 119, "L3": 147}


def footer_ok(path):
    if not os.path.exists(path) or os.path.getsize(path) < 8:
        return False
    with open(path, "rb") as fh:
        fh.seek(-4, 2)
        return fh.read(4) == b"PAR1"


def fetch(level, first, last, dest):
    n = N_SHARDS[level]
    os.makedirs(dest, exist_ok=True)
    for i in range(first, last + 1):
        name = f"UltraData-Code-{level}-py-part-{i:05d}-of-{n:05d}.parquet"
        out = os.path.join(dest, name)
        if footer_ok(out):
            print(f"SKIP {name} ({os.path.getsize(out)} bytes)", flush=True)
            continue
        t0 = time.time()
        # -4: the pod's IPv6 egress is broken; -C - resumes; -f fails loud on HTTP errors.
        r = subprocess.run(["curl", "-4", "-fSL", "-C", "-", "--retry", "3",
                            "--connect-timeout", "15", "-o", out,
                            f"{BASE}/data/UltraData-Code-{level}/py/{name}"])
        if r.returncode != 0 or not footer_ok(out):
            print(f"FAIL {name} rc={r.returncode} footer={footer_ok(out)}", flush=True)
            continue
        print(f"OK {name} ({os.path.getsize(out)} bytes, {time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, choices=["L2", "L3"])
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=3)
    ap.add_argument("--dest", default="data/raw/ultradata")
    fetch(**vars(ap.parse_args()))


if __name__ == "__main__":
    main()

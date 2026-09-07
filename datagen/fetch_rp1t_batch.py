#!/usr/bin/env python3
# restartable: per-file, and per-chunk inside a file -- _ranged_get skips a complete span and
# resumes a partial one from lo+bytes_on_disk, so an interrupt costs the chunk in flight.
"""Fetch a named list of RedPajama-1T github files through fetch_corpus._ranged_get.

Replaces runs/fetch8p.sh, whose failure branch (`wait $p || ok=0` then `rm -f "$out".c*`)
deleted six complete chunks when two throttled ones were killed -- 1.8 GB of a 2.02 GB file,
gate_failure_incidents 263. It is a separate entry point from fetch_corpus.fetch() because
that one walks a source's whole manifest from the top; this takes an explicit file list, which
is how the 30B code batches are actually fetched.

    python3 datagen/fetch_rp1t_batch.py --list runs/next8.txt --out data/raw/rp1t_github_b2
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch_corpus import _ranged_get  # noqa: E402

BASE = "https://data.together.xyz/redpajama-data-1T/v1.0.0/github"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="file of shard names, one per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--chunks", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = [n for n in open(a.list).read().split() if n]
    print(f"{len(names)} file(s) from {a.list}", flush=True)

    done = failed = 0
    for f in names:
        dst = os.path.join(a.out, f)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"skip {f} ({os.path.getsize(dst)}B)", flush=True)
            done += 1
            continue
        ok, why = _ranged_get(f"{a.base}/{f}", dst, f, chunks=a.chunks)
        if ok:
            print(f"done {f} {why}B", flush=True)
            done += 1
        else:
            # The chunks that DID complete stay on disk; re-running resumes from them.
            print(f"FAILED {f}: {why}", file=sys.stderr, flush=True)
            failed += 1
    print(f"FETCH COMPLETE {done} done, {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Tokenize every data/mix.json domain into its cache before training starts.

train.py does this itself on rank 0 inside build_mix(), with the other ranks parked on a barrier.
Doing it up front instead means a tokenizer or missing-domain failure shows up in seconds rather
than after eight ranks have allocated their GPUs, and the training launch starts on warm caches.

    python datagen/pretokenize.py [--mix data/mix_scale_3.24b.json] [--domains web,math]
    python datagen/pretokenize.py --workers 8   # t50: process-parallel, 9-14 min for 15B
"""

import argparse
import json
import os
import sys
import time

# t50: with --workers N>1, cap the tokenizers rayon pool per process BEFORE
# importing train (which imports tokenizers). fork()ed encode workers inherit the
# pool, so workers x (nproc/workers) = nproc total threads; setting it any later
# has no effect on an already-initialized pool.
if "--workers" in sys.argv:
    _w = int(sys.argv[sys.argv.index("--workers") + 1])
    if _w > 1:
        os.environ.setdefault("RAYON_NUM_THREADS", str(max(1, (os.cpu_count() or 1) // _w)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import harness  # single source of truth for the configured mix

import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    # The configured mix, not a hardcoded name: a hardcoded path goes stale when the mix is retired.
    ap.add_argument("--mix", default=os.path.join(ROOT, harness.cfg_default("mix")))
    ap.add_argument("--domains", help="comma-separated subset (default: every domain in the mix)")
    ap.add_argument("--workers", type=int, default=1,
                    help="processes per domain for tokenization (t50; 1 = train.py's own path)")
    a = ap.parse_args()

    mix = json.load(open(a.mix, encoding="utf-8"))
    names = a.domains.split(",") if a.domains else list(mix["domains"])
    assert os.path.exists(train.TOK_PATH), f"no tokenizer at {train.TOK_PATH}"
    tok = train.build_tokenizer([])

    total = 0
    for d in names:
        t0 = time.time()
        seqs = train._domain_seqs(d, tok, is_main=True, ddp=False, workers=a.workers)
        n = seqs.numel()
        total += n
        print(
            f"{d:<6} {len(seqs):>9} rows x {train.Cfg.seq + 1} = {n / 1e9:.2f}B tokens  "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
    print(f"TOTAL {total / 1e9:.2f}B tokens across {len(names)} domains")
    print("PRETOKENIZE_DONE")


if __name__ == "__main__":
    main()

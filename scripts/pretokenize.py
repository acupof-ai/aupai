#!/usr/bin/env python3
"""Tokenize every data/mix.json domain into its cache before training starts.

train.py does this itself on rank 0 inside build_mix(), with the other ranks parked on a barrier.
Doing it up front instead means a tokenizer or missing-domain failure shows up in seconds rather
than after eight ranks have allocated their GPUs, and the training launch starts on warm caches.

    python scripts/pretokenize.py [--mix data/mix_v3.json] [--domains web,math]
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import harness  # single source of truth for the configured mix

import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    # The configured mix, not a hardcoded name: a hardcoded path goes stale when the mix is retired.
    ap.add_argument("--mix", default=os.path.join(ROOT, harness.cfg_default("mix")))
    ap.add_argument("--domains", help="comma-separated subset (default: every domain in the mix)")
    a = ap.parse_args()

    mix = json.load(open(a.mix, encoding="utf-8"))
    names = a.domains.split(",") if a.domains else list(mix["domains"])
    assert os.path.exists(train.TOK_PATH), f"no tokenizer at {train.TOK_PATH}"
    tok = train.build_tokenizer([])

    total = 0
    for d in names:
        t0 = time.time()
        seqs = train._domain_seqs(d, tok, is_main=True, ddp=False)
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

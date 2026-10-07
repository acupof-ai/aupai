#!/usr/bin/env python3
"""Driver for build_corpus's calibrated near-dedup post-pass over en_fwe4.

The engine is a library function (no CLI): bands=64 rows=2 -> recall ~1.0 at
J>=0.5, exact word-3gram decision, _norm_skeleton (lowercase+ws) for English web.

SCALE FIX (6.65M fineweb docs): the stock _lsh_candidates enumerates every pair
inside every band bucket. A handful of giant buckets (boilerplate/template docs
sharing a 2-row signature) make that quadratic set unbounded -- the stock run sat
in _lsh_candidates for 4h with RSS climbing 36->46GB and never reached rewrite.
We cap per-bucket membership at BUCKET_CAP when generating candidate pairs; giant
buckets are non-distinctive templates, and any genuinely near-duplicate pair in
them still collides in the other 63 bands under a normal-sized bucket."""
import argparse
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "datagen"))
import build_corpus as B

BUCKET_CAP = 100


def _lsh_candidates_capped(sigs, bands, rows, cap=BUCKET_CAP):
    """Same as B._lsh_candidates but a band bucket contributes at most cap docs'
    pairs (deterministic: take the lowest ordinals after sort)."""
    rivals = set()
    for b in range(bands):
        table = {}
        for o, sig in sigs.items():
            table.setdefault(sig[b * rows * 8 : (b + 1) * rows * 8], []).append(o)
        for members in table.values():
            if len(members) > 1:
                m = sorted(members)
                if len(m) > cap:
                    m = m[:cap]
                for i in range(len(m)):
                    for j in range(i + 1, len(m)):
                        rivals.add((m[i], m[j]))
    return rivals


B._lsh_candidates = _lsh_candidates_capped  # engine reads this name at call time

ap = argparse.ArgumentParser()
ap.add_argument("--domain", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=32)
args = ap.parse_args()
ns = SimpleNamespace(out=args.out, domain=args.domain, workers=args.workers)
sys.exit(B._near_dedup_postpass(ns))

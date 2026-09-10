#!/usr/bin/env python3
"""Build token caches for named mix domains with train.py's own path, CPU side.

Thin driver over train._domain_seqs: it shard-globs data/corpus/<d>, shuffles with the
sample seed, encodes via the fork worker pool, and writes tokens_<d>.pt plus the
.vocab/.srcfp/.seed stamps. A fresh cache is skipped (mmap load), so re-running reports
without re-encoding. CPU only -- set CUDA_VISIBLE_DEVICES empty in the environment.

workers x RAYON_NUM_THREADS must stay <= nproc (train._encode_domain contract); this
script does not set RAYON_NUM_THREADS for you because nproc is machine-specific.

Report line per domain (stdout):
  REPORT <d> rows=<pool rows> pool_tokens=<rows*(seq+1)> seconds=<s> sha256=<16> bytes=<n>
For an already-cached domain seconds is the mmap load, not the encode.
"""
import argparse
import hashlib
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import train  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "eval"))
import cache_guard  # noqa: E402


def sha256_prefix(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 26), b""):
            h.update(blk)
    return h.hexdigest()[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domains", nargs="+")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    tok = train.build_tokenizer([])
    print("cache dir", train._token_cache_dir(), "vocab", train.VOCAB_ID[:12], flush=True)
    for d in a.domains:
        # Building writes the same tens of GB to /data00 that a read does, so the co-residency
        # rule applies on this side too: refuse while a live claim holds the cards. No claim ->
        # silent pass. An unrecorded domain size WARNs and proceeds on a lower bound.
        cache_guard.assert_not_co_resident([d])
        t = time.time()
        seqs = train._domain_seqs(d, tok, True, False, workers=a.workers)
        dt = time.time() - t
        rows = len(seqs)
        path = train._domain_cache_path(d)
        print(
            f"REPORT {d} rows={rows} pool_tokens={rows * (train.Cfg.seq + 1)} "
            f"seconds={dt:.0f} sha256={sha256_prefix(path)} bytes={os.path.getsize(path)}",
            flush=True,
        )
        del seqs
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

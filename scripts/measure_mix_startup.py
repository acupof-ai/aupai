#!/usr/bin/env python3
# restartable: read-only timing report; re-running is idempotent and does not write the corpus or caches
"""fb: throughput push — measure build_mix startup for the 200M@4B retrain.

For a 2-hour run, startup cost is dominated by torch.load of the token caches.
_corpus_fp (dir fingerprint) is a separate cost, measured here by calling the real
train._corpus_fp (train.py:1355-1376) directly, NOT a re-implementation.

This measures, per domain cache: file bytes, torch.load seconds (one read), and the
pre-load `_corpus_fp` fingerprint seconds (if the corpus dir is present). Reports
per row and a summed startup estimate. Host-side, no GPU, read-only.

CORRECTION (2026-09-02, de + fb): the earlier version hashed every byte of every
shard; the real _corpus_fp reads only first/last 64KB per shard. That version timed
a function startup never calls. This one calls train._corpus_fp directly.

Usage (pod):
  python3 scripts/measure_mix_startup.py data/mix_200m_4b.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mix_path", nargs="?", help="mix json (optional if --domains given)")
    ap.add_argument("--domains", nargs="*", help="override domains (mix file may not exist/be stale)")
    ap.add_argument("--cache_dir", default="/data00")
    ap.add_argument("--rows_only", action="store_true", help="time torch.load only, skip corpus_fp walk")
    a = ap.parse_args()

    mix = json.load(open(a.mix_path, encoding="utf-8")) if a.mix_path else {}
    domains = a.domains or list(mix.get("domains", {}).keys())
    print(f"mix {a.mix_path or '(domains-only)'}: total_tokens={mix.get('total_tokens')}, {len(domains)} domains")

    import torch

    # THE CO-RESIDENCY REFUSAL, for the whole mix at once because that is what this tool
    # reads: a full non-mmap torch.load of every cache in the file, which at mix_200m_8b is
    # 161 GB off /data00. This is the read the "no lane card" rule was written for, and it
    # had no refusal in front of it because the guard's chokepoint is train._domain_seqs and
    # this file opens the caches by path (e1, 2026-09-05).
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))
    from cache_guard import assert_not_co_resident

    assert_not_co_resident(domains)
    rows = []
    total_bytes = total_load_s = total_walk_s = 0.0
    for name in domains:
        cache = os.path.join(a.cache_dir, f"tokens_{name}.pt")
        if not os.path.exists(cache):
            print(f"  {name}: NO CACHE {cache}")
            continue
        b = os.path.getsize(cache)
        # fresh-read torch.load
        t0 = time.perf_counter()
        obj = torch.load(cache, map_location="cpu", weights_only=True)
        load_s = time.perf_counter() - t0
        # decode length for a row-count note
        try:
            n = len(obj["ids"]) if isinstance(obj, dict) and "ids" in obj else "?"
        except Exception:
            n = "?"
        rows.append((name, b, load_s, n))
        total_bytes += b
        total_load_s += load_s
        del obj
    # corpus_fp overlaps with the cache load (it runs BEFORE, so they're sequential)
    if not a.rows_only:
        import train  # 3b: call the real fingerprint, not a re-implementation
        for name, b, load_s, n in rows:
            corpus_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                      "data", "corpus", name)
            if os.path.isdir(corpus_dir):
                t0 = time.perf_counter()
                train._corpus_fp(corpus_dir)
                ws = time.perf_counter() - t0
                total_walk_s += ws
                print(f"  {name}: fp {ws:.2f}s (train._corpus_fp)")
            else:
                print(f"  {name}: corpus dir absent (skip fp)")

    print(json.dumps({"mix": a.mix_path, "n_domains_cached": len(rows),
                      "total_cache_bytes_GB": total_bytes / 1e9,
                      "total_torch_load_s": round(total_load_s, 2),
                      "total_corpus_fp_walk_s": round(total_walk_s, 2),
                      "est_startup_s": round(total_load_s + total_walk_s, 2),
                      "pct_of_2h_run": round((total_load_s + total_walk_s) / 7200 * 100, 2),
                      "rows": [{"domain": d, "bytes_GB": round(x / 1e9, 3), "load_s": round(l, 2), "n": n}
                               for d, x, l, n in rows]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
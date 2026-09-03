#!/usr/bin/env python3
"""Doc-level subset check between two corpus dirs: is every doc in B (by content
hash) present in A? w/o re-reading A per B doc. Hashes each doc's content
(sha256 of the raw content string) in both dirs, then computes |B ∩ A| / |B|.
Decisive for the en_c4 two-dir question (2026-09-03, controller): if B ⊆ A it is
a deletion candidate; extra docs are supply not to double-count. shard-names
alone cannot prove subset (same name pattern, could be different content), so
this hashes content.

    python3 datagen/corpus_subset_check.py --root data/corpus --sub en_c4_stage2 \
        --sup en_c4
"""
import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os


def doc_hashes(shard):
    hs = []
    with open(shard, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                c = json.loads(line).get("content", "")
            except Exception:
                continue
            if c:
                hs.append(hashlib.sha256(c.encode("utf-8", "replace")).digest())
    return hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sup", required=True)  # the larger dir
    ap.add_argument("--sub", required=True)  # the candidate subset
    a = ap.parse_args()
    sup_shards = sorted(glob.glob(os.path.join(a.root, a.sup, "*.jsonl")))
    sub_shards = sorted(glob.glob(os.path.join(a.root, a.sub, "*.jsonl")))
    with mp.Pool(16) as pool:
        sup_h = [h for part in pool.map(doc_hashes, sup_shards) for h in part]
    sup_set = set(sup_h)
    with mp.Pool(16) as pool:
        sub_h = [h for part in pool.map(doc_hashes, sub_shards) for h in part]
    overlap = sum(1 for h in sub_h if h in sup_set)
    sub_n = len(sub_h)
    frac = overlap / max(1, sub_n)
    extra = sub_n - overlap
    print(json.dumps({
        "sup": a.sup, "sup_shards": len(sup_shards), "sup_docs": len(sup_set),
        "sub": a.sub, "sub_shards": len(sub_shards), "sub_docs": sub_n,
        "overlap": overlap, "subset_fraction": frac,
        "extra_not_in_sup": extra,
        "verdict": "subset" if frac >= 0.999 else ("mostly_subset" if frac >= 0.9 else "not_subset"),
    }, indent=1))


if __name__ == "__main__":
    main()
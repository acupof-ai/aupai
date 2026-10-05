#!/usr/bin/env python3
"""Sample-estimate swallow-code's REAL near-duplicate rate before deciding on a MinHash pass.

Read-only. Draws a fixed-seed random sample from the built, exact-deduped
data/corpus/swallowcode_scor/swallowcode_scor_*.jsonl shards, generates 128-perm MinHash
candidate pairs with the production Fork-C banding (64 bands x 2 rows), then makes the
DECISION by exact normalized char-5gram-set Jaccard >= 0.5 and counts union-find removals.

This is the code analogue of bb's swallow-math sample; it can NOT reuse that number: code
normalises by stripping [\\s\\W_]+ and shingles char 5-grams (build_corpus MinHashLSH), math
uses normalized word 3-grams. MinHash/LSH is a candidate generator only (the 99.6% false
positive on math is why the exact-J decision and the candidate false-positive rate are both
printed).

    python3 scripts/neardup_sample_swallowcode.py --n 150000 --seed 17 [--root /work/aupai]
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--domain", default="swallowcode_scor")
    ap.add_argument("--n", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--bands", type=int, default=64)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--jaccard", type=float, default=0.5)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.root, "datagen"))
    import build_corpus as bc  # noqa: E402

    paths = sorted(glob.glob(os.path.join(args.root, "data", "corpus", args.domain,
                                          f"{args.domain}_*.jsonl")))
    if not paths:
        raise SystemExit(f"no built {args.domain}_*.jsonl shards under {args.root}")

    # Reservoir sample over the whole built corpus (shard-order independent, fixed seed).
    rng = random.Random(args.seed)
    sample, seen = [], 0
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                seen += 1
                if len(sample) < args.n:
                    sample.append(line)
                else:
                    j = rng.randrange(seen)
                    if j < args.n:
                        sample[j] = line
    n = len(sample)

    perms = args.bands * args.rows
    ab, mask = bc._near_coeffs(perms, args.seed)
    norm = bc._NORM
    sigs, texts, shingle_cache = {}, {}, {}

    def shingles(text):
        s = norm.sub("", text)
        return {s[i : i + 5] for i in range(max(1, len(s) - 4))}

    for o, line in enumerate(sample):
        text = json.loads(line).get("content") or ""
        texts[o] = text
        sh = shingles(text)
        hs = [int.from_bytes(hashlib.blake2b(x.encode(), digest_size=8).digest(), "little")
              for x in sh]
        if hs:
            sigs[o] = bc._minhash(hs, ab, mask)

    candidates = bc._lsh_candidates(sigs, args.bands, args.rows)
    exact_edges = []
    for lo, hi in candidates:
        a = shingle_cache.setdefault(lo, shingles(texts[lo]))
        b = shingle_cache.setdefault(hi, shingles(texts[hi]))
        if a and b and len(a & b) / len(a | b) >= args.jaccard:
            exact_edges.append((lo, hi))
    fp = 100.0 * (1 - len(exact_edges) / len(candidates)) if candidates else 0.0

    # Union-find over exact edges; a component of size k removes k-1 docs.
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for lo, hi in exact_edges:
        parent[find(lo)] = find(hi)
    size = {}
    for x in range(n):
        r = find(x)
        size[r] = size.get(r, 0) + 1
    removed = sum(k - 1 for k in size.values() if k > 1)
    big = sorted((k for k in size.values() if k > 1), reverse=True)[:5]

    out = {
        "domain": args.domain,
        "corpus_docs_seen": seen,
        "sample_n": n,
        "seed": args.seed,
        "banding": {"perms": perms, "bands": args.bands, "rows": args.rows},
        "exact_decision": "normalized char-5gram-set Jaccard",
        "jaccard_threshold": args.jaccard,
        "lsh_candidate_pairs": len(candidates),
        "candidate_false_positive_pct": round(fp, 3),
        "exact_near_dup_pairs": len(exact_edges),
        "union_find_removed": removed,
        "near_dup_rate_pct": round(100.0 * removed / n, 4),
        "largest_clusters": big,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

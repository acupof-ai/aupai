#!/usr/bin/env python3
# restartable: read-only report tool — an interrupt costs only the re-scan; it never writes the corpus
"""3b-8 FULL: near-dedup report across BOTH pre-train code domains (starcoder 28G + rp1t 1.5G).

fb (2026-09-02): run code_py_starcoder + code_py_rp1t together — cross-domain dupes are
the thing we want to see. 28G cannot be pairwise, so MinHash+LSH (build_corpus.MinHashLSH).
Thresholds PRE-REGISTERED at Jaccard 0.7 / 0.8 / 0.9 (report all three). Output is a REPORT,
not data. NEVER rewrites a shard (these dirs are ladder-mix frozen).

Streaming shape (28G text must not all live in RAM):
  per doc -> normalise_code -> shingles -> minhash signature -> LSH bands -> bucket[(b,key)] appends ordinal
  bucket ordinals (sigs dropped) -> candidate pairs -> Jaccard computed by re-locating the two
  original lines (ordinal -> domain/shard/offset) and re-reading just those two docs.
  Cross-domain flag: ordinal's domain differs.

Report per threshold: docs_in_clustered_pairs, duplication rate, cluster-size distribution,
20 sample pair (path,path, J), cross-domain pair share. Exit 0 always (report, not gate).

Run (pod, /work/aupai): python3 datagen/near_dedup_scale.py --root data/corpus --out runs/3b8_code_dedup.json --domains code_py_starcoder code_py_rp1t
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_corpus as B  # noqa: E402
import near_dedup_postpass as ND  # noqa: E402  (normalise_code, word_shingles, jaccard)

THRESHOLDS = (0.7, 0.8, 0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perms", type=int, default=96)
    ap.add_argument("--bands", type=int, default=12)
    ap.add_argument("--max_docs", type=int, default=0, help="cap for a smoke run")
    a = ap.parse_args()

    lsh = B.MinHashLSH(perms=a.perms, bands=a.bands)
    band_rows = a.perms // a.bands

    shards = []  # (domain, path)
    for dom in a.domains:
        for p in sorted(glob.glob(os.path.join(a.root, dom, "*.jsonl"))):
            shards.append((dom, p))

    # pass 1: stream signatures, bucket (band,key)->[ordinal]; record doc location
    loc = []  # ordinal -> (domain, shard_path, shard_local_line_no)
    buckets = defaultdict(list)
    t0 = time.perf_counter()
    n = 0
    for dom, p in shards:
        ln = 0
        for line in open(p, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line:
                continue
            d = json.loads(line)
            t = d.get("content") or ""
            if t:
                sig = lsh.signature(t)
                idx = n
                loc.append((dom, p, ln))
                for b in range(a.bands):
                    key = tuple(sig[b * band_rows:(b + 1) * band_rows])
                    buckets[(b, key)].append(idx)
                n += 1
            ln += 1
            if a.max_docs and n >= a.max_docs:
                break
        if a.max_docs and n >= a.max_docs:
            break
    print(f"streamed {n} docs in {time.perf_counter()-t0:.1f}s, {len(buckets)} nonempty buckets", flush=True)

    # candidate pairs from buckets (dedup unordered pairs)
    pairs = set()
    for (b, key), ords in buckets.items():
        if len(ords) < 2:
            continue
        for i in range(len(ords)):
            for j in range(i + 1, len(ords)):
                u, v = ords[i], ords[j]
                pairs.add((u, v) if u < v else (v, u))
    print(f"{len(pairs)} candidate pairs from LSH", flush=True)

    # read the two docs to compute exact Jaccard (relocate by domain/shard/line)
    cache = {}

    def gettext(ordinal):
        if ordinal in cache:
            return cache[ordinal]
        dom, p, ln = loc[ordinal]
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == ln:
                    cache[ordinal] = json.loads(line).get("content", "")
                    return cache[ordinal]
        return ""

    def jac_pair(u, v):
        return ND.jaccard(ND.word_shingles(ND.normalise_code(gettext(u))),
                          ND.word_shingles(ND.normalise_code(gettext(v))))

    # compute all pair Jaccards once, populate cluster graph per threshold
    scored = {}
    for u, v in pairs:
        scored[(u, v)] = jac_pair(u, v)

    out = {"domains": a.domains, "docs": n, "candidate_pairs": len(pairs),
           "thresholds": {}, "pre_registered": True, "method": "MinHash-LSH (perms=%d bands=%d) + exact Jaccard" % (a.perms, a.bands)}
    for th in THRESHOLDS:
        e = [(u, v, j) for (u, v), j in scored.items() if j >= th]
        # cluster graph
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[max(rx, ry)] = min(rx, ry)

        for u, v, _ in e:
            union(u, v)
        cl = defaultdict(list)
        for i in range(n):
            cl[find(i)].append(i)
        multi = {r: mem for r, mem in cl.items() if len(mem) > 1}
        docs_in = sum(len(m) for m in multi.values())
        cross = sum(1 for u, v, _ in e
                    if loc[u][0] != loc[v][0])
        sizes = sorted(len(m) for m in multi.values())
        from collections import Counter
        size_dist = dict(sorted(Counter(sizes).items()))
        samples = sorted(e, key=lambda x: -x[2])[:20]
        out["thresholds"][str(th)] = {
            "pairs": len(e), "docs_in_clustered_pairs": docs_in,
            "duplication_rate": round(docs_in / max(1, n), 4),
            "clusters": len(multi), "cluster_size_dist": size_dist,
            "cross_domain_pairs": cross, "cross_domain_share": round(cross / max(1, len(e)), 4),
            "sample_pairs": [{"a": "%s:%s" % (loc[u][0], loc[u][1]), "b": "%s:%s" % (loc[v][0], loc[v][1]),
                              "j": round(j, 3)} for u, v, j in samples],
        }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False)[:1200])


if __name__ == "__main__":
    main()
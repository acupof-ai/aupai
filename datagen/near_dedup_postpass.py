#!/usr/bin/env python3
# restartable: read-only report tool; an interrupt costs only the re-scans, and re-running is idempotent (it never writes the corpus)
"""3b-8: near-dedup post-pass dry-run (report-only) over a cleaned code/math domain.

fb: run near-dedup per 44's six conditions. This DRY RUN (condition 2: cluster
report before any deletion) mines the domain shards, normalises code, bands
MinHash signatures into LSH candidate buckets, computes pairwise Jaccard>=0.5 on
those, union-finds clusters, and REPORTS the clusters + retention WITHOUT deleting.
Holdout rows protected (never clustered/removed). For the acceptance: per-domain
retention + a 100-cluster hand-read sample written out for manual adjudication.

Method (near_dedup_gate.md, 44): normalise code = strip comments + string
literals, numbers->#, identifiers->placeholder (70-word keyword stoplist kept),
collapse whitespace; word 3-gram shingles; Jaccard>=0.5; union-find; keep lowest
ordinal. Signature: build_corpus.MinHashLSH (128-perm, 16 bands x 8 rows).

Usage (pod, read-only): python3 datagen/near_dedup_postpass.py data/corpus/<domain> --out <dir>
Only reads the corpus; deletes NOTHING.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_corpus as B  # noqa: E402

# code normalisation (44 gate, near_dedup_gate.md)
_COMMENT = re.compile(r"#[^\n]*|/\*.*?\*/|//[^\n]*", re.S)
_STRING = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|"""(?:\\.|[^"\\])*?"""', re.S)
_NUM = re.compile(r"\b\d[\d_.]*\b")
_IDENT = re.compile(r"\b[a-zA-Z_]\w*\b")
_KEYWORDS = set("""and as assert async await break class continue def del elif else except finally for
from global if import in is lambda nonlocal not or pass raise return try while with yield True False
None self cls int float str bool list dict set tuple range len min max sum abs print return def
class import from if elif else while for in not and or is return lambda""".split())
_SPACE = re.compile(r"\s+")


def normalise_code(text):
    t = _COMMENT.sub(" ", text)
    t = _STRING.sub(" ", t)
    t = _NUM.sub("#", t)
    t = _IDENT.sub(lambda m: m.group(0) if m.group(0) in _KEYWORDS else "@", t)
    return _SPACE.sub(" ", t)


def word_shingles(text, k=3):
    words = text.split()
    if len(words) < k:
        return set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir")
    ap.add_argument("--out", required=True, help="dir for cluster report + hand-read sample")
    ap.add_argument("--max_docs", type=int, default=0)
    a = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(a.corpus_dir, "*.jsonl")))
    if not shards:
        print("no shards", file=sys.stderr); sys.exit(1)
    lsh = B.MinHashLSH(perms=128, bands=16)

    docs = []  # (ordinal, orig_text, shingle_set)
    for p in shards:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t = d.get("content") or ""
            if not t:
                continue
            norm = normalise_code(t)
            docs.append((len(docs), t, word_shingles(norm)))
            if a.max_docs and len(docs) >= a.max_docs:
                break
        if a.max_docs and len(docs) >= a.max_docs:
            break

    # LSH candidate buckets (banded), then pairwise Jaccard>=0.5 -> union-find
    print(f"docs={len(docs)} shards={len(shards)}", flush=True)
    parent = list(range(len(docs)))
    sigs = [lsh.signature(d[1]) for d in docs]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    from collections import defaultdict
    buckets = defaultdict(list)
    for i, s in enumerate(sigs):
        for b in range(16):
            key = tuple(s[b * 8 : (b + 1) * 8])
            buckets[(b, key)].append(i)
    n_edges = 0
    for (b, key), idxs in buckets.items():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                ai, aj = idxs[i], idxs[j]
                if docs[ai][1] == docs[aj][1]:
                    n_edges += 1; union(ai, aj); continue
                if jaccard(docs[ai][2], docs[aj][2]) >= 0.5:
                    n_edges += 1; union(ai, aj)
    # cluster membership
    clusters = defaultdict(list)
    for i in range(len(docs)):
        clusters[find(i)].append(i)
    multi = {r: v for r, v in clusters.items() if len(v) > 1}
    # retention
    kept = sum(1 for r, v in clusters.items() if len(v) == 1)
    os.makedirs(a.out, exist_ok=True)
    # cluster report (condition 2: BEFORE any delete)
    rep = os.path.join(a.out, "clusters_report.jsonl")
    sample95 = os.path.join(a.out, "handread_sample_100.txt")
    with open(rep, "w", encoding="utf-8") as fr:
        for r, members in sorted(multi.items()):
            fr.write(json.dumps({"cluster_root": r, "n_members": len(members),
                                 "members": [i for i in members], "ordinals": [docs[i][0] for i in members]},
                                ensure_ascii=False) + "\n")
    with open(sample95, "w", encoding="utf-8") as fs:
        import random as _r
        _r.seed(3)
        s = sorted(multi.items())   # deterministic
        for r, members in s[:100]:
            fs.write(f"=== cluster {r} n={len(members)} ===\n")
            for i in members[:2]:
                fs.write(docs[i][1][:200].replace("\n", " / ") + "\n---\n")
    print(json.dumps({
        "docs": len(docs), "n_edges_jac": n_edges, "clusters_total": len(clusters),
        "dup_clusters": len(multi), "docs_in_dup_clusters": sum(len(v) for v in multi.values()),
        "kept_retention": kept / max(1, len(docs)),
        "report": rep, "handread_sample": sample95,
        "method": "normalise(code)+3-word-shingle, J>=0.5, union-find, 16-band LSH candidates",
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
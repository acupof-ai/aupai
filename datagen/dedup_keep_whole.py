#!/usr/bin/env python3
# restartable: reads cached .sig.npy/.loc.json (never re-hashes) and writes shards into a
# fresh out_dir; an interrupt loses only the write pass, which is minutes.
"""Shape (b): dedup b2v2 against dd09 AND within itself, dd09 kept WHOLE.

Same engine as datagen/code_dedup_build.py -- 96 perms, 12 bands, char 5-gram, est>=th,
union-find, keep the min ordinal per cluster. Two differences, both deliberate:

  1. dd09 is listed FIRST, so its ordinals are globally lower. min(cluster) is therefore a
     dd09 document whenever a cluster spans both domains, which is exactly "dd09 kept whole":
     the b2v2 copy is dropped and dd09 never loses a document to b2v2.
  2. ONLY b2v2 documents are written out. code_dedup_build rewrites every input, which here
     would copy dd09's 20 GB unchanged. dd09 stays where it is, untouched.

The assertion that makes (1) checkable rather than assumed: zero dd09 documents may be
deleted. If any is, the ordinal ordering is not what this script claims and it refuses.
"""

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = "/work/aupai"
CK = os.path.join(ROOT, "runs/code_rp1t_ck")
KEEP_WHOLE, DEDUP = "code_rp1t_dd09", "code_rp1t_b2v2"
TH = 0.9
PERMS, BANDS = 96, 12
BAND_ROWS = PERMS // BANDS
OUT = os.path.join(ROOT, "data/corpus/code_rp1t_b2v2_dd")

t0 = time.perf_counter()


def log(m):
    r = round(time.perf_counter() - t0)
    print(f"[{r // 60}m{r % 60:02d}s] {m}", flush=True)


sigs, locs, bound = [], [], []
for dom in (KEEP_WHOLE, DEDUP):
    S = np.load(os.path.join(CK, f"{dom}.sig.npy"))
    with open(os.path.join(CK, f"{dom}.loc.json")) as _f:
        L = json.load(_f)
    assert S.shape[0] == len(L), f"{dom}: {S.shape[0]} sigs vs {len(L)} locs"
    sigs.append(S)
    locs.append(L)
    log(f"{dom}: {S.shape[0]} sigs")
all_sigs = np.vstack(sigs)
n_keep = sigs[0].shape[0]
n = all_sigs.shape[0]
log(f"{n} docs total; ordinals [0,{n_keep}) are {KEEP_WHOLE} and are globally lower")

buckets = defaultdict(list)
for g in range(n):
    sig = all_sigs[g]
    for b in range(BANDS):
        buckets[(b, tuple(sig[b * BAND_ROWS : (b + 1) * BAND_ROWS].tolist()))].append(g)
log(f"banded into {len(buckets)} buckets")

pairs = set()
for _k, ords in buckets.items():
    if len(ords) < 2:
        continue
    for i in range(len(ords)):
        for j in range(i + 1, len(ords)):
            u, v = ords[i], ords[j]
            pairs.add((u, v) if u < v else (v, u))
del buckets
pv = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)
del pairs
est = (all_sigs[pv[:, 0]] == all_sigs[pv[:, 1]]).mean(axis=1)
e = pv[est >= TH]
log(f"{pv.shape[0]} candidates, est>={TH}: {e.shape[0]} edges")

parent = list(range(n))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for u, v in e:
    rx, ry = find(u), find(v)
    if rx != ry:
        parent[max(rx, ry)] = min(rx, ry)
cl = defaultdict(list)
for i in range(n):
    cl[find(i)].append(i)
multi = {r: m for r, m in cl.items() if len(m) > 1}
deleted = {i for m in multi.values() for i in m if i != min(m)}
transitive_spared = sum(1 for m in multi.values() for i in m if i != min(m) and i < n_keep)
log(
    f"{len(multi)} clusters >=2, {len(deleted)} non-representatives "
    f"({transitive_spared} of them {KEEP_WHOLE}, spared)"
)

# KEEP WHOLE AS A CONSTRAINT, NOT AS A CONSEQUENCE OF ORDERING. Listing dd09 first makes
# min(cluster) a dd09 ordinal, which is necessary and NOT sufficient: union-find merges by
# TRANSITIVITY, so two dd09 docs that are not near-dups of each other join one cluster when a
# b2v2 doc is near both, and the higher dd09 ordinal becomes a non-representative. Measured
# 2026-09-07: 8,355 dd09 docs, every one of them transitive (dd09-dd09 edges above 0.9: ZERO).
# So the rule is stated directly -- a KEEP_WHOLE ordinal is never deleted -- and the assertion
# below now checks the CAUSE that would make that wrong rather than the symptom.
deleted = {i for i in deleted if i >= n_keep}
dd_dd_edges = int(((e[:, 0] < n_keep) & (e[:, 1] < n_keep)).sum())
if dd_dd_edges:
    raise SystemExit(
        f"REFUSE: {dd_dd_edges} edge(s) join two {KEEP_WHOLE} documents at est>={TH}. "
        f"{KEEP_WHOLE}'s own dedup should have removed those, so its docs_kept is wrong and "
        f"this pass must not build on it -- do not write."
    )

cross = sum(1 for u, v in e if (u < n_keep) != (v < n_keep))
within = int(e.shape[0]) - cross
b2_deleted = sorted(deleted)
log(
    f"edges: {cross} cross-domain, {within} within; {len(b2_deleted)} b2v2 docs to drop "
    f"of {n - n_keep} ({len(b2_deleted) / (n - n_keep):.4f})"
)

os.makedirs(OUT, exist_ok=True)
kept_by_shard = defaultdict(set)
for g in range(n_keep, n):
    if g in deleted:
        continue
    shard, ln = locs[1][g - n_keep]
    kept_by_shard[shard].add(ln)
written = bytes_w = docs_w = 0
for shard, keep_lines in kept_by_shard.items():
    tgt = os.path.join(OUT, os.path.basename(shard).replace(DEDUP, "code_rp1t_b2v2_dd"))
    with open(shard, encoding="utf-8") as fsrc, open(tgt, "w", encoding="utf-8") as fdst:
        for i, line in enumerate(fsrc):
            if i in keep_lines and line.strip():
                fdst.write(line)
                bytes_w += len(line.encode("utf-8", "replace"))
                docs_w += 1
    written += 1
    if written % 30 == 0:
        log(f"  wrote {written} shards, {bytes_w / 1e6:.0f} MB")
log(f"wrote {written} shards, {docs_w} docs, {bytes_w / 1e6:.0f} MB -> {OUT}")

sys.path.insert(0, os.path.join(ROOT, "datagen"))
from corpus_fingerprint import fp_dir  # noqa: E402

with open(os.path.join(ROOT, "data/corpus", DEDUP, "build_corpus_stats.json")) as _f:
    src = json.load(_f)
with open(os.path.join(ROOT, "data/corpus", KEEP_WHOLE, "build_corpus_stats.json")) as _f:
    keep = json.load(_f)
stats = {
    "domain": "code_rp1t_b2v2_dd",
    "inputs": {
        DEDUP: {
            "srcfp": src.get("fingerprint"),
            "filters": src.get("filters"),
            "filters_fp": src.get("filters_fp"),
            "role": "deduped",
        },
        KEEP_WHOLE: {
            "srcfp": keep.get("fingerprint"),
            "filters": keep.get("filters"),
            "role": "kept whole; not rewritten, not modified",
        },
    },
    "dedup": {
        "threshold": TH,
        "method": "MinHash-J char 5-gram",
        "n_perm": PERMS,
        "bands": BANDS,
        "shape": f"b2v2 against {KEEP_WHOLE} AND within b2v2; {KEEP_WHOLE} kept whole",
        "docs_in": n - n_keep,
        "docs_kept": docs_w,
        "docs_deleted": len(b2_deleted),
        "clusters_gt1": len(multi),
        "edges_cross_domain": cross,
        "edges_within": within,
        "edges_keep_whole_internal": 0,
        "keep_whole_transitive_spared": int(transitive_spared),
        "keep_whole_note": f"{KEEP_WHOLE} ordinals are never deleted. {transitive_spared} of "
        f"its documents were non-representatives by TRANSITIVITY (a b2v2 doc "
        f"near two dd09 docs merges them) and were spared by that rule; "
        f"direct {KEEP_WHOLE}-{KEEP_WHOLE} edges above {TH} were 0, asserted.",
        "drop_rate": round(len(b2_deleted) / (n - n_keep), 6),
        "drop_rate_config": f"docs_deleted/docs_in over b2v2 only; {KEEP_WHOLE} contributed "
        f"{n_keep} documents as dedup targets and lost none (asserted)",
    },
    "filters": f"near-dedup-th{TH}-ordinal-rep-vs-{KEEP_WHOLE}",
    "n_shards": written,
    "docs": docs_w,
    "bytes": bytes_w,
    "fingerprint": fp_dir(OUT),
}
with open(os.path.join(OUT, "build_corpus_stats.json"), "w") as _f:
    json.dump(stats, _f, ensure_ascii=False, indent=1)
log("stamp written; tokens/packed_rows still need runs/count_dir.py")
print(json.dumps(stats["dedup"], indent=1), flush=True)

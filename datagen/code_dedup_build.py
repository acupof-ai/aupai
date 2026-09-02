#!/usr/bin/env python3
# restartable: dedup_ck per-domain sigs checkpoint (reload, never re-hash); the new dir is
# written once; an interrupt mid-write leaves a partial NEW dir (never touches inputs).
"""3b-10: build data/corpus/code_dedup08/ from code_py_starcoder + code_py_rp1t.

fb 2026-09-02 plan: th 0.8 est-cluster (MinHash-J char 5-gram, verified), keep the ordinal
representative per cluster, write a NEW dir (ladder dirs untouched). Stamp carries filters_fp
+ the two inputs' srcfp/fingerprint + dedup params. Tokens before/after measured separately by
datagen/count_cleaned_code.py. 20 deleted docs written beside their representative for an eyeball
near-dup check. CPU, setsid, log to runs/, progress counts docs + bytes.

Pipeline: load per-domain MinHash sigs (dedup_ck) -> LSH band -> pairwise est -> est>=0.80 union-find
clusters -> kept_set = singletons + min-ordinal rep of each cluster -> group kept by (shard,line) and
read each input shard ONCE, write kept lines to data/corpus/code_dedup08/<same shard name> -> stamp
json -> 20-sample (deleted vs representative) side by side.
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TH = 0.80
PERMS = 96
BANDS = 12
BAND_ROWS = PERMS // BANDS


def dom_loc_of(g, dom_bound):
    d = int(np.searchsorted(dom_bound, g, side="right"))
    return d, g - (0 if d == 0 else dom_bound[d - 1])


def read_doc_text(shard, ln):
    with open(shard, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == ln:
                d = json.loads(line)
                return d.get("content", "")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckdir", required=True)
    ap.add_argument("--sample_out", default="")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    dom_sigs, dom_locs = [], []
    for dom in a.inputs:
        S = np.load(os.path.join(a.ckdir, f"{dom}.sig.npy"))
        L = json.load(open(os.path.join(a.ckdir, f"{dom}.loc.json")))
        dom_sigs.append(S)
        dom_locs.append(L)
        print(f"{dom}: {S.shape[0]} sigs cached", flush=True)
    all_sigs = np.vstack(dom_sigs)
    dom_bound = np.cumsum([S.shape[0] for S in dom_sigs])
    n = all_sigs.shape[0]
    t0 = time.perf_counter()

    # LSH band -> candidate pairs -> est >= TH
    buckets = defaultdict(list)
    for g in range(n):
        sig = all_sigs[g]
        for b in range(BANDS):
            buckets[(b, tuple(sig[b * BAND_ROWS:(b + 1) * BAND_ROWS].tolist()))].append(g)
    pairs = set()
    for (b, key), ords in buckets.items():
        if len(ords) < 2:
            continue
        for i in range(len(ords)):
            for j in range(i + 1, len(ords)):
                u, v = ords[i], ords[j]
                pairs.add((u, v) if u < v else (v, u))
    pv = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    est = (all_sigs[pv[:, 0]] == all_sigs[pv[:, 1]]).mean(axis=1)
    e = pv[est >= TH]
    print(f"{len(pairs)} candidates, est>={TH}: {e.shape[0]} edges", flush=True)

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
    for u, v in e:
        union(u, v)
    cl = defaultdict(list)
    for i in range(n):
        cl[find(i)].append(i)
    multi = {r: mem for r, mem in cl.items() if len(mem) > 1}
    deleted = sorted(i for r, mem in multi.items() for i in mem if i != min(mem))
    kept_set = set(range(n)) - set(deleted)
    print(f"{len(multi)} clusters >=2; delete {len(deleted)}/{n} ({len(deleted)/max(1,n):.4f})", flush=True)

    # group kept by shard (per dom_loc), so each input shard is read once
    kept_by_shard = defaultdict(set)
    for g in kept_set:
        d, local = dom_loc_of(g, dom_bound)
        shard, ln = dom_locs[d][local]
        kept_by_shard[shard].add(ln)

    # write kept lines, one read per shard
    bytes_w = shards_written = 0
    for shard, keep_lines in kept_by_shard.items():
        target = os.path.join(a.out_dir, os.path.basename(shard))
        with open(shard, encoding="utf-8") as fsrc, open(target, "w", encoding="utf-8") as fdst:
            for i, line in enumerate(fsrc):
                if i in keep_lines and line.strip():
                    fdst.write(line)
                    bytes_w += len(line.encode("utf-8", "replace"))
        shards_written += 1
        if shards_written % 20 == 0:
            rr = round(time.perf_counter() - t0)
            print(f"  wrote {shards_written} shards, {bytes_w/1e6:.0f} MB ({rr // 60}m{rr % 60}s)", flush=True)

    # 20-sample: deleted doc vs its cluster representative
    if a.sample_out and multi:
        rng = random.Random(11)
        chosen = rng.sample(list(multi.keys()), min(20, len(multi)))
        with open(a.sample_out, "w", encoding="utf-8") as f:
            for r in chosen:
                mem = multi[r]
                rep = min(mem)
                victim = rng.choice([m for m in mem if m != rep])
                rd, rl = dom_loc_of(rep, dom_bound)
                vd, vl = dom_loc_of(victim, dom_bound)
                f.write(f"=== cluster rep ord {rep} vs deleted ord {victim} ===\n")
                f.write("DELETED: " + read_doc_text(dom_locs[vd][vl][0], dom_locs[vd][vl][1])[:300].replace("\n", " ↳ ") + "\n")
                f.write("REP: " + read_doc_text(dom_locs[rd][rl][0], dom_locs[rd][rl][1])[:300].replace("\n", " ↳ ") + "\n---\n")
        print(f"sample -> {a.sample_out}", flush=True)

    # stamp
    stamps = {}
    for dom in a.inputs:
        p = os.path.join(a.root, dom, "build_corpus_stats.json")
        if os.path.exists(p):
            s = json.load(open(p))
            stamps[dom] = {"srcfp": s.get("fingerprint"), "filters": s.get("filters"),
                           "filters_fp": s.get("filters_fp")}
    stats = {"domain": os.path.basename(a.out_dir),
             "inputs": stamps,
             "dedup": {"threshold": TH, "method": "MinHash-J char 5-gram", "n_perm": PERMS, "bands": BANDS,
                       "docs_in": n, "docs_kept": len(kept_set), "docs_deleted": len(deleted),
                       "clusters_gt1": len(multi),
                       "est_note": "est verified est≈exact on char 5-gram (dq.near_dedup_code_rate)"},
             "filters": "code_dedup08-th0.8-ordinal-rep", "n_shards": shards_written}
    with open(os.path.join(a.out_dir, "build_corpus_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    rr = round(time.perf_counter() - t0)
    print(json.dumps({"docs_in": n, "docs_deleted": len(deleted), "docs_kept": len(kept_set),
                      "expected_rate_th08": 0.0418, "bytes_written_MB": round(bytes_w / 1e6, 1)}, ensure_ascii=False), flush=True)
    print(f"DONE in {rr // 60}m{rr % 60}s -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
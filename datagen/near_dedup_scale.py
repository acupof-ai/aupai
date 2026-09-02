#!/usr/bin/env python3
# restartable: signatures checkpoint per domain (.sig.npy); an interrupt costs only
# re-band since already-signed domains are reloaded, never re-hashed.
"""3b-8 FULL (multi-core): near-dedup report across BOTH pre-train code domains.

fb 2026-09-02 conditions: Pool<=16 + nice -n 10 + CUDA_VISIBLE_DEVICES='' (three GPU jobs'
host dataloaders must not starve); one progress line every N docs (counts only, byte-capped
log); signatures saved per domain (.sig.npy) so a second run does not re-hash; report states
MinHash/LSH params (n_perm, band), full-vs-sample, and gives near-dup rate +- a CI.

Pipeline: stream each domain with a 16-proc Pool -> MinHash signature per doc -> write
<domain>.sig.npy (int32 (n_docs, n_perm)) and <domain>.loc.json. Rebuild = load sigs.
All-domain LSH banding (16 bands x rows) -> candidate pairs -> exact Jaccard (relocate 2 docs
by domain/shard/line) -> thresholds 0.7/0.8/0.9 -> report. Read-only, never writes corpus.
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_corpus as B  # noqa: E402
import near_dedup_postpass as ND  # noqa: E402

THRESHOLDS = (0.7, 0.8, 0.9)
PERMS = 96
BANDS = 12
BAND_ROWS = PERMS // BANDS


def sig_one(shard_path):
    """One worker: read a shard, return (sigs int32(n,PERMS), locs [(shard,line)], docids, content_cache).
    content_cache only returns for the (few) hits — but Jaccard relocation needs original text,
    so we ALSO stash every doc's text? No: too big. We relocate by re-reading later; this fn only signs."""
    lsh = B.MinHashLSH(perms=PERMS, bands=BANDS)
    sigs, locs = [], []
    for ln, line in enumerate(open(shard_path, encoding="utf-8")):
        if not line.strip():
            continue
        try:
            content = json.loads(line).get("content", "")
        except Exception:
            continue
        if not content:
            continue
        sig = np.asarray(lsh.signature(content), dtype=np.int64)
        sigs.append(sig)
        locs.append((shard_path, ln))
    return np.stack(sigs) if sigs else np.zeros((0, PERMS), np.int64), locs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckdir", required=True, help="dir for per-domain .sig.npy/.loc.json checkpoints")
    ap.add_argument("--pool", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.ckdir, exist_ok=True)

    BUCKETS = defaultdict(list)
    dom_sigs, dom_locs = [], []  # per-domain concatenated
    dom_shards = {dom: sorted(glob.glob(os.path.join(a.root, dom, "*.jsonl"))) for dom in a.domains}
    doc_count = 0
    t0 = time.perf_counter()

    for dom in a.domains:
        sig_path = os.path.join(a.ckdir, f"{dom}.sig.npy")
        loc_path = os.path.join(a.ckdir, f"{dom}.loc.json")
        if os.path.exists(sig_path) and os.path.exists(loc_path):
            S = np.load(sig_path)
            L = json.load(open(loc_path))
            print(f"{dom}: cached {S.shape[0]} sigs (skip re-hash)", flush=True)
        else:
            rd = round(time.perf_counter() - t0)
            print(f"{dom}: signing {len(dom_shards[dom])} shards (elapsed {rd // 60}m{rd % 60}s)", flush=True)
            shard_list = dom_shards[dom]
            with Pool(a.pool) as pool:
                parts = []
                done = 0
                for i, (sigs, locs) in enumerate(pool.imap_unordered(sig_one, shard_list, chunksize=1)):
                    parts.append((sigs, locs))
                    done += len(locs)
                    if done % 200000 < 4000:
                        rr = round(time.perf_counter() - t0)
                        print(f"  {dom}: {done} docs signed ({rr}s, e{round(rr / max(1, done) * 1000000)} s/100k)", flush=True)
            if not parts:
                S = np.zeros((0, PERMS), np.int64); L = []
            else:
                S = np.vstack([p[0] for p in parts])
                L = [x for p in parts for x in p[1]]
            np.save(sig_path, S)
            json.dump(L, open(loc_path, "w"))
            print(f"{dom}: {S.shape[0]} sigs -> {sig_path}", flush=True)
        dom_sigs.append(S)
        dom_locs.append(L)
        doc_count += S.shape[0]

    # LSH banding across ALL domains (cross-domain is the point)
    # build global arrays
    all_sigs = np.vstack(dom_sigs)
    # ordinal -> (dom_idx, local_i)
    dom_bound = np.cumsum([S.shape[0] for S in dom_sigs])
    for g in range(all_sigs.shape[0]):
        sig = all_sigs[g]
        for b in range(BANDS):
            key = tuple(sig[b * BAND_ROWS:(b + 1) * BAND_ROWS].tolist())
            BUCKETS[(b, key)].append(g)
    rr = round(time.perf_counter() - t0)
    print(f"banded {all_sigs.shape[0]} docs into {len(BUCKETS)} buckets (elapsed {rr // 60}m{rr % 60}s)", flush=True)

    def dom_loc(g):
        d = int(np.searchsorted(dom_bound, g, side="right"))
        local = g - (0 if d == 0 else dom_bound[d - 1])
        return d, local

    # candidate pairs
    pairs = set()
    for (b, key), ords in BUCKETS.items():
        if len(ords) < 2:
            continue
        for i in range(len(ords)):
            for j in range(i + 1, len(ords)):
                u, v = ords[i], ords[j]
                pairs.add((u, v) if u < v else (v, u))
    print(f"{len(pairs)} candidate pairs", flush=True)

    # exact Jaccard by relocating the two docs' text
    cache = {}
    def text_of(g):
        if g in cache:
            return cache[g]
        d, local = dom_loc(g)
        shard, ln = dom_locs[d][local]
        with open(shard, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == ln:
                    cache[g] = json.loads(line).get("content", "")
                    return cache[g]
        return ""

    def jac(g, h):
        return ND.jaccard(ND.word_shingles(ND.normalise_code(text_of(g))),
                          ND.word_shingles(ND.normalise_code(text_of(h))))

    scored = {}
    for u, v in pairs:
        scored[(u, v)] = jac(u, v)

    n = all_sigs.shape[0]
    out = {"domains": a.domains, "docs_total": n, "sample": "FULL (all shards)",
           "method": f"MinHash-LSH n_perm={PERMS} bands={BANDS}({BAND_ROWS}/band) + exact Jaccard",
           "candidate_pairs": len(pairs), "thresholds": {}}
    for th in THRESHOLDS:
        e = [(u, v, j) for (u, v), j in scored.items() if j >= th]
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
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
        rate = docs_in / max(1, n)
        # Wilson 95% CI on the doc-level near-dup rate
        z = 1.96
        denom = 1 + z * z / n
        p_hat = (rate + z * z / (2 * n)) / denom
        half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denom
        cross = sum(1 for u, v, _ in e if dom_loc(u)[0] != dom_loc(v)[0])
        samples = sorted(e, key=lambda x: -x[2])[:20]
        out["thresholds"][str(th)] = {
            "pairs": len(e), "docs_in_clustered_pairs": docs_in,
            "near_dup_rate": round(rate, 6), "rate_ci95": [round(p_hat - half, 6), round(p_hat + half, 6)],
            "clusters": len(multi), "cross_domain_pairs": cross,
            "cross_domain_share": round(cross / max(1, len(e)), 4),
            "sample_pairs": [{"a": dom_locs[dom_loc(u)[0]][dom_loc(u)[1]][0] + ":" + str(dom_locs[dom_loc(u)[0]][dom_loc(u)[1]][1]),
                              "b": dom_locs[dom_loc(v)[0]][dom_loc(v)[1]][0] + ":" + str(dom_locs[dom_loc(v)[0]][dom_loc(v)[1]][1]),
                              "j": round(j, 3)} for u, v, j in samples],
        }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    rr = round(time.perf_counter() - t0)
    print(json.dumps(out, ensure_ascii=False)[:900], flush=True)
    print(f"DONE in {rr // 60}m{rr % 60}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deterministically reproduce the code_dedup08 clusters and emit a stratified
hand-read sheet for the aupai-6e ②-first ruling (2026-09-03): judge whether, in
cross-domain clusters, the code_py_rp1t member is a genuine near-duplicate of the
code_py_starcoder ORDINAL representative.

Reuses near_dedup_scale's signing+clustering (same params: TH 0.8, PERMS 96,
BANDS 12, char 5-gram) so the reproduction is the build's, not a new measure.
The code_dedup08 ckdir was cleaned, so sigs re-hash here (deterministic, few min).

Outcome is judged by a human reviewer OFF the sheet; this script only persists
cluster membership + picks the stratified sample. Criterion (aupai-6e, verbatim,
written BEFORE the read): near-duplicate = same code modulo whitespace, identifiers,
or comments; different program = not. If the rp1t member is a genuine duplicate of
the starcoder representative in >= 80% of mixed clusters, code_dedup08 stands;
below that, rerun with a domain-fair representative (prefer the rp1t member).

    python datagen/code_dedup_handread.py --root data/corpus \
        --domains code_py_starcoder code_py_rp1t --ckdir runs/code_dedup08_ck \
        --rep math 40 --n_rep 100 --out runs/code_dedup_handread_sheet.json
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import near_dedup_scale as ND  # noqa: E402

TH = 0.80  # code_dedup08 was th0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--ckdir", required=True)
    ap.add_argument("--n_rp1t_clusters", type=int, default=40)
    ap.add_argument("--n_total_clusters", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.ckdir, exist_ok=True)
    rng = random.Random(5)

    # Stage A signatures (re-hash if cache absent, else reload) via near_dedup_scale
    dom_sigs, dom_locs, dom_shards = [], [], {}
    for dom in a.domains:
        sig_path = os.path.join(a.ckdir, f"{dom}.sig.npy")
        loc_path = os.path.join(a.ckdir, f"{dom}.loc.json")
        if os.path.exists(sig_path) and os.path.exists(loc_path):
            S = np.load(sig_path); L = json.load(open(loc_path))
            print(f"{dom}: cached {S.shape[0]} sigs", flush=True)
        else:
            # replicate near_dedup_scale's signing block
            import glob
            from multiprocessing import Pool
            shards = sorted(glob.glob(os.path.join(a.root, dom, "*.jsonl")))
            parts, done = [], 0
            print(f"{dom}: signing {len(shards)} shards", flush=True)
            with Pool(16) as pool:
                for sigs, locs in pool.imap_unordered(ND.sig_one, shards, chunksize=1):
                    parts.append((sigs, locs)); done += len(locs); print(f"  {done} signed", flush=True)
            S = np.vstack([p[0] for p in parts]) if parts else np.zeros((0, ND.PERMS), np.int64)
            L = [x for p in parts for x in p[1]]
            np.save(sig_path, S); json.dump(L, open(loc_path, "w"))
        dom_sigs.append(S); dom_locs.append(L); dom_shards[dom] = True

    all_sigs = np.vstack(dom_sigs)
    n = all_sigs.shape[0]
    dom_bound = np.cumsum([S.shape[0] for S in dom_sigs])

    def dom_of(g):
        d = int(np.searchsorted(dom_bound, g, side="right"))
        return a.domains[d], g - (0 if d == 0 else dom_bound[d - 1])

    # Stage B: LSH band -> candidate pairs -> est>=TH edges -> union-find clusters
    BUCKETS = defaultdict(list)
    for g in range(n):
        sig = all_sigs[g]
        for b in range(ND.BANDS):
            BUCKETS[(b, tuple(sig[b * ND.BAND_ROWS:(b + 1) * ND.BAND_ROWS].tolist()))].append(g)
    pairs = set()
    for ords in BUCKETS.values():
        if len(ords) > 1:
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
            parent[x] = parent[parent[x]]; x = parent[x]
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
    multi = {r: [i for i in mem if len(mem) > 1] for r, mem in cl.items() if len(mem) > 1}

    # clusters with >=1 rp1t member, and the all-starcoder ones (for blinding)
    mixed = {r: mem for r, mem in multi.items() if any(dom_of(g)[0] == a.domains[1] for g in mem)}
    pure_star = {r: mem for r, mem in multi.items() if all(dom_of(g)[0] == a.domains[0] for g in mem)}
    print(f"clusters>=2: {len(multi)} | with rp1t member: {len(mixed)} | pure-starcoder: {len(pure_star)}", flush=True)
    rp1t_members = sum(1 for mem in multi.values() for g in mem if dom_of(g)[0] == a.domains[1])
    print(f"total rp1t docs in clusters: {rp1t_members}", flush=True)

    # ordinal rep per cluster = min global index (code_dedup08 kept min ordinal)
    rng.shuffle(list(mixed.keys()))
    rp1t_sample = list(mixed.keys())[: a.n_rp1t_clusters]
    star_sample = rng.sample(sorted(pure_star), max(0, a.n_total_clusters - a.n_rp1t_clusters))
    chosen = rp1t_sample + star_sample

    def doc(g):
        d, local = dom_of(g)
        shard, ln = dom_locs[d][local]
        with open(shard, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == ln:
                    return json.loads(line).get("content", "")
        return ""

    sheet = {"threshold": TH, "domains": a.domains,
             "criterion": "near-duplicate = same code modulo whitespace, identifiers, or comments; different program = not (aupai-6e, written before reading)",
             "rep_selection": "ordinal min global index (code_dedup08 kept min ordinal)",
             "n_mixed_clusters": len(mixed), "n_rp1t_members_in_clusters": rp1t_members,
             "sample": []}
    for r in chosen:
        mem = multi[r]
        rep = min(mem)
        rep_dom, _ = dom_of(rep)
        rp1t = [g for g in mem if dom_of(g)[0] == a.domains[1]]
        if rep_dom == a.domains[1]:
            rep_dom = "rp1t(rep)"  # note if the rep itself is rp1t
        sheet["sample"].append({
            "cluster_ord": int(r), "rep_ord": int(rep), "rep_domain": rep_dom,
            "n_members": len(mem), "rp1t_members": [int(g) for g in rp1t],
            "mixed": bool(rp1t) and any(dom_of(g)[0] == a.domains[0] for g in mem),
            "rep_text_excerpt": doc(rep)[:400], "member_text_excerpt": doc(rp1t[0])[:400],
        })
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(sheet, f, ensure_ascii=False, indent=1)
    print(f"hand-read sheet ({len(chosen)} clusters, {len(rp1t_sample)} with rp1t) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
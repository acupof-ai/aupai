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

    python3 datagen/code_dedup_handread.py --root data/corpus \
        --domains code_py_starcoder code_py_rp1t --ckdir runs/code_dedup08_ck \
        --n_rp1t_clusters 40 --n_total_clusters 100 \
        --out runs/code_dedup_handread_sheet.json

`--rep math 40 --n_rep 100` stood here until 2026-09-08 and does not parse: the parser
has never had either flag. doc_commands_exist checks that a cited FILE exists, not that
a cited command's flags are accepted, so a documented invocation can be wrong for as
long as nobody types it -- this one was, and the run that needed it lost the time to
argparse's error. Verified by `--help` before this edit, not by reading the parser.
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


def cluster_row(ord_, mem, dom_of, doc, domains):
    """One hand-read sheet row for one cluster. Extracted so it can be tested.

    A PURE-STARCODER cluster has no rp1t member, and the previous inline version indexed
    rp1t[0] unconditionally -- IndexError on all 60 of them, so the 100-cluster sheet the
    acceptance asks for could not be produced at all by the tool written for it. Such a
    cluster is not a defect and is not dropped: it is the negative control for the
    asymmetry b0 measured (rp1t deleted 33.02% against starcoder 1.32%), the case where no
    cross-domain choice was made.
    """
    rep = min(mem)
    rep_dom, _ = dom_of(rep)
    rp1t = [g for g in mem if dom_of(g)[0] == domains[1]]
    if rep_dom == domains[1]:
        rep_dom = "rp1t(rep)"  # note if the rep itself is rp1t
    return {
        "cluster_ord": int(ord_),
        "rep_ord": int(rep),
        "rep_domain": rep_dom,
        "n_members": len(mem),
        "rp1t_members": [int(g) for g in rp1t],
        "mixed": bool(rp1t) and any(dom_of(g)[0] == domains[0] for g in mem),
        "rep_text_excerpt": doc(rep)[:400],
        "member_text_excerpt": doc(rp1t[0])[:400] if rp1t else None,
        # The judgement 4c's 2026-09-06 ruling asks the reader for: is the kept starcoder
        # copy the SAME FILE as the dropped rp1t one? That decides whether the 33.02% rp1t
        # deletion rate is duplicate removal or supply loss. Blank until a human fills it.
        "same_file": None,
    }


def draw_mixed(mixed, n, rng):
    """n mixed clusters, drawn rather than sliced -- and the assertion lives HERE, not in the
    caller, so the selftest exercises the production path instead of a copy of it.

    `rng.shuffle(list(mixed.keys()))` stood in the caller until 2026-09-08 and was a NO-OP:
    shuffle mutates in place and the list it was handed was discarded on the same line, so the
    slice that followed took the LOWEST-ORDINAL n clusters. Ordinal is position in the corpus
    scan, so that was a contiguous slice of the first shards, not a sample of the stratum -- and
    the mixed stratum is the arm #70's whole question rests on.

    The first fix asserted the property in the caller and re-tested both spellings on a synthetic
    pool inside the selftest. The mutant SURVIVED: restoring the no-op in the caller left the
    selftest green, because the selftest was checking its own copy of the logic. Hence one
    function, called by both.
    """
    keys = sorted(mixed)
    got = rng.sample(keys, min(n, len(keys)))
    if len(keys) > n:
        # A no-op shuffle is indistinguishable from a working draw by the output's SIZE -- both
        # return n. The property that separates them: a random draw cannot be the pool's first n
        # in order.
        if got == keys[:n]:
            raise SystemExit(
                f"REFUSE: the mixed draw of {n} equals the {n} lowest-ordinal clusters exactly. "
                f"That is corpus position, not a sample; the 2026-09-08 no-op shuffle is back."
            )
    return got


def _selftest():
    """The pure-starcoder cluster must produce a row, not an IndexError.

    Both directions: a mixed cluster still carries the rp1t excerpt (the fix must not
    blank the field that the hand read actually reads), and a pure cluster produces a row
    with member_text_excerpt None and mixed False.
    """
    fails = []
    doms = ["code_py_starcoder", "code_rp1t"]
    world = {
        0: ("code_py_starcoder", "sc0"),
        1: ("code_py_starcoder", "sc1"),
        2: ("code_rp1t", "rp2"),
        3: ("code_rp1t", "rp3"),
    }

    def dom_of(g):
        return world[g]

    def doc(g):
        return f"text-of-{world[g][1]}"

    pure = cluster_row(7, [0, 1], dom_of, doc, doms)
    if pure["member_text_excerpt"] is not None:
        fails.append("pure-starcoder cluster invented an rp1t excerpt")
    if pure["mixed"] or pure["rp1t_members"]:
        fails.append(f"pure cluster reported mixed={pure['mixed']} members={pure['rp1t_members']}")
    if pure["rep_domain"] != "code_py_starcoder" or pure["rep_ord"] != 0:
        fails.append(f"pure cluster rep wrong: {pure['rep_domain']} {pure['rep_ord']}")

    mixed = cluster_row(8, [1, 2], dom_of, doc, doms)
    if mixed["member_text_excerpt"] != "text-of-rp2":
        fails.append(f"mixed cluster lost the rp1t excerpt: {mixed['member_text_excerpt']!r}")
    if not mixed["mixed"] or mixed["rp1t_members"] != [2]:
        fails.append("mixed cluster not reported as mixed")
    if mixed["rep_ord"] != 1 or mixed["rep_domain"] != "code_py_starcoder":
        fails.append("ordinal-first representative not selected in the mixed cluster")

    # rp1t itself first: the rep IS rp1t, which the sheet must label distinctly or a reader
    # counting "rep_domain == starcoder" undercounts the cases where rp1t won.
    rp_first = cluster_row(9, [2, 3], dom_of, doc, doms)
    if rp_first["rep_domain"] != "rp1t(rep)":
        fails.append(f"rp1t-representative cluster labelled {rp_first['rep_domain']!r}")

    if any(r["same_file"] is not None for r in (pure, mixed, rp_first)):
        fails.append("same_file must start blank; it is the human's judgement")

    # THE DRAW, through draw_mixed itself. The three cluster_row cases above cannot see this
    # defect: they test the SHAPE of one row and the no-op was in WHICH rows get built.
    pool = {k: [k, k + 1] for k in range(200)}
    got = draw_mixed(pool, 40, random.Random(5))
    if len(got) != 40:
        fails.append(f"draw_mixed returned {len(got)}, not 40")
    if got == sorted(pool)[:40]:
        fails.append("draw_mixed returned the pool's first 40 in order -- not a draw")
    if len(set(got)) != len(got):
        fails.append("draw_mixed returned a duplicate cluster; a cluster must be read once")
    if not set(got) <= set(pool):
        fails.append("draw_mixed invented a cluster not in the pool")
    # and the REFUSAL: a pool whose draw would be the first n must raise, not return
    try:
        draw_mixed({k: [k] for k in range(3)}, 3, random.Random(5))
    except SystemExit:
        fails.append("draw_mixed refused when n >= pool size, where every draw IS the whole pool")
    for f in fails:
        print(f"  FAIL {f}", file=sys.stderr)
    if fails:
        print(f"code_dedup_handread selftest: {len(fails)} failure(s)", file=sys.stderr)
        return 1
    print(
        "code_dedup_handread selftest OK: pure-starcoder cluster yields a row with a None "
        "excerpt (the old rp1t[0] raised IndexError on all 60 of them), the mixed cluster "
        "keeps its excerpt, an rp1t-first cluster is labelled rp1t(rep), and same_file is "
        "blank for the reader; and draw_mixed -- the production draw, not a copy of it -- "
        "returns 40 distinct clusters from the pool and never the pool's first 40 in order, "
        "which is the only property that separates a draw from the no-op shuffle it replaced"
    )
    return 0


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
            S = np.load(sig_path)
            L = json.load(open(loc_path))
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
                    parts.append((sigs, locs))
                    done += len(locs)
                    print(f"  {done} signed", flush=True)
            S = np.vstack([p[0] for p in parts]) if parts else np.zeros((0, ND.PERMS), np.int64)
            L = [x for p in parts for x in p[1]]
            np.save(sig_path, S)
            json.dump(L, open(loc_path, "w"))
        dom_sigs.append(S)
        dom_locs.append(L)
        dom_shards[dom] = True

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
            BUCKETS[(b, tuple(sig[b * ND.BAND_ROWS : (b + 1) * ND.BAND_ROWS].tolist()))].append(g)
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
    multi = {r: [i for i in mem if len(mem) > 1] for r, mem in cl.items() if len(mem) > 1}

    # clusters with >=1 rp1t member, and the all-starcoder ones (for blinding)
    mixed = {r: mem for r, mem in multi.items() if any(dom_of(g)[0] == a.domains[1] for g in mem)}
    pure_star = {r: mem for r, mem in multi.items() if all(dom_of(g)[0] == a.domains[0] for g in mem)}
    print(
        f"clusters>=2: {len(multi)} | with rp1t member: {len(mixed)} | pure-starcoder: {len(pure_star)}",
        flush=True,
    )
    rp1t_members = sum(1 for mem in multi.values() for g in mem if dom_of(g)[0] == a.domains[1])
    print(f"total rp1t docs in clusters: {rp1t_members}", flush=True)

    # ordinal rep per cluster = min global index (code_dedup08 kept min ordinal)
    #
    # `rng.shuffle(list(mixed.keys()))` stood here until 2026-09-08 and was a NO-OP: shuffle
    # mutates in place, and the list it was handed is discarded on the same line, so `mixed` was
    # untouched and the slice below took the 40 LOWEST-ORDINAL mixed clusters. Ordinal is position
    # in the corpus scan, so that is a contiguous slice of the first shards, not a sample of the
    # stratum -- and the mixed stratum is the arm the whole question rests on. The intent was
    # visible in the code and the effect was absent, which is why the sample below is asserted
    # rather than assumed. `sorted()` first, because dict order depends on insertion and a seeded
    # draw over an unordered set is not reproducible.
    rp1t_sample = draw_mixed(mixed, a.n_rp1t_clusters, rng)
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

    sheet = {
        "threshold": TH,
        "domains": a.domains,
        "criterion": "near-duplicate = same code modulo whitespace, identifiers, or comments; different program = not (aupai-6e, written before reading)",
        "rep_selection": "ordinal min global index (code_dedup08 kept min ordinal)",
        "n_mixed_clusters": len(mixed),
        "n_rp1t_members_in_clusters": rp1t_members,
        "sample": [],
    }
    for r in chosen:
        sheet["sample"].append(cluster_row(r, multi[r], dom_of, doc, a.domains))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(sheet, f, ensure_ascii=False, indent=1)
    print(f"hand-read sheet ({len(chosen)} clusters, {len(rp1t_sample)} with rp1t) -> {a.out}", flush=True)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    main()

#!/usr/bin/env python3
# restartable: writes the sheet once at the end from an in-memory sample; an interrupt
# leaves no partial sheet and nothing is consumed downstream. Re-run costs one LSH pass.
"""3b-10 #70: stratified hand-read sheet for rp1t-vs-starcoder near-dup clusters.

runs/3b8_code_dedup.json says cross-domain edges are 9.89% / 11.96% / 25.06% of all
edges at th 0.7 / 0.8 / 0.9 over code_py_starcoder + code_py_rp1t. That is an EDGE
count over documents and it cannot answer the question the mix asks -- whether a
starcoder top-up on top of code_rp1t_dd09 buys unseen tokens. Two gaps: an edge is
not a document (a cluster of k documents has up to k(k-1)/2 edges, so a few large
clusters inflate the share), and a document is not a token (the removed documents in
code_rp1t are 2.1106x mean length, cs.code_rp1t_near_dup_rate).

This writes the sheet a human reads: N clusters drawn stratified over the est band,
each printed with its cross-domain membership and both documents' heads, so the reader
can rule on whether a cross-domain pair is the SAME FILE (a top-up buys nothing there)
or two files that a 5-gram MinHash calls similar (boilerplate, a license header, a
generated __init__). The ruling is the reader's; this only builds the sheet.

Each pair also carries WHICH SIDE SURVIVES a dedup (62, 2026-09-07). code_dedup_build
keeps the minimum ordinal, and ordinal order follows the domain concatenation order, not
the corpus you are trying to top up -- so a 25% cross-source edge share means a top-up
loses almost nothing if the kept representative is usually the starcoder side, and most
of the gain if it is not. Without that column the sheet returns a token fraction that
still cannot be converted into a supply figure, which is the shape of an answer that
does not answer. The sheet prints the kept side per pair and the tally at the top.

    python3 datagen/build_xsource_handread.py --ckdir runs/owm_dedup_ck \
        --domains code_py_starcoder code_py_rp1t --th 0.8 --n 100 \
        --out runs/xsource_handread_th08.txt
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

PERMS = 96
BANDS = 12
BAND_ROWS = PERMS // BANDS


def dom_loc_of(g, dom_bound):
    d = int(np.searchsorted(dom_bound, g, side="right"))
    return d, g - (0 if d == 0 else dom_bound[d - 1])


def read_doc(shard, ln):
    with open(shard, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == ln:
                return json.loads(line)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckdir", required=True)
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--th", type=float, default=0.8)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--bands_sample", type=int, default=0,
                    help="if >0, hash about this many ordinals drawn PROPORTIONALLY from each "
                         "domain. Never a prefix: ordinals run in domain order, so the first N "
                         "of a two-domain concatenation are one domain and cross-domain is 0 by "
                         "construction -- measured 2026-09-07, the first version of this flag "
                         "printed 0 cross of 178 pairs and the 0 meant nothing")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=70)
    a = ap.parse_args()

    sigs, locs = [], []
    for dom in a.domains:
        S = np.load(os.path.join(a.ckdir, f"{dom}.sig.npy"), mmap_mode="r")
        L = json.load(open(os.path.join(a.ckdir, f"{dom}.loc.json")))
        sigs.append(S)
        locs.append(L)
        print(f"{dom}: {S.shape[0]} sigs", flush=True)
    dom_bound = np.cumsum([S.shape[0] for S in sigs])
    n = int(dom_bound[-1])

    if a.bands_sample and a.bands_sample < n:
        # Proportional per domain, not a prefix. dom_bound is cumulative, so ordinals
        # [0, dom_bound[0]) are all domain 0: a prefix of a two-domain concatenation
        # contains one domain and reports 0 cross-domain pairs for arithmetic reasons.
        rng0 = random.Random(a.seed)
        pool = []
        lo = 0
        for d, hi in enumerate(dom_bound):
            share = max(1, int(a.bands_sample * (int(hi) - lo) / n))
            pool.extend(rng0.sample(range(lo, int(hi)), min(share, int(hi) - lo)))
            lo = int(hi)
        ordinals = sorted(pool)
        print(f"  sampling {len(ordinals)} of {n} ordinals, proportional per domain", flush=True)
    else:
        ordinals = range(n)

    buckets = defaultdict(list)
    for count, g in enumerate(ordinals):
        d, local = dom_loc_of(g, dom_bound)
        sig = np.asarray(sigs[d][local])
        for b in range(BANDS):
            buckets[(b, tuple(sig[b * BAND_ROWS:(b + 1) * BAND_ROWS].tolist()))].append(g)
        if count and count % 500000 == 0:
            print(f"  banded {count}", flush=True)

    pairs = set()
    for _key, ords in buckets.items():
        if len(ords) < 2 or len(ords) > 200:
            continue
        for i in range(len(ords)):
            for j in range(i + 1, len(ords)):
                u, v = ords[i], ords[j]
                pairs.add((u, v) if u < v else (v, u))
    print(f"{len(pairs)} candidate pairs", flush=True)
    if not pairs:
        print("no candidate pairs -- nothing to hand-read", file=sys.stderr)
        return 2

    # est per pair, keep those at or above the threshold, and split cross vs within.
    rng = random.Random(a.seed)
    scored = []
    for u, v in pairs:
        du, lu = dom_loc_of(u, dom_bound)
        dv, lv = dom_loc_of(v, dom_bound)
        est = float((np.asarray(sigs[du][lu]) == np.asarray(sigs[dv][lv])).mean())
        if est >= a.th:
            scored.append((u, v, est, du != dv))
    cross = [s for s in scored if s[3]]
    within = [s for s in scored if not s[3]]
    print(f"est>={a.th}: {len(scored)} pairs, {len(cross)} cross-domain, {len(within)} within",
          flush=True)
    if not cross:
        # A 0 here is only a measurement if both domains were actually sampled. Say which,
        # because 0-by-construction and 0-by-measurement print the same digit.
        seen = {a.domains[dom_loc_of(g, dom_bound)[0]] for g in ordinals} if a.bands_sample \
            else set(a.domains)
        if len(seen) < 2:
            print(f"REFUSING: only {sorted(seen)} was sampled, so 0 cross-domain pairs is "
                  f"arithmetic, not a measurement. Raise --bands_sample or drop it.",
                  file=sys.stderr)
            return 2
        print(f"no cross-domain pairs at est>={a.th} over {sorted(seen)} -- that is a "
              f"measurement; report it, do not fill the sheet", file=sys.stderr)

    # stratify the cross-domain pairs over the est band so the sheet is not all est=1.0
    bands = defaultdict(list)
    for s in cross:
        bands[round(min(s[2], 0.999) * 10) / 10].append(s)
    picked = []
    keys = sorted(bands)
    per = max(1, a.n // max(1, len(keys)))
    for k in keys:
        rng.shuffle(bands[k])
        picked.extend(bands[k][:per])
    rng.shuffle(picked)
    picked = picked[:a.n]

    # Which side a dedup would keep, over the WHOLE cross-domain population rather than
    # the sample: code_dedup_build keeps the minimum ordinal, and ordinals run in domain
    # concatenation order, so for a cross-domain pair the earlier-listed domain always
    # wins. Counting it makes that visible instead of leaving it to be re-derived.
    kept_tally = defaultdict(int)
    for u, v, _est, _x in cross:
        keep = min(u, v)
        kd, _ = dom_loc_of(keep, dom_bound)
        kept_tally[a.domains[kd]] += 1

    with open(a.out, "w", encoding="utf-8") as f:
        f.write(f"# {len(picked)} cross-domain pairs at est>={a.th}, stratified over the est band\n")
        f.write(f"# domains: {a.domains}; population: {len(cross)} cross of {len(scored)} pairs\n")
        f.write(f"# kept side over all {len(cross)} cross-domain pairs (min ordinal wins): "
                f"{dict(kept_tally)}\n")
        f.write("# The kept side follows ordinal order, i.e. the order --domains was given,\n"
                "# NOT which corpus is being topped up. Read it before converting any rate\n"
                "# into a supply figure: a cross-source duplicate costs the top-up only when\n"
                "# the SURVIVING copy is the one already in the mix.\n")
        f.write("# RULE EACH: SAME (the same file) | SIMILAR (boilerplate/generated) | DIFFERENT\n\n")
        for u, v, est, _ in picked:
            du, lu = dom_loc_of(u, dom_bound)
            dv, lv = dom_loc_of(v, dom_bound)
            sa, la = locs[du][lu]
            sb, lb = locs[dv][lv]
            da, db = read_doc(sa, la), read_doc(sb, lb)
            kd, _ = dom_loc_of(min(u, v), dom_bound)
            f.write(f"=== est {est:.3f} | {a.domains[du]} vs {a.domains[dv]} "
                    f"| dedup keeps: {a.domains[kd]} ===\n")
            f.write(f"A {sa}:{la}  ({len(da.get('content', ''))} chars)\n")
            f.write((da.get("content", "")[:600]).replace("\n", " | ") + "\n")
            f.write(f"B {sb}:{lb}  ({len(db.get('content', ''))} chars)\n")
            f.write((db.get("content", "")[:600]).replace("\n", " | ") + "\n")
            f.write("RULING: \n\n")
    print(f"sheet -> {a.out} ({len(picked)} pairs of {len(cross)} cross-domain); "
          f"kept side {dict(kept_tally)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

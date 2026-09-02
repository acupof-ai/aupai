#!/usr/bin/env python3
# restartable: signatures checkpoint per domain (.sig.npy); re-run reloads, never re-hashes.
"""3b-8 FULL (v5): near-dedup report over BOTH pre-train code domains.

fb rule (2026-09-02): near-dup rate by MinHash-J ESTIMATE (est>=th clustering, Wilson CI);
exact Jaccard only on a RANDOM SAMPLE of est>=0.85 top pairs to verify est≈J (sample size + its
own CI in the report); the near-threshold band (est in [0.60,0.85), 2,978,812 pairs) is reported
as a morphological finding, not exact-ranged. Method column: "MinHash-J estimate + top-pair
exact verify". The 181-min exact-abort decision is kept (fb approved).

Stage A: 16-proc Pool streaming -> MinHash signature -> <domain>.sig.npy + .loc.json (checkpoint).
Stage B: all-domain LSH banding -> candidate pairs -> MinHash-J estimate est = elementwise sig
equality. Per threshold th in {0.7,0.8,0.9}: cluster docs on est>=th edges; rate = clustered/total
with Wilson 95% CI; cross-domain share; 20 sample pairs. Verify: exact J on a fixed-seed random
sample of est>=0.85 pairs; report sample n, mean(est-J), and a CI on that. Read-only.
"""
import argparse
import glob
import json
import os
import random
import sys
import time
from collections import OrderedDict, defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_corpus as B  # noqa: E402
import near_dedup_postpass as ND  # noqa: E402

THRESHOLDS = (0.7, 0.8, 0.9)
PERMS = 96
BANDS = 12
BAND_ROWS = PERMS // BANDS
BAND_LO = 0.60
BAND_HI = 0.85
VERIFY_N = 200  # top-pair exact-verify sample cap

_GB = None
_GL = None
_SHL = OrderedDict()
_SHL_MAX = 20000


def sig_one(shard_path):
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
        sigs.append(np.asarray(lsh.signature(content), dtype=np.int64))
        locs.append((shard_path, ln))
    return (np.stack(sigs) if sigs else np.zeros((0, PERMS), np.int64)), locs


def _read_doc(g):
    d = int(np.searchsorted(_GB, g, side="right"))
    local = g - (0 if d == 0 else _GB[d - 1])
    shard, ln = _GL[d][local]
    with open(shard, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == ln:
                return json.loads(line).get("content", "")
    return ""


def _shingles(g):
    if g in _SHL:
        _SHL.move_to_end(g)
        return _SHL[g]
    sh = frozenset(ND.word_shingles(ND.normalise_code(_read_doc(g))))
    _SHL[g] = sh
    if len(_SHL) > _SHL_MAX:
        _SHL.popitem(last=False)
    return sh


def _init_verify(g_dom_bound, g_dom_locs):
    global _GB, _GL
    _GB, _GL = g_dom_bound, g_dom_locs
    _SHL.clear()


def _exact_jac(pair):
    return ND.jaccard(_shingles(pair[0]), _shingles(pair[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckdir", required=True)
    ap.add_argument("--pool", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.ckdir, exist_ok=True)

    BUCKETS = defaultdict(list)
    dom_sigs, dom_locs = [], []
    dom_shards = {dom: sorted(glob.glob(os.path.join(a.root, dom, "*.jsonl"))) for dom in a.domains}
    t0 = time.perf_counter()
    for dom in a.domains:
        sig_path = os.path.join(a.ckdir, f"{dom}.sig.npy")
        loc_path = os.path.join(a.ckdir, f"{dom}.loc.json")
        if os.path.exists(sig_path) and os.path.exists(loc_path):
            S = np.load(sig_path)
            L = json.load(open(loc_path))
            print(f"{dom}: cached {S.shape[0]} sigs (skip re-hash)", flush=True)
        else:
            parts, done = [], 0
            print(f"{dom}: signing {len(dom_shards[dom])} shards", flush=True)
            with Pool(a.pool) as pool:
                for sigs, locs in pool.imap_unordered(sig_one, dom_shards[dom], chunksize=1):
                    parts.append((sigs, locs))
                    done += len(locs)
                    if done % 200000 < 4000:
                        rr = round(time.perf_counter() - t0)
                        print(f"  {dom}: {done} signed ({rr}s)", flush=True)
            S = np.vstack([p[0] for p in parts]) if parts else np.zeros((0, PERMS), np.int64)
            L = [x for p in parts for x in p[1]]
            np.save(sig_path, S)
            json.dump(L, open(loc_path, "w"))
            print(f"{dom}: {S.shape[0]} sigs -> {sig_path}", flush=True)
        dom_sigs.append(S)
        dom_locs.append(L)

    all_sigs = np.vstack(dom_sigs)
    n = all_sigs.shape[0]
    dom_bound = np.cumsum([S.shape[0] for S in dom_sigs])
    for g in range(all_sigs.shape[0]):
        sig = all_sigs[g]
        for b in range(BANDS):
            BUCKETS[(b, tuple(sig[b * BAND_ROWS:(b + 1) * BAND_ROWS].tolist()))].append(g)
    rr = round(time.perf_counter() - t0)
    print(f"banded {n} docs into {len(BUCKETS)} buckets ({rr // 60}m{rr % 60}s)", flush=True)

    pairs = set()
    for (b, key), ords in BUCKETS.items():
        if len(ords) < 2:
            continue
        for i in range(len(ords)):
            for j in range(i + 1, len(ords)):
                u, v = ords[i], ords[j]
                pairs.add((u, v) if u < v else (v, u))
    pv = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    est = (all_sigs[pv[:, 0]] == all_sigs[pv[:, 1]]).mean(axis=1)
    rr = round(time.perf_counter() - t0)
    print(f"{len(pairs)} candidates; est band [{BAND_LO},{BAND_HI}): {int(((est >= BAND_LO) & (est < BAND_HI)).sum())} pairs ({rr // 60}m{rr % 60}s)", flush=True)

    def dom_loc(g):
        d = int(np.searchsorted(dom_bound, g, side="right"))
        return d, g - (0 if d == 0 else dom_bound[d - 1])

    out = {"domains": a.domains, "docs_total": n, "sample": "FULL (all shards)",
           "method": "MinHash-J estimate + top-pair exact verify (n_perm=%d, bands=%d(%d/band)); est=MinHash-J" % (PERMS, BANDS, BAND_ROWS),
           "candidate_pairs": len(pairs),
           "near_threshold_band": {"range": [BAND_LO, BAND_HI], "pairs": int(((est >= BAND_LO) & (est < BAND_HI)).sum())},
           "thresholds": {}}
    for th in THRESHOLDS:
        emask = est >= th
        edges = pv[emask]
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
        for u, v in edges:
            union(u, v)
        cl = defaultdict(list)
        for i in range(n):
            cl[find(i)].append(i)
        multi = {r: mem for r, mem in cl.items() if len(mem) > 1}
        docs_in = sum(len(m) for m in multi.values())
        rate = docs_in / max(1, n)
        z = 1.96
        denom = 1 + z * z / n
        p_hat = (rate + z * z / (2 * n)) / denom
        half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denom
        cross = sum(1 for u, v in edges if dom_loc(u)[0] != dom_loc(v)[0])
        samples = sorted([(int(u), int(v), float(e)) for u, v, e in zip(edges[:, 0], edges[:, 1], est[emask])],
                         key=lambda x: -x[2])[:20]
        out["thresholds"][str(th)] = {
            "near_dup_rate": round(rate, 6), "rate_ci95": [round(p_hat - half, 6), round(p_hat + half, 6)],
            "docs_in_clustered_pairs": docs_in, "edges": int(emask.sum()), "clusters": len(multi),
            "cross_domain_pairs": cross, "cross_domain_share": round(cross / max(1, int(emask.sum())), 4),
            "sample_pairs": [
                {"a": dom_locs[dom_loc(u)[0]][dom_loc(u)[1]][0] + ":" + str(dom_locs[dom_loc(u)[0]][dom_loc(u)[1]][1]),
                 "b": dom_locs[dom_loc(v)[0]][dom_loc(v)[1]][0] + ":" + str(dom_locs[dom_loc(v)[0]][dom_loc(v)[1]][1]),
                 "est": round(e, 3)} for u, v, e in samples],
        }
        rr = round(time.perf_counter() - t0)
        print(f"  est>=th{th}: rate {rate:.5f} (CI {round(p_hat-half,5)},{round(p_hat+half,5)}), {int(emask.sum())} edges ({rr // 60}m{rr % 60}s)", flush=True)

    # exact verify: random sample of est>=BAND_HI top pairs, fixed seed
    topmask = est >= BAND_HI
    top = pv[topmask]
    top_est = est[topmask]
    rng = random.Random(3)
    sample_idx = rng.sample(range(top.shape[0]), min(VERIFY_N, top.shape[0])) if top.shape[0] else []
    verify = {}
    if sample_idx:
        sv = [(int(u), int(v)) for u, v in top[sample_idx].tolist()]
        with Pool(a.pool, initializer=_init_verify, initargs=(dom_bound, dom_locs)) as pool:
            jexact = pool.map(_exact_jac, sv, chunksize=16)
        samp_est = top_est[sample_idx]
        jex = np.asarray(jexact)
        diff = (jex - samp_est)
        mean_diff = float(diff.mean())
        mae = float(np.abs(diff).mean())
        sd = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
        verify = {"sample_n": int(len(sample_idx)), "sample_of": int(top.shape[0]),
                  "mean_est": float(samp_est.mean()), "mean_exact": float(jex.mean()),
                  "mean_diff_exact_minus_est": round(mean_diff, 4), "mae": round(mae, 4),
                  "diff_sd": round(sd, 4), "est_approx_exact": bool(abs(mean_diff) < 0.05)}
        print(f"verify: n={len(sample_idx)} est>={BAND_HI}, mean est {samp_est.mean():.3f} vs exact {jex.mean():.3f}, mae {mae:.3f}", flush=True)
    out["exact_verify"] = verify
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    rr = round(time.perf_counter() - t0)
    print(json.dumps(out, ensure_ascii=False)[:900], flush=True)
    print(f"DONE in {rr // 60}m{rr % 60}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
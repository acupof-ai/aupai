#!/usr/bin/env python3
# restartable: signatures checkpoint per domain (.sig.npy); an interrupt costs only
# re-band + re-verify since signed domains reload, never re-hash.
"""3b-8 FULL (v4): near-dedup report over BOTH pre-train code domains.

fb conditions: Pool<=16 + nice -n 10 + CUDA_VISIBLE_DEVICES=''; per-N-doc progress (counts);
signatures checkpointed per domain; report states MinHash/LSH params + full-vs-sample + rate + CI.

Stage A: stream each domain (16-proc Pool) -> MinHash signature -> <domain>.sig.npy + .loc.json.
Stage B: all-domain LSH banding -> candidate pairs -> coarse MinHash-J estimate (est, elementwise
sig equality, fast) -> exact Jaccard ONLY on est >= EXACT_FL (0.60; est<0.60 means J is below every
report floor with margin, MinHash est std ~0.03) -> report thresholds 0.7/0.8/0.9.
Verify phase is now parallel with LRU per-doc normalised-shingle reuse (same doc in many pairs is
not re-normalised/re-read; fixes the unbounded-cache RSS blowup) and imap progress + --max_pairs cap.

Reporting: near-dup rate = docs in clustered (>=th) pairs / total; Wilson 95% CI; est<EXACT_FL pairs
are assumed below every floor (documented in boundary). Read-only; never writes corpus.
"""
import argparse
import glob
import json
import os
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
EXACT_FL = 0.60

_GB = None
_GL = None
# per-worker LRU: gid -> frozenset of normalised 3-word shingles (reused across pairs)
_SHL = OrderedDict()
_SHL_MAX = 20000
_BYTES = 0


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
    global _BYTES
    d = int(np.searchsorted(_GB, g, side="right"))
    local = g - (0 if d == 0 else _GB[d - 1])
    shard, ln = _GL[d][local]
    with open(shard, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == ln:
                txt = json.loads(line).get("content", "")
                _BYTES += len(txt.encode("utf-8", "replace"))
                return txt
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
    global _BYTES
    u, v = pair
    _BYTES = 0
    j = ND.jaccard(_shingles(u), _shingles(v))
    return j, _BYTES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckdir", required=True)
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--max_pairs", type=int, default=0, help="cap exact-Jaccard pairs (0=unlimited)")
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
    print(f"banded {all_sigs.shape[0]} docs into {len(BUCKETS)} buckets ({rr // 60}m{rr % 60}s)", flush=True)

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
    exact_pv = pv[est >= EXACT_FL]
    rr = round(time.perf_counter() - t0)
    print(f"{len(pairs)} candidates, est>={EXACT_FL}: {exact_pv.shape[0]} for exact Jaccard ({rr // 60}m{rr % 60}s)", flush=True)

    scored = {}
    n_exact = exact_pv.shape[0]
    if a.max_pairs and n_exact > a.max_pairs:
        exact_pv = exact_pv[:a.max_pairs]
        n_exact = exact_pv.shape[0]
    ETA_CEIL = 7200  # 2h; abort before full verify (fb rule)
    if exact_pv.size:
        with Pool(a.pool, initializer=_init_verify, initargs=(dom_bound, dom_locs)) as pool:
            done = bytes_total = 0
            probe_t0 = time.perf_counter()
            probed = False
            for (u, v), (j, br) in zip(exact_pv.tolist(),
                                       pool.imap_unordered(_exact_jac, exact_pv.tolist(), chunksize=512)):
                scored[(int(u), int(v))] = j
                done += 1
                bytes_total += br
                if not probed and done == 10000:
                    per = (time.perf_counter() - probe_t0) / 10000.0
                    eta = (n_exact - done) * per
                    print(f"probe 10k pairs: {per * 1000:.2f} ms/pair, ETA {eta / 60:.1f} min ({eta:.0f}s)", flush=True)
                    probed = True
                    if eta > ETA_CEIL:
                        print(f"ABORT (fb rule): ETA {eta / 60:.1f} min > 2h ceiling; not entering full verify", flush=True)
                        with open(a.out, "w", encoding="utf-8") as f:
                            json.dump({"aborted": True, "reason": f"ETA {eta/60:.1f} min > 2h",
                                       "docs_total": n, "verified_pairs": done}, f, indent=1)
                        return
                if done % 200000 == 0:
                    rr = round(time.perf_counter() - t0)
                    mb = bytes_total / 1e6
                    print(f"  exact: {done}/{n_exact} pairs, {mb:.0f} MB read ({rr // 60}m{rr % 60}s)", flush=True)
        rr = round(time.perf_counter() - t0)
        print(f"exact Jaccard done: {len(scored)}/{n_exact} pairs, {bytes_total/1e6:.0f} MB ({rr // 60}m{rr % 60}s)", flush=True)

    out = {"domains": a.domains, "docs_total": n, "sample": "FULL (all shards)",
           "method": f"MinHash-LSH n_perm={PERMS} bands={BANDS}({BAND_ROWS}/band) + exact Jaccard on est>={EXACT_FL}",
           "candidate_pairs": len(pairs), "est_exact": int(exact_pv.shape[0]),
           "verified_pairs": len(scored), "thresholds": {}}
    for th in THRESHOLDS:
        e = [(u, v, j) for (u, v), j in scored.items() if j >= th]
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
        rate = docs_in / max(1, n)
        z = 1.96
        denom = 1 + z * z / n
        p_hat = (rate + z * z / (2 * n)) / denom
        half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denom

        def dom_loc(g):
            d = int(np.searchsorted(dom_bound, g, side="right"))
            return d, g - (0 if d == 0 else dom_bound[d - 1])

        cross = sum(1 for u, v, _ in e if dom_loc(u)[0] != dom_loc(v)[0])
        samples = sorted(e, key=lambda x: -x[2])[:20]
        out["thresholds"][str(th)] = {
            "pairs": len(e), "docs_in_clustered_pairs": docs_in,
            "near_dup_rate": round(rate, 6), "rate_ci95": [round(p_hat - half, 6), round(p_hat + half, 6)],
            "clusters": len(multi), "cross_domain_pairs": cross,
            "cross_domain_share": round(cross / max(1, len(e)), 4),
            "sample_pairs": [
                {"a": dom_locs[dom_loc(u)[0]][dom_loc(u)[1]][0] + ":" + str(dom_locs[dom_loc(u)[0]][dom_loc(u)[1]][1]),
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

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
import inspect
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
from build_corpus import _NORM  # noqa: E402
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
    s = _NORM.sub("", _read_doc(g))
    # char 5-gram — the SAME shingle the MinHash signature uses (build_corpus.MinHashLSH.signature 168-169),
    # so exact Jaccard and the est metric measure the SAME J (fb: verify failed on definition mismatch).
    sh = frozenset(s[i:i + 5] for i in range(max(1, len(s) - 4)))
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
           "method": "MinHash-J estimate + top-pair exact verify (n_perm=%d, bands=%d(%d/band), char 5-gram; est and exact use the same shingle, so J is J of that shingle)" % (PERMS, BANDS, BAND_ROWS),
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
        # PARTICIPATION, not drop. docs_in counts every document that has a near-dup;
        # a dedup keeps one representative per cluster, so it removes docs_in - clusters.
        # The two were one field named near_dup_rate until 2026-09-07, and multiplying
        # code_rp1t's th0.9 participation by a token count overstated the loss by
        # 701,466,449 tokens. Both are written, and neither is named near_dup_rate.
        drop = (docs_in - len(multi)) / max(1, n)
        if multi and drop >= rate:
            raise AssertionError(
                f"th{th}: drop {drop} >= participation {rate} over {len(multi)} cluster(s). "
                f"A cluster has >=2 members by construction, so a kept representative must "
                f"make drop strictly smaller. Equal means clusters was counted wrong or the "
                f"two rates are reading one number.")
        z = 1.96
        denom = 1 + z * z / n
        p_hat = (rate + z * z / (2 * n)) / denom
        half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denom
        cross = sum(1 for u, v in edges if dom_loc(u)[0] != dom_loc(v)[0])
        samples = sorted([(int(u), int(v), float(e)) for u, v, e in zip(edges[:, 0], edges[:, 1], est[emask])],
                         key=lambda x: -x[2])[:20]
        out["thresholds"][str(th)] = {
            "participation_rate": round(rate, 6), "drop_rate": round(drop, 6),
            "rate_ci95": [round(p_hat - half, 6), round(p_hat + half, 6)],
            "rate_ci95_is_for": "participation_rate",
            "docs_in_clustered_pairs": docs_in, "edges": int(emask.sum()), "clusters": len(multi),
            "cross_domain_pairs": cross, "cross_domain_share": round(cross / max(1, int(emask.sum())), 4),
            "sample_pairs": [
                {"a": dom_locs[dom_loc(u)[0]][dom_loc(u)[1]][0] + ":" + str(dom_locs[dom_loc(u)[0]][dom_loc(u)[1]][1]),
                 "b": dom_locs[dom_loc(v)[0]][dom_loc(v)[1]][0] + ":" + str(dom_locs[dom_loc(v)[0]][dom_loc(v)[1]][1]),
                 "est": round(e, 3)} for u, v, e in samples],
        }
        rr = round(time.perf_counter() - t0)
        print(f"  est>=th{th}: participation {rate:.5f} (CI {round(p_hat-half,5)},{round(p_hat+half,5)}), "
              f"drop {drop:.5f}, {int(emask.sum())} edges, {len(multi)} clusters ({rr // 60}m{rr % 60}s)",
              flush=True)

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


def _rates(cluster_sizes, n):
    """participation and drop from cluster sizes. Both rates in one place, so the writer
    and the selftest cannot disagree about which is which."""
    multi = [s for s in cluster_sizes if s > 1]
    docs_in = sum(multi)
    part = docs_in / max(1, n)
    drop = (docs_in - len(multi)) / max(1, n)
    if multi and drop >= part:
        raise AssertionError(
            f"drop {drop} >= participation {part} over {len(multi)} cluster(s). A cluster has "
            f">=2 members by construction, so a kept representative must make drop strictly "
            f"smaller. Equal means clusters was counted wrong or the two rates read one number.")
    return part, drop, len(multi), docs_in


def _selftest():
    """The two rates are different numbers, and a count with no representative subtracted
    is refused. Against the REAL measurement rather than an invented one: the published
    code_rp1t th0.9 row must reproduce, including the identity that its drop equals what
    the build actually removed. The field was named near_dup_rate and held participation
    for weeks -- a name cannot fail, so the distinction is enforced here."""
    n = 3747157
    sizes = [2] * 114888
    sizes[0] += 427723 - 2 * 114888
    part, drop, nc, docs_in = _rates(sizes, n)
    assert docs_in == 427723, docs_in
    assert nc == 114888, nc
    assert round(part, 6) == 0.114146, part
    assert round(drop, 6) == 0.083486, drop
    assert docs_in - nc == 3747157 - 3434322, docs_in - nc
    assert drop < part

    part0, drop0, nc0, _ = _rates([1, 1, 1], 3)
    assert (part0, drop0, nc0) == (0.0, 0.0, 0), (part0, drop0, nc0)

    # The guard cannot be tripped through _rates' own arithmetic: the subtraction of one
    # representative per cluster happens inside it, so with any cluster list drop < part
    # holds identically. It is a tripwire for a FUTURE edit to that line, and a test that
    # pretended otherwise would be a world built from the implementation. So mutate the
    # line and assert the guard catches the mutant, which is the thing actually claimed.
    src = inspect.getsource(_rates)
    mutant_src = src.replace("(docs_in - len(multi)) / max(1, n)", "docs_in / max(1, n)")
    assert mutant_src != src, "the drop line moved; this mutation no longer applies"
    ns = {}
    exec(mutant_src, {"__builtins__": __builtins__}, ns)  # noqa: S102
    caught = False
    try:
        ns["_rates"]([2, 3], 100)
    except AssertionError as e:
        caught = "drop" in str(e)
    assert caught, "a drop computed without subtracting the representative was NOT refused"

    # (4) the control that makes (3) mean something: with the GUARD removed as well, the
    # same mutant must pass silently. Without this, case 3 proves only "something raises",
    # and a future edit that makes _rates raise for an unrelated reason keeps it green.
    # 62 called this as the finding on PR #1; it was run by hand and belongs in the file.
    noguard = mutant_src.replace("    if multi and drop >= part:\n", "    if False:\n")
    assert noguard != mutant_src, "the guard line moved; this control no longer applies"
    ns2 = {}
    exec(noguard, {"__builtins__": __builtins__}, ns2)  # noqa: S102
    slipped = True
    try:
        ns2["_rates"]([2, 3], 100)
    except AssertionError:
        slipped = False
    assert slipped, ("with the guard disabled the mutant still raised -- case 3 is passing for "
                     "some other reason and proves nothing about the guard")

    print("near_dedup_scale selftest OK: 4 cases. code_rp1t th0.9 reproduces participation "
          "0.114146 and drop 0.083486 (427,723 - 114,888 = 312,835 = the build's exact removal, "
          "3,747,157 - 3,434,322); an empty world gives 0/0 and does not raise; a mutant that "
          "drops the representative subtraction is refused; and with the guard also removed that "
          "same mutant passes silently, so case 3 fails on the guard and not on something else")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    main()

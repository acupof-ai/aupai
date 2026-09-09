#!/usr/bin/env python3
# 3b cross-domain near-overlap at J>=0.5 (2026-09-09).
# The three kept code domains were each deduped at build time (dd09/b2v2 th=0.9,
# dedup08 th=0.8); what remains is CROSS-domain overlap. Exact channel is measured
# by exact_overlap.py; this measures the near channel [0.5, 1.0).
#
# Method: 96-perm MinHash char 5-gram (same shingle and perm count as the
# builds: near_dedup_scale/code_dedup_build/dedup_keep_whole all use 96),
# 24 bands x 4 rows (LSH threshold ~0.45; the builds use 12 bands x 8 rows,
# ~0.73 -- this instrument deliberately casts a wider net). Signatures
# checkpointed per domain.
# Jaccard estimate: 50K-doc sample of each domain queried against the band
# index built over ALL docs; participation = docs with >=1 est>=0.5 neighbor in
# the other domain. Exact char-5gram Jaccard on a 100-pair sample calibrates est.
# Every est>=0.5 hit pair is persisted to data/decontam/near_overlap_hits_0909.jsonl
# (4c 2026-09-10): the keep-set participation cut is a separate join over that
# file, not a statistic this instrument emits.
import glob, json, os, random, sys, time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/work/aupai/datagen")
import build_corpus as B
from build_corpus import _NORM

PERMS, BANDS, ROWS = 96, 24, 4
TH = 0.5
SAMPLE_N = 50000
CK = "/work/aupai/runs/overlap_ck"
DOMAINS = [
    ("dd09", "/work/aupai/data/corpus/code_rp1t_dd09/*.jsonl"),
    ("b2v2", "/work/aupai/data/corpus/code_rp1t_b2v2_dd/*.jsonl"),
    ("dedup08", "/work/aupai/data/corpus/code_dedup08/*.jsonl"),
]


def sig_one(shard_path):
    lsh = B.MinHashLSH(perms=PERMS, bands=BANDS)
    sigs, locs = [], []
    try:
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
    except Exception as e:
        print(f"  SHARD FAIL {shard_path}: {e}", flush=True)
    return (np.stack(sigs) if sigs else np.zeros((0, PERMS), np.int64)), locs


def sign_domain(name, pat):
    os.makedirs(CK, exist_ok=True)
    sp, lp = f"{CK}/{name}.sig.npy", f"{CK}/{name}.loc.json"
    if os.path.exists(sp) and os.path.exists(lp):
        S = np.load(sp)
        print(f"{name}: cached {S.shape[0]} sigs", flush=True)
        return S
    shards = sorted(glob.glob(pat))
    t0 = time.perf_counter()
    parts, done, n_shards = [], 0, 0
    with Pool(16) as pool:
        for sigs, locs in pool.imap_unordered(sig_one, shards, chunksize=1):
            parts.append((sigs, locs))
            done += len(locs)
            n_shards += 1
            print(f"  {name}: shard {n_shards}/{len(shards)} ({os.path.basename(locs[0][0]) if locs else 'empty'}), "
                  f"{done} total ({round(time.perf_counter()-t0)}s)", flush=True)
    S = np.vstack([p[0] for p in parts])
    np.save(sp, S)
    # locs in the SAME completion order as the stacked sigs. A sorted-glob
    # rebuild misaligns ~85% of rows (b0, PR #177 review 2026-09-10): the
    # sigs are stacked in imap_unordered completion order, not shard order.
    json.dump([loc for p in parts for loc in p[1]], open(lp, "w"))
    print(f"{name}: {S.shape[0]} sigs -> {sp} ({round(time.perf_counter()-t0)}s)", flush=True)
    return S


def shingles_of(text):
    s = _NORM.sub("", text)
    return frozenset(s[i:i + 5] for i in range(max(1, len(s) - 4)))


def read_doc(loc):
    shard, ln = loc
    with open(shard, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == ln:
                return json.loads(line).get("content", "")
    return ""


def main():
    random.seed(20260909)
    sigs, sizes = [], []
    for name, pat in DOMAINS:
        S = sign_domain(name, pat)
        sigs.append(S)
        sizes.append(S.shape[0])
    all_sigs = np.vstack(sigs)
    bound = np.cumsum(sizes)
    dom_of = np.concatenate([np.full(n, d) for d, n in enumerate(sizes)])
    n = all_sigs.shape[0]
    print(f"total {n} docs; banding {BANDS}x{ROWS}", flush=True)

    # band index: key -> global ids
    idx = defaultdict(list)
    for g in range(n):
        sig = all_sigs[g]
        for b in range(BANDS):
            idx[(b, sig[b * ROWS:(b + 1) * ROWS].tobytes())].append(g)
    print(f"{len(idx)} buckets", flush=True)

    # participation: sample SAMPLE_N docs per domain, query against all
    rng = random.Random(20260909)
    out = {"method": f"MinHash {PERMS}perm char5gram, LSH {BANDS}x{ROWS} (th~0.45), est>={TH}, "
                     f"sample {SAMPLE_N}/domain queried vs full index", "domains": {}}
    verify_pairs = []
    hit_pairs = set()
    for d, (name, _) in enumerate(DOMAINS):
        lo = 0 if d == 0 else bound[d - 1]
        hi = bound[d]
        sample = rng.sample(range(lo, hi), min(SAMPLE_N, hi - lo))
        # neighbors per other domain
        hits = {e: set() for e in range(len(DOMAINS)) if e != d}
        max_est = {}
        for g in sample:
            sig = all_sigs[g]
            cands = set()
            for b in range(BANDS):
                cands.update(idx.get((b, sig[b * ROWS:(b + 1) * ROWS].tobytes()), ()))
            for c in cands:
                if c == g:
                    continue
                e = int(dom_of[c])
                if e == d:
                    continue
                est = float((sig == all_sigs[c]).mean())
                if est >= TH:
                    hits[e].add(g)
                    hit_pairs.add((min(int(g), int(c)), max(int(g), int(c)), est))
                    if g not in max_est or est > max_est[g][0]:
                        max_est[g] = (est, c)
        rec = {"sampled": len(sample)}
        for e, hs in hits.items():
            oname = DOMAINS[e][0]
            rec[f"hit_in_{oname}"] = len(hs)
            rec[f"participation_in_{oname}"] = round(len(hs) / len(sample), 6)
            # collect verify pairs (highest est first, capped)
            vp = sorted(((max_est[g][0], g, max_est[g][1]) for g in hs), reverse=True)
            verify_pairs.extend(vp[:50])
        out["domains"][name] = rec
        print(f"{name}: {rec}", flush=True)

    # persist every unique hit pair: the keep-set participation cut joins this
    # file by doc id, it does not re-run the instrument
    locs = []
    for name, _ in DOMAINS:
        lp = f"{CK}/{name}.loc.json"
        if not os.path.exists(lp):
            raise SystemExit(f"missing {lp} -- sign_domain must persist locs in sig order")
        locs.append(json.load(open(lp)))
    # content guard: len equality cannot prove loc/sig alignment -- a sorted-glob
    # rebuild passes len while misaligning ~85% of rows (b0, PR #177, 2026-09-10)
    guard_rng = random.Random(0)
    lsh = B.MinHashLSH(perms=PERMS, bands=BANDS)
    for d, (name, _) in enumerate(DOMAINS):
        L, S = locs[d], sigs[d]
        assert len(L) == S.shape[0], f"{name}: {len(L)} locs vs {S.shape[0]} sigs"
        for i in guard_rng.sample(range(len(L)), min(14, len(L))):
            doc = read_doc(L[i])
            assert (np.asarray(lsh.signature(doc), dtype=np.int64) == S[i]).all(), \
                f"{name}: loc[{i}] sig mismatch -- loc/sig order misaligned"
    print("loc guard: 14/doc content spot-check OK", flush=True)
    pair_path = "/work/aupai/data/decontam/near_overlap_hits_0909.jsonl"
    os.makedirs(os.path.dirname(pair_path), exist_ok=True)
    with open(pair_path, "w") as f:
        for g, c, est in sorted(hit_pairs):
            d, e = int(dom_of[g]), int(dom_of[c])
            lo_g = 0 if d == 0 else bound[d - 1]
            lo_c = 0 if e == 0 else bound[e - 1]
            f.write(json.dumps({"shard_a": locs[d][g - lo_g][0], "row_a": locs[d][g - lo_g][1],
                                "domain_a": DOMAINS[d][0], "shard_b": locs[e][c - lo_c][0],
                                "row_b": locs[e][c - lo_c][1], "domain_b": DOMAINS[e][0],
                                "est_jaccard": round(est, 4)}, ensure_ascii=False) + "\n")
    out["hit_pairs_unique"] = len(hit_pairs)
    print(f"hit pairs: {len(hit_pairs)} unique -> {pair_path}", flush=True)

    # exact-J calibration on a fixed sample of hit pairs
    random.shuffle(verify_pairs)
    vp = verify_pairs[:100]
    diffs = []
    for est, g, c in vp:
        d, e = int(dom_of[g]), int(dom_of[c])
        lo_g = 0 if d == 0 else bound[d - 1]
        lo_c = 0 if e == 0 else bound[e - 1]
        try:
            tg = read_doc(locs[d][g - lo_g])
            tc = read_doc(locs[e][c - lo_c])
        except Exception:
            continue
        sa, sb = shingles_of(tg), shingles_of(tc)
        if not sa or not sb:
            continue
        jex = len(sa & sb) / len(sa | sb)
        diffs.append((est, jex))
    if diffs:
        mae = sum(abs(e - j) for e, j in diffs) / len(diffs)
        bias = sum(e - j for e, j in diffs) / len(diffs)
        n_above = sum(1 for e, j in diffs if j >= TH)
        out["exact_verify"] = {"sample_n": len(diffs), "mae_est_minus_exact": round(mae, 4),
                               "mean_bias": round(bias, 4),
                               "exact_J_ge_0.5": n_above,
                               "note": "pairs sampled from est>=0.5 hits; exact-J<0.5 are est false positives"}
        print(f"verify: n={len(diffs)}, mae={mae:.3f}, bias={bias:+.3f}, exact J>=0.5: {n_above}/{len(diffs)}", flush=True)

    json.dump(out, open("/work/aupai/runs/near_overlap_0909.json", "w"), ensure_ascii=False, indent=2)
    print("DONE -> /work/aupai/runs/near_overlap_0909.json", flush=True)


if __name__ == "__main__":
    main()

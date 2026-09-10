#!/usr/bin/env python3
"""Synthetic self-repetition metrics (3b, 2026-09-09).

See docs/standards/synth_selfrep_metric.md for the metric design. Two modes:

  --mode stream   R1 over full jsonl corpora (stdlib+numpy):
                  distinct-8gram rate via the kth-hash estimator + exact-dup rate.
  --mode sample   R2+R3 on a doc sample (needs sklearn): reskin-band near-dup
                  rate [0.3,0.7) and top-10 cluster mass.

Usage:
  python3 synth_selfrep.py --mode stream --corpus a.jsonl [b.jsonl ...] [--field content]
  python3 synth_selfrep.py --mode sample --corpus a.jsonl [--field content] [--n 100000]
"""
import argparse, hashlib, json, random, re
from collections import Counter

import numpy as np

TOK = re.compile(r"[a-z0-9]+|[一-鿿]")
P = 12  # kth-hash modulus 2^P: expected 2^-P of hashes kept


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def stream_mode(paths, field):
    kept8 = 0
    total8 = 0
    seen_doc = set()
    dup_doc = 0
    n_doc = 0
    for path in paths:
        for line in open(path, encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            text = str(d.get(field, ""))
            toks = TOK.findall(text.lower())
            n_doc += 1
            dh = h64(text[:4096])
            if dh in seen_doc:
                dup_doc += 1
            else:
                seen_doc.add(dh)
            for i in range(len(toks) - 7):
                total8 += 1
                if h64(" ".join(toks[i:i + 8])) % (1 << P) == 0:
                    kept8 += 1
    est_distinct = kept8 * (1 << P)
    print(json.dumps({
        "docs": n_doc,
        "exact_dup_rate": round(dup_doc / max(n_doc, 1), 5),
        "total_8grams": total8,
        "distinct_8gram_rate": round(est_distinct / max(total8, 1), 5),
        "estimator": f"kth-hash p={P}, kept={kept8}",
    }, ensure_ascii=False, indent=2))


# ---------- sample mode (R2/R3) ----------

PERMS = 64
RNG = np.random.default_rng(20260909)
A = RNG.integers(1, 1 << 32, size=PERMS, dtype=np.uint64)
B = RNG.integers(0, 1 << 32, size=PERMS, dtype=np.uint64)
MERSENNE = np.uint64((1 << 61) - 1)


def minhash(tokens):
    """64-perm MinHash signature over the token set. None if empty."""
    if not tokens:
        return None
    hs = np.array([h64(t) for t in set(tokens)], dtype=np.uint64)
    V = (A[None, :] * hs[:, None] + B[None, :]) & MERSENNE  # (n_tokens, PERMS)
    return V.min(axis=0)


def reskin_band(sigs):
    """Pair-rate in Jaccard bands from MinHash signatures (upper triangle only).

    sigs: (m, PERMS) uint64. Returns (pairs, band_counts).
    """
    m = len(sigs)
    band = {"[0.3,0.5)": 0, "[0.5,0.7)": 0, ">=0.7": 0}
    pairs = 0
    BLOCK = 2000
    for i0 in range(0, m, BLOCK):
        i1 = min(i0 + BLOCK, m)
        # accumulate signature matches in perm-chunks to bound memory:
        # full (BLOCK, m, 64) bool tensor would be ~2.5GB
        acc = np.zeros((i1 - i0, m - i0 - 1), dtype=np.int16)
        for p0 in range(0, PERMS, 8):
            p1 = min(p0 + 8, PERMS)
            acc += (sigs[i0:i1, None, p0:p1] == sigs[None, i0 + 1:, p0:p1]).sum(axis=2)
        j = acc / PERMS
        # upper triangle only: row r (global i=i0+r) pairs with columns
        # c >= r (global j=i0+1+c > i)
        r_idx = np.arange(i1 - i0)[:, None]
        c_idx = np.arange(m - i0 - 1)[None, :]
        upper = c_idx >= r_idx
        j = j[upper]
        pairs += j.size
        band[">=0.7"] += int((j >= 0.7).sum())
        band["[0.5,0.7)"] += int(((j >= 0.5) & (j < 0.7)).sum())
        band["[0.3,0.5)"] += int(((j >= 0.3) & (j < 0.5)).sum())
    return pairs, band


def sample_mode(path, field, n):
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    random.seed(20260909)
    random.shuffle(lines)
    docs = []
    for line in lines[:n]:
        try:
            d = str(json.loads(line).get(field, ""))
        except Exception:
            continue
        if d.strip():
            docs.append(d)
    print(f"sample: {len(docs)} docs")

    # R2: MinHash near-dup rate in the reskin band [0.3, 0.7)
    sigs = [s for s in (minhash(TOK.findall(d.lower())) for d in docs[:20000]) if s is not None]
    pairs, band = reskin_band(np.stack(sigs))
    reskin = band["[0.3,0.5)"] + band["[0.5,0.7)"]
    print(json.dumps({"R2_docs_scored": len(sigs), "R2_pairs_scored": pairs,
                      "R2_reskin_band": band,
                      "R2_reskin_rate": round(reskin / max(pairs, 1), 6)},
                     ensure_ascii=False, indent=2))

    # R3: cluster mass
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=3, max_features=50000)
    X = vec.fit_transform(docs[:50000])
    km = KMeans(n_clusters=200, random_state=20260909, n_init=3).fit(X)
    sizes = Counter(km.labels_)
    top10 = sum(c for _, c in sizes.most_common(10))
    print(json.dumps({"R3_top10_cluster_mass": round(top10 / X.shape[0], 4),
                      "R3_largest_cluster_pct": round(sizes.most_common(1)[0][1] / X.shape[0], 4)},
                     ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stream", "sample"], required=True)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--field", default="content")
    ap.add_argument("--n", type=int, default=100000)
    args = ap.parse_args()
    if args.mode == "stream":
        stream_mode(args.corpus, args.field)
    else:
        sample_mode(args.corpus[0], args.field, args.n)


if __name__ == "__main__":
    main()

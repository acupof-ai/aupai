#!/usr/bin/env python3
"""DCLM-style fastText clone for the CCI3 junk/usable decision.

Hashed char (2-5) and word (1-2) n-grams, L2-normalized, logistic regression
with L-BFGS. CPU-only, numpy+scipy. Trains on the 150 hand-read benchmark and
tests on the locked 400 (2x2) and web_labels (non-degradation vs the deployed
quality head's 0.823 AUC).

    python datagen/fasttext_junk.py --raw-dir data/raw/cci3_hq

The 150 and 400 samples carry (shard, id); full text is pulled from the raw
CCI3-HQ shards by id. The locked 400 is test-only: never train on it.
"""
import argparse, json, re
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize

DIM = 1 << 20
CAP = 20000  # chars per doc


def ngram_hashes(text):
    text = text[:CAP]
    h = set()
    b = text.encode("utf-8")
    for n in (2, 3, 4, 5):
        for i in range(len(b) - n + 1):
            h.add(hash(b[i:i + n]) % DIM)
    words = re.findall(r"\w+", text, re.UNICODE)
    for w in words:
        h.add(hash(("W" + w).encode("utf-8")) % DIM)
    for i in range(len(words) - 1):
        h.add(hash(("W" + words[i] + "_" + words[i + 1]).encode("utf-8")) % DIM)
    return h


def build_matrix(texts):
    rows, cols = [], []
    for i, t in enumerate(texts):
        for c in ngram_hashes(t):
            rows.append(i)
            cols.append(c)
    X = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(texts), DIM))
    norms = np.sqrt(X.multiply(X).sum(1)).A1
    norms[norms == 0] = 1
    return X.multiply(1.0 / norms[:, None]).tocsr()


def fit(X, y, reg):
    d = X.shape[1]

    def loss(w):
        z = X @ w
        ll = np.sum(np.logaddexp(0, z) - y * z) + 0.5 * reg * np.dot(w, w)
        p = 1.0 / (1.0 + np.exp(-z))
        return ll, X.T @ (p - y) + reg * w

    return minimize(loss, np.zeros(d), jac=True, method="L-BFGS-B", options={"maxiter": 300}).x


def auc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos, nneg = y.sum(), len(y) - y.sum()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def pull_texts(raw_dir, pairs):
    """pairs: list of (shard, id). Returns {id: text} from the raw shards."""
    by_shard = {}
    for shard, doc_id in pairs:
        by_shard.setdefault(shard, set()).add(doc_id)
    total = len(pairs)
    out = {}
    for shard, want in by_shard.items():
        with open(f"{raw_dir}/{shard}") as f:
            for line in f:
                d = json.loads(line)
                if d["id"] in want:
                    out[d["id"]] = d["text"]
                    if len(out) == total:
                        return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/cci3_hq")
    ap.add_argument("--handread", default="data/corpus/sample/cci3_handread_150.jsonl")
    ap.add_argument("--audit400", default="data/corpus/sample/cci3_audit_400.jsonl")
    ap.add_argument("--audit400-labels", default="data/corpus/sample/cci3_audit_400_labels.jsonl")
    ap.add_argument("--web-labels", default="data/corpus/sample/web_labels.jsonl")
    a = ap.parse_args()

    hand = [json.loads(l) for l in open(a.handread)]
    assert all(r.get("label_set_version") == "v2" for r in hand), "handread must be v2 (0cf1eb8fa4ae=junk)"
    pairs = [(r["shard"], r["id"]) for r in hand]
    texts = pull_texts(a.raw_dir, pairs)
    Xtr = build_matrix([texts[r["id"]] for r in hand])
    y = np.array([1.0 if r["hand_label"] == "junk" else 0.0 for r in hand])
    print(f"train: {len(hand)} docs, {y.sum():.0f} junk", flush=True)

    best = None
    idx = np.arange(len(y))
    for reg in (1e-6, 1e-5, 1e-4, 1e-3):
        aucs = []
        for k in range(5):
            va = idx[k::5]
            tr = np.setdiff1d(idx, va)
            w = fit(Xtr[tr], y[tr], reg)
            aucs.append(auc(y[va], Xtr[va] @ w))
        m = np.mean(aucs)
        print(f"  reg={reg:.0e} CV-AUC={m:.3f}", flush=True)
        if best is None or m > best[1]:
            best = (reg, m)
    reg = best[0]
    w = fit(Xtr, y, reg)
    print(f"chosen reg={reg:.0e} in-sample AUC={auc(y, Xtr @ w):.3f}", flush=True)

    audit = [json.loads(l) for l in open(a.audit400)]
    labels = {json.loads(l)["id"]: json.loads(l)["junk"] for l in open(a.audit400_labels)}
    t400 = pull_texts(a.raw_dir, [(r["shard"], r["id"]) for r in audit])
    junk400 = np.array([1.0 if labels[r["id"]] else 0.0 for r in audit])
    s400 = build_matrix([t400[r["id"]] for r in audit]) @ w
    print(f"\nLOCKED-400: AUC(junk)={auc(junk400, s400):.3f}")
    for q in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        t = np.quantile(s400, 1 - q)
        drop = s400 >= t
        tp = int((drop & (junk400 == 1)).sum())
        fp = int((drop & (junk400 == 0)).sum())
        fn = int((~drop & (junk400 == 1)).sum())
        rec = tp / (tp + fn)
        prec = tp / (tp + fp) if tp + fp else 0
        mark = "  <-- PASS" if rec >= 0.5 and prec >= 0.8 else ""
        print(f"  drop {q:.0%}: recall={rec:.3f} precision={prec:.3f} (TP={tp} FP={fp} FN={fn}){mark}")

    web = [json.loads(l) for l in open(a.web_labels)]
    yw = np.array([1.0 - r["y"] for r in web])  # web y=1 means keep/educational
    sw = build_matrix([r["t"] for r in web]) @ w
    print(f"\nWEB_LABELS: AUC(junk)={auc(yw, sw):.3f}  (old head AUC(keep)=0.823 on this set)")


if __name__ == "__main__":
    main()

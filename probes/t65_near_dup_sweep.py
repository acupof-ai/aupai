"""Near-duplicate rate per domain, at the normaliser that ACTUALLY EXISTS, swept over threshold.

Imports build_corpus's own _word_shingle_hashes / _near_coeffs / _minhash_sig / _lsh_candidates
rather than reimplementing them, so this measures the shipped engine and not a lookalike.

WHY A SWEEP AND NOT A NUMBER. near_dedup_gate.md calibrates J>=0.5 against per-domain
normalisers (norm_code, norm_en_c4, norm_math with a 70-word keyword stoplist and LaTeX
mapping). Those do not exist in the repo -- grep returns nothing. The only normaliser present
is _norm_skeleton: lowercase + collapse whitespace. Under it the gate doc's own example
cluster scores 0.181, so 0.5 is a threshold calibrated for one measuring stick applied to
another. A single number at 0.5 would therefore be a fact about the wrong stick; the sweep
reports the curve and lets the threshold be chosen with the number in view.

    python3 t65.py <corpus_root> <domain> [rows] [--selftest]
"""
import json
import os
import sys

sys.path.insert(0, os.environ.get("AUPAI_ROOT", "/work/aupai"))
sys.path.insert(0, os.path.join(os.environ.get("AUPAI_ROOT", "/work/aupai"), "datagen"))
import build_corpus as bc  # noqa: E402

PERMS, BANDS, ROWS = 128, 64, 2
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def load(root, domain, cap):
    import glob
    docs = []
    for p in sorted(x for x in glob.glob(os.path.join(root, domain, "*.jsonl"))
                    if "build_corpus_stats" not in os.path.basename(x)):
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                t = o.get("text") or o.get("content") or ""
                if t:
                    docs.append(t)
                    if len(docs) >= cap:
                        return docs
    return docs


def measure(texts):
    ab, mask = bc._near_coeffs(PERMS, 0)
    sigs, sh = {}, {}
    for i, t in enumerate(texts):
        h = bc._word_shingle_hashes(bc._norm_skeleton(t))
        if not h:
            continue
        sh[i] = set(h)
        sigs[i] = bc._minhash(h, ab, mask)
    cands = bc._lsh_candidates(sigs, BANDS, ROWS)
    # exact Jaccard once per candidate pair, then bucket by threshold
    js = []
    for a, b in cands:
        A, B = sh[a], sh[b]
        u = len(A | B)
        if u:
            js.append((len(A & B) / u, a, b))
    out = {}
    for th in THRESHOLDS:
        removable = set()
        pairs = 0
        for j, a, b in js:
            if j >= th:
                pairs += 1
                removable.add(max(a, b))
        out[str(th)] = {"pairs": pairs, "rows_removable": len(removable),
                        "pct": round(100 * len(removable) / max(len(sh), 1), 2)}
    return {"docs_shingled": len(sh), "candidate_pairs": len(cands), "by_threshold": out}


def selftest():
    """A near-dup probe that finds nothing must first be shown to find a planted pair."""
    base = " ".join(f"word{i}" for i in range(200))
    texts = [f"unique document {i} " + " ".join(f"tok{i}_{j}" for j in range(60))
             for i in range(60)]
    texts += [base, base + " and one extra clause at the end"]   # high J, not exact
    got = measure(texts)
    hi = got["by_threshold"]["0.9"]["rows_removable"]
    lo = got["by_threshold"]["0.3"]["rows_removable"]
    assert lo >= 1, f"planted near-dup pair not found at J>=0.3: {got}"
    assert hi >= 1, f"planted pair should survive to J>=0.9 (it is ~0.97): {got}"
    print(f"selftest OK: planted near-dup found, removable at J>=0.3 {lo}, at J>=0.9 {hi}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        root, dom = sys.argv[1], sys.argv[2]
        cap = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
        r = measure(load(root, dom, cap))
        r["domain"], r["normaliser"] = dom, "_norm_skeleton (lowercase+collapse); the doc's per-domain normalisers do not exist"
        print(json.dumps(r))

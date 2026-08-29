#!/usr/bin/env python3
"""Bidirectional contamination scan: eval holdouts vs a candidate corpus.

Finding #1 of docs/review_2026-08-26.md was stage-1 SFT on the corpus the eval problems
were drawn from. Every new source gets this scan before it enters the mix, and every
existing source can be re-scanned. This is the math-specific sibling of
scripts/scan_contamination.py (which scans generic corpus with exact line matching):
this one adds containment and a cross-language screen.

Screens, cheapest first:
  1. exact-normalized match (same key as scripts/holdout.py: whitespace/punctuation-insensitive)
  2. containment: |holdout bigrams ∩ text bigrams| / |holdout bigrams| >= threshold.
     Jaccard is the wrong metric here: a verbatim question embedded in an 841-char
     document scores Jaccard 0.339. Containment answers the right question -- how much
     of the holdout is present in the text -- and is monotone in text length.
  3. cross-language number-multiset screen (for en sources vs zh holdouts, where surface
     bigrams cannot match): identical number multiset with >= 2 numbers -> flagged for
     human review. Same-language precision is ~1% (2M flags on the math corpus), so use
     it ONLY across languages.

Containment is one sparse matmul: R (rows x vocab) @ H (vocab x holdouts), exact, in C.
Measured 2026-08-30 on the 245 MB math corpus (530k rows, ~73M tok): 0.74M tok/s with
the old Python postings loop, 6.8M tok/s single-process matmul, 22M tok/s with 9 fork
workers -- 100B tok rescans in 1.26h. LSH was considered and rejected: it is approximate
(needs a recall proof against the 223 known hits) where the exact scan already clears
the 2h budget. Deps: numpy, scipy (the pod needs both).

Incremental: data/scan_ledger.jsonl records (path, bytes, mtime, holdout_hash, threshold)
-> verdict. A re-run skips shards whose fingerprint, holdout set, and threshold are
unchanged; a holdout-set change invalidates every entry and forces a full rescan.
--force bypasses.
ponytail: bytes+mtime is a make-style fingerprint -- content-hash if a shard can be
replaced in place adversarially.

Min-length guard: holdouts with <20 bigrams are scored in a separate bucket. Containment
saturates on short holdouts ("1+1 等于几" is nearly contained in any text with digits),
so their hits are reported separately, never mixed into the main count.

Output: full distribution, not just threshold counts -- deciles of per-holdout max
containment, and hit counts at 0.7 / 0.8 / 0.9, so the reader can see how sensitive the
verdict is to the threshold.

Usage:
    python scripts/scan_math_contamination.py <path> [--q-field NAME] [--full-doc]
                                                 [--threshold 0.8] [--jobs N] [--force]
                                                 [--fpr-baseline CLEAN_PATH]
    python scripts/scan_math_contamination.py --self-check

Modes:
    default      jsonl/parquet with a question field (math sources); the question part
                 of 问：...答：... content is extracted
    --full-doc   scan whole documents (web/wiki/textbook corpora)
    --fpr-baseline <clean_path>   also scan a known-clean corpus and report the false-
                 positive rate at the threshold; the only evidence the threshold means
                 anything. Not recorded in the ledger.

Exit code: 1 if any holdout is hit at the threshold (REJECT THE WHOLE SOURCE -- do not
filter rows), 0 if clean. Field-extraction failures never pass as clean: if >1% of rows
yield empty text the scan errors out instead of reporting 0 hits.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import get_context

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from holdout import norm  # noqa: E402

# math_hard_eval_1k retired 2026-08-30: every math_short batch contaminated it
# (cont.math_short_leak) -- it shared the bank's elementary-olympiad canon.
# Replaced by math_hard_eval_v2_1k (type-disjoint: symbolic algebra/geometry,
# cont.math_hard_v2). Old math-hard scores are void (cont.math_hard_v1_void).
HOLDOUT_FILES = [
    "data/eval/math_test_500.jsonl",
    "data/synthetic/math_hard_eval_v2_1k.jsonl",
]
DEFAULT_THRESHOLD = 0.8
MIN_BIGRAMS = 20
CHUNK = 20000
NUM_RE = __import__("re").compile(r"\d+\.?\d*")
QUESTION_FIELDS = ("question", "instruction", "problem", "core_question", "content")
LEDGER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "scan_ledger.jsonl")


def qhash(q):
    return hashlib.sha1(norm(q).encode("utf-8")).hexdigest()[:16]


def bigrams(q):
    """Char-bigrams for zh, word tokens for en -- same scheme the Jaccard era used."""
    q = str(q)
    if __import__("re").search(r"[一-鿿]", q):
        return {q[i : i + 2] for i in range(len(q) - 1)}
    return set(__import__("re").findall(r"[a-z0-9]+", q.lower()))


def numbers(q):
    return tuple(sorted(NUM_RE.findall(str(q))))


def load_holdouts(files=None):
    qs = []
    for path in files or HOLDOUT_FILES:
        if not os.path.exists(path):
            print(f"  missing (skipped): {path}")
            continue
        for line in open(path, encoding="utf-8"):
            if line.strip():
                qs.append(json.loads(line)["instruction"])
    return qs


def extract_question(text, qfield):
    """Question part of a 问：...答：... doc, or the field itself. Never silent-empty."""
    t = str(text or "")
    if qfield and qfield != "content":
        return t.strip()
    t = t.split("\n答：")[0]
    if t.startswith("问："):
        t = t[2:]
    return t.strip()


def iter_texts(path, qfield, full_doc):
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        names = pq.ParquetFile(path).schema_arrow.names
        col = None
        for q in [qfield] + list(QUESTION_FIELDS):
            if q and q in names:
                col = q
                break
        if col is None:
            sys.exit(f"no question-like column in {path}; pass --q-field")
        for t in pq.ParquetFile(path).read(columns=[col]).to_pydict()[col]:
            yield t if full_doc else extract_question(t, col)
    else:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            t = d.get(qfield) if qfield else None
            if t is None:
                for f in QUESTION_FIELDS:
                    if d.get(f):
                        t = d[f]
                        break
            yield t if full_doc else extract_question(t, qfield or "content")


class HoldoutIndex:
    """Sparse matmul containment: R (rows x V) @ H (V x holdouts), exact, in C."""

    def __init__(self, holdouts):
        self.holdouts = holdouts
        self.h_norm = {qhash(q) for q in holdouts}
        h_bigrams = [bigrams(q) for q in holdouts]
        self.sizes = np.array([len(b) for b in h_bigrams], dtype=np.float64)
        vocab = sorted({g for bg in h_bigrams for g in bg})
        self.g2i = {g: i for i, g in enumerate(vocab)}
        self.H = np.zeros((len(vocab), len(holdouts)), dtype=np.int16)
        for j, bg in enumerate(h_bigrams):
            for g in bg:
                self.H[self.g2i[g], j] = 1
        self.short = [i for i, b in enumerate(h_bigrams) if len(b) < MIN_BIGRAMS]
        self.nums = defaultdict(list)
        for i, q in enumerate(holdouts):
            nums = numbers(q)
            if len(nums) >= 2:
                self.nums[nums].append(i)

    def scan_chunk(self, texts):
        """One batch. Exact-hit rows are excluded from containment (same as the old
        per-row path: an exact hit returned early, contributing no bigrams)."""
        ridx, cidx = [], []
        exact_hits, num_flags = [], 0
        max_cont = np.zeros(len(self.holdouts), dtype=np.float64)
        for r, text in enumerate(texts):
            if not text:
                continue
            if qhash(text) in self.h_norm:
                exact_hits.append((r, str(text)[:120]))
                continue
            for g in bigrams(text):
                i = self.g2i.get(g)
                if i is not None:
                    ridx.append(r)
                    cidx.append(i)
            nums = numbers(text)
            if len(nums) >= 2:
                num_flags += len(self.nums.get(nums, ()))
        if ridx:
            R = sparse.csr_matrix(
                (np.ones(len(ridx), dtype=np.int16), (ridx, cidx)),
                shape=(len(texts), self.H.shape[0]),
            )
            cont = np.asarray(R @ self.H, dtype=np.float64) / self.sizes
            max_cont = np.maximum(max_cont, cont.max(axis=0))
        return exact_hits, max_cont, num_flags


def scan_path(path, idx, qfield, full_doc, threshold):
    n = empty = 0
    max_cont = np.zeros(len(idx.holdouts), dtype=np.float32)
    exact_hits, num_flags = [], 0
    chunk = []
    chunk_start = 0
    for text in iter_texts(path, qfield, full_doc):
        n += 1
        if not text:
            empty += 1
            continue
        chunk.append(text)
        if len(chunk) >= CHUNK:
            ex, mc, nf = idx.scan_chunk(chunk)
            exact_hits += [(chunk_start + r, t) for r, t in ex]
            max_cont = np.maximum(max_cont, mc)
            num_flags += nf
            chunk_start += len(chunk)
            chunk = []
    if chunk:
        ex, mc, nf = idx.scan_chunk(chunk)
        exact_hits += [(chunk_start + r, t) for r, t in ex]
        max_cont = np.maximum(max_cont, mc)
        num_flags += nf
    if n and empty / n > 0.01:
        sys.exit(f"REFUSED: {empty}/{n} rows ({empty/n:.1%}) extracted empty -- "
                 f"field name mismatch? Pass --q-field. Not reporting clean on empty text.")
    return n, empty, exact_hits, max_cont.tolist(), num_flags


def _worker(args):
    path, idx, qfield, full_doc, threshold = args
    return path, *scan_path(path, idx, qfield, full_doc, threshold)


def deciles(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    return {f"p{p}": round(xs[min(len(xs) - 1, p * len(xs) // 100)], 3)
            for p in (50, 75, 90, 95, 99, 100)}


def report(name, n, empty, exact_hits, max_cont, num_flags, threshold, holdouts, short):
    long_i = [i for i in range(len(holdouts)) if i not in set(short)]
    long_vals = [max_cont[i] for i in long_i]
    short_vals = [max_cont[i] for i in short]
    hits = {t: sum(1 for v in long_vals if v >= t) for t in (0.7, 0.8, 0.9)}
    print(f"\n=== {name}: {n} rows, {empty} empty")
    print(f"exact-normalized hits: {len(exact_hits)}")
    for row, t in exact_hits[:5]:
        print(f"  row {row}: {t}")
    print(f"containment distribution (holdouts with >= {MIN_BIGRAMS} bigrams, n={len(long_vals)}):")
    print(f"  deciles: {deciles(long_vals)}")
    print(f"  holdouts hit at 0.7 / 0.8 / 0.9: {hits[0.7]} / {hits[0.8]} / {hits[0.9]}")
    if short:
        s_hits = sum(1 for v in short_vals if v >= threshold)
        print(f"short-holdout bucket (<{MIN_BIGRAMS} bigrams, n={len(short)}): "
              f"max-containment deciles {deciles(short_vals)}, {s_hits} at {threshold} "
              f"-- REPORTED SEPARATELY, not in the main count")
    print(f"number-multiset flags (cross-language screen, review-only): {num_flags}")
    verdict = "REJECT" if hits[threshold] or exact_hits else "clean"
    print(f"verdict at {threshold}: {verdict}")
    return 1 if verdict == "REJECT" else 0


def holdout_hash(holdouts):
    return hashlib.sha1("".join(sorted(qhash(q) for q in holdouts)).encode()).hexdigest()[:16]


def ledger_read():
    rows = []
    if os.path.exists(LEDGER):
        for line in open(LEDGER, encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ledger_cached(ledger, path, hhash, threshold):
    st = os.stat(path)
    for row in ledger:
        if (row.get("path") == path and row.get("bytes") == st.st_size
                and row.get("mtime") == int(st.st_mtime) and row.get("holdout_hash") == hhash
                and row.get("threshold") == threshold):
            return row
    return None


def ledger_append(path, hhash, threshold, n, hits_08, exact, verdict):
    st = os.stat(path)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "path": path, "bytes": st.st_size, "mtime": int(st.st_mtime),
            "holdout_hash": hhash, "threshold": threshold, "scanned_at": int(time.time()),
            "rows": n, "hits_0.8": hits_08, "exact": exact, "verdict": verdict,
        }, ensure_ascii=False) + "\n")


def self_check():
    """Known-answer pair: one contaminated, one clean; containment must differ by >= 0.6."""
    holdouts = ["小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？"]
    idx = HoldoutIndex(holdouts)
    contaminated = "老师出题：" + holdouts[0] + " 请列式计算。"
    clean = "办公室里有个不起眼的现象：谁在周会上被点名夸过一次，下次他的周报就写得更细。"
    _, mc1, _ = idx.scan_chunk([contaminated])
    _, mc2, _ = idx.scan_chunk([clean])
    v1, v2 = float(mc1[0]), float(mc2[0])
    assert v1 == 1.0, f"verbatim embed must be containment 1.0, got {v1}"
    assert v1 - v2 >= 0.6, f"known-answer pair must differ by >= 0.6, got {v1} vs {v2}"
    # field fall-through must not pass as clean
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write(json.dumps({"wrong_field": holdouts[0]}, ensure_ascii=False) + "\n")
        tmp = f.name
    try:
        rc = 0
        try:
            scan_path(tmp, idx, "question", False, 0.8)
        except SystemExit as e:
            rc = e.code
        assert rc not in (0, None), "field mismatch must error, not report clean"
    finally:
        os.unlink(tmp)
    print(f"self-check OK (contaminated {v1} vs clean {v2}, field-mismatch refused)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--q-field", default=None)
    ap.add_argument("--full-doc", action="store_true")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers over shards (fork)")
    ap.add_argument("--force", action="store_true", help="rescan even if the ledger has a fresh verdict")
    ap.add_argument("--fpr-baseline", metavar="CLEAN_PATH",
                    help="known-clean corpus to estimate the false-positive rate")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--holdout", action="append",
                    help="holdout jsonl (repeatable); overrides the default math-500+math-hard set")
    args = ap.parse_args()
    if args.self_check:
        sys.exit(self_check())
    if not args.path:
        ap.error("path required (or --self-check)")
    holdouts = load_holdouts(args.holdout)
    if not holdouts:
        sys.exit("no holdout files found; run scripts/holdout.py first or run from repo root")
    idx = HoldoutIndex(holdouts)
    hhash = holdout_hash(holdouts)
    paths = sorted(glob.glob(args.path)) if "*" in args.path else [args.path]
    ledger = [] if args.force else ledger_read()
    rc = 0
    fresh, cached = [], []
    for p in paths:
        row = None if args.fpr_baseline else ledger_cached(ledger, p, hhash, args.threshold)
        if row and not args.force:
            cached.append((p, row))
            rc |= 1 if row["verdict"] == "REJECT" else 0
        else:
            fresh.append(p)
    if args.jobs > 1 and len(fresh) > 1:
        with get_context("fork").Pool(min(args.jobs, len(fresh))) as pool:
            results = pool.map(_worker, [(p, idx, args.q_field, args.full_doc, args.threshold)
                                         for p in fresh])
    else:
        results = [_worker((p, idx, args.q_field, args.full_doc, args.threshold)) for p in fresh]
    short = set(idx.short)
    for p, n, empty, exact, mc, nf in results:
        rc |= report(os.path.basename(p), n, empty, exact, mc, nf, args.threshold, holdouts, short)
        long_i = [i for i in range(len(holdouts)) if i not in short]
        hits_08 = sum(1 for i in long_i if mc[i] >= args.threshold)
        verdict = "REJECT" if hits_08 or exact else "clean"
        ledger_append(p, hhash, args.threshold, n, hits_08, len(exact), verdict)
    for p, row in cached:
        print(f"=== {os.path.basename(p)}: cached {row['verdict']} "
              f"({row['hits_0.8']} hits at 0.8, scanned_at={row['scanned_at']}) -- ledger")
    if args.fpr_baseline:
        fn, fe, fex, fmc, fnf = scan_path(args.fpr_baseline, idx, args.q_field, True, args.threshold)
        long_i = [i for i in range(len(holdouts)) if i not in short]
        fp = sum(1 for i in long_i if fmc[i] >= args.threshold)
        print(f"\nFPR baseline ({os.path.basename(args.fpr_baseline)}, {fn} docs, assumed clean): "
              f"{fp}/{len(long_i)} holdouts hit at {args.threshold} = {fp/len(long_i):.2%}")
    sys.exit(rc)


if __name__ == "__main__":
    main()

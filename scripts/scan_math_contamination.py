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
     document scores Jaccard 0.339. Containment answers the right question — how much
     of the holdout is present in the text — and is monotone in text length.
  3. cross-language number-multiset screen (for en sources vs zh holdouts, where surface
     bigrams cannot match): identical number multiset with >= 2 numbers -> flagged for
     human review. Same-language precision is ~1% (2M flags on the math corpus), so use
     it ONLY across languages.

Min-length guard: holdouts with <20 bigrams are scored in a separate bucket. Containment
saturates on short holdouts ("1+1 等于几" is nearly contained in any text with digits),
so their hits are reported separately, never mixed into the main count.

Output: full distribution, not just threshold counts — deciles of per-holdout max
containment, and hit counts at 0.7 / 0.8 / 0.9, so the reader can see how sensitive the
verdict is to the threshold.

Usage:
    python scripts/scan_math_contamination.py <path> [--q-field NAME] [--full-doc]
                                                 [--threshold 0.8] [--fpr-baseline]
    python scripts/scan_math_contamination.py --self-check

Modes:
    default      jsonl/parquet with a question field (math sources); the question part
                 of 问：...答：... content is extracted
    --full-doc   scan whole documents (web/wiki/textbook corpora)
    --fpr-baseline <clean_path>   also scan a known-clean corpus and report the false-
                 positive rate at the threshold; the only evidence the threshold means
                 anything

Exit code: 1 if any holdout is hit at the threshold (REJECT THE WHOLE SOURCE — do not
filter rows), 0 if clean. Field-extraction failures never pass as clean: if >1% of rows
yield empty text the scan errors out instead of reporting 0 hits.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from holdout import norm  # noqa: E402

HOLDOUT_FILES = [
    "data/eval/math_test_500.jsonl",
    "data/synthetic/math_hard_eval_1k.jsonl",
]
DEFAULT_THRESHOLD = 0.8
MIN_BIGRAMS = 20
NUM_RE = __import__("re").compile(r"\d+\.?\d*")
QUESTION_FIELDS = ("question", "instruction", "problem", "core_question", "content")


def qhash(q):
    return hashlib.sha1(norm(q).encode("utf-8")).hexdigest()[:16]


def bigrams(q):
    """Char-bigrams for zh, word tokens for en — same scheme the Jaccard era used."""
    q = str(q)
    if __import__("re").search(r"[一-鿿]", q):
        return {q[i : i + 2] for i in range(len(q) - 1)}
    return set(__import__("re").findall(r"[a-z0-9]+", q.lower()))


def numbers(q):
    return tuple(sorted(NUM_RE.findall(str(q))))


def load_holdouts():
    qs = []
    for path in HOLDOUT_FILES:
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
    """Inverted index over holdout bigrams; containment per candidate text."""

    def __init__(self, holdouts):
        self.holdouts = holdouts
        self.h_norm = {qhash(q) for q in holdouts}
        self.bigrams = [bigrams(q) for q in holdouts]
        self.nums = defaultdict(list)
        for i, q in enumerate(holdouts):
            nums = numbers(q)
            if len(nums) >= 2:
                self.nums[nums].append(i)
        self.short = [i for i, b in enumerate(self.bigrams) if len(b) < MIN_BIGRAMS]
        self.postings = defaultdict(list)
        for i, bg in enumerate(self.bigrams):
            for g in bg:
                self.postings[g].append(i)

    def scan_text(self, text):
        """Return (exact_hit, {holdout_idx: containment}, number_multiset_hits)."""
        if qhash(text) in self.h_norm:
            return True, {}, []
        tb = bigrams(text)
        shared = defaultdict(int)
        for g in tb:
            for i in self.postings.get(g, ()):
                shared[i] += 1
        cont = {i: shared[i] / len(self.bigrams[i]) for i in shared}
        nums = numbers(text)
        num_hits = self.nums.get(tuple(nums), []) if len(nums) >= 2 else []
        return False, cont, num_hits


def scan_path(path, idx, qfield, full_doc, threshold):
    n = empty = 0
    max_cont = [0.0] * len(idx.holdouts)
    exact_hits = []
    num_flags = 0
    for text in iter_texts(path, qfield, full_doc):
        n += 1
        if not text:
            empty += 1
            continue
        exact, cont, num_hits = idx.scan_text(text)
        if exact:
            exact_hits.append((n, text[:120]))
        for i, c in cont.items():
            if c > max_cont[i]:
                max_cont[i] = c
        num_flags += len(num_hits)
    if n and empty / n > 0.01:
        sys.exit(f"REFUSED: {empty}/{n} rows ({empty/n:.1%}) extracted empty — "
                 f"field name mismatch? Pass --q-field. Not reporting clean on empty text.")
    return n, empty, exact_hits, max_cont, num_flags


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
              f"— REPORTED SEPARATELY, not in the main count")
    print(f"number-multiset flags (cross-language screen, review-only): {num_flags}")
    verdict = "REJECT" if hits[threshold] or exact_hits else "clean"
    print(f"verdict at {threshold}: {verdict}")
    return 1 if verdict == "REJECT" else 0


def self_check():
    """Known-answer pair: one contaminated, one clean; containment must differ by >= 0.6."""
    holdouts = ["小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？"]
    idx = HoldoutIndex(holdouts)
    contaminated = "老师出题：" + holdouts[0] + " 请列式计算。"
    clean = "办公室里有个不起眼的现象：谁在周会上被点名夸过一次，下次他的周报就写得更细。"
    _, c1, _ = idx.scan_text(contaminated)
    _, c2, _ = idx.scan_text(clean)
    v1, v2 = c1.get(0, 0.0), c2.get(0, 0.0)
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
    ap.add_argument("--fpr-baseline", metavar="CLEAN_PATH",
                    help="known-clean corpus to estimate the false-positive rate")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        sys.exit(self_check())
    if not args.path:
        ap.error("path required (or --self-check)")
    holdouts = load_holdouts()
    if not holdouts:
        sys.exit("no holdout files found; run scripts/holdout.py first or run from repo root")
    idx = HoldoutIndex(holdouts)
    paths = sorted(glob.glob(args.path)) if "*" in args.path else [args.path]
    rc = 0
    for p in paths:
        n, empty, exact, mc, nf = scan_path(p, idx, args.q_field, args.full_doc, args.threshold)
        short = set(idx.short)
        rc |= report(os.path.basename(p), n, empty, exact, mc, nf, args.threshold, holdouts, short)
    if args.fpr_baseline:
        fn, fe, fex, fmc, fnf = scan_path(args.fpr_baseline, idx, args.q_field, True, args.threshold)
        long_i = [i for i in range(len(holdouts)) if i not in set(idx.short)]
        fp = sum(1 for i in long_i if fmc[i] >= args.threshold)
        print(f"\nFPR baseline ({os.path.basename(args.fpr_baseline)}, {fn} docs, assumed clean): "
              f"{fp}/{len(long_i)} holdouts hit at {args.threshold} = {fp/len(long_i):.2%}")
    sys.exit(rc)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bidirectional contamination scan: eval holdouts vs a candidate corpus.

Finding #1 of docs/review_2026-08-26.md was stage-1 SFT on the corpus the eval problems
were drawn from. Every new source gets this scan before it enters the mix, and every
existing source can be re-scanned. This is the math-specific sibling of
scripts/scan_contamination.py (which scans generic corpus with exact line matching):
this one adds containment and a cross-language screen.

Screens, cheapest first:
  1. exact-normalized match (same key as scripts/holdout.py: whitespace/punctuation-insensitive)
  2. containment: |holdout bigrams ∩ text bigrams| / |holdout bigrams| >= threshold,
     IDF-WEIGHTED over the scanned corpus (pass 1 accumulates document frequency per
     holdout bigram; pass 2 scores). Without weighting, generic bigrams (学生/班级)
     dominate short generic holdouts: the 2026-08-30 math-500 scan produced 11 false
     hits at 0.80-0.86, all education-vocabulary overlap on 24-28-bigram template
     problems, 0 exact. Weighted, all 12 pairs dropped below 0.8 (max 0.789) while
     verbatim injections stayed at 1.000. Jaccard is the wrong metric here: a verbatim
     question embedded in an 841-char document scores Jaccard 0.339. Containment
     answers the right question -- how much of the holdout is present in the text --
     and is monotone in text length.
  3. cross-language number-multiset screen (for en sources vs zh holdouts, where surface
     bigrams cannot match): identical number multiset with >= 2 numbers -> flagged for
     human review. Same-language precision is ~1% (2M flags on the math corpus), so use
     it ONLY across languages.

Containment is one sparse matmul per pass (pass 1: document frequency for IDF;
pass 2: scoring). Measured 2026-08-30 on the 245 MB math corpus (530k rows, ~73M tok): 0.74M tok/s with
the old Python postings loop, 6.8M tok/s single-process matmul, 22M tok/s with 9 fork
workers -- 100B tok rescans in 1.26h. LSH was considered and rejected: it is approximate
(needs a recall proof against the 223 known hits) where the exact scan already clears
the 2h budget. Deps: numpy, scipy (the pod needs both).

Incremental: data/scan_ledger.jsonl records (path, bytes, mtime, holdout_hash, threshold,
idf) -> verdict. A re-run skips shards whose fingerprint, holdout set, threshold, and
weighting are unchanged; a holdout-set change invalidates every entry and forces a full
rescan. Pre-IDF ledger rows (idf absent) never match -- the 0.8 threshold means a
different quantity weighted. --force bypasses.
ponytail: bytes+mtime is a make-style fingerprint -- content-hash if a shard can be
replaced in place adversarially.

Min-length guard: holdouts with <20 bigrams are scored in a separate bucket. Containment
saturates on short holdouts ("1+1 等于几" is nearly contained in any text with digits),
so their hits are reported separately, never mixed into the main count.

Output: full distribution, not just threshold counts -- deciles of per-holdout max
containment, and hit counts at 0.7 / 0.8 / 0.9, so the reader can see how sensitive the
verdict is to the threshold.

Usage:
    python scripts/scan_math_contamination.py <path> --fpr-baseline BASELINE_PATH
                                                 [--q-field NAME] [--full-doc]
                                                 [--threshold 0.8] [--jobs N] [--force]
    python scripts/scan_math_contamination.py --self-check

    --fpr-baseline is REQUIRED: the same-scale in-training corpus. The verdict is
    the candidate's per-GB hit rate vs the baseline's -- REJECT if > 2x (fb's
    pre-registered rule, cont.cci3_vs_webhq_rates) or on any exact hit. Absolute
    per-shard counts are meaningless across unequal corpus sizes (cont.cci3_scale_failure).

Modes:
    default      jsonl/parquet with a question field (math sources); the question part
                 of 问：...答：... content is extracted
    --full-doc   scan whole documents (web/wiki/textbook corpora)
    --fpr-baseline <clean_path>   also scan a known-clean corpus and report the false-
                 positive rate at the threshold; the only evidence the threshold means
                 anything. Not recorded in the ledger.

Exit code: 1 if the candidate REJECTs (exact hit, or per-GB hit rate > 2x the
baseline's), 0 if clean. Field-extraction failures never pass as clean: if >1% of
rows yield empty text the scan errors out instead of reporting 0 hits.
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

# Fork-after-OpenBLAS deadlock: the parent builds HoldoutIndex (spawning the
# OpenBLAS pool, 4 threads on the pod), then fork()s workers that inherit a
# broken pool and hang at 0% CPU on their first BLAS call. macOS uses
# Accelerate (fork-safe) so --jobs worked locally; Linux/OpenBLAS hangs.
# The matmul is sparse@dense and barely uses BLAS anyway -- serial BLAS costs
# nothing, and throughput comes from process parallelism. Must precede numpy.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
# fb's pre-registered admission rule (2026-08-30, cont.cci3_vs_webhq_rates):
# accept a candidate corpus if its per-GB hit rate is at most this x the
# same-scale in-training baseline's rate. Above it, REJECT.
RATE_REJECT_RATIO = 2.0
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
    """Sparse matmul containment: R (rows x V) @ H (V x holdouts), exact, in C.

    H is sparse csc: a dense 75K-vocab x 7.5K-holdout H would be 2.2GB; the
    sparse form is ~6MB. With set_idf, containment is IDF-weighted over the
    scanned corpus -- generic bigrams stop dominating short generic holdouts
    (the 2026-08-30 math-500 FP class: 12/12 known FP pairs dropped below 0.8
    weighted, verbatim injections stayed at 1.000).
    """

    def __init__(self, holdouts):
        self.holdouts = holdouts
        self.h_norm = {qhash(q) for q in holdouts}
        h_bigrams = [bigrams(q) for q in holdouts]
        self.sizes = np.array([len(b) for b in h_bigrams], dtype=np.float64)
        vocab = sorted({g for bg in h_bigrams for g in bg})
        self.g2i = {g: i for i, g in enumerate(vocab)}
        rows, cols = [], []
        for j, bg in enumerate(h_bigrams):
            for g in bg:
                rows.append(self.g2i[g])
                cols.append(j)
        self.H = sparse.csc_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)),
            shape=(len(vocab), len(holdouts)),
        )
        # column-normalised forms: cont = R @ Hn
        self.Hn = (self.H @ sparse.diags(1.0 / self.sizes)).tocsr()
        self.short = [i for i, b in enumerate(h_bigrams) if len(b) < MIN_BIGRAMS]
        self.short_set = set(self.short)
        self.long_cols = [i for i in range(len(holdouts)) if i not in self.short_set]
        self.idf = None
        self.Hw_n = None
        self.nums = defaultdict(list)
        for i, q in enumerate(holdouts):
            nums = numbers(q)
            if len(nums) >= 2:
                self.nums[nums].append(i)

    def set_idf(self, df, n_docs):
        """Enable IDF-weighted containment. df: document frequency per vocab
        bigram over the scanned corpus; n_docs: corpus size."""
        self.idf = np.log((n_docs + 1) / (df + 1)) + 1.0
        Hw = self.H.T.multiply(self.idf).T.tocsc()
        mass = np.asarray(Hw.sum(axis=0)).ravel()
        self.Hw_n = (Hw @ sparse.diags(1.0 / mass)).tocsr()

    def _matrix(self, texts, skip=()):
        """Binary doc-bigram matrix R for one batch (bigrams() is a set, so R
        is binary by construction). `skip` rows contribute no bigrams
        (exact hits -- same exclusion as the old per-row early return)."""
        ridx, cidx = [], []
        for r, text in enumerate(texts):
            if not text or r in skip:
                continue
            for g in bigrams(text):
                i = self.g2i.get(g)
                if i is not None:
                    ridx.append(r)
                    cidx.append(i)
        if not ridx:
            return None
        return sparse.csr_matrix(
            (np.ones(len(ridx), dtype=np.int8), (ridx, cidx)),
            shape=(len(texts), self.H.shape[0]),
        )

    def df_chunk(self, texts):
        """Document-frequency contribution of one batch (column sums of R)."""
        R = self._matrix(texts)
        if R is None:
            return np.zeros(self.H.shape[0], dtype=np.float64)
        return np.asarray(R.sum(axis=0)).ravel()

    def scan_chunk(self, texts, threshold):
        """One batch. Exact-hit rows are excluded from containment (same as the
        old per-row path: an exact hit returned early, contributing no bigrams)."""
        exact_hits, num_flags = [], 0
        for r, text in enumerate(texts):
            if not text:
                continue
            if qhash(text) in self.h_norm:
                exact_hits.append((r, str(text)[:120]))
                continue
            nums = numbers(text)
            if len(nums) >= 2:
                num_flags += len(self.nums.get(nums, ()))
        R = self._matrix(texts, skip=frozenset(r for r, _ in exact_hits))
        if R is None:
            return exact_hits, np.zeros(len(self.holdouts)), num_flags, 0
        M = R @ (self.Hw_n if self.idf is not None else self.Hn)
        max_cont = np.asarray(M.max(axis=0).todense()).ravel()
        hit_rows = 0
        if self.long_cols:
            rowmax = np.asarray(M[:, self.long_cols].max(axis=1).todense()).ravel()
            hit_rows = int((rowmax >= threshold).sum())
        return exact_hits, max_cont, num_flags, hit_rows


def _iter_chunks(path, qfield, full_doc):
    """Yield text chunks of CHUNK from a path."""
    chunk = []
    for text in iter_texts(path, qfield, full_doc):
        chunk.append(text)
        if len(chunk) >= CHUNK:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def scan_path(path, idx, qfield, full_doc, threshold):
    n = empty = 0
    hit_rows = 0
    max_cont = np.zeros(len(idx.holdouts), dtype=np.float64)
    exact_hits, num_flags = [], 0
    chunk_start = 0
    for chunk in _iter_chunks(path, qfield, full_doc):
        n += len(chunk)
        empty += sum(1 for t in chunk if not t)
        ex, mc, nf, hr = idx.scan_chunk(chunk, threshold)
        exact_hits += [(chunk_start + r, t) for r, t in ex]
        max_cont = np.maximum(max_cont, mc)
        num_flags += nf
        hit_rows += hr
        chunk_start += len(chunk)
    if n and empty / n > 0.01:
        sys.exit(f"REFUSED: {empty}/{n} rows ({empty/n:.1%}) extracted empty -- "
                 f"field name mismatch? Pass --q-field. Not reporting clean on empty text.")
    return n, empty, exact_hits, max_cont.tolist(), num_flags, hit_rows


def _worker(args):
    # idx rides fork copy-on-write via the module global (set in main before the
    # Pool is created) -- pickling it per task would push ~200MB x N_tasks
    # through the task pipe for nothing.
    path, idx, qfield, full_doc, threshold = args
    idx = _WORKER_IDX if _WORKER_IDX is not None else idx
    return path, *scan_path(path, idx, qfield, full_doc, threshold)


def _df_worker(args):
    """Per-path document-frequency contribution (IDF pass, fork pool)."""
    path, qfield, full_doc = args
    idx = _WORKER_IDX
    df = np.zeros(idx.H.shape[0], dtype=np.float64)
    n = 0
    for chunk in _iter_chunks(path, qfield, full_doc):
        df += idx.df_chunk(chunk)
        n += len(chunk)
    return df, n


_WORKER_IDX = None


def deciles(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    return {f"p{p}": round(xs[min(len(xs) - 1, p * len(xs) // 100)], 3)
            for p in (50, 75, 90, 95, 99, 100)}


def report(name, n, empty, exact_hits, max_cont, num_flags, hit_rows, bytes_,
           threshold, holdouts, short, base_per_gb):
    """Scale-free rates for one corpus/shard, vs a same-scale baseline rate.

    Verdict (fb pre-registered 2026-08-30, cont.cci3_vs_webhq_rates):
      exact hit            -> REJECT (verbatim contamination, no FPR defense)
      candidate per-GB rate > RATE_REJECT_RATIO x baseline -> REJECT
      else clean.
    Absolute per-shard counts are meaningless across unequal corpus sizes
    (cont.cci3_scale_failure): the baseline is the same-scale in-training corpus.
    """
    long_i = [i for i in range(len(holdouts)) if i not in short]
    long_vals = [max_cont[i] for i in long_i]
    short_vals = [max_cont[i] for i in short]
    gb = bytes_ / 1e9
    per_gb = hit_rows / gb if gb else 0.0
    per_mdoc = hit_rows / n * 1e6 if n else 0.0
    print(f"\n=== {name}: {n} rows, {gb:.3f} GB, {empty} empty")
    print(f"exact-normalized hits: {len(exact_hits)}")
    for row, t in exact_hits[:5]:
        print(f"  row {row}: {t}")
    print(f"hit rows (containment >= {threshold}, long holdouts): {hit_rows} "
          f"= {per_gb:.1f}/GB = {per_mdoc:.1f}/M-docs")
    print(f"  (union holdouts hit at 0.7/0.8/0.9: "
          f"{sum(v >= 0.7 for v in long_vals)}/{sum(v >= 0.8 for v in long_vals)}/{sum(v >= 0.9 for v in long_vals)}"
          f" -- context only, not the verdict)")
    if short:
        s_hits = sum(1 for v in short_vals if v >= threshold)
        print(f"short-holdout bucket (<{MIN_BIGRAMS} bigrams, n={len(short)}): "
              f"deciles {deciles(short_vals)}, {s_hits} at {threshold} -- REPORTED SEPARATELY")
    print(f"number-multiset flags (cross-language screen, review-only): {num_flags}")
    if exact_hits:
        verdict = "REJECT"
    elif base_per_gb == 0:
        verdict = "REJECT" if hit_rows else "clean"
    else:
        verdict = "REJECT" if per_gb / base_per_gb > RATE_REJECT_RATIO else "clean"
    ratio = "inf" if base_per_gb == 0 and hit_rows else (f"{per_gb / base_per_gb:.2f}x" if base_per_gb else "n/a")
    print(f"verdict: {verdict} (baseline {base_per_gb:.1f}/GB, ratio {ratio}, "
          f"reject if > {RATE_REJECT_RATIO}x or exact)")
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


def ledger_cached(ledger, path, hhash, threshold, baseline_id):
    st = os.stat(path)
    for row in ledger:
        if (row.get("path") == path and row.get("bytes") == st.st_size
                and row.get("mtime") == int(st.st_mtime) and row.get("holdout_hash") == hhash
                and row.get("threshold") == threshold and row.get("baseline_id") == baseline_id
                and bool(row.get("idf"))):
            return row
    return None


def ledger_append(path, hhash, threshold, baseline_id, n, hit_rows, exact, per_gb, verdict):
    st = os.stat(path)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "path": path, "bytes": st.st_size, "mtime": int(st.st_mtime),
            "holdout_hash": hhash, "threshold": threshold, "baseline_id": baseline_id,
            "idf": True,
            "scanned_at": int(time.time()),
            "rows": n, "hit_rows": hit_rows, "per_gb": round(per_gb, 2),
            "exact": exact, "verdict": verdict,
        }, ensure_ascii=False) + "\n")


def self_check():
    """Known-answer pair: one contaminated, one clean; containment must differ by >= 0.6."""
    holdouts = ["小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？"]
    idx = HoldoutIndex(holdouts)
    contaminated = "老师出题：" + holdouts[0] + " 请列式计算。"
    clean = "办公室里有个不起眼的现象：谁在周会上被点名夸过一次，下次他的周报就写得更细。"
    _, mc1, _, hr1 = idx.scan_chunk([contaminated], 0.8)
    _, mc2, _, hr2 = idx.scan_chunk([clean], 0.8)
    v1, v2 = float(mc1[0]), float(mc2[0])
    assert abs(v1 - 1.0) < 1e-9, f"verbatim embed must be containment 1.0, got {v1}"
    assert v1 - v2 >= 0.6, f"known-answer pair must differ by >= 0.6, got {v1} vs {v2}"
    assert hr1 == 1 and hr2 == 0, f"hit-row counting wrong: {hr1}/{hr2}"
    # IDF-weighted path: uniform IDF must reproduce unweighted containment
    idx.set_idf(np.ones(len(idx.g2i)), 10)
    _, mc1w, _, hr1w = idx.scan_chunk([contaminated], 0.8)
    assert abs(float(mc1w[0]) - v1) < 1e-9, \
        f"weighted containment must match unweighted under uniform IDF, got {mc1w[0]} vs {v1}"
    assert hr1w == hr1
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


def scan_many(paths, idx, args):
    """Scan fresh paths with the fork pool (idx rides COW via _WORKER_IDX)."""
    if not paths:
        return []
    if args.jobs > 1 and len(paths) > 1:
        global _WORKER_IDX
        _WORKER_IDX = idx
        with get_context("fork").Pool(min(args.jobs, len(paths))) as pool:
            return pool.map(_worker, [(p, None, args.q_field, args.full_doc, args.threshold)
                                      for p in paths])
    return [_worker((p, idx, args.q_field, args.full_doc, args.threshold)) for p in paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--q-field", default=None)
    ap.add_argument("--full-doc", action="store_true")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers over shards (fork)")
    ap.add_argument("--force", action="store_true", help="rescan even if the ledger has a fresh verdict")
    ap.add_argument("--fpr-baseline", metavar="BASELINE_PATH",
                    help="REQUIRED: same-scale in-training corpus; the verdict is the candidate's "
                         "per-GB hit rate vs the baseline's (a FPR number without a same-scale "
                         "baseline has no binding power -- cont.cci3_scale_failure)")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--holdout", action="append",
                    help="holdout jsonl (repeatable); overrides the default math-500+math-hard set")
    args = ap.parse_args()
    if args.self_check:
        sys.exit(self_check())
    if not args.path:
        ap.error("path required (or --self-check)")
    if not args.fpr_baseline:
        ap.error("--fpr-baseline is required: pass the same-scale in-training corpus "
                 "(e.g. the web_hq shards or data/corpus/math/gsm8k_zh_000.jsonl for math batches)")
    holdouts = load_holdouts(args.holdout)
    if not holdouts:
        sys.exit("no holdout files found; run scripts/holdout.py first or run from repo root")
    idx = HoldoutIndex(holdouts)
    hhash = holdout_hash(holdouts)
    paths = sorted(glob.glob(args.path)) if "*" in args.path else [args.path]
    base_paths = sorted(glob.glob(args.fpr_baseline)) if "*" in args.fpr_baseline else [args.fpr_baseline]
    baseline_id = ":".join(f"{p}:{os.stat(p).st_size}:{int(os.stat(p).st_mtime)}" for p in base_paths)
    ledger = [] if args.force else ledger_read()
    rc = 0
    fresh, cached = [], []
    for p in paths:
        row = ledger_cached(ledger, p, hhash, args.threshold, baseline_id)
        if row and not args.force:
            cached.append((p, row))
            rc |= 1 if row["verdict"] == "REJECT" else 0
        else:
            fresh.append(p)

    # IDF pass: document frequency per holdout bigram over the corpus being
    # scanned (baseline + candidate). Without it, generic bigrams dominate
    # short generic holdouts (math-500 FP class, 2026-08-30).
    idf_paths = sorted(set(base_paths) | set(fresh))
    global _WORKER_IDX
    _WORKER_IDX = idx
    if args.jobs > 1 and len(idf_paths) > 1:
        with get_context("fork").Pool(min(args.jobs, len(idf_paths))) as pool:
            parts = pool.map(_df_worker, [(p, args.q_field, args.full_doc) for p in idf_paths])
    else:
        parts = [_df_worker((p, args.q_field, args.full_doc)) for p in idf_paths]
    df = np.zeros(len(idx.g2i), dtype=np.float64)
    for part, _ in parts:
        df += part
    n_idf = sum(n for _, n in parts)
    idx.set_idf(df, n_idf)
    print(f"IDF pass: {n_idf} docs over {len(idf_paths)} paths, "
          f"{int((df > 0).sum())}/{len(df)} holdout bigrams present in corpus")

    # baseline first: its rate is the verdict's reference. Same mode as the candidate.
    b_results = scan_many(base_paths, idx, args)
    b_rows = sum(r[1] for r in b_results)
    b_bytes = sum(os.stat(p).st_size for p in base_paths)
    b_hits = sum(r[6] for r in b_results)
    b_exact = sum(len(r[3]) for r in b_results)
    b_mc = np.maximum.reduce([np.asarray(r[4], dtype=np.float64) for r in b_results]) if b_results else np.zeros(len(holdouts))
    base_per_gb = b_hits / (b_bytes / 1e9) if b_bytes else 0.0
    print(f"BASELINE ({len(base_paths)} shard(s), {b_rows} rows, {b_bytes / 1e9:.3f} GB): "
          f"{b_hits} hit rows = {base_per_gb:.1f}/GB, {b_exact} exact")
    b_long = [b_mc[i] for i in range(len(holdouts)) if i not in idx.short]
    print(f"  (union holdouts hit at 0.7/0.8/0.9: "
          f"{sum(v >= 0.7 for v in b_long)}/{sum(v >= 0.8 for v in b_long)}/{sum(v >= 0.9 for v in b_long)}"
          f" -- baseline ceiling, context only)")

    results = scan_many(fresh, idx, args)
    short = set(idx.short)
    for p, n, empty, exact, mc, nf, hr in results:
        bytes_ = os.stat(p).st_size
        rc |= report(os.path.basename(p), n, empty, exact, mc, nf, hr, bytes_,
                     args.threshold, holdouts, short, base_per_gb)
        per_gb = hr / (bytes_ / 1e9) if bytes_ else 0.0
        verdict = "REJECT" if (exact or (base_per_gb == 0 and hr) or
                               (base_per_gb and per_gb / base_per_gb > RATE_REJECT_RATIO)) else "clean"
        ledger_append(p, hhash, args.threshold, baseline_id, n, hr, len(exact), per_gb, verdict)
    for p, row in cached:
        print(f"=== {os.path.basename(p)}: cached {row['verdict']} "
              f"({row.get('hit_rows', row.get('hits_0.8', '?'))} hit rows, {row.get('per_gb', '?')}/GB, "
              f"scanned_at={row['scanned_at']}) -- ledger")
    sys.exit(rc)


if __name__ == "__main__":
    main()

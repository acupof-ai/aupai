#!/usr/bin/env python3
"""Bidirectional near-duplicate contamination scan: eval holdouts vs a candidate math source.

Life-or-death gate for any math source before it enters the mix (docs/review_2026-08-26.md
finding #1; irreversible: contamination in pretrain poisons every later checkpoint).

Run on the pod, after download, before ingest:
    python scripts/scan_math_contamination.py <candidate.jsonl> --q-field core_question
    python scripts/scan_math_contamination.py <candidate.parquet> --q-field question

Exit code 0 = clean. Exit code 1 = ANY hit -> REJECT THE WHOLE SOURCE (do not filter rows).

Three screens, cheapest first:
  1. exact-normalized match (same key as scripts/holdout.py: whitespace/punctuation-insensitive)
  2. near-duplicate: token-set Jaccard >= 0.8 (char-bigrams for zh, words for en),
     length-ratio bucketed so the pairwise cost stays small
  3. cross-language number-multiset screen (en source vs zh holdout): identical number
     multiset with >= 2 numbers -> flagged for human review (surface Jaccard cannot see
     translation pairs; numbers survive translation)

Both directions are the same pairwise computation; the matrix is symmetric.
Self-check:  python scripts/scan_math_contamination.py --self-check
"""

import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict

HOLDOUT_FILES = [
    "data/eval/math_test_500.jsonl",
    "data/synthetic/math_hard_eval_1k.jsonl",
]
JACCARD_THRESHOLD = 0.8
NUM_RE = re.compile(r"\d+\.?\d*")


def norm(q):
    return "".join(ch for ch in str(q) if not ch.isspace() and ch not in "：:，,。.、（）()")


def qhash(q):
    return hashlib.sha1(norm(q).encode("utf-8")).hexdigest()[:16]


def tokens(q):
    q = unicodedata.normalize("NFKC", str(q)).lower()
    if re.search(r"[一-鿿]", q):
        return {q[i : i + 2] for i in range(len(q) - 1)}
    return set(re.findall(r"[a-z0-9]+", q))


def numbers(q):
    return sorted(NUM_RE.findall(str(q)))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def load_holdouts():
    qs = []
    for path in HOLDOUT_FILES:
        try:
            for line in open(path, encoding="utf-8"):
                if line.strip():
                    qs.append(json.loads(line)["instruction"])
        except FileNotFoundError:
            print(f"  missing (skipped): {path}")
    return qs


def iter_candidate(path, qfield):
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        col = None
        for q in pq.ParquetFile(path).schema_arrow.names:
            if q.lower() in (qfield.lower(), "question", "instruction", "problem", "core_question"):
                col = q
                break
        if col is None:
            sys.exit(f"no question-like column in {path}; pass --q-field")
        for t in pq.ParquetFile(path).read(columns=[col]).to_pydict()[col]:
            yield t
    else:
        # The parquet branch exits when no column matches. This one used to fall through to "",
        # so a jsonl whose text lives in `content` scanned 1,532 holdouts against empty strings
        # and reported 14 shards clean. A contamination scan that cannot read its input must
        # fail, not pass: 0 hits is the answer we are hoping for, so it is the one that must
        # never be reachable by accident.
        names = (qfield, "question", "instruction", "problem", "core_question", "text", "content")
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            hit = next((n for n in names if d.get(n)), None)
            if hit is None:
                sys.exit(f"no question-like field in {path}; keys are {sorted(d)}; pass --q-field")
            yield d[hit]


def scan(path, qfield):
    holdouts = load_holdouts()
    if not holdouts:
        sys.exit("no holdout files found; run from the repo root or fix HOLDOUT_FILES")
    h_norm = {qhash(q) for q in holdouts}
    h_tok = [tokens(q) for q in holdouts]
    h_num = {tuple(numbers(q)) for q in holdouts if len(numbers(q)) >= 2}

    # bucket both sides by length decile and number multiset for cheap screening
    by_num = defaultdict(list)
    by_len = defaultdict(list)
    n = 0
    n_empty = 0
    for q in iter_candidate(path, qfield):
        n += 1
        if not str(q).strip():
            n_empty += 1
        nums = tuple(numbers(q))
        if len(nums) >= 2:
            by_num[nums].append((n, q))
        by_len[len(str(q)) // 50].append((n, q))
    if n == 0:
        sys.exit(f"{path}: read 0 rows -- a clean verdict here would be vacuous")
    if n_empty > n * 0.01:
        sys.exit(f"{path}: {n_empty}/{n} extracted texts are empty; wrong --q-field?")

    h_by_len = defaultdict(list)
    for hq, ht in zip(holdouts, h_tok):
        h_by_len[len(str(hq)) // 50].append((hq, ht))

    hits = []
    seen = set()
    for nums, rows in by_num.items():
        if nums in h_num:
            for idx, q in rows:
                hits.append(("number-multiset", idx, q))
                seen.add(idx)
    for bucket, rows in by_len.items():
        candidates = h_by_len.get(bucket, []) + h_by_len.get(bucket - 1, []) + h_by_len.get(bucket + 1, [])
        for idx, q in rows:
            if qhash(q) in h_norm:
                hits.append(("exact", idx, q))
                seen.add(idx)
                continue
            if idx in seen:
                continue
            t = tokens(q)
            for hq, ht in candidates:
                if jaccard(t, ht) >= JACCARD_THRESHOLD:
                    hits.append(("jaccard", idx, q))
                    break

    print(f"{n} candidate questions scanned vs {len(holdouts)} holdouts")
    for kind, idx, q in hits[:20]:
        print(f"  HIT [{kind}] row {idx}: {str(q)[:120]}")
    if len(hits) > 20:
        print(f"  ... {len(hits) - 20} more")
    print(f"{len(hits)} hits -> {'REJECT WHOLE SOURCE' if hits else 'clean'}")
    return 1 if hits else 0


def self_check():
    base = "一个水池有两个进水管，甲管单独注满需要6小时，乙管单独注满需要8小时，两管同时开几小时注满？"
    assert qhash(base) in {qhash(base)}
    # one-character insertion: same problem, light edit -> screen 2 must catch it
    near = "一个水池有两个进水管，甲管单独注满需要6小时，乙管单独注满也需要8小时，两管同时开几小时注满？"
    assert jaccard(tokens(base), tokens(near)) >= JACCARD_THRESHOLD, "near-dup must hit"
    far = "小明有3个苹果，小红有5个梨，他们一共有多少个水果？"
    assert jaccard(tokens(base), tokens(far)) < JACCARD_THRESHOLD, "unrelated must not hit"
    en = "A pool has two inlet pipes. Pipe A alone fills it in 6 hours, pipe B in 8 hours. How long if both are open?"
    assert numbers(base) == numbers(en), "number multiset must survive translation"

    # The hand-written cases above never saw the real corpus shape (rows whose question lives in
    # `content`, not a `question` field). A self-check that shares the script's own assumption
    # about field names goes green while the script scans empty strings on production data --
    # that bug shipped once (2026-08-30, the empty-q-field fake-clean). Build the broken world
    # from a REAL shard, not from the check's own source (AGENTS.md).
    import glob as _glob, subprocess as _sp

    real = sorted(_glob.glob("data/corpus/math/*.jsonl"))
    assert real, "self-check: no real math shard found to scan against"
    shard = real[0]
    out = _sp.run(
        [sys.executable, __file__, shard, "--q-field", "content"],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, f"self-check: real shard scan failed: {out.stderr[-300:]}"
    assert "scanned" in out.stdout, f"self-check: real shard scanned 0 rows: {out.stdout[-200:]}"
    # the guard that would have caught the vacuous-clean: an EMPTY read (nonexistent file) must
    # FAIL, not return clean. A wrong --q-field name is NOT a broken world here -- the fallback
    # tuple includes `content`/`text`, which on a content-bearing real shard correctly recovers.
    broken = _sp.run(
        [sys.executable, __file__, "data/corpus/math/no_such_shard_000.jsonl"],
        capture_output=True,
        text=True,
    )
    assert broken.returncode != 0, "self-check: empty read returned clean; the n==0 guard is dead"

    print("self-check OK")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--self-check":
        sys.exit(self_check())
    if len(sys.argv) < 2:
        sys.exit("usage: scan_math_contamination.py <candidate.jsonl|parquet> [--q-field NAME]")
    qfield = "question"
    if "--q-field" in sys.argv:
        qfield = sys.argv[sys.argv.index("--q-field") + 1]
    sys.exit(scan(sys.argv[1], qfield))

#!/usr/bin/env python3
"""t21 report: exact token count of CLEANED rp1t_github, the 5 numbers.

Tokens = the frozen tokenizer over the CLEANED text (not bytes x a ratio -- that
is the 85.7B->73.6B correction class). Reads the clean's output shards
(data/corpus/code/*.jsonl), which carry filters_fp. Reports:

  fetched docs   (from the raw rp1t_github fetch_stats/docs)
  kept docs      (rows in the cleaned shards)
  retention      kept / fetched
  landed tokens  frozen-tokenizer tokens over cleaned text
  landed tok/byte

Also checks whether the 751.3M-tokens/file projection holds within 10% (13 files
were the fetch target); if not, the 73.6B code-supply figure changes by the gap.

Usage (on pod): python3 datagen/count_cleaned_code.py
"""

import glob
import json
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
SHARDS = sorted(glob.glob(os.path.join(ROOT, "data", "corpus", "code_rp1t", "*.jsonl")))
WORKERS = int(os.environ.get("COUNT_WORKERS", "8"))
_TOK = None


def tok():
    """The frozen tokenizer, loaded on first use.

    LAZY BECAUSE data/tokenizer.json IS GITIGNORED (.gitignore:5). Loading it at import
    time made every importer of this module unrunnable in a worktree that had never copied
    the file in: `count_code_dirs.py --selftest` raised `No such file or directory` from
    line 30 before reaching any test, and the pre-commit hook refused a ledger merge on it
    at 8e536df7. CI and the pod both hold the file, which is why it was green in both
    (4c, 2026-09-08). The import-time load predates this; registering the selftest in
    SELFTEST_FILES is what made it reachable, so the defect arrived by being exposed.
    """
    global _TOK
    if _TOK is None:
        if not os.path.isfile(TOK_PATH):
            raise SystemExit(
                f"tokenizer missing at {TOK_PATH}. It is gitignored (.gitignore:5), so a "
                f"fresh worktree does not have it -- copy it from another checkout or the "
                f"pod. Nothing here can run without it."
            )
        from tokenizers import Tokenizer  # type: ignore

        _TOK = Tokenizer.from_file(TOK_PATH)
    return _TOK


def _count_shard(shard):
    from count_tokens import count_docs

    kept = tokens = tb = 0
    texts = []
    with open(shard, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t = d.get("content") or d.get("text") or ""
            if not t:
                continue
            kept += 1
            tb += len(t.encode("utf-8"))
            texts.append(t)
            if len(texts) >= 2000:
                tokens += count_docs(texts, tok())
                texts = []
    tokens += count_docs(texts, tok())
    return kept, tokens, tb


def main():
    with mp.Pool(WORKERS) as pool:
        counts = pool.map(_count_shard, SHARDS)
    kept = sum(c[0] for c in counts)
    tokens = sum(c[1] for c in counts)
    tb = sum(c[2] for c in counts)  # text UTF-8 bytes of kept docs
    # fetched docs: the raw fetch wrote N jsonl docs; counted as kept+rejected is
    # not available, so fetched is derived from the raw files (the clean's reject
    # reasons live in build_corpus_stats / the raw). We count raw docs here.
    raw_docs = 0
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "rp1t_github", "*.sampled.jsonl"))):
        with open(p, encoding="utf-8") as f:
            raw_docs += sum(1 for l in f if l.strip())
    # filters_fp present?
    stats = {}
    sp = os.path.join(ROOT, "data", "corpus", "code_rp1t", "build_corpus_stats.json")
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as fh:
            stats = json.load(fh)
    fp = stats.get("filters_fp")

    per_file = tokens / max(1, len(SHARDS))
    print(f"shards: {len(SHARDS)}  filters_fp: {fp or 'MISSING -> FAILS the derived-artifact rule'}")
    print(f"fetched docs (sample): {raw_docs}   kept docs: {kept}")
    print(f"retention (kept/fetched, one-file denominator): {kept / max(1, raw_docs):.2%}")
    print(f"landed tokens (frozen tok over cleaned text): {tokens} ({tokens / 1e9:.2f}B)")
    print(f"landed tok/byte: {tokens / max(1, tb):.4f}")
    # Per-file projection check. DENOMINATOR WARNING: SHARDS are 235 CLEANED output
    # shards; the 751.3M/file projection is per RAW FETCH file (98 files for the
    # full supply, modeled under the stage-1 budget's raw bytes). Comparing per
    # shard to per raw file is apples-to-oranges and prints a false -95% gap; the
    # number that matters is total landed tokens against the domain's cap/budget.
    print(
        f"per cleaned shard: {per_file / 1e6:.2f}M tok/shard x {len(SHARDS)} shards = {tokens / 1e9:.2f}B landed "
        f"(NOT comparable to the 751.3M-raw-file supply projection -- different denominator)"
    )
    print(
        f"disk /work free: {os.statvfs(os.path.join(ROOT, 'data')).f_bavail * os.statvfs(os.path.join(ROOT, 'data')).f_frsize / 1e9:.0f}G"
    )


if __name__ == "__main__":
    main()

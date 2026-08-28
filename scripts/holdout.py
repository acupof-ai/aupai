#!/usr/bin/env python3
"""The eval holdout, as a hash set every data path must exclude.

Finding #1 of docs/review_2026-08-26.md: stage-1 SFT trained on the full Belle-derived
corpus that the 500 eval problems were drawn from, with no exclusion anywhere and
no script that produced the split. Every reported accuracy before this file existed
may be partly memorization.

`python scripts/holdout.py` regenerates data/eval/holdout_hashes.txt from the eval
files; importers use `is_holdout(question)`.
"""
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_FILES = [
    os.path.join(ROOT, "data", "eval", "math_test_500.jsonl"),
    os.path.join(ROOT, "data", "synthetic", "math_hard_eval_1k.jsonl"),
]
HASH_PATH = os.path.join(ROOT, "data", "eval", "holdout_hashes.txt")


def norm(q):
    """Whitespace- and punctuation-insensitive key, so a reformatted copy still matches."""
    return "".join(ch for ch in str(q) if not ch.isspace() and ch not in "：:，,。.、（）()")


def qhash(q):
    return hashlib.sha1(norm(q).encode("utf-8")).hexdigest()[:16]


def load():
    if not os.path.exists(HASH_PATH):
        return set()
    return {l.strip() for l in open(HASH_PATH, encoding="utf-8") if l.strip()}


_CACHE = None


def is_holdout(q):
    global _CACHE
    if _CACHE is None:
        _CACHE = load()
    return qhash(q) in _CACHE


def main():
    hs = set()
    for path in EVAL_FILES:
        if not os.path.exists(path):
            print(f"  missing (skipped): {path}")
            continue
        n = 0
        for line in open(path, encoding="utf-8"):
            if line.strip():
                hs.add(qhash(json.loads(line)["instruction"]))
                n += 1
        print(f"  {os.path.basename(path)}: {n}")
    os.makedirs(os.path.dirname(HASH_PATH), exist_ok=True)
    with open(HASH_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(hs)) + "\n")
    print(f"{len(hs)} unique holdout hashes -> {HASH_PATH}")


if __name__ == "__main__":
    main()

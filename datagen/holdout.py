#!/usr/bin/env python3
"""The eval holdout, as a hash set every data path must exclude.

`python datagen/holdout.py` regenerates data/eval/holdout_hashes.txt from the eval
files; importers use `is_holdout(question)`.

Fail-closed: the hash file carries a fingerprint of the eval files that produced
it. is_holdout() RAISES if the file is missing, predates fingerprints, or is
stale (an eval file changed since regeneration). A guard that silently returns
False is indistinguishable from no guard at all -- the 2026-08-30 sft_all.pt
contamination (19/20 holdout questions packed because the hash set was empty)
is the reason this is loud, not silent.
"""

import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_FILES = [
    os.path.join(ROOT, "data", "eval", "math_test_500.jsonl"),
    os.path.join(ROOT, "data", "synthetic", "math_hard_eval_1k.jsonl"),
    os.path.join(ROOT, "data", "eval", "code_holdout_500.jsonl"),
    os.path.join(ROOT, "data", "eval", "code_holdout_v2_500.jsonl"),
]
HASH_PATH = os.path.join(ROOT, "data", "eval", "holdout_hashes.txt")


def norm(q):
    """Whitespace- and punctuation-insensitive key, so a reformatted copy still matches."""
    return "".join(ch for ch in str(q) if not ch.isspace() and ch not in "：:，,。.、（）()")


def qhash(q):
    return hashlib.sha1(norm(q).encode("utf-8")).hexdigest()[:16]


def _fingerprint():
    """sha1 of (basename, file-sha1) for every existing EVAL_FILES entry. Changes
    when an eval file is added, removed, or edited -- which is exactly when the
    hash set must be regenerated."""
    h = hashlib.sha1()
    for path in sorted(EVAL_FILES):
        if os.path.exists(path):
            h.update(os.path.basename(path).encode())
            h.update(hashlib.sha1(open(path, "rb").read()).digest())
    return h.hexdigest()[:16]


def load():
    if not os.path.exists(HASH_PATH):
        raise RuntimeError(
            f"{HASH_PATH} missing -- the holdout guard is unloaded. "
            "Run `python datagen/holdout.py` to regenerate."
        )
    lines = [l.strip() for l in open(HASH_PATH, encoding="utf-8") if l.strip()]
    fp = next((l[5:] for l in lines if l.startswith("# fp:")), None)
    if fp is None:
        raise RuntimeError(
            f"{HASH_PATH} has no fingerprint (old format) -- the guard may be stale. "
            "Run `python datagen/holdout.py` to regenerate."
        )
    if fp != _fingerprint():
        raise RuntimeError(
            f"{HASH_PATH} is stale: eval files changed since it was generated. "
            "Run `python datagen/holdout.py` to regenerate."
        )
    return {l for l in lines if not l.startswith("#")}


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
        f.write(f"# fp:{_fingerprint()}\n")
        f.write("\n".join(sorted(hs)) + "\n")
    print(f"{len(hs)} unique holdout hashes (fp {_fingerprint()}) -> {HASH_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Holdout-leak scan of the DELIVERED code_py_starcoder corpus (fb P0).
tilerl's merge catch (09-02): my parallel build_starcoder_py.py dropped the is_holdout
check (grep 0), so the delivered corpus MAY contain eval-holdout rows. This scans ALL
6.18M rows for is_holdout overlap; 0 hits = the corpus is clean (starcoder-python ~ zero
eval overlap, accidentally), >0 = needs rebuild with the holdout-checking script."""
import json
import glob
import sys

sys.path.insert(0, "/work/aupai/datagen")
import build_corpus as B  # noqa: E402


def main():
    shards = sorted(glob.glob("/work/aupai/data/corpus/code_py_starcoder/*.jsonl"))
    n = hits = 0
    for p in shards:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            n += 1
            c = json.loads(line).get("content") or ""
            if B.is_holdout(c) or B.is_holdout(B.QA_PREFIX.sub("", B.ANSWER_TAIL.split(c, 1)[0]).strip()):
                hits += 1
    print(f"FULL {n} rows | holdout HITS {hits}")


if __name__ == "__main__":
    main()
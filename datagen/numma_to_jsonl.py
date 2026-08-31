#!/usr/bin/env python3
"""NuminaMath-CoT parquet -> jsonl for build_corpus.

The parquet has no text column (source/problem/solution/messages); build_corpus's
parquet reader takes one text column, so the two-column composition happens here.
Plain problem+solution, not ChatML: build_corpus strips <|...|> control tokens,
and the pretraining chat domain already teaches the ChatML format.

Usage: python datagen/numma_to_jsonl.py <data/raw/hf_numma> <data/raw/hf_numma_jsonl>
"""
import json
import os
import sys

import pyarrow.parquet as pq


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(src)):
        if not f.endswith(".parquet"):
            continue
        out = os.path.join(dst, f.replace(".parquet", ".jsonl"))
        t = pq.read_table(os.path.join(src, f))
        cols = t.to_pydict()
        with open(out, "w", encoding="utf-8") as w:
            for problem, solution in zip(cols["problem"], cols["solution"], strict=True):
                problem, solution = (problem or "").strip(), (solution or "").strip()
                if not problem or not solution:
                    continue
                w.write(json.dumps(
                    {"content": f"{problem}\n\n{solution}", "source": "numina_cot", "url": ""},
                    ensure_ascii=False,
                ) + "\n")
                n += 1
        print(f"{f}: {len(cols['problem'])} rows -> {out}", flush=True)
    print(f"total {n} docs")


if __name__ == "__main__":
    main()

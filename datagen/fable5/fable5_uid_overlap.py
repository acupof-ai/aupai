#!/usr/bin/env python3
"""Is fable5_2m a new source, or the same rows as fable5_traces?

Answers it by uid, not by row count: the two files are different encodings of overlapping
exports, and a count comparison cannot tell a subset from a disjoint set of the same size.

    python3 datagen/fable5/fable5_uid_overlap.py
"""

import argparse
import json

import pyarrow.parquet as pq

POD = "/work/aupai"


def uids_parquet(path):
    out = set()
    f = pq.ParquetFile(path)
    for b in f.iter_batches(batch_size=4000, columns=["row_json"]):
        for r in b.to_pylist():
            try:
                d = json.loads(r["row_json"])
            except Exception:
                continue
            if "cot" in d and d.get("uid"):
                out.add(d["uid"])
    return out


def uids_jsonl(path):
    out = set()
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            try:
                u = json.loads(ln).get("uid")
            except Exception:
                continue
            if u:
                out.add(u)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=f"{POD}/data/raw/fable5_2m/data/train.parquet")
    ap.add_argument("--jsonl", default=f"{POD}/data/raw/fable5_traces/fable5_cot_merged.jsonl")
    a = ap.parse_args()
    p, j = uids_parquet(a.parquet), uids_jsonl(a.jsonl)
    print(
        json.dumps(
            {
                "parquet_cot_uids": len(p),
                "jsonl_uids": len(j),
                "overlap": len(p & j),
                "parquet_only": len(p - j),
                "jsonl_only": len(j - p),
            },
            indent=1,
        )
    )

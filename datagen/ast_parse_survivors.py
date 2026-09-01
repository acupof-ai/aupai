#!/usr/bin/env python3
"""FULL-corpus surviving count after the ast.parse filter on code_rp1t (fb P0 2026-09-01).
fb ruling: ast.parse(py3) success = Language ID AND syntax filter in one pass (Java/Ruby
do not accidentally parse py3); Python-only, no py2 normalization (py2 fails parse -> drops
for free = the filter working). THE load-bearing number: how many tokens AND rows survive
ast.parse across ALL 3,747,157 code_rp1t rows -- a count, not a sample rate (a sample
denominator already burned the mix once today). Filter chain final: ast.parse -> boilerplate
-> length floor; this counts only the ast.parse stage."""
import ast
import glob
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/work/aupai")


def _count_shard(args):
    path, = args
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file("/work/aupai/data/tokenizer.json")
    rows = rows_ok = 0
    tokens_ok = 0
    chars_ok = 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rows += 1
        t = None
        try:
            t = json.loads(line).get("content") or ""
        except json.JSONDecodeError:
            continue
        if not t:
            continue
        try:
            ast.parse(t)
        except SyntaxError:
            continue
        rows_ok += 1
        chars_ok += len(t)
        tokens_ok += len(tk.encode(t).ids)
    return rows, rows_ok, tokens_ok, chars_ok


def main():
    shards = sorted(glob.glob("/work/aupai/data/corpus/code_rp1t/*.jsonl"))
    jobs = min(16, len(shards))
    with mp.Pool(jobs) as pool:
        parts = pool.map(_count_shard, [(p,) for p in shards])
    rows = sum(p[0] for p in parts)
    rows_ok = sum(p[1] for p in parts)
    tokens_ok = sum(p[2] for p in parts)
    chars_ok = sum(p[3] for p in parts)
    print(json.dumps({
        "domain": "code_rp1t", "shards": len(shards), "total_rows": rows,
        "parse_ok_rows": rows_ok, "parse_survivor_rate": rows_ok / rows if rows else 0.0,
        "surviving_tokens": tokens_ok, "surviving_chars": chars_ok,
        "config": {"filter": "ast.parse(t) under python3; successful = python3, kept",
                   "note": "full corpus (3.7M rows), count not sample; this is ast.parse stage only, before boilerplate-drop and length floor"},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
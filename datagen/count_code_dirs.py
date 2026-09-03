#!/usr/bin/env python3
"""3b-10: frozen-tokenizer token count over a list of corpus dirs, same method as
count_cleaned_code (reuses its TOK + _count_shard), so before/after is comparable.
Usage: python3 datagen/count_code_dirs.py DIR [DIR ...]
Prints per dir: kept docs, shards, landed tokens. CPU, no GPU."""
import glob
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import count_cleaned_code as C  # noqa: E402


def count_dir(d):
    shards = sorted(glob.glob(os.path.join(C.ROOT, "data", "corpus", d, "*.jsonl")))
    kept = tokens = 0
    done = 0
    with mp.Pool(C.WORKERS) as pool:
        for c in pool.imap_unordered(C._count_shard, shards):
            kept += c[0]; tokens += c[1]
            done += 1
            if done % 50 == 0:
                print(f"{d}: {done}/{len(shards)} shards done, {tokens/1e9:.3f}B tokens so far", flush=True)
    print(f"{d}: shards={len(shards)} kept_docs={kept} landed_tokens={tokens} ({tokens/1e9:.3f}B)", flush=True)
    return d, len(shards), kept, tokens


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        import tempfile
        import shutil
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "s.jsonl")
            rows = [{"content": "def add(a,b): return a+b  # tests" * 4} for _ in range(10)]
            with open(p, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            kept, tok, tb = C._count_shard(p)
            assert kept == 10 and tb > 0
            expect = sum(len(C.TOK.encode(r["content"]).ids) for r in rows)
            assert tok == expect and tok > 100, (tok, expect)
            print(f"selftest OK: 10 fake docs, {tok} tokens == manual encode")
        finally:
            shutil.rmtree(d)
        sys.exit(0)
    for d in sys.argv[1:]:
        count_dir(d)
    print("ALL-DONE")
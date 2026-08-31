#!/usr/bin/env python3
"""t50 check: _encode_domain(workers=N) is element-identical to the single-process encode().

Run on the pod (needs data/tokenizer.json + the math_seed corpus):
    python3 scripts/test_parallel_encode.py
"""
import glob
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("RAYON_NUM_THREADS", "4")  # 4 workers x 4 threads = 16 total
import train  # noqa: E402


def main():
    tok = train.build_tokenizer([])
    shards = sorted(glob.glob(os.path.join(train.DATA, "corpus", "math_seed", "*.jsonl")))
    assert shards, "math_seed corpus missing"
    texts = [json.loads(ln)["content"] for ln in open(shards[0], encoding="utf-8") if ln.strip()][:2000]
    random.Random(0).shuffle(texts)
    # workers=4 FIRST: a process that has already encoded holds a live rayon pool,
    # and fork()ing one deadlocks the child in the pool's inherited locks. The
    # parallel call must precede any in-process encode (pretokenize.py satisfies
    # this by construction -- the parent never encodes when workers>1).
    four = train._encode_domain(texts, tok, 4)
    one = train._encode_domain(texts, tok, 1)
    assert one.dtype == four.dtype == torch.int32, (one.dtype, four.dtype)
    assert torch.equal(one, four), f"streams differ: {one.shape} vs {four.shape}"
    print(f"OK: workers=1 and workers=4 identical ({one.numel()} tokens, {len(texts)} docs)")


if __name__ == "__main__":
    main()

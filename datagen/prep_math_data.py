#!/usr/bin/env python3
"""Stage-2 math SFT data prep: hold out 500, dedup, normalize, sample, contamination check."""
import hashlib
import json
import os
import random
import re

random.seed(42)
ROOT = "/work/aupai"
os.makedirs(f"{ROOT}/data/eval", exist_ok=True)
os.makedirs(f"{ROOT}/data/workbatch", exist_ok=True)

rows = [json.loads(l) for l in open(f"{ROOT}/data/school_math_r1_zh.jsonl", encoding="utf-8")]
hold = set(random.sample(range(len(rows)), 500))
with open(f"{ROOT}/data/eval/math_test_500.jsonl", "w", encoding="utf-8") as f:
    for i in sorted(hold):
        f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")

seen, kept = set(), 0
with open(f"{ROOT}/data/workbatch/school_math_train.jsonl", "w", encoding="utf-8") as f:
    for i, r in enumerate(rows):
        if i in hold:
            continue
        h = hashlib.md5(r["instruction"].strip().encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        kept += 1
print(f"school_math train (deduped): {kept}", flush=True)

with open(f"{ROOT}/data/gsm8k_zh.jsonl", encoding="utf-8") as f, \
     open(f"{ROOT}/data/workbatch/gsm8k_zh_train.jsonl", "w", encoding="utf-8") as g:
    n = 0
    for line in f:
        r = json.loads(line)
        r["output"] = re.sub(r"####\s*", "答案是：", r["output"])
        g.write(json.dumps(r, ensure_ascii=False) + "\n")
        n += 1
print(f"gsm8k_zh normalized: {n}", flush=True)

coig = [json.loads(l) for l in open(f"{ROOT}/data/coig.jsonl", encoding="utf-8")]
with open(f"{ROOT}/data/workbatch/coig_50k.jsonl", "w", encoding="utf-8") as f:
    for r in random.sample(coig, 50_000):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("coig sampled: 50000", flush=True)

rlvr = set()
for line in open(f"{ROOT}/data/rl/rlvr_math.jsonl", encoding="utf-8"):
    r = json.loads(line)
    rlvr.add(hashlib.md5(r["prompt"].strip().encode()).hexdigest())
contam = sum(
    1 for i in sorted(hold)
    if hashlib.md5(rows[i]["instruction"].strip().encode()).hexdigest() in rlvr
)
print(f"CONTAMINATION: {contam}/500 holdout rows are in rlvr_math.jsonl", flush=True)

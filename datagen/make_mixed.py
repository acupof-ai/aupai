#!/usr/bin/env python3
"""Create mixed SFT+pretrain dataset (1:1 ratio) to prevent catastrophic forgetting.
Run on pod after SFT data is in place.

Usage: python3 make_mixed.py
Output: /work/aupai/data/sft/sft_mixed.jsonl
"""

import json, random

random.seed(42)

SFT_PATH = "/work/aupai/data/sft/sft_all.jsonl"
PRETRAIN_PATH = "/work/aupai/data/mix/mixed_v3.jsonl"
OUT_PATH = "/work/aupai/data/sft/sft_mixed.jsonl"

sft = []
with open(SFT_PATH) as f:
    for line in f:
        d = json.loads(line)
        text = f"问：{d['instruction']}\n答：{d['output']}"
        sft.append({"content": text})
print(f"SFT: {len(sft)}")

pre = []
with open(PRETRAIN_PATH) as f:
    lines = f.readlines()
random.shuffle(lines)
for line in lines[: len(sft)]:
    d = json.loads(line)
    pre.append({"content": d["content"]})
print(f"Pretrain sample: {len(pre)}")

mixed = sft + pre
random.shuffle(mixed)
with open(OUT_PATH, "w") as f:
    for d in mixed:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"Mixed: {len(mixed)} → {OUT_PATH}")

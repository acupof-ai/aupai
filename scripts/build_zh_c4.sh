#!/bin/bash
# 0e-6: build zh_c4_dc from chinese-c4 jsonl.zst shards, gate vocab, CPU.
set -u
cd /work/aupai
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=/work/aupai
python3 -u datagen/build_zh_domain.py \
  --src '/data00/aupai_raw/zh_c4/chinese-c4-*.jsonl.zst' \
  --out data/corpus/zh_c4_dc \
  --content-key text \
  --source AI-ModelScope/chinese-c4 \
  --target-tokens ${ZH_C4_TARGET:-530000000}

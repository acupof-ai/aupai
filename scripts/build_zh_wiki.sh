#!/bin/bash
# 0e-6: build zh_wiki_dc from fetched wikipedia-cn, gate vocab, CPU.
set -u
cd /work/aupai
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=/work/aupai
python3 -u datagen/build_zh_domain.py \
  --src /data00/aupai_raw/zhwiki/wikipedia-cn.json \
  --out data/corpus/zh_wiki_dc \
  --target-tokens ${ZH_TARGET:-800000000}

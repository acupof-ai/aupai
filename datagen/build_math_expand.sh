#!/usr/bin/env bash
# Reproduce the math-domain corpus expansion: ~0.07B tokens of new math (LF/dedup/
# length-filtered, holdout-clean) into data/corpus/math/.
#   - math_short_v6/v7/v8: fresh-seed mathbank drain (seed + instance-cap; different
#     seeds -> non-overlapping instruction pools)
#   - ape210k / math23k / mxode / gsm8k_zh: HF fetch via datagen/fetch_math_data.py
#   - math_short_sol_v1: short-solution line from mathbank/run_short_sol.py
# belle is skipped on purpose: it duplicates school_math_r1_zh already in corpus.
#
# Rerun from scratch:
#   cd mathbank && python run_math_short.py 200000 ../data/synthetic/math_short_v6.jsonl --seed 20260827
#   cd mathbank && python run_math_short.py 200000 ../data/synthetic/math_short_v7.jsonl --seed 31337
#   cd mathbank && python run_math_short.py 200000 ../data/synthetic/math_short_v8.jsonl --seed 999983
#   python datagen/fetch_math_data.py ape210k math23k mxode gsm8k_zh
#   python mathbank/run_short_sol.py 12000 data/synthetic/math_short_sol_v1.jsonl --seed 7
#   bash datagen/build_math_expand.sh

set -euo pipefail
cd "$(dirname "$0")/.."
MATH=math
# Gate: no math_short batch enters the corpus contaminated (finding 2026-08-30:
# the bank shares template DNA with math_hard_eval_1k; v3/v5/v6/v7/v8/v10/v11 all
# REJECT at 0.8). This WILL refuse until the bank is fixed -- that is the point.
bash eval/gate_math_short.sh \
    data/synthetic/math_short_v6.jsonl \
    data/synthetic/math_short_v7.jsonl \
    data/synthetic/math_short_v8.jsonl \
    data/synthetic/math_short_sol_v1.jsonl
rm -rf "data/corpus/$MATH"
python datagen/build_corpus.py --domain "$MATH" \
  --source "jsonl:data/synthetic/math_short_v6.jsonl" \
  --source "jsonl:data/synthetic/math_short_v7.jsonl" \
  --source "jsonl:data/synthetic/math_short_v8.jsonl" \
  --source "jsonl:data/math/ape210k.jsonl" \
  --source "jsonl:data/math/math23k.jsonl" \
  --source "jsonl:data/math/mxode.jsonl" \
  --source "jsonl:data/math/gsm8k_zh.jsonl" \
  --source "jsonl:data/synthetic/math_short_sol_v1.jsonl" \
  --target_tokens 0.5e9 --filters light --no_near_dedup
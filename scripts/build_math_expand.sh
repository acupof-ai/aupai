#!/usr/bin/env bash
# Reproduce the math-domain corpus expansion (2026-08-27).
#
# Adds ~0.07B tokens of new math (LF/dedup/length-filtered, holdout-clean):
#   - math_short_v6/v7/v8: fresh-seed mathbank drain (reproducible from mathbank/,
#     seed + instance-cap; different seeds -> non-overlapping instruction pools)
#   - ape210k / math23k / mxode / gsm8k_zh: HF fetch via scripts/fetch_math_data.py
#   - math_short_sol_v1: short-solution line from mathbank/run_short_sol.py
# belle is skipped on purpose: it reproduces school_math_r1_zh already in corpus.
#
# Output: data/corpus/math/  (~0.07B tokens, 530K docs). aupai-01's corpus cache
# freshness check re-tokenizes automatically when these files change.
#
# Rerun from scratch:
#   cd mathbank && python run_math_short.py 200000 ../data/synthetic/math_short_v6.jsonl --seed 20260827
#   cd mathbank && python run_math_short.py 200000 ../data/synthetic/math_short_v7.jsonl --seed 31337
#   cd mathbank && python run_math_short.py 200000 ../data/synthetic/math_short_v8.jsonl --seed 999983
#   python scripts/fetch_math_data.py ape210k math23k mxode gsm8k_zh
#   python mathbank/run_short_sol.py 12000 data/synthetic/math_short_sol_v1.jsonl --seed 7
#   bash scripts/build_math_expand.sh

set -euo pipefail
cd "$(dirname "$0")/.."
MATH=math
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
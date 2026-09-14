#!/bin/bash
# Timing calibration ONLY -- do NOT run until r3 finishes (fb 2026-09-14, cards
# belong to the v41_r3 block). One card, single-sample greedy, HumanEval 164 and
# MBPP 427. Read per-task seconds off the progress lines; multiply by n (10) and
# the task count to budget the E0/ET/EC n=10 T=0.2 stage-2 evals. GPU greedy is
# the floor; temperature sampling at n=10 is ~n times serial work with this scorer.
#
# Usage (after run end, on the granted lane card):
#   CUDA_VISIBLE_DEVICES=1 bash runs/stage2_timing_calib.sh <ckpt>
set -u
CKPT=${1:?"usage: stage2_timing_calib.sh <ckpt>"}
cd "$(dirname "$0")/.."
source eval/_devs.sh 1
CARD=${_DEVS[0]}
echo "=== HumanEval 164 greedy rstrip (1 card GPU $CARD) ==="
python3 eval/humaneval_gen.py --ckpt "$CKPT" --device cuda:0 --rstrip_nl \
    --run timing_he_greedy_1card --force

echo "=== MBPP 427 greedy sig-rstrip (1 card GPU $CARD) ==="
python3 eval/mbpp_gen.py --data data/eval/sanitized-mbpp.json --ckpt "$CKPT" \
    --device cuda:0 --run timing_mbpp_greedy_1card --force

echo "Stage-2 budget: wall_per_task_greedy * n(10) * tasks, per checkpoint x 3 (E0/ET/EC)."

#!/bin/bash
# Shape-fault probe on CARD 5. Grant: runs/card_assignment.json cards[5], "GRANTED
# 2026-09-05T04:34Z -> e1: kda_shape_fault probe, then control floor", ruling sha 86314a7e.
#
# --cards 5, not a derived lane. lane_card is "5,7" and the launcher takes the first card
# measured idle, which is tilerl's card 7 until their P1 rerun starts (d4296bb9 refuses that
# now and names this flag instead). The probe claims by its own pid inside harness launch.
#
# A SCRIPT RATHER THAN AN INLINE pod COMMAND: the hypothesis text contains parentheses and
# `~/bin/pod` passes the command through a shell, which failed with "syntax error near
# unexpected token `('". Quoting it correctly through two shells is a second thing to get
# right for no benefit.
set -euo pipefail
cd /work/aupai
python3 /work/aupai/scripts/harness.py launch e1_kda_shape_fault \
  --cards 5 \
  --hypothesis "the card-7 misaligned-address crash is a SHAPE fault -- the autotuner benchmarking a per-rank shape the arms never ran -- not a model or weights fault" \
  --output /work/aupai/runs/kda_shape_fault.json \
  -- probes/kda_shape_fault.py --device cuda:0 --allow_cuda \
     --out /work/aupai/runs/kda_shape_fault.json

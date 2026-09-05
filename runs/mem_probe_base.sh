#!/bin/bash
# TEMPLATE for one cell of the six-cell readout-4 probe (4c's ruling 2026-09-05, user order 07:40).
# Copy per cell and set the five __PLACEHOLDER__ values from the table below; nothing else changes.
#
# WHAT THIS PROBES. M1/M2/M3 were stopped under readout 4 with key-usage collapse (M1
# pool_touched_frac 0.0945 at step 1000, key_gini 0.9192, topk_entropy 0.927 of ln 32 = 3.466).
# Readout 6 was healthy throughout (rows_changed_since_prev 1.09-1.25 of touched rows), so the
# value writes land and the defect is in WHICH rows get selected. Each cell runs 300 steps of M1's
# own launch line with one candidate change to the selection path.
#
# ONE CARD, world 1, and that costs one comparison: the diag block MAX-reduces `touched` across
# ranks at world 2, so a world-1 window sees half the tokens. These cells are therefore NOT
# comparable to M1's 0.306 / 0.214 / 0.137 -- read them against mem_probe_base on card 0, which is
# the world-1 control. If a winner passes here, 4c wants one 300-step world-2 confirmation on cards
# 1+2 before amendment 12, so the quoted number is at the arm's own scale.
#
# --mem_arm probe_<cell>, NOT m1 (4c's ruling). M1's line carries --mem_arm m1 and train.py refuses
# --mem_values without one, so inheriting it would write all six cells' rows into
# runs/memory_diag.jsonl under name "m1" -- append-only, folded on (name, step), at the same step
# numbers M1 already occupies. M1's readout-4 curve would become unreadable and nothing could undo
# it. probe_<cell> keeps every cell separable and outside the m1/m2/m3 namespace; memory_diag's
# _arm_key does not recognise it, which is correct -- these are probes, not arms, and the run names
# mem_probe_* deliberately do not match harness's arm regex either (verified: all six False).
#
# --save_every 100000 so a 300-step run writes NO checkpoint. Six 7.4 GiB checkpoints would be
# 44 GiB on a /work already at 89%.
#
# ABSOLUTE PATHS ONLY: a `cd ... &` shape does not reach the backgrounded half, so a relative path
# resolves under /sgl-workspace/sglang and the redirect silently writes nothing (pod shape 166).
# Launch with: setsid bash /work/aupai/runs/mem_probe_<cell>.sh > /work/aupai/runs/mem_probe_<cell>.wrap.log 2>&1 &
#
# CELL TABLE -- cell / card / port / mem_arm / flags
#   base      0  29510  probe_base     (none)
#   sel2      1  29511  probe_sel2     --mem_sel_lr 0.002
#   sel3      2  29512  probe_sel3     --mem_sel_lr 0.0002
#   l2        3  29513  probe_l2       --mem_query_norm l2
#   bn        4  29514  probe_bn       --mem_query_norm bn
#   sel2_l2   6  29516  probe_sel2_l2  --mem_sel_lr 0.002 --mem_query_norm l2
#
# THE CARD CLAIM. harness launch's device poll is 90 s and the 155 GiB cache load outlasts it, so
# the claim does not land on its own (measured on M1, M2 and M3). Claim by hand the moment the card
# shows memory, against a RANK pid, not the wrapper:
#   python3 scripts/card_claim.py acquire --name mem_probe_<cell> --cards <card> \
#     --pid <rank pid> --require-device --note "readout-4 probe cell <cell>"
set -u
cd /work/aupai || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# THE CARD COMES FROM THE CALLER, never from this file. CUDA_VISIBLE_DEVICES is not additive:
# assigning it here REPLACES whatever lane the launcher confined this run to, which is how a
# lane-card launch landed on a training-block card on 2026-08-31 (2f97e4a). The cell table above
# says which card each cell is FOR; the launcher is what puts it there:
#   CUDA_VISIBLE_DEVICES=0 setsid bash /work/aupai/runs/mem_probe_base.sh \
#     > /work/aupai/runs/mem_probe_base.wrap.log 2>&1 &
# Refused rather than defaulted: a probe that silently picks a card is the same escape by another
# route, and the six cells run concurrently on six different cards.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "refusing: mem_probe_base needs CUDA_VISIBLE_DEVICES from the caller (cell base is for card 0)." >&2
  exit 2
fi
case "$CUDA_VISIBLE_DEVICES" in
  *,*) echo "refusing: this is a world-1 probe (NGPU=1) but CUDA_VISIBLE_DEVICES exposes more than one device ($CUDA_VISIBLE_DEVICES)." >&2; exit 2 ;;
esac
export NGPU=1
export PORT=29510
exec python3 scripts/harness.py launch mem_probe_base --training \
  --hypothesis "readout-4 probe cell base on card 0: 300 steps, world 1, M1's launch line plus no change (the world-1 control for the other five). Pass = pool_touched_frac at step 300 above 0.50 and not falling from step 200 to 300, read against mem_probe_base on card 0 (world 1) and NOT against M1's world-2 log." \
  -- ./run_ddp.sh --mix data/mix_200m_8b.json --name mem_probe_base \
     --dim 1024 --layers 12 --heads 8 --ffn_hidden 3072 --batch 16 --accum 2 \
     --no-grad_ckpt --lr_scale 1.0 --warmdown 0.1 --anneal_frac 0 --warmup 300 \
     --save_every 100000 --max_steps 300 --seed 42 \
     --mem_values 1048576 --mem_top_k 32 --mem_layers 6 --no-mem_sparse \
     --mem_arm probe_base \
     

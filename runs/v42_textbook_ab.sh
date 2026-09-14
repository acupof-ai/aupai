#!/usr/bin/env bash
# v42 textbook-continuation stage-2 A/B (de; prereg runs/prereg.jsonl#textbook_continuation_ab_0914).
#
# COMMITTED DRAFT -- DOES NOT LAUNCH. Both arms resume the r3 FINAL checkpoint after step
# 38070 and run a 137-step single-phase segment (world 8 x batch 4 x accum 6 = 786,432
# tokens/step) to total step 38207. Do not run until:
#   1. r3 has reached step 38070 and its FINAL checkpoint exists (the step23000 cursor used to
#      build the dry-run mixes is structurally identical but the launch mixes MUST be regenerated
#      against the final ckpt: scripts/write_mix_v42_stage2.py --ckpt <final r3 ckpt>),
#   2. the CPU dry-run is green on those final mixes and fb has approved the merge,
#   3. the controller grants the 8 cards.
#
# LR SHAPE (proved on a CPU replica, prereg amendment_3). Warmup is ABSOLUTE steps and dead
# post-resume, so --warmup 0. The segment is one cosine warmdown tail: warmdown_start must be
# exactly the join step 38070, so --warmdown = 137/38207 = 0.003586. --lr_scale 0.20 makes the
# join peak 0.20 of the r3 base LR and the end floor 0.05*0.20 = 0.01; --anneal_frac 0 (the
# segment has no main/anneal split -- the mix weights are the single phase and anneal==weight).
#
# The two arms differ ONLY in --mix/--name. Architecture, csa/csa2 flash, rope, MoE and batch
# geometry are byte-for-byte the r3 run's, so the comparison isolates the textbook share.
# Both bind PLAIN *_dc caches (the mix carries no cache_exclude).
#
# Usage (only on the go, detached per AGENTS.md pod rules):
#   setsid nohup bash -c 'cd /work/aupai && <one of the lines below> > runs/<name>.log 2>&1' \
#       </dev/null >/dev/null 2>&1 &
set -euo pipefail

CKPT="${CKPT:-ckpt_v41_r3_0914.pt}"   # the FINAL r3 checkpoint; override with the real path at go

ARCH=(--dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 --batch 4 --accum 6
      --attn_every 1 --csa --csa2 --csa2_win_flash --rope_dims 64 --n_swa_only_layers 2
      --no-attn_res --moe_experts 48 --moe_top_k 3 --moe_shared 1 --moe_expert_ffn 1728
      --moe_layers 0-11 --moe_arm v41r3 --no-grad_ckpt)
SEG=(--lr_scale 0.20 --warmup 0 --warmdown 0.003586 --anneal_frac 0 --save_every 100)

# T: textbook at 0.2999 (exactly 4 trainable-pool epochs = 7,888 rows) + six domains on 0.7001.
LINE_T="./run_ddp.sh --mix data/mix_textbook_cont.json --name v42_textbook_t ${ARCH[*]} ${SEG[*]} --resume ${CKPT}"

# C: the same six domains at their r3 anneal proportions over the full segment, no textbook.
LINE_C="./run_ddp.sh --mix data/mix_cont_ctrl.json --name v42_textbook_c ${ARCH[*]} ${SEG[*]} --resume ${CKPT}"

echo "TREATMENT (T):"
echo "  ${LINE_T}"
echo
echo "CONTROL (C):"
echo "  ${LINE_C}"
echo
echo "This is a draft; it prints the two launch lines and exits without launching."

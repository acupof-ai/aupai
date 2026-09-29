#!/bin/bash
# v42 architecture probe, one arm (1e, 2026-09-29). The control is the existing v41_ced_0926 run:
# this is its launch line (runs/ced_w8_0926_launch.sh) with the same mix, seed, 30B schedule,
# warmup 500, B4/accum6 world 8, stochastic rounding and router lr 0.001 -- only the architecture
# changes, and it stops at step 2000. Read against runs/v41_ced_0926.log at the same steps:
#   val 2.491 / 2.135 / 2.027 / 1.954 at 500 / 1000 / 1500 / 2000; HumanEval greedy 8/164 at 2000.
# Candidate: 24 layers (CED 12/12), first two pure SWA, encoder CSA2 F,R,R,R,R,R,F,R,R,R (Full
# every 6), decoders Full, 64 experts top-8 x 640 + 1 shared, sqrtsoftplus router x1.5, untied
# head. 3.26B total / 620.7M active (control 3.22B / 355.4M). Seven changes at once: this answers
# "adopt the package", not which change carries the delta.
cd /work/aupai || exit 1
NAME="v42_arch_b_$(date +%m%d)"
# MB is the per-rank micro-batch; accum follows so a step stays 786,432 tokens like the control.
# 24 layers carry about twice the control's activations; if B4 does not fit, run MB=2.
MB=${MB:-4}
case "$MB" in 1|2|3|4|6) ;; *) echo "MB must divide 24" >&2; exit 2 ;; esac
ACC=$((24 / MB))
export NGPU=8
exec python3 scripts/harness.py launch "$NAME" \
  --training --class incremental --gate-timeout 3000 \
  --hypothesis "V4.1-aligned 24-layer CED (620.7M active) on the v41_ced_0926 recipe reaches lower val than v41_ced_0926 at steps 500/1000/1500/2000 (2.491/2.135/2.027/1.954); step time read beside it" \
  -- ./run_ddp.sh --mix data/mix_v41_gate.json --name "$NAME" --max_steps 2000 \
  --dim 1024 --heads 8 --batch "$MB" --accum "$ACC" \
  --lr_scale 1.0 --warmdown 0.65 --anneal_frac 0.10 --warmup 500 --save_every 2000 --no-grad_ckpt \
  --attn_every 1 --csa --csa2 --csa2_win_flash --rope_dims 64 --no-attn_res --ced \
  --moe_shared 1 --moe_arm v42b --stochastic_round --moe_router_lr 0.001 \
  --layers 24 --ffn_hidden 5760 --ced_enc_layers 12 --attn_hybrid --n_swa_only_layers 2 \
  --csa2_modes F,R,R,R,R,R,F,R,R,R,F,F,F,F,F,F,F,F,F,F,F,F \
  --moe_experts 64 --moe_top_k 8 --moe_expert_ffn 640 --moe_layers 0-23 \
  --router_score sqrtsoftplus --moe_routed_scale 1.5 --untie_head

#!/bin/bash
# Pre-flight for the v41 stage-2 full launch (1e order 2026-09-28): exercise the PERIODIC
# save path with the REAL full mix, which the 50-step smoke could not (its mix kept cot_dc,
# the full mix retires it). Runs ~6 optimizer steps from ckpt_v41_ced_0926.pt and writes one
# .step checkpoint at step 38150 (save_every 10). The cursor sum must equal 38150 x 192.
# Mix/LR/model flags are byte-identical to runs/ced_stage2_launch.sh full mode.
cd /work/aupai || exit 1
export NGPU=8

exec python3 scripts/harness.py launch v41_ced_stage2_pre_0928 \
  --training --class infra-verification --gate-timeout 3000 \
  --hypothesis "stage-2 full-mix periodic-save preflight: retired cot_dc cursor keeps the absolute sum exact at the first save" \
  -- ./run_ddp.sh --mix data/mix_v41_stage2.json --name v41_ced_stage2_pre_0928 \
  --resume ckpt_v41_ced_0926.pt \
  --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 --batch 4 --accum 6 \
  --lr_scale 1.0 --lr_origin_step 38146 --lr_peak_mult 0.30 \
  --warmdown 1.0 --anneal_frac 0.0 --warmup 500 \
  --save_every 10 --val_every 0 --max_steps 38152 --no-grad_ckpt \
  --stochastic_round \
  --attn_every 1 --csa --csa2 --csa2_win_flash --rope_dims 64 --no-attn_res \
  --ced --ced_enc_layers 6 \
  --moe_experts 48 --moe_top_k 3 --moe_shared 1 --moe_expert_ffn 1728 --moe_layers 0-11 \
  --moe_router_lr 0.001 --router_score sigmoid --moe_arm v41ced

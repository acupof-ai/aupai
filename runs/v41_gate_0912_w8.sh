#!/bin/bash
# DRAFT w8 from runs/v41_gate_0911.sh (NOT LAUNCHED): NGPU 8, --accum 6, name v41_gate_0912_w8.
cd /work/aupai || exit 1
export NGPU=8
exec python3 scripts/harness.py launch v41_gate_0912_w8 --training --class incremental --hypothesis "V4.1 flat CSA2 MoE world 8 on the UltraData gate mix clears HumanEval pass@1 >= 30% (prereg v41_gate_0911, w8 follow-on)" \
  -- ./run_ddp.sh --mix data/mix_v41_gate.json --name v41_gate_0912_w8 \
  --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 --batch 4 --accum 6 \
  --lr_scale 1.0 --warmdown 0.65 --anneal_frac 0.10 --warmup 500 --save_every 2000 --no-grad_ckpt \
  --attn_every 1 --csa --csa2 --csa2_win_flash --rope_dims 64 --n_swa_only_layers 2 --no-attn_res \
  --moe_experts 48 --moe_top_k 3 --moe_shared 1 --moe_expert_ffn 1728 --moe_layers 0-11 --moe_arm v41gate

#!/bin/bash
# DRAFT round 3 (USER DECISION 2026-09-13 03:4xZ): phi-1 route -- small high-quality set x3
# epochs, 30B token budget. World 8, B4/accum6 = 786,432 tokens/step, same V4.1 flat stack as
# v41_gate_0911 (csa2_win_flash). Mix data/mix_v41_r3.json is drafted by ae and MUST be on main
# with measured caches before launch. NO LAUNCH from this file: needs 8 cards granted from tileRL
# and the controller's explicit go.
cd /work/aupai || exit 1
export NGPU=8
exec python3 scripts/harness.py launch v41_r3_0914 --training --class incremental --hypothesis "V4.1 flat CSA2 MoE round 3, phi-style small high-quality mix x3 epochs to 30B (prereg v41_r3_0914)" \
  -- ./run_ddp.sh --mix data/mix_v41_r3.json --name v41_r3_0914 \
  --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 --batch 4 --accum 6 \
  --lr_scale 1.0 --warmdown 0.65 --anneal_frac 0.10 --warmup 500 --save_every 1000 --no-grad_ckpt \
  --attn_every 1 --csa --csa2 --csa2_win_flash --rope_dims 64 --n_swa_only_layers 2 --no-attn_res \
  --moe_experts 48 --moe_top_k 3 --moe_shared 1 --moe_expert_ffn 1728 --moe_layers 0-11 --moe_arm v41r3

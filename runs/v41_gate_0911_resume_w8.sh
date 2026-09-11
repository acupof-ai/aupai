#!/bin/bash
# Continue v41_gate_0911 at world 8 from .step6000 (user ruling 2026-09-11): NGPU8 accum6 resume, same run name.
cd /work/aupai || exit 1
export NGPU=8
exec python3 scripts/harness.py launch v41_gate_0911 --training --class incremental --gate-timeout 300 --hypothesis "V4.1 flat CSA2 MoE continued world 8 from step6000 on the UltraData gate mix (prereg v41_gate_0911, w8 continuation amendment 9)" \
  -- ./run_ddp.sh --resume ckpt_v41_gate_0911.pt.step6000 --mix data/mix_v41_gate.json --name v41_gate_0911 \
  --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 --batch 4 --accum 6 \
  --lr_scale 1.0 --warmdown 0.65 --anneal_frac 0.10 --warmup 500 --save_every 2000 --no-grad_ckpt \
  --attn_every 1 --csa --csa2 --csa2_win_flash --rope_dims 64 --n_swa_only_layers 2 --no-attn_res \
  --moe_experts 48 --moe_top_k 3 --moe_shared 1 --moe_expert_ffn 1728 --moe_layers 0-11 --moe_arm v41gate

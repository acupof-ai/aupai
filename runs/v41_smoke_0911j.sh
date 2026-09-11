#!/bin/bash
# v41_smoke_0911j: de-109 csa2_win_flash validation vs i.
#   Exact i line (mix_smoke_warmup, new-vocab f1f8 caches present) PLUS --csa2_win_flash.
#   Four cards 0-3 via --cards (the grant block is 0-5; the harness would otherwise take all
#   six). tileRL holds 4/6/7 on 2026-09-11 -- card 4 is excluded on purpose, do not add it.
#   B4/accum4 (i's per-rank shape), 150 steps, harness launch --training.
# Acceptance vs i at the same steps (fb 2026-09-11):
#   loss within 0.05 at steps 50/100/150; no NaN; tok/s/gpu >= 1.4x i's 18K; peak < 72 GiB.
cd /work/aupai || exit 1
exec python3 scripts/harness.py launch v41_smoke_0911j --training --cards 0,1,2,3 --class infra-verification \
  -- ./run_ddp.sh --mix data/mix_smoke_warmup.json --name v41_smoke_0911j \
  --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 --batch 4 --accum 4 \
  --lr_scale 1.0 --warmdown 0.65 --anneal_frac 0.0 --warmup 20 --save_every 100000 --no-grad_ckpt \
  --attn_every 1 --csa --csa2 --csa2_win_flash --rope_dims 64 --n_swa_only_layers 2 --no-attn_res \
  --moe_experts 48 --moe_top_k 3 --moe_shared 1 --moe_expert_ffn 1728 --moe_layers 0-11 --moe_arm v41smoke

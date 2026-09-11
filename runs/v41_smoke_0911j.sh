#!/bin/bash
# smoke j: i's line + --csa2_win_flash on 0-3; acceptance in prereg v41_gate_0911 amendment 5
cd /work/aupai || exit 1
export NGPU=4  # run_ddp.sh sizes torchrun from NGPU; --cards names the 4 visible devices
exec python3 scripts/harness.py launch v41_smoke_0911j --training --cards 0,1,2,3 --class infra-verification \
  -- ./run_ddp.sh --mix data/mix_smoke_warmup.json --name v41_smoke_0911j \
  --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 --batch 4 --accum 4 \
  --lr_scale 1.0 --warmdown 0.65 --anneal_frac 0.0 --warmup 20 --save_every 100000 --no-grad_ckpt \
  --attn_every 1 --csa --csa2 --csa2_win_flash --rope_dims 64 --n_swa_only_layers 2 --no-attn_res \
  --moe_experts 48 --moe_top_k 3 --moe_shared 1 --moe_expert_ffn 1728 --moe_layers 0-11 --moe_arm v41smoke
# Stop is by exact PID, never --max_steps: max_steps shortens total_steps, moving the warmdown
# start from step 134 (i, total 381) to 53 (total 150), so the step-150 losses would sit on
# different LR schedules and could not be compared. After step 150 is logged: read the torchrun
# PID from the claim, kill it by exact PID, verify cards 0-3 free in nvidia-smi, close the exp
# row keyed with --started carrying loss@50/100/150, NaN, tok/s/gpu, peak GiB.

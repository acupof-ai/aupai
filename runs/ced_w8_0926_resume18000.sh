#!/bin/bash
# Resume of runs/ced_w8_0926_launch.sh from step18000 after the step19220 watchdog stop.
# One delta against that launcher: --resume. The watchdog restarts separately (runs/wd_0926.sh, --from_step 18000).
cd /work/aupai || exit 1
export NGPU=8
exec python3 scripts/harness.py launch v41_ced_0926 \
  --training --class incremental --gate-timeout 3000 \
  --hypothesis "resume v41_ced_0926 from step18000, recipe unchanged, after the divergence watchdog stopped it at step19220 (gnorm 55/31/49408 pre-clip, loss 0.66/1.70/5.70 at 19200/19210/19220). Same data order: a recurrence at ~19200 says the batches trigger it; no recurrence says it was stochastic. v41_ced_0923 diverged twice and resumed the same way" \
  -- ./run_ddp.sh --mix data/mix_v41_gate.json --name v41_ced_0926 \
  --resume ckpt_v41_ced_0926.pt.step18000 \
  --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 --batch 4 --accum 6 \
  --lr_scale 1.0 --warmdown 0.65 --anneal_frac 0.10 --warmup 500 --save_every 2000 --no-grad_ckpt \
  --attn_every 1 --csa --csa2 --csa2_win_flash --rope_dims 64 --n_swa_only_layers 2 --no-attn_res \
  --ced --ced_enc_layers 6 \
  --moe_experts 48 --moe_top_k 3 --moe_shared 1 --moe_expert_ffn 1728 --moe_layers 0-11 --moe_arm v41ced \
  --stochastic_round --router_score sigmoid --moe_router_lr 0.001

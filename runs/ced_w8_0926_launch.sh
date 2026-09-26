#!/bin/bash
# CED world-8 formal 30B launch -- the HumanEval gate line, architecture = CED (bottom 6
# encoder / top 6 decoder, per-layer W_KV/W_Z global KV projected from H_6).
#
# Derivation: runs/ced_w8_smoke.sh (S3), which proved 8-rank NCCL / MoE all-to-all / CSA2
# start and step together. The run_ddp.sh line below is S3's byte-for-byte except --name and
# --save_every, --moe_arm v41ced included. Four deltas, all above the run_ddp.sh line:
#   name          v41_ced_w8smoke_0923 -> v41_ced_0926
#   --class       infra-verification   -> incremental  (the class runs/v41_gate_0922.sh uses)
#   steps         --max_steps 60 --val_every 30 REMOVED. Both were the smoke's; with neither
#                 passed the gate line's own values apply: val_every falls through to
#                 Cfg.val_every = 500 (train.py:446) and the run goes the full budget.
#   --save_every  100000 (the smoke's "never") -> 2000 (the gate line's cadence)
#
# SCHEDULE, from the gate line rather than re-derived here: data/mix_v41_gate.json schedules
# 30.00B tokens = 7,324,212 rows at seq 4096; per rank at world 8 that is 915,526 rows, so
# total_steps = 915,526 // (batch 4 * accum 6) = 38,146 -- the same 38,146 the world-6/accum-8
# 0911 line computed, which is why accum went 8 -> 6 when the world went to 8.
#
# --gate-timeout 3000 is the smoke's, kept for the same reason: the first launch pays compile
# and cache warming before step 0.
#
# NOT STARTED. Placed on the pod for review; the go is the controller's.
cd /work/aupai || exit 1
export NGPU=8
exec python3 scripts/harness.py launch v41_ced_0926 \
  --training --class incremental --gate-timeout 3000 \
  --hypothesis "v41_ced_0923 recipe (same mix, seed, 30B, B4/accum6 world 8) plus three fixes -- stochastic bf16 rounding, per-expert sigmoid router, router AdamW lr 0.01 -> 0.001 -- trains 38146 steps without router collapse and the base clears HumanEval pass@1 >= 30% after SFT; short runs r2/r3 0926 read val 2.211/2.137 vs 0923 2.317 at step 1000" \
  -- ./run_ddp.sh --mix data/mix_v41_gate.json --name v41_ced_0926 \
  --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 --batch 4 --accum 6 \
  --lr_scale 1.0 --warmdown 0.65 --anneal_frac 0.10 --warmup 500 --save_every 2000 --no-grad_ckpt \
  --attn_every 1 --csa --csa2 --csa2_win_flash --rope_dims 64 --n_swa_only_layers 2 --no-attn_res \
  --ced --ced_enc_layers 6 \
  --moe_experts 48 --moe_top_k 3 --moe_shared 1 --moe_expert_ffn 1728 --moe_layers 0-11 --moe_arm v41ced \
  --stochastic_round --router_score sigmoid --moe_router_lr 0.001

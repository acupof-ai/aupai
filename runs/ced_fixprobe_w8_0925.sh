#!/bin/bash
# CED two-fix 3-hour probe, world 8 (user order relayed by 1e 2026-09-25).
#
# Same recipe and the SAME seed as runs/ced_w8_launch.sh (v41_ced_0923); a fresh start, not a
# resume -- the probe measures the two fixes from init over 3000 steps. Exactly two flags are
# added to the 0923 line:
#   --stochastic_round   option B: fp32 candidate Bernoulli-rounded into bf16 weights, fp32 Muon
#                        momentum (1e option B, PR de-sft-stochastic-round)
#   --router_score sigmoid  per-expert sigmoid gate, normalized only within the selected top-k,
#                        bias selection-only (DeepSeek-V3 2412.19437 §2.1.2; PR ci-moe-router-0925)
# Both PRs must be merged and pushed before this script can launch (--router_score is unknown to
# train.py until ci's PR lands). save_every 500; --max_steps 3000.
#
# NOT STARTED. The go is the controller's.
cd /work/aupai || exit 1
export NGPU=8
exec python3 scripts/harness.py launch v41_ced_fixprobe_0925 \
  --training --class incremental --gate-timeout 3000 \
  --hypothesis "CED with the two post-0923 fixes (stochastic bf16 rounding + per-expert sigmoid router) trains 3000 steps without router top-1 collapse and with nonzero sub-ULP weight movement; same mix/recipe/seed as v41_ced_0923, two flags the only delta" \
  -- ./run_ddp.sh --mix data/mix_v41_gate.json --name v41_ced_fixprobe_0925 \
  --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 --batch 4 --accum 6 \
  --lr_scale 1.0 --warmdown 0.65 --anneal_frac 0.10 --warmup 500 --save_every 500 \
  --max_steps 3000 --no-grad_ckpt \
  --attn_every 1 --csa --csa2 --csa2_win_flash --rope_dims 64 --n_swa_only_layers 2 --no-attn_res \
  --ced --ced_enc_layers 6 \
  --moe_experts 48 --moe_top_k 3 --moe_shared 1 --moe_expert_ffn 1728 --moe_layers 0-11 \
  --moe_arm v41ced \
  --stochastic_round --router_score sigmoid

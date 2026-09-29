#!/bin/bash
# v42 architecture probe, one arm (1e, 2026-09-29). The control is v41_ced_0926
# (runs/ced_w8_0926_launch.sh): same mix, seed, 30B schedule, warmup 500, warmdown 0.65,
# B4/accum6 world 8, stopped at step 2000 with --stop_at_step, which keeps the 30B LR schedule, so
# steps 500-2000 run at the control's lr multipliers. Read against runs/v41_ced_0926.log.20260927T015240Z (pod; the live 0926 log starts at 18500):
#   val 2.491 / 2.135 / 2.027 / 1.954 at 500 / 1000 / 1500 / 2000; HumanEval greedy 8/164 at 2000.
# Arm: --arch v42, the v41f V4.1 stack at preset v41f.config.v42_s24 -- 24 layers CED 12/12,
# CSA2 m=2 encoder / m=1 decoder with V4.1's Full/Reuse/Reindex pattern, MQA 16x256, mHC x4,
# 64 experts top-8 x 640 + 1 shared, sqrtsoftplus x1.5, SwiGLU clamp 10, untied fp32 head, no
# softcap -- trained by the V4.1 optimizer (Muon RMS 0.18 head-wise Q, Sinkhorn embed/head, AdamW
# 0.9/0.95/1e-20) at one base lr --v42_lr 1e-3. 3,256.6M total / 614.1M active (control 3.22B /
# 355.4M). Everything changes at once: this answers "adopt the package", not which part carries
# the delta. Also different from the control: no --stochastic_round (the V4.1 optimizer has no
# stochastic-rounding path; train.py refuses the pair).
# --dim/--layers/--heads/--ffn_hidden are required recipe flags and inert under --arch v42.
cd /work/aupai || exit 1
NAME="v42_arch_b_$(date +%m%d)"
# MB is the per-rank micro-batch; accum follows so a step stays 786,432 tokens like the control.
# Peak memory at MB=4 is unmeasured; if B4 does not fit, run MB=2.
MB=${MB:-4}
case "$MB" in 1|2|3|4|6) ;; *) echo "MB must divide 24" >&2; exit 2 ;; esac
ACC=$((24 / MB))
export NGPU=8
exec python3 scripts/harness.py launch "$NAME" \
  --training --class incremental --gate-timeout 3000 \
  --hypothesis "V4.1-aligned v42 stack (--arch v42, 614.1M active) on the v41_ced_0926 recipe reaches lower val than v41_ced_0926 at steps 500/1000/1500/2000 (2.491/2.135/2.027/1.954); step time read beside it" \
  -- ./run_ddp.sh --mix data/mix_v41_gate.json --name "$NAME" --stop_at_step 2000 \
  --arch v42 --v42_lr 1e-3 --moe_arm v42b \
  --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 --batch "$MB" --accum "$ACC" \
  --lr_scale 1.0 --warmdown 0.65 --anneal_frac 0.10 --warmup 500 --save_every 2000 --no-grad_ckpt \
  ${EXTRA_ARGS:-}

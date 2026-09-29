#!/bin/bash
# V4.1 CED stage-2: ~10B-token continuation from the 30B final checkpoint, large-batch
# train.py path WITH optimizer state (the sft_math small-batch/fresh-opt attempt lost 1.5
# HumanEval pt, 2026-09-28).
#
# MODE=smoke (default for the pre-flight) schedules 50 steps over the six domains already
# on the pod; MODE=full schedules 12,715 steps (~10B) once zh_web_dc and math_cot2_dc
# exist. Both resume the SAME checkpoint and carry the SAME LR design; smoke differs only
# in mix file, name, val cadence and save cadence.
#
# LR (train.py --lr_origin_step, mix-0928): the schedule rebases at the 38,146-step join.
# Absolute 500-step linear re-warmup to 0.30 of each optimizer group's peak, then cosine
# to EXACTLY ZERO over the rest -- --warmdown 1.0 fills the segment, --anneal_frac 0 makes
# it one phase, and the stage-2 branch bypasses final_lr_frac=0.05, the nonzero floor the
# 30B run ended shaking at. lr_scale stays 1.0; optimizer state loads from the checkpoint.
#
# --stochastic_round is MANDATORY: the 30B checkpoint was trained with option B
# (StochasticAdamW, state keys m/v; cfg.stochastic_round=True). Without the flag the
# resume builds stock AdamW (state keys exp_avg), load_state_dict accepts both shapes in
# silence and the first opt.step() dies with KeyError 'exp_avg' -- measured on the first
# smoke attempt 2026-09-28.
#
# Mixes are generated on the pod against the checkpoint cursor:
#   python3 scripts/write_mix_stage2_v41.py --ckpt ckpt_v41_ced_0926.pt --mode smoke|full
cd /work/aupai || exit 1
export NGPU=8

MODE="${MODE:-smoke}"
CKPT="${CKPT:-ckpt_v41_ced_0926.pt}"  # override to resume a stage-2 save (the 0929 SR-fix repair)
JOIN=38146

if [ "$MODE" = "full" ]; then
  NAME="${NAME:-v41_ced_stage2_0928}"
  MIX=data/mix_v41_stage2.json
  SAVE=2000
  VAL=500
  WARMUP="${WARMUP:-500}"  # absolute re-warmup, matches the 30B run's own warmup (1e ruling)
  WATCH=1
  HYPO="${HYPO:-stage-2 10B continuation from the 30B final: 30%-peak re-warmup then cosine to zero, code-heavy re-mix; HumanEval beats the 30B endpoint}"
else
  NAME=v41_ced_stage2_smoke_0928
  MIX=data/mix_v41_stage2_smoke.json
  SAVE=100000
  VAL=25
  WARMUP="${WARMUP:-20}"  # 50 steps cannot show a 500-step ramp; smoke shrinks it to exercise the peak+decay
  WATCH=0
  HYPO="stage-2 join mechanics: opt+cursor load, loss continues at the 30B endpoint (~1.4-1.8), stage-2 LR ramp, 0 NaN"
fi

# Divergence watchdog (full only; same rules as v41_ced_0923: gnorm>10 on 3 consecutive
# progress lines, 50-step mean train loss >3.0, any card nvidia-smi RSS >80 GiB/60s). It
# waits for the log to appear and seeks to EOF, so arming before the launch is race-free;
# it reports, never restarts. Smoke is 50 steps and needs none.
if [ "$WATCH" = "1" ]; then
  setsid nohup python3 scripts/ced_diverge_watch.py --log "runs/$NAME.log" --name "$NAME" \
    > "runs/$NAME.watchdog.log" 2>&1 </dev/null &
  echo "watchdog armed: runs/$NAME.watchdog.log (rules gnorm>10x3 | loss50mean>3.0 | RSS>80GiB/60s)"
fi

exec python3 scripts/harness.py launch "$NAME" \
  --training --class incremental --gate-timeout 3000 \
  --hypothesis "$HYPO" \
  -- ./run_ddp.sh --mix "$MIX" --name "$NAME" --resume "$CKPT" \
  --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 --batch 4 --accum 6 \
  --lr_scale 1.0 --lr_origin_step "$JOIN" --lr_peak_mult 0.30 \
  --warmdown 1.0 --anneal_frac 0.0 --warmup "$WARMUP" \
  --save_every "$SAVE" --val_every "$VAL" --no-grad_ckpt --stochastic_round \
  --attn_every 1 --csa --csa2 --csa2_win_flash --rope_dims 64 --no-attn_res \
  --ced --ced_enc_layers 6 \
  --moe_experts 48 --moe_top_k 3 --moe_shared 1 --moe_expert_ffn 1728 --moe_layers 0-11 \
  --moe_router_lr 0.001 --router_score sigmoid --moe_arm v41ced

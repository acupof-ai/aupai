#!/usr/bin/env bash
# Pretrain -> SFT -> RL -> benchmark, unattended, one stage at a time.
#
#   PRETRAIN=k6_fone bash scripts/run_pipeline.sh              # wait for a running pretrain, then go
#   FROM=sft bash scripts/run_pipeline.sh                      # resume the chain at a later stage
#
# Every stage writes runs/pipeline.log, records itself with scripts/exp.py, and stops the chain on
# failure rather than feeding a broken artefact to the next one. Stage outputs are the checkpoints,
# so a stage that already produced its checkpoint is skipped on a rerun.
set -uo pipefail
cd "$(dirname "$0")/.."

PRETRAIN=${PRETRAIN:-k6_fone}
SFT=${SFT:-sft_k6}
RL=${RL:-rl_k6}
NGPU=${NGPU:-8}
FROM=${FROM:-pretrain}
SFT_PT=${SFT_PT:-data/sft/sft_mix_fone.pt}
# Default SFT arm: verified synthetic plus the real rows that survived hand sampling.
# Belle is excluded -- 38.7% of it is semantically defective.
SFT_SOURCES=${SFT_SOURCES:-data/synthetic/math_short_v8.jsonl,data/sft/real_math_noBelle.jsonl,data/alpaca_gpt4_zh.jsonl}
LOG=runs/pipeline.log
mkdir -p runs data/rl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
die() { say "PIPELINE FAILED at $1"; exit 1; }
declare -A STAGE=([pretrain]=1 [sft]=2 [probe]=3 [rl]=4 [bench]=5)
want() { [ "${STAGE[$1]}" -ge "${STAGE[$FROM]}" ]; }

# ---- 1. pretrain: adopt a run already in flight rather than starting a second one
if want pretrain; then
  say "stage pretrain: waiting for ckpt_$PRETRAIN.pt"
  while pgrep -f "train\.py .*--name $PRETRAIN" > /dev/null; do sleep 120; done
  [ -f "ckpt_$PRETRAIN.pt" ] || die "pretrain (no ckpt_$PRETRAIN.pt)"
  say "stage pretrain: done — $(grep -E '^(ep |step )' "runs/$PRETRAIN.log" | tail -1)"
fi

# ---- 2. SFT
if want sft && [ ! -f "ckpt_$SFT.pt" ]; then
  say "stage sft: packing"
  # --fone is not optional against a FoNE base: it has only ever seen a number as one
  # [NUM] carrying a Fourier value, and sft_math.py refuses a pack without values.
  python3 datagen/prepare_sft_math.py --fone --out "$SFT_PT" --sources "$SFT_SOURCES" >> "$LOG" 2>&1 \
    || die "sft packing"
  say "stage sft: training on ckpt_$PRETRAIN.pt"
  NGPU=$NGPU PORT=29660 bash scripts/run_sft.sh "$SFT" "ckpt_$PRETRAIN.pt" "$SFT_PT" \
    --batch 24 --epochs 2 --lr_scale 0.1 >> "$LOG" 2>&1 || die "sft"
  say "stage sft: $(python3 scripts/exp.py list 2>/dev/null | grep " $SFT " | tail -1)"
fi

# ---- 3. difficulty probe: keep only the problems this model gets right 20-80% of the time
if want probe && [ ! -f data/rl/rl_band.jsonl ]; then
  say "stage probe: 10,382 instances x (1 greedy + 8 sampled), sharded over $NGPU GPUs"
  bash eval/probe_band.sh "ckpt_$SFT.pt" "$NGPU" >> "$LOG" 2>&1 || die "probe"
  say "stage probe: $(grep -m1 '^band:' "$LOG" | tail -1)"
fi

# ---- 4. RL, run to completion
if want rl && [ ! -f "ckpt_$RL.pt" ]; then
  DATA=data/rl/rl_band.jsonl
  [ -s "$DATA" ] || DATA=data/rl/rlvr_clean.jsonl
  say "stage rl: GSPO on $DATA ($(wc -l < "$DATA") problems)"
  python3 scripts/exp.py start --name "$RL" --cmd "torchrun --nproc_per_node=$NGPU algorithms/rlvr.py --resume ckpt_$SFT.pt --data $DATA --steps 500" \
    --notes "GSPO, group 8, T=0.9, KL 0.02, prompts filtered to the 20-80% solve-rate band" \
    --hypothesis "The previous RL run had nothing to amplify: 30-55% of groups were all-right or all-wrong, so half the compute produced no gradient. With prompts restricted to the band this model actually solves 20-80% of the time, does accuracy move at all?" >> "$LOG" 2>&1
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPU - 1))) torchrun --nproc_per_node="$NGPU" \
    --master_port=29662 algorithms/rlvr.py --resume "ckpt_$SFT.pt" --data "$DATA" \
    --steps 500 --temperature 0.9 --top_p 0.95 --max_new 320 --kl_beta 0.02 \
    --out "ckpt_$RL.pt" >> "$LOG" 2>&1
  RC=$?
  LAST=$(grep -E "^step " runs/rlvr.log 2>/dev/null | tail -1)
  python3 scripts/exp.py done --name "$RL" --status $([ $RC -eq 0 ] && echo ok || echo fail) \
    --result "rc=$RC | $LAST" >> "$LOG" 2>&1
  [ $RC -eq 0 ] || die "rl"
  say "stage rl: done — $LAST"
fi

# ---- 5. benchmark every checkpoint the chain produced
if want bench; then
  say "stage bench"
  for CK in "ckpt_$PRETRAIN.pt" "ckpt_$SFT.pt" "ckpt_$RL.pt"; do
    [ -f "$CK" ] || continue
    M5=$(bash eval/eval_math.sh "$CK" "$NGPU" 2>>"$LOG" | tail -1)
    MH=$(bash eval/eval_hard.sh "$CK" "$NGPU" 2>>"$LOG" | tail -1)
    PK=$(python3 eval/math_hard.py --ckpt "$CK" --k 8 --temperature 0.8 2>>"$LOG" | tail -1)
    say "bench $CK | $M5 | $MH | pass@8 $PK"
  done
fi

say "PIPELINE DONE"

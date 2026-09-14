#!/bin/bash
# v42 phi-1-route post-r3 SFT launcher (de prereg companion). Shell only; does not modify
# train.py/sft_math.py. Loads the finished r3 checkpoint and SFTs on the phi code-exercises
# pack in CONTINUATION format, then scores HumanEval through the rstrip continuation gate
# (NOT ChatML).
#
# PRE-LAUNCH CHECKLIST — this job is NOT checkpointable:
#   * 738 steps/epoch x N=6 = 4,428 steps, ~2.4-3.0 h of continuous 8-card hold (exact
#     steps read from the built pack: 23,637 rows -> min 2,954/rank -> //4).
#   * Confirm an unbroken 8-card GPU window for the whole run BEFORE launch.
#   * Launch detached the r3 way: pod-side launch file under `setsid nohup ... </dev/null &`
#     so it survives the session (pod foreground dies with the tn tunnel).
#   * An interrupt restarts the ENTIRE run from the r3 (ET) base at epoch 0. sft_math.py
#     has no resume-from-output path: --resume loads pretrained weights with a FRESH
#     optimizer, the loop starts at step 0 with a fresh per-epoch randperm, and the
#     --save_every mid-run saves cannot be continued.
#
# PACK: data/sft/sft_phi_codeexercises_v42_65m_0914.pt — 23,637 rows, 63.97M supervised
# tokens, vocab f1f860970d15d623 (matches r3). NOT the 14,846-row 40M 0913 pack.
#
# warmup pitfall (do NOT "fix" in train/sft code): sft_math.py has no --warmup CLI, so the
# run inherits the resumed r3 cfg Cfg.warmup = 500 and Cfg.warmdown = 0.65. lr_mult
# (train.py:2962) tests warmup FIRST: while step < 500 the multiplier ramps (step+1)/500
# regardless of warmdown, so the effective cosine start is max(500, wd_start), where
# wd_start = total - 0.65*total.
#   N=1: 738  steps, wd_start 259 < warmup 500 (warmup 68%). Ramps to peak at step 499,
#         then step 500 drops from mult 1.0 to ~0.52 and cosines to final_lr_frac 0.05 --
#         no plateau, unusable.
#   N=4: 2952 steps, wd_start 1034, warmup 17% (clean ramp -> plateau -> cosine)
#   N=6: 4428 steps, wd_start 1550, warmup 11%   <- default
#   N=8: 5904 steps, wd_start 2067, warmup 8.5%
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=v42_phi_sft
# r3-final checkpoint. Training finishes to this base name; override with RESUME= for a
# specific .stepN while r3 is still running.
RESUME=${RESUME:-ckpt_v41_r3_0914.pt}
PACK=data/sft/sft_phi_codeexercises_v42_65m_0914.pt
# Timestamped so an interrupted-then-rerun job can never silently overwrite a completed
# run's checkpoint.
TS=$(date -u +%Y%m%dT%H%M%SZ)
# Devices come from the CALLER via CUDA_VISIBLE_DEVICES; never hard-code physical
# indices here (a script that writes them escapes any lane the caller confined it to).
# Default to the 8-card block only if the caller set nothing.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
# shellcheck disable=SC1091
source eval/_devs.sh 8   # validates >=8 visible devices; populates ${_DEVS[@]}
CARDS=$(IFS=,; echo "${_DEVS[*]}")
NGPU=${#_DEVS[@]}
EPOCHS=${EPOCHS:-6}          # 738 steps/epoch -> 4428 total at N=6, warmup 500 = 11%
BATCH=${BATCH:-4}            # per-rank microbatch; effective = 4 x 8 = 32 rows/step
LR_SCALE=${LR_SCALE:-0.1}
SAVE_EVERY=${SAVE_EVERY:-200}
OUT=${OUT:-ckpt_v42_phisft_n${EPOCHS}_${TS}.pt}
# fp8 is ON by default in sft_math.py (only --no_fp8 disables it); grad_ckpt stays ON
# (FP8 backward NaNs without it).

if [ -z "${HYPOTHESIS:-}" ]; then
  echo "REFUSING: set HYPOTHESIS='<what this phi SFT is meant to show>' before launching."
  exit 2
fi
[ -f "$RESUME" ] || { echo "REFUSING: r3 resume ckpt $RESUME not present (r3 finished?)"; exit 2; }
[ -f "$PACK" ]   || { echo "REFUSING: phi pack $PACK not present"; exit 2; }

echo "== cardless pack gate: vocab_id / holdout / fone =="
CUDA_VISIBLE_DEVICES= python3 sft_math.py \
  --resume "$RESUME" --sft_path "$PACK" --check_pack

echo "== live-claim gate on cards $CARDS =="
python3 - "$CARDS" <<'PY'
import sys
sys.path.insert(0, "scripts")
import card_claim
wanted = {c.strip() for c in sys.argv[1].split(",") if c.strip()}
live, _ = card_claim.claims()
held = set()
for c in live:
    held |= {str(x) for x in c.get("cards", [])} & wanted
if held:
    print(f"REFUSING: cards {sorted(held)} held by a live claim; phi SFT launches only after r3 ends.")
    sys.exit(1)
print(f"cards {sorted(wanted)}: no live claim")
PY

NOTES=$(python3 - "$PACK" <<'PY'
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
n = d["input_ids"].shape[0]
sup = int((d["labels"] != -100).sum())
print(f"{n} rows, {sup/1e6:.2f}M supervised tokens, vocab_id {d['vocab_id']}")
PY
)
python3 scripts/exp.py start --name "$NAME" \
  --cmd "torchrun x8 sft_math.py --resume $RESUME --sft_path $PACK --out $OUT --epochs $EPOCHS --batch $BATCH --lr_scale $LR_SCALE --save_every $SAVE_EVERY (fp8, cards $CARDS); eval/humaneval_gen.py --rstrip_nl" \
  --hypothesis "$HYPOTHESIS" --notes "$NOTES" >/dev/null

python3 scripts/card_claim.py acquire --name "$NAME" --cards "$CARDS" \
  --note "post-r3 phi continuation SFT $PACK" --wait 0 --wait-for-device 300 || {
  echo "REFUSING to launch: card_claim acquire refused on $CARDS"
  python3 scripts/exp.py done --name "$NAME" --status fail --result "card claim refused"
  exit 1
}
trap 'python3 scripts/card_claim.py release --name "$NAME" >/dev/null 2>&1 || true' EXIT

set +e
torchrun --nproc_per_node="$NGPU" \
  --master_port="${PORT:-29540}" \
  sft_math.py --resume "$RESUME" --sft_path "$PACK" --out "$OUT" \
  --epochs "$EPOCHS" --batch "$BATCH" --lr_scale "$LR_SCALE" \
  --save_every "$SAVE_EVERY" &
TORCH_PID=$!
wait "$TORCH_PID"
TRAIN_RC=$?
set -e
if [ $TRAIN_RC -ne 0 ]; then
  python3 scripts/exp.py done --name "$NAME" --status fail --result "sft exited $TRAIN_RC"
  exit $TRAIN_RC
fi

echo "== acceptance read: HumanEval CONTINUATION rstrip gate pass@1 (non-ChatML) =="
HE_LOG="runs/he_rstrip_${NAME}.log"
set +e
# Continuation arm: prompt.rstrip(newline), no ChatML wrapper, gate column. This is the
# format the phi pack actually trains (signature-repeated complete functions in continuation).
python3 eval/humaneval_gen.py \
  --ckpt "$OUT" --rstrip_nl --run "$NAME" --force 2>&1 | tee "$HE_LOG"
EVAL_RC=${PIPESTATUS[0]}
set -e
PREDS=$(grep "preds saved:" "$HE_LOG" | tail -1 | sed 's/^preds saved: //' || true)
RESULT=$(grep "HUMANEVAL pass@1" "$HE_LOG" | tail -1 || true)
if [ $EVAL_RC -ne 0 ]; then
  python3 scripts/exp.py done --name "$NAME" --status fail --result "humaneval rstrip eval failed rc=$EVAL_RC"
  exit 1
fi
python3 scripts/exp.py done --name "$NAME" --status ok \
  --result "HumanEval rstrip continuation by-name (preds $PREDS); see runs/prereg.jsonl#v42_phi_sft criterion"
echo "$NAME done: $RESULT"

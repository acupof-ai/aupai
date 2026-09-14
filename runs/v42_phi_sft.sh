#!/bin/bash
# v42 phi-1-route post-r3 SFT launcher (DRAFT, de prereg companion). Shell only; does not
# modify train.py/sft_math.py. Restarts from the finished r3 checkpoint and SFTs on the
# phi code-exercises pack in CONTINUATION format, then scores HumanEval through the rstrip
# continuation gate (NOT ChatML).
#
# restartable: mid-run saves carry optimizer state (--save_every), so an interrupted run
# resumes on one LR curve; an interrupt only costs the current partial epoch.
#
# warmup pitfall (do NOT "fix" in train/sft code): sft_math.py has no --warmup CLI, so the
# run inherits the resumed r3 cfg Cfg.warmup = 500. total_steps =
# epochs * (per_rank_rows // batch); DDP shards the pack X[rank::world] and trims to the
# min rank multiple (ddp_even_len). Current pack sft_phi_codeexercises_0913.pt = 14,846
# rows: world 8 -> min 1,855 rows/rank; batch 4/rank -> 463 steps/epoch.
#   N=1: 463  steps, warmup 463/463 = 100% (the whole run is ramping -- unusable)
#   N=2: 926  steps, warmup 500 = 54%
#   N=3: 1389 steps, warmup 500 = 36%
#   N=4: 1852 steps, warmup 500 = 27%
#   N=6: 2778 steps, warmup 500 = 18%   <- first N with total comfortably > 500
#   N=8: 3704 steps, warmup 500 = 14%
# With a 40M-token pack, N>=6 is required for total >> 500 (warmup <= ~20%). Default 6.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=v42_phi_sft
# r3-final checkpoint. Training finishes to this base name; override with RESUME= for a
# specific .stepN while r3 is still running.
RESUME=${RESUME:-ckpt_v41_r3_0914.pt}
PACK=data/sft/sft_phi_codeexercises_0913.pt
OUT=ckpt_${NAME}.pt
# Devices come from the CALLER via CUDA_VISIBLE_DEVICES; never hard-code physical
# indices here (a script that writes them escapes any lane the caller confined it to).
# Default to the 8-card block only if the caller set nothing.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
# shellcheck disable=SC1091
source eval/_devs.sh 8   # validates >=8 visible devices; populates ${_DEVS[@]}
CARDS=$(IFS=,; echo "${_DEVS[*]}")
NGPU=${#_DEVS[@]}
EPOCHS=${EPOCHS:-6}          # 463 steps/epoch -> 2778 total at N=6, warmup 500 = 18%
BATCH=${BATCH:-4}            # per-rank microbatch; effective = 4 x 8 = 32 rows/step
LR_SCALE=${LR_SCALE:-0.1}
SAVE_EVERY=${SAVE_EVERY:-200}
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
  --cmd "torchrun x8 sft_math.py --resume $RESUME --sft_path $PACK --epochs $EPOCHS --batch $BATCH --lr_scale $LR_SCALE --save_every $SAVE_EVERY (fp8, cards $CARDS); eval/humaneval_gen.py --rstrip_nl" \
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

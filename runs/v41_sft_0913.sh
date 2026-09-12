#!/bin/bash
# restartable: one-shot post-gate ChatML SFT launcher for prereg v41_sft_0913. DRAFT -- no
# launch until the controller's go and the gate run has ended.
#
# Stage: resume the FINAL gate checkpoint, one epoch+ over the 3b-22 ChatML pack, then read
# HumanEval pass@1 through the ChatML/by-name arm (eval/humaneval_gen.py --chatml).
# One card, default card 0 (override CARD=). The gate run trains world-8 on block 0-7, so
# this script refuses hard while any live claim holds the chosen card -- the gate claim
# included -- and relies on card_claim.py acquire --wait 0 as the second gate.
#
# Recipe (justification in runs/prereg.jsonl#v41_sft_0913):
#   pack data/sft/sft_v41_chatml_post30b_0912.pt = 9,610 rows x 4096, 27.97M supervised
#   tokens (71.0% of packed), vocab_id f1f860970d15d623 matching the gate ckpt.
#   EPOCHS=2, BATCH=8 (single card) -> 2 * (9610 // 8) = 2402 optimizer steps, 55.9M
#   supervised tokens seen. LR_SCALE=0.1 (sft_math.py default; gate peak LR x 0.1).
#   BATCH 8 seq4096 fp8 grad-ckpt single card: the resumed cfg carries --csa2_win_flash,
#   whose smoke peak was 43.5 GiB at B4; B8 is estimated ~60 GiB, under the 80 GiB ceiling;
#   first launch confirms nvidia-smi and the prereg is amended if B8 OOMs (drop to B4).
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=v41_sft_0913
RESUME=${RESUME:-ckpt_v41_gate_0911.pt}
PACK=data/sft/sft_v41_chatml_post30b_0912.pt
OUT=ckpt_${NAME}.pt
# The device is the caller's CUDA_VISIBLE_DEVICES; CARD is only the fallback when none was
# exported. The safe idiom keeps this launcher from writing a physical index past a lane the
# controller confined it to, and lets the go set the free card via CUDA_VISIBLE_DEVICES.
CARD=${CARD:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$CARD}
DEV=$CUDA_VISIBLE_DEVICES
EPOCHS=${EPOCHS:-2}
BATCH=${BATCH:-8}
LR_SCALE=${LR_SCALE:-0.1}

if [ -z "${HYPOTHESIS:-}" ]; then
  echo "REFUSING: set HYPOTHESIS='<what this SFT is meant to show>' before launching."
  exit 2
fi
[ -f "$RESUME" ] || { echo "REFUSING: resume ckpt $RESUME not present (gate run finished?)"; exit 2; }
[ -f "$PACK" ]   || { echo "REFUSING: pack $PACK not present"; exit 2; }

echo "== cardless pack gate: vocab_id / holdout / fone =="
CUDA_VISIBLE_DEVICES= python3 sft_math.py \
  --resume "$RESUME" --sft_path "$PACK" --check_pack

echo "== live-claim gate on device $DEV =="
python3 - "$DEV" <<'PY'
import sys
sys.path.insert(0, "scripts")
import card_claim
card = sys.argv[1]
live, _ = card_claim.claims()
holders = [c for c in live if card in [str(x) for x in c.get("cards", [])]]
if holders:
    for c in holders:
        print(f"REFUSING: card {card} held by live claim {c.get('name')} pid {c.get('pid')} "
              f"({c.get('note', '')})")
    print("The gate run still holds a card; SFT launches only after it ends.")
    sys.exit(1)
print(f"card {card}: no live claim")
PY

NOTES=$(python3 - "$PACK" <<'PY'
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
n = d["input_ids"].shape[0]
sup = int((d["labels"] != -100).sum())
print(f"{n} rows x 4096, {sup/1e6:.2f}M supervised tokens, vocab_id {d['vocab_id']}")
PY
)
python3 scripts/exp.py start --name "$NAME" \
  --cmd "sft_math.py --resume $RESUME --sft_path $PACK --epochs $EPOCHS --batch $BATCH --lr_scale $LR_SCALE (card $CARD); eval/humaneval_gen.py --chatml" \
  --hypothesis "$HYPOTHESIS" --notes "$NOTES" >/dev/null

python3 scripts/card_claim.py acquire --name "$NAME" --cards "$DEV" \
  --note "post-gate ChatML SFT $PACK" --wait 0 || {
  echo "REFUSING to launch: card_claim acquire refused on device $DEV"
  python3 scripts/exp.py done --name "$NAME" --status fail --result "card claim refused"
  exit 1
}
trap 'python3 scripts/card_claim.py release --name "$NAME" >/dev/null 2>&1 || true' EXIT

set +e
torchrun --nproc_per_node=1 \
  --master_port="${PORT:-29530}" \
  sft_math.py --resume "$RESUME" --sft_path "$PACK" --out "$OUT" \
  --epochs "$EPOCHS" --batch "$BATCH" --lr_scale "$LR_SCALE"
TRAIN_RC=$?
set -e
if [ $TRAIN_RC -ne 0 ]; then
  python3 scripts/exp.py done --name "$NAME" --status fail --result "sft exited $TRAIN_RC"
  exit $TRAIN_RC
fi

echo "== acceptance read: HumanEval ChatML by-name pass@1 =="
HE_LOG="runs/he_chatml_${NAME}.log"
set +e
python3 eval/humaneval_gen.py \
  --ckpt "$OUT" --chatml --run "$NAME" --force 2>&1 | tee "$HE_LOG"
EVAL_RC=${PIPESTATUS[0]}
set -e
PREDS=$(grep "preds saved:" "$HE_LOG" | tail -1 | sed 's/^preds saved: //' || true)
RESULT=$(grep "HUMANEVAL pass@1" "$HE_LOG" | tail -1 || true)
if [ $EVAL_RC -ne 0 ]; then
  python3 scripts/exp.py done --name "$NAME" --status fail --result "humaneval chatml eval failed rc=$EVAL_RC"
  exit 1
fi
python3 scripts/exp.py done --name "$NAME" --status ok \
  --result "HumanEval ChatML by-name (preds $PREDS); see runs/prereg.jsonl#v41_sft_0913 criterion"
echo "$NAME done: $RESULT"

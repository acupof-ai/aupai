#!/bin/bash
# restartable:
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=v41_sft_0913
RESUME=${RESUME:-ckpt_v41_gate_0911.pt}
PACK=data/sft/sft_v41_chatml_post30b_0912.pt
OUT=ckpt_${NAME}.pt
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

set +e
torchrun --nproc_per_node=1 \
  --master_port="${PORT:-29530}" \
  sft_math.py --resume "$RESUME" --sft_path "$PACK" --out "$OUT" \
  --epochs "$EPOCHS" --batch "$BATCH" --lr_scale "$LR_SCALE" &
TORCH_PID=$!
python3 scripts/card_claim.py acquire --name "$NAME" --cards "$DEV" \
  --note "post-gate ChatML SFT $PACK" --wait 0 --wait-for-device 300 || {
  echo "REFUSING to launch: card_claim acquire refused on device $DEV"
  kill "$TORCH_PID" 2>/dev/null || true
  python3 scripts/exp.py done --name "$NAME" --status fail --result "card claim refused"
  exit 1
}
trap 'python3 scripts/card_claim.py release --name "$NAME" >/dev/null 2>&1 || true' EXIT
wait "$TORCH_PID"
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

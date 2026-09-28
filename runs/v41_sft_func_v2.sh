#!/bin/bash
# restartable: the SFT loop itself resumes from the latest --save_every checkpoint via sft_math;
# a relaunch re-runs the cardless --check_pack gate and card claim, then continues. The expensive
# reasoning pack (fetch + sandbox verify + 13-gram decontam) is produced by separate scripts, never
# rebuilt here.
set -euo pipefail
cd "$(dirname "$0")/.."

# SFT v2 (1e 2026-09-28): HumanEval-shaped function completion. Sandbox-verified TACO/APPS
# call-based problems (signature + docstring -> body, x3) plus code_if short answers, trained on the
# rstrip-nl prompt boundary HumanEval is scored on. Base: the 3-checkpoint average of the 0926 run
# (47/164 greedy). One epoch: v1 lost ground from epoch 1 to epoch 2 (35 -> 33).
NAME=v41_sft_func_v2
RESUME=${RESUME:-ckpt_v41_ced_0926_avg3.pt}
PACK=${PACK:-data/sft/sft_func_v2/sft_func_v2.pt}
OUT=ckpt_${NAME}.pt
# Take the caller's CUDA_VISIBLE_DEVICES; only default to all eight when unset, the same
# contract run_ddp.sh uses (an unconditional assignment would escape a lane restriction).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
source eval/_devs.sh 8
NPROC=${#_DEVS[@]}
CARDS=$(IFS=,; echo "${_DEVS[*]}")
EPOCHS=${EPOCHS:-1}
BATCH=${BATCH:-4}
LR_SCALE=${LR_SCALE:-0.1}
SAVE_EVERY=${SAVE_EVERY:-1000}

if [ -z "${HYPOTHESIS:-}" ]; then
  echo "REFUSING: set HYPOTHESIS='<what this SFT is meant to show>' before launching."
  exit 2
fi
[ -f "$RESUME" ] || { echo "REFUSING: resume ckpt $RESUME not present (30B run finished?)"; exit 2; }
[ -f "$PACK" ]   || { echo "REFUSING: pack $PACK not present (run scripts/sft_func_build.py pack)"; exit 2; }
[ -f data/sft/sft_func_v2/heldout_func.jsonl ] || {
  echo "REFUSING: heldout_code.jsonl missing -- RL held-out must be carved before SFT"; exit 2; }

echo "== cardless pack gate: vocab_id / holdout / fone =="
CUDA_VISIBLE_DEVICES= python3 sft_math.py \
  --resume "$RESUME" --sft_path "$PACK" --check_pack

echo "== live-claim gate on devices $CARDS =="
python3 - "$CARDS" <<'PY'
import sys
sys.path.insert(0, "scripts")
import card_claim
wanted = {c.strip() for c in sys.argv[1].split(",")}
live, _ = card_claim.claims()
held = []
for c in live:
    if wanted & {str(x) for x in c.get("cards", [])}:
        held.append(f"{c.get('name')} pid {c.get('pid')} ({c.get('note','')})")
if held:
    for h in held:
        print(f"REFUSING: a wanted card is held by live claim {h}")
    print("The pretraining run still holds a card; SFT launches only after it ends.")
    sys.exit(1)
print(f"cards {sorted(wanted)}: no live claim")
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
  --cmd "torchrun n$NPROC sft_math.py --resume $RESUME --sft_path $PACK --stochastic_round --epochs $EPOCHS --batch $BATCH --lr_scale $LR_SCALE --save_every $SAVE_EVERY (cards $CARDS)" \
  --hypothesis "$HYPOTHESIS" --notes "$NOTES" >/dev/null

set +e
torchrun --nproc_per_node="$NPROC" \
  --master_port="${PORT:-29531}" \
  sft_math.py --resume "$RESUME" --sft_path "$PACK" --out "$OUT" \
  --no_fp8 --stochastic_round \
  --epochs "$EPOCHS" --batch "$BATCH" --lr_scale "$LR_SCALE" \
  --save_every "$SAVE_EVERY" &
TORCH_PID=$!
python3 scripts/card_claim.py acquire --name "$NAME" --cards "$CARDS" \
  --note "function-completion SFT v2 $PACK" --wait 0 --wait-for-device 300 || {
  echo "REFUSING to launch: card_claim acquire refused on devices $CARDS"
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
python3 scripts/exp.py done --name "$NAME" --status ok \
  --result "function-completion SFT v2 trained; pack $NOTES"
echo "$NAME done -> $OUT"

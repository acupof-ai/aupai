#!/usr/bin/env bash
# The cross-arm number, in the only order that produces a comparable one.
#
#   bash scripts/heldout_crossarm.sh <ours_ckpt> <control_hf_dir> [outdir]
#
# WHY A SCRIPT AND NOT FOUR COMMANDS. The four steps have an order and a shared argument,
# and getting either wrong produces a NUMBER rather than an error: score each arm on its own
# keep-set and you get two losses over different questions. The refusal that protects
# against that lives in eval_heldout.py, but only if --ids is passed, and only if it is the
# INTERSECTION. This file is what makes "both arms, same population" the default path
# instead of a thing to remember.
#
# SEQ PER ARM IS NOT A DEFAULT. Each arm's keep-set must be computed at the seq that arm was
# TRAINED at -- 4096 for ours, 2048 for the control (was 1024 before fb's 2026-09-03 ruling;
# 1024 dropped 11.91% of held-out questions carrying 50.41% of the supervised bytes). Pass
# OURS_SEQ / CTRL_SEQ if either arm's training seq changes; a wrong seq here silently scores
# a population the model never trained on.
set -euo pipefail
cd "$(dirname "$0")/.."

OURS_CKPT=${1:?need our SFT checkpoint, e.g. ckpt_control_ours.pt}
CTRL_DIR=${2:?need the control arm .hf directory, e.g. runs/control_lr_scan/pythia160m_lr1e-4.hf}
OUT=${3:-runs/heldout_crossarm}
OURS_SEQ=${OURS_SEQ:-4096}
CTRL_SEQ=${CTRL_SEQ:-2048}
TEXT=${TEXT:-data/sft/control_sft_text_heldout.jsonl}
mkdir -p "$OUT"

for p in "$OURS_CKPT" "$CTRL_DIR" "$TEXT"; do
  [ -e "$p" ] || { echo "CANNOT RUN: $p does not exist"; exit 2; }
done

# 1-2. Each arm's keep-set. No model, no card -- so CUDA_VISIBLE_DEVICES is emptied rather
# than left inherited: a CPU step that opens a card is what turned into an ownership
# investigation on 2026-09-03 (1e's standing rule: a pure-CPU script sets it to "").
echo "== keep-sets (no card)"
CUDA_VISIBLE_DEVICES="" python3 scripts/eval_heldout.py --arm ours --seq "$OURS_SEQ" \
  --text "$TEXT" --emit_ids "$OUT/ids_ours.txt"
CUDA_VISIBLE_DEVICES="" python3 scripts/eval_heldout.py --arm control --seq "$CTRL_SEQ" \
  --text "$TEXT" --emit_ids "$OUT/ids_ctrl.txt"

# 3. The intersection. This is the population; both arms are scored on exactly it.
echo
echo "== shared population"
CUDA_VISIBLE_DEVICES="" python3 scripts/eval_heldout.py \
  --intersect "$OUT/ids_ours.txt" "$OUT/ids_ctrl.txt" --emit_ids "$OUT/ids_shared.txt"

# 4. Score both. A card is held here, so it is claimed -- and released even on failure.
CLAIM_NAME=${CLAIM_NAME:-heldout_crossarm}
CLAIM_CARDS=${CUDA_VISIBLE_DEVICES:-}
if [ -n "$CLAIM_CARDS" ]; then
  python3 scripts/card_claim.py acquire --name "$CLAIM_NAME" --cards "$CLAIM_CARDS" \
    --note "cross-arm held-out scoring, minutes" --wait "${CLAIM_WAIT:-0}" || {
    echo "REFUSING to score: could not claim card(s) $CLAIM_CARDS"; exit 1; }
  trap 'python3 scripts/card_claim.py release --name "$CLAIM_NAME" >/dev/null 2>&1 || true' EXIT
else
  echo "NOTE: CUDA_VISIBLE_DEVICES unset -- scoring will use every visible card and claims"
  echo "      nothing. Set it to the card you were allocated."
fi

echo
echo "== ours"
python3 scripts/eval_heldout.py --arm ours --seq "$OURS_SEQ" --text "$TEXT" \
  --ckpt "$OURS_CKPT" --ids "$OUT/ids_shared.txt" --json_out "$OUT/ours.json"
echo
echo "== control"
python3 scripts/eval_heldout.py --arm control --seq "$CTRL_SEQ" --text "$TEXT" \
  --ckpt "$CTRL_DIR" --ids "$OUT/ids_shared.txt" --json_out "$OUT/control.json"

# 5. The comparison, and the check that makes it one. Two arms' numbers are comparable only
# if they were computed over the same ids AND the same byte denominator; both are in the json,
# so this asserts rather than trusts.
echo
python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
a = json.load(open(os.path.join(out, "ours.json")))
b = json.load(open(os.path.join(out, "control.json")))
bad = []
if a["evaluated_ids_sha256"] != b["evaluated_ids_sha256"]:
    bad.append(f"evaluated_ids_sha256 differ: ours {a['evaluated_ids_sha256']} vs "
               f"control {b['evaluated_ids_sha256']} -- the arms scored DIFFERENT questions")
if a["supervised_bytes"] != b["supervised_bytes"]:
    bad.append(f"supervised_bytes differ: {a['supervised_bytes']:,} vs "
               f"{b['supervised_bytes']:,} -- the per-byte losses have different divisors")
if a["examples_scored"] != b["examples_scored"]:
    bad.append(f"examples_scored differ: {a['examples_scored']} vs {b['examples_scored']}")
if bad:
    print("=== NOT COMPARABLE")
    for m in bad:
        print(f"  {m}")
    sys.exit(1)
print("=== cross-arm held-out (same ids, same denominator)")
print(f"  population        {a['examples_scored']:,} examples, "
      f"ids {a['evaluated_ids_sha256']}")
print(f"  supervised bytes  {a['supervised_bytes']:,}  (the shared divisor)")
print()
print(f"{'arm':10s} {'seq':>5s} {'NLL/byte':>10s} {'NLL/token':>10s} {'sup tokens':>12s}")
for name, r in (("ours", a), ("control", b)):
    print(f"{name:10s} {r['seq']:>5d} {r['nll_per_supervised_byte']:>10.6f} "
          f"{r['nll_per_supervised_token']:>10.6f} {r['supervised_tokens']:>12,}")
d = a["nll_per_supervised_byte"] - b["nll_per_supervised_byte"]
who = "ours" if d < 0 else "control"
print()
print(f"  lower (better) per supervised byte: {who}, by {abs(d):.6f} "
      f"({100*abs(d)/max(a['nll_per_supervised_byte'], b['nll_per_supervised_byte']):.2f}%)")
print()
print("  per-token is NOT cross-arm comparable (different tokenizers); it is printed because")
print("  it is the number in each arm's training log.")
print("  This compares data+architecture+recipe TOGETHER. The optimizers differ (Muon vs")
print("  AdamW) and the seqs differ, so no single factor can be credited.")
PY

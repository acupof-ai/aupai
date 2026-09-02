#!/usr/bin/env bash
# The control arm's lr scan. fb's requirement, 2026-09-02: a single lr chosen by a 4-step CPU
# smoke test cannot be the control's final lr, or "our architecture won" is explainable by
# "the control was undertuned". Three points, same pack, same step count, selected on
# HELD-OUT loss -- never training loss, which would pick whichever lr memorised hardest.
#
#   bash scripts/scan_control_lr.sh <pack> [outdir]
#
# One card, three runs serial, ~3h. Every point's held_out_loss lands in its own meta.json
# and the summary below goes in the report's footnote.
set -euo pipefail
cd "$(dirname "$0")/.."

PACK=${1:?need the shared text pack}
OUT=${2:-runs/control_lr_scan}
DEV=${DEVICE:-cuda}
EXTRA=${EXTRA:-}
mkdir -p "$OUT"

# Claim the card for the WHOLE scan, not per point: three serial runs on one card are one
# occupancy, and releasing between them would let another job in mid-scan and make the three
# points' wall times incomparable. Card comes from CUDA_VISIBLE_DEVICES, which is how this arm
# is pointed at a card (fb's allocation: control arm on card 1).
CLAIM_NAME=${CLAIM_NAME:-control_lr_scan}
CLAIM_CARDS=${CUDA_VISIBLE_DEVICES:-}
if [ -n "$CLAIM_CARDS" ] && [ "$DEV" != "cpu" ]; then
  python3 scripts/card_claim.py acquire --name "$CLAIM_NAME" --cards "$CLAIM_CARDS" \
    --note "control lr scan, 3 points serial" --wait "${CLAIM_WAIT:-0}" || {
    echo "REFUSING to launch: could not claim card(s) $CLAIM_CARDS (card_claim.py status)"
    exit 1
  }
  trap 'python3 scripts/card_claim.py release --name "$CLAIM_NAME" >/dev/null 2>&1 || true' EXIT
elif [ "$DEV" != "cpu" ]; then
  echo "NOTE: CUDA_VISIBLE_DEVICES is unset, so this scan claims nothing and will use every"
  echo "      visible card. Set it (fb's allocation puts the control arm on card 1)."
fi

# Three points a factor of ~3 apart, bracketing the 1e-4 the smoke test found. A scan whose
# best point is at an endpoint has not bracketed the optimum -- the summary says so rather
# than reporting the endpoint as if it were a minimum.
LRS="3e-5 1e-4 3e-4"

# One exp row for the SCAN, not per point: the three points are one experiment with one
# question ("which lr does held-out loss pick"), and three rows would read as three findings.
if [ -z "${HYPOTHESIS:-}" ]; then
  echo "REFUSING: set HYPOTHESIS='<what this scan is meant to test>' before launching."
  exit 2
fi
python3 scripts/exp.py start --name "$CLAIM_NAME" \
  --cmd "scan_control_lr.sh $PACK ($LRS)" --hypothesis "$HYPOTHESIS" \
  --notes "3 lr points serial on card(s) ${CLAIM_CARDS:-unset}, selected on held-out loss" \
  >/dev/null || true

for LR in $LRS; do
  CKPT="$OUT/pythia160m_lr$LR"
  if [ -f "$CKPT.meta.json" ]; then
    echo "== lr $LR already done ($CKPT.meta.json), skipping"
    continue
  fi
  echo "== lr $LR -> $CKPT"
  # Serial by design: three concurrent runs on one card would contend and the wall times
  # would not be comparable. Restartable: an interrupted point is redone, finished ones skip.
  python3 scripts/sft_hf_control.py --pack "$PACK" --lr "$LR" --out "$CKPT" \
    --device "$DEV" $EXTRA 2>&1 | tee "$OUT/lr$LR.log"
done

echo
echo "=== control lr scan summary (selected on HELD-OUT loss)"
python3 - "$OUT" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(out, "*.meta.json"))):
    m = json.load(open(p))
    rows.append((m.get("lr"), m.get("held_out_loss"), m.get("held_out_loss_before"),
                 m.get("final_train_loss"), m.get("steps"), os.path.basename(p)))
if not rows:
    sys.exit("no meta.json found -- the scan produced nothing to select from")
rows.sort(key=lambda r: (r[1] is None, r[1]))
print(f"{'lr':>8}  {'held-out':>9}  {'before':>8}  {'train':>8}  {'steps':>6}")
for lr, ho, hb, tr, st, _ in rows:
    f = lambda v: "    n/a" if v is None else f"{v:8.4f}"  # noqa: E731
    print(f"{lr:>8}  {f(ho)}  {f(hb)}  {f(tr)}  {st:>6}")
best = rows[0]
print(f"\nBEST held-out: lr {best[0]}  loss {best[1]:.4f}  ({best[5]})")
if best[3] is not None and rows[-1][3] is not None:
    tr_best = min(rows, key=lambda r: (r[3] is None, r[3]))
    if tr_best[0] != best[0]:
        print(f"NOTE: training loss would have picked lr {tr_best[0]} instead -- which is why "
              f"the selection is on held-out loss.")
lrs = [float(r[0]) for r in rows if r[0] is not None]
if lrs and float(best[0]) in (min(lrs), max(lrs)):
    print(f"WARNING: the best point {best[0]} is at an ENDPOINT of the scanned range "
          f"[{min(lrs):g}, {max(lrs):g}]. The optimum is not bracketed, so this is the best "
          f"of what was tried, not a minimum. Extend the range before quoting it as tuned.")
PY

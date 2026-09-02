#!/bin/bash
# Score e1-27's four new points on the SAME population as everything else in the control report.
# restartable: reads four checkpoints and writes one small JSON each, ~2 min per point on CPU.
# An interrupt costs the unfinished point; finished JSONs are complete and the guard below refuses
# to score while any training process still holds a checkpoint.
#
#
# The whole value of this sweep is that its numbers sit beside the control arm's five points and
# the two floors. That requires one population, one denominator, one evaluator -- the three
# conditions defect A/B/C in docs/audits/control_pythia160m_vs_ours.md section 3 were each a
# violation of. So: eval_heldout.py, --ids runs/heldout_v2/ids_shared.txt, and every output must
# report evaluated_ids_sha256 cae4daf7ad59388c or the number does not get used.
#
# The fifth point is NOT scored here. lr_scale 0.1 is ckpt_control_ours.pt, already scored
# through the current guarded evaluator as runs/heldout_v2/ours_sft_reguarded.json (0.293989,
# next-token beats skip-one 11.19x). Rescoring it would produce a second implementation of one
# number, which is the root cause section 3 names.
set -uo pipefail
cd /work/aupai

IDS=runs/heldout_v2/ids_shared.txt
WANT_SHA=cae4daf7ad59388c
OUT=runs/heldout_v2

[ -f "$IDS" ] || { echo "REFUSING: $IDS absent -- the population must come from the durable copy"; exit 2; }

# Refuse while any sweep arm is still training: a checkpoint mid-write scores as garbage, and
# "the file exists" is not "the file is finished".
#
# Discovered from the process table, NOT from a pid file: my first version read
# /tmp/e1_27_sweep.pids, which the sweep driver never writes, so `cat` yielded nothing and the
# loop refused nothing. A guard whose input is empty passes silently -- the same shape as the
# leak check that returned "clean" on an empty population.
alive=$(ps -eo pid=,stat=,args= 2>/dev/null | grep '[s]ft_math.py' | grep 'e1_27' || true)
if [ -n "$alive" ]; then
  echo "REFUSING: e1-27 training is still running:"
  echo "$alive" | sed 's/^/  /'
  exit 2
fi
# And prove the discovery works at all, rather than trusting an empty result: ps must be able to
# see THIS shell. If it cannot, the emptiness above meant nothing.
ps -o pid= -p $$ >/dev/null 2>&1 || { echo "REFUSING: cannot read the process table, so"; \
  echo "  'no training running' is unverified"; exit 2; }
echo "no e1-27 training processes alive (process table readable)"

for s in 0.01 0.03 0.3 1.0; do
  [ -f "ckpt_e1_27_lr${s}.pt" ] || { echo "REFUSING: ckpt_e1_27_lr${s}.pt absent -- that arm did not finish"; exit 2; }
done

# ONE card, taken out of the caller's lane rather than named here. Scoring runs the four points
# sequentially, so it needs a single device -- but writing a physical index in a child REPLACES the
# caller's CUDA_VISIBLE_DEVICES instead of indexing into it, and that is how a lane-card launch
# landed on a training-block GPU on 2026-08-31 (2f97e4a).
: "${CUDA_VISIBLE_DEVICES:=0}"
source eval/_devs.sh 1 || exit 2
card=${_DEVS[0]}
[ "$(nvidia-smi -i "$card" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] \
  || { echo "REFUSING: card $card is not idle"; exit 2; }

for s in 0.01 0.03 0.3 1.0; do
  json="$OUT/ours_lr${s}.json"
  log="$OUT/ours_lr${s}.log"
  echo "=== scoring lr_scale $s on card $card ==="
  CUDA_VISIBLE_DEVICES=${_DEVS[0]} python3 -u scripts/eval_heldout.py \
    --arm ours --ckpt "ckpt_e1_27_lr${s}.pt" --seq 4096 \
    --ids "$IDS" --json_out "$json" > "$log" 2>&1 &
  p=$!
  python3 scripts/card_claim.py acquire --name "e1_27_score_$s" --cards "$card" --pid "$p" \
    --note "e1-27 scoring lr_scale $s" >/dev/null 2>&1 \
    || { echo "REFUSING: could not claim card $card; killing $p"; kill "$p" 2>/dev/null; exit 2; }
  wait "$p"; rc=$?
  python3 scripts/card_claim.py release --name "e1_27_score_$s" >/dev/null 2>&1
  tail -8 "$log"
  [ "$rc" -eq 0 ] || { echo "SCORING FAILED/REFUSED for lr_scale $s (rc=$rc) -- stop, report, do not adjust"; exit "$rc"; }
done

python3 scripts/e1_27_table.py

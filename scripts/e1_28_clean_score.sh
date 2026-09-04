#!/bin/bash
# e1-28: the three verdict numbers on the CLEAN subset, beside the 10,421 versions.
# restartable: three eval_heldout runs writing one small JSON each, ~2 min per point on one card.
# An interrupt costs the unfinished point; finished JSONs are complete and self-describing.
#
# THIS SCRIPT'S POPULATION IS SUPERSEDED AND IT NOW REFUSES TO RUN. Read this before editing
# the refusal away.
#
# It scores runs/heldout_v2/ids_clean.txt: 10,421 minus the 316 items scripts/e1_28_leak_scan.py
# flagged. That scan is retracted (facts/contamination.json#cont.heldout_in_pretrain_corpus): it
# read 5.0-13.2% of each domain's documents because it counted token-block cursor rows as
# documents, and its coverage guard was in the same wrong unit so it never fired. The
# whole-corpus scan found 2,114 contaminated items, not 316.
#
# So the four numbers already in runs/heldout_v2/clean_*.json -- ours floor 0.457462, control
# floor 0.926041, and the SFT pair -- are measurements of a population that is still 1,818
# contaminated items wide. They are not wrong arithmetic; they answer a question nobody asked.
# docs/audits/control_pythia160m_vs_ours.md 5.3d carries them marked 作废 for that reason.
#
# THE TWO EXCLUSION SETS CROSS, which is why this cannot be fixed by pointing IDS at a bigger
# file and keeping the old numbers as a baseline: 296 of the 316 are among the 2,114, and 20
# are NOT -- items the retracted scan dropped that the full scan says are fine. The new
# population is not the old one shrunk, so 0.457462 is not a point on the same curve.
#
# WHAT TO RUN INSTEAD: scripts/e1_28_clean_ids.py writes runs/heldout_v2/ids_clean_v2_notknown.txt
# (8,307 ids, ids_sha ff496ed9ed26c44d) and reports the 5,409-item verified-clean count beside
# it -- 8,307 includes 2,898 items whose answers are under 13 words, which have no 13-gram and
# were never testable, so that set is "not known to be dirty" rather than clean. Choosing
# between the two populations is a judgement about what the floor is meant to support; it is not
# this script's to make silently. Then re-run the three points with IDS and WANT_SHA set to the
# chosen set, on a card that is granted in runs/card_assignment.json.
set -uo pipefail

cat >&2 <<'REFUSAL'
REFUSING: this script's population is superseded.

  ids_clean.txt      10,105 ids = 10,421 - the 316-item exclusion from e1_28_leak_scan.py
  that scan           RETRACTED: 5.0-13.2% document coverage, coverage guard in the wrong unit
  the real figure     2,114 contaminated of 7,523 measurable (28.10%), whole-corpus scan
  consequence         1,818 contaminated items are still inside ids_clean.txt

  The four numbers in runs/heldout_v2/clean_*.json are kept as a record and marked 作废 in
  docs/audits/control_pythia160m_vs_ours.md 5.3d. Do not cite them as a clean-subset result.

  Next: CUDA_VISIBLE_DEVICES= python3 scripts/e1_28_clean_ids.py --write
        then set IDS/WANT_SHA here to the chosen population and delete this refusal in the
        same commit that records which population was chosen and why.
REFUSAL
exit 2

# ---- unreachable below; kept so the recipe is not lost when the population is settled ----
cd /work/aupai

IDS=runs/heldout_v2/ids_clean.txt
OUT=runs/heldout_v2
WANT_SHA=7231156c5698c210
WANT_N=10105

# THE IDS FILE MUST BE THE ONE THAT WAS AUDITED. A file with this name but different contents would
# score a different population and every number below would be a number about something else --
# and it would look exactly like a result. The digest is computed by IMPORTING eval_heldout.ids_sha,
# because guessing that recipe once already produced e64914b26d8562f3 against the real
# cae4daf7ad59388c.
[ -f "$IDS" ] || { echo "REFUSING: $IDS absent"; exit 2; }
got=$(CUDA_VISIBLE_DEVICES= python3 -c "
import sys; sys.path.insert(0,'scripts')
from eval_heldout import ids_sha
ids=[int(x) for x in open('$IDS') if x.strip()]
print(f'{len(ids)} {ids_sha(ids)}')") || { echo "REFUSING: could not digest $IDS"; exit 2; }
[ "$got" = "$WANT_N $WANT_SHA" ] || {
  echo "REFUSING: $IDS is $got, expected $WANT_N $WANT_SHA"
  echo "  The clean subset is defined by 70718736; a different digest means a different population."
  exit 2; }
echo "population verified: $got"

# ONE card, out of the caller's lane. Writing a physical index in a child REPLACES the caller's
# restriction instead of indexing into it (2f97e4a: a lane-card launch landed on a training GPU).
: "${CUDA_VISIBLE_DEVICES:=1}"
source eval/_devs.sh 1 || exit 2
card=${_DEVS[0]}
used=$(nvidia-smi -i "$card" --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -lt 1000 ] || { echo "REFUSING: card $card holds ${used} MiB"; exit 2; }

# The three points, each named by what it is in section 5.
run() {  # name arm ckpt_or_dir
  local name=$1 arm=$2 ckpt=$3
  local json="$OUT/clean_${name}.json" log="$OUT/clean_${name}.log"
  echo "=== $name ($arm) ==="
  if [ -f "$json" ]; then echo "  exists, skipping"; return 0; fi
  CUDA_VISIBLE_DEVICES=${_DEVS[0]} python3 -u scripts/eval_heldout.py \
    --arm "$arm" ${ckpt:+--ckpt "$ckpt"} --seq 4096 \
    --ids "$IDS" --json_out "$json" > "$log" 2>&1 &
  local p=$!
  python3 scripts/card_claim.py acquire --name "e1_28_clean_$name" --cards "$card" --pid "$p" \
    --note "e1-28: $name on the clean subset $WANT_SHA" >/dev/null 2>&1 \
    || { echo "REFUSING: could not claim card $card; killing $p"; kill "$p" 2>/dev/null; exit 2; }
  wait "$p"; local rc=$?
  python3 scripts/card_claim.py release --name "e1_28_clean_$name" >/dev/null 2>&1
  [ $rc -eq 0 ] || { echo "  FAILED rc=$rc; tail:"; tail -3 "$log"; return 1; }
  # A json that scored a different population is worse than no json.
  CUDA_VISIBLE_DEVICES= python3 -c "
import json,sys
d=json.load(open('$json'))
s=d.get('evaluated_ids_sha256')
if s!='$WANT_SHA': sys.exit(f'WRONG POPULATION in $json: {s} != $WANT_SHA')
print(f\"  nll/byte {d['nll_per_supervised_byte']:.6f}  n={d.get('n_scored')}  bytes={d.get('supervised_bytes')}  sha {s}\")" || return 1
}

rc=0
run floor_ours    ours    ckpt_p200m_4b_0902.pt      || rc=1
# THE CONTROL FLOOR'S MODEL IS NAMED EXPLICITLY, NOT LEFT TO THE DEFAULT. --model_dir defaults to
# data/controls/pythia-160m-step2000, but runs/heldout_v2/floor_control.json records ckpt
# "e1_untrained.hf" -- the published 0.903758 was measured on THAT directory. Taking the default
# here would score a different model and produce a number that is not comparable to the floor it
# is meant to sit beside, while looking exactly like a result.
CTRL_DIR=/tmp/e1_untrained.hf
[ -d "$CTRL_DIR" ] || { echo "REFUSING: $CTRL_DIR absent -- it is the model behind the published"
  echo "  control floor 0.903758 (runs/heldout_v2/floor_control.json ckpt=e1_untrained.hf), it lives"
  echo "  only on the pod's /tmp, and nothing in git can reconstruct it. Re-download before scoring."
  exit 2; }
run floor_control control "$CTRL_DIR"                 || rc=1
run ours_sft      ours    ckpt_control_ours.pt        || rc=1
echo
echo "three verdict numbers on $WANT_N ids ($WANT_SHA); the other six points were NOT recomputed"
exit $rc

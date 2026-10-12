#!/bin/bash
# P0 control for the hand-run launcher claim pattern (handover doc section 5, P0:
# "显卡 claim 失败仍可能继续 -> claim 失败必须停止").
#
# Three worlds, all on ONE card so this never collides with a running job:
#   1. negative: `--pid $$` (this script's own shell) must be REFUSED -- de-34. This is the
#      shape that silently killed the 07:49 and 08:29 engbase launches.
#   2. positive: start a non-shell process, then claim ITS pid -- must be held.
#   3. negative: a second claim of the same card under another name must be REFUSED, which is
#      the condition the launcher turns into "kill the job and exit 4" instead of running
#      unclaimed.
# Usage: bash runs/claim_pattern_selftest.sh [card]   (default card 7)
set -uo pipefail
cd /work/aupai
CARD=${1:-7}
CLAIM="python3 scripts/card_claim.py"
fails=0

echo "== 1. negative control: --pid \$\$ (a shell) must REFUSE"
if $CLAIM acquire --name claimtest_shellpid --cards "$CARD" --wait 0 --pid $$ >/tmp/claimtest_1.txt 2>&1; then
  echo "FAIL: a claim bound to shell pid $$ was accepted"; fails=$((fails + 1))
else
  echo "PASS: refused -- $(tail -1 /tmp/claimtest_1.txt)"
fi

echo "== 2. positive control: claim the job's own pid after starting it"
bash -c 'exec sleep 40' &
JOB=$!
if $CLAIM acquire --name claimtest_jobpid --cards "$CARD" --wait 0 --pid "$JOB" >/tmp/claimtest_2.txt 2>&1; then
  echo "PASS: held -- $(tail -1 /tmp/claimtest_2.txt)"
else
  echo "FAIL: could not claim card $CARD for pid $JOB -- $(tail -1 /tmp/claimtest_2.txt)"
  fails=$((fails + 1))
fi

echo "== 3. negative control: the same card under another name must REFUSE"
if $CLAIM acquire --name claimtest_second --cards "$CARD" --wait 0 --pid "$JOB" >/tmp/claimtest_3.txt 2>&1; then
  echo "FAIL: a second claim on an occupied card was accepted"; fails=$((fails + 1))
else
  echo "PASS: refused -- $(tail -1 /tmp/claimtest_3.txt)"
fi

echo "== status"
$CLAIM status 2>&1 | grep -E "claimtest|ORPHAN.*$CARD" || echo "(no claimtest row visible)"

echo "== cleanup"
$CLAIM release --name claimtest_jobpid >/dev/null 2>&1 || true
$CLAIM release --name claimtest_shellpid >/dev/null 2>&1 || true
$CLAIM release --name claimtest_second >/dev/null 2>&1 || true
kill "$JOB" 2>/dev/null || true
wait "$JOB" 2>/dev/null || true
if [ "$fails" -ne 0 ]; then
  echo "CLAIM_PATTERN_CONTROL_FAILED ($fails)"
  exit 1
fi
echo "CLAIM_PATTERN_CONTROL_PASSED"

#!/usr/bin/env bash
# The 2026-09-04 12:03Z pod checkpoint prune. ONE rm per named file, plan taken fresh at run time.
#
# WHY THE PLAN IS TAKEN HERE AND NOT REUSED. A plan computed minutes ago describes a pod that has
# since moved: two training runs finished at 11:09Z and 11:37Z and their rollers wrote .stepN files
# while the earlier plan was being read. gen_ckpt_listing.py --plan-deletion re-scans the pod every
# time, so the only safe input is the one produced at the moment of deletion (6e's instruction, and
# the reason a stale `stat` nearly cost a 497-row artifact on 2026-09-04).
#
# WHAT IT REFUSES TO DO:
#   - no `rm -f`, no globs, no `xargs`: one `rm` per name that the plan printed, and the name comes
#     from the plan's own `rm <name>` line rather than from a pattern this script composes.
#   - the four checkpoints of the live A/B (ckpt_b0_headmix_arm{A,B}.pt and their .stepN) and the
#     Stage E files are asserted ABSENT from the plan before anything is deleted. They are the
#     subject of a running comparison; the Stage E rows were voided by user order at ~11:05Z and
#     their checkpoints are to be pruned in a LATER pass with the void on the record, not folded
#     into this one silently.
#   - the three pin classes are asserted present in the plan's skip list, not merely absent from
#     the rm list: "not deleted" and "deliberately protected" are different states, and only the
#     second proves the protection ran.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
ROOT=$PWD
LOG=runs/audit_0904/prune_1203Z.log
PLAN=runs/audit_0904/prune_1203Z_plan.txt

# FIRST LINE IS date -u, PASTED, NOT COMPUTED BY HAND. On 2026-09-04 I wrote "12:03Z has passed"
# while `date -u` on screen read 10:14:29Z, having pasted that output sixty seconds earlier. The
# log's authority is this line and nothing else.
: > "$LOG"
{
  date -u
  echo "prune of /work/aupai checkpoints, run by b0, plan taken fresh at this minute"
  echo "operator: scripts/gen_ckpt_listing.py --plan-deletion (deletes nothing itself)"
} | tee -a "$LOG"

echo "=== df BEFORE ===" | tee -a "$LOG"
~/bin/pod 'df -h /work | tail -2' 2>&1 | tee -a "$LOG"

echo "=== fresh plan ===" | tee -a "$LOG"
if ! timeout 600 python3 scripts/gen_ckpt_listing.py --plan-deletion > "$PLAN" 2>>"$LOG"; then
  echo "REFUSING: --plan-deletion exited nonzero; nothing deleted" | tee -a "$LOG"
  exit 2
fi
RM_N=$(grep -c '^rm ' "$PLAN")
echo "plan: $RM_N rm line(s); $(grep -E '^# [0-9]+ file' "$PLAN")" | tee -a "$LOG"

# GUARD 1: the live A/B and the voided Stage E must not be in this plan.
echo "=== guard: live A/B + Stage E absent from the rm list ===" | tee -a "$LOG"
BAD=$(grep '^rm ' "$PLAN" | grep -E 'headmix_arm|se_16lnew|se_looped' || true)
if [ -n "$BAD" ]; then
  echo "REFUSING: the plan names files of the live A/B or of voided Stage E:" | tee -a "$LOG"
  echo "$BAD" | tee -a "$LOG"
  exit 2
fi
echo "ok: no headmix / se_16lnew / se_looped name on any rm line" | tee -a "$LOG"

# GUARD 2: each pin class must appear in the SKIP list, which proves the protection evaluated.
echo "=== guard: pin classes protected (present in the skip list) ===" | tee -a "$LOG"
fail=0
for pin in \
  'ckpt_0830v1_3.24b.pt.ep1' \
  'ckpt_sft_p324_v4.pt' \
  'ckpt_w7_b32a1.pt' ; do
  if grep -qE "^#   ${pin}: " "$PLAN"; then
    echo "ok: $pin -> $(grep -E "^#   ${pin}: " "$PLAN" | head -1 | sed 's/.*: //')" | tee -a "$LOG"
  else
    echo "REFUSING: $pin is not in the plan's skip list -- it is not demonstrably protected" | tee -a "$LOG"
    fail=1
  fi
  # And it must not be on an rm line under its EXACT name (a derived .stepN is a different file).
  if grep -qE "^rm ${pin}\s" "$PLAN"; then
    echo "REFUSING: $pin appears on an rm line" | tee -a "$LOG"
    fail=1
  fi
done
if grep -qE '^rm .*\.milestone_' "$PLAN"; then
  echo "REFUSING: a *.milestone_* hardlink is on an rm line -- deleting it unpins what it protects" | tee -a "$LOG"
  fail=1
fi
[ "$fail" -eq 0 ] || exit 2

# THE NAME LIST, validated HERE before a single byte of it reaches the pod. Every name must be a
# plain ckpt_* basename: no slash (so nothing outside /work/aupai can be named), no glob character,
# no shell metacharacter, no leading dot. A name that fails is reported and dropped, never repaired.
NAMES=runs/audit_0904/prune_1203Z_names.txt
: > "$NAMES"
bad_n=0
CAP=400   # iteration cap: the plan is finite, but every loop here carries one
i=0
while IFS= read -r name; do
  i=$((i+1))
  if [ "$i" -gt "$CAP" ]; then
    echo "REFUSING: plan has more than $CAP names; nothing deleted" | tee -a "$LOG"; exit 2
  fi
  case "$name" in
    ckpt_*)
      case "$name" in
        *[!A-Za-z0-9._-]*)
          echo "REFUSING name with a character outside [A-Za-z0-9._-]: $name" | tee -a "$LOG"
          bad_n=$((bad_n+1)) ;;
        *) printf '%s\n' "$name" >> "$NAMES" ;;
      esac ;;
    *) echo "REFUSING name that is not a ckpt_ basename: $name" | tee -a "$LOG"; bad_n=$((bad_n+1)) ;;
  esac
done < <(grep '^rm ' "$PLAN" | awk '{print $2}')
if [ "$bad_n" -ne 0 ]; then
  echo "REFUSING the run: $bad_n name(s) failed validation; nothing deleted" | tee -a "$LOG"; exit 2
fi
N_NAMES=$(wc -l < "$NAMES" | tr -d ' ')
echo "validated $N_NAMES name(s) for deletion" | tee -a "$LOG"

# THE DELETION, one `rm` per named file, executed INSIDE the container in a single session.
# ONE SESSION RATHER THAN 189: each ~/bin/pod call is tn exec -> crictl exec, ~3 s, so a call per
# file (plus one per stat) is ~20 minutes of tunnel round trips and 378 independent chances for the
# tunnel to drop mid-prune with no record of where it stopped. The discipline that matters is
# preserved exactly: the loop deletes ONE NAMED FILE PER `rm`, with no glob, no `-f`, no `xargs`,
# and no pattern composed by this script -- the names come from the plan and were validated above.
# stat runs immediately before each rm, in the same shell, so the size logged is the size at delete
# time rather than a snapshot from minutes earlier.
echo "=== rm, one file at a time (remote loop) ===" | tee -a "$LOG"
if ! ~/bin/pod "cd /work/aupai && cat > /tmp/prune_names.txt <<'PRUNE_EOF'
$(cat "$NAMES")
PRUNE_EOF
done_n=0; miss_n=0; err_n=0; freed=0
while IFS= read -r n; do
  [ -n \"\$n\" ] || continue
  sz=\$(stat -c %s -- \"/work/aupai/\$n\" 2>/dev/null)
  if [ -z \"\$sz\" ]; then echo \"absent at delete time: \$n\"; miss_n=\$((miss_n+1)); continue; fi
  if rm -- \"/work/aupai/\$n\"; then
    echo \"rm \$n    \$sz B\"; done_n=\$((done_n+1)); freed=\$((freed+sz))
  else
    echo \"FAILED to rm: \$n (\$sz B)\"; err_n=\$((err_n+1))
  fi
done < /tmp/prune_names.txt
echo \"=== totals ===\"
echo \"deleted \$done_n file(s), \$freed B\"
echo \"absent already: \$miss_n\"
echo \"errors: \$err_n\"
rm -f /tmp/prune_names.txt
date -u" 2>&1 | tee -a "$LOG"; then
  echo "WARNING: the remote prune loop exited nonzero -- read the lines above for how far it got" | tee -a "$LOG"
fi

echo "=== df AFTER ===" | tee -a "$LOG"
~/bin/pod 'df -h /work | tail -2' 2>&1 | tee -a "$LOG"
echo "next: regenerate the listing, then commit the after-state" | tee -a "$LOG"

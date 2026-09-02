#!/bin/bash
# Relaunch a training run ONCE if it crashes. The auto-resume that `harness launch`
# provides, for a run that cannot go through `harness launch`.
#
#   setsid nohup bash scripts/supervise_run.sh <name> -- <command...> &
#
# WHY THIS EXISTS. --auto-resume lives in harness.py's _supervise (:9315) and needs a
# live parent to reap the exit code. The 500M launch had to bypass `harness launch`
# entirely, because _allocation_cards (:8829) reads the six-point ladder's frozen
# config and would have forced world=7 onto a run that is not a ladder point. That
# bypass was correct and it silently took auto-resume with it: the run went 40 minutes
# with no supervisor while its own gate test had proven resume works (fb, 2026-09-02).
#
# ONE RESUME, NEVER MORE. train.py:2308 sets _plan_step_origin = resume_step and :1386
# computes the row cursor as (step - origin) * batch * accum, so the cursor is
# per-segment: it describes rows drawn since THIS resume, not since the start. A second
# resume re-reads what the first one already trained on -- 10,240 rows at this shape --
# and no test has ever covered it. Exhausted is a stop, not a third attempt.
set -u

NAME="${1:?usage: supervise_run.sh <name> -- <command...>}"
shift
[ "${1:-}" = "--" ] || { echo "usage: supervise_run.sh <name> -- <command...>" >&2; exit 2; }
shift
[ $# -gt 0 ] || { echo "no command given after --" >&2; exit 2; }

ROOT="${SUPERVISE_ROOT:-/work/aupai}"
cd "$ROOT" || exit 1
LOG="runs/${NAME}.supervisor.log"

say() { echo "[supervisor $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# The newest ckpt_<name>.pt.stepN. Sorted numerically on the suffix, not lexically:
# .step9000 sorts above .step10000 as a string, and resuming from the older one silently
# discards real training.
latest_ckpt() {
  ls -1 "ckpt_${NAME}.pt.step"* 2>/dev/null \
    | sed 's/.*\.step\([0-9]*\)$/\1 &/' | sort -k1,1n | tail -1 | cut -d' ' -f2-
}

say "watching: $*"
"$@" &
CHILD=$!
say "child pid $CHILD"
wait $CHILD
rc=$?

if [ $rc -eq 0 ]; then
  say "exit 0 -- clean finish, nothing to resume"
  exit 0
fi

# 42 is train.py's reserved kill-criterion exit: a deliberate abort (NaN, kill
# criterion), not a crash. Resuming it would relaunch into the condition that stopped it.
if [ $rc -eq 42 ]; then
  say "exit 42 -- deliberate stop (kill criterion), NOT resuming"
  exit 42
fi

CKPT=$(latest_ckpt)
if [ -z "$CKPT" ]; then
  say "exit $rc -- no ckpt_${NAME}.pt.stepN exists, crashed before the first save. NOT resuming."
  exit $rc
fi

# A checkpoint still being written is a truncated file, and resuming from it reads as a
# corrupt-resume failure with the wrong cause. Two identical sizes 5s apart is the save
# having finished, whatever the disk was doing (the same wait prove_resume.sh uses).
# `wc -c`, not `stat -c %s`: -c is GNU-only, and on a BSD stat it fails into `|| echo 0`,
# where a size that is never non-zero never satisfies the break -- the wait then burns its
# full 5 minutes and resumes anyway, silently. Measured on this Mac before it shipped.
prev=-1
for _ in $(seq 1 60); do
  sz=$(wc -c < "$CKPT" 2>/dev/null | tr -d ' ' || echo 0)
  [ "$sz" = "$prev" ] && [ "${sz:-0}" != "0" ] && break
  prev=$sz
  sleep 5
done

say "exit $rc -- resuming ONCE from $CKPT ($prev bytes)"
sleep 30   # let the dead ranks release their cards before eight more ask for them

"$@" --resume "$CKPT" &
CHILD=$!
say "resumed child pid $CHILD"
wait $CHILD
rc2=$?
say "exit $rc2 after one resume -- auto-resume exhausted, no further attempt"
exit $rc2

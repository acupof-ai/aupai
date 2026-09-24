#!/bin/bash
# Auto HumanEval on every new ckpt_v41_ced_0923.pt.stepN, CPU-sharded, co-resident with the run.
# Task: ~/.aupai-team/task_gpu_eval.md step 3 (1e, 2026-09-23). GPU path refused by the card
# claim guard (all 8 cards held by v41_ced_0923, lane_card null), so this is CPU-only.
#
# Two pods-specific facts are pinned here because both cost a failed launch:
#   * /bin/sh is dash, so bash arrays are not available -- the core groups are plain strings.
#   * --device cpu refuses unless CUDA_VISIBLE_DEVICES is set (empty means cardless).
#
# Single instance via flock (never a pid file: a stale pid file and a live job look alike).
# Iteration cap because a loop keyed on `ls` runs forever if a new ckpt keeps appearing.
export CUDA_VISIBLE_DEVICES=
NAME=v41_ced_0923
CK=/work/aupai/ckpt_v41_ced_0923.pt
LOCK=/work/aupai/runs/heval_auto.lock
DONE=/work/aupai/runs/heval_auto.done
FAILED=/work/aupai/runs/heval_auto.failed
GIVEUP=/work/aupai/runs/heval_auto.giveup
MERGE_RETRIES=3
SHARDS=8
THREADS=8
# THE CAP COUNTS POLLS, NOT EVALUATIONS: a round with no complete checkpoint increments it too.
# A shorter poll means MORE iterations per checkpoint, so POLL and MAX_ITER move together: at
# POLL=60 and ~2000 steps between saves at ~3.7 s/step, one checkpoint cycle costs ~124
# iterations, and the remaining checkpoints of the 30B schedule want ~1000. 800 is kept because
# the loop is restarted by hand between evaluations anyway; raise it if the run is left alone.
MAX_ITER=${MAX_ITER:-800}
# SCOPE: only steps >= MIN_STEP are evaluated. 2000 and 4000 already have their readings by
# other means (2000 from the single-process run, 4000 from the first 8-shard run). They are
# NOT marked evaluated here: the loop simply does not look below MIN_STEP.
#
# LIMIT OF THIS SCRIPT: the self-heal re-queues a step whose result file is missing, but it
# re-runs ALL EIGHT shards. On the step12000 shape (one shard timed out, the other seven
# finished) that repeats ~7 shards' work. Re-running just the short shard needs the merge's
# gap reported per shard, which this does not read. Accepted for now: correctness first,
# wasted CPU second.
MIN_STEP=${MIN_STEP:-6000}
# AND ONLY ON THE GRID, because the arithmetic does not close otherwise. The resumed run
# saves every 500 steps (~30 min at ~3.7 s/step) while one evaluation takes ~41 min, so
# scoring every checkpoint falls behind by construction and the backlog only grows. Score
# step % GRID == 0, plus the final checkpoint of the schedule whatever its step -- that one
# is the gate reading and must not be skipped for being off-grid.
#
# The step is the SAVE NUMBER, which after a resume is the absolute training step (the run
# reads `step` from the checkpoint), so `% 2000` selects the same points the first segment
# saved on the 2000 grid.
GRID=${GRID:-2000}
TOTAL_STEPS=${TOTAL_STEPS:-38146}
SAVE_EVERY=${SAVE_EVERY:-500}
# The FINAL checkpoint is written by train.py:4555 as the suffixless ckpt_<name>.pt, AFTER the
# loop reaches total_steps -- it is not a .stepN file at all. Every earlier completion criterion
# here is keyed on a .step suffix, so the one checkpoint the gate actually reads was unreachable:
# 38146 is not a multiple of GRID, and no glob of "$CK".step* can match a name without the suffix.
FINAL_CKPT=$CK
# COMPLETION CRITERIA -- not an exact byte count. Two, and both must hold:
#   (a) the training log has printed the step line for step+10. save_checkpoint is called
#       SYNCHRONOUSLY in the training loop -- it is the `save_checkpoint(ckpt_path + f".step{step}",
#       ...)` call, which sits above the `runlog(` that emits the step line. train.py has no
#       threads or asyncio, so that line being on disk
#       proves step N's torch.save returned. (A fixed size cannot work: at the time of writing
#       step2000/4000 measured 13021167860 B while step6000 measured 13021167924 B -- the
#       save is not byte-stable. 2000 and 4000 have since been rolled off the pod.)
#   (b) the file size has not changed for 120 s -- the host backup loop's second line of
#       defence, already exercised on three checkpoints.
SETTLE_S=120
# Poll interval for a new checkpoint. 600 -> 60 buys ~4.5 min earlier start on average (half
# the poll), and a poll that finds nothing costs one `ls` of a handful of files.
POLL=${POLL:-60}
LOG=/work/aupai/runs/${NAME}.log

# Core groups: 8 x 8 CPUs, disjoint physical cores, avoiding the cores the training dataloader
# uses AND their SMT siblings (topology read from /sys, not an assumed offset).
G0="2,13,29,37,45,55,63,73";  G1="3,14,30,38,46,56,64,74"
G2="4,15,31,39,47,57,65,75";  G3="5,20,32,40,50,58,66,76"
G4="7,21,33,41,51,59,67,77";  G5="10,26,34,42,52,60,68,78"
G6="11,27,35,43,53,61,69,79"; G7="12,28,36,44,54,62,71,80"

# A step is done only when BOTH the DONE line and its result file exist. One definition, used
# by the skip test and by the post-merge bookkeeping, so a step recorded done without a result
# (the step12000 shape: merge refused on a shard gap, the append ran anyway) is re-queued on
# the next pass instead of being trusted forever.
fsize() { wc -c < "$1" 2>/dev/null || echo 0; }

done_and_produced() {
  grep -qx "$1" "$DONE" 2>/dev/null || return 1
  [ -f "runs/heval_merged_step$1_result.json" ]
}

# step_wanted <n> -- is this checkpoint's step one this loop scores? Pure arithmetic, so the
# selftest drives it without a pod, a checkpoint or a clock.
step_wanted() {
  [ "$1" -ge "$MIN_STEP" ] || return 1
  [ "$(( $1 % GRID ))" -eq 0 ] && return 0
  # Off the grid it is wanted only as the FINAL checkpoint. The save may land partway through
  # the last block, so it is matched against the schedule's end with one save interval of
  # slack rather than by equality.
  [ "$1" -ge "$(( TOTAL_STEPS - SAVE_EVERY ))" ]
}

# pick_step -- print the next checkpoint to evaluate (a step number or FINAL), or nothing.
# Extracted from the loop body so the selftest can drive the REAL selection: the defect this
# function exists to fix was in the selection itself (a glob of "$CK".step* cannot match the
# suffixless final ckpt), so a test that calls a helper the loop does not use proves nothing.
pick_step() {
  local f n sz1 sz2 next_n
  # Smallest step checkpoint that is COMPLETE by both criteria and not yet evaluated.
  #
  # ORDER BY THE NUMBER, not by the path string. `sort -t. -k3 -n` looks numeric but is not:
  # with the field separator at the last `.`, `-k3` runs to the END OF THE LINE, so `-n` parses
  # "6000" as a number only while the field holds nothing else -- and the shell's `$( )` word
  # splitting then re-reads the paths in glob order. Measured: step10000 and step12000 sort
  # BEFORE step6000/8000. The selection below scans for a `break`, so a wrong order does not
  # skip a step, but the note is here because reading it as numeric is the easy mistake.
  #
  # THE self-heal lives in one place: done_and_produced. A step counts as done only when the
  # DONE line AND its result file both exist. Both the skip test below and the post-merge
  # bookkeeping use it, so "done but never produced" cannot survive a restart.
  step=""
  for f in $(ls -1 "$CK".step* 2>/dev/null | sed 's/.*\.step//' | sort -n); do
    f="$CK.step$f"
    [ -f "$f" ] || continue
    case "$f" in *.ep*|*interrupt*) continue;; esac
    n=${f##*.step}
    case "$n" in ''|*[!0-9]*) continue;; esac
    step_wanted "$n" || continue
    done_and_produced "$n" && continue
    grep -qx "$n" "$GIVEUP" 2>/dev/null && continue
    next_n=$((n + 10))
    grep -qE "^step $next_n/" "$LOG" 2>/dev/null || continue
    sz1=$(fsize "$f")
    sleep "$SETTLE_S"
    sz2=$(fsize "$f")
    [ "$sz1" = "$sz2" ] && [ "$sz1" -gt 0 ] || continue
    echo "$n"; return 0
  done
  # THE FINAL CHECKPOINT, tried only when no .stepN is pending. It has no number of its own, so
  # it is tagged FINAL in the DONE/GIVEUP/give-up bookkeeping. The completion signal cannot be the
  # step+10 log line (there is no step+10) -- it is train.py:4557's `print(f"saved {ckpt_path}")`,
  # which reaches this log because the launcher captures stdout. That line is emitted AFTER the
  # save returns, so its presence proves the file is whole.
  if [ -f "$FINAL_CKPT" ] \
     && ! done_and_produced FINAL && ! grep -qx FINAL "$GIVEUP" 2>/dev/null \
     && grep -qF "saved $FINAL_CKPT" "$LOG" 2>/dev/null; then
    sz1=$(fsize "$FINAL_CKPT")
    sleep "$SETTLE_S"
    sz2=$(fsize "$FINAL_CKPT")
    [ "$sz1" = "$sz2" ] && [ "$sz1" -gt 0 ] && { echo FINAL; return 0; }
  fi
  return 0
}

if [ "${1:-}" = "--selftest" ]; then
  # Every assertion here is a world a reviewer had to build by hand, committed so the next
  # reader does not have to. Nothing touches the pod: DONE/CK are redirected to a temp dir.
  #
  # WHY THIS IS IN THE FILE AND NOT IN A TEST HARNESS: no CI hook in this repo runs a shell
  # script's behaviour, so a selftest that lives only in a report is a claim, not a check.
  _st_fail=0
  _st() { # _st <label> <expected rc> <actual rc>
    if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1 (want $2, got $3)"; _st_fail=1; fi
  }
  d=$(mktemp -d) || exit 1
  DONE="$d/done"; GIVEUP="$d/giveup"; CK="$d/ckpt"
  : > "$DONE"; : > "$GIVEUP"
  mkdir -p "$d/runs"

  # --- selection grid: only step % 2000, plus the final checkpoint.
  step_wanted 6000;  _st "grid 6000" 0 $?
  step_wanted 8000;  _st "grid 8000" 0 $?
  step_wanted 14500; _st "off-grid 14500" 1 $?
  step_wanted 15000; _st "off-grid 15000" 1 $?
  step_wanted 4000;  _st "below MIN_STEP" 1 $?
  step_wanted 38000; _st "grid point near the end" 0 $?
  step_wanted 37000; _st "off-grid, not near the end" 1 $?
  # THE FINAL-CHECKPOINT CLAUSE NEEDS A TOTAL THAT PUTS ITS THRESHOLD OFF THE GRID, or it is
  # untested. With the live constants the threshold is TOTAL-SAVE = 38146-500 = 37646, and the
  # last save lands on 38000 -- already a multiple of 2000, so a world written against those
  # constants passes on the grid clause alone. Measured: replacing this clause with `return 1`
  # left that world green.
  #
  # The threshold moves with TOTAL, and the two worlds below straddle the THRESHOLD it creates
  # (37900-500 = 37400), not any step number quoted in prose: 37500 is above it and off-grid,
  # 37000 is below it. Both numbers are derived from the constants on the two lines above them,
  # so a change to either constant moves the test with it.
  _st_saved_total=$TOTAL_STEPS
  TOTAL_STEPS=37900                                   # threshold becomes 37400
  step_wanted 37500; _st "off-grid final checkpoint (above threshold)" 0 $?
  step_wanted 37000; _st "off-grid, below the threshold" 1 $?
  TOTAL_STEPS=$_st_saved_total

  # --- done_and_produced: the step12000 shape is the first of these.
  run_dp() { ( cd "$d" && DONE="$DONE" ; done_and_produced "$1" ); }
  echo 12000 > "$DONE"
  run_dp 12000; _st "DONE line, no result -> not done" 1 $?
  : > "$d/runs/heval_merged_step12000_result.json"
  run_dp 12000; _st "DONE line and result -> done" 0 $?
  : > "$d/runs/heval_merged_step6000_result.json"
  run_dp 6000;  _st "result, no DONE line -> not done" 1 $?
  run_dp 9999;  _st "neither -> not done" 1 $?

  # --- the FINAL checkpoint: the gate's own reading, and the one the old loop could never reach.
  # It is selected from a file that is NOT $CK.stepN, so it is tested through pick_step -- the
  # real selection loop -- rather than by calling a helper, because the defect WAS the selection:
  # a glob of "$CK".step* cannot match a name with no .step suffix, and 38146 is off the grid.
  run_pick() { ( cd "$d" && CK="$CK" LOG="$LOG" DONE="$DONE" GIVEUP="$GIVEUP" FINAL_CKPT="$FINAL_CKPT" \
                 SETTLE_S=0 ; pick_step ); }
  LOG="$d/run.log"; FINAL_CKPT="$d/final"; : > "$LOG"; : > "$DONE"; : > "$GIVEUP"
  mkdir -p "$d/data/eval"; cd "$d" || exit 1

  # (a) file present, no "saved" line yet: the save may still be in flight, so NOT picked.
  printf 'x' > "$FINAL_CKPT"
  got=$(run_pick); _st "final ckpt with no saved-line -> not picked" "" "$got"
  # (b) a 0-byte file is refused even WITH the saved-line -- "exists" is not "written".
  echo "saved $FINAL_CKPT" > "$LOG"; : > "$FINAL_CKPT"
  got=$(run_pick); _st "0-byte final ckpt is not picked even with the saved-line" "" "$got"
  # (c) saved-line present and non-empty -> picked, tagged FINAL (it has no step number).
  printf 'x' > "$FINAL_CKPT"
  got=$(run_pick); _st "final ckpt + saved-line -> picked as FINAL" "FINAL" "$got"
  # (d) once it has a result file, it is not re-picked.
  : > "$d/runs/heval_merged_stepFINAL_result.json"; echo FINAL > "$DONE"
  got=$(run_pick); _st "final ckpt already produced -> not re-picked" "" "$got"

  # WORLD (e) NEEDS ITS OWN CLEAN STATE. The "already produced" predicate is clearable in two
  # places, and resetting only $DONE leaves (d)'s result file on disk -- measured: with a
  # reordered pick_step that tried FINAL first, this world still passed, i.e. it asserted nothing.
  : > "$DONE"; : > "$GIVEUP"; rm -f "$d/runs/heval_merged_stepFINAL_result.json"
  # (e) a .stepN on the grid still WINS over FINAL -- the final ckpt is the gate reading, not a
  # reason to skip an earlier one. BOTH log lines must be present, or the FINAL branch is
  # disqualified by its saved-line grep rather than by the ordering.
  printf 'x' > "$CK.step6000"
  printf 'saved %s\nstep 6010/100000\n' "$FINAL_CKPT" > "$LOG"
  got=$(run_pick); _st "pending .stepN is picked before FINAL" "6000" "$got"
  rm -f "$CK.step6000"

  # --- ordering: the smallest step must come first, which is what the old sort got wrong.
  for n in 2000 6000 8000 10000 12000; do : > "$CK.step$n"; done
  got=$(ls -1 "$CK".step* | sed 's/.*\.step//' | sort -n | head -1)
  _st "smallest step first" 2000 "$got"

  rm -rf "$d"
  [ "$_st_fail" -eq 0 ] && echo "selftest: all worlds pass" || echo "selftest: FAILURES above"
  exit "$_st_fail"
fi

exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "another instance holds $LOCK"; exit 0; }

cd /work/aupai || exit 1
touch "$DONE" "$FAILED" "$GIVEUP"


iter=0
while [ "$iter" -lt "$MAX_ITER" ]; do
  iter=$((iter + 1))
  step=$(pick_step)
  if [ -z "$step" ]; then sleep "$POLL"; continue; fi

  # ONE place where FINAL becomes concrete paths. The tag is what DONE/GIVEUP carry and what the
  # result file is named after; ckf is the file to load; base is the preds-name stem, which for
  # FINAL is the suffixless ckpt name.
  if [ "$step" = FINAL ]; then tag=FINAL; ckf=$FINAL_CKPT; base=$PAT; else tag=$step; ckf=$CK.step$step; base=$PAT.step$step; fi
  echo "=== $(date -u +%H:%M:%SZ) evaluating step $tag"
  pids=""
  i=0
  while [ "$i" -lt "$SHARDS" ]; do
    eval "cores=\$G$i"
    setsid nohup taskset -c "$cores" python3 eval/humaneval_gen.py \
      --ckpt "$ckf" --device cpu --threads "$THREADS" --rstrip_nl \
      --shard_i "$i" --shard_n "$SHARDS" --run "ced_s${tag}_rstrip_sh$i" --force \
      > "runs/heval_auto_${tag}_sh${i}.log" 2>&1 < /dev/null &
    i=$((i + 1))
  done
  wait
  echo "=== $(date -u +%H:%M:%SZ) shards done for step $tag"
  # THE MERGE'S EXIT STATUS IS THE DECISION, and it has to be captured WITHOUT a pipe: `$?`
  # after `cmd | tail` is tail's status, so the original code appended to DONE even when
  # e0_merge_score refused. `merge()` raises SystemExit on a shard gap, a duplicate
  # (task_id, sample_idx) or a short sample count -- all before writing anything -- so the
  # guard has to be the command itself.
  #
  # The rule is the RESULT FILE, not the merged jsonl (e0_merge_score writes the jsonl first)
  # and not "non-empty" (a partial file is still a file). Only when it exists does the step
  # count as produced.
  merge_log="runs/heval_auto_merge_${tag}.log"
  if python3 eval/e0_merge_score.py --bench humaneval --n 1 \
      --glob "data/eval/preds_humaneval_${base}.rstripnl.shard*of8.ced_s${tag}_rstrip_sh*.jsonl" \
      --out "runs/heval_merged_step${tag}.jsonl" \
      --result "runs/heval_merged_step${tag}_result.json" > "$merge_log" 2>&1 \
      && [ -f "runs/heval_merged_step${tag}_result.json" ]; then
    echo "$tag" >> "$DONE"
    tail -2 "$merge_log"
    echo "=== $(date -u +%H:%M:%SZ) step $tag DONE (result written)"
  else
    # LOUD, and retried -- but not forever. The attempt count is derived from the log file so
    # it survives a restart of this loop.
    att=$(grep -c "MERGE-FAILED step $tag" "$FAILED" 2>/dev/null); att=${att:-0}
    echo "MERGE-FAILED step $tag (attempt $((att + 1)) of $MERGE_RETRIES)" | tee -a "$FAILED"
    tail -3 "$merge_log" 2>/dev/null
    if [ "$((att + 1))" -ge "$MERGE_RETRIES" ]; then
      echo "=== GIVING UP on step $tag after $MERGE_RETRIES attempts; leaving it un-done and moving on" | tee -a "$FAILED"
      echo "$tag" >> "$GIVEUP"
    fi
  fi
done
echo "=== STOPPED: iteration cap MAX_ITER=$MAX_ITER reached (polls counted, not evaluations) at $(date -u +%H:%M:%SZ); last evaluated step: $(tail -1 "$DONE" 2>/dev/null)"

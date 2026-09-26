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
# The script's own directory, resolved ONCE at the top. A later `_st_root` computed from
# BASH_SOURCE inside the selftest resolved to the TEMP dir instead -- the hook copies the file
# to a scratch tree before running it, so by then BASH_SOURCE points there, not at the repo.
SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
REPO_ROOT=$(cd "$SELF_DIR/.." && pwd)
NAME=${NAME:-v41_ced_0923}
CK=${CK:-/work/aupai/ckpt_${NAME}.pt}
# Run-scoped state and result names. Empty for v41_ced_0923 so its existing files keep their
# names; any other run gets its own, or its steps would read as done from 0923's DONE list and
# its merged results would overwrite 0923's heval_merged_step<N> files.
SFX=""; [ "$NAME" = v41_ced_0923 ] || SFX="_${NAME}"
PAT=${PAT:-$(basename "$CK")}
LOCK=/work/aupai/runs/heval_auto${SFX}.lock
DONE=/work/aupai/runs/heval_auto${SFX}.done
FAILED=/work/aupai/runs/heval_auto${SFX}.failed
GIVEUP=/work/aupai/runs/heval_auto${SFX}.giveup
MERGE_RETRIES=3
SHARDS=11
THREADS=8
# A worker holds the whole model: measured 13.0GB RSS, 98% ANONYMOUS (not reclaimable cache),
# and the ckpt is 12.1GiB loaded per process with no sharing. So a worker is only started when
# its node has room, and the check is per-node because node0 and node1 were measured at 144GB
# and 41GB free while the 8 real shards were running. 14GB = 13.0 measured + headroom.
MIN_FREE_KB=${MIN_FREE_KB:-$((14 * 1024 * 1024))}
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

# Core groups: 11 workers x 8 CPUs (node0 8, node1 3), excluded against TRAINING's hot cores.
# Each worker's 8 threads sit on 4 physical cores (two threads per core), the same shape the
# previous 8-way groups used -- G0 and G1 there are each other's SMT siblings.
#
# SELECTION IS FROM TOPOLOGY, NOT BY HAND. A physical core is excluded when ANY of its threads
# is hot, so no eval thread ever shares a core with a trainer thread. The hot set is ONE
# SAMPLE (busy = user+nice+system+irq+softirq, idle and iowait excluded) and training is not
# pinned, so the scheduler may move it -- which is why the ACCEPTANCE TEST IS TRAINING'S OWN
# SPEED, not this list. 4 further cores per node are left unassigned for the scheduler.
G0="2,3,4,5,8,9,10,11";       G1="14,15,16,17,18,19,20,21"
G2="22,23,24,25,26,27,28,29"; G3="30,31,32,33,34,35,36,37"
G4="38,39,40,41,42,43,44,45"; G5="46,47,48,49,50,51,52,53"
G6="54,55,56,57,58,59,60,61"; G7="64,65,68,69,70,71,72,73"
G8="90,91,92,93,94,95,96,97"; G9="98,99,100,101,102,103,104,105"
G10="106,107,108,109,110,111,112,113"
# node of each group, for numactl --cpunodebind/--membind. G0..G7 are node0, G8..G10 node1.
G_NODE="0 0 0 0 0 0 0 0 1 1 1"

# A step is done only when BOTH the DONE line and its result file exist. One definition, used
# by the skip test and by the post-merge bookkeeping, so a step recorded done without a result
# (the step12000 shape: merge refused on a shard gap, the append ran anyway) is re-queued on
# the next pass instead of being trusted forever.
fsize() { wc -c < "$1" 2>/dev/null || echo 0; }

done_and_produced() {
  grep -qx "$1" "$DONE" 2>/dev/null || return 1
  [ -f "runs/heval_merged${SFX}_step$1_result.json" ]
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
# paths_for <tag> -- emit ckf/base as shell assignments for a step number or FINAL.
# A FUNCTION because the glob built from `base` fails SILENTLY: `preds_humaneval_${base}.rstripnl
# .shard*of8...jsonl` with an empty or wrong base matches nothing, e0_merge_score exits on
# "no shard files match", and the loop records MERGE-FAILED without saying which half was wrong.
# That is exactly how a never-defined PAT shipped: $PAT expanded to the empty string under `set -u`
# being unset here, both paths broke, and no test looked at the paths at all.
paths_for() {
  local t="$1"
  if [ "$t" = FINAL ]; then
    printf 'ckf=%q base=%q\n' "$FINAL_CKPT" "$PAT"
  else
    printf 'ckf=%q base=%q\n' "$CK.step$t" "$PAT.step$t"
  fi
}

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
  # GROUND TRUTH for the preds-filename shape, read from the REAL ckpt path before CK is
  # reassigned to a temp fixture below. Naming the fixture from $PAT instead makes the test
  # self-referential and blind: a wrong PAT renames the fixture too, so pattern and file always
  # agree. Measured -- with PAT=ckpt_WRONG_name.pt every world stayed green.
  _st_ckname=$(basename "$CK")
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

  # --- run scoping: another run's name must not read 0923's DONE list or result files. The
  # rule is read from the script's own top-level `SFX=` line and executed, never re-typed here:
  # a grep for its text would also match this comment block and could not fail.
  _st_rule=$(grep '^SFX=' "$0")
  _st "exactly one top-level SFX rule" 1 "$(printf '%s\n' "$_st_rule" | grep -c .)"
  _st "0926 gets its own suffix" "_v41_ced_0926" "$(NAME=v41_ced_0926; eval "$_st_rule"; echo "$SFX")"
  _st "0923 keeps the unsuffixed names" "" "$(NAME=v41_ced_0923; eval "$_st_rule"; echo "$SFX")"
  ( cd "$d" && SFX=_v41_ced_0926 && echo 6000 > "$DONE.x" && DONE="$DONE.x" && : > "runs/heval_merged_step6000_result.json" \
      && done_and_produced 6000 ); _st "0926 does not count 0923's result file as its own" 1 $?
  rm -f "$DONE.x"

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

  # --- paths_for: the glob must MATCH A REAL FILENAME, not merely be non-empty.
  # This is the check whose absence let an undefined PAT ship. The failure is silent in the
  # tightest way: an empty base builds a glob that matches nothing, e0_merge_score exits
  # "no shard files match", and every step records MERGE-FAILED. Nothing printed the path.
  # CK is the real pod path (defined at the top of the file), but the glob is matched from the
  # selftest's temp dir: build the fixture under $d/data/eval and run the match there.
  run_paths() { ( eval "$(paths_for "$1")"; printf '%s|%s' "$ckf" "$base" ); }
  # The fixture is named from _st_ckname (the real ckpt basename), NOT from $PAT. CK is also
  # reassigned to $d/ckpt above, so `basename $CK` would name it `..._ckpt.` while paths_for
  # returned `..._ckpt_v41_ced_0923.pt.` -- a mismatch that reads as "the glob is broken" when
  # the fixture is what is wrong. Measured: both worlds failed for exactly that reason.
  mkdir -p "$d/data/eval"
  : > "$d/data/eval/preds_humaneval_${_st_ckname}.step6000.rstripnl.shard0of8.ced_s6000_rstrip_sh0.jsonl"
  : > "$d/data/eval/preds_humaneval_${_st_ckname}.rstripnl.shard0of8.ced_sFINAL_rstrip_sh0.jsonl"

  # THE GLOB IS EXPANDED THE WAY THE CONSUMER EXPANDS IT. e0_merge_score does
  # `glob.glob(args.glob)` in python; the loop passes the pattern as a QUOTED argument. A bash
  # `ls "…*…"` would hand the pattern over literally and report 0 matches against a file that is
  # right there -- measured, on all four of these worlds, which is how a passing test would have
  # "confirmed" the defect. Match with python's glob, the subject's own mechanism.
  pglob() { python3 -c 'import glob,sys; print(len(glob.glob(sys.argv[1])))' "$1"; }

  got=$(run_paths 6000); _st "paths_for 6000: ckf is the .stepN file" "$CK.step6000|$PAT.step6000" "$got"
  gg=$(pglob "$d/data/eval/preds_humaneval_$(run_paths 6000 | cut -d'|' -f2).rstripnl.shard*of8.ced_s6000_rstrip_sh*.jsonl")
  _st "paths_for 6000: the merge glob matches a real file" 1 "$gg"

  got=$(run_paths FINAL); _st "paths_for FINAL: ckf is the suffixless ckpt" "$FINAL_CKPT|$PAT" "$got"
  gf=$(pglob "$d/data/eval/preds_humaneval_$(run_paths FINAL | cut -d'|' -f2).rstripnl.shard*of8.ced_sFINAL_rstrip_sh*.jsonl")
  _st "paths_for FINAL: the merge glob matches a real file" 1 "$gf"
  # AND the control: the old undefined-PAT shape matches ZERO, which is what made the defect
  # silent rather than loud. Without this line the worlds above would also pass for a pattern so
  # broad it matches anything.
  badg=$(pglob "$d/data/eval/preds_humaneval_.step6000.rstripnl.shard*of8.ced_s6000_rstrip_sh*.jsonl")
  _st "empty base matches nothing (the shipped failure shape)" 0 "$badg"

  # --- the worker groups: 11 disjoint sets that never touch a hot core's physical core.
  # Read from the SCRIPT, so a hand-edit that reintroduces an overlap fails here. The failure
  # is quiet otherwise: two workers sharing a physical core just run slower, and nothing in
  # the merge or the counts would notice.
  _st_groups() { for k in $(seq 0 $((SHARDS - 1))); do eval "echo \$G$k"; done; }
  # (i) every group has THREADS cpus
  bad=$(for g in $(_st_groups); do [ "$(echo "$g" | tr ',' '\n' | wc -l)" -eq "$THREADS" ] || echo "$g"; done)
  _st "every group has THREADS cpus" "" "$bad"
  # (ii) no cpu appears in two groups -- disjointness is what makes the shards independent
  tot=$(echo "$(_st_groups)" | tr ',' '\n' | sort -n | wc -l)
  uniq=$(echo "$(_st_groups)" | tr ',' '\n' | sort -nu | wc -l)
  _st "groups are disjoint" "$tot" "$uniq"
  # (iii) NO GROUP MEMBER SHARES A PHYSICAL CORE WITH A HOT CORE. The hot set is one sample
  # (busy = user+nice+system+irq+softirq; idle and iowait excluded) and training is unpinned, so
  # this cannot promise current exclusion -- it pins the EXCLUSION THE GROUPS WERE BUILT WITH,
  # which is the thing a hand-edit would silently undo. The old groups failed exactly this:
  # 13 and 63 were sampled hot and both were group members.
  _st_hot="1 6 13 63 66 122 123 138 142 150 152 168"
  # The sibling lists are 'N' or 'A-B' RANGES (measured: cpu13 -> '12-13'), so a plain
  # membership test misses half of every pair. Expand the range.
  # Siblings come from a COMMITTED SNAPSHOT by default, not from /sys -- see the block below.
  # The root is a parameter so the same code runs against the snapshot and against a live host.
  _st_sibs() { # _st_sibs <cpu> <snapshot-path|"">
    local c="$1" snap="$2" sh="$3" f="$4"
    if [ -n "$snap" ]; then
      awk -v c="$c" '$1=="cpu" && $2==c {print $4; exit}' "$snap"
      return
    fi
    f="${4:-/sys/devices/system/cpu/cpu$c/topology/}/thread_siblings_list"
    [ -r "$f" ] || { echo "UNREADABLE"; return; }
    tr ',' '\n' < "$f"
  }
  # Expand 'A-B' or 'N' into a list of cpus.
  _st_expand() {
    tr ',' '\n' | while read -r x; do
      case "$x" in
        *-*) a=${x%-*}; b=${x#*-}; seq "$a" "$b" | tr '\n' ' ';;
        "")  ;;
        *)   printf '%s ' "$x";;
      esac
    done
  }
  # AN UNREADABLE TOPOLOGY MUST NOT READ AS A PASS. On macOS /sys does not exist, so the old
  # guard `[ -r ] || continue` skipped every cpu and this world went green while asserting
  # nothing -- measured: a group holding hot cores 1/6/13/63 passed. The hook runs this selftest
  # on macOS, so the vacuous pass was the NORMAL case.
  #
  # It cannot simply FAIL there: the same hook gates every commit on a dev box, and refusing
  # every commit for a check that needs Linux is a worse failure than the one it detects. So it
  # SKIPS OUT LOUD when the top-level dir is absent, and the skip is itself asserted -- a silent
  # skip and a vacuous pass are the same defect one level apart.
  # THE TOPOLOGY COMES FROM A COMMITTED SNAPSHOT, so this world RUNS EVERYWHERE -- CI, macOS,
  # the pod. An earlier version read /sys and SKIPPED when it was absent, which meant it passed
  # on the two machines that run it most (a dev box, and a GitHub runner with a small single-node
  # /sys) and only ever bit on the pod. genB measured that: CI red on `topology readable on every
  # group cpu`, 33 of 34 worlds green. A world that skips where it is usually run is not a check.
  # From REPO_ROOT, captured at the top of the file. Two earlier attempts failed here: a
  # relative path misses because the selftest runs from a temp dir, and BASH_SOURCE resolved to
  # the temp tree because the hook copies this file there before running it.
  _st_snap=${TOPO_SNAPSHOT:-$REPO_ROOT/data/eval/topo_snapshot_30b.tsv}
  if [ -r "$_st_snap" ]; then
    # (a) every group cpu exists in the snapshot, and the node it is bound to has it
    _st_top=0
    _st_i=0
    for g in $(_st_groups); do
      _st_i=$((_st_i + 1))
      n=$(echo $G_NODE | cut -d" " -f$_st_i)
      # The node cpulist is a RANGE ("0-89"), so a literal membership test never matches a bare
      # cpu number. Expand it first -- measured: `case ",0-89," in *",2,"*` is false.
      nlist=$(awk -v n="$n" '$1=="node" && $2==n {print $4}' "$_st_snap" | _st_expand | tr -s ' ' ',')
      for c in $(echo "$g" | tr ',' ' '); do
        awk -v c="$c" '$1=="cpu" && $2==c {found=1} END{exit !found}' "$_st_snap" || {
          echo "  group $_st_i cpu $c is not in the snapshot"; _st_top=1; }
        case ",$nlist," in *",$c,"*) ;; *) echo "  group $_st_i cpu $c is not on node$n (cpulist $nlist)"; _st_top=1;; esac
      done
    done
    _st "every group cpu exists and sits on its declared node" 0 "$_st_top"
    clash=""
    for c in $(_st_groups | tr ',' ' '); do
      for s in $(_st_sibs "$c" "$_st_snap" | _st_expand); do
        case " $_st_hot " in *" $s "*) clash="$clash $c/$s";; esac
      done
    done
    # AN EMPTY CLASH HERE IS A MEASUREMENT, NOT A SKIP: the snapshot has real sibling data, so a
    # group holding a hot core fails on any machine. Verified by mutation (see the PR).
    _st "no group cpu shares a physical core with a sampled-hot core" "" "$clash"
  else
    _st "topology snapshot present (else this world cannot run)" 1 0
  fi

  # (iv) the group count matches SHARDS, and G_NODE has one entry per group
  _st "one group per shard" "$SHARDS" "$(_st_groups | wc -l | tr -d ' ')"
  _st "G_NODE has one node per group" "$SHARDS" "$(echo $G_NODE | wc -w | tr -d ' ')"

  # --- a worker is DROPPED, not failed, when its node is short on memory -- and the drop must
  # not leave a hole in shard_i, because the merge refuses a gap. Modelled on the launch loop's
  # own arithmetic: index by launch order, not by group number.
  assign() { # assign <free_kb...> -> the shard_i values that would be launched
    local idx=0 out=""
    local k=0
    for f in "$@"; do
      if [ "$f" -ge "$MIN_FREE_KB" ]; then out="$out $idx"; idx=$((idx + 1)); fi
      k=$((k + 1))
    done
    echo "$out" | sed 's/^ //'
  }
  _st "no drop: 0..2" "0 1 2" "$(assign 99999999 99999999 99999999)"
  _st "middle node short: still contiguous" "0 1" "$(assign 99999999 1 99999999)"
  # A HOLE IS THE FAILURE THIS GUARDS: assigning by group number would give 0 2 here, and the
  # merge's `shard set {0,2} is not 0..k` refusal would fire at the end of a 40-minute eval.
  _st "first node short: no hole (would be 1 2 by group number)" "0 1" "$(assign 1 99999999 99999999)"
  _st "all short: nothing to merge" "" "$(assign 1 1 1)"

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

  # ONE place where FINAL becomes concrete paths -- see paths_for, which is also what the
  # selftest drives. The tag is what DONE/GIVEUP carry and what the result file is named after;
  # ckf is the file to load; base is the preds-name stem (the suffixless ckpt name for FINAL).
  tag=$step
  eval "$(paths_for "$tag")"
  echo "=== $(date -u +%H:%M:%SZ) evaluating step $tag"
  # ONE SHARED QUEUE per evaluation: every worker claims from it, so a worker that finishes
  # early takes the next task instead of idling while the slowest peer drains its fixed slice.
  # --shard_i/--shard_n still NAME the worker (the merge needs 0..k contiguous); only the
  # selection changes.
  qdir="runs/heval_q${SFX}_${tag}"
  rm -rf "$qdir"; mkdir -p "$qdir"
  # NWORK is the number actually launched, and it is what the merge glob is matched against.
  # A worker is dropped when its node lacks memory for the weights, and the drop must not leave
  # a hole in shard_i: assign indices in launch order, not by group number.
  nwork=0
  i=0
  while [ "$i" -lt "$SHARDS" ]; do
    eval "cores=\$G$i"
    node=$(echo $G_NODE | cut -d" " -f$((i + 1)))
    need=$(awk "/MemFree/{print \$4}" /sys/devices/system/node/node${node}/meminfo 2>/dev/null || echo 0)
    if [ "${need:-0}" -lt "$MIN_FREE_KB" ]; then
      echo "=== skipping worker $i: node$node free $((need / 1048576))GB < $((MIN_FREE_KB / 1048576))GB"
      i=$((i + 1)); continue
    fi
    # numactl, not taskset: --membind pins the weights to the node the CPUs are on, which is the
    # half of NUMA placement that taskset does not do. Both are passed so a numactl failure on
    # --cpunodebind alone cannot silently drop the cpu restriction.
    setsid nohup numactl --physcpubind="$cores" --membind="$node" \
      python3 eval/humaneval_gen.py \
      --ckpt "$ckf" --device cpu --threads "$THREADS" --rstrip_nl \
      --queue_dir "$qdir" \
      --shard_i "$nwork" --shard_n "$SHARDS" --run "ced_s${tag}_rstrip_sh$nwork" --force \
      > "runs/heval_auto${SFX}_${tag}_sh${nwork}.log" 2>&1 < /dev/null &
    nwork=$((nwork + 1))
    i=$((i + 1))
  done
  wait
  echo "=== $nwork of $SHARDS worker(s) ran; merging shard_i 0..$((nwork - 1))"
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
  merge_log="runs/heval_auto_merge${SFX}_${tag}.log"
  if python3 eval/e0_merge_score.py --bench humaneval --n 1 \
      --glob "data/eval/preds_humaneval_${base}.rstripnl.shard*of${SHARDS}.ced_s${tag}_rstrip_sh*.jsonl" \
      --out "runs/heval_merged${SFX}_step${tag}.jsonl" \
      --result "runs/heval_merged${SFX}_step${tag}_result.json" > "$merge_log" 2>&1 \
      && [ -f "runs/heval_merged${SFX}_step${tag}_result.json" ]; then
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

#!/bin/bash
# DDP pretraining on all 8 GPUs. Flags: see `python train.py --help`.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
cd "$(dirname "$0")"

# A training run on stale code is worse than no run: it produces a number we read as
# the new recipe's result. train.py's manifest gate cannot catch it -- it compares the
# pod against the pod's OWN manifest, so a whole-tree push of an old sha is internally
# consistent and passes. Only the stamp pod_push.sh leaves says WHICH sha that is.
#
# The pod has no git and no route back to the pushing machine, so this cannot sync;
# it can only refuse and name the command. Missing stamp = a partial push cleared it,
# or nobody ever ran --all. Either way the tree is not a known sha.
if [ ! -d .git ] && [ "${ALLOW_UNSYNCED:-}" != "1" ]; then
  if [ ! -f data/pod_synced_head ]; then
    echo "REFUSING: no data/pod_synced_head -- this pod is not at a known commit." >&2
    echo "  Run: scripts/pod_push.sh --all   (from an up-to-date main)" >&2
    echo "  A partial push clears the stamp; only --all can set it." >&2
    echo "  Deliberate exception: ALLOW_UNSYNCED=1 -- the run then cannot say what code it ran." >&2
    exit 1
  fi
  read -r _sha _dirty _when < data/pod_synced_head
  if [ "${_dirty:-1}" != "0" ]; then
    echo "REFUSING: pod was pushed from a tree with $_dirty uncommitted manifest file(s)" >&2
    echo "  ($_sha at $_when). That code exists on no commit, so the run is unreproducible." >&2
    echo "  Commit them, then: scripts/pod_push.sh --all" >&2
    exit 1
  fi
  # The stamp dates the PUSH, and files change after a push: build_starcoder_py.py was
  # edited on the pod 99 seconds after the stamp below was written, and the stamp still
  # read clean. So the stamp answers "which sha" and the manifest answers "is the tree
  # still that sha" -- neither alone is the question. runs/ divergence is expected
  # (the pod writes those rows) and --check already reports rather than fails it.
  if ! python3 scripts/pod_drift.py --check >/dev/null 2>&1; then
    echo "REFUSING: pod files drifted from the manifest since the $_when push:" >&2
    python3 scripts/pod_drift.py --check 2>&1 | grep -oE "[0-9]+ drifted: [^;]*" | head -3 >&2
    echo "  Someone edited the pod directly. Commit that change and re-run" >&2
    echo "  scripts/pod_push.sh --all, or ALLOW_UNSYNCED=1 to train on it knowingly." >&2
    exit 1
  fi
  echo "pod code: $_sha (clean, synced $_when, manifest verified)"
  # THE THIRD CLAUSE, which had no code behind it (de-14). :23 refuses dirty!=0 and :34
  # refuses drift, but "the stamp's sha is the commit the repository says the pod should
  # hold" was only PRINTED on the line above -- verified by a human reading two hex strings
  # off a 66-hour log, when the point of freezing condition 2' was that anyone could verify
  # it unattended.
  #
  # WHAT IS ANSWERABLE HERE, and the honest limit: the pod has no git and no route back, so
  # it cannot ask whether main's HEAD is still $_sha. It CAN refuse a stamp that is not a
  # full 40-char sha, which is what a hand-edit or a truncating writer produces -- and that
  # is not hypothetical: exp.py wrote 7- and 8-character shas for the same commit until
  # de-38, because `rev-parse --short` auto-scales with the object count. A stamp abbreviated
  # by any writer resolves for nobody.
  #
  # "Is $_sha main's HEAD" is answered where git is: harness check pod_stamp_is_main. Split
  # this way on purpose -- a check that cannot be run where the evidence lives returns a
  # believable answer from a filesystem that cannot hold the evidence, which is worse than
  # no answer (launch_gate.run's rule).
  case "$_sha" in
    *[!0-9a-f]* | "")
      echo "REFUSING: the stamp's sha is not hexadecimal: '$_sha'" >&2
      echo "  data/pod_synced_head is written only by scripts/pod_push.sh --all." >&2
      exit 1 ;;
  esac
  if [ ${#_sha} -ne 40 ]; then
    echo "REFUSING: the stamp holds a ${#_sha}-character sha ('$_sha'), not a full 40." >&2
    echo "  An abbreviated sha is not an identity: git's short form auto-scales with the" >&2
    echo "  object count, so the same commit prints 7 characters on one day and 8 on the" >&2
    echo "  next (de-38). Re-run: scripts/pod_push.sh --all" >&2
    exit 1
  fi
fi

torchrun --nproc_per_node="${NGPU:-8}" --master_port="${PORT:-29500}" train.py --fp8 "$@"
rc=$?
# A training run without a score-matrix record is what the score_matrix_present
# check catches; score here so the record exists by construction.
NAME=
for arg in "$@"; do
  case "$arg" in --name=*) NAME=${arg#--name=};; --name) ;; *) [ "${prev:-}" = "--name" ] && NAME=$arg;; esac
  prev=$arg
done
if [ $rc -eq 0 ] && [ -n "$NAME" ] && [ -f "ckpt_${NAME}.pt" ]; then
  # Not the block. This scoring runs inside the training shell, where
  # CUDA_VISIBLE_DEVICES is still the seven-card block, so it used to take whatever
  # card 0 was doing -- on 2026-09-01 a process holding 14.37 GiB, and the scorer
  # died asking for 96 MiB. Measure a free lane card, and queue rather than force.
  CARD=$(python scripts/harness.py free-card --wait 1800)
  SCORING_RC=0
  if [ -n "$CARD" ]; then
    CUDA_VISIBLE_DEVICES="$CARD" python eval/score_matrix.py --ckpt "ckpt_${NAME}.pt" --json runs/score_matrix.jsonl &
    SCORER_PID=$!
    # SAY THAT A SCORE IS IN FLIGHT, in the run's own row, before it can be double-started.
    # This chain runs after torchrun exits and used to write nothing, so nothing distinguished
    # "being scored right now" from "never scored" -- b0 double-scored the params leg on that
    # gap. Two events, not one: a line written only at the end cannot tell a reader whether a
    # currently-running score is theirs.
    #
    # The CARD is the identity that crosses namespaces; SCORER_PID does not. Measured
    # 2026-09-04: nvidia-smi INSIDE the container reports HOST pids (1079933, 1079934,
    # 1187488 -- none resolve in the container's own ps), so no pid this shell can print
    # ever appears in the compute-apps list. It is labelled container-pid because that is
    # the only namespace it resolves in, and it is what `ps -p` and `kill` take there.
    #
    # --quiet-if-absent: a run launched without `harness launch` has no open row, and
    # bookkeeping with nowhere to write must not turn a successful score into a failure.
    # `|| true` for the same reason. But stderr is NOT redirected away: a mistyped flag here
    # would otherwise be invisible forever, which is the same silence this annotation exists
    # to remove. Loud in the log, harmless to the exit code.
    python scripts/exp.py note --name "$NAME" --quiet-if-absent \
      --text "score_matrix STARTED on card $CARD (container-pid $SCORER_PID)" >/dev/null || true
    wait "$SCORER_PID" || SCORING_RC=$?
    python scripts/exp.py note --name "$NAME" --quiet-if-absent \
      --text "score_matrix FINISHED on card $CARD (container-pid $SCORER_PID) rc=$SCORING_RC" >/dev/null || true
  else
    echo "FATAL: no free lane card in 30min -- ckpt_${NAME}.pt unscored, training succeeded but this run produced NO metrics. Re-score: CUDA_VISIBLE_DEVICES=<lane> python eval/score_matrix.py --ckpt ckpt_${NAME}.pt --json runs/score_matrix.jsonl" >&2
    SCORING_RC=1
  fi
  # CLOSE THE ROW HERE, because nothing else will (de-47). no_stale_running FAILs on a
  # row still `running` after 24h, and this chain is the last thing that runs -- a 66h
  # pretrain ends, scores, and exits, and the row it opened stays open until a human
  # remembers. That is what stalled launch_gate: an open row from a finished run reads
  # identical to a job still on the cards, so the gate cannot tell a free lane from a
  # busy one and says NO-GO.
  #
  # UNCONDITIONAL, both paths. The failure path is the one that most needs closing: a
  # run that produced no metrics is exactly the row a human is least likely to come
  # back to, and `exit 1` below would otherwise leave it open forever. status carries
  # which happened, so a closed row is not a claim that the run was good.
  #
  # The result is the val loss read from the log, not a score: the score_matrix record
  # holds the metrics, and this field exists so `exp.py render` shows a number instead
  # of an empty cell. `finding` says the close was automatic -- the human interpretation
  # is still missing and must not read as supplied.
  #
  # Best-effort for the same reason as the notes: --quiet-if-absent for a run launched
  # without `harness launch`, `|| true` so bookkeeping cannot turn a good run bad, and
  # stderr left alone so a mistyped flag is loud in the log.
  VAL=$(grep -oE "val [0-9]+\.[0-9]+" "runs/${NAME}.log" 2>/dev/null | tail -1)
  if [ "$SCORING_RC" -eq 0 ]; then
    DONE_STATUS=ok
    DONE_RESULT="${VAL:-training completed}, score_matrix on card $CARD"
  else
    DONE_STATUS=error
    DONE_RESULT="${VAL:-training completed}, scoring FAILED rc=$SCORING_RC -- no metrics"
  fi
  python scripts/exp.py done --name "$NAME" --status "$DONE_STATUS" \
    --result "$DONE_RESULT" \
    --finding "chained close by run_ddp.sh; finding pending a human reading" \
    --decision "pending" >/dev/null || true
  # fb, 2026-09-02: a scoring failure must make the run's exit code nonzero. The
  # old `|| echo WARN` swallowed it: a 66h run's ~28 milestones would each fail
  # silently while the run exited 0, and a red nobody can act on is no signal.
  # The checkpoint is fine -- re-score with the command above -- but a run that
  # produced no metrics must not read as a success.
  if [ "$SCORING_RC" -ne 0 ]; then
    echo "FATAL: scoring failed for ckpt_${NAME}.pt (rc=$SCORING_RC) -- exiting nonzero" >&2
    exit 1
  fi
fi
exit $rc

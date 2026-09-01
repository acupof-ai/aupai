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
  echo "pod code: $_sha (clean, synced $_when)"
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
  if [ -n "$CARD" ]; then
    CUDA_VISIBLE_DEVICES="$CARD" python eval/score_matrix.py --ckpt "ckpt_${NAME}.pt" --json runs/score_matrix.jsonl \
      || echo "WARN: score_matrix failed for ckpt_${NAME}.pt -- the harness check will flag the missing record" >&2
  else
    echo "WARN: no free lane card in 30min; ckpt_${NAME}.pt unscored. Run: python eval/score_matrix.py --ckpt ckpt_${NAME}.pt --json runs/score_matrix.jsonl" >&2
  fi
fi
exit $rc

#!/bin/bash
# Push code files to the pod from an UP-TO-DATE tree. The pod is not a git repo, so a
# push copies one session's local state into the pod's global state; in a multi-session
# tree that state is stale by default (2026-08-30: a push rolled back 3b's
# datagen/build_corpus.py row-group feature, commit e39146e, and its new launcher died
# on "unrecognized arguments: --rg_mod"). Refuses to push a file with uncommitted
# changes, pulls --rebase --autostash first, and re-runs the drift gate after.
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 1 ] || { echo "usage: $0 <file>..."; exit 2; }
# The pushed files themselves must be committed -- pushing uncommitted code is what the
# drift guard forbids. The rest of the tree may be dirty (another session's work);
# --autostash keeps it out of the pull and restores it after.
for f in "$@"; do
  if [ -n "$(git status --porcelain -- "$f")" ]; then
    echo "refusing: $f has uncommitted changes -- commit or stash it first"
    exit 1
  fi
done
git pull --rebase --autostash
for f in "$@"; do
  ~/bin/podput "$f" "/work/aupai/$f"
done
~/bin/pod "cd /work/aupai && python3 scripts/pod_drift.py --check" < /dev/null

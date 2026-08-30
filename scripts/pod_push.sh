#!/bin/bash
# Push code files to the pod from an UP-TO-DATE tree. The pod is not a git repo, so a
# push copies one session's local state into the pod's global state; in a multi-session
# tree that state is stale by default (2026-08-30: a push rolled back 3b's
# datagen/build_corpus.py row-group feature, commit e39146e, and its new launcher died
# on "unrecognized arguments: --rg_mod"). Refuses to push a file with uncommitted
# changes, then re-runs the drift gate after.
#
# It does NOT pull. Every session works in this same tree and the same .git, so another
# session's commit is already in HEAD the moment it is made -- there is nothing to fetch
# from them, and origin is not how they reach each other. The `git pull --rebase
# --autostash` this script used to run therefore bought nothing and cost something real:
# --autostash stashes and restores the WHOLE dirty tree, which in a six-session tree is
# five other sessions' uncommitted work, every time anyone pushes a file to the pod. That
# is the same hazard as `git checkout` on a file you did not write, run automatically.
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 1 ] || { echo "usage: $0 <file>..."; exit 2; }
# The pushed files themselves must be committed -- pushing uncommitted code is what the
# drift guard forbids. The rest of the tree may be dirty (another session's work) and is
# left exactly as it is.
for f in "$@"; do
  if [ -n "$(git status --porcelain -- "$f")" ]; then
    echo "refusing: $f has uncommitted changes -- commit or stash it first"
    exit 1
  fi
done
for f in "$@"; do
  ~/bin/podput "$f" "/work/aupai/$f"
done
~/bin/pod "cd /work/aupai && python3 scripts/pod_drift.py --check" < /dev/null

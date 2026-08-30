#!/bin/bash
# Push code files to the pod from an UP-TO-DATE tree. The pod is not a git repo, so a
# push copies one session's local state into the pod's global state; in a multi-session
# tree that state is stale by default (2026-08-30: a push rolled back 3b's
# datagen/build_corpus.py row-group feature, commit e39146e, and its new launcher died
# on "unrecognized arguments: --rg_mod"). Refuses a dirty tree, pulls --rebase first,
# then re-runs the pod drift gate so a push that forgets the manifest fails loud.
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 1 ] || { echo "usage: $0 <file>..."; exit 2; }
dirty=$(git status --porcelain --untracked-files=no)
[ -z "$dirty" ] || { echo "refusing to push from a dirty tree; commit or stash first:"; echo "$dirty"; exit 1; }
git pull --rebase
for f in "$@"; do
  ~/bin/podput "$f" "/work/aupai/$f"
done
~/bin/pod "cd /work/aupai && python3 scripts/pod_drift.py --check" < /dev/null

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
#
# Large files (>100KB gzip+base64) bypass podput's argv limit by pushing directly to the
# container's emptyDir host path via `tn push`. The file lands at /work/aupai/<path> in
# the container, same as podput.
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 1 ] || { echo "usage: $0 <file>..."; exit 2; }
# The pushed files themselves must be committed -- pushing uncommitted code is what the
# drift guard forbids. The rest of the tree may be dirty (another session's work) and is
# left exactly as it is.
# They must also be reachable from main: the pod runs main's tree, and a branch-only
# version pushed there would be rolled back by the next pusher's merge (2026-08-31
# worktree ruling). Compare the working blob against main's -- commit to your branch,
# merge, then push.
for f in "$@"; do
  if [ -n "$(git status --porcelain -- "$f")" ]; then
    echo "refusing: $f has uncommitted changes -- commit or stash it first"
    exit 1
  fi
  want=$(git rev-parse "main:$f" 2>/dev/null)
  if [ -z "$want" ]; then
    echo "refusing: $f is not in main -- merge your branch first"
    exit 1
  fi
  if [ "$(git hash-object "$f")" != "$want" ]; then
    echo "refusing: $f differs from main -- merge your branch first (the pod runs main, not your branch)"
    exit 1
  fi
done

# The manifest must describe HEAD. A scoped change without a regenerated manifest
# makes the post-push drift line report the pusher's own file as drifted -- a false
# alarm that trains people to ignore it. Regenerate and refuse; the pusher commits
# the manifest and re-runs, so the commit discipline stays and the pod gate can
# never be satisfied by an unregenerated manifest.
if ! python3 scripts/pod_drift.py --check-head >/dev/null 2>&1; then
  echo "pod_push: manifest was stale, regenerated -- commit it and push again" >&2
  python3 scripts/pod_drift.py --write >/dev/null
  exit 1
fi

# Find the host path of the container's /work emptyDir (cached for this run).
# podput's argv limit (~100KB gzip+base64) cannot carry large files; tn push to the
# emptyDir host path bypasses the container's argv entirely.
EMPTYPATH=""
find_emptydir() {
  [ -n "$EMPTYPATH" ] && return
  EMPTYPATH=$(tn exec "for d in /var/lib/kubelet/pods/*/volumes/kubernetes.io~empty-dir/work; do [ -d \"\$d/aupai\" ] && echo \"\$d\" && break; done" 2>/dev/null | head -1)
  if [ -z "$EMPTYPATH" ]; then
    echo "pod_push: cannot find /work emptyDir host path (is the pod running?)" >&2
    exit 1
  fi
}

for f in "$@"; do
  b64_size=$(gzip -9c "$f" | base64 | tr -d '\n' | wc -c | tr -d ' ')
  if [ "$b64_size" -le 100000 ]; then
    ~/bin/podput "$f" "/work/aupai/$f"
  else
    find_emptydir
    echo "pod_push: $f ($b64_size b64 chars) via emptyDir path" >&2
    tn push "$f" "$EMPTYPATH/aupai/$f"
  fi
done

# Always push the manifest: a file can never land on the pod without the reference
# that describes it. 2026-08-31: a pushed fetch_corpus.py with a stale manifest killed
# a healthy training launch because the pod-side --check compared fresh file vs old hash.
manifest="data/pod_head_manifest.txt"
b64_size=$(gzip -9c "$manifest" | base64 | tr -d '\n' | wc -c | tr -d ' ')
if [ "$b64_size" -le 100000 ]; then
  ~/bin/podput "$manifest" "/work/aupai/$manifest"
else
  find_emptydir
  tn push "$manifest" "$EMPTYPATH/aupai/$manifest"
fi

~/bin/pod "cd /work/aupai && python3 scripts/pod_drift.py --check" < /dev/null

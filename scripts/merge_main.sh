#!/bin/bash
# merge_main.sh <branch>: merge a branch into the integration tree under a lock.
# Two concurrent merges in one worktree race on HEAD and the index (2026-09-04: e1's
# fast-forward landed inside the controller's three-way merge, "cannot lock ref HEAD",
# half-applied index). git's ref lock does not protect the shared working tree; this does.
set -euo pipefail
MAIN=/Users/bytedance/code/aupai
LOCK=$MAIN/.git/merge_main.lock
[ $# -eq 1 ] || { echo "usage: scripts/merge_main.sh <branch>" >&2; exit 2; }
for _ in $(seq 1 120); do
  if mkdir "$LOCK" 2>/dev/null; then
    trap 'rmdir "$LOCK"' EXIT
    if git -C "$MAIN" merge --no-edit "$1"; then exit 0; fi
    if [ -e "$MAIN/.git/MERGE_HEAD" ]; then
      if [ -n "$(git -C "$MAIN" diff --name-only --diff-filter=U)" ]; then
        why="$1 conflicts with main: run 'git merge main' in your worktree, resolve there, commit, retry"
      else
        why="the merge commit was refused (hook or harness check above): fix on your branch, retry"
      fi
      git -C "$MAIN" merge --abort
      echo "merge_main: aborted so the integration tree stays clean; $why" >&2
    fi
    exit 1
  fi
  if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +10 2>/dev/null)" ]; then
    echo "merge_main: lock older than 10 min, removing $LOCK" >&2; rmdir "$LOCK" 2>/dev/null || true
  fi
  sleep 1
done
echo "merge_main: could not take $LOCK in 120 s" >&2; exit 1

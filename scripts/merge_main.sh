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
    if git -C "$MAIN" merge --no-edit "$1"; then
      # THE MERGE IS ALREADY A COMMIT HERE, so a drop cannot be aborted -- `git merge --abort`
      # only works before the commit exists, and `reset --hard` in the shared integration tree
      # would discard whatever else landed. So: detect, RESTORE the dropped paths from the
      # parent that held them, and refuse to return 0 until someone has looked.
      #
      # runs/redaction_handread_v14.tsv was lost by seven merges and restored four times because
      # each restore was itself dropped by the next merge. Restoring in the same step is what
      # breaks that cycle; leaving it to a human is what produced the four attempts.
      # `|| true` on the capture, then judge the EXIT CODE, not just the text. Exit 1 is "drops
      # found", 0 is clean, and anything else means the check could not run -- an old harness
      # without the flag exits 2 with empty stdout, which reads identically to clean. Silence
      # from a tool that failed to start is the shape this whole guard exists to prevent.
      # `set -e` would take the script down on the guard's own nonzero exit before rc is read,
      # so the assignment is split from the capture: `|| rc=$?` keeps the exit code without
      # letting errexit fire.
      rc=0
      drops=$(python3 "$MAIN/scripts/harness.py" --merge-drops 2>/dev/null) || rc=$?
      if [ "$rc" -gt 1 ]; then
        echo "merge_main: WARNING -- the merge-drop guard did not run (harness --merge-drops" >&2
        echo "  exit $rc). The merge stands and was NOT checked for dropped paths." >&2
        exit 0
      fi
      if [ -n "$drops" ]; then
        echo "merge_main: this merge dropped path(s) a parent held, with nobody deleting them:" >&2
        printf '%s\n' "$drops" | while IFS=$'\t' read -r path parent; do
          [ -n "$path" ] || continue
          echo "  $path (held by ${parent:0:8})" >&2
          git -C "$MAIN" checkout "$parent" -- "$path" 2>/dev/null || true
        done
        git -C "$MAIN" status --short >&2
        echo "merge_main: the paths above are RESTORED in the working tree and staged. Commit" >&2
        echo "  them in the integration tree, then re-run. The merge itself stands." >&2
        # The guard will still report these paths until that commit lands: the merge COMMIT's
        # tree is history and a restore cannot change it, only the tree that follows. So this
        # exit is not a retry -- re-running before committing prints the same thing (verified
        # against the recorded merge d9c9614f: restore stages the file, merge_drops still names
        # it). Commit first.
        exit 1
      fi
      exit 0
    fi
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

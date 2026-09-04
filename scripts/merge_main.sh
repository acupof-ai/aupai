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
    # ALREADY AN ANCESTOR: nothing will ship, and git says only "Already up to date."
    # 2026-09-04: `merge_main.sh b0` merged b0 at ccbc0891, already in main, while b0's real
    # work was on b0-ve-rownorms. The merge printed success, exited 0, and what caught it was
    # pod_push's unrelated "differs from main" refusal minutes later. The likeliest cause is
    # the operator naming the wrong branch -- a stale local ref, or a branch that was renamed
    # -- so the message names the tip and asks whether that is the branch meant.
    #
    # READ BEFORE THE MERGE, because afterwards the question cannot be asked: a merge that
    # fast-forwards makes the branch an ancestor, so the same test run after would be true of
    # every successful merge. WARN, not a refusal (6e): merging an ancestor is harmless and a
    # refusal would break a legitimate no-op re-run. Exit code is unchanged.
    if git -C "$MAIN" merge-base --is-ancestor "$1" main 2>/dev/null; then
      _tip=$(git -C "$MAIN" rev-parse --short "$1" 2>/dev/null || echo "?")
      _sub=$(git -C "$MAIN" log -1 --format=%s "$1" 2>/dev/null || echo "?")
      echo "merge_main: WARNING -- $1 ($_tip) is already an ancestor of main, so this merge" >&2
      echo "  ships nothing. Its tip is: $_sub" >&2
      echo "  If that is not the work you meant to merge, you have named the wrong branch:" >&2
      echo "  \`git branch --sort=-committerdate | head\` shows what moved most recently." >&2
    fi
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
      # SHARED-FILE CLAIM RELEASE ON MERGE (T0, 2026-09-04). The claim that let the branch's
      # shared-file commit pass lives in the branch worktree, not main (the non-merge commit's
      # hook reads the tree it ran in). Now that the edit is merged the file is handed back:
      # release the merging session's claims in the branch's worktree. A session that acquired
      # a claim and merged a DIFFERENT branch still needs its claim, so scope by owner ($USER),
      # and the branch may have no worktree (merged after --delete) -- then its claims are moot
      # and the 6h TTL bounds anything left.
      # THE OFFSET IS 19, NOT 16. `branch refs/heads/` is 18 characters, so substr($0,16) starts
      # three too early and yields `ds/de` for branch `de` -- it never equals the bare name, `_wt`
      # is always empty, and the release is silently skipped for EVERY branch. Measured 2026-09-05
      # on this tree: substr(16)="ds/de", substr(19)="de", and with 19 the awk matches
      # /Users/bytedance/code/aupai-de. 58's AGENTS.md claim survived a successful merge and had to
      # be released by hand. Reported as a branch-naming problem (`ds/<name>` prefixes); it is not
      # -- `git worktree list --porcelain` prints the full ref and the coincidence is that the
      # three characters at 16-18 are `ds/`, the tail of `refs/heads/`.
      #
      # The consequence is not cosmetic even with the 6h TTL: $USER is `bytedance` for every
      # session on this box, so one leaked claim blocks every other session's shared-file commits
      # until it expires.
      _wt=$(git -C "$MAIN" worktree list --porcelain 2>/dev/null \
        | awk -v b="$1" '/^worktree /{w=substr($0,10)} /branch refs\/heads\// && substr($0,19)==b && w!="" {print w; exit}')
      if [ -n "$_wt" ] && [ -f "$_wt/scripts/file_claim.py" ]; then
        # No --owner: file_claim's own default is LAUNCH_OWNER, else the worktree name, and both
        # it and the claim dir derive from the script's own path -- so invoking $_wt's copy by
        # absolute path already targets $_wt's claims as $_wt's owner. Passing $USER scoped
        # nothing: it is `bytedance` for every session here, so one session's merge handed back
        # every other session's claims.
        _rel=$(python3 "$_wt/scripts/file_claim.py" release-all 2>/dev/null \
          || echo "release-all failed")
        echo "merge_main: shared-file claims on $1: $_rel" >&2
      else
        # SAY SO WHEN NOTHING WAS RELEASED. The absence of the line above was the only signal,
        # and it read as "merged after --delete" -- which is how the substr(16) bug survived:
        # the release had never fired for any branch and silence looked like the normal case.
        if [ -z "$_wt" ]; then
          echo "merge_main: no worktree matched branch $1 -- shared-file claims NOT released;" >&2
          echo "  release by hand in that tree, or the 6h TTL clears them." >&2
        else
          echo "merge_main: $_wt has no scripts/file_claim.py -- claims NOT released" >&2
        fi
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

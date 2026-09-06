#!/bin/bash
# merge_main.sh <branch>: merge a branch into the integration tree under a lock.
# Two concurrent merges in one worktree race on HEAD and the index (2026-09-04: e1's
# fast-forward landed inside the controller's three-way merge, "cannot lock ref HEAD",
# half-applied index). git's ref lock does not protect the shared working tree; this does.
set -euo pipefail
MAIN=/Users/bytedance/code/aupai
# Overridable so --selftest can drive the real predicate against fixture locks instead of a
# reimplementation of it. Nothing else sets these.
LOCK=${MERGE_LOCK_DIR:-$MAIN/.git/merge_main.lock}
HOLDER=$LOCK/holder
[ $# -eq 1 ] || { echo "usage: scripts/merge_main.sh <branch>|--hold|--release|--selftest" >&2; exit 2; }

# REFUSE A DEADLINE SHORTER THAN A HOOK RUN. A merge commit runs the full pre-commit hook -- ~50-60 s
# on a normal commit and longer when the selftest set is large -- and a parent that kills the process
# group before it finishes leaves the WORST state of any failure mode here: main unmoved, the shared
# index staged with the merge's content, and .hookstaged_* copies in the working tree, because SIGKILL
# does not run the hook's `finally`. The next session's merge then aborts on "local changes would be
# overwritten" and every merge into main is blocked until someone cleans up by hand.
#
# MEASURED 2026-09-05: `timeout 900 bash scripts/merge_main.sh de` run through a wrapper that caps at
# 120 s. The log ends at `hook: ruff 0.05s`, the stage before selftests, with no traceback -- the hook
# was killed, not broken. b0's next merge aborted, 4c's pod_push was blocked, and E1's launch waited.
# Two sessions then diagnosed it as a defect in the hook itself, which cost more than the merge.
#
# READ FROM THE PARENT'S CMDLINE, because that is where the deadline actually is. A `timeout N` parent
# shows as `timeout N <cmd>` in ps, verified on this machine in both directions (under a timeout the
# direct parent is `timeout 30 /tmp/de_detect.sh`; bare, it is the invoking shell). The signal
# disposition is NOT readable -- timeout sends SIGTERM when the deadline arrives and inherits nothing
# beforehand -- so a trap cannot see this coming and only the process tree can.
# NOT FOR THE MODES THAT NEITHER MERGE NOR COMMIT. --selftest, --hold and --release take no lock
# for a merge, run no hook, and finish in under a second, so a deadline cannot strand the shared
# index in the way described above. The guard fired on `timeout 120 merge_main.sh --selftest`
# (measured 2026-09-05, minutes after it landed) and refused to run the test of itself -- a guard
# whose only failure was against its own verification.
if [ "${MERGE_MAIN_ALLOW_TIMEOUT:-0}" != "1" ] && [ "$1" != "--selftest" ] && [ "$1" != "--hold" ] \
   && [ "$1" != "--release" ]; then
  _parent_cmd=$(ps -o command= -p "$PPID" 2>/dev/null || true)
  # ARGV[0], NOT ANYWHERE IN THE LINE (4c, 2026-09-06). The test was `*timeout\ *`, which matches
  # the substring wherever it falls -- and ps prints the parent's WHOLE cmdline, heredoc body
  # included. A merge invoked from a shell command that merely CONTAINED the words "timeout 20"
  # in quoted text was refused for a deadline that existed only inside a string. Measured both
  # directions: under a real `timeout 30` the first field is `timeout`; with the words in a
  # heredoc it is `/bin/zsh`.
  #
  # A path is allowed before the name (/usr/bin/timeout, gtimeout) but the deadline must be the
  # parent's own argv[0], not text it happens to carry.
  _parent_argv0=${_parent_cmd%% *}
  case "${_parent_argv0##*/}" in
    timeout|gtimeout)
      # The first bare number after `timeout`, skipping its flags (-k, -s TERM, --preserve-status).
      #
      # THE SUFFIX IS SCALED, NOT STRIPPED. My first version did `gsub(/[smhd]/,"")`, which reads
      # `20m` as 20 and `1h` as 1 -- so it would have REFUSED a 20-minute deadline that is generous
      # and a 1-hour one that is ample, while `2m` refused for the right reason by coincidence (120).
      # Measured on all six forms before this line was trusted: 120, 2m, 20m, 900, `-k 5 -s TERM 90`,
      # 1h. A guard that refuses a correct invocation is worse than none: people pass the override and
      # stop reading the reason.
      _deadline=$(printf '%s\n' "$_parent_cmd" | tr ' ' '\n' | awk '
        /^timeout$/ {seen=1; next}
        seen && /^-/ {skip=1; next}
        seen && skip {skip=0; next}
        seen && /^[0-9]+(\.[0-9]+)?[smhd]?$/ {
          n = $0 + 0
          if ($0 ~ /m$/) n *= 60
          else if ($0 ~ /h$/) n *= 3600
          else if ($0 ~ /d$/) n *= 86400
          print int(n); exit
        }')
      if [ -n "${_deadline:-}" ] && [ "$_deadline" -lt 600 ]; then
        echo "REFUSING: this merge is running under \`timeout ${_deadline}\`, and a merge commit needs" >&2
        echo "  longer than that -- the pre-commit hook alone takes 50-60 s and more when the" >&2
        echo "  selftest set is large. A kill mid-hook leaves main unmoved, the SHARED index staged" >&2
        echo "  with this merge's content, and .hookstaged_* files in the working tree (SIGKILL does" >&2
        echo "  not run the hook's cleanup), which blocks every other session's merge until someone" >&2
        echo "  clears it by hand. Measured 2026-09-05: a 120 s cap did exactly this and stalled" >&2
        echo "  three sessions and one launch." >&2
        echo "  Run it with no timeout, or in the background and poll:" >&2
        echo "    nohup bash scripts/merge_main.sh $1 > /tmp/merge_$1.log 2>&1 </dev/null &" >&2
        echo "  MERGE_MAIN_ALLOW_TIMEOUT=1 overrides, for a deadline you know exceeds a hook run." >&2
        exit 2
      fi
      ;;
  esac
fi

# --hold, --release and --selftest are handled AFTER the two functions below, which they call.

# TAKE THE LOCK, then write the holder file. There IS a window, two statements wide, in which the
# lock exists with no holder -- and a waiter arriving inside it must not read "no holder" as dead,
# or it removes a lock a live merge is holding, which is worse than the age rule it replaces.
#
# I tried to close the window with a staged directory moved into place, and it does not work: `mv a
# b` where b is an existing directory moves a INTO b on both Linux and macOS, so it silently
# creates $LOCK/staging instead of failing, and the mutual exclusion is gone. `mv -T` is GNU-only.
# So the window stays and the WAITER carries the grace: a lock with no holder file is dead only
# after _NO_HOLDER_GRACE consecutive reads a second apart. The window is two statements; the grace
# is three seconds; a pre-change lock (there is no holder file in any of them) costs three extra
# seconds once.
_NO_HOLDER_GRACE=3
_take_lock() {  # $1=purpose $2=deliberate $3=pid to record (default $$)
  mkdir "$LOCK" 2>/dev/null || return 1
  { echo "pid=${3:-$$}"; echo "purpose=$1"; echo "deliberate=$2";
    echo "since=$(date -u '+%Y-%m-%d %H:%M:%SZ')"; } > "$HOLDER"
  return 0
}

# any lock older than 10 minutes, which is right for the case it was written for -- a merge killed
# before its EXIT trap ran leaves the directory behind -- and wrong for the case that actually
# happened: 4c held the lock as a quiet window, and two of e1's merges (e92b9bbd, 7c20e080) landed
# on main inside it because 10 minutes had passed. The holder was never told; the waiter reported
# it as clearing a stale lock. Age is a proxy for deadness and it fails in exactly the case where
# someone is using the lock as intended. Same shape as guarding on [ -d /proc/<pid> ] to mean
# is-this-running: the criterion does not express the property (AGENTS.md, two incidents).
#
# ZOMBIES ARE NOT ALIVE. `kill -0` succeeds on a process that has exited and not been reaped, so
# it answers "has this pid been reaped", the same defect in a new place. The state is read too, and
# Z means dead. Both are needed: ps alone cannot distinguish a pid that never existed from one it
# cannot see.
#
# A LOCK WITH NO holder FILE IS DEAD BY CONSTRUCTION. It predates this change, or a crash happened
# between mkdir and the write; either way nobody can be asked, and refusing forever would need a
# hand-run rmdir every time. That window is two statements wide and the file is written before the
# merge starts.
_lock_is_dead() {
  if [ ! -f "$HOLDER" ]; then
    # THE GRACE, not an immediate verdict: see _take_lock. A caller that has not passed
    # $_NO_HOLDER_GRACE consecutive holderless reads is told "not yet", so the loop keeps waiting.
    _no_holder_seen=$((${_no_holder_seen:-0} + 1))
    if [ "$_no_holder_seen" -lt "$_NO_HOLDER_GRACE" ]; then _dead_why=""; return 1; fi
    _dead_why="no holder file after ${_no_holder_seen}s (a lock predating this change, or a crash between mkdir and the write)"
    return 0
  fi
  _no_holder_seen=0
  local p st
  p=$(sed -n 's/^pid=//p' "$HOLDER" 2>/dev/null || true)
  case "$p" in ''|*[!0-9]*) _dead_why="holder file names no usable pid"; return 0;; esac
  st=$(ps -o stat= -p "$p" 2>/dev/null | tr -d ' ' || true)
  if [ -z "$st" ]; then _dead_why="holder pid $p is gone"; return 0; fi
  case "$st" in Z*) _dead_why="holder pid $p is a zombie ($st) -- exited, not reaped"; return 0;; esac
  _dead_why=""
  return 1
}

# --hold / --release: take the lock as a quiet window and keep it across sessions. This is what the
# age rule made impossible -- a deliberate hold and a lock left by a killed merge are identical to
# `find -mmin +10`, so on 2026-09-05 4c's window silently expired and two of e1's merges landed
# inside it. A hold is now a declared state, not an accident of timing.
#
# A HOLD RECORDS THE CALLER'S PARENT, NOT $$. This script exits immediately after taking the hold,
# so $$ would name a process that is already gone and the very next waiter would read the hold as
# dead -- an age rule with a zero-second timeout. $PPID is the shell or session that ran the hold
# and is the thing whose liveness actually means "someone is still coordinating". If the holder does
# not care about liveness at all (a hold that must outlive its terminal), MERGE_HOLD_PID=0 records
# an unparseable pid, which reads as dead: the deliberate=yes refusal is then what protects it, and
# only an explicit --release clears it.
if [ "$1" = "--hold" ]; then
  if _take_lock "${MERGE_HOLD_PURPOSE:-a deliberate quiet window}" yes "${MERGE_HOLD_PID:-$PPID}"; then
    echo "merge_main: holding $LOCK for pid ${MERGE_HOLD_PID:-$PPID} -- release it with" >&2
    echo "  'scripts/merge_main.sh --release'." >&2
    echo "  A waiter will NOT time this out. It waits while that pid lives, and even once the pid" >&2
    echo "  is gone it refuses to remove a deliberate hold, printing this file so it can ask." >&2
    exit 0
  fi
  echo "merge_main: already locked:" >&2; sed 's/^/  /' "$HOLDER" 2>/dev/null || true; exit 1
fi
if [ "$1" = "--release" ]; then
  if [ ! -d "$LOCK" ]; then echo "merge_main: no lock to release" >&2; exit 0; fi
  _p=$(sed -n 's/^pid=//p' "$HOLDER" 2>/dev/null || true)
  # A LIVE HOLDER THAT IS NOT YOURS is someone else's lock, and removing it is the age rule's defect
  # typed by hand instead of fired by a timer. Yours means the pid recorded is this process or its
  # parent -- the same identity --hold writes.
  _no_holder_seen=$_NO_HOLDER_GRACE
  if [ -n "$_p" ] && [ "$_p" != "$$" ] && [ "$_p" != "$PPID" ] && ! _lock_is_dead; then
    echo "merge_main: the lock is held by a LIVE process that is not you -- refusing to release" >&2
    sed 's/^/  /' "$HOLDER" >&2
    echo "  Ask that session. If it is wedged, kill pid $_p and release again." >&2
    exit 1
  fi
  rm -f "$HOLDER"; rmdir "$LOCK"; echo "merge_main: released $LOCK" >&2; exit 0
fi

# SECOND READER BEFORE MERGE (4c's ruling 2026-09-05, user-approved). A branch whose
# not-yet-on-main commits touch train.py or model.py is refused unless runs/review.jsonl
# carries a row whose `artifact` names one of those shas and whose reviewer is not the author.
#
# WHY A REFUSAL RATHER THAN THE PROSE RULE IT REPLACES: on 2026-09-05 every defect that was
# caught was caught by a second reader and none by the author -- b0's two optimizer misroutings,
# the SwiGLU mutant that reached main and the pod, my own witness transposed twice and its
# tolerance set below the noise floor of the device it runs on. The rule already existed as
# words; what it lacked was a moment where it fires.
#
# TRAIN.PY AND MODEL.PY ONLY. They are where a defect is silent: an arm trains, the loss moves,
# and nothing says the optimizer grouped a 56M-parameter FFN as depth-attention queries. A
# broader net would refuse routine work and be worked around, which is worse than no gate.
_review_gate() {  # $1 = branch. Echoes the refusal reason; returns 1 to refuse.
  local shas subject author sha found row
  # ONLY the commits this merge would ship, and only non-merge ones: a merge commit's diff
  # against its first parent shows the other side's files, which the other side already
  # answered for. --no-merges also exempts reverts of merges; a plain revert is exempted by
  # its subject below.
  shas=$(git -C "$MAIN" rev-list --no-merges "main..$1" 2>/dev/null) || return 0
  [ -n "$shas" ] || return 0
  found=""
  for sha in $shas; do
    # A REVERT IS EXEMPT (4c): it restores a state that was already reviewed, and refusing one
    # would make the fastest correction the hardest commit to land -- exactly backwards when a
    # bad commit is on main and on the pod.
    subject=$(git -C "$MAIN" log -1 --format=%s "$sha")
    case "$subject" in Revert*|revert*) continue ;; esac
    if git -C "$MAIN" show --stat --format= --name-only "$sha" \
         | grep -qxE 'train\.py|model\.py'; then
      found="$found $sha"
    fi
  done
  [ -n "$found" ] || return 0

  for sha in $found; do
    author=$(git -C "$MAIN" log -1 --format=%an "$sha")
    # The row must name the sha AND be written by someone else. `git log --format=%an` is the
    # git author, while review.jsonl's reviewer is a roster name -- these are different
    # namespaces, so the comparison that can actually be made is reviewer-vs-the-branch: a row
    # whose reviewer is the branch's own name is a self-review. Checked in python because
    # review.jsonl rows are JSON and a grep for the sha would match it inside any field.
    row=$(python3 - "$sha" "$1" <<'PY'
import json, sys
sha, branch = sys.argv[1], sys.argv[2]
short = sha[:8]
try:
    rows = [json.loads(l) for l in open("runs/review.jsonl", encoding="utf-8") if l.strip()]
except (OSError, json.JSONDecodeError):
    sys.exit(0)  # unreadable ledger: say nothing, the caller refuses for want of a row
for r in rows:
    if not isinstance(r, dict):
        continue
    art = str(r.get("artifact", "")) + " " + str(r.get("item", ""))
    if short in art or sha in art:
        rev = str(r.get("reviewer", "")).strip()
        # SELF-REVIEW IS NOT A REVIEW. The reviewer field is free text ("b0 (self-reported)"),
        # so the test is whether the branch's own name appears in it, not equality.
        if rev and branch.lower() not in rev.lower():
            print(rev)
            break
PY
)
    if [ -z "$row" ]; then
      echo "merge_main: REFUSING -- $sha touches train.py or model.py and no second reader" >&2
      echo "  has signed it. $(git -C "$MAIN" log -1 --format='%h %s' "$sha")" >&2
      echo "  Author: $author. Needed: a row in runs/review.jsonl whose \`artifact\` names" >&2
      echo "  $sha (or ${sha:0:8}) with a \`reviewer\` that is not '$1'." >&2
      echo "  Every defect caught on 2026-09-05 was caught by a second reader, none by the" >&2
      echo "  author -- that is what this refusal is for." >&2
      echo "  Controller override: AUPAI_CONTROLLER=1 (logged to runs/friction.jsonl)." >&2
      return 1
    fi
  done
  return 0
}

# WHY `git merge` RETURNED NONZERO, as a function so --selftest drives it against both worlds.
# Two different states return nonzero and reporting one as the other is 4c's 2026-09-06 defect:
#   conflict        unmerged index entries; conflict markers are in the files.
#   refused-commit  the merge RESOLVED and its commit was refused by the pre-commit hook: zero
#                   unmerged paths, main's versions of every merged file staged.
#   none            no merge is in flight at all.
# MERGE_HEAD is set in the first two alike, so it cannot be the discriminator; `git ls-files -u`
# can. Prints one of the three words.
_merge_failure_kind() {
  if [ -n "$(git ls-files -u)" ]; then
    echo conflict
  elif [ -f "$(git rev-parse --git-dir)/MERGE_HEAD" ]; then
    echo refused-commit
  else
    echo none
  fi
}

# THE STAGED-INDEX CARRY (tilerl-31), as two functions so --selftest drives the REAL code in a
# scratch repo rather than a reimplementation of it. A reimplemented predicate shares the
# original's assumptions and its agreement is not evidence (gate_failure_shapes §231).
#
# _carry_stage <branch>   prints the wip ref it created, or nothing when no carry was needed.
#                         Returns 1 and explains when it refuses.
# _carry_restore <ref>    writes the carried paths over the merge result and drops the ref.
_carry_stage() {
  _cs_staged=$(git diff --cached --name-only)
  [ -n "$_cs_staged" ] || return 0
  # A PENDING MERGE IS NOT A STAGED INDEX TO CARRY. When a merge resolves and its commit is
  # refused by the hook, MERGE_HEAD stays set and main's versions of every merged file are
  # staged -- so this function sees 10-30 "staged paths" that are not the operator's work at
  # all, and carrying them writes main's own content to refs/wip/ and then restores it over the
  # merge. 4c hit this twice in aupai-fb on 2026-09-06. Refuse and name the recovery: the
  # pending merge has to be concluded or aborted before there is anything here to carry.
  if [ -f "$(git rev-parse --git-dir)/MERGE_HEAD" ]; then
    echo "merge_main: a merge is already in progress in this worktree -- refusing to carry" >&2
    echo "  its staged paths. $(printf '%s\n' "$_cs_staged" | wc -l | tr -d ' ') path(s) are staged, and after a refused merge" >&2
    echo "  commit those are MAIN's versions, not your work. Conclude or abandon it first:" >&2
    echo "    git commit --no-edit     # if the merge was resolved and only its commit failed" >&2
    echo "    git merge --abort        # to throw the merge away" >&2
    return 1
  fi
  # THE TRIGGER IS "WILL THIS MERGE TOUCH A STAGED PATH", and fast-forward vs true merge only
  # decides WHICH paths that is. My first version returned early on any fast-forward, on the
  # measurement that a fast-forward tolerates a staged index -- and that measurement was taken
  # with the staged path held at one the incoming side never touches, which is the single case
  # where it is true. MEASURED AGAINST ITSELF within the hour (de, 2026-09-06): this function
  # printed "the merge is a fast-forward -- no carry needed" and the fast-forward then aborted
  # with "Your local changes to the following files would be overwritten by merge: runs/tasks.jsonl,
  # scripts/harness.py" -- both staged, both changed by main. A fast-forward checks out the files
  # the range changed, so it refuses a staged path INSIDE that set and ignores one outside it.
  #
  # A true merge runs `ort`, which needs a clean index TREE-WIDE and refuses ANY staged path
  # regardless of what the merge touches (measured with the staged path held constant: fast-
  # forwardable rc=0, diverged rc=2). So:
  #   true merge    -> carry, always
  #   fast-forward  -> carry only when a staged path is in the fast-forward's changed set
  # W3 asserted the old rule and passed because its staged path was outside the set; it now
  # asserts the same world AND the in-set case, which is the one that bit.
  if git merge-base --is-ancestor HEAD main; then
    # Exact-line intersection without a process substitution: `comm` needs both sides sorted,
    # and grep -Fxf on a temp file is one fewer moving part than <(...) inside `set -u`.
    _cs_ffl=$(mktemp)
    git diff --name-only HEAD main > "$_cs_ffl"
    _cs_overlap=$(printf '%s\n' "$_cs_staged" | grep -Fxf "$_cs_ffl" || true)
    rm -f "$_cs_ffl"
    if [ -z "$_cs_overlap" ]; then
      echo "merge_main: $(printf '%s\n' "$_cs_staged" | wc -l | tr -d ' ') staged path(s), fast-forward touches none of them -- no carry needed" >&2
      return 0
    fi
    echo "merge_main: fast-forward, but it changes $(printf '%s\n' "$_cs_overlap" | wc -l | tr -d ' ') staged path(s) -- carrying" >&2
  fi
  _cs_ref="refs/wip/$1"
  # REFUSE RATHER THAN OVERWRITE. An existing ref means a previous carry never restored, so
  # clobbering it discards whatever that run staged, silently.
  if git rev-parse -q --verify "$_cs_ref" >/dev/null; then
    echo "REFUSING: $_cs_ref already exists -- a previous carry never restored." >&2
    echo "  Inspect it, then remove it once its content is safe:" >&2
    echo "    git show $_cs_ref" >&2
    echo "    git update-ref -d $_cs_ref" >&2
    return 1
  fi
  # A COMMIT ON A PRIVATE REF, NEVER A STASH: refs/stash is ONE stack shared by every worktree,
  # so two sessions each pop the other's entry (AGENTS.md; e1 and b0, 2026-09-02). refs/wip/<branch>
  # is per-branch and shows in `git log --all`, so an interrupted carry is recoverable by name
  # rather than by position in a stack somebody else is also pushing to.
  _cs_tree=$(git write-tree) || { echo "REFUSING: git write-tree failed" >&2; return 1; }
  _cs_wip=$(git commit-tree "$_cs_tree" -p HEAD -m "wip: staged index carried through merge of main") \
    || { echo "REFUSING: git commit-tree failed" >&2; return 1; }
  git update-ref "$_cs_ref" "$_cs_wip" || { echo "REFUSING: could not write $_cs_ref" >&2; return 1; }
  # RESET THE INDEX **AND** THE WORKING TREE, for the carried paths only. `git reset HEAD -- .`
  # alone unstages and leaves the file MODIFIED, and git refuses a merge on a dirty working tree
  # exactly as it refuses one on a dirty index -- so the abort simply moved from "staged" to
  # "local changes". Measured on the W3b world (de, 2026-09-06): after the reset, staged was empty,
  # unstaged held f.txt, and the fast-forward still aborted with "Your local changes to the
  # following files would be overwritten by merge: f.txt".
  #
  # TWO KINDS OF STAGED PATH, and W6 found the second. A path that exists in HEAD is restored with
  # `checkout HEAD -- <path>`. A NEWLY ADDED path does not exist in HEAD at all, so that checkout
  # errors ("pathspec did not match") and the carry reports no-carry -- which is what W6 saw when
  # it staged a new other.txt. A new path is instead unstaged and removed from the working tree;
  # both are safe because the content is already committed on $_cs_ref, and _carry_restore writes
  # it back by name.
  #
  # Scoped to the carried paths, never `-- .`: this must not touch a path the session left dirty
  # on purpose and did not stage. Those are the `_dirty` gate's business, above, which refuses
  # before we get here.
  printf '%s\n' "$_cs_staged" | while IFS= read -r _cs_p; do
    [ -n "$_cs_p" ] || continue
    if git cat-file -e "HEAD:$_cs_p" 2>/dev/null; then
      git checkout HEAD -- "$_cs_p" || exit 1
    else
      git rm -q --cached -- "$_cs_p" || exit 1
      rm -f -- "$_cs_p"
    fi
  done || { echo "REFUSING: could not clear the carried paths (they are safe at $_cs_ref)" >&2; return 1; }
  echo "merge_main: carried $(printf '%s\n' "$_cs_staged" | wc -l | tr -d ' ') staged path(s) to $_cs_ref (${_cs_wip:0:8}); restoring after the merge" >&2
  # FIRST LINE IS THE REF, THE REST ARE THE PATHS. _carry_restore needs the names: restoring by
  # `.` reverts the merge (see the comment there), so the list has to survive the round trip.
  printf '%s\n' "$_cs_ref"
  printf '%s\n' "$_cs_staged"
  return 0
}

_carry_restore() {
  # RESTORE ONLY THE CARRIED PATHS, NAMED. `git checkout <ref> -- .` was the first version and it
  # is WRONG in a way nothing else here would have caught: the ref is a whole-tree commit, so `.`
  # writes EVERY path back to its pre-merge content and silently reverts the merge. Measured
  # (/tmp/de_w1_diag.sh, 2026-09-06): after the merge shared.txt read `other` (main's change) and
  # mine.txt read `staged`; after `checkout <ref> -- .` shared.txt was back to `base`. The merge
  # succeeded, the carry "worked", and main's change was gone from the working tree with nothing
  # raising. W1 asserts the CONSEQUENCE -- both changes present -- which is the only reason this
  # was found instead of shipped.
  #
  # $2 is the newline-separated path list _carry_stage recorded. Restoring by name touches nothing
  # the merge wrote to any other path.
  [ -n "${2:-}" ] || { echo "REFUSING: _carry_restore needs the carried path list" >&2; return 1; }
  # A UNION-MERGE LEDGER IS UNIONED BACK, NEVER OVERWRITTEN. This is the third defect in this
  # function and the most expensive: an overwrite restore is correct for a source file, where my
  # version supersedes the merged one, and is a DELETION for a file git merges by union, where the
  # merged version holds rows other sessions appended. Measured (de, 2026-09-06): the carry
  # restored runs/tasks.jsonl over the merge and dropped e1-48 and e1-49, two rows that arrived in
  # main during that merge. The hook's ledger-append-only gate caught it -- "runs/tasks.jsonl loses
  # 2 record(s)" -- so the loss never reached a commit, but nothing in the carry itself would have
  # noticed, and the carry exists precisely to be used on a tree with staged ledger rows.
  #
  # Scope read from .gitattributes rather than a list here: a second list of union files would drift
  # from the one git actually honours, and this whole function is a study in a rule keyed on the
  # wrong property.
  _cr_ok=1
  # A MARKER FILE, not a variable: the loop below runs in a pipe subshell, so an assignment there
  # is invisible to this shell. This is the same class as the defect the loop is fixing, and it
  # would have made a conflict silent.
  _cr_flag=$(mktemp)
  printf '%s\n' "$2" | while IFS= read -r _cr_p; do
    [ -n "$_cr_p" ] || continue
    if git check-attr merge -- "$_cr_p" 2>/dev/null | grep -q ': merge: union$'; then
      # Union: keep every line from the merged file AND from the carried copy, first-seen order.
      # Duplicate LINES collapse; nothing else does. A row is an event here (230 of 283 task ids
      # appear more than once -- an `open` row and a later `dropped` row), so folding by id would
      # delete history, which is the same defect one level down: my first recovery attempt folded
      # 563 rows to 286 before its own output caught it.
      _cr_tmp=$(mktemp)
      git show "$1:$_cr_p" > "$_cr_tmp" 2>/dev/null || : > "$_cr_tmp"
      _cr_merged=$(mktemp)
      cat "$_cr_p" > "$_cr_merged" 2>/dev/null || : > "$_cr_merged"
      awk '!seen[$0]++' "$_cr_merged" "$_cr_tmp" > "$_cr_p" || exit 1
      rm -f "$_cr_tmp" "$_cr_merged"
      echo "merge_main:   $_cr_p unioned (merge=union), not overwritten" >&2
    else
      # SOURCE: A THREE-WAY MERGE, NOT A RESTORE. This is the fourth defect in this function and
      # the largest. An overwrite restore assumes my carried version supersedes the merged one,
      # which is true only when the merge did not touch that path -- and the carry now fires
      # precisely BECAUSE the merge touches it. Measured (de, 2026-09-06): the carry restored
      # scripts/harness.py from the carry ref and reverted 517 lines of other sessions' work,
      # including e1's 9a382298; the visible symptom was owner_queue_depth reading FAIL in this
      # tree while main returns WARN, which I mis-reported to the controller as e1 having no
      # assigned task. Nothing raised -- the file simply went backwards.
      #
      # `git merge-file` on the three versions is the same operation git would have done had the
      # path not been staged: BASE is the path as of the carry's parent (pre-merge HEAD), OURS is
      # the merged file now in the working tree, THEIRS is my carried edit. A conflict is left in
      # the file with markers AND reported, because a conflict here is a real disagreement between
      # my edit and the merge, and silently picking a side is what produced this defect.
      _cr_base=$(mktemp); _cr_ours=$(mktemp); _cr_theirs=$(mktemp)
      git show "$1^:$_cr_p" > "$_cr_base" 2>/dev/null || : > "$_cr_base"
      cat "$_cr_p" > "$_cr_ours" 2>/dev/null || : > "$_cr_ours"
      git show "$1:$_cr_p" > "$_cr_theirs" 2>/dev/null || : > "$_cr_theirs"
      if git merge-file -q -L merged -L pre-merge -L "carried" \
           "$_cr_ours" "$_cr_base" "$_cr_theirs"; then
        cat "$_cr_ours" > "$_cr_p"
        echo "merge_main:   $_cr_p three-way merged (my edit onto the merge result)" >&2
      else
        cat "$_cr_ours" > "$_cr_p"
        echo "merge_main:   CONFLICT in $_cr_p -- conflict markers are IN THE FILE." >&2
        echo "merge_main:   Your carried edit and the merge disagree on the same lines. Resolve" >&2
        echo "merge_main:   by hand; the carried version is also at $1:$_cr_p" >&2
        printf '%s\n' "$_cr_p" >> "$_cr_flag"
      fi
      rm -f "$_cr_base" "$_cr_ours" "$_cr_theirs"
    fi
  done || _cr_ok=0
  if [ "$_cr_ok" -ne 1 ]; then
    echo "REFUSING: the merge landed but the staged index could not be restored." >&2
    echo "  It is intact at $1. Restore the carried paths by name, then drop it:" >&2
    printf '%s\n' "$2" | sed "s|^|    git checkout $1 -- |" >&2
    echo "    git update-ref -d $1" >&2
    rm -f "$_cr_flag"
    return 1
  fi
  # THE REF IS KEPT ON A CONFLICT. A conflicted file holds markers, so the carried content is only
  # recoverable from the ref -- dropping it here would leave the operator with a broken file and no
  # clean copy of what they staged.
  if [ -s "$_cr_flag" ]; then
    echo "REFUSING to drop $1: $(wc -l < "$_cr_flag" | tr -d ' ') path(s) conflicted and hold" >&2
    echo "  conflict markers. Resolve them, then drop the ref yourself:" >&2
    sed 's/^/    /' "$_cr_flag" >&2
    echo "    git update-ref -d $1" >&2
    rm -f "$_cr_flag"
    return 1
  fi
  rm -f "$_cr_flag"
  git update-ref -d "$1"
  echo "merge_main: restored $(printf '%s\n' "$2" | wc -l | tr -d ' ') carried path(s) on top of the merge; $1 dropped" >&2
  return 0
}

# --selftest drives the REAL _lock_is_dead against fixture locks, in BOTH directions -- a
# predicate that only ever says "dead" passes every positive case: the live-pid and deliberate-hold
# rows are the ones that would have prevented 2026-09-05.
if [ "$1" = "--selftest" ]; then
  _fails=0
  _t=$(mktemp -d)
  _case() {  # $1=name $2=want dead|alive
    if _lock_is_dead; then _got=dead; else _got=alive; fi
    if [ "$_got" != "$2" ]; then
      echo "  FAIL $1: want $2, got $_got ($_dead_why)" >&2; _fails=$((_fails + 1))
    else
      echo "  ok   $1 -> $_got${_dead_why:+ ($_dead_why)}"
    fi
  }
  LOCK=$_t/l1; HOLDER=$LOCK/holder; mkdir -p "$LOCK"
  _no_holder_seen=$_NO_HOLDER_GRACE   # as if the grace had already elapsed
  _case "holderless lock past its grace" dead
  _no_holder_seen=0
  _case "holderless lock inside its grace" alive
  LOCK=$_t/l2; HOLDER=$LOCK/holder; mkdir -p "$LOCK"; _no_holder_seen=0
  printf 'pid=%s\npurpose=merge x\ndeliberate=no\n' "$$" > "$HOLDER"
  _case "live holder, this very process" alive
  # A DELIBERATE HOLD BY A LIVE PID IS ALIVE, which is the case that broke: the old rule removed it
  # at 10 minutes with the holder never told.
  LOCK=$_t/l3; HOLDER=$LOCK/holder; mkdir -p "$LOCK"; _no_holder_seen=0
  printf 'pid=%s\npurpose=quiet window\ndeliberate=yes\n' "$$" > "$HOLDER"
  _case "live deliberate hold (the 2026-09-05 case)" alive
  # A pid that cannot exist. 2^22 is above every default pid_max on Linux and macOS.
  LOCK=$_t/l4; HOLDER=$LOCK/holder; mkdir -p "$LOCK"; _no_holder_seen=0
  printf 'pid=4194304\npurpose=killed merge\ndeliberate=no\n' > "$HOLDER"
  _case "holder pid gone" dead
  LOCK=$_t/l5; HOLDER=$LOCK/holder; mkdir -p "$LOCK"; _no_holder_seen=0
  printf 'pid=notanumber\ndeliberate=no\n' > "$HOLDER"
  _case "unparseable pid" dead
  # A REAL ZOMBIE, because `kill -0` succeeds on one and that is the entire reason the predicate
  # reads ps stat as well. Made with a python parent that forks, lets the child exit, and does NOT
  # wait -- a bash `( exit ) &` does not work here: bash reaps its own children asynchronously, so
  # the first version of this case SKIPped every run with stat=gone, and a case that can silently
  # skip the assertion it exists for is the one that will be skipped on the day it matters.
  LOCK=$_t/l6; HOLDER=$LOCK/holder; mkdir -p "$LOCK"; _no_holder_seen=0
  python3 -c '
import os, sys, time
pid = os.fork()
if pid == 0:
    os._exit(0)
open(sys.argv[1], "w").write(str(pid))
time.sleep(20)
' "$_t/zpid" & _zparent=$!
  for _i in 1 2 3 4 5 6 7 8 9 10; do [ -s "$_t/zpid" ] && break; sleep 0.2; done
  _z=$(cat "$_t/zpid" 2>/dev/null || true)
  _zst=$(ps -o stat= -p "${_z:-0}" 2>/dev/null | tr -d ' ' || true)
  case "$_zst" in
    Z*) printf 'pid=%s\ndeliberate=no\n' "$_z" > "$HOLDER"
        _case "zombie holder (kill -0 would say alive)" dead
        # THE NEGATIVE CONTROL: if kill -0 rejected this pid too, the case would pass for the wrong
        # reason and prove nothing about reading stat.
        if kill -0 "$_z" 2>/dev/null; then
          echo "  ok   negative control: kill -0 accepts pid $_z, so reading ps stat is load-bearing"
        else
          echo "  FAIL negative control: kill -0 rejected the zombie, so this case says nothing about stat" >&2
          _fails=$((_fails + 1))
        fi;;
    *)  echo "  FAIL zombie holder: no zombie was created (stat=${_zst:-gone}, pid=${_z:-unset}) --" >&2
        echo "       the case that motivates reading ps stat did not run" >&2
        _fails=$((_fails + 1));;
  esac
  kill "$_zparent" 2>/dev/null || true
  wait "$_zparent" 2>/dev/null || true
  # AGE IS NOT THE CRITERION: an hour-old lock with a live holder is alive. This is the assertion
  # that fails if anyone reintroduces `find -mmin +10`.
  LOCK=$_t/l7; HOLDER=$LOCK/holder; mkdir -p "$LOCK"; _no_holder_seen=0
  printf 'pid=%s\npurpose=long merge\ndeliberate=no\n' "$$" > "$HOLDER"
  touch -t 200001010000 "$LOCK" "$HOLDER" 2>/dev/null || true
  _case "26-year-old lock, live holder (age must not decide)" alive
  rm -rf "$_t"
  # THE SECOND-READER GATE, six worlds, driven against a REAL scratch repo rather than a
  # reimplementation of the predicate. Both directions: a gate that only ever refuses passes
  # every negative case, and one that only ever accepts is the prose rule it replaced.
  _g=$(mktemp -d)
  (
    cd "$_g" && git init -q . && git config user.email t@t && git config user.name T
    mkdir -p runs && echo x > model.py && echo y > train.py && echo z > other.txt
    : > runs/review.jsonl && git add -A && git commit -qm init
    _base=$(git rev-parse HEAD)
    git checkout -q -b feat && echo x2 >> model.py && git commit -qam "feat: touch model"
    git checkout -q -b docsonly "$_base" && echo w >> other.txt && git commit -qam "docs: elsewhere"
    git checkout -q -b revonly "$_base" && echo r >> model.py && git commit -qam 'Revert "x"'
    git checkout -q "$_base" 2>/dev/null
  ) >/dev/null 2>&1
  _sha=$(git -C "$_g" rev-parse feat)
  _gcase() {  # $1=name $2=branch $3=want refused|accepted
    if ( MAIN=$_g; cd "$_g"; _review_gate "$2" ) 2>/dev/null; then _got=accepted; else _got=refused; fi
    if [ "$_got" != "$3" ]; then
      echo "  FAIL review-gate $1: want $3, got $_got" >&2; _fails=$((_fails + 1))
    else
      echo "  ok   review-gate $1: $_got"
    fi
  }
  # main is where the gate compares from, so point it at the base commit.
  git -C "$_g" branch -f main "$(git -C "$_g" rev-list --max-parents=0 HEAD | head -1)" >/dev/null 2>&1
  : > "$_g/runs/review.jsonl"
  _gcase "model.py, no row" feat refused
  printf '{"reviewer": "b0", "artifact": "model.py @ %s"}\n' "${_sha:0:8}" > "$_g/runs/review.jsonl"
  _gcase "model.py, row by another" feat accepted
  # SELF-REVIEW IS NOT A REVIEW -- the case that decides whether this gate enforces anything.
  printf '{"reviewer": "feat", "artifact": "model.py @ %s"}\n' "${_sha:0:8}" > "$_g/runs/review.jsonl"
  _gcase "model.py, self-review" feat refused
  printf '{"reviewer": "b0", "artifact": "model.py @ deadbeef"}\n' > "$_g/runs/review.jsonl"
  _gcase "row names another sha" feat refused
  : > "$_g/runs/review.jsonl"
  _gcase "touches neither file" docsonly accepted
  _gcase "lone revert is exempt" revonly accepted
  rm -rf "$_g"

  # THE DEADLINE GUARD MUST NOT REFUSE THE MODES THAT NEITHER MERGE NOR COMMIT. It did:
  # `timeout 120 bash scripts/merge_main.sh --selftest` was refused minutes after the guard
  # landed, so the guard's only observed failure was against its own verification. Driven by
  # running this script for real under a timeout, in both directions -- the exempt modes must
  # exit 0 and a merge must still be refused -- because a case that only inspected the `case`
  # pattern would pass for a version whose exemption never reached the branch.
  _dcase() {  # $1=name $2=args $3=want exempt|refused
    # THREE OUTCOMES, NOT TWO (4c, 2026-09-06). This was `case $_out in *REFUSING*) refused;;
    # *) exempt;; esac`, so ANY output that lacked the refusal line read as `exempt` -- including
    # the case where the guard never answered at all. Measured under three concurrent merges:
    # `timeout 20` killed the inner run while it was reading the parent cmdline through ps, the
    # output had no REFUSING line, and the world reported `want refused, got exempt` -- a red
    # against the guard for a fact about machine load. Direct reruns passed.
    #
    # The exit code is the discriminator and it was being thrown away by `|| true`. 124 is
    # timeout's own kill; anything else nonzero with no verdict in the output is a crash. Both
    # are "could not check", which is a RETRY here rather than a failure, for the same reason
    # pod_sync_check exits 2 on "cannot run": a check that cannot run has not found anything.
    _drc=0; _out=$(timeout 20 bash "$0" $2 2>&1) || _drc=$?
    case "$_out" in
      *"REFUSING: this merge is running under"*) _got=refused;;
      *) if [ "$_drc" -eq 124 ] || { [ "$_drc" -ne 0 ] && [ "$_drc" -ne 2 ]; }; then
           _got=no-answer
         else
           _got=exempt
         fi;;
    esac
    if [ "$_got" = "no-answer" ]; then
      # One retry, then say what happened rather than scoring it. Under load the read succeeds
      # on the second try; if it does not, "the guard could not be exercised" is the honest
      # result and it is not the same claim as "the guard is wrong".
      _drc=0; _out=$(timeout 20 bash "$0" $2 2>&1) || _drc=$?
      case "$_out" in
        *"REFUSING: this merge is running under"*) _got=refused;;
        *) [ "$_drc" -eq 124 ] && _got=no-answer || _got=exempt;;
      esac
    fi
    if [ "$_got" = "no-answer" ]; then
      echo "  SKIP deadline guard, $1: the guard did not answer twice (exit $_drc, likely a slow" >&2
      echo "       ps under load). NOT a failure and NOT a pass -- rerun when the box is quiet." >&2
    elif [ "$_got" != "$3" ]; then
      echo "  FAIL deadline guard, $1: want $3, got $_got" >&2; _fails=$((_fails + 1))
    else
      echo "  ok   deadline guard, $1 -> $_got"
    fi
  }
  # --release, NOT --selftest, FOR THE EXEMPT CASE. Recursing into --selftest cannot work: a full
  # selftest takes 41 s (measured 2026-09-06) and _dcase runs it under `timeout 20`, so the inner
  # run was ALWAYS killed and its empty output scored `exempt` by the catch-all. The world passed
  # for the wrong reason -- it never once reached the exemption it claims to test, on main or here.
  # --release is exempt by the same clause, does no merge, and returns in milliseconds, so what is
  # scored is the guard's verdict rather than a timeout kill. It is a no-op when no lock is held.
  _dcase "an exempt mode under a timeout is not refused" --release exempt
  _dcase "a branch merge under a timeout is refused" _no_such_branch_selftest refused

  # THE DEADLINE MUST BE THE PARENT'S OWN argv[0], NOT TEXT IT CARRIES (4c, 2026-09-06). Run with
  # NO timeout parent, but from a shell whose cmdline contains the words in a heredoc -- which is
  # what refused 4c's merge for a `timeout 1` that existed only inside a string. _dcase cannot
  # express this: every case it runs is already under `timeout 20`, so the property needs its own
  # world with no deadline at all.
  _o_quoted=$(bash -c 'cat <<XX >/dev/null
a heredoc that mentions timeout 1 in quoted text
XX
bash "$0" _no_such_branch_selftest 2>&1' "$0" 2>&1 || true)
  case "$_o_quoted" in
    *"REFUSING: this merge is running under"*)
      echo "  FAIL deadline guard, quoted text: refused for a deadline that exists only inside a" >&2
      echo "       string in the parent's cmdline -- match argv[0], not the whole line" >&2
      _fails=$((_fails + 1));;
    *) echo "  ok   deadline guard, the words in quoted text are not a deadline";;
  esac

  # THE STAGED-INDEX CARRY (tilerl-31), six worlds against a REAL scratch repo, driving the real
  # _carry_stage/_carry_restore. The whole point is that the CONSEQUENCE is asserted -- the merge
  # succeeds and BOTH changes are present -- not that the functions ran without error.
  _c=$(mktemp -d)
  _ccase() {  # $1=name $2=want ok|fail  $3..=nothing; the body is inline per world
    if [ "$_cgot" != "$2" ]; then
      echo "  FAIL carry $1: want $2, got $_cgot${_cwhy:+ ($_cwhy)}" >&2; _fails=$((_fails + 1))
    else
      echo "  ok   carry $1 -> $_cgot"
    fi
  }
  _cworld() {  # build: main advanced on shared.txt; work diverged; mine.txt staged
    rm -rf "$_c/w"; mkdir -p "$_c/w"
    (
      cd "$_c/w" && git init -q -b main . && git config user.email t@t && git config user.name T
      printf 'base\n' > shared.txt; printf 'base\n' > mine.txt
      git add -A && git commit -qm base
      git checkout -q -b work
      printf 'other\n' > shared.txt
      git checkout -q main && printf 'other\n' > shared.txt && git commit -qam "main edits shared"
      git checkout -q work
      if [ "${1:-diverge}" = "diverge" ]; then
        printf 'localcommit\n' > mine.txt && git commit -qam "work commits mine"
      fi
      printf 'staged\n' > mine.txt && git add mine.txt
    ) >/dev/null 2>&1
  }
  # W1: THE JOINT ACCEPTANCE TEST. Staged path main never touches, histories diverged. The merge
  # must succeed and the RESULT must hold both changes -- main's shared.txt AND the staged mine.txt.
  _cworld diverge
  _cgot=fail; _cwhy=""
  _cout=$( cd "$_c/w" && _co=$(_carry_stage work) \
           && _cr=$(printf '%s\n' "$_co" | head -1) && _cp=$(printf '%s\n' "$_co" | tail -n +2) \
           && git merge --no-edit main >/dev/null 2>&1 \
           && { [ -n "$_cr" ] && _carry_restore "$_cr" "$_cp" >/dev/null 2>&1; } \
           && grep -q other shared.txt && grep -q staged mine.txt && echo BOTH ) 2>&1 || true
  case "$_cout" in *BOTH*) _cgot=ok;; *) _cwhy="$_cout";; esac
  _ccase "W1 staged path + diverged history: merge succeeds, BOTH changes present" ok
  # W1-CONTROL: the same world with NO carry must FAIL, or W1 proves nothing about the carry.
  _cworld diverge
  _cgot=ok; _cwhy=""
  ( cd "$_c/w" && git merge --no-edit main ) >/dev/null 2>&1 && _cwhy="the bare merge succeeded" || _cgot=fail
  _ccase "W1-control bare merge on the same world is refused by git" fail
  # W2: the ref already exists -- a previous carry never restored. Must refuse, not clobber.
  _cworld diverge
  _cgot=ok; _cwhy=""
  ( cd "$_c/w" && git update-ref refs/wip/work HEAD && _carry_stage work ) >/dev/null 2>&1 \
    && _cwhy="_carry_stage overwrote an existing refs/wip/work" || _cgot=fail
  _ccase "W2 refs/wip/<branch> already exists: refuse" fail
  # ...and the refusal must NAME the ref, or the operator cannot find their own work.
  _cworld diverge
  _cout=$( cd "$_c/w" && git update-ref refs/wip/work HEAD >/dev/null 2>&1; \
           cd "$_c/w" && _carry_stage work 2>&1 >/dev/null ) 2>/dev/null || true
  case "$_cout" in
    *"refs/wip/work"*) echo "  ok   carry W2 refusal names the ref";;
    *) echo "  FAIL carry W2 refusal does not name refs/wip/work: $_cout" >&2; _fails=$((_fails + 1));;
  esac
  # W3: FAST-FORWARD whose changed set does NOT include the staged path -- no carry. _cworld
  # stages mine.txt and main only edits shared.txt, so the fast-forward will not touch it.
  _cworld ff
  _cgot=fail; _cwhy=""
  _cout=$( cd "$_c/w" && _carry_stage work ) 2>/dev/null || true
  if [ -z "$_cout" ]; then _cgot=ok; else _cwhy="carried on a fast-forward that touches nothing staged: $_cout"; fi
  _ccase "W3 fast-forward, staged path outside its changed set: no carry" ok
  # ...and the world must really be a fast-forward, or W3 passes for the wrong reason.
  if ( cd "$_c/w" && git merge-base --is-ancestor HEAD main ); then
    echo "  ok   carry W3 world control: HEAD really is an ancestor of main"
  else
    echo "  FAIL carry W3 world control: the world is not fast-forwardable, so it tests nothing" >&2
    _fails=$((_fails + 1))
  fi
  # W3b: FAST-FORWARD THAT DOES CHANGE THE STAGED PATH -- must carry, and the merge must then
  # succeed. THIS IS THE CASE THAT BIT (de, 2026-09-06): the first version returned early on any
  # fast-forward, on a measurement taken with the staged path outside the changed set, and the
  # real merge aborted with "Your local changes to the following files would be overwritten by
  # merge: runs/tasks.jsonl, scripts/harness.py" -- both staged, both changed by main. W3 alone
  # passed the broken version, so the world it lacked is the whole finding.
  rm -rf "$_c/w"; mkdir -p "$_c/w"
  (
    cd "$_c/w" && git init -q -b main . && git config user.email t@t && git config user.name T
    printf 'top\nmiddle\nbottom\n' > f.txt && git add -A && git commit -qm base
    git checkout -q -b work                       # work == main, so this is a fast-forward
    git checkout -q main && printf 'TOP_FROM_MAIN\nmiddle\nbottom\n' > f.txt \
      && git commit -qam "main edits the top of f.txt"
    # SAME path main changed, DIFFERENT lines. The first version of this world had both sides
    # rewrite the single line `base`, which is a genuine three-way conflict -- so once the restore
    # became a merge rather than an overwrite, the world asserted "succeeds" on inputs that must
    # conflict. The property under test is the TRIGGER (a fast-forward that touches a staged path
    # must carry), not conflict handling; W9 covers the conflict.
    git checkout -q work && printf 'top\nmiddle\nBOTTOM_FROM_ME\n' > f.txt && git add f.txt
  ) >/dev/null 2>&1
  _cgot=fail; _cwhy=""
  _cout=$( cd "$_c/w" && _co=$(_carry_stage work) \
           && _cr=$(printf '%s\n' "$_co" | head -1) && _cp=$(printf '%s\n' "$_co" | tail -n +2) \
           && [ -n "$_cr" ] \
           && git merge --no-edit main >/dev/null 2>&1 \
           && _carry_restore "$_cr" "$_cp" >/dev/null 2>&1 \
           && grep -q TOP_FROM_MAIN f.txt && grep -q BOTTOM_FROM_ME f.txt && echo CARRIED ) 2>&1 || true
  case "$_cout" in *CARRIED*) _cgot=ok;; *) _cwhy="$_cout";; esac
  _ccase "W3b fast-forward that changes the staged path: carry, both edits survive" ok
  # W9: A REAL CONFLICT REFUSES, KEEPS THE REF, AND LEAVES MARKERS. Both sides rewrite the SAME
  # line, so no merge can pick a side -- and the carried content is then only recoverable from the
  # ref, which is why the ref must NOT be dropped here. Silently choosing one side is the defect
  # the three-way restore exists to remove, so it must not reappear as a silent success.
  rm -rf "$_c/w3"; mkdir -p "$_c/w3"
  (
    cd "$_c/w3" && git init -q -b main . && git config user.email t@t && git config user.name T
    printf 'one\n' > f.txt && git add -A && git commit -qm base
    git checkout -q -b work && printf 'x\n' > sentinel && git add sentinel && git commit -qm s
    git checkout -q main && printf 'MAIN\n' > f.txt && git commit -qam "main rewrites the line"
    git checkout -q work && printf 'MINE\n' > f.txt && git add f.txt
  ) >/dev/null 2>&1
  _cgot=ok; _cwhy=""
  _cout=$( cd "$_c/w3" && _co=$(_carry_stage work) \
           && _cr=$(printf '%s\n' "$_co" | head -1) && _cp=$(printf '%s\n' "$_co" | tail -n +2) \
           && git merge --no-edit main >/dev/null 2>&1 \
           && _carry_restore "$_cr" "$_cp" ) 2>&1 && _cwhy="restore reported success on a real conflict" || _cgot=fail
  _ccase "W9 real conflict: the restore refuses" fail
  if ( cd "$_c/w3" && git rev-parse -q --verify refs/wip/work >/dev/null ) \
     && ( cd "$_c/w3" && grep -q '<<<<<<<' f.txt ); then
    echo "  ok   carry W9 the ref is kept and the file holds conflict markers"
  else
    echo "  FAIL carry W9: a conflict must keep refs/wip/work (the only clean copy) and leave" >&2
    echo "       markers in the file; ref present: $( cd "$_c/w3" && git rev-parse -q --verify refs/wip/work >/dev/null && echo yes || echo NO), markers: $( cd "$_c/w3" && grep -q '<<<<<<<' f.txt && echo yes || echo NO)" >&2
    _fails=$((_fails + 1))
  fi
  # W10: A PENDING MERGE (MERGE_HEAD set, index full of MAIN's versions) IS REFUSED, not carried.
  # 4c hit this twice in aupai-fb on 2026-09-06: a merge resolved, its COMMIT was refused by the
  # hook, and the next run read main's 10-30 staged files as the operator's index and carried them
  # to refs/wip/. The world reproduces it with `git merge --no-commit`, which leaves exactly that
  # state -- MERGE_HEAD set, zero unmerged paths, main's content staged.
  rm -rf "$_c/w10"; mkdir -p "$_c/w10"
  (
    cd "$_c/w10" && git init -q -b main . && git config user.email t@t && git config user.name T
    printf 'one\n' > f.txt && git add -A && git commit -qm base
    git checkout -q -b work && printf 'x\n' > sentinel && git add -A && git commit -qm s
    git checkout -q main && printf 'MAIN\n' > g.txt && git add -A && git commit -qm "main adds g"
    git checkout -q work
    git merge --no-commit --no-ff main
  ) >/dev/null 2>&1
  _cgot=ok; _cwhy=""
  # The 2>&1 is INSIDE the substitution: the refusal goes to stderr, so with the redirect
  # outside, _cout is empty and any grep of it passes for the wrong reason.
  _cout=$( cd "$_c/w10" && _carry_stage work 2>&1 ) \
    && _cwhy="carried a pending merge's index instead of refusing" || _cgot=fail
  _ccase "W10 pending merge: carry refuses instead of carrying main's staged files" fail
  if printf '%s' "$_cout" | grep -q "merge is already in progress"; then
    echo "  ok   carry W10 the refusal names the pending merge"
  else
    echo "  FAIL carry W10: the refusal must say a merge is in progress, so the reader knows to" >&2
    echo "       run \`git commit --no-edit\`; it said: $(printf '%s' "$_cout" | head -1)" >&2
    _fails=$((_fails + 1))
  fi
  if ( cd "$_c/w10" && ! git rev-parse -q --verify refs/wip/work >/dev/null ); then
    echo "  ok   carry W10 no ref was written"
  else
    echo "  FAIL carry W10: refusing must write NO ref -- a refs/wip/ holding main's own content" >&2
    echo "       is worse than none, because the restore would put it back over the merge" >&2
    _fails=$((_fails + 1))
  fi
  # W10-CONTROL, IN THE OPPOSITE DIRECTION: an ordinary staged index in a world with no pending
  # merge must still carry. Without it, a _carry_stage that refused every call would pass W10.
  # It builds its OWN world rather than concluding w10's: under a mutant that never refuses, w10's
  # merge is left mid-flight and the control then fails downstream, on state rather than on merit.
  rm -rf "$_c/w10c"; mkdir -p "$_c/w10c"
  (
    cd "$_c/w10c" && git init -q -b main . && git config user.email t@t && git config user.name T
    printf 'one\n' > f.txt && git add -A && git commit -qm base
    git checkout -q -b work && printf 'x\n' > sentinel && git add -A && git commit -qm s
    git checkout -q main && printf 'MAIN\n' > f.txt && git add -A && git commit -qm "main edits f"
    git checkout -q work && printf 'mine\n' > f.txt && git add f.txt
  ) >/dev/null 2>&1
  _cgot=fail; _cwhy=""
  _cout=$( cd "$_c/w10c" && _carry_stage work 2>/dev/null ) && _cgot=ok \
    || _cwhy="refused an ordinary staged index in a world with no merge in progress"
  _ccase "W10-control no pending merge: a real staged index still carries" ok

  # K1-K3: _merge_failure_kind's THREE states, each in its own world. The whole defect was one
  # state being reported as another, so every state is driven and the two nonzero ones must
  # DISAGREE -- a predicate answering the same word for both would restore the bug.
  _kcase() {  # <label> <dir> <want>
    _kgot=$( cd "$2" && _merge_failure_kind )
    if [ "$_kgot" = "$3" ]; then
      echo "  ok   kind $1 -> $3"
    else
      echo "  FAIL kind $1: want $3, got $_kgot" >&2
      _fails=$((_fails + 1))
    fi
  }
  # K1 CONFLICT: both sides edit one line, so the index holds unmerged entries.
  rm -rf "$_c/k1"; mkdir -p "$_c/k1"
  (
    cd "$_c/k1" && git init -q -b main . && git config user.email t@t && git config user.name T
    printf 'one\n' > f.txt && git add -A && git commit -qm base
    git checkout -q -b work && printf 'WORK\n' > f.txt && git add -A && git commit -qm w
    git checkout -q main && printf 'MAIN\n' > f.txt && git add -A && git commit -qm m
    git checkout -q work && git merge --no-edit main
  ) >/dev/null 2>&1 || true   # the merge conflicts BY DESIGN; under set -e it would abort the run
  _kcase "K1 both sides edited one line" "$_c/k1" conflict
  # K2 REFUSED COMMIT: the merge resolves (disjoint files) and a hook that always exits 1 refuses
  # its commit. A REAL failing hook, not `--no-commit` -- the state has to arrive the way 4c's did.
  # The hook is `pre-merge-commit`, NOT `pre-commit`: `git merge` runs only the former, so a world
  # built on `pre-commit` merges cleanly and reads `none`. This repo symlinks both to one script
  # (`git rev-parse --git-common-dir`/hooks, read 2026-09-07), which is why 4c's merge hit it.
  rm -rf "$_c/k2"; mkdir -p "$_c/k2"
  (
    cd "$_c/k2" && git init -q -b main . && git config user.email t@t && git config user.name T
    printf 'one\n' > f.txt && git add -A && git commit -qm base
    git checkout -q -b work && printf 'x\n' > s.txt && git add -A && git commit -qm w
    git checkout -q main && printf 'MAIN\n' > g.txt && git add -A && git commit -qm m
    git checkout -q work
    printf '#!/bin/sh\nexit 1\n' > .git/hooks/pre-merge-commit
    chmod +x .git/hooks/pre-merge-commit
    git merge --no-edit main
  ) >/dev/null 2>&1 || true   # the hook refuses the commit BY DESIGN, so this exits nonzero too
  _kcase "K2 resolved merge whose commit the hook refused" "$_c/k2" refused-commit
  if ( cd "$_c/k2" && [ -z "$(git ls-files -u)" ] \
       && [ -f "$(git rev-parse --git-dir)/MERGE_HEAD" ] \
       && [ -n "$(git diff --cached --name-only)" ] ); then
    echo "  ok   kind K2 world control: zero unmerged, MERGE_HEAD set, main's files staged"
  else
    echo "  FAIL kind K2 world control: the world did not reproduce the state -- the hook may" >&2
    echo "       have been skipped, so K2 would pass against something else entirely" >&2
    _fails=$((_fails + 1))
  fi
  # K3 NONE: a clean worktree with no merge in flight.
  _kcase "K3 no merge in flight" "$_c/w10c" none

  # W3b-CONTROL: the bare merge on that world must FAIL, or W3b proves nothing.
  rm -rf "$_c/w"; mkdir -p "$_c/w"
  (
    cd "$_c/w" && git init -q -b main . && git config user.email t@t && git config user.name T
    printf 'base\n' > f.txt && git add -A && git commit -qm base
    git checkout -q -b work
    git checkout -q main && printf 'main\n' > f.txt && git commit -qam "main edits f.txt"
    git checkout -q work && printf 'staged\n' > f.txt && git add f.txt
  ) >/dev/null 2>&1
  _cgot=ok; _cwhy=""
  ( cd "$_c/w" && git merge --no-edit main ) >/dev/null 2>&1 \
    && _cwhy="the bare fast-forward succeeded, so W3b tests nothing" || _cgot=fail
  _ccase "W3b-control bare fast-forward on the same world is refused by git" fail
  # W4: an empty index is untouched -- no ref written, nothing to restore.
  _cworld diverge
  ( cd "$_c/w" && git reset -q HEAD -- . ) >/dev/null 2>&1
  _cgot=fail; _cwhy=""
  _cout=$( cd "$_c/w" && _carry_stage work ) 2>/dev/null || true
  if [ -z "$_cout" ] && ! ( cd "$_c/w" && git rev-parse -q --verify refs/wip/work >/dev/null ); then
    _cgot=ok
  else
    _cwhy="wrote a ref for an empty index"
  fi
  _ccase "W4 empty index: no ref, no carry" ok
  # W5: NO STASH. The carry must never touch refs/stash -- that stack is shared with every
  # worktree, and two sessions popping it apply diffs they never wrote (e1 and b0, 2026-09-02).
  _cworld diverge
  ( cd "$_c/w" && _carry_stage work ) >/dev/null 2>&1
  if ( cd "$_c/w" && git rev-parse -q --verify refs/stash >/dev/null ); then
    echo "  FAIL carry W5: the carry wrote refs/stash -- that stack is shared across worktrees" >&2
    _fails=$((_fails + 1))
  else
    echo "  ok   carry W5 refs/stash untouched"
  fi
  # W6: A CONFLICTING MERGE LEAVES THE CARRY INTACT AND SAYS SO. The staged work must survive a
  # failed merge, because the operator's next move is to resolve and restore it by name.
  rm -rf "$_c/w"; mkdir -p "$_c/w"
  (
    cd "$_c/w" && git init -q -b main . && git config user.email t@t && git config user.name T
    printf 'base\n' > both.txt && git add -A && git commit -qm base
    git checkout -q -b work && printf 'work\n' > both.txt && git commit -qam "work edits both"
    git checkout -q main && printf 'main\n' > both.txt && git commit -qam "main edits both"
    git checkout -q work && printf 'staged\n' > other.txt && git add other.txt
  ) >/dev/null 2>&1
  _cgot=fail; _cwhy=""
  _cref=$( cd "$_c/w" && _carry_stage work ) 2>/dev/null || true
  if [ -n "$_cref" ]; then
    ( cd "$_c/w" && git merge --no-edit main ) >/dev/null 2>&1 || true
    if ( cd "$_c/w" && git rev-parse -q --verify refs/wip/work >/dev/null ) \
       && ( cd "$_c/w" && git show "refs/wip/work:other.txt" 2>/dev/null | grep -q staged ); then
      _cgot=ok
    else
      _cwhy="the carry ref or its content did not survive the conflicting merge"
    fi
  else
    _cwhy="no carry was made in the conflict world"
  fi
  _ccase "W6 conflicting merge: the carried index survives at refs/wip/work" ok
  # W7: A UNION-MERGE LEDGER IS UNIONED BACK, NOT OVERWRITTEN. THE CASE THAT COST REAL ROWS
  # (de, 2026-09-06): the carry restored runs/tasks.jsonl over the merge and dropped e1-48 and
  # e1-49, rows that had arrived in main during that same merge. Caught by the hook's
  # ledger-append-only gate, not by anything here, and the carry is meant for exactly this tree.
  #
  # Both sides append a DIFFERENT row to the same ledger, and the result must hold both. A world
  # where only one side appends passes an overwrite restore, which is why the two appends matter.
  rm -rf "$_c/w"; mkdir -p "$_c/w"
  (
    cd "$_c/w" && git init -q -b main . && git config user.email t@t && git config user.name T
    mkdir -p runs
    printf 'runs/led.jsonl merge=union\n' > .gitattributes
    printf '{"id":"base"}\n' > runs/led.jsonl
    git add -A && git commit -qm base
    git checkout -q -b work && printf 'x\n' > code.py && git commit -qm "work adds code.py" -- . 2>/dev/null \
      || { git add code.py && git commit -qm "work adds code.py"; }
    git checkout -q main
    printf '{"id":"base"}\n{"id":"theirs"}\n' > runs/led.jsonl   # main appends a row
    git commit -qam "main appends theirs"
    git checkout -q work
    printf '{"id":"base"}\n{"id":"mine"}\n' > runs/led.jsonl     # I append a different row
    git add runs/led.jsonl
  ) >/dev/null 2>&1
  _cgot=fail; _cwhy=""
  _cout=$( cd "$_c/w" && _co=$(_carry_stage work) \
           && _cr=$(printf '%s\n' "$_co" | head -1) && _cp=$(printf '%s\n' "$_co" | tail -n +2) \
           && [ -n "$_cr" ] \
           && git merge --no-edit main >/dev/null 2>&1 \
           && _carry_restore "$_cr" "$_cp" >/dev/null 2>&1 \
           && grep -q '"theirs"' runs/led.jsonl && grep -q '"mine"' runs/led.jsonl && echo BOTHROWS ) 2>&1 || true
  case "$_cout" in *BOTHROWS*) _cgot=ok;; *) _cwhy="$_cout";; esac
  _ccase "W7 union-merge ledger: both sides' rows survive the carry" ok
  # W8: A SOURCE FILE MAIN ALSO CHANGED KEEPS BOTH EDITS. The fourth defect and the largest
  # (de, 2026-09-06): the restore overwrote scripts/harness.py with the carried copy and reverted
  # 517 lines of other sessions' work, e1's 9a382298 among them. The visible symptom was a check
  # reading FAIL in this tree while main returned WARN, and I mis-reported that to the controller
  # as a fact about another session's queue. Nothing raised; the file went backwards.
  #
  # W1 does not cover it: there, main changes shared.txt and I stage mine.txt, so the merge and my
  # edit touch different paths and an overwrite of mine.txt loses nothing. Here BOTH sides edit the
  # SAME file in DIFFERENT places, which is the case the carry now fires on by construction.
  rm -rf "$_c/w"; mkdir -p "$_c/w"
  (
    cd "$_c/w" && git init -q -b main . && git config user.email t@t && git config user.name T
    printf 'top\nmiddle\nbottom\n' > src.py && git add -A && git commit -qm base
    git checkout -q -b work && printf 'x\n' > sentinel && git add sentinel && git commit -qm "work commits sentinel"
    git checkout -q main
    printf 'TOP_FROM_MAIN\nmiddle\nbottom\n' > src.py && git commit -qam "main edits the top"
    git checkout -q work
    printf 'top\nmiddle\nBOTTOM_FROM_ME\n' > src.py && git add src.py   # my edit, far from main's
  ) >/dev/null 2>&1
  _cgot=fail; _cwhy=""
  _cout=$( cd "$_c/w" && _co=$(_carry_stage work) \
           && _cr=$(printf '%s\n' "$_co" | head -1) && _cp=$(printf '%s\n' "$_co" | tail -n +2) \
           && [ -n "$_cr" ] \
           && git merge --no-edit main >/dev/null 2>&1 \
           && _carry_restore "$_cr" "$_cp" >/dev/null 2>&1 \
           && grep -q TOP_FROM_MAIN src.py && grep -q BOTTOM_FROM_ME src.py && echo BOTHEDITS ) 2>&1 || true
  case "$_cout" in *BOTHEDITS*) _cgot=ok;; *) _cwhy="$_cout";; esac
  _ccase "W8 source file both sides edited: main's change AND mine survive" ok
  # W8-CONTROL: the world must be a real divergent merge over one staged source path, or W8 could
  # pass because no carry ran. Assert the bare merge is refused.
  rm -rf "$_c/w2"; mkdir -p "$_c/w2"
  (
    cd "$_c/w2" && git init -q -b main . && git config user.email t@t && git config user.name T
    printf 'top\nmiddle\nbottom\n' > src.py && git add -A && git commit -qm base
    git checkout -q -b work && printf 'x\n' > sentinel && git add sentinel && git commit -qm "work commits sentinel"
    git checkout -q main && printf 'TOP_FROM_MAIN\nmiddle\nbottom\n' > src.py && git commit -qam "main edits the top"
    git checkout -q work && printf 'top\nmiddle\nBOTTOM_FROM_ME\n' > src.py && git add src.py
  ) >/dev/null 2>&1
  _cgot=ok; _cwhy=""
  ( cd "$_c/w2" && git merge --no-edit main ) >/dev/null 2>&1 \
    && _cwhy="the bare merge succeeded, so W8 tests nothing" || _cgot=fail
  _ccase "W8-control bare merge on the same world is refused by git" fail
  # W7-CONTROL: the world must really be a divergent merge that stages the ledger, or W7 could
  # pass because no carry happened at all.
  if ( cd "$_c/w" && git log --oneline -1 | grep -q . ); then
    echo "  ok   carry W7 world control: the merge world was built"
  else
    echo "  FAIL carry W7 world control: the world was not built, so it tests nothing" >&2
    _fails=$((_fails + 1))
  fi
  rm -rf "$_c"

  if [ "$_fails" -gt 0 ]; then echo "merge_main selftest: $_fails failure(s)" >&2; exit 1; fi
  echo "merge_main selftest OK: liveness decides, not age -- a live holder and a live deliberate"
  echo "  hold both read alive at any age; gone, zombie and unparseable read dead; a holderless"
  echo "  lock is alive inside its ${_NO_HOLDER_GRACE}s grace and dead after it."
  exit 0
fi


# SHARED-FILE CLAIM RELEASE (T0, 2026-09-04), unchanged in behaviour, moved into a function so
# the integration path reads as one sequence. The claim that let the branch's shared-file commit
# pass lives in the branch worktree, not main, so it is released there.
#
# THE OFFSET IS 19, NOT 16. `branch refs/heads/` is 18 characters, so substr($0,16) starts three
# too early and yields `ds/de` for branch `de` -- it never equals the bare name and the release
# was silently skipped for EVERY branch. Not cosmetic even with the 6h TTL: $USER is `bytedance`
# for every session on this box, so one leaked claim blocks every other session.
_release_claims() {
  _wt=$(git -C "$MAIN" worktree list --porcelain 2>/dev/null \
    | awk -v b="$1" '/^worktree /{w=substr($0,10)} /branch refs\/heads\// && substr($0,19)==b && w!="" {print w; exit}')
  if [ -n "$_wt" ] && [ -f "$_wt/scripts/file_claim.py" ]; then
    # No --owner: file_claim's default derives from the script's own path, so invoking $_wt's
    # copy by absolute path already targets $_wt's claims as $_wt's owner. Passing $USER scoped
    # nothing -- it is `bytedance` for every session here.
    _rel=$(python3 "$_wt/scripts/file_claim.py" release-all 2>/dev/null || echo "release-all failed")
    echo "merge_main: shared-file claims on $1: $_rel" >&2
  elif [ -z "$_wt" ]; then
    # SAY SO WHEN NOTHING WAS RELEASED -- the absence of this line read as "merged after
    # --delete", which is how the substr(16) bug survived: it had never fired for any branch.
    echo "merge_main: no worktree matched branch $1 -- claims NOT released; 6h TTL clears them." >&2
  else
    echo "merge_main: $_wt has no scripts/file_claim.py -- claims NOT released" >&2
  fi
}

for _ in $(seq 1 120); do
  if _take_lock "merge $1 into main" no; then
    trap 'rm -f "$HOLDER"; rmdir "$LOCK"' EXIT
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
    if [ "${AUPAI_CONTROLLER:-0}" = "1" ]; then
      if ! _review_gate "$1" 2>/dev/null; then
        # THROUGH THE CLI, not a hand-appended JSON line. This append used to build the row itself,
        # which is how kind="override" existed in a writer while FRICTION_KINDS rejected it -- and it
        # was never caught because the writer had never fired (0 override rows in 62). A second
        # writer means the vocabulary is advisory: the CLI's `choices=FRICTION_KINDS` refuses an
        # unknown kind, a hand-append cannot. 4c/user ruling 2026-09-05: the CLI is the only writer.
        #
        # `|| true` because a failed ledger write must not abort a merge the controller has already
        # authorised; the CLI prints its own refusal, so a rejected kind is loud rather than silent.
        python3 "$MAIN/scripts/harness.py" friction add \
          --kind override --who tilerl \
          --blocked "merge $1 with unreviewed train.py/model.py commits" \
          --cause "AUPAI_CONTROLLER=1 used to bypass the second-reader refusal" \
          --commit || true
        echo "merge_main: second-reader gate OVERRIDDEN by AUPAI_CONTROLLER=1; logged to friction." >&2
      fi
    else
      _review_gate "$1" || exit 1
    fi
    # WHERE THE MERGE HAPPENS, and this is the whole rebuild. It used to run `git merge` HERE,
    # inside the shared integration tree, which made integrating a four-step non-atomic write
    # (worktree, index, ~30 s hook, commit) to a directory every session depends on. Any kill,
    # conflict or hook failure stranded it in a state only another session's files could repair.
    # The index-equals-HEAD rule, "never edit in the integration tree", the hook-runs-main's-copy
    # defect and the .hookstaged_* leftovers were all compensation for that one design.
    #
    # Now: the merge happens in the CALLER's own worktree, where only the caller depends on the
    # result, and main advances by an atomic compare-and-swap that touches no working tree at all.
    _wt_self=$(git rev-parse --show-toplevel 2>/dev/null || echo "")
    if [ -z "$_wt_self" ] || [ "$(cd "$_wt_self" && git rev-parse --abbrev-ref HEAD)" != "$1" ]; then
      echo "merge_main: run this from the worktree that holds branch $1 -- the merge happens" >&2
      echo "  there now, not in the integration tree." >&2
      exit 1
    fi
    # DETACH IT RATHER THAN REFUSING, WHEN IT IS SAFE TO (4c's ruling, 2026-09-05, after the
    # third occurrence in one day). A CAS onto a checked-out branch does NOT refuse: measured,
    # it returns 0, moves the ref, and leaves that tree's HEAD and index at the old commit so
    # every changed file reads as staged-modified -- silent, where a refusal would be loud. So
    # the tree must be detached before the CAS; the only question is who does it.
    #
    # Nobody may commit in the integration tree -- the pre-commit hook refuses there -- so a
    # detach of a CLEAN tree loses nothing that could exist. What keeps re-attaching it is a
    # person typing `git checkout main` to look at main, and a refusal aimed at that person
    # blocks every OTHER session's merge until someone notices. The refusal survives for the
    # case where something could actually be lost: a DIRTY tree, where `checkout --detach`
    # would carry the modifications along and hide them under a detached HEAD.
    if [ "$(git -C "$MAIN" rev-parse --abbrev-ref HEAD 2>/dev/null)" = "main" ]; then
      if [ -n "$(git -C "$MAIN" status --porcelain 2>/dev/null)" ]; then
        echo "REFUSING: the integration tree is on main AND dirty, so I will not detach it:" >&2
        git -C "$MAIN" status --porcelain 2>/dev/null | sed 's/^/    /' >&2
        echo "  A detach would carry these along and hide them under a detached HEAD. Nobody" >&2
        echo "  commits in that tree, so this is someone's work in the wrong place or a tool" >&2
        echo "  writing where it should not. Look before clearing it." >&2
        echo "  main is unmoved at $(git -C "$MAIN" rev-parse --short main 2>/dev/null)." >&2
        exit 1
      fi
      if ! git -C "$MAIN" checkout --detach main -q 2>/dev/null; then
        echo "REFUSING: the integration tree is on main and could not be detached." >&2
        echo "  git -C $MAIN checkout --detach main" >&2
        exit 1
      fi
      echo "merge_main: detached the integration tree (it was on main; a CAS there is silent)"
    fi
    _old=$(git -C "$MAIN" rev-parse main)
    # A DIRTY LEDGER ABORTS THE MERGE BEFORE IT STARTS, and git's own advice for it is `git
    # stash`, which is forbidden here (.git/refs/stash is shared across every worktree).
    # Naming the order costs three lines and is the whole recovery: commit the ledger row
    # path-scoped FIRST -- git can merge those without a person (merge=union, or prereg.jsonl's
    # key-union driver), so a row commits cleanly on its own and
    # the pre-commit behind-main refusal exempts a union-only commit -- and merge after. Doing
    # it the other way round means letting `friction add` commit mid-merge, which is the
    # dirty-during-commit shape the flip exists to remove (3b, 2026-09-05).
    # TESTED WITHOUT ATTEMPTING THE MERGE: a `git merge` in the condition would CONSUME the
    # merge when it succeeded, and the second call would then say "Already up to date" and
    # skip the drop check. Dirtiness alone is the test -- git aborts on it before merging.
    _dirty=$(git diff --name-only)
    if [ -n "$_dirty" ]; then
      echo "merge_main: uncommitted changes will abort the merge:" >&2
      echo "$_dirty" | sed 's/^/    /' >&2
      echo "  If these are ledger rows (runs/*.jsonl -- union, or prereg's key-union driver)," >&2
      echo "  commit them ALONE first, then re-run: git commit -m '<msg>' -- <the ledger file>" >&2
      echo "  A union-only commit is exempt from the behind-main refusal, so this works from" >&2
      echo "  a stale worktree. Do NOT git stash: .git/refs/stash is shared with every peer." >&2
      echo "  main is unmoved at ${_old:0:8}." >&2
      exit 1
    fi
    # THE STAGED INDEX, CARRIED THROUGH THE MERGE (tilerl-31). `_dirty` above reads
    # `git diff --name-only` -- UNSTAGED only -- so a staged path fell straight into the merge
    # below and aborted there, with git's own advice being `git stash`, which is forbidden here.
    # The mechanism and the refusal cases are documented on _carry_stage; --selftest drives it.
    # _carry_stage prints the ref on its first line and the carried paths after it.
    _carry_out=$(_carry_stage "$1") || { echo "  main is unmoved at ${_old:0:8}." >&2; exit 1; }
    _carry=$(printf '%s\n' "$_carry_out" | head -1)
    _carry_paths=$(printf '%s\n' "$_carry_out" | tail -n +2)
    if ! git merge --no-edit main; then
      # WHY `git merge` FAILED, WHICH IS TWO DIFFERENT THINGS. A content conflict leaves
      # unmerged index entries and no commit. A merge that resolved cleanly and then had its
      # COMMIT refused by the pre-commit hook leaves MERGE_HEAD set, ZERO unmerged paths, and
      # main's versions of every merged file staged. Both return nonzero, and reporting the
      # second as a conflict is what 4c hit twice in aupai-fb on 2026-09-06 (16:4xZ and
      # 17:5xZ): the message named no file because there was no conflicting file, and the next
      # run then carried 10-30 of main's staged paths to refs/wip/. Both times the recovery was
      # `git commit` of the pending merge, which the hook passed on retry.
      #
      # `git ls-files -u` is the discriminator, not MERGE_HEAD: MERGE_HEAD is set in BOTH cases.
      # It is `_merge_failure_kind`, a function, so --selftest drives the REAL predicate against
      # both worlds; inline here it would have been reachable only by running a whole merge.
      _unmerged=$(git ls-files -u | awk '{print $4}' | sort -u)
      if [ "$(_merge_failure_kind)" = "refused-commit" ]; then
        echo "merge_main: the merge RESOLVED, and its commit was refused -- read the refusal" >&2
        echo "  above this line, not a conflict message. Zero unmerged paths; main's versions" >&2
        echo "  of $(git diff --cached --name-only | wc -l | tr -d ' ') file(s) are staged and MERGE_HEAD is set." >&2
        echo "  Fix what the refusal names, then CONCLUDE the merge -- do not re-run this" >&2
        echo "  script first, it would carry main's staged paths to refs/wip/$1:" >&2
        echo "    git commit --no-edit        # the hook runs again; it passed on retry both" >&2
        echo "                               # times this was seen (4c, 2026-09-06)" >&2
        echo "    bash scripts/merge_main.sh $1" >&2
        echo "  To abandon instead: git merge --abort" >&2
      else
        echo "merge_main: $1 conflicts with main. Resolve in THIS worktree, commit, retry --" >&2
        if [ -n "$_unmerged" ]; then
          echo "  the conflicting path(s):" >&2
          printf '%s\n' "$_unmerged" | sed 's/^/    /' >&2
        fi
      fi
      if [ -n "$_carry" ]; then
        echo "  YOUR STAGED INDEX IS AT $_carry -- it is NOT lost and NOT restored. After you" >&2
        echo "  resolve, restore the carried paths BY NAME (never \`-- .\`, which reverts the" >&2
        echo "  merge) and drop the ref:" >&2
        printf '%s\n' "$_carry_paths" | sed "s|^|    git checkout $_carry -- |" >&2
        echo "    git update-ref -d $_carry" >&2
      fi
      echo "  nothing was written to the integration tree and main is unmoved at ${_old:0:8}." >&2
      exit 1
    fi
    if [ -n "$_carry" ]; then
      _carry_restore "$_carry" "$_carry_paths" || exit 1
    fi    # THE PENDING OVERRIDE ROWS, written here rather than by the hook. A hook that appends to a
    # ledger mid-commit dirties the tree during the commit and refused the next merge; the hook
    # records the event under its own git dir (invisible to `git status`, per-worktree) and this
    # step drains it through the CLI, which owns the accepted-kind list.
    #
    # MOVED BEFORE THE CAS (2026-09-06). It drained AFTER, so every row it wrote landed on the
    # BRANCH one commit behind the ref the CAS had just advanced: measured, 008ee382. A row
    # delay rather than a row loss -- the rows arrive inside whoever integrates next -- which is
    # why it went unnoticed. The CAS advances main to `_new`, so anything that must land has to
    # be committed before `_new` is read.
    #
    # AND IT GOES THROUGH THE SAME QUEUE, not `--commit` per row. That flag commits once PER ROW,
    # which is the cost tilerl-24 exists to remove; --defer appends to aupai_pending_rows and the
    # drain immediately below writes all of them in one commit. Two queues, one exit -- which is
    # also why this step must run BEFORE that drain and not merely before the CAS.
    _pend="$(git rev-parse --git-dir)/aupai_pending_friction"
    if [ -s "$_pend" ]; then
      _n=0
      while IFS= read -r _row; do
        [ -n "$_row" ] || continue
        python3 "$_wt_self/scripts/harness.py" friction add --kind override --who "$1" \
          --blocked "commit from behind main on $1" --cause "$_row" --defer >/dev/null 2>&1 \
          && _n=$((_n + 1)) || true
      done < "$_pend"
      [ "$_n" -gt 0 ] && : > "$_pend"
      echo "merge_main: queued $_n pending override row(s) for the drain" >&2
    fi
    # THE QUEUED FRICTION ROWS, ONE COMMIT FOR ALL OF THEM (4c, 2026-09-06). 601 commits landed
    # on main in 24 h and 98 were single ledger rows. The 105 merge commits cannot batch -- each
    # is one integration -- but these can, because nobody reads friction.jsonl in real time.
    # `friction add --defer` queues a whole JSON row under the git dir; this appends every queued
    # row and commits ONCE, path-scoped.
    #
    # DRAINED BEFORE `_new` IS READ, NOT AFTER THE CAS, and that ordering is the whole
    # correctness of this. Measured on its own first run (2026-09-06): draining after the CAS
    # produced commit c12576ea carrying both rows on the BRANCH while main already pointed at
    # 793ff59e -- one commit behind it. The rows were never lost, but they never reached main
    # either, and the next merge would have carried them silently. The CAS advances main to
    # `_new`, so anything that must land has to be committed before `_new` is read.
    #
    # The pre-existing override drain below still has this defect (008ee382, same run). It is
    # left for its owner rather than fixed here: it is a different queue with a different
    # writer, and changing both in one commit would make neither bisectable.
    #
    # TRUNCATED ONLY AFTER THE COMMIT SUCCEEDS, so a kill anywhere in between leaves the queue
    # intact and the rows land next time. The failure this rules out is losing a row, which is
    # the only irreversible outcome here: a row committed twice would be visible and fixable,
    # a row dropped is gone.
    _rows="$(git rev-parse --git-dir)/aupai_pending_rows"
    if [ -s "$_rows" ]; then
      _rn=$(grep -c . "$_rows" 2>/dev/null || echo 0)
      if cat "$_rows" >> "$_wt_self/runs/friction.jsonl" \
         && git -C "$_wt_self" commit -q -m "friction: $_rn queued row(s) from $1" \
              -- runs/friction.jsonl >/dev/null 2>&1; then
        : > "$_rows"
        echo "merge_main: drained $_rn queued friction row(s) into one commit" >&2
      else
        # The rows are in the file but not committed, or the commit was refused. Leave the
        # QUEUE intact -- draining again would duplicate, so say what state it is in rather
        # than guessing. `git checkout` on the ledger reverts the append; the queue re-drains.
        echo "merge_main: WARNING -- $_rn queued friction row(s) appended but NOT committed." >&2
        echo "  The queue is intact, so nothing is lost. Undo the append and re-run:" >&2
        echo "    git -C $_wt_self checkout -- runs/friction.jsonl" >&2
      fi
    fi
    _new=$(git rev-parse HEAD)
    # THE DROP CHECK RUNS BEFORE THE CAS, ON THE CANDIDATE, and a drop is now a REFUSAL. It used
    # to restore the dropped paths INTO the shared tree and stage them for someone else to
    # commit -- the shared-tree defect in miniature. Here main has not moved, so the fix happens
    # on the integrator's own side. --rev is required: merge_drops defaults to HEAD, and against
    # a detached integration tree that would check the wrong commit and print nothing, which
    # reads identically to clean (the failure its own docstring names).
    rc=0
    drops=$(python3 "$_wt_self/scripts/harness.py" --merge-drops --rev "$_new" 2>/dev/null) || rc=$?
    if [ "$rc" -gt 1 ]; then
      echo "merge_main: WARNING -- the merge-drop guard did not run (exit $rc). Main is unmoved." >&2
      exit 1
    fi
    if [ -n "$drops" ]; then
      echo "merge_main: this merge drops path(s) a parent held, with nobody deleting them:" >&2
      printf '%s\n' "$drops" | while IFS=$'\t' read -r path parent; do
        [ -n "$path" ] || continue
        echo "  $path (held by ${parent:0:8})" >&2
      done
      echo "merge_main: REFUSED before main moved. Restore them here and amend:" >&2
      echo "  git checkout <parent> -- <path> && git commit --amend --no-edit" >&2
      exit 1
    fi
    # THE ATOMIC STEP. On mismatch someone else landed first: re-merge and re-run, which is a
    # loop the caller drives rather than a lock we hold across a 30 s gate.
    #
    # -m SIGNS THE REFLOG ENTRY, and that is what makes a hand write detectable at all. A bare
    # `git update-ref refs/heads/main <new>` that happens to be a FAST-FORWARD passes
    # check_main_advances_by_ancestry -- ancestry holds -- so the §245 rule "only merge_main
    # writes main" had no enforcement for that shape (44-37). Measured 2026-09-06: an unsigned
    # CAS and a bare hand write produce IDENTICAL reflog lines, both with an empty message, so
    # nothing downstream could tell them apart. All 60 entries in main's window were empty.
    # With -m the legitimate writer is the only one that leaves a mark, and the CAS still
    # refuses a stale old-value (verified: "cannot lock ref ... but expected <old>").
    if ! git -C "$MAIN" update-ref -m "merge_main: $1" refs/heads/main "$_new" "$_old" 2>/dev/null; then
      echo "merge_main: main moved while this ran (expected ${_old:0:8}, now" >&2
      echo "  $(git -C "$MAIN" rev-parse --short main)). Nothing landed. Re-run: the merge above" >&2
      echo "  is already in this worktree, so this is one more \`merge_main.sh $1\`." >&2
      exit 1
    fi
    echo "merge_main: main ${_old:0:8} -> ${_new:0:8}" >&2
    # ADVANCE THE INTEGRATION TREE'S WORKING FILES TO THE NEW main. The CAS moves a ref and
    # touches no working tree -- which is the point -- but that tree's FILES are executed by
    # everyone: .git/hooks/pre-commit is a symlink to ../../scripts/hooks/pre-commit, i.e. the
    # integration tree's checked-out copy, and every worktree shares one .git. Before the flip
    # that tree was always on main, so the executed hook was main's by coincidence. Detaching
    # froze it: measured 2026-09-06 (b0 found it, e1 filed #48), the executing hook was md5
    # 56e6cf2e = 4b364cac's blob while main's was f9b95b1d, 69 commits later. Any hook fix was
    # inert everywhere until someone re-checked-out that tree by hand.
    #
    # `checkout --detach` rather than a branch: the tree must stay detached, or the next CAS is
    # the silent one. Guarded on cleanliness for the same reason the detach above is -- nobody
    # commits there, so a clean tree loses nothing, and a dirty one is somebody's misplaced work.
    # A failure here is a WARNING, not a refusal: main has already moved and the merge succeeded,
    # so exiting nonzero would report a landed integration as failed. It names the command
    # instead, because a stale hook is silent and this line is the only notice anyone gets.
    if [ -z "$(git -C "$MAIN" status --porcelain 2>/dev/null)" ]; then
      if git -C "$MAIN" checkout --detach "$_new" -q 2>/dev/null; then
        echo "merge_main: integration tree advanced to ${_new:0:8} (it holds the hook everyone runs)" >&2
      else
        echo "merge_main: WARNING -- could not advance the integration tree; the pre-commit hook" >&2
        echo "  every worktree executes is that tree's copy and is now stale. Fix by hand:" >&2
        echo "    git -C $MAIN checkout --detach main" >&2
      fi
    else
      echo "merge_main: WARNING -- the integration tree is dirty, so its files were NOT advanced." >&2
      echo "  Every worktree executes ITS copy of scripts/hooks/pre-commit, which is now stale:" >&2
      git -C "$MAIN" status --porcelain 2>/dev/null | sed 's/^/    /' >&2
    fi
    # THE PUSH IS PART OF THE STEP. Measured 2026-09-05: main took 670 commits in 24 h against
    # 139 origin/main push events, so it advanced ~5x per push and every gap was a window where a
    # peer's fetch and the pod read a stale main. A FAILING push does NOT roll the ref back: the
    # commit is durable and reachable, and undoing it to match origin would discard work to fix a
    # delivery problem.
    if git -C "$MAIN" push -q origin main 2>/dev/null; then
      echo "merge_main: pushed origin/main" >&2
    else
      echo "merge_main: WARNING -- main is at ${_new:0:8} but the push FAILED. Main stays" >&2
      echo "  advanced; retry with: git -C $MAIN push origin main" >&2
    fi
    # THE POD PUSH ONLY PRINTS. pod_push.sh refuses any file differing from main and stamps
    # main's sha, but it also carries a running-.sh refusal and an emptyDir path for large files,
    # and a training run mid-flight is exactly when an automatic pod push is most dangerous and
    # least expected. Turning it on is a separate change with its own second reader.
    _scoped=$(git -C "$MAIN" diff --name-only "$_old" "$_new" -- train.py model.py run_ddp.sh \
              2>/dev/null | tr '\n' ' ')
    [ -n "$_scoped" ] && echo "merge_main: POD PUSH DUE for: $_scoped" >&2
    _release_claims "$1"
    exit 0
  fi
  # THE WAITER'S RULE: liveness, never age. A live holder is waited for however long it takes,
  # because the alternative is what happened on 2026-09-05 -- landing a merge inside someone's
  # quiet window and telling neither party. A deliberate hold is never removed even if its pid has
  # gone, because whoever set it up is coordinating something and the right answer is to ask them,
  # not to guess; the message names them so asking is possible.
  if _lock_is_dead; then
    if grep -q '^deliberate=yes' "$HOLDER" 2>/dev/null; then
      echo "merge_main: the lock is a DELIBERATE hold whose holder pid is gone -- not removing it." >&2
      sed 's/^/  /' "$HOLDER" >&2
      echo "  Ask that session, or clear it with 'scripts/merge_main.sh --release'." >&2
      exit 1
    fi
    echo "merge_main: removing a dead lock -- $_dead_why" >&2
    [ -f "$HOLDER" ] && sed 's/^/  /' "$HOLDER" >&2 || true
    rm -f "$HOLDER"; rmdir "$LOCK" 2>/dev/null || true
  fi
  sleep 1
done
echo "merge_main: could not take $LOCK in 120 s; it is held by a live process:" >&2
sed 's/^/  /' "$HOLDER" 2>/dev/null || echo "  (no holder file)" >&2
exit 1

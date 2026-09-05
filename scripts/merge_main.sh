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
  case "$_parent_cmd" in
    *timeout\ *)
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
    _out=$(timeout 20 bash "$0" $2 2>&1 || true)
    case "$_out" in
      *"REFUSING: this merge is running under"*) _got=refused;;
      *) _got=exempt;;
    esac
    if [ "$_got" != "$3" ]; then
      echo "  FAIL deadline guard, $1: want $3, got $_got" >&2; _fails=$((_fails + 1))
    else
      echo "  ok   deadline guard, $1 -> $_got"
    fi
  }
  _dcase "--selftest under a timeout is exempt" --selftest exempt
  _dcase "a branch merge under a timeout is refused" _no_such_branch_selftest refused

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
    # path-scoped FIRST -- those files are merge=union, so a row commits cleanly on its own and
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
      echo "  If these are ledger rows (runs/*.jsonl, merge=union), commit them ALONE first," >&2
      echo "  then re-run: git commit -m '<msg>' -- <the ledger file>" >&2
      echo "  A union-only commit is exempt from the behind-main refusal, so this works from" >&2
      echo "  a stale worktree. Do NOT git stash: .git/refs/stash is shared with every peer." >&2
      echo "  main is unmoved at ${_old:0:8}." >&2
      exit 1
    fi
    if ! git merge --no-edit main; then
      echo "merge_main: $1 conflicts with main. Resolve in THIS worktree, commit, retry --" >&2
      echo "  nothing was written to the integration tree and main is unmoved at ${_old:0:8}." >&2
      exit 1
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
    if ! git -C "$MAIN" update-ref refs/heads/main "$_new" "$_old" 2>/dev/null; then
      echo "merge_main: main moved while this ran (expected ${_old:0:8}, now" >&2
      echo "  $(git -C "$MAIN" rev-parse --short main)). Nothing landed. Re-run: the merge above" >&2
      echo "  is already in this worktree, so this is one more \`merge_main.sh $1\`." >&2
      exit 1
    fi
    echo "merge_main: main ${_old:0:8} -> ${_new:0:8}" >&2
    # THE PENDING OVERRIDE ROWS, written here rather than by the hook. A hook that appends to a
    # ledger mid-commit dirties the tree during the commit and refused the next merge; the hook
    # now records the event under its own git dir (invisible to `git status`, per-worktree) and
    # this step drains it through the CLI, which owns the accepted-kind list. Drained AFTER the
    # CAS and truncated only on success, so a kill between the two leaves the rows for next time.
    _pend="$(git rev-parse --git-dir)/aupai_pending_friction"
    if [ -s "$_pend" ]; then
      _n=0
      while IFS= read -r _row; do
        [ -n "$_row" ] || continue
        python3 "$_wt_self/scripts/harness.py" friction add --kind override --who "$1" \
          --blocked "commit from behind main on $1" --cause "$_row" --commit >/dev/null 2>&1 \
          && _n=$((_n + 1)) || true
      done < "$_pend"
      [ "$_n" -gt 0 ] && : > "$_pend"
      echo "merge_main: drained $_n pending override row(s) to friction" >&2
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

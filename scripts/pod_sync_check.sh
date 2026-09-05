#!/bin/bash
# Minimal drift check: sha256 of the files that execute on the pod (pretrain ->
# score flow), local working tree vs /work/aupai. The pod is not a git repo and
# code arrives by hand-push, so without this the two diverge silently.
# Exit 1 on any DIFF or MISSING. The file set comes from scripts/pod_drift.py
# (the same set the pod's manifest gate enforces) -- one scope, two directions.
#
# runs/*.jsonl are EXCLUDED, not compared and skipped quietly: they are union-merged
# ledgers the pod appends rows to, so they diverge by design and pod_push.sh never
# syncs them in either direction. Comparing them made 8 of 9 lines permanently red and
# buried the one real finding (datagen/build_starcoder_py.py, 123 lines on the pod
# against main's 114) -- fb only saw it by filtering runs/ out by hand. A gate that is
# always red is not a gate. pod_drift.py --check already reports them separately.
#
# A CHECKER THAT CANNOT RUN EXITS NONZERO AND NEVER PRINTS THE PASS STRING.
# MEASURED 2026-09-05 (b0 found it, 4c relayed): run pod-side, where there is no git,
# `pod_drift.py --list-scoped` dies inside `git ls-files` with CalledProcessError 128,
# FILES comes back empty, the for loop iterates zero times, fail stays 0, and this
# printed `pod in sync (0 files)` and exited 0 -- with the traceback above it on stderr,
# which a caller reading the exit code never sees. That is the loudest possible failure
# reported as the strongest possible pass. Three guards below, in the order the failure
# passed through them:
#   1. `set -o pipefail` and an explicit rc test on the FILES command, so a dead
#      subprocess is a refusal rather than an empty string.
#   2. a zero-file set is refused ANYWHERE, not only pod-side: comparing nothing always
#      succeeds, so "0 files" can never be evidence of sync. This is the general form of
#      the bug and it also covers a SCOPE that stops matching, a pathspec typo, and an
#      empty checkout.
#   3. the pod probe's own failure is a refusal too: an empty REMOTE means the sha256sum
#      never ran, which is indistinguishable from "every file is missing" if you only
#      read the loop's output.
# Each refusal says `cannot run: <reason>` and exits 2, so a caller can tell "could not
# check" from "checked and found drift" (exit 1) -- the distinction the old code erased.
set -o pipefail
cd "$(dirname "$0")/.."

die() { echo "cannot run: $1" >&2; exit 2; }

# --selftest: the three refusals, each asserted BY ITS OWN REASON, plus a control that must
# pass. Worlds are real -- a copy of this script in a throwaway tree -- so world 1 is exactly
# the pod-side invocation b0 hit.
#
# ASSERTING "exit 2" IS NOT ENOUGH, MEASURED 2026-09-05 by mutating each guard away. All three
# refusals exit 2, so a world that only checks the code cannot tell which guard fired, and two
# mutations survived on that:
#   - guard 1 removed: --list-scoped's TRACEBACK becomes $RAW, which is 72 words, so NFILES is
#     non-zero and guard 3 refuses instead ("the pod returned nothing for 72 file(s)"). Same
#     exit code, different guard, and the world could not see the difference.
#   - a version that refuses UNCONDITIONALLY passed every refusal world, because every one of
#     them expects a refusal. Three refusals are satisfiable by turning the checker off.
# So each world now matches its guard's own message, and the control asserts a real run is not
# refused. That is what makes the set non-vacuous.
if [ "${1:-}" = "--selftest" ]; then
  _self=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
  _fails=0
  _t=$(mktemp -d)
  trap 'rm -rf "$_t"' EXIT

  # W1: no git. `git ls-files` exits 128, --list-scoped dies, and the OLD code printed
  # "pod in sync (0 files)" and exited 0 with the traceback on stderr.
  mkdir -p "$_t/w1/scripts"
  cp "$_self" "$_t/w1/scripts/pod_sync_check.sh"
  cp "$(dirname "$_self")/pod_drift.py" "$_t/w1/scripts/pod_drift.py"
  _o=$(cd "$_t/w1/scripts" && bash pod_sync_check.sh 2>&1); _rc=$?
  if [ "$_rc" != "2" ]; then
    echo "  FAIL W1 no-git: exit $_rc, expected 2 -- 'could not check' must be distinguishable from 'checked, found drift' (1)"; _fails=1
  fi
  case "$_o" in
    *"pod in sync"*) echo "  FAIL W1 no-git: printed the pass string; this is the exact defect (b0, 2026-09-05)"; _fails=1 ;;
  esac
  # BY GUARD 1'S OWN REASON. Without this clause, removing guard 1 still passes: the traceback
  # becomes the file list and guard 3 refuses with a different message at the same exit code.
  case "$_o" in
    *"--list-scoped failed"*) ;;
    *) echo "  FAIL W1 no-git: refused, but not by the --list-scoped guard: $(printf '%s' "$_o" | grep -o 'cannot run:.*' | head -1)"; _fails=1 ;;
  esac

  # W2: a zero-file set with git working. The GENERAL form -- comparing nothing always
  # succeeds -- so it must be refused on any machine, not just pod-side.
  mkdir -p "$_t/w2/scripts"
  cp "$_self" "$_t/w2/scripts/pod_sync_check.sh"
  printf '#!/usr/bin/env python3\nimport sys\nsys.exit(0)\n' > "$_t/w2/scripts/pod_drift.py"
  _o2=$(cd "$_t/w2/scripts" && bash pod_sync_check.sh 2>&1); _rc2=$?
  if [ "$_rc2" != "2" ]; then
    echo "  FAIL W2 empty-set: exit $_rc2, expected 2 -- 0 files 'matching' is not evidence of sync"; _fails=1
  fi
  case "$_o2" in
    *"pod in sync"*) echo "  FAIL W2 empty-set: 0 files read as sync"; _fails=1 ;;
  esac
  case "$_o2" in
    *"file set is empty"*) ;;
    *) echo "  FAIL W2 empty-set: refused, but not by the empty-set guard: $(printf '%s' "$_o2" | grep -o 'cannot run:.*' | head -1)"; _fails=1 ;;
  esac

  # W3: ~/bin/pod ABSENT, with a real file set. Must refuse rather than report every file
  # MISSING -- that would be a claim about the pod's contents made without reading it.
  _o3=$(cd "$(dirname "$_self")/.." && HOME=/nonexistent-selftest-home bash scripts/pod_sync_check.sh 2>&1); _rc3=$?
  if [ "$_rc3" != "2" ]; then
    echo "  FAIL W3 no-pod-binary: exit $_rc3, expected 2"; _fails=1
  fi
  case "$_o3" in
    *"MISSING on pod"*) echo "  FAIL W3 no-pod-binary: claimed files are missing on a pod it never reached"; _fails=1 ;;
  esac
  # BY THE -x GUARD'S OWN MESSAGE, not by any pod-shaped refusal. Removing that guard still
  # refuses here -- the missing binary makes REMOTE empty and guard 3 fires -- but with the
  # WRONG DIAGNOSIS: "the pod returned nothing / the tunnel is down" for a machine that has no
  # wrapper at all. Pod-side that sends the reader after a healthy tunnel. Measured 2026-09-05:
  # this clause is the only thing that distinguishes the two.
  case "$_o3" in
    *"bin/pod is not executable"*) ;;
    *) echo "  FAIL W3 no-pod-binary: refused, but not by the ~/bin/pod guard -- a missing wrapper must not be reported as a dead tunnel: $(printf '%s' "$_o3" | grep -o 'cannot run:.*' | head -1)"; _fails=1 ;;
  esac

  # W4: ~/bin/pod PRESENT and answering with nothing -- a live wrapper, a dead container. This
  # is guard 3's own world, and the one W3 cannot reach.
  mkdir -p "$_t/fakehome/bin"
  printf '#!/bin/sh\nexit 0\n' > "$_t/fakehome/bin/pod"
  chmod +x "$_t/fakehome/bin/pod"
  _o5=$(cd "$(dirname "$_self")/.." && HOME="$_t/fakehome" bash scripts/pod_sync_check.sh 2>&1); _rc5=$?
  if [ "$_rc5" != "2" ]; then
    echo "  FAIL W4 pod-silent: exit $_rc5, expected 2 -- an empty answer means the probe failed, not that every file is missing"; _fails=1
  fi
  case "$_o5" in
    *"MISSING on pod"*) echo "  FAIL W4 pod-silent: reported files MISSING from an answer the pod never gave"; _fails=1 ;;
  esac
  case "$_o5" in
    *"pod returned nothing"*) ;;
    *) echo "  FAIL W4 pod-silent: refused, but not by the empty-answer guard: $(printf '%s' "$_o5" | grep -o 'cannot run:.*' | head -1)"; _fails=1 ;;
  esac

  # CONTROL, THE ONE WORLD THAT MUST NOT REFUSE. Three refusal worlds are all satisfied by a
  # script that refuses everything -- measured: that mutation passed all three. This is the
  # clause that fails it.
  #
  # Guards 1 and 2 only, deliberately: both are about THIS tree, so a refusal there means the
  # guards are too broad. Guard 3 is about the pod, and a genuinely dropped tunnel makes it
  # fire correctly -- asserting a full pass would make every commit touching this file a
  # liveness test for the pod, a fail-closed on an unrelated condition. It also keeps the cost
  # at 0.5s where a full run sha256s ~460 files and calls the pod, at 20s.
  _o4=$(cd "$(dirname "$_self")/.." && bash -c 'set -o pipefail
    if ! RAW=$(python3 scripts/pod_drift.py --list-scoped 2>&1); then echo "GUARD1_WOULD_REFUSE"; exit 0; fi
    N=$(printf "%s\n" "$RAW" | grep -v "^runs/" | wc -w | tr -d " ")
    [ "$N" -gt 0 ] || echo "GUARD2_WOULD_REFUSE"
    echo "N=$N"' 2>&1)
  case "$_o4" in
    *GUARD1_WOULD_REFUSE*|*GUARD2_WOULD_REFUSE*)
      echo "  FAIL control: on this checkout guards 1-2 would refuse a run that should proceed ($_o4)"; _fails=1 ;;
  esac
  case "$_o4" in
    *"N=0"*) echo "  FAIL control: read 0 in-scope files on a real checkout, so the control proves nothing"; _fails=1 ;;
  esac

  # W5-W8: THE CLASSIFIER, in a world where all four states exist by construction. The other
  # worlds drive the whole script and so need the real pod; this one drives the loop alone with
  # the three sha sources supplied by hand, which is the only way to build a pod that is behind
  # main and a pod that is diverged in the same run.
  #
  # WHY IT IS ASSERTED PER STATE, not as "some output appeared": the defect being fixed is that
  # one output covered three states. A world that checks only "it printed something" would pass
  # against the OLD code, which also printed something. Each state must produce ITS OWN label,
  # and the negative control -- a file that matches -- must produce none.
  _cls=$(cd "$_t" && mkdir -p w5 && cd w5 && \
    L_same=$(printf a | shasum -a 256 | cut -d' ' -f1) \
    L_new=$(printf b | shasum -a 256 | cut -d' ' -f1) \
    L_old=$(printf c | shasum -a 256 | cut -d' ' -f1) \
    L_odd=$(printf d | shasum -a 256 | cut -d' ' -f1) \
    bash -c '
      # local | pod | main  ->  expected label
      #   a   |  a  |  a    ->  (silent: in sync)          W5 negative control
      #   b   |  c  |  b    ->  POD-BEHIND-MAIN            W6
      #   b   |  d  |  c    ->  CONTENT-DIVERGED           W7  (pod matches neither)
      #   b   | --- |  ---  ->  UNTRACKED-BY-MANIFEST      W8  (absent both sides)
      #
      # W7 NEEDS local != main AS WELL. My first version wrote local==main there, which is the
      # DEFINITION of pod-behind-main, so it was labelled that and the world failed -- correctly.
      # Divergence means the pod holds a blob nobody has, which requires all three to differ.
      for row in "insync:$L_same:$L_same:$L_same" "behind:$L_new:$L_old:$L_new" \
                 "diverged:$L_new:$L_odd:$L_old" "unsent:$L_new::"; do
        IFS=: read -r name lh rh mh <<<"$row"
        if [ -z "$rh" ]; then
          if [ -z "$mh" ]; then echo "$name UNTRACKED-BY-MANIFEST"; else echo "$name MISSING"; fi
        elif [ "$lh" = "$rh" ]; then echo "$name CLEAN"
        elif [ -n "$mh" ] && [ "$rh" = "$mh" ]; then echo "$name local-ahead"
        elif [ -n "$mh" ] && [ "$lh" = "$mh" ]; then echo "$name POD-BEHIND-MAIN"
        else echo "$name CONTENT-DIVERGED"; fi
      done' 2>&1)
  for _want in "insync CLEAN" "behind POD-BEHIND-MAIN" "diverged CONTENT-DIVERGED" \
               "unsent UNTRACKED-BY-MANIFEST"; do
    case "$_cls" in
      *"$_want"*) ;;
      *) echo "  FAIL classifier: expected '$_want', got: $(printf '%s' "$_cls" | tr '\n' '; ')"; _fails=1 ;;
    esac
  done
  # AND THE STATES MUST BE DISTINCT. Four labels that are all the same string would satisfy
  # every clause above, which is exactly the defect: one output for several states.
  _n_uniq=$(printf '%s\n' "$_cls" | awk '{print $2}' | sort -u | wc -l | tr -d ' ')
  if [ "$_n_uniq" != "4" ]; then
    echo "  FAIL classifier: 4 states produced $_n_uniq distinct labels -- the report still conflates them"; _fails=1
  fi

  [ "$_fails" = "0" ] || { echo; echo "pod_sync_check selftest FAILED"; exit 1; }
  echo "  pod_sync_check: no-git, empty-set and no-pod each refuse by their OWN guard's reason at exit 2 and never print the pass string; guards 1-2 pass this checkout through; and the four sync states (clean, pod-behind-main, content-diverged, never-manifested) produce four DISTINCT labels"
  exit 0
fi

# GUARD 1: --list-scoped must SUCCEED. Captured separately from the pipeline so its exit
# code survives; `$(a | b)` would report b's.
if ! RAW=$(python3 scripts/pod_drift.py --list-scoped 2>&1); then
  die "scripts/pod_drift.py --list-scoped failed (this needs a git checkout; run it on the laptop, not the pod). Its output: $(printf '%s' "$RAW" | tail -2 | tr '\n' ' ')"
fi
FILES=$(printf '%s\n' "$RAW" | grep -v '^runs/' | tr '\n' ' ')
NFILES=$(printf '%s' "$FILES" | wc -w | tr -d ' ')

# GUARD 2: nothing to compare is not a pass, on any machine.
[ "$NFILES" -gt 0 ] || die "the scoped file set is empty, so there is nothing to compare -- 0 files always 'matches'. Check pod_drift.py's SCOPE and that this is a git checkout"

[ -x "$HOME/bin/pod" ] || die "$HOME/bin/pod is not executable here; this check reads the pod through it (laptop only)"
REMOTE=$(~/bin/pod "cd /work/aupai && sha256sum $FILES 2>/dev/null" < /dev/null)

# GUARD 3: an empty remote answer means the probe failed, not that every file is missing.
[ -n "$REMOTE" ] || die "the pod returned nothing for $NFILES file(s) -- the tunnel or the container is down, and reporting $NFILES MISSING would be a claim about the pod's contents we cannot make"

# MAIN'S OWN SHAS, THE THIRD VALUE. Local-vs-pod has one output for three states, which is
# what made every run end in a hand-rolled md5 loop (84 and 3b, 2026-09-05, independently):
# "the pod is behind main" and "the pod holds something nobody has" both printed DIFF, and so
# did "my worktree is ahead", which is not drift at all. Two shas cannot separate three states;
# main's is the missing one.
#
# READ FROM git, NOT FROM data/pod_head_manifest.txt: that file is generated by pod_push and
# describes whatever ref the last --all used, so trusting it here would make this check inherit
# the staleness it exists to report. `git cat-file --batch` hashes main's blobs directly, one
# process for all ~460 paths (a `git show` per file took 13s; this is under 1s).
MAINSHAS=$(mktemp) || die "cannot create a temporary file"
trap 'rm -f "$MAINSHAS"' EXIT
if git rev-parse --verify -q main >/dev/null 2>&1; then
  for f in $FILES; do
    s=$(git cat-file blob "main:$f" 2>/dev/null | shasum -a 256 | cut -d' ' -f1) || continue
    [ -n "$s" ] && printf '%s %s\n' "$s" "$f"
  done
else
  # No main here (a bare fixture, or the pod). The check still runs; it just cannot tell
  # POD-BEHIND-MAIN from CONTENT-DIVERGED, and says so rather than guessing.
  echo "note: no 'main' ref here, so a difference cannot be attributed -- reporting DIFF only" >&2
fi \
> "$MAINSHAS" || die "cannot write a temporary file"

fail=0
diverged=0
behind=0
for f in $FILES; do
  lh=$(shasum -a 256 "$f" | cut -d' ' -f1)
  rh=$(printf '%s\n' "$REMOTE" | awk -v f="$f" '$2==f{print $1}')
  mh=$(awk -v f="$f" '$2==f{print $1}' "$MAINSHAS")
  if [ -z "$rh" ]; then
    if [ -z "$mh" ]; then
      # NOT IN main's MANIFEST AND NOT ON THE POD. This is the harness.py shape and it needs
      # its own line: a path the manifest omits is never offered to pod_push's per-file gate,
      # so it is SKIPPED rather than refused -- silently, exit 0. Falling through to MISSING
      # would say the pod lost a file, when in fact nothing ever tried to send it.
      echo "UNTRACKED-BY-MANIFEST: $f  (absent from the pod AND from main's manifest; pod_push would skip it, not refuse it)"
    else
      echo "MISSING on pod: $f"
    fi
    fail=1
  elif [ "$lh" = "$rh" ]; then
    :
  elif [ -n "$mh" ] && [ "$rh" = "$mh" ]; then
    # The pod matches MAIN; this worktree is what differs. Not drift.
    echo "local-ahead: $f  (pod == main; your worktree differs)"
  elif [ -n "$mh" ] && [ "$lh" = "$mh" ]; then
    echo "POD-BEHIND-MAIN: $f  (your tree == main, pod has an older blob -- push it)"
    behind=1
    fail=1
  else
    echo "CONTENT-DIVERGED: $f  (pod matches neither main nor this worktree)"
    diverged=1
    fail=1
  fi
done
if [ $fail -eq 0 ]; then
  echo "pod in sync ($NFILES files)"
else
  echo "-- POD-BEHIND-MAIN is fixed by pod_push.sh --all from a worktree; CONTENT-DIVERGED is not."
fi
exit $fail

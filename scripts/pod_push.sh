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
export PODPUT_TRACKED_OK=1
cd "$(dirname "$0")/.."

# Stamp WHAT is on the pod and from WHERE. The pod has no git and no route back to
# this machine, so it cannot ask whether main has moved -- run_ddp.sh can only read a
# stamp somebody left. Called after --check, so a stamp means the manifest gate agreed.
#
# Only a whole-manifest push may claim a sha. A named-file push leaves every other file
# at whatever it was, so it CLEARS the stamp instead: the pod is then a mix of one sha's
# tree and one file from another, and the honest state is "unknown". The failure this
# guards is a three-day run on code somebody pushed one file into.
stamp_sync() {
  if [ "$1" = all ]; then
    local head_sha dirty main_sha
    head_sha=$(git rev-parse HEAD)
    # THE SHA MUST BE MAIN'S, NOT THIS WORKTREE'S BRANCH TIP (de-14). Every session pushes
    # from its own worktree, so `rev-parse HEAD` is that branch's tip: measured 2026-09-03,
    # this tree's HEAD was 1b85dd0c while main was 69c8bd87. The pod runs main -- push_one
    # already refuses any file that differs from main -- so a stamp naming a branch tip
    # describes a tree that does not exist anywhere: main's file contents under a sha only
    # one laptop has. run_ddp.sh then compares against a value nobody else can resolve.
    main_sha=$(git rev-parse main 2>/dev/null || echo "")
    if [ -n "$main_sha" ] && [ "$head_sha" != "$main_sha" ]; then
      if git merge-base --is-ancestor "$head_sha" "$main_sha" 2>/dev/null; then
        # Behind main: the files pushed are main's (push_one enforced that), so main's sha
        # is what describes them.
        echo "pod sync stamp: using main ($main_sha) not this branch tip ($head_sha)"
        head_sha=$main_sha
      else
        echo "refusing to stamp: HEAD ($head_sha) is not reachable from main ($main_sha)." >&2
        echo "  The pod runs main. A stamp naming an unmerged commit describes a tree that" >&2
        echo "  exists on no branch, and run_ddp.sh cannot verify it. Merge into main first." >&2
        return 1
      fi
    fi
    dirty=$(git status --porcelain -- $(awk '{print $2}' data/pod_head_manifest.txt \
            | grep -v '^runs/') 2>/dev/null | wc -l | tr -d ' ')
    ~/bin/pod "cd /work/aupai && printf '%s %s %s\n' $head_sha $dirty $(date -u +%Y-%m-%dT%H:%M:%SZ) > data/pod_synced_head" < /dev/null
    echo "pod sync stamp: $head_sha (dirty=$dirty)"
  else
    ~/bin/pod "cd /work/aupai && rm -f data/pod_synced_head" < /dev/null
    echo "pod sync stamp CLEARED (partial push) -- run '$0 --all' before a training launch"
  fi
}

ALL=0
if [ "${1:-}" = "--all" ]; then ALL=1; shift; fi
[ $# -ge 1 ] || [ $ALL -eq 1 ] || { echo "usage: $0 [--all] <file>..."; echo "       $0 --all   (sync the whole manifest: push changed, delete manifest-left)"; exit 2; }

find_emptydir() {
  [ -n "${EMPTYPATH:-}" ] && return
  EMPTYPATH=$(tn exec "for d in /var/lib/kubelet/pods/*/volumes/kubernetes.io~empty-dir/work; do [ -d \"\$d/aupai\" ] && echo \"\$d\" && break; done" 2>/dev/null | head -1)
  if [ -z "$EMPTYPATH" ]; then
    echo "pod_push: cannot find /work emptyDir host path (is the pod running?)" >&2
    exit 1
  fi
}

# Push one committed, main-reachable file. Large files (>100KB gzip+base64) bypass
# podput's argv limit via the container's emptyDir host path.
# True when a script of this name is executing on the pod. ps with STAT Z filtered,
# not pgrep -f: a ZOMBIE keeps its argv, and run_ddp.sh had three of them beside the
# one live process, so pgrep would match the dead ones forever and make the guard a
# permanent refusal -- the trap _drop_zombies exists for.
#
# MATCHED ON THE POD PATH, NOT THE BASENAME. The basename version refused a push of
# scripts/e1_27_sweep.sh because an UNRELATED /tmp/e1_27_sweep.sh was running (e1,
# 2026-09-03): same basename, different file, and overwriting the tracked one could not
# have corrupted the running one. The hazard this guard exists for is byte-offset
# corruption of the file being written, which is a property of the PATH -- so the test
# is the path podput will write to. A bare `scripts/foo.sh` in someone's argv still
# matches, because /work/aupai/scripts/foo.sh contains it as a suffix; that direction
# of looseness is the safe one (a false refusal, never a false permit).
running_on_pod() {
  if [ -n "${POD_PUSH_ALLOW_RUNNING_SH:-}" ]; then
    # THE OVERRIDE IS NOW CHECKED, NOT TRUSTED. It used to return 1 unconditionally: the
    # operator asserted the edit was safe and nothing recomputed it. The safety is a
    # property of the DIFF, not of the flag -- on 2026-09-04 an edit to the scoring block
    # was pushed under this override while two runs were mid-script and it WAS safe, because
    # every added byte landed after byte 4391, which both shells had already read. The same
    # flag on an edit touching byte 4000 would have been unsafe with no warning at all.
    # pod_sh_offset.py reads each live shell's offset from /proc/<pid>/fdinfo and refuses
    # unless every differing byte is at or after the earliest of them.
    if python3 scripts/pod_sh_offset.py --check "$1"; then
      return 1   # verified safe: allow the push
    fi
    echo "  POD_PUSH_ALLOW_RUNNING_SH is set, but the offset check above REFUSED." >&2
    echo "  The flag asserts the edit is safe; that assertion is now recomputed and it does" >&2
    echo "  not hold. Wait for the run to finish." >&2
    return 0     # treated as running: refuse
  fi
  ~/bin/pod "ps -eo stat,args | awk '\$1 !~ /^Z/' | grep -v grep | grep -qF '$1'" \
    >/dev/null 2>&1
}

push_one() {
  local f="$1"
  if [ -n "$(git status --porcelain -- "$f")" ]; then
    echo "refusing: $f has uncommitted changes -- commit or stash it first"
    exit 1
  fi
  # `|| true`: under `set -e` a failing command substitution kills the script HERE,
  # before the refusal below can print. That is what happened on 2026-09-01 -- a
  # branch-only file produced exit 128 and ZERO output, which is indistinguishable from
  # a push that worked, and three GPU cells were nearly launched against a script that
  # was never delivered. A refusal that produces no evidence is not a refusal, it is a
  # silence. stderr is kept rather than discarded so git's own reason survives.
  want=$(git rev-parse "main:$f" 2>/dev/null) || true
  if [ -z "$want" ]; then
    echo "refusing: $f is not in main -- merge your branch first (the pod runs main)" >&2
    exit 1
  fi
  if [ "$(git hash-object "$f")" != "$want" ]; then
    echo "refusing: $f differs from main -- merge your branch first (the pod runs main, not your branch)"
    exit 1
  fi
  # A .sh THAT IS RUNNING RIGHT NOW must not be overwritten. podput writes with `>`,
  # which truncates the SAME inode, and bash reads a script incrementally by byte
  # offset -- so a running shell resumes at its old offset inside the new bytes and
  # executes whatever now sits there. Demonstrated, not assumed: replacing a sleeping
  # script mid-run made it print the REPLACEMENT's lines. Nearly overwrote run_ddp.sh
  # while it was driving the lr probe's second arm, 40 minutes into a 7-card run
  # (2026-09-02). --all pre-checks the whole batch; this covers a named-file push.
  case "$f" in
    *.sh)
      if running_on_pod "$f"; then
        echo "REFUSING: $f is executing on the pod right now. podput truncates in place and" >&2
        echo "  bash reads scripts by byte offset, so overwriting it can make the running" >&2
        echo "  shell execute a corrupted position. Wait for it to finish, or override with" >&2
        echo "  POD_PUSH_ALLOW_RUNNING_SH=1 if you know nothing is mid-script." >&2
        exit 1
      fi
      ;;
  esac
  local b64_size
  b64_size=$(gzip -9c "$f" | base64 | tr -d '\n' | wc -c | tr -d ' ')
  if [ "$b64_size" -le 100000 ]; then
    ~/bin/podput "$f" "/work/aupai/$f" || { echo "REFUSING: podput failed for $f; nothing after it shipped" >&2; exit 1; }
  else
    find_emptydir
    echo "pod_push: $f ($b64_size b64 chars) via emptyDir path" >&2
    tn push "$f" "$EMPTYPATH/aupai/$f"
  fi
  # RESTORE THE MODE GIT RECORDS. Neither transport carries it: podput pipes into `> $R`
  # and tn push writes content, so the pod file gets whatever the remote umask says --
  # 644. Every .sh that git marks 100755 landed non-executable, and a pod call naming
  # the script path then dies on "Permission denied" (b0-17's first launch, 2026-09-02;
  # 16 tracked .sh were in that state, measured, not the 5 the task estimated).
  #
  # The mode comes from `git ls-files -s` rather than from the local file's stat: the
  # local bit can be anything (a fresh clone, a copy through a filesystem with no exec
  # bit), and what the pod should run is what main records. Only the exec bit is
  # honoured -- git tracks exactly two modes for blobs, 100644 and 100755.
  local gitmode
  gitmode=$(git ls-files -s -- "$f" | awk '{print $1}')
  case "$gitmode" in
    100755) ~/bin/pod "chmod 755 /work/aupai/$f" >/dev/null 2>&1 || {
              echo "WARNING: $f pushed but chmod 755 failed -- it will not be executable" >&2; } ;;
    100644) ;;  # nothing to do: the umask already gives a non-executable file
    "")     echo "WARNING: $f has no git mode (untracked?) -- mode not set on the pod" >&2 ;;
    *)      echo "WARNING: $f has unexpected git mode $gitmode -- mode not set on the pod" >&2 ;;
  esac
}

if [ $ALL -eq 1 ]; then
  # Whole-tree sync after a layout-changing merge. The manifest must describe HEAD
  # and be reachable from main; the pod's last manifest defines the delete set, so
  # throwaway probes (never in any manifest) are untouched.
  [ $# -eq 0 ] || { echo "pod_push --all takes no file arguments" >&2; exit 2; }
  # Generate from HEAD; not tracked, so there is nothing to be stale against and no
  # main-reachability gate to apply to it (shape A). The old `push_one manifest` call
  # existed only to run that gate on a tracked file.
  python3 scripts/pod_drift.py --write >/dev/null
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  ~/bin/pod cat /work/aupai/data/pod_head_manifest.txt > "$tmp/old" 2>/dev/null || true
  # Pod shas for every new-manifest path, one batch. Missing files error to stderr
  # and are simply absent from stdout -> pushed.
  # Space-separated: a newline inside the quoted command becomes a command
  # separator in the pod's bash -lc, so only the first path would reach sha256sum.
  paths=$(awk '{print $2}' data/pod_head_manifest.txt | grep -v '^runs/' | tr '\n' ' ')
  ~/bin/pod "cd /work/aupai && sha256sum $paths 2>/dev/null" > "$tmp/pod" || true
  pushes=(); dels=(); blocked=()
  while read -r op p; do
    [ -n "$op" ] || continue
    if [ "$op" = push ]; then pushes+=("$p"); else dels+=("$p"); fi
  done < <(python3 scripts/pod_drift.py --plan-sync "$tmp/old" "$tmp/pod")
  echo "pod_push --all: ${#pushes[@]} push, ${#dels[@]} delete"
  # CHECK THE WHOLE BATCH BEFORE PUSHING ANY OF IT. push_one refuses a .sh that is
  # executing, and refusing mid-loop leaves the pod HALF UPDATED with a stale sync
  # stamp -- measured: the first attempt pushed 3 files, then refused on run_ddp.sh,
  # and the pod sat at 3 drifted files. The drift check and run_ddp's own stamp gate
  # both caught it, so nothing unsafe shipped, but a partial push is a worse state
  # than either pushing or not pushing. All-or-nothing.
  for p in ${pushes[@]+"${pushes[@]}"}; do
    case "$p" in *.sh) running_on_pod "$p" && blocked+=("$p") ;; esac
  done
  if [ ${#blocked[@]} -gt 0 ]; then
    echo "REFUSING the whole push: ${#blocked[@]} script(s) executing on the pod: ${blocked[*]}" >&2
    echo "  A running .sh is a cursor into a file, not a file: podput truncates in place" >&2
    echo "  and bash reads by byte offset, so the live shell would resume inside new bytes." >&2
    echo "  Nothing was pushed -- a partial sync is worse than none. Wait, or set" >&2
    echo "  POD_PUSH_ALLOW_RUNNING_SH=1." >&2
    exit 1
  fi
  for p in ${pushes[@]+"${pushes[@]}"}; do push_one "$p"; done
  if [ ${#dels[@]} -gt 0 ]; then
    ~/bin/pod "cd /work/aupai && rm -f -- ${dels[*]}"
    echo "deleted: ${dels[*]}"
  fi
  # The manifest last: it must describe exactly what landed.
  ~/bin/podput data/pod_head_manifest.txt /work/aupai/data/pod_head_manifest.txt
  ~/bin/pod "cd /work/aupai && python3 scripts/pod_drift.py --check" < /dev/null
  stamp_sync all
  exit 0
fi

# GENERATE the manifest from the HEAD being pushed. It is not tracked (shape A, 6e ruling
# 2026-09-04): a file that is a pure function of HEAD does not belong in the tree, and
# tracking it was the source of every "local changes would be overwritten" abort -- the
# pre-commit hook regenerated it on every commit, so any two branches that both committed
# collided on it, 4 of 15 rows in runs/friction.jsonl. The tree check above already refuses
# a dirty push, so HEAD is what these files are.
python3 scripts/pod_drift.py --write >/dev/null

for f in "$@"; do
  push_one "$f"
done

# VERIFY WHAT WAS PUSHED, PER PATH, before the manifest gate below. That gate compares the
# pod against data/pod_head_manifest.txt and is silent about every path the manifest does
# not list -- 398 of 815 tracked files are in it, so docs/lessons/, docs/audits/ and
# scripts/pod_sync_check.sh (explicitly out of SCOPE) can be pushed by name with nothing
# afterwards reading their bytes. `pod_sync_check` reported `4 UNREGISTERED .py not in
# manifest` on 2026-09-03, which is this same gap from the other side.
podshas=$(mktemp)
~/bin/pod "cd /work/aupai && sha256sum $* 2>/dev/null" < /dev/null > "$podshas" || true
python3 scripts/pod_verify_landed.py "$podshas" "$@"
rm -f "$podshas"

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
stamp_sync partial

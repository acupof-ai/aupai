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
    local head_sha dirty
    head_sha=$(git rev-parse HEAD)
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
  # (2026-09-02). Refuses rather than skips: a silent skip means the pod keeps old
  # code while the push reports success, which is the drift this script exists to stop.
  case "$f" in
    *.sh)
      # ps, not pgrep: a ZOMBIE keeps its argv, and run_ddp.sh had three of them beside
      # the one live process. pgrep -f matches the dead ones forever, which would turn
      # this guard into a permanent refusal -- the same trap _drop_zombies exists for.
      if [ -z "${POD_PUSH_ALLOW_RUNNING_SH:-}" ] && ~/bin/pod \
           "ps -eo stat,args | awk '\$1 !~ /^Z/' | grep -v grep | grep -q '$(basename "$f")'" \
           >/dev/null 2>&1; then
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
    ~/bin/podput "$f" "/work/aupai/$f"
  else
    find_emptydir
    echo "pod_push: $f ($b64_size b64 chars) via emptyDir path" >&2
    tn push "$f" "$EMPTYPATH/aupai/$f"
  fi
}

if [ $ALL -eq 1 ]; then
  # Whole-tree sync after a layout-changing merge. The manifest must describe HEAD
  # and be reachable from main; the pod's last manifest defines the delete set, so
  # throwaway probes (never in any manifest) are untouched.
  [ $# -eq 0 ] || { echo "pod_push --all takes no file arguments" >&2; exit 2; }
  if ! python3 scripts/pod_drift.py --check-head >/dev/null 2>&1; then
    echo "pod_push --all: manifest stale vs HEAD -- regenerate, commit, merge, then re-run" >&2
    exit 1
  fi
  push_one data/pod_head_manifest.txt >/dev/null  # main-reachability gate on the manifest itself
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  ~/bin/pod cat /work/aupai/data/pod_head_manifest.txt > "$tmp/old" 2>/dev/null || true
  # Pod shas for every new-manifest path, one batch. Missing files error to stderr
  # and are simply absent from stdout -> pushed.
  # Space-separated: a newline inside the quoted command becomes a command
  # separator in the pod's bash -lc, so only the first path would reach sha256sum.
  paths=$(awk '{print $2}' data/pod_head_manifest.txt | grep -v '^runs/' | tr '\n' ' ')
  ~/bin/pod "cd /work/aupai && sha256sum $paths 2>/dev/null" > "$tmp/pod" || true
  pushes=(); dels=()
  while read -r op p; do
    [ -n "$op" ] || continue
    if [ "$op" = push ]; then pushes+=("$p"); else dels+=("$p"); fi
  done < <(python3 scripts/pod_drift.py --plan-sync "$tmp/old" "$tmp/pod")
  echo "pod_push --all: ${#pushes[@]} push, ${#dels[@]} delete"
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

for f in "$@"; do
  push_one "$f"
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
stamp_sync partial

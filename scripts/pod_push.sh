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


# WHICH REF IS `main`. `main` is a LOCAL ref, and since the 2026-09-07 PR flip it is a
# STALE CACHE of the shared branch: `gh pr merge` advances origin/main on GitHub and
# touches nothing in this clone, so a file byte-identical to main's tip reads as differing
# from it. Measured (3b, 2026-09-08): local main held blob 83af0af0 for a file whose
# origin/main copy and worktree copy were both 7c4b30bc, and push_one refused with
# "differs from main -- merge your branch first" on a branch that was ALREADY merged. A
# refusal naming a remedy that cannot work is the expensive shape, because the operator
# does the remedy, gets the same refusal, and concludes the file is the problem.
#
# THE FALSE PERMIT IS THE WORSE HALF, and it was nobody's report: with local main stale, a
# file matching the OLD blob and not the new one PASSES this gate, and the pod then runs
# bytes main no longer holds. Both directions are one defect -- the gate compares against
# a cache of main rather than main -- so both are fixed by resolving the ref, not by
# rewording the refusal.
#
# Three outcomes, each stated rather than silent:
#   fetch works              -> origin/main, authoritative
#   fetch fails, cached ref  -> origin/main anyway. A cache of the remote is never BEHIND
#                               local main in the case this function exists for, so it is
#                               strictly better evidence; WARN that it may be behind the
#                               real main.
#   no origin/main at all    -> local main. That is what a fixture and a standalone clone
#                               have, and such a tree has no remote to be stale against.
#
# Resolved ONCE into MAIN_REF at the bottom of this block: push_one runs per file, and a
# fetch per file would put a network round trip inside the push loop.
resolve_main_ref() {
  if git fetch --quiet origin +refs/heads/main:refs/remotes/origin/main 2>/dev/null; then
    echo origin/main
    return 0
  fi
  if git rev-parse --verify -q refs/remotes/origin/main >/dev/null; then
    echo "WARNING: could not fetch origin -- comparing against the CACHED origin/main" >&2
    echo "  ($(git rev-parse --short refs/remotes/origin/main)), which may be behind the" >&2
    echo "  real main. A refusal below may be this staleness rather than the file." >&2
    echo origin/main
    return 0
  fi
  echo "WARNING: no origin/main in this tree -- comparing against local main, which a" >&2
  echo "  merge on GitHub does not advance." >&2
  echo main
}


# The sha a whole-manifest stamp may claim, or a refusal — BEFORE anything ships.
# Same test stamp_sync used to run at the end, hoisted for the reason the rule exists:
# a `refusing:` line printed after every file and the manifest have landed names a
# condition that no longer prevents anything, and "only a refusing: line means nothing
# shipped" was then false on this path. Echoes the sha; refuses nonzero.
resolve_stamp_sha() {
  local head_sha main_sha
  head_sha=$(git rev-parse HEAD)
  # THE SHA MUST BE MAIN'S, NOT THIS WORKTREE'S BRANCH TIP (de-14). Every session pushes
  # from its own worktree, so `rev-parse HEAD` is that branch's tip: measured 2026-09-03,
  # this tree's HEAD was 1b85dd0c while main was 69c8bd87. The pod runs main -- push_one
  # already refuses any file that differs from main -- so a stamp naming a branch tip
  # describes a tree that does not exist anywhere: main's file contents under a sha only
  # one laptop has. run_ddp.sh then compares against a value nobody else can resolve.
  main_sha=$(git rev-parse "$MAIN_REF" 2>/dev/null || echo "")
  if [ -n "$main_sha" ] && [ "$head_sha" != "$main_sha" ]; then
    if git merge-base --is-ancestor "$head_sha" "$main_sha" 2>/dev/null; then
      # Behind main: the files pushed are main's (push_one enforced that), so main's sha
      # is what describes them.
      echo "pod sync stamp: using $MAIN_REF ($main_sha) not this branch tip ($head_sha)" >&2
      head_sha=$main_sha
    else
      echo "refusing: HEAD ($head_sha) is not reachable from $MAIN_REF ($main_sha)." >&2
      echo "  The pod runs main. A stamp naming an unmerged commit describes a tree that" >&2
      echo "  exists on no branch, and run_ddp.sh cannot verify it. Merge into main first." >&2
      echo "  Nothing was pushed." >&2
      return 1
    fi
  fi
  echo "$head_sha"
}

# Stamp WHAT is on the pod and from WHERE. The pod has no git and no route back to
# this machine, so it cannot ask whether main has moved -- run_ddp.sh can only read a
# stamp somebody left. Called after --check, so a stamp means the manifest gate agreed.
#
# A whole-manifest push claims its sha directly. A named-file push USED TO CLEAR the
# stamp unconditionally, on the reasoning that the pod is then "one sha's tree plus one
# file from another" and the honest state is unknown. That reasoning is right about the
# risk and wrong about the evidence: the pod's own drift gate has just compared every
# SCOPE file against the manifest, so after a partial push whether the pod is a mix is
# a MEASURED question, not an assumed one. Cleared three times in one hour on
# 2026-09-06 (3b, 44, the b0 resume) by pushes that left the pod exactly at main, and
# launch_gate then refused on a stamp describing no divergence.
#
# So: recompute. If the drift gate says every manifest file matches, the pod IS the
# manifest's tree and the stamp is that sha. If anything differs, clear it -- same
# outcome as before, now for a reason that was checked. The guarded failure (a
# three-day run on code somebody pushed one file into) is unchanged: that pod has a
# drifted file, so the gate says so and the stamp goes.
stamp_sync() {
  if [ "$1" = all ]; then
    local head_sha dirty
    head_sha=$2  # resolved and refused-on before the first push
    dirty=$(git status --porcelain -- $(awk '{print $2}' data/pod_head_manifest.txt \
            | grep -v '^runs/') 2>/dev/null | wc -l | tr -d ' ')
    ~/bin/pod "cd /work/aupai && printf '%s %s %s\n' $head_sha $dirty $(date -u +%Y-%m-%dT%H:%M:%SZ) > data/pod_synced_head" < /dev/null
    echo "pod sync stamp: $head_sha (dirty=$dirty)"
  elif [ "$1" = partial ] && [ "${2:-}" = clean ] && [ -n "${3:-}" ]; then
    # The drift gate above exited 0: every manifest file on the pod matches this tree's,
    # and this tree is main (resolve_stamp_sha refused otherwise). Nothing is mixed.
    ~/bin/pod "cd /work/aupai && printf '%s %s %s\n' $3 0 $(date -u +%Y-%m-%dT%H:%M:%SZ) > data/pod_synced_head" < /dev/null
    echo "pod sync stamp: $3 (partial push, but every manifest file matches -- recomputed, not cleared)"
  else
    ~/bin/pod "cd /work/aupai && rm -f data/pod_synced_head" < /dev/null
    echo "pod sync stamp CLEARED -- the pod does not match this tree; run '$0 --all'"
  fi
}

# THE MAIN-REACHABILITY GATE, its own function so the selftest below can drive it in a
# real fixture repo. Inlined in push_one it was unreachable from any test: push_one goes on
# to call podput and the pod, so exercising it meant faking the transport to measure a
# comparison that touches neither. Exits nonzero on refusal, as it did inline -- callers
# run it in a subshell only in the selftest.
assert_matches_main_ref() {
  local f="$1" want
  # `|| true`: under `set -e` a failing command substitution kills the script HERE,
  # before the refusal below can print. That is what happened on 2026-09-01 -- a
  # branch-only file produced exit 128 and ZERO output, which is indistinguishable from
  # a push that worked, and three GPU cells were nearly launched against a script that
  # was never delivered. A refusal that produces no evidence is not a refusal, it is a
  # silence. stderr is kept rather than discarded so git's own reason survives.
  want=$(git rev-parse "$MAIN_REF:$f" 2>/dev/null) || true
  if [ -z "$want" ]; then
    echo "refusing: $f is not in $MAIN_REF -- merge your branch first (the pod runs main)" >&2
    exit 1
  fi
  if [ "$(git hash-object "$f")" != "$want" ]; then
    echo "refusing: $f differs from $MAIN_REF -- merge your branch first (the pod runs main, not your branch)"
    # THE REMEDY MUST BE ONE THAT WORKS. When MAIN_REF is the local `main`, "merge your
    # branch first" is what the operator already did -- the branch is merged on GitHub and
    # this clone has not heard about it -- so the line above sends them around the loop
    # again. Name the staleness instead, and only when it is actually the case.
    if [ "$MAIN_REF" = main ] && git rev-parse --verify -q refs/remotes/origin/main >/dev/null \
       && [ "$(git rev-parse main)" != "$(git rev-parse refs/remotes/origin/main)" ]; then
      echo "  NOTE: comparing against LOCAL main ($(git rev-parse --short main)), which differs" >&2
      echo "  from origin/main ($(git rev-parse --short refs/remotes/origin/main)). A PR merged" >&2
      echo "  with gh advances origin and not this ref, so merging again will not change this." >&2
      echo "  Run: git fetch origin main" >&2
    fi
    exit 1
  fi
}

# --selftest: drive stamp_sync's three outcomes against a FAKE pod, so the branch that
# decides whether a launch is allowed has a check that runs on this machine. It must come
# before the tree/refusal gates below, which talk to the real pod.
if [ "${1:-}" = "--selftest" ]; then
  # STRIP THE INHERITED GIT ENVIRONMENT FIRST. The pre-commit hook runs this file with
  # GIT_DIR and GIT_INDEX_FILE exported, and those override every `cd` the fixture does:
  # measured under `GIT_DIR=.../aupai-de/.git`, `git init -q -b main "$_g/up"` re-initialised
  # THIS repository's git dir instead of the fixture's, so the clones were never created and
  # cases F/G/H failed with rc=9 `cd: no such file or directory`. Three ways this could have
  # gone unnoticed: it only reproduces at commit time, the fixture builder's output is sent
  # to /dev/null, and a `cd` failure in a subshell reads as a broken world rather than as
  # leaked state. HOME is not touched here -- the fake-pod cases below set it deliberately.
  unset GIT_DIR GIT_INDEX_FILE GIT_WORK_TREE GIT_OBJECT_DIRECTORY GIT_COMMON_DIR GIT_PREFIX
  _d=$(mktemp -d); mkdir -p "$_d/bin" "$_d/pod/data"
  cat > "$_d/bin/pod" <<'FAKEPODEOF'
#!/bin/bash
cd "$FAKEPOD" || exit 1
cmd="$1"; cmd="${cmd#cd /work/aupai && }"
eval "$cmd"
FAKEPODEOF
  chmod +x "$_d/bin/pod"
  export FAKEPOD="$_d/pod" HOME="$_d"
  _stamp="$_d/pod/data/pod_synced_head"
  _fails=0
  # A: every manifest file matched and the sha resolved -- the stamp is RECOMPUTED, not
  # cleared. This is the case that cost three cleared stamps in one hour on 2026-09-06.
  rm -f "$_stamp"
  stamp_sync partial clean deadbeefdeadbeefdeadbeefdeadbeefdeadbeef >/dev/null
  if ! grep -q deadbeef "$_stamp" 2>/dev/null; then
    echo "FAIL A: a clean partial push did not stamp; launch_gate would refuse on no divergence" >&2
    _fails=1
  fi
  # B: the drift gate found something -- clear, same as before. Without this the fix would
  # be "always stamp", which passes A and stamps a pod that really is a mix.
  stamp_sync partial >/dev/null
  if [ -f "$_stamp" ]; then
    echo "FAIL B: a drifted partial push left a stamp behind" >&2
    _fails=1
  fi
  # C: clean but NO sha (resolve_stamp_sha refused -- an unmerged HEAD). Must fall through
  # to the clear, never stamp an empty sha, which run_ddp.sh cannot resolve.
  printf 'pre-existing\n' > "$_stamp"
  stamp_sync partial clean >/dev/null
  if [ -f "$_stamp" ]; then
    echo "FAIL C: stamped with no sha resolved: $(cat "$_stamp")" >&2
    _fails=1
  fi
  rm -rf "$_d"
  # ---- D-G: the main-reachability gate against a REAL two-repo fixture ----
  # A real `git clone` with a real origin, not a hand-written world: the subject is what
  # `git rev-parse <ref>:<path>` answers when the local ref lags its remote, and no fixture
  # that stubs git can be wrong about that in the same way the real thing is. The bug is
  # exactly this shape -- `gh pr merge` advances origin/main and leaves refs/heads/main
  # where it was -- so the world has to be able to hold two different shas for one branch.
  _g=$(mktemp -d)
  (
    set -e
    export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null
    git init -q -b main "$_g/up"
    cd "$_g/up"
    git config user.email t@t; git config user.name t
    printf 'v1\n' > f.txt; git add f.txt; git commit -qm v1
    cd "$_g"
    git clone -q "$_g/up" work
    # A SECOND clone for the fetch-is-broken case: it is fetched while the remote still
    # works, so its cached origin/main holds v2, and only then is the remote broken. That
    # is the one world where the cached ref is better evidence than local main -- without
    # it, resolve_main_ref's middle branch is never executed and reverting it passes.
    git clone -q "$_g/up" nofetch
    cd "$_g/up"; printf 'v2\n' > f.txt; git commit -qam v2
    cd "$_g/nofetch"; git fetch -q origin main
    # BREAK THE REMOTE ONLY AFTER PROVING WE ARE IN THE FIXTURE. This line is the one
    # destructive command in the selftest, and .git/config is SHARED by every worktree, so
    # running it in the real tree rewrites every session's origin. That is not hypothetical:
    # before the `unset GIT_DIR` above existed, a hook-environment probe pointed these
    # commands at this repository and remote.origin.url became the fixture's path (recovered
    # from .git/logs and `gh repo view`, no refs lost). The unset prevents the leak; this
    # asserts the consequence independently, because the next leak will arrive by a route
    # the unset does not cover.
    #
    # COMPARE THE GIT DIR, NOT THE WORKTREE. `--show-toplevel` was the first version and it
    # is a SELF-MATCHING predicate: with GIT_DIR leaked and no GIT_WORK_TREE, git treats the
    # cwd as the worktree top, so it answers the cwd the test compares it against and passes
    # in exactly the world it exists to catch. Measured -- with the unset removed, the guard
    # was present, passed, and the real remote.origin.url was still overwritten. The git dir
    # is the thing `set-url` writes into, so that is the thing to identify.
    #
    # Resolved with `cd -P` because mktemp -d hands back /var/folders/... while git answers
    # /private/var/folders/... (/var is a symlink on macOS), and the textual test refused its
    # own fixture -- a guard that fires on the correct world is the same defect as one that
    # never fires.
    _fixdir=$(cd -P "$_g/nofetch" && pwd)
    _gitdir=$(git rev-parse --absolute-git-dir)
    _gitdir=$(cd -P "$_gitdir" && pwd)
    if [ "$_gitdir" != "$_fixdir/.git" ]; then
      echo "REFUSING: about to rewrite remote.origin.url in $_gitdir, which is not the" >&2
      echo "  fixture's ($_fixdir/.git). .git/config is shared by every worktree, so this" >&2
      echo "  would repoint every session's origin at a temp path." >&2
      exit 1
    fi
    git remote set-url origin "$_g/gone"
  ) >/dev/null 2>&1 || { echo "FAIL: could not build the git fixture" >&2; rm -rf "$_g"; exit 1; }

  _case() {  # _case <label> <ref-mode> <file-content> <want-rc> <want-in-output> [subdir]
    local label="$1" mode="$2" content="$3" want_rc="$4" want_txt="$5" dir="${6:-work}" out rc=0
    out=$(
      cd "$_g/$dir" || exit 9
      printf '%s' "$content" > f.txt
      # The tree's own refs/heads/main is deliberately NOT advanced: that IS the stale-cache
      # state a `gh pr merge` leaves behind.
      # if/elif rather than `case`, because bash 3.2 (macOS /bin/bash) misreads a case
      # pattern's `)` inside $( ) as the end of the substitution -- "syntax error near
      # unexpected token `;;'", measured here before this comment existed.
      if [ "$mode" = stale ]; then
        # EACH CASE POPULATES ITS OWN CACHED REF. Without this line G passed only because
        # D had run first and D's resolve_main_ref did the fetch -- an order dependency
        # between cases, so reverting the resolution turned G red for the wrong reason.
        git fetch -q origin main 2>/dev/null || true
        MAIN_REF=main
      else
        MAIN_REF=$(resolve_main_ref 2>/dev/null)
      fi
      assert_matches_main_ref f.txt 2>&1
    ) || rc=$?
    if [ "$rc" -ne "$want_rc" ]; then
      echo "FAIL $label: rc=$rc, wanted $want_rc -- $out" >&2
      _fails=1
    elif [ -n "$want_txt" ] && ! printf '%s' "$out" | grep -qF "$want_txt"; then
      echo "FAIL $label: output does not contain '$want_txt' -- $out" >&2
      _fails=1
    fi
  }
  # D: the reported bug. Content matches origin/main and NOT the stale local main, and the
  #    push must go through. Under the old code this was the refusal whose remedy -- merge
  #    again -- could not work, because the branch was already merged.
  _case "D origin/main content passes with a stale local main" resolved 'v2
' 0 ""
  # E: THE FALSE PERMIT, the direction nobody reported. Content matches the OLD local main
  #    exactly, so the pre-fix gate PASSED it and the pod ran bytes main no longer holds.
  #    Refusing here is what makes D a fix rather than a widening.
  _case "E content matching only the STALE main is refused" resolved 'v1
' 1 "differs from"
  # F: matches neither -- still refused, resolution or not. The narrowness control.
  _case "F content matching neither ref is still refused" resolved 'v3
' 1 "differs from"
  # G: fetch unavailable, so MAIN_REF falls back to local main. The refusal must NAME the
  #    staleness and a remedy that works; "merge your branch first" alone is the loop.
  _case "G a stale-main refusal names git fetch, not another merge" stale 'v2
' 1 "git fetch origin main"
  # H: the CACHED-REF branch, in the clone whose origin is gone. Content matches v2, which
  #    is what the cached origin/main holds and what local main does not -- so it passes only
  #    if resolve_main_ref prefers the cache over local main when the fetch fails. This is
  #    the world M2 needed: with the middle branch reverted, MAIN_REF is local main (v1) and
  #    this refuses. It also asserts the WARNING, since a silent fallback to possibly-stale
  #    evidence is the thing that has to be visible.
  _case "H an unreachable origin still prefers the cached origin/main over local main" \
    resolved 'v2
' 0 "" nofetch
  _hout=$(cd "$_g/nofetch" && resolve_main_ref 2>&1 >/dev/null)
  if ! printf '%s' "$_hout" | grep -qF "CACHED origin/main"; then
    echo "FAIL H2: the cached-ref fallback did not warn that it may be behind: $_hout" >&2
    _fails=1
  fi
  rm -rf "$_g"
  # ---- I: every --write call passes the ref, checked on THIS FILE'S SOURCE ----
  # A source-level assertion, because --all's body cannot be driven from here: it talks to the
  # real pod. Removing --ref from both call sites left this selftest green before this case
  # existed (measured), and the consequence is silent -- pod_drift's default is the local `main`,
  # so the manifest is built from a stale ref while push_one and the stamp use origin/main, and
  # the pod-side --check then reports drift on the file the push just landed.
  #
  # COUNTED BOTH WAYS, and neither pattern is a literal that matches its own line: the total
  # number of --write calls must EQUAL the number carrying --ref, so a third call site added
  # later without the ref fails here rather than passing by absence.
  # ANCHORED ON THE INVOCATION, not on the mention. The first version counted
  # 'pod_drift\.py --write' and its own FAIL message contains that string, so the total read 3
  # against 2 and the case failed on a correct file -- §267's self-matching grep, third instance
  # today. `^python3 ` at line start matches only a command, never prose or an echo.
  _w_total=$(grep -cE '^ *python3 scripts/pod_drift\.py --write' "$0")
  _w_ref=$(grep -cE '^ *python3 scripts/pod_drift\.py --write --ref' "$0")
  if [ "$_w_total" -lt 2 ] || [ "$_w_total" -ne "$_w_ref" ]; then
    echo "FAIL I: $_w_ref of $_w_total manifest-write call(s) pass --ref. All three" >&2
    echo "  halves of --all must name ONE ref: push_one gates against \$MAIN_REF, the stamp" >&2
    echo "  uses it, and pod_drift's own default is the LOCAL main, a stale cache since the" >&2
    echo "  PR flip. A manifest built from a different ref asserts the wrong blob for a file" >&2
    echo "  that landed correctly." >&2
    _fails=1
  fi
  [ "$_fails" -eq 0 ] || { echo "pod_push selftest: FAIL"; exit 1; }
  echo "pod_push selftest ok: a clean partial push recomputes the stamp, a drifted one clears it, an unresolved sha clears rather than stamps, and the main gate compares against origin/main -- so a file matching the merged tip passes while one matching only a stale local main is refused with a remedy that works; every --write call passes that same ref"
  exit 0
fi

ALL=0
if [ "${1:-}" = "--all" ]; then ALL=1; shift; fi
[ $# -ge 1 ] || [ $ALL -eq 1 ] || { echo "usage: $0 [--all] <file>..."; echo "       $0 --all   (sync the whole manifest: push changed, delete manifest-left)"; exit 2; }

# ONE fetch for the whole invocation: push_one runs per file and resolve_stamp_sha reads
# the same ref, so resolving inside either would put a network round trip in the loop.
MAIN_REF=$(resolve_main_ref)

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
  assert_matches_main_ref "$f"
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
  # REFUSE FROM THE INTEGRATION TREE. pod_push cds to `dirname $0/..`, so it runs in whichever
  # COPY you invoked -- and a copy lives in /Users/bytedance/code/aupai. Since the 2026-09-05
  # flip that tree is DETACHED, so every `git show HEAD:` underneath answered about a commit
  # behind main: measured HEAD 0425accb vs main 1595220e, one rewritten file silently skipped
  # (not refused -- a path the manifest omits is never offered to the per-file gate below) and
  # the stamp still claimed main. Same script, two behaviours, decided by which path you typed.
  #
  # The predicate is db's is_integration_tree (main worktree AND has linked worktrees), the same
  # one the pre-commit hook uses -- not a sixth spelling of the idea. It is stronger than testing
  # HEAD != main because it also fires if someone re-attaches that tree at main, where the HEAD
  # test would pass and the next detach would break it again.
  if python3 -c "
import sys, os
sys.path.insert(0, os.path.join('$(pwd)', 'scripts'))
try:
    from integration_tree import is_integration_tree
except Exception:
    sys.exit(1)
sys.exit(0 if is_integration_tree('$(pwd)') else 1)
" 2>/dev/null; then
    echo "REFUSING: this is the integration tree's copy of pod_push." >&2
    echo "  It cds to its own directory, so every git read below would use that tree -- which is" >&2
    echo "  detached and behind main, and the manifest built there SKIPS files rather than" >&2
    echo "  refusing them. Run your own worktree's copy instead:" >&2
    echo "    cd <your worktree> && bash scripts/pod_push.sh --all" >&2
    echo "  That worktree's HEAD must be REACHABLE FROM main -- behind is fine and stamps main's" >&2
    echo "  sha, an unmerged commit is refused. Merge before pushing if you have local work." >&2
    exit 1
  fi
  # BEFORE the first push, not at stamp time: an unmergeable HEAD must refuse while
  # refusing still means nothing shipped.
  STAMP_SHA=$(resolve_stamp_sha)
  # ONE REF FOR ALL THREE HALVES OF --all, passed explicitly rather than defaulted. push_one
  # gates each file against $MAIN_REF and resolve_stamp_sha stamps that ref's sha; pod_drift
  # --write defaulted to the LOCAL `main`, which since the PR flip is a stale cache. Measured in
  # a two-repo fixture with local main one commit behind origin/main: push_one required blob
  # 8c1384d8, the stamp said 3439ccdf (both origin/main), and the manifest wrote 626799f0 (local
  # main) -- so the manifest asserted the OLD blob for a file pushed at the NEW one and the
  # pod-side --check reported drift on the file the push had just landed. That is the symptom
  # tilerl-0a hand-fixed twice on 2026-09-08 by fast-forwarding local main, which worked because
  # it collapsed the three refs onto one; passing the ref does the same thing without touching a
  # ref every worktree shares. (The comment here used to say "from HEAD", which was already
  # false: --write moved to `main` in the 2026-09-05 flip.)
  python3 scripts/pod_drift.py --write --ref "$MAIN_REF" >/dev/null
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
  # THE LEDGERS, BOTH DIRECTIONS, because pod_push excludes runs/ by design and nothing
  # else ran the reverse path. Measured 2026-09-04 (6e): e1_31b_loop_500 and
  # params_leg_final_score_matrix were closed on main for hours and still read `running` on
  # the pod, no_ghost_running was NO-GO there, and b0's Stage E arm 2 did not launch on a
  # lane the ledger said was busy. A ledger the two sides disagree about is worse than a
  # stale one: each side's guard is correct about its own file and wrong about the world.
  #
  # ORDER: pull first (pod rows home), then push (closes out to the pod). The pull direction
  # is what makes the push safe -- pushing first would send closes for rows whose pod-side
  # events this repo has not yet seen, and append_pod_rows keys on (name, started), so a row
  # only the pod knows about would get a second event rather than being folded onto.
  #
  # REFUSE ON EITHER DIRECTION'S ERROR, before the manifest ships. pod_pull_ledgers returns
  # nonzero when an append fails or a re-read cannot verify it, and a manifest that vouches
  # for a tree whose ledgers half-synced is the state that reads as clean and is not.
  # Conflicts where both sides hold different non-empty values are NOT an error -- the tool
  # reports them and stops, because a human decides which measurement is right.
  echo "pod_push --all: ledgers, pull then push"
  if ! python3 scripts/pod_pull_ledgers.py --apply; then
    echo "REFUSING: pulling pod ledger rows home failed -- not pushing the manifest." >&2
    echo "  The pod holds rows this repo lacks; fix that first or the manifest vouches" >&2
    echo "  for a tree whose ledgers disagree with the pod's." >&2
    exit 1
  fi
  if ! python3 scripts/pod_pull_ledgers.py --push --apply; then
    echo "REFUSING: pushing local ledger closes to the pod failed -- manifest not shipped." >&2
    echo "  Rows closed here still read running there, which is what makes no_ghost_running" >&2
    echo "  NO-GO on the pod and blocks the next launch." >&2
    exit 1
  fi
  # The manifest last: it must describe exactly what landed.
  ~/bin/podput data/pod_head_manifest.txt /work/aupai/data/pod_head_manifest.txt
  ~/bin/pod "cd /work/aupai && python3 scripts/pod_drift.py --check" < /dev/null
  stamp_sync all "$STAMP_SHA"
  exit 0
fi

# GENERATE the manifest from the HEAD being pushed. It is not tracked (shape A, 6e ruling
# 2026-09-04): a file that is a pure function of HEAD does not belong in the tree, and
# tracking it was the source of every "local changes would be overwritten" abort -- the
# pre-commit hook regenerated it on every commit, so any two branches that both committed
# collided on it, 4 of 15 rows in runs/friction.jsonl. The tree check above already refuses
# a dirty push, so HEAD is what these files are.
#
# THE SAME REF THE PER-FILE GATE USES, for the reason spelled out at the --all call above: a
# manifest built from a different ref than the files were gated against asserts the wrong blob
# for a file that landed correctly.
python3 scripts/pod_drift.py --write --ref "$MAIN_REF" >/dev/null

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

# The drift gate decides the stamp now, so its exit code has to be CAPTURED rather than
# left to `set -e`. `_rc=0; cmd || _rc=$?` survives set -e; `cmd; _rc=$?` does not --
# the shell exits at cmd before the assignment runs.
#
# A partial push may only claim a sha this tree can legitimately stamp, which is the same
# question --all asks: HEAD must be main or reachable from it. resolve_stamp_sha prints a
# refusal and returns nonzero otherwise, and then the stamp is cleared rather than claimed.
_drift_rc=0
~/bin/pod "cd /work/aupai && python3 scripts/pod_drift.py --check" < /dev/null || _drift_rc=$?
_partial_sha=""
if [ "$_drift_rc" -eq 0 ]; then
  _partial_sha=$(resolve_stamp_sha 2>/dev/null) || _partial_sha=""
fi
if [ "$_drift_rc" -eq 0 ] && [ -n "$_partial_sha" ]; then
  stamp_sync partial clean "$_partial_sha"
else
  stamp_sync partial
fi
exit "$_drift_rc"

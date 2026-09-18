#!/usr/bin/env bash
# lib_ff_main.sh -- safely advance an integration tree's LOCAL main REF to origin/main.
#
# WHY THIS EXISTS. The integration checkout is DETACHED at an old commit while the merge
# machinery compares the local main BRANCH REF (`rev-list main..origin/main`). After a PR
# merges, that ref lags origin/main and blocks the next merge / makes the symlinked
# pre-commit hook stale, so selftests registered on main go silently dark for every worktree.
# A `git checkout --detach origin/main` moves HEAD, not the compared ref -- it does not help.
#
# WHAT IT DOES. `ff_main_ref <integ_dir>` moves ONLY refs/heads/main to origin/main:
#   - refuses (rc 3) if any linked worktree has main CHECKED OUT, naming the path and giving
#     the detach command (moving a checked-out branch ref from under a worktree corrupts its
#     index; `git fetch origin main:main` itself refuses this, and update-ref must not bypass);
#   - fetches origin/main; network/fetch failure -> rc 2 (freshness unknown, never assumed);
#   - level / ahead (ahead is the in-progress merge) -> rc 0, unchanged;
#   - STRICT ANCESTOR (behind>0, ahead=0) -> `git fetch origin main:main`, then asserts the ref
#     equals origin/main; detached HEAD and the working tree are untouched -> rc 0;
#   - DIVERGED (behind>0 and ahead>0) -> rc 1, names local-only/origin-only, never forces.
#
# SINGLE IMPLEMENTATION. Both scripts/merge_main.sh (bash) and scripts/hooks/pre-commit
# (python, via subprocess) call THIS so the two call sites cannot drift. The helper never uses
# update-ref to bypass the checked-out guard, and never `|| true`s a failed ff.
#
# Run its self-check with: bash scripts/lib_ff_main.sh --selftest   (rc 0 on success)

_ff_die() { echo "REFUSING: $*" >&2; }

ff_main_ref() {
  _fm="$1"
  [ -d "$_fm/.git" ] || [ -f "$_fm/.git" ] || { _ff_die "ff_main_ref: not a git tree: $_fm"; return 2; }

  # 1. A linked worktree with main checked out blocks a safe ref move.
  _wt_main=""
  _wt_main_path=""
  while IFS= read -r _line; do
    case "$_line" in
      "worktree "*) _cur="${_line#worktree }" ;;
      "branch refs/heads/main")
        if [ -n "$_cur" ]; then _wt_main="yes"; _wt_main_path="$_cur"; fi ;;
    esac
  done < <(git -C "$_fm" worktree list --porcelain)
  if [ "$_wt_main" = "yes" ]; then
    _ff_die "main is checked out in a worktree: $_wt_main_path"
    echo "  A checked-out branch ref cannot be moved from under that worktree (its index would" >&2
    echo "  desync). Free it there, then re-run:" >&2
    echo "    git -C \"$_wt_main_path\" checkout --detach   # or switch to its work branch" >&2
    return 3
  fi

  # 2. Fetch remote-tracking only; never assume freshness when offline.
  if ! git -C "$_fm" fetch -q origin main; then
    _ff_die "could not fetch origin/main; local main freshness is unknown (offline?)"
    echo "  Fix network/tunnel access and re-run; do not merge on an unfetched main." >&2
    return 2
  fi

  # 3. Three-way relationship.
  _behind=$(git -C "$_fm" rev-list --count main..origin/main 2>/dev/null)
  _ahead=$(git -C "$_fm" rev-list --count origin/main..main 2>/dev/null)
  case "$_behind" in ''|*[!0-9]*) _ff_die "cannot compare main to origin/main"; return 2 ;; esac
  case "$_ahead"  in ''|*[!0-9]*) _ff_die "cannot compare main ahead-count"; return 2 ;; esac

  if [ "$_behind" -eq 0 ]; then
    return 0  # level, or ahead (the merge-in-progress commit)
  fi
  if [ "$_ahead" -ne 0 ]; then
    _ff_die "local main DIVERGED from origin/main ($_behind behind, $_ahead ahead); not fast-forwardable"
    echo "  A force would sideways-move main past a local-only commit and is forbidden. Inspect:" >&2
    echo "    git -C \"$_fm\" log --oneline origin/main..main   # local-only" >&2
    echo "    git -C \"$_fm\" log --oneline main..origin/main   # origin-only" >&2
    return 1
  fi

  # 4. Strict ancestor: ff the ref only. The integration checkout is detached off main, so this
  # moves refs/heads/main without touching HEAD or the working tree. No --force, no bypass.
  if ! git -C "$_fm" fetch -q origin main:main; then
    _ff_die "main was fast-forwardable but 'git fetch origin main:main' failed"
    echo "  Run it by hand in the integration tree and re-run:" >&2
    echo "    git -C \"$_fm\" fetch origin main:main" >&2
    return 2
  fi
  if [ "$(git -C "$_fm" rev-parse main)" != "$(git -C "$_fm" rev-parse origin/main)" ]; then
    _ff_die "after fetch origin main:main the local ref still differs from origin/main"
    echo "  Inspect the refs manually; do not force: git -C \"$_fm\" rev-list --left-right --count main...origin/main" >&2
    return 2
  fi
  echo "ff_main_ref: local main was $_behind commit(s) behind; fast-forwarded the ref (working tree untouched)." >&2
  return 0
}

_ff_selftest() {
  _fails=0
  _ok() { echo "  ok   $1"; }
  _no() { echo "  FAIL $1" >&2; _fails=$((_fails + 1)); }
  _d="$(mktemp -d 2>/dev/null || mktemp -t ffmain)"
  trap 'rm -rf "$_d"' EXIT
  git init -q --bare -b main "$_d/origin.git"
  git clone -q "$_d/origin.git" "$_d/integ"
  ( cd "$_d/integ" && git config user.email t@t && git config user.name T \
    && echo b > f && git add f && git commit -qm base && git push -q origin main )
  _base=$(git -C "$_d/integ" rev-parse main)

  # Advance ORIGIN only.
  git clone -q --branch main "$_d/origin.git" "$_d/adv"
  ( cd "$_d/adv" && git config user.email t@t && git config user.name T \
    && echo n > g && git add g && git commit -qm adv && git push -q origin main )
  _origin=$(git --git-dir="$_d/origin.git" rev-parse main)

  # WORLD 1: strict ancestor + detached HEAD -> auto ff, ref moves, HEAD pinned at base.
  ( cd "$_d/integ" && git checkout -q --detach "$_base" )
  _o=$(ff_main_ref "$_d/integ" 2>&1) && _rc=0 || _rc=$?
  if [ "$_rc" -eq 0 ] && [ "$(git -C "$_d/integ" rev-parse main)" = "$_origin" ] \
     && [ "$(git -C "$_d/integ" rev-parse HEAD)" = "$_base" ]; then
    _ok "W1 strict ancestor auto-ff moves ref, leaves detached HEAD/worktree at base"
  else
    _no "W1: rc=$_rc main=$(git -C "$_d/integ" rev-parse main 2>/dev/null) want $_origin; HEAD=$(git -C "$_d/integ" rev-parse HEAD) want $_base; $_o"
  fi

  # WORLD 1b: level now -> rc0 silent.
  _o=$(ff_main_ref "$_d/integ" 2>&1) && _rc=0 || _rc=$?
  [ "$_rc" -eq 0 ] && [ -z "$_o" ] && _ok "W1b level ref passes silently" || _no "W1b rc=$_rc: $_o"

  # WORLD 2: a worktree has main checked out -> rc3 hard FAIL, ref untouched.
  ( cd "$_d/integ" && git worktree add -q "$_d/namemain" main )
  _o=$(ff_main_ref "$_d/integ" 2>&1) && _rc=0 || _rc=$?
  if [ "$_rc" -eq 3 ] && echo "$_o" | grep -q "checked out in a worktree" \
     && [ "$(git -C "$_d/integ" rev-parse main)" = "$_origin" ]; then
    _ok "W2 checked-out-main worktree hard FAILs (rc3) naming the path, ref unchanged"
  else
    _no "W2: want rc3 + path + unchanged ref, got rc=$_rc: $_o"
  fi
  ( cd "$_d/integ" && git worktree remove --force "$_d/namemain" >/dev/null 2>&1 )

  # WORLD 3: diverged -> rc1, ref not force-moved. Fresh clone off the bare; give integ a
  # local-only commit, then advance origin with a second commit -> mutual divergence.
  rm -rf "$_d/div"
  git clone -q "$_d/origin.git" "$_d/div"
  ( cd "$_d/div" && git config user.email t@t && git config user.name T && git checkout -q --detach HEAD
    _l=$(echo local | git commit-tree HEAD^{tree} -p "$(git rev-parse HEAD)" -m localonly)
    git update-ref refs/heads/main "$_l" )
  _dlocal=$(git -C "$_d/div" rev-parse main)
  ( cd "$_d/adv" && echo o > h && git add h && git commit -qam originonly2 && git push -q origin main )
  _o=$(ff_main_ref "$_d/div" 2>&1) && _rc=0 || _rc=$?
  if [ "$_rc" -eq 1 ] && echo "$_o" | grep -q "DIVERGED" \
     && [ "$(git -C "$_d/div" rev-parse main)" = "$_dlocal" ]; then
    _ok "W3 diverged refs hard FAIL (rc1), local ref not force-moved"
  else
    _no "W3: want rc1 DIVERGED + ref kept, got rc=$_rc: $_o"
  fi

  if [ "$_fails" -eq 0 ]; then echo "lib_ff_main selftest OK: 3 worlds (ff / checked-out / diverged)"; return 0; fi
  echo "lib_ff_main selftest: $_fails FAIL(s)" >&2; return 1
}

# Run its self-check ONLY when executed directly (bash scripts/lib_ff_main.sh --selftest),
# not when sourced: callers (merge_main.sh, the python hook) pass their own "$1", and matching
# on sourced "$1" would double-run this and interleave output. BASH_SOURCE vs $0 distinguishes.
if [ "${BASH_SOURCE[0]:-}" = "${0:-}" ] && [ "${1:-}" = "--selftest" ]; then
  _ff_selftest
fi

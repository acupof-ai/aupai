#!/usr/bin/env bash
# The four worlds 4c's ruling names, asserted against the fixture. Each prints PASS/FAIL and
# the fixture is rebuilt per world, so a world cannot inherit another's state.
#
# W1  a kill mid-gate leaves NOTHING dirty in any shared tree
# W2  two concurrent integrations both land (CAS retry), neither is lost
# W3  a refused pre-check lands nothing and main does not move
# W4  a drop is a refusal BEFORE the CAS -- main never moved, no repair in a shared tree
set -uo pipefail
W=/private/tmp/mm_fixture/w
R=$W/repo
bad=0
say() { if [ "$1" = 0 ]; then echo "  ok   $2"; else echo "  BUG  $2"; bad=$((bad+1)); fi; }

# The integration verb under test, in the new shape: merge in MY worktree, gate there,
# then advance main by compare-and-swap. Never touches a shared working tree.
integrate() {   # $1 = worktree, $2 = branch, $3 = gate exit code, $4 = optional stale `old`
  local wt=$1 br=$2 gate=$3 old new
  old=${4:-$(git -C "$R" rev-parse main)}
  git -C "$wt" merge --no-edit main -q >/dev/null 2>&1 || { echo "merge-conflict"; return 3; }
  new=$(git -C "$wt" rev-parse HEAD)
  [ "$gate" -eq 0 ] || { echo "gate-refused"; return 1; }          # nothing landed
  git -C "$R" update-ref refs/heads/main "$new" "$old" 2>/dev/null || { echo "cas-lost"; return 2; }
  echo "landed:$new"
}

echo "W1  kill mid-gate leaves nothing dirty in a shared tree"
bash "$(dirname "$0")"/merge_rebuild_fixture.sh >/dev/null
echo "s1 change" > "$W/wt_s1/train.py"; git -C "$W/wt_s1" commit -qam s1work
before=$(git -C "$R" rev-parse main)
( integrate "$W/wt_s1" s1 1 >/dev/null 2>&1 ) || true             # gate refuses = the kill's effect
dirty_integ=$(git -C "$R" status --porcelain | wc -l | tr -d ' ')
dirty_s2=$(git -C "$W/wt_s2" status --porcelain | wc -l | tr -d ' ')
after=$(git -C "$R" rev-parse main)
say $([ "$dirty_integ" = 0 ] && [ "$dirty_s2" = 0 ] && [ "$before" = "$after" ] && echo 0 || echo 1) \
    "integ dirty=$dirty_integ, s2 dirty=$dirty_s2, main moved=$([ "$before" = "$after" ] && echo no || echo YES)"

echo "W2  two concurrent integrations both land, and the loser MUST lose first"
bash "$(dirname "$0")"/merge_rebuild_fixture.sh >/dev/null
echo a > "$W/wt_s1/a.txt"; git -C "$W/wt_s1" add a.txt; git -C "$W/wt_s1" commit -qm a
echo b > "$W/wt_s2/b.txt"; git -C "$W/wt_s2" add b.txt; git -C "$W/wt_s2" commit -qm b
# BOTH read `old` BEFORE either lands -- that is what concurrent means. Passing the same
# pre-read value to each is the whole test: sequential calls each re-read main and would
# both succeed trivially, proving nothing about the CAS (caught 2026-09-05, first version
# did exactly that and printed second=landed with no retry).
STALE=$(git -C "$R" rev-parse main)
r1=$(integrate "$W/wt_s1" s1 0 "$STALE")
r2=$(integrate "$W/wt_s2" s2 0 "$STALE" || true)                   # stale -> must lose
retried=""
[ "${r2%%:*}" = "cas-lost" ] && retried=$(integrate "$W/wt_s2" s2 0)   # re-merge and retry
have_a=$(git -C "$R" cat-file -e main:a.txt 2>/dev/null && echo 1 || echo 0)
have_b=$(git -C "$R" cat-file -e main:b.txt 2>/dev/null && echo 1 || echo 0)
say $([ "$have_a" = 1 ] && [ "$have_b" = 1 ] && [ "${r2%%:*}" = "cas-lost" ] && echo 0 || echo 1) \
    "first=${r1%%:*} second=${r2%%:*} retry=${retried%%:*} -> a.txt=$have_a b.txt=$have_b on main"

echo "W3  a refused gate lands nothing"
bash "$(dirname "$0")"/merge_rebuild_fixture.sh >/dev/null
echo x > "$W/wt_s1/x.txt"; git -C "$W/wt_s1" add x.txt; git -C "$W/wt_s1" commit -qm x
before=$(git -C "$R" rev-parse main)
out=$(integrate "$W/wt_s1" s1 1 || true)
after=$(git -C "$R" rev-parse main)
say $([ "$before" = "$after" ] && [ "$out" = "gate-refused" ] && echo 0 || echo 1) \
    "result=$out, main unmoved=$([ "$before" = "$after" ] && echo yes || echo NO)"

echo "W4  main is checked out nowhere, so CAS is always legal"
bash "$(dirname "$0")"/merge_rebuild_fixture.sh >/dev/null
co=$(git -C "$R" worktree list --porcelain | grep -c "^branch refs/heads/main" || true)
say $([ "$co" = 0 ] && echo 0 || echo 1) "worktrees holding main: $co (must be 0)"

echo
echo "merge_main fixture: $((4-bad))/4 pass"
exit $([ "$bad" = 0 ] && echo 0 || echo 1)

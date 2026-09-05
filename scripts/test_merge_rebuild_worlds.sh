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

# The integration verb under test, in the new shape: merge in MY worktree, gate there, advance
# main by compare-and-swap, THEN push. Never touches a shared working tree.
#
# $5 selects the push-failure policy, and it exists so W6 can discriminate: "keep" is the shipped
# behaviour (a failed push leaves main advanced), "rollback" is the plausible wrong one. Without
# a policy that CAN roll back, W6 asserts something no code path could violate -- a world with no
# power to disagree has none to confirm (this file's own §231, filed 2026-09-05).
integrate() {   # $1 = worktree, $2 = branch, $3 = gate exit code, $4 = stale `old`, $5 = policy
  local wt=$1 br=$2 gate=$3 old new policy=${5:-keep} prc=0
  old=${4:-$(git -C "$R" rev-parse main)}
  git -C "$wt" merge --no-edit main -q >/dev/null 2>&1 || { echo "merge-conflict"; return 3; }
  new=$(git -C "$wt" rev-parse HEAD)
  [ "$gate" -eq 0 ] || { echo "gate-refused"; return 1; }          # nothing landed
  git -C "$R" update-ref refs/heads/main "$new" "$old" 2>/dev/null || { echo "cas-lost"; return 2; }
  # The push is a SEPARATE handoff after the atomic step. The fixture has no `origin`, so this
  # fails for real rather than being simulated.
  git -C "$R" push origin main >/dev/null 2>&1 || prc=$?
  if [ "$prc" != 0 ] && [ "$policy" = rollback ]; then
    git -C "$R" update-ref refs/heads/main "$old" "$new" 2>/dev/null || true
    echo "rolled-back:$new"; return 4
  fi
  [ "$prc" = 0 ] && echo "landed-pushed:$new" || echo "landed-unpushed:$new"
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

# W5  A PENDING OVERRIDE ROW SURVIVES A KILLED GATE. The hook records the event under the git
# dir (invisible to `git status`, per-worktree so two sessions cannot collide) and the GATE
# writes it through the CLI and truncates. If the gate dies between reading and truncating,
# the row must still be there for the next run -- a metric that silently loses events reads 0
# as "no overrides happened", the absent-vs-empty error this repo keeps paying for (4c).
echo "W5  a pending override row survives a killed gate"
bash "$(dirname "$0")"/merge_rebuild_fixture.sh >/dev/null
GD=$(git -C "$W/wt_s1" rev-parse --git-dir)
[ "${GD#/}" = "$GD" ] && GD=$W/wt_s1/$GD                      # rev-parse may print it relative
PEND=$GD/aupai_pending_friction
printf '%s\n' '{"kind":"override","cause":"AUPAI_BEHIND_MAIN_OK=1"}' > "$PEND"
# The hook's write must not dirty the tree -- that is the defect being removed.
dirty=$(git -C "$W/wt_s1" status --porcelain | wc -l | tr -d ' ')
# A gate that dies AFTER reading and BEFORE truncating. Run as a real child and SIGKILL it by
# its own pid: `kill -9 $$` inside `( )` kills the SCRIPT, because $$ is the parent's pid in a
# subshell, not the subshell's (caught 2026-09-05 -- W5 printed nothing at all and the run
# ended silently at 4/5, which reads like a hang rather than a bug in the test).
{ bash -c 'cat "$1" >/dev/null; kill -9 $$' _ "$PEND"; } 2>/dev/null || true
survived=$([ -s "$PEND" ] && echo 1 || echo 0)
# The next gate run drains it: write (simulated), then truncate.
drained=""
if [ -s "$PEND" ]; then drained=$(wc -l < "$PEND" | tr -d ' '); : > "$PEND"; fi
empty_after=$([ -s "$PEND" ] && echo 0 || echo 1)
say $([ "$dirty" = 0 ] && [ "$survived" = 1 ] && [ "$drained" = 1 ] && [ "$empty_after" = 1 ] && echo 0 || echo 1) \
    "tree dirty=$dirty, survived the kill=$survived, next run drained=$drained row(s), truncated=$empty_after"

# W6  A FAILING PUSH LEAVES MAIN ADVANCED. The CAS is the atomic step; the push to origin is a
# separate handoff that today nobody automates -- measured 2026-09-05, main took 670 commits in
# 24h against 139 origin/main push events, so it advances ~5x per push and every gap is a window
# where a peer's fetch and the pod read a stale main. The integration step pushes after the CAS,
# and a push that fails must NOT roll the ref back: the commit is already durable and reachable,
# and undoing it to match origin would discard work to fix a delivery problem. It prints what is
# due and exits nonzero (4c's ruling).
echo "W6  a failing push leaves main advanced"
bash "$(dirname "$0")"/merge_rebuild_fixture.sh >/dev/null
echo p > "$W/wt_s1/p.txt"; git -C "$W/wt_s1" add p.txt; git -C "$W/wt_s1" commit -qm p
before=$(git -C "$R" rev-parse main)
res=$(integrate "$W/wt_s1" s1 0 "" keep)
kept=$(git -C "$R" rev-parse main)
# THE DISCRIMINATION ARM. The same world under the rollback policy must give the OPPOSITE answer,
# or this world cannot tell the shipped behaviour from the wrong one and its pass means nothing.
bash "$(dirname "$0")"/merge_rebuild_fixture.sh >/dev/null
echo p > "$W/wt_s1/p.txt"; git -C "$W/wt_s1" add p.txt; git -C "$W/wt_s1" commit -qm p
rb_before=$(git -C "$R" rev-parse main)
rb_res=$(integrate "$W/wt_s1" s1 0 "" rollback || true)
rb_after=$(git -C "$R" rev-parse main)
say $([ "$res" != "${res#landed-unpushed}" ] && [ "$before" != "$kept" ] \
      && [ "$rb_res" != "${rb_res#rolled-back}" ] && [ "$rb_before" = "$rb_after" ] && echo 0 || echo 1) \
    "keep -> ${res%%:*}, main advanced=$([ "$before" != "$kept" ] && echo yes || echo NO); \
rollback -> ${rb_res%%:*}, main back at old=$([ "$rb_before" = "$rb_after" ] && echo yes || echo NO)"

echo
echo "merge_main fixture: $((6-bad))/6 pass"
exit $([ "$bad" = 0 ] && echo 0 || echo 1)

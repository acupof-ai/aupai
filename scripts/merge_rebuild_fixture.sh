#!/usr/bin/env bash
# Fixture world for the merge_main rebuild: a repo whose `main` is NOT checked out anywhere,
# an integration tree detached at some older commit, and two session worktrees on their own
# branches. This is the shape the redesign proposes; every world below is asserted against it.
set -euo pipefail
W=/private/tmp/mm_fixture/w
rm -rf "$W" && mkdir -p "$W"
R=$W/repo
git init -q -b main "$R"
git -C "$R" config user.email a@b
git -C "$R" config user.name t
mkdir -p "$R/scripts" "$R/runs"
echo "code v1"   > "$R/train.py"
echo '{"r":1}'   > "$R/runs/ledger.jsonl"
echo "#gate"     > "$R/scripts/gate.sh"
git -C "$R" add -A
git -C "$R" commit -qm base
BASE=$(git -C "$R" rev-parse HEAD)

# THE INTEGRATION TREE IS THE PRIMARY WORKTREE, detached -- which is the real shape:
# /Users/bytedance/code/aupai IS the repo dir, not a secondary worktree. Detaching a
# secondary one while the primary still holds `main` would leave CAS illegal and the
# fixture would prove nothing about the real layout.
git -C "$R" checkout -q --detach "$BASE"
# Two sessions, each on its own branch.
git -C "$R" worktree add -q -b s1 "$W/wt_s1" "$BASE"
git -C "$R" worktree add -q -b s2 "$W/wt_s2" "$BASE"

echo "main            $(git -C "$R" rev-parse --short main)"
echo "integ (detached) $(git -C "$R" rev-parse --short HEAD)"
echo "s1              $(git -C "$W/wt_s1" rev-parse --abbrev-ref HEAD)"
echo "s2              $(git -C "$W/wt_s2" rev-parse --abbrev-ref HEAD)"
echo "BASE=$BASE"

---
question: When a pre-commit is refused for being behind main, which moves land the commit without AUPAI_BEHIND_MAIN_OK, and is the override still needed?
status: measured
source: de-113, fixture run importing scripts/hooks/pre-commit predicates, 2026-09-12
---

# Behind-main gate: the four states and the one blocked path

The gate (scripts/hooks/pre-commit) refuses a commit when it is behind main AND
stages a file main has also changed AND the staged set is not all driver-merged
AND the invocation is not itself a merge:

```
_clash = _main_touched_staged(staged) if behind and not ledger_only and not merging else []
```

Each of the four clauses is an independent exemption. Measured 2026-09-12 by
importing the installed hook and driving `_behind_main`, `staged`,
`_union_merged`, `_merging`, and `_main_touched_staged` against scratch repos:

| state | behind | ledger-only | merging | clash | lands without flag |
|---|---|---|---|---|---|
| merge commit finishing a merge | yes | no | **yes** | none | yes |
| staged set is all union/prereg-driver ledgers | yes | **yes** | no | none | yes |
| code staged; main moved a DIFFERENT file | yes | no | no | **empty** (3-dot intersection) | yes |
| local code commit staging a file main also changed | yes | no | no | non-empty | **no — needs the flag** |

3 of 4 states land without the override; exactly one needs it.

## The loop the one blocked state caused

The old refusal printed only `git merge --no-edit main`. In the blocked state
the working change is staged and overlaps a file main moved, so that merge
refuses with "your local changes to the following files would be overwritten by
merge"; `git stash` is forbidden (.git/refs/stash is shared across worktrees).
The prescribed move could not run. This was the top-2 friction cause.

## Why the override is kept

The blocked state is a real refusal, not a process bug: it stops a commit
writing on top of moved code without a tested version beneath it. The
flag-free escape already exists and is sanctioned — `scripts/merge_main.sh
<branch>`'s staged carry (`_carry_stage`/`_carry_restore`): it commits the
index to a private `refs/wip/<branch>` (never the shared stash), clears it,
runs the merge as state 1 (the merge-commit exemption), then three-way merges
the carried work back onto the merge result. Its W1-W10 selftest worlds cover
fast-forward, true merge, conflict, new-file, and union-ledger cases. So no
normal integration needs the flag; the override survives only for deliberately
reproducing a failure on older code, which is what the message says it costs.

Post-2026-09-07 code goes through a PR rather than merge_main, where the branch
is pushed and GitHub performs the merge, so the local gate is largely moot for
new code. The override rows still in runs/friction.jsonl (117, latest
2026-09-10) are hand commits of local wip, not merges.

## The fix

The refusal message now names both moves: a bare `git merge` when nothing is
staged, and `scripts/merge_main.sh <branch>` when staged code overlaps. The
predicate and the override are unchanged. The 3-of-4 table above is the
evidence that retiring the flag would remove a deliberate-exception path
without closing any legitimate loop.

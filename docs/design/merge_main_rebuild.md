# merge_main rebuild: integrate in your own worktree, advance main by CAS

Status: design, verified on a fixture. Cutover after E1's step-30 readout (4c's ruling
2026-09-05). Second reader: b0. Claimed files: `scripts/merge_main.sh`, `scripts/pod_push.sh`,
`scripts/harness.py`, `AGENTS.md`.

## The root cause, and what is downstream of it

`merge_main.sh:339` runs `git merge` INSIDE the shared integration working tree
(`/Users/bytedance/code/aupai`). Integrating is therefore a four-step non-atomic write —
worktree, index, ~30 s hook, commit — to a directory every session depends on. Any kill,
conflict or hook failure strands it in a state only another session's files can repair.

Downstream of that one design, all compensation: the index-equals-HEAD rule, "never edit in
the integration tree", the hook-runs-main's-copy defect, `.hookstaged_*` leftovers, the `cp -r`
mutant, and de's sub-600 s `timeout` guard.

Second surface, same root (e1, 2026-09-05): a `git merge main` in a session's OWN worktree was
killed by a 2-minute tool timeout mid-hook — index written, no commit, `AUTO_MERGE` without
`MERGE_HEAD`. The hook runs ~30 s inside EVERY commit, so any git command can exceed a session's
tool timeout and die half-written.

## The design

1. **Merge in your own worktree.** `git merge main` runs where only you depend on the result.
2. **Gate once, explicitly, detached.** Not an implicit pre-commit on every commit. Its result is
   recorded against the CANDIDATE SHA, so a second reader can see the gate ran on what landed and
   not on an earlier tip (4c).
3. **Advance main by compare-and-swap.** `git update-ref refs/heads/main <new> <expected-old>` —
   atomic, touches no working tree. On mismatch: re-merge and retry.
4. **The integration tree stops being a checkout of main** — detached. This is a PRECONDITION of
   (3), not a fallback: see the measurement below.

## Measured, not reasoned (2026-09-05, fixture repo)

CAS onto a branch that IS checked out does not refuse. It returns 0, moves the ref, and leaves
that worktree's HEAD and index at the old commit — so every changed file reads as staged-modified
(`M f.txt`; disk holds the old content, ref holds the new). A refusal would be loud; this is
silent, and the shared tree then looks dirty to the next session that walks in.

Detached first: the same CAS returns 0 and `git status --porcelain` is EMPTY. `git show main:path`
and `git archive main` read fine from a detached tree. A stale expected-old refuses loudly:
`cannot lock ref 'refs/heads/main': is at X but expected Y`.

## Hook split (measured on this session's own commits)

| stage | cost |
|---|---|
| tree check / behind-main / shared-file / blob+allowlist / credential / ledger-append / ruff | ~0.2 s total |
| selftests | 0.9–11.8 s |
| harness check | 28.3–30.4 s |

The seven sub-second stages stay in `pre-commit`. The two slow ones become the gate step. That is
99.5% of a 30–42 s hook moved out, and it removes the reason a `git commit` can exceed a 2-minute
tool timeout at all.

## Port list (from the grep, not from the description)

Four hardcoded paths to the integration tree exist; two matter.
`progress_feed.py:266` is an href in generated HTML and `build_agentic_sft.py:1133` is a string
inside training data — neither reads the tree.

Of `merge_main.sh`'s 18 `$MAIN` uses: **9 need only the object database** (`rev-list`, `log -1`,
`show --stat`, `merge-base`, `rev-parse`, `worktree list`) and survive detaching untouched.
**9 need a checkout** — and 7 of those exist only to run or clean up the in-tree merge (339 the
merge, 367 `checkout -- path`, 369 `status`, 422/423/428 `MERGE_HEAD`/unmerged/`merge --abort`).
Deleted, not ported. Real ports: the lock path (10) and the friction append (333).

`harness.py`: `real = "/Users/bytedance/code/aupai"` (14901). Used at 14901/14958 for merge-class
selftests that clone at a sha (history — fine), and at 2813/10158/15246 to symlink `scripts/` and
read `runs/claims/*` AS FILES. Those three keep working (a detached tree still has files) but
AGENTS.md must not claim nothing reads the integration tree as files.

`--merge-drops` reads NO files: `merge_drops` (harness.py:5167, 54 lines) shells only `rev-list`,
`ls-tree`, `log`. But its rev defaults to HEAD and the CLI passes none, so against a detached
integration tree it would check the wrong commit and print nothing — and empty output reads
identically to clean, the failure its own docstring names. Port: `--merge-drops --rev <candidate>`,
threaded to the `rev` parameter that already exists and is unused.

## Behaviour change, approved by 4c

Today a dropped path is RESTORED into the integration working tree and staged for someone else to
commit — one of the incidents this rebuild removes. New: the drop check runs on the candidate merge
commit in the integrating session's own worktree BEFORE the CAS, and a drop is a REFUSAL. Main
never moved; the fixing happens on the integrator's side. No restore, no staging in a shared tree.

## AGENTS.md

Deleted: index-equals-HEAD, never-edit-in-the-integration-tree, hook-runs-main's-copy.
**Kept: the stash rule** — `.git/refs/stash` is shared across worktrees regardless of what the
integration tree is, so it is NOT downstream of this design. Checked, not assumed.
Also deleted: de's sub-600 s `timeout` guard (4c) — two guards for one failure, and once the 30 s
gate is out of every commit the duration it protects against no longer occurs.

## Fixture worlds (4/4, `/private/tmp/mm_fixture/worlds.sh`)

- **W1** a refused gate mid-integration leaves `integ dirty=0, s2 dirty=0`, main unmoved.
- **W2** two concurrent integrations: first lands, second gets `cas-lost`, retry lands. Both files
  end up on main.
- **W3** a refused gate lands nothing and main does not move.
- **W4** `git worktree list` shows 0 worktrees holding main, so CAS is always legal.

Two of these were wrong first and are worth recording. W4 failed because the fixture detached a
SECONDARY worktree while the primary still held `main` — the real layout has the integration tree
as the primary, so the fixture was proving the opposite of the deployed shape. And W2 passed by
construction: each call re-read `main` before its CAS, so the two were sequential, not concurrent,
and the loser could not lose. Both now read a single pre-read `old`.

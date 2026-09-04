---
question: How does aupai-infra come out of this tree without a half-state, and what does it cost?
status: proposed — no cut without 4c's ruling; 6e owns steps 3-4 of the original plan
source: scripts/infra_inventory.py (re-run 2026-09-05), docs/standards/compute_contract.md, 6e's 2026-09-04 rulings
---

# The aupai-infra cut: plan

The contract (`docs/standards/compute_contract.md`) named the line. This says where it
falls in files today, what has to move together, and what it costs. **Nothing is cut until
4c rules.** The inventory numbers below are from a re-run on 2026-09-05 and move with the
tree; `scripts/infra_inventory.py --json` is the current list.

## What the measurement says

| class | files | code lines |
|---|---|---|
| mixed | 32 | 38,451 |
| project-only | 25 | 8,832 |
| contract-caller | 27 | 4,675 |
| infra-only | 11 | 1,426 |
| reference | 52 | 0 (docs and ledgers) |

Two numbers decide the shape of the cut, and neither is the one the class counts suggest.

**The harness cut is 723 lines, not 18,242.** `scripts/harness.py` dominates `mixed`
(38,451 code lines across the class, 18,242 of them here) and reads like the blocker. It is
not. 91 checks live there; 25 are pod-authoritative; of those, most ask a PROJECT question
that only the pod can answer — `mix_supply`, `corpus_fp_matches`, `sft_pack_holdout`,
`ladder_config_frozen` — and stay. The checks whose SUBJECT is the compute are 19 functions
including their broken worlds, 723 lines total: `pod_drift`, `no_ghost_running`,
`root_durable`, `no_foreground_pod_training`, `lane_respected`, `card_held_without_claim`,
`snapshot_logs_say_so_at_the_tail`, `pod_stamp_is_main`, `test_pod_wrappers`,
`refusal_precedes_push`. They need 8 harness-local helpers (`_pod_ps_rows`, `judge_pod_ps`,
`_busy_training_cards`, `_expand_cards`, `_gpu_present`, `_has_training_process`,
`_is_mount`, `_exp_fold`) totalling ~196 lines, and exactly one of those — `_exp_fold` —
reaches into the project. It is already lazy-imported, which is the seam the contract's
ledger ruling predicted.

**Only 6 files import an infra module; the other 27 callers shell out.** Every one of the 27
contract-callers touches `alloc` and nothing else — they call `card_claim.py` as a
subprocess. The real import edges are four: `loader.py -> card_claim`, `launch_gate.py ->
pod_drift`, `train.py -> pod_drift`, `merge_score_row.py -> pod_pull_ledgers` (plus
`harness.py` and `pod_drift.py` itself). A subprocess call survives a repo split with a path
change; an import does not. So the cut's API surface is four edges, not 27.

## The cut, in order

Each step leaves the tree working. No step half-lands: if a step cannot finish, it reverts.

**1. Vendored — tracked, but the install is per-laptop and this one had not run it.**
`scripts/pod`, `scripts/podput`, `scripts/test_pod_wrappers.sh` are tracked (6e's ruling 2,
2026-09-04). AGENTS.md:234 names the one-time step `ln -sf "$PWD/scripts/pod" ~/bin/pod`, and
on this machine it had never been run: `~/bin/pod` was still the original untracked file,
**missing both refusals** — no `cd … &` refusal and no `--view`. Everything that reaches the
pod through `~/bin/pod` was therefore running the unvendored wrapper, including
`harness.py:169`'s `pod_reachable` probe and `podput`'s last line. Fixed 2026-09-05 by
symlinking to the integration tree (`/Users/bytedance/code/aupai/scripts/pod`, not a
worktree — a worktree path is branch-scoped and dies with the worktree).

The general form is the finding, and it is the same shape as §199: **vendoring a file and
installing it are different claims.** The repo contained the entry point; the machine that
runs it did not. A tracked copy nothing executes is not a vendored entry point, and the
split's precondition is the second claim, not the first. Worth a check — compare
`realpath ~/bin/pod` against the tracked path — because no session can see this from inside
the repo, and every session on a fresh laptop starts in the broken state.

**2. Publish the ledger schema file.** 6e's ruling 3 left its name open. Nothing moves until
it exists, because the transport verb's contract is "rows are opaque, keyed only by what the
schema declares" and there is no schema file to point at. Proposal: `runs/ledger_schema.json`
holding `ledger_audit.KEYS` and the per-file fold predicate name. The project owns it;
neither side adds a ledger without changing it first. **Blocked on de and 6e.**

**3. Move the 11 infra-only files.** 1,426 lines, zero project imports, nothing else
changes. `git subtree split` on a path list, or a fresh repo plus `git log --follow` export —
history matters here because `pod_push.sh`'s refusal ordering and `pod`'s `cd &` refusal each
have an incident behind them, and a squashed import loses the commit that explains why.

**4. Move the 723 harness lines and their helpers.** They become the infra repo's own check
module with its own selftests and broken worlds. `_exp_fold` stays a lazy import across the
boundary, which is the one edge that has to be designed rather than moved.

**5. Convert the four import edges** to the infra package's public names, in one commit each,
each verified by running the caller. The 27 subprocess callers need a path change only.

**6. `docs/lessons/infra_incidents.md` follows** — it says so in its own header. 44's file,
their call on when.

## What this costs, and what it buys

Cost: four import edges to redesign, one lazy cross-boundary call (`_exp_fold`), a
two-repo CI, and every session learning a second checkout. The pod wrappers stop being
editable in the same commit as the code that calls them.

Buys: the 27 contract-callers already use the compute as if the five verbs existed, so the
split makes a boundary that is being respected in practice. A second target (Colab, another
cluster) implements five verbs instead of reading 38,451 lines of mixed code to find out
what the project needs.

## What I am not proposing

- **Not moving `card_claim.py` whole.** 67 alloc hits and 42 project hits; `grant_lane` is
  policy. It splits, and I have not measured where.
- **Not moving the pod checks by their EVIDENCE tag.** 25 are pod-authoritative and most
  stay. The subject is the criterion, not the authority — 6e said this and the count is what
  tempts you away from it.
- **Not cutting before step 2.** A split whose transport verb has no schema to point at is a
  half-state by the contract's own definition.

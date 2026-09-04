---
question: What may the project ask of the compute, and what does the compute promise back?
status: accepted (6e, 2026-09-04); step 3 pending their read of the landed files
source: infra inventory (scripts/infra_inventory.py, 2026-09-04), 6e ruling on the aupai-infra split
---

# The compute contract

Five verbs. The project depends on this file; it never depends on the pod. Colab,
a second cluster, or a laptop implements the same five and the project does not
change.

Written because the split needs a line to cut along. `scripts/infra_inventory.py`
measured where that line falls today: **13 infra-only files, 29 seams, 28 files
that call a verb without implementing one, 24 project files with a passing
mention, 58 docs and ledgers that only name it.** The 28 are the argument for a
contract — they already use the compute as if these verbs existed.

## The five verbs

| verb | CLI today | in | out | nonzero when |
|---|---|---|---|---|
| **exec** | `pod '<cmd>'` | a command string | stdout+stderr, container exit code | the container is not Running; the command's own failure |
| **sync** | `pod_push.sh [--all] <files>` | files reachable from main | manifest + sync stamp | any refusal, decided **before** the first byte ships |
| **allocate** | `card_claim.py acquire\|status\|release` | card ids, a job name | a claim file on the compute | the cards are held by another claim |
| **transport** | `pod_pull_ledgers.py` | ledger paths, both directions | merged rows on both sides | either direction fails |
| **sweep** | `sweep.py [--execute]` | nothing | a verdict per process, `runs/sweeper.jsonl` | never kills without `--execute` |

## What each verb never does

**exec** never interprets the command. It does not `cd` for you: the container's
default cwd is `/sgl-workspace/sglang`, and a command that does not cd inherits
it — three inline background forms produce **no log file at all** rather than an
empty one (shape 166). It never reports host state as container state: `tn exec`
is the host view, `pod` is the container view, and the same path names two
different directories in the two.

**sync** never ships a file that differs from main, and never prints a refusal
after shipping. The refusal must precede the first byte or it is not a refusal —
this failed twice (shape 136; the stamp_sync hoist, 2026-09-04). `--all` is
all-or-nothing: a partial push leaves the compute in a state worse than either
pushing or not pushing.

**allocate** never infers ownership. Not from cwd (it is the container default
and carries no information in either direction), not from a pid's presence in a
list (`nvidia-smi` gives a GPU UUID, not an index — shape 106), not from prose
markers in a json file. Identity across two queries joins on a unique id.

**transport** never parses the payload. It carries JSONL lines keyed only by what
the schema file declares, and never reads further into a row. It also never lets
one side win by being read second: a ledger the two sides disagree about is worse
than a missing one.

**sweep** never kills a process it cannot positively classify. `unclassified` is
the safe answer and it is the common one.

## What the contract does not cover, stated so nobody assumes it

- **Which cards a job should get.** Allocation is mechanism; policy (the 7-card
  block, the lane) stays in the project.
- **Whether a run is finished.** That is `exp.fold`, a project concept the sweep
  verb consumes but does not own.
- **The zombie problem.** PID 1 must reap; no verb can fix it from inside
  (`docs/standards/env_hygiene.md` §1).

## The seam, measured

29 files implement a verb and know project concepts. The three that carry most
of it:

- **`scripts/harness.py`** — the largest cut, and smaller than "the pod checks
  move" suggests. 24 checks are pod-authoritative (`EVIDENCE == "pod"`), not the
  19 in the earlier note, but most of those ask a PROJECT question that only the
  pod can answer: `mix_supply`, `corpus_fp_matches`, `sft_pack_holdout`,
  `ladder_config_frozen`. Those stay. What moves is the checks whose subject is
  the compute itself — `pod_drift`, `no_ghost_running`, `root_durable`,
  `no_foreground_pod_training`, `lane_respected`, `card_held_without_claim`. de
  names the final set; the count is not the criterion, the subject is.
- **`scripts/card_claim.py`** — 64 alloc hits, 15 project. Nearly all infra;
  `grant_lane` is policy and stays.
- **`scripts/pod_drift.py`**, **`pod_pull_ledgers.py`**, **`sweep.py`** — infra
  with a project-shaped input (the manifest, the ledger schema, `exp.fold`).

`scripts/infra_inventory.py --json` is the current list; re-run it rather than
quoting these numbers, since they move with the tree.

## Three questions the inventory raised, ruled 2026-09-04 (6e)

1. **`run_ddp.sh` and `train.py` both carry sync hits.** The drift gate that
   guards a launch is infra and moves; the launch itself is project and stays,
   calling the contract's sync-verify verb.
2. **The wrappers** (`~/bin/pod`, `podput`) are untracked today and every session
   depends on them. Vendor first, then split — otherwise step 3 splits a tree
   that does not contain its own entry points.
3. **Ledgers.** The **project owns the schema**: `ledger_audit.KEYS` and the
   per-file fold predicate stay in the project and are published as one small
   machine-readable file the infra repo reads. The transport verb treats rows as
   opaque payload keyed only by what that file declares. **Neither side may add a
   ledger without the schema file changing first.** Its name and location are
   still open between de and me.

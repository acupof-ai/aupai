---
question: Do the register and the rules hold to world-class bar? Cross-review of 3b-3: task commands, register checks, retro/review schemas, AGENTS.md rule coverage, writing.md compliance
status: recorded
source: scripts/harness.py (task/review_present/tasks_well_formed/ledgers_one_line_per_row/device_set_honoured/agents_rules_covered), AGENTS.md lines 244-330, runs/retro.jsonl, runs/review.jsonl, docs/standards/writing.md
---

# Harness cross-review 0901, 3b: register and rules

Scope: the task commands (add/done/reopen/list), the register checks (`review_present`, `tasks_well_formed`, `ledgers_one_line_per_row`), the `runs/retro.jsonl` + `runs/review.jsonl` schemas, and the AGENTS.md rule-coverage table. Verdict per row names the check, the property it tests, and whether it is the rule's property. Findings ordered by severity.

## Principle-severity findings

| # | rule | mapped check | what the check actually tests | does it test the rule's property? |
|---|---|---|---|---|
| P1 | CUDA_VISIBLE_DEVICES, not cuda:N | `gemm_dims_aligned` | tensor-shape alignment (vocab/d/ffn_hidden) to avoid a cuBLAS SM75 kernel fallback | **no** — the rule is about device selection, not shapes |
| P2 | Every delivery has a second reader | `review_present` | a review.jsonl row exists for a done task with the named reviewer | **partial** — the rule demands the row name an artifact/failing case; the check never examines the row's content |
| P3 | runs/.jsonl ledgers merge by union; row identity is (name,started)/id not position | `no_ghost_running` | running rows have a live process | **indirect** — the union-row-identity property is what `ledgers_one_line_per_row` guards, not no_ghost_running |

P1 details. The real enforcer of the cuda:N rule is `check_device_set_honoured` (harness.py:4259): it scans repo-owned `.sh` for `CUDA_VISIBLE_DEVICES=<physical index>` assignments and requires sourcing `eval/_devs.sh`. That check is a registered check (CHECKS line 4772) but maps to **no rule** in `_RULE_CHECKS`. The table maps the rule to `gemm_dims_aligned` instead. So the coverage table names the wrong check, and the correct check is invisible in coverage. `check_agents_rules_covered` cannot see either: `gemm_dims_aligned` exists (line 4554), so the mapping passes, and `device_set_honoured` being unmapped is not a violation it checks. This is the most dangerous green: a reader believes shape-alignment enforces device-selection. Fix: remap the rule to `device_set_honoured`.

P2 details. AGENTS.md:330 states a review must "name the artifact path or failing case they actually opened; a review that names neither is not a review." `check_review_present` (line 2262) verifies a review.jsonl row for the task exists and its `reviewer` equals the named one (line 2299). It never reads the row's `finding`/`summary`/`notes`/`verdict` fields. A review row `{task, reviewer, verdict}` naming nothing satisfies the check, violating the rule's own definition. The review ledger's 24 rows do carry `finding`/`summary`; the check just does not use them. Fix: require a non-empty artifact/finding field on the review row.

P1 and P2 are both classes `agents_rules_covered` cannot catch, because coverage proves a mapping was made, not that it is honest — the check's own docstring says this.

## Checks that enforce their rule's property

| check | property it tests | rule | verdict |
|---|---|---|---|
| `tasks_well_formed` | done → evidence; open → owner; has why; id collision by `opened` | done requires evidence, a task starts owned+reasoned | enforces it, and matches the command contract (`done --evidence` required) |
| `ledgers_one_line_per_row` | one JSON object per physical line across union ledgers | ledgers merge by union; a multi-line row interleaves and corrupts | enforces it; broken world is the real retro pretty-printed, 3b's actual failure |
| `curl_ipv4` | every curl invocation passes -4 | curl -4, always | enforces it |
| `task done` refuses owner-as-reviewer and non-roster reviewer | reviewer is a second reader, from fixed pairs | the second-reader rule | enforces it |
| `task add` owner-scoped ids | `<owner>-<n>` collision-free across branches | concurrent branch ids must not collide | enforces it |

## Schemas

`review.jsonl` and `retro.jsonl` are held to one-object-per-line by `ledgers_one_line_per_row`. Neither has a field-schema check beyond that. `review_present` reads `task`+`reviewer` (P2 covers the missing content check). `retro.jsonl` rows vary by author: some carry `incidents`/`change_landed`/`differently`, others `class_removed`/`note`/`self_assessment`. A retro row with no `owner` or `incidents` still passes every check. For a retrospective this loose schema may be acceptable; it is a stated property of the register, not a guard.

## Review-row production

`review.jsonl` has no harness writer: the reviewer hand-appends a row (documented in AGENTS.md:330). The producing half is therefore manual and outside the harness's copy. `review_present` catches a missing review after the 30-minute grace but, combined with P2, cannot catch a review that exists but names no artifact. The 30-minute `review_present` FAIL is the only enforcement of a review arriving.

## writing.md compliance of tonight's docs

| doc | verdict |
|---|---|
| docs/lessons/near_dedup_stage2_postpass.md | partial: tables and numbers (J ≥ 0.5, 21%, 3-grams) meet the standard; several "not X but Y" constructions and prose paragraphs remain; needs a fourth pass |
| runs/retro.jsonl row (3b) | data, not prose; schema holds, one-line per row |
| this audit | written to the standard; pending the fourth pass |

## Forward actions

1. **Remap** the `CUDA_VISIBLE_DEVICES` rule to `device_set_honoured` (P1).
2. **Require** a review row to carry an artifact/finding for `review_present` to count it (P2).
3. Optionally extend `agents_rules_covered` to flag a registered check that maps to no rule (device_set_honoured today) — that would have surfaced P1.
4. State the retro row-schema looseness as a decision, or add a `retro_well_formed`-style guard.
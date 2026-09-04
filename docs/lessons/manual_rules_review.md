---
question: Which of the 26 manual rules in AGENTS.md's coverage table have recorded incidents, and what should happen to each?
status: open
source: 4 parallel agents, each matching a rule group against docs/lessons/infra_incidents.md, docs/lessons/gate_failure_incidents.md, runs/friction.jsonl, runs/review.jsonl; 2026-09-05
---

# Manual rules review — KEEP / BRIEF / DELETE proposal

Criterion: a rule is kept by evidence, not by sounding right. KEEP = ≥2 recorded incidents, stays prose until a gate exists. BRIEF = 1 incident, moves into `harness brief <kind>`'s per-kind reminder list. DELETE = 0 incidents.

Sources searched: `docs/lessons/infra_incidents.md` (84 §), `docs/lessons/gate_failure_incidents.md` (60 §), `runs/friction.jsonl` (51 rows), `runs/review.jsonl` (170 rows). Agents were strict: tangential mentions excluded; same event recorded by two sessions counted once.

| # | Rule (short name) | Count | Proposal | Key evidence |
|---|---|---|---|---|
| 1 | GPUs — ownership is a controller decision | 9 | KEEP | infra:§57, §15, §126; friction:#12,#45,#47; review:b0-16,b0-24,fb-2f2f |
| 2 | Kill not finished until nvidia-smi says free | 4 | KEEP | friction:#12,#47; review:b0-16,fb-2f2f |
| 3 | Lanes: 7-card block + 1 lane card | 3 | KEEP | friction:#10; review:tilerl-17,b0-24 |
| 4 | Small jobs queue on lane, never spill | 3 | KEEP | friction:#10; review:tilerl-17,b0-24 |
| 5 | Lane holds one job at a time | 1 | BRIEF | friction:#10 (probes serialize on lane) |
| 6 | Judge cost in seconds vs run's own spend | 2 | KEEP | infra:§15; review:tilerl-17 |
| 7 | Dropped tn tunnel doesn't end command | 0 | DELETE | no incident in any source |
| 8 | Card claims live where the job runs | 1 | BRIEF | friction:#45 (b0 launched via run_ddp.sh, no claim in pod's runs/claims) |
| 9 | File transfer via podput | 3 | KEEP | friction:#29,#30,#31; review:#27 (bare redirect truncated file) |
| 10 | tn exec vs ~/bin/pod filesystem views | 0 | DELETE | no incident in any source |
| 11 | Index must equal HEAD before merge | 7 | KEEP | friction:#13,#21,#33,#35,#37,#48; infra:§175 |
| 12 | Conflicting path needs a commit first | 4 | KEEP | friction:#21,#35,#37,#48 |
| 13 | pod_push only ever ADDS | 0 | DELETE | no incident about a deletion not propagated |
| 14 | Only refusing: line means nothing shipped | 2 | KEEP | friction:#31,#35 (output filter hid refusal/error) |
| 15 | pod_head_manifest.txt NOT tracked | 14 | KEEP | friction:#1-4,#13,#18-24; review:e1-19,e1-20 |
| 16 | What is reachable, measured with -4 | 0 | DELETE | no incident; it's a measurement record, not a rule |
| 17 | Reachability changes; fetcher carries mirror chain | 0 | DELETE | no incident in any source |
| 18 | Pod frozen from training launch until run ends | 1 | BRIEF | friction:#32 (pod_push refused while run_ddp.sh live) |
| 19 | cfg_default raises not returns None | 0 | DELETE | no incident; it's a code convention, not an operational rule |
| 20 | Ledger takes names from scores (--name X) | 0 | DELETE | no incident about orphan-score attribution |
| 21 | Each session in own worktree | 2 | KEEP | friction:#33,#37 (e1 staged edits in integration tree, raced controller merge) |
| 22 | Commit within 30 min of touching a file | 1 | BRIEF | friction:#48 (uncommitted append silently discarded) |
| 23 | ruff format only on files you created | 0 | DELETE | origin event (2026-08-31) deleted in restructure; not in any source |
| 24 | Commit as soon as change works (dup of 22) | 1 | BRIEF | friction:#48 (same event as rule 22) |
| 25 | Stage by path, never git add -A/. | 1 | BRIEF | friction:#35 (pathspec error hidden by output filter) |
| 26 | harness task/friction write tree's ledger | 0 | DELETE | origin event (AGENTS.md:292) not in any source |

## Summary

- **KEEP (11)**: rules 1, 2, 3, 4, 6, 9, 11, 12, 14, 15, 21
- **BRIEF (6)**: rules 5, 8, 18, 22, 24, 25
- **DELETE (9)**: rules 7, 10, 13, 16, 17, 19, 20, 23, 26

## Notes

- Rules 22 and 24 are near-duplicates in the table. Both have 1 incident (the same friction:#48 event). If merged, the combined rule has 1 incident → BRIEF.
- Rule 15 (manifest NOT tracked) has the highest count (14) but the underlying issue was fixed by untracking the file (56fa71b5). The 14 incidents are historical. The rule stays as KEEP because the manifest's generated status is still a live discipline.
- Rule 19 (cfg_default raises) and rule 20 (ledger names from scores) are code conventions, not operational rules. They belong in the harness selftest contract, not in AGENTS.md's operational rule table.
- Rules 16 and 17 are about network reachability. They have zero incidents because the environment has been stable. DELETE is safe — if reachability changes, a new rule can be written from the incident.
- Rule 26's origin event (AGENTS.md:292: "harness task refused to run in a worktree, so 'run it in the main checkout' sent a session into the one tree where sessions overwrite each other") is recorded in AGENTS.md prose but not in any of the four incident sources. The agent counted it as 0 because the task said to search the four sources.

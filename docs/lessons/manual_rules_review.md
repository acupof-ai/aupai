---
question: Which of the 31 manual rules in _MANUAL_RULES have recorded incidents, and what should happen to each?
status: open
source: 4 parallel agents + AGENTS.md prose as fifth source; 2026-09-05
---

# Manual rules review — KEEP / BRIEF / DELETE proposal

Criterion: a rule is kept by evidence, not by sounding right. KEEP = ≥2 recorded incidents, stays prose until a gate exists. BRIEF = 1 incident, moves into `harness brief <kind>`'s per-kind reminder list. DELETE = 0 incidents across all five sources.

Sources searched: `docs/lessons/infra_incidents.md` (84 §), `docs/lessons/gate_failure_incidents.md` (60 §), `runs/friction.jsonl` (51 rows), `runs/review.jsonl` (170 rows), `AGENTS.md` prose (dated incidents in rule bullets). Agents were strict: tangential mentions excluded; same event recorded by two sessions counted once.

The first pass searched four sources and missed 5 rules that live in `_MANUAL_RULES` but not the coverage table. 4c ruled AGENTS.md prose is a fifth source: rules whose origin incidents are dated there count. This pass adds all 31 rules.

| # | Rule (short name) | Count | Proposal | Key evidence |
|---|---|---|---|---|
| 1 | GPUs — ownership is a controller decision | 9 | KEEP | infra:§57,§15,§126; friction:#12,#45,#47; review:b0-16,b0-24,fb-2f2f |
| 2 | Kill not finished until nvidia-smi says free | 4 | KEEP | friction:#12,#47; review:b0-16,fb-2f2f |
| 3 | Lanes: 7-card block + 1 lane card | 3 | KEEP | friction:#10; review:tilerl-17,b0-24 |
| 4 | Small jobs queue on lane, never spill | 3 | KEEP | friction:#10; review:tilerl-17,b0-24 |
| 5 | Lane holds one job at a time | 1 | BRIEF | friction:#10 (probes serialize on lane) |
| 6 | Judge cost in seconds vs run's own spend | 2 | KEEP | infra:§15; review:tilerl-17 |
| 7 | Dropped tn tunnel doesn't end command | 1 | BRIEF | AGENTS.md:241 (2026-09-03, 206 GB cp filled pod to 100%, rm -rf lost 10 min) |
| 8 | Card claims live where the job runs | 1 | BRIEF | friction:#45 (b0 launched via run_ddp.sh, no claim in pod's runs/claims) |
| 9 | File transfer via podput | 3 | KEEP | friction:#29,#30,#31; review:#27 (bare redirect truncated file) |
| 10 | tn exec vs ~/bin/pod filesystem views | 1 | BRIEF | AGENTS.md:239 (2026-09-03, host-view grep agreed with stale sha; "two wrong views agreeing") |
| 11 | Index must equal HEAD before merge | 7 | KEEP | friction:#13,#21,#33,#35,#37,#48; infra:§175 |
| 12 | Conflicting path needs a commit first | 4 | KEEP | friction:#21,#35,#37,#48 |
| 13 | pod_push only ever ADDS | 1 | BRIEF | AGENTS.md:255 (2026-09-02, 69 files deleted from main still on pod, all gates green) |
| 14 | Only refusing: line means nothing shipped | 2 | KEEP | friction:#31,#35 (output filter hid refusal/error) |
| 15 | pod_head_manifest.txt NOT tracked | 14 | KEEP | friction:#1-4,#13,#18-24; review:e1-19,e1-20 |
| 16 | What is reachable, measured with -4 | 1 | BRIEF | AGENTS.md:258 (2026-08-30, Errno 99 false-negative matrix nearly retired pretraining slot) |
| 17 | Reachability changes; fetcher carries mirror chain | 1 | BRIEF | AGENTS.md:260 (2026-08-31, hf-mirror dead all day, math/CoT fetches stalled hours) |
| 18 | Pod frozen from training launch until run ends | 1 | BRIEF | friction:#32 (pod_push refused while run_ddp.sh live) |
| 19 | cfg_default raises not returns None | 0 | MOVE | code convention, not operational rule → selftest contract section |
| 20 | Ledger takes names from scores (--name X) | 0 | MOVE | code convention, not operational rule → selftest contract section |
| 21 | Each session in own worktree | 2 | KEEP | friction:#33,#37 (e1 staged edits in integration tree, raced controller merge) |
| 22 | Commit within 30 min of touching a file | 1 | BRIEF | friction:#48 (uncommitted append silently discarded) |
| 23 | ruff format only on files you created | 1 | BRIEF | AGENTS.md:398 (2026-08-31, 61-line reformat buried others' work, invited checkout that deleted device gate) |
| 24 | Commit as soon as change works (dup of 22) | 1 | BRIEF | friction:#48 (same event as rule 22) |
| 25 | Stage by path, never git add -A/. | 1 | BRIEF | friction:#35 (pathspec error hidden by output filter) |
| 26 | harness task/friction write tree's ledger | 1 | BRIEF | AGENTS.md:292 (b0 2026-09-04, harness task refused in worktree, session sent to integration tree) |
| A | Hook edit in branch doesn't run until merged | 3 | KEEP | AGENTS.md:401 (2026-09-01 selftest red under 5 green hook lines; e1 2026-09-02 build_agentic_sft.py registered on branch, never ran); friction:#11 |
| B | PID only meaningful in namespace that read it | 5 | KEEP | AGENTS.md:240 (2026-09-01 host pid in container, ~80 steps contention; 2026-09-01 pid sets unresolvable; 2026-09-03 zombie pid, 31 min wait on dead pid; for-loop launch race); friction:#12 |
| C | Never checkout/restore a file you didn't write | 4 | KEEP | AGENTS.md:253 (2026-08-30, --autostash rolled back 3b's build_corpus.py); friction:#48,#42 |
| D | Deletion needs per-file check for glob/loaders | 2 | KEEP | AGENTS.md:61 (de-5 deleted progress_feed.py on reachability reading, froze page; vet_programs.py:37 glob, 23 generators read as unreferenced) |
| E | Every delivery has a second reader | 1 | BRIEF | AGENTS.md:391 (user order 2026-08-31; 170 review rows are practice, not incidents) |

## Summary

- **KEEP (15)**: rules 1, 2, 3, 4, 6, 9, 11, 12, 14, 15, 21, A, B, C, D
- **BRIEF (14)**: rules 5, 7, 8, 10, 13, 16, 17, 18, 22, 23, 24, 25, 26, E
- **MOVE to selftest contract (2)**: rules 19, 20
- **DELETE (0)**: none — every rule has at least one dated incident in AGENTS.md prose or the other four sources

## BRIEF assignment table (rule → kind)

For de's `harness brief <kind>`. Rules 22+24 merged (same incident, same rule).

| kind | rule | short name |
|---|---|---|
| gpu | 5 | lane holds one job at a time |
| gpu | 8 | card claims live where the job runs |
| pod | 7 | dropped tn tunnel doesn't end command |
| pod | 10 | tn exec vs ~/bin/pod filesystem views |
| pod | 13 | pod_push only ever ADDS |
| pod | 18 | pod frozen from training launch until run ends |
| net | 16 | what is reachable, measured with -4 |
| net | 17 | reachability changes; fetcher carries mirror chain |
| git | 22+24 | commit within 30 min / as soon as change works |
| git | 23 | ruff format only on files you created |
| git | 25 | stage by path, never git add -A/. |
| git | 26 | harness task/friction write tree's ledger |
| review | E | every delivery has a second reader |

## Notes

- Rules 22 and 24 are near-duplicates. Both have 1 incident (the same friction:#48 event). If merged, the combined rule has 1 incident → BRIEF.
- Rule 15 (manifest NOT tracked) has the highest count (14) but the underlying issue was fixed by untracking the file (56fa71b5). The 14 incidents are historical. The rule stays as KEEP because the manifest's generated status is still a live discipline.
- Rules 19 and 20 are code conventions, not operational rules. They belong in the harness selftest contract, not in AGENTS.md's operational rule table. MOVE, not DELETE.
- The 5 missing rules (A-E) are in `_MANUAL_RULES` but not in the coverage table. They are in AGENTS.md prose (lines 401, 240, 397, 61, 391). All have ≥1 incident.
- Rule E (second reader) has 170 review rows, but those are routine practice, not incidents. The origin event (user order 2026-08-31) is the one incident.

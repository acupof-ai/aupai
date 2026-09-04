---
question: After the 2026-09-04 audit, what is true now, and what is removed, closed or fixed to leave a clean environment before any new experiment?
status: open
source: user order 2026-09-04 ~04:50Z ("把现状理清楚，把环境弄干净"); findings in runs/audit_0904/
---

# Post-audit: state and cleanup

The freeze lifts for the two things below and nothing else. No new experiment, fetch, build or
scoring launches until both are closed; the one GPU job in this phase is the doc_cu re-score,
last.

## 1. State of record — `docs/standards/state_0904.md` (owner 44, by 08:00Z)

One page. For each area, three lists: **stands** (numbers and decisions that survive the audit,
each with its fact id or ledger row), **retracted or qualified** (with the finding id), **unmeasured**
(named). Area owners send 44 their three lists by 06:30Z from their own reports; 44 joins, no
new claims. 98 renders it above the audit section.

## 2. Cleanup — `runs/audit_0904/cleanup.jsonl` (one row per item: id, owner, action, state, evidence)

Rules: one item per commit; a deletion reads the target first and, for anything not created by the
deleter, is broadcast in the row 24 h before it runs (existing rule); a kill is by exact PID in the
namespace that read it, and is done when `ps -o stat=` and `nvidia-smi` agree; no fix widens beyond
the finding it closes.

| id | owner | item | from |
|---|---|---|---|
| C1 | tilerl | kill by exact PID: 3 `until [ -f …]` loops, 6 stale aupai `tail -f`, 307 orphaned `tail -F runs/events.jsonl` (PR-11 called them another project's on cwd alone; cwd is the container default, so origin is unknown; ruling: a watcher whose stdout pipe has no reader delivers to nobody and is disposable whatever spawned it, verified per PID before the kill); state why the 3,958 zombies cannot be reaped from inside the container and what would | PR-11 |
| C2 | b0 | 12:03Z prune as scheduled; then delete the 17 stale pod copies and the `_b0_*` scratch files (own), list the other 168 untracked pod files by owner for 24 h broadcast | PR-6/7 |
| C3 | 3b | broadcast the ~80 GB unreferenced corpus (24 `web_cci3_p*`, 115 loose `batch_*.jsonl`) for 24 h, then delete; stamps stay | CD-6 |
| C4 | de | one transport for every `runs/*.jsonl`, union by event; then pull the pod's rows and close every `running` row with no process; triage the 62 open tasks (close or re-block on a task id) | DL-11, DL-3 |
| C5 | de | `merge_main.sh --no-ff`; `no_conflict_markers` cost (walk once, cache); register `check_test_record_after_last_stage`; `_ckpt_names` reads value too; `walk_tracked` scope stated | DL-22, DL-1, DL-20, DL-8, DL-2 |
| C6 | e1 | score_matrix capture `(stdout + stderr)` at :281 and :304; `#cu` suffix on rows; `cu_path` plumbed through score_matrix, ppl, domain_bpb, math_bpb; loader warning only when a cfg was given | MT-12, E5, E10, MT-13 |
| C7 | tilerl | `eff.kv_pool_undersized_for_serving` → retracted with `retracted_value`; new fact against `48ae458`, citing repository and commit | MT-1 |
| C8 | 44 | fact edits: `ds.n2_params_vs_data_matched_compute` boundary "cu_none, doc_cu pending"; `eff.eval_path_cu_artifact_ce` sentence qualified; `dq.agentic_credential_split` and the four `/tmp` contamination facts gain the instrument-lost boundary; E15/UF timestamps; `adversarial_review.md` citation carries its sha | E3, E2, F10, E14, E15, F2 |
| C9 | 98 | amend the three unamended page lines (UF-1/2/3) with a dated correction; store lines gain a date | UF-1..4 |
| C10 | b0 | `launch_tests` L16 row re-recorded under the (test, shape) key; `recipe_provenance` d768 group already on main, note it in the roadmap | MT-4, MT-7 |
| C11 | e1 | after C6 lands: ONE lane job re-scoring on doc_cu the checkpoints behind published numbers: both N2 legs, the Stage D/E arms, the control arm (domain_bpb first) | E1, E3, E10 |

C11 is the only card job and runs last. Everything else is CPU or an edit and runs in parallel,
one commit each, merged with `scripts/merge_main.sh`. Done means the row's `state` is `done` with
the commit or the `ps`/`df` reading as evidence.

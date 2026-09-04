---
question: What is the true state of the project on 2026-09-04, read from artifacts rather than from messages, ledgers or memory?
status: open
source: user order 2026-09-04 (full stop, whole-team audit); reports under runs/audit_0904/
---

# Whole-project audit, 2026-09-04

User order: all new work stops. Jobs already running finish; nothing new is launched, fetched,
built or hand-read. Every session audits one area. The controller designs the method, samples
the reports, and rejects or returns them; the controller writes no report.

## Principles

1. **Artifacts, not claims.** A finding cites the file and line, the artifact path, or the exact
   command whose output it quotes. A ledger row, a message, a fact's `value` or a doc sentence is
   a claim to be checked, never evidence for itself.
2. **Every number carries its basis.** Population, instrument, unit, date. A "clean" or "passes"
   statement names what the instrument cannot see.
3. **Population before property.** For every guard, check, scan or filter: is the set it runs over
   the set the property is about? Today's incidents (holdout population, launch_tests key,
   selftest gating, launch-gate pod partition) were all this shape.
4. **Instruments are tested on a broken world before their output is believed.** An audit script
   that has not been shown to fail on a known defect has not run.
5. **Audit only.** No fixes, no refactors, no ledger edits, no new launches. The only writes are
   the report and audit scripts under `runs/audit_0904/`. A defect found is written down, not
   repaired. A branch merged with `scripts/merge_main.sh` ships every commit on it, so a session
   holding pre-freeze commits merges only its report branch or rebases the report onto main
   first; "no new work" did not say this and two frozen fixes reached main that way (de, §6).
   Exception: a live security exposure (credential in a tracked or shared file) is
   reported to the controller at once, and the file is quarantined by rename, never printed.
6. **Reproduce a sample, do not re-derive everything.** Where a report covers hundreds of items,
   recompute a fixed-seed sample of at least 30 and state the sample.
7. **Report what was not checked** as a named list, not by silence.

## Deliverable

`runs/audit_0904/<area>.md`, committed on the auditor's branch, merged with
`scripts/merge_main.sh`. Sections, in this order:

1. Scope: the paths and artifacts covered; the paths deliberately excluded.
2. Method: the commands run, the sample seeds, the broken-world test of each audit script.
3. Population counts: how many items exist, how many were read, how many sampled.
4. Findings table: `id | severity | claim as published | evidence (path/cmd) | what contradicts it`.
   - S1: a published number, decision or fact is wrong or unsupported.
   - S2: a guard, check or filter is silent on part of its population, or its population is
     narrower than its property; nothing wrong is known yet, but nothing would be caught.
   - S3: hygiene: stale docs, dead rows, naming, drift with no consequence found.
5. Blind spots of this audit: what the method could not see.
6. Open questions for the controller: at most five, each answerable with one decision.

First report within 3 hours of this file landing; a partial report at 3 hours beats a complete
one at 6. Reports are read by the auditor's roster pair, who recomputes three findings and appends
a `## Pair check` section naming which three and whether they held.

## Areas and owners

| area | owner | pair | scope |
|---|---|---|---|
| model and training code | b0 | tilerl | `model.py`, `train.py`, `sft_math.py`: document isolation, loop seam, cursor/resume, fp8, checkpoint save/load, `Cfg` frozen keys; every `facts/efficiency.json` and `facts/smelt_deeploop.json` entry against its cited source |
| evaluation and held-out | e1 | 3b | `eval/*`, `datagen/holdout.py`, the registry, `runs/score_matrix.jsonl`, `runs/*blocks*.jsonl`; every eval's prompt format on a base checkpoint; which published numbers were taken on the cu_none path; contamination facts and their populations |
| instruments and ledgers | de | 44 | `scripts/harness.py` checks (each: population vs property, broken world real), `scripts/launch_gate.py`, hooks, `scripts/card_claim.py`, `scripts/pod_pull_ledgers.py`; ledger state pod vs repo for every `runs/*.jsonl`; `runs/tasks.jsonl` open rows vs reality |
| corpus and data | 3b | b0 | every domain named by `data/mix_200m_*.json` and `data/mix_30b*.json`: stamp, fingerprint, `filters_fp`, supply vs claimed tokens, which holdout population it was built against, dedup state; `data/raw/*` inventory with provenance or its absence |
| pod and repository state | tilerl | b0 | pod vs main file drift beyond the manifest, orphan processes, disk by directory, checkpoint inventory vs `runs/pod_ckpt_candidates_2026-09-04.txt` vs pins vs facts; card assignment vs `nvidia-smi` vs claims |
| facts and documents | 44 | de | every `facts/*.json` entry: status, source exists, config non-empty, a fixed-seed sample of 30 values recomputed from the cited artifact; `docs/lessons`, `docs/audits`, `docs/standards/roadmap_0903.md` claims against facts |
| user-facing statements | 98 | e1 | `~/aupai-progress.html`, `EXPERIMENTS.md`, the controller's replies logged in `runs/board.jsonl`: every number and verdict traced to a fact or ledger row; the untraceable ones listed |

Running jobs that continue: Stage E arm 1 (`b0_se_16lnew_1b`, cards 0+1), the v14 agentic SFT
build (laptop, e1). Stage E arm 2 (`b0_se_looped_2b`) was launched 03:22:03Z and killed by b0 on
the stop order ~80 s later, in compile, with zero steps and no checkpoint; it is not running and is
not relaunched during the audit. Not started and not to be started until the audit closes: the
third 2B arm, the OT3 fetch, hand-read #70, the v14 hand-read, any re-score. The 12:03Z
checkpoint prune is a scheduled deletion under a prior user order and executes as planned.

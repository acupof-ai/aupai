---
question: What is measured next on v41_ced_0923, which decisions wait on the user, and what happens after training ends?
status: open
source: runs/prereg.jsonl#v41_ced_0923@amended_1; runs/heval_auto_loop.sh; open PRs #670 #671 #673 #664 #674; flat_refactor_sequence_0923.md
---

# 07 — Next measurements and decisions

## While training runs (now → step 34,331)

The auto loop scores one rstrip point per checkpoint, so the 12k–34k points at every
2,000-step save land without intervention (pod `runs/heval_auto_loop.sh`, PR #670).
Reading cadence:

| point | step | what it answers |
|---|---|---|
| 10k | done | 8/164; first point past the early phase |
| 20k (~15.7B tok) | next milestone | whether the single-digit climb resumes after the 8k→10k dip (8k was 13/164; ±3–4 tasks is noise, so one more point does not decide a trend) |
| 30k (~23.6B tok) | pre-anneal | trajectory entering the anneal phase |
| 34,331 | anneal start | last main-composition point; anneal mix runs for the final 3,815 steps |

Val points continue every 500 steps. Stop rule 4 (HumanEval inconsistent with reaching
30%) escalates to the user rather than stopping the run unilaterally; no point so far
has triggered it.

## User decision before the anneal: L3

The anneal composition raises code_ultra_l3_noexec_dc from weight 0.314634 to anneal
0.421333 — the largest composition shift in the schedule. The open decision is whether
to change the L3 input before that phase: the mix currently trains the no-exec
aggregate (static dedup + nontriviality + decontam; user order 2026-09-11), while the
execution-filtered L3 build exists outside the mix and was retained for an A/B. Supply
is not a constraint either way (L3 ratio 0.366, 26.7B pool). Changing the anneal mix
after training starts is a recipe change and is the user's call; the registered row
does not pre-authorize it.

## At run end (step 38,146)

1. GPU gate read on the freed 8-card block: rstrip pass@1 n=164 (the gate number) plus
   the standard arm beside it. Prerequisite: the `humaneval_sample.py` truncate fix
   passing `entry_point` (genA), named in the registered amendment.
2. Full score matrix on one card (`eval/score_matrix.py`); generative metrics use
   continuation/rstrip form, never an unseen prompt prefix.
3. Gate verdict against ≥30%; close the experiment row with the number and the reading
   artifact. The historical end-of-run checklist is
   [`v41_gate_0911_end.md`](v41_gate_0911_end.md) (its commands reference the destroyed
   0911 run and do not execute; the step order carries over).

## Code cleanup after the run

Ruling 2026-09-23: while the run is live, no PR touching its import path
(model.py/train.py) merges, because a crash-resume would load the new code. The work is
prepared as drafts, sequenced in [`flat_refactor_sequence_0923.md`](flat_refactor_sequence_0923.md)
with the surface inventory in [`non_ced_surface_analysis_0923.md`](non_ced_surface_analysis_0923.md):

| item | state | PR |
|---|---|---|
| remove the flat single-pass branch and flat-only launchers (CED encoder shares the entry builder, so this is a branch refactor; byte-identical CED forward digest is the acceptance) | draft, waits for run end | #673 open |
| AttnRes removal (separate architecture axis; one old k3 checkpoint still exercises the loader's auto-disable path, which must keep loading legacy state) | contract test first, then removal | #664 open, step 5 of the sequence |
| HumanEval CPU sharding + auto loop | open | #670 |
| resume-gate CI fix (does not touch model code) | open | #671 |
| merge_main resolved-tip verdict | open | #674 |

Not scheduled: deleting the shared `entries_per_doc` builder (it is the CED encoder),
HCA removal (recipe decision), and anything under `v41f/` (a separate live track with
its own preregs).

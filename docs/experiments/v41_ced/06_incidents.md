---
question: What incidents hit the CED line between the pivot and now, what caused each, and what fixed it?
status: measured
source: PR descriptions #648 #661 #667 #668 #671 #674; runs/friction.jsonl; runs/review.jsonl
---

# 06 — Incidents and fixes

One row each: symptom, cause, fix, PR. Read 2026-09-23; OPEN fixes are marked.

| # | symptom | cause | fix | PR |
|---|---|---|---|---|
| 1 | S2 smoke crashed before stepping, fp8 shape error in the CED decoder global-KV projection (M=2072) | fp8 linear requires the M axis divisible by 16; the projected doc-block count is not | pad the decoder W_KV/W_Z projection M-axis to 16; S2 then ran 300 steps, 0 NaN | #648 merged 2026-09-22 |
| 2 | CI gate `gate_resume_equivalent_to_uninterrupted` intermittently red; 15 shas; same-sha push vs pull_request runs diverged 6/7; in-VM retry reproduced the identical signature (max\|Δ\|=1.334e-2, 506,533/524,288 diffs) | runner-persistent oneDNN bf16 brg_matmul OpenMP K-reduction under 2 intra-op threads plus CPU oversubscription on Intel-class CI runners; control and restart are sibling processes in one VM, so the divergence rides the runner, not the invocation — measured 6/6 red with oneDNN on + 2 threads under burners, 0/6 with either oneDNN off or 1 thread | per-arm host/ISA/env fingerprint in the on-red dump; same-VM attempt-2 relabeled as the known VM-correlated red; workers pinned to 1 thread plus an independent-runner verify matrix | #667 merged, #668 merged, #671 OPEN |
| 3 | `merge_main.sh` refused merges on a stale main CI verdict; four measured occasions returned an ancestor's failed/cancelled run as "newest on main" (17-, 10-, 2-day-old rows) | `gh run list --branch main --limit 1` row order is unstable, and main push runs cancel each other in the concurrency group, so the newest completed run is usually not the tip's | key the verdict on the resolved tip sha (fallback created-after-tip-time), page --limit 20, auto-recover a moved origin | #674 OPEN |
| 4 | every merge_main hook in the repo went red for a window; unrelated PRs' CI failed | three docs cited `ci.yml:<line>`; each was accurate when written and wrong after unrelated pushes (two of them in `runs/non_ced_surface_0923.md`); the new check and the citations landed in three separately-green PRs, jointly red only on main | check `ci_line_number_citations` rejects ci.yml line citations; the three citations replaced with step-command text | #634 added the check, #661 fixed the citations; friction row runs/friction.jsonl:578 |

Incident 2 is CI-only and never affected the training process: the gate compares two
short CPU train steps on the runner, not the pod run. Its open fix #671 changes CI
worker threads and adds the runner matrix; the model code is untouched.

Incidents during the run window that did not become code changes: the S3 smoke's rc=1
was the scoring stage failing to get a lane card on the 8-card block (training itself
passed); one S2 launch was refused by the training-scope drift gate (unpushed model.py,
correct refusal); the long-running step-2000 single-process HumanEval is slow by CPU
cost, not failed.

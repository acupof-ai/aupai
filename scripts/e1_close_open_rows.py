"""Close e1's 9 open task rows into 6e's four buckets, carrying every field forward.

6e's constraint: "a closing row carries opened/task/why/prior/pair forward and adds state+evidence,
or the hooks read it as a second task". So each row is the ORIGINAL dict with state/closed/result/
evidence/commit added, never a fresh dict -- a rebuilt row loses `opened` and the hooks then see a
new task rather than a closed one.

Buckets, and the evidence each was decided on (from the TREE, per 6e -- b0 found two rows called
"obsolete" that were actually delivered):
  A delivered   e1-22 (E9-E12 in the audit report), e1-24 (data/sft/agentic_v14.jsonl, 4823 rows,
                gate passed), e1-28 (scripts/e1_28_verified_ids.py + runs/e1_28/*.json)
  B small       e1-26 (MFU basis: measured 17 of 18 facts lack it -- but this EDITS FACTS, which is
                44's area under C8, so it closes as handed off, not as done by me)
  C needs cards e1-25 (Pythia control SFT), e1-27 (SFT lr sweep), e1-29 (N3 v2 seed sigma)
  D obsolete    e1-21, e1-30 (both are docs/lessons/post_pretrain_plan.md section 5, and both
                predate the audit's finding that the eval instruments they depend on were defective)

C11/C12 stay OPEN as 6e required, so the queue is never empty. They are appended separately as
e1-36 and e1-37 -- NOT e1-31/e1-32, which the harness rejected as an id collision: the folded
"latest row per id" view hides ids whose newest row is `done`, so the highest id in use has to be
read over EVERY line, not over the fold. e1-31 and e1-32 are N7 Stage A and N8, both closed.
"""
# restartable: an interrupt is cheap because the only write is one append of <=9 lines to
# runs/tasks.jsonl at the very end, after every row has been read and validated. A partial run writes
# nothing; a re-run re-reads the ledger and skips any id whose latest state is already done/dropped,
# so appending twice is impossible. Nothing here touches a card, the pod, or a checkpoint.
import json
import os
import subprocess

ROOT = "/Users/bytedance/code/aupai-e1"
TASKS = os.path.join(ROOT, "runs", "tasks.jsonl")
NOW = subprocess.run(["date", "-u", "+%Y-%m-%d %H:%M"], capture_output=True,
                     text=True).stdout.strip()
COMMIT = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()

CLOSE = {
    "e1-22": dict(
        state="done",
        result="DELIVERED as audit findings E9-E12. All 43 eval/*.py enumerated (not 46 -- that "
               "count included __init__.py, _devs.sh and __pycache__), each read for the prompt "
               "builder it calls and whether that prefix occurs in pretraining. E9 is the clean "
               "result (the ChatML-prefix rule holds where it applies), E10 names four scorers "
               "passing no document mask, E11 two scorers I had claimed to read and had not, E12 "
               "the sibling-scorer duplication. The sweep was delegated; every line cited in "
               "E9-E12 I re-verified at the file.",
        evidence="runs/audit_0904/eval_heldout.md (E9, E10, E11, E12; findings table rows)",
    ),
    "e1-24": dict(
        state="done",
        result="DELIVERED. data/sft/agentic_v14.jsonl, 4,823 rows / 21,602,682 B, built by "
               "scripts/build_agentic_sft.py through scripts/loader.format_agentic. The secret-scan "
               "gate passed (0 real-credential rows, 2 allowed types over 16,288,781 chars) and 44's "
               "50-row hand-read found 0 residual credentials. The build also produced the finding "
               "that matters more than the pack: 59 of 62 credential-bearing episodes were caught "
               "ONLY by the new CLI/env tier, so they were in v13 and every earlier pack. E16 "
               "establishes no checkpoint ever trained on any agentic pack.",
        evidence="data/sft/agentic_v14.jsonl; runs/e1_v14_agentic_build_2026-09-04.log; "
                 "runs/redaction_handread_v14.tsv; runs/audit_0904/eval_heldout.md (E16, E17)",
    ),
    "e1-28": dict(
        state="done",
        result="DELIVERED with the population corrected twice. The measurable held-out set is 7,523 "
               "of 10,421 items (an item can only hit if its answer has >=13 word tokens), of which "
               "2,114 hit the mix_200m_4b corpus and 5,409 are verified clean; ids_sha "
               "54cef7869c7b57c0, scanner fingerprint 0abccb3c850b969d, both reproduced locally "
               "against the pulled artifact. The 316-item predecessor is RETRACTED: "
               "datagen/holdout.py's 4-path EVAL_FILES omitted control_sft_text_heldout.jsonl, so "
               "the guard fingerprinted all 9 domains and excluded nothing. 296 of the old 316 are "
               "among the new 2,114 and 20 are not, so the new set is not the old one shrunk.",
        evidence="scripts/e1_28_verified_ids.py; runs/e1_28/e1_28_per_domain_alone.json; "
                 "runs/audit_0904/eval_heldout.md (E7, and the per-domain holdout-population table "
                 "in section 3)",
    ),
    "e1-26": dict(
        state="done",
        result="MEASURED AND HANDED OFF, not fixed by me. 18 facts/*.json entries quote MFU and "
               "exactly 1 carries the basis; 17 do not: repo.loop_from_scratch_stage_d, "
               "repo.loop_not_adopted_equal_compute, be.ctx_length_p324, eff.h20_mfu_200m, "
               "eff.attnres_internal, eff.kda_kernel_path, eff.nccl_proto_simple_ab, "
               "eff.short_conv_shifted_madd, eff.microbatch_32_oom, "
               "eff.max_autotune_dynamic_shape_noship, eff.step_remainder_attribution, "
               "eff.seam_dynamo_disable, eff.grad_ckpt_inverts_with_depth, "
               "eff.depth_is_not_the_mfu_gap, eff.w7_peak_memory_b32_fits, "
               "eff.step_roofline_p200m_4card, eff.padded_vocab_table_no_pay_200m. The row asked me "
               "to EDIT those facts; facts are 44's area and cleanup_0904 C8 is the fact-edit item, "
               "so the list goes to 44 rather than my writing 17 entries in someone else's file. "
               "The basis itself is mfu = 6*n_params*tps/peak_tflops at train.py:2560, and the "
               "reason it matters is b0's VE arm reading 55% vs base 48% at identical 57K tok/s/gpu "
               "-- the move is the +16.3% params in the numerator, not throughput.",
        evidence="the 17 ids above, enumerated over facts/*.json; train.py:2560; "
                 "docs/standards/cleanup_0904.md C8 (44's fact-edit item)",
    ),
    "e1-25": dict(
        state="dropped",
        result="PARKED: EXPERIMENT. Needs the OT3 fetch chain and two SFT runs on cards, and the "
               "audit's freeze allows only C11. Also now rests on a defective instrument: this row's "
               "deliverable is a readable score for the 200M run, and the metrics it would use are "
               "the ones E10 shows were taken cu-blind and E19 shows never ran at all (domain_bpb). "
               "Re-open after C12's domain_bpb fix is exercised on a real checkpoint, not before -- "
               "otherwise the control comparison reproduces the instrument, not the models.",
        evidence="runs/card_assignment.json (no grant to e1 beyond C11's spent lane); "
                 "runs/audit_0904/eval_heldout.md (E10, E19)",
    ),
    "e1-27": dict(
        state="dropped",
        result="PARKED: EXPERIMENT. A 5-point SFT lr sweep is 5 card jobs. The row's premise also "
               "shifted: it exists because the control's lr was swept while ours ran v5's recipe "
               "once, and the floors it cites (0.451 vs 0.904) are from the comparison whose "
               "held-out population e1-28 retracted. Re-open with the 5,409 verified-clean set as "
               "the evaluation population.",
        evidence="runs/card_assignment.json; scripts/e1_28_verified_ids.py (the 5,409 set that "
                 "replaces the population the floors were read on)",
    ),
    "e1-29": dict(
        state="dropped",
        result="PARKED: EXPERIMENT. Two same-recipe seeds per metric is a card job by construction. "
               "One part is now known-bad rather than merely unrun: the row names humaneval_bpb, "
               "math_bpb and lambada_en as 'the only metrics with cited resolution', and E10/E19 "
               "show math_bpb and humaneval_bpb are single-document scorers (so cu=None is correct "
               "there) while domain_bpb never produced a number. The seed-sigma design stands; the "
               "metric list needs rewriting first.",
        evidence="runs/audit_0904/eval_heldout.md (E10, E19); eval/humaneval_bpb.py:216 "
                 "(one document per forward)",
    ),
    "e1-21": dict(
        state="dropped",
        result="SUPERSEDED by docs/lessons/post_pretrain_plan.md, which is where the SFT-side "
               "long line now lives, and by e1-24 for the pack half (delivered). Checked in the "
               "tree rather than assumed: the plan file exists at docs/lessons/, not "
               "docs/standards/ where the row's sibling e1-30 points. What this row asked for that "
               "the plan does not yet carry is num_id derived from the tokenizer rather than a "
               "literal -- recorded here so dropping the row does not drop that item.",
        evidence="docs/lessons/post_pretrain_plan.md; e1-24 (agentic pack delivered)",
    ),
    "e1-30": dict(
        state="dropped",
        result="SUPERSEDED by docs/standards/cleanup_0904.md and docs/standards/state_0904.md, "
               "which are the post-audit versions of 'what is ready'. The row's own reference is "
               "stale: it names docs/standards/post_pretrain_plan.md section 5 and the file is at "
               "docs/lessons/post_pretrain_plan.md. Its condition (section 5 reaches zero open "
               "items) cannot be the gate any more, because the audit found the eval instruments "
               "that section depends on were defective (E1, E10, E19) -- readiness measured against "
               "them would have certified a broken path.",
        evidence="docs/standards/cleanup_0904.md; docs/lessons/post_pretrain_plan.md; "
                 "runs/audit_0904/eval_heldout.md (E1, E10, E19)",
    ),
}

rows = []
for line in open(TASKS, encoding="utf-8"):
    line = line.strip()
    if line:
        rows.append(json.loads(line))
latest = {}
for r in rows:
    if r.get("id"):
        latest[r["id"]] = r

out = []
for tid, close in CLOSE.items():
    base = latest.get(tid)
    if base is None:
        raise SystemExit(f"{tid} not in the ledger; refusing to invent a row")
    if base.get("state") in ("done", "dropped"):
        print(f"  {tid} already {base['state']}; skipping")
        continue
    row = dict(base)          # carry opened/task/why/prior/pair/owner/socket forward
    row.update(close)
    row["closed"] = NOW
    row["commit"] = COMMIT
    out.append(row)

with open(TASKS, "a", encoding="utf-8") as fh:
    for row in out:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"appended {len(out)} closing row(s) at {NOW}, commit {COMMIT}")
for row in out:
    print(f"  {row['id']:8} {row['state']:8} {row['result'][:66]}")

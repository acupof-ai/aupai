"""Fill drop_reason on e1's five grandfathered dropped rows (de's tasks_well_formed, 7fd8bc68).

Each value is a CONDENSATION of the `result` this session already wrote when it dropped the row,
not a new claim: the reason the work stopped was recorded at the time, in the wrong field for a
check that did not exist yet. Nobody else may write these -- only the owner knows why each stopped,
and a value written by a third party is a fabricated reason in the field whose purpose is to carry a
real one (harness.py's own docstring).

The two shapes, kept distinct because they mean different things to a reader deciding what to
re-open: SUPERSEDED means the work moved somewhere and the row is redundant; PARKED means the work
is still wanted and is blocked on a named resource, with the re-open condition stated.

# restartable: one append per id, all at the end, after every row is read and validated. A re-run
# skips any id whose latest event already carries a non-empty drop_reason, so appending twice is
# impossible. A partial run writes nothing. Nothing here touches a card, the pod, or a checkpoint.
"""
import json
import os
import subprocess

ROOT = "/Users/bytedance/code/aupai-e1"
TASKS = os.path.join(ROOT, "runs", "tasks.jsonl")
NOW = subprocess.run(["date", "-u", "+%Y-%m-%d %H:%M"], capture_output=True,
                     text=True).stdout.strip()

REASONS = {
    "e1-21": "SUPERSEDED: the SFT-side long line moved to docs/lessons/post_pretrain_plan.md and "
             "the pack half was delivered by e1-24, so nothing was left for this row to carry. "
             "One item did NOT move with it and is recorded in the closing result rather than "
             "lost: num_id derived from the tokenizer instead of a literal.",
    "e1-25": "PARKED, needs cards AND a working instrument. Two SFT runs plus the OT3 fetch chain "
             "is card work, and the audit's freeze allowed only C11. It also stopped resting on a "
             "sound instrument mid-row: its deliverable is a readable score for the 200M run, and "
             "audit_0904 E10/E19 show those metrics were taken cu-blind and that domain_bpb never "
             "produced a number at all. Re-open after C12's fix is exercised on a real checkpoint "
             "-- earlier, the control comparison would reproduce the instrument, not the models.",
    "e1-27": "PARKED, needs cards, and its premise shifted. A 5-point SFT lr sweep is 5 card jobs. "
             "The floors that motivated it (0.451 vs 0.904) come from the comparison whose "
             "held-out population e1-28 RETRACTED, so the row would have swept against a "
             "population that no longer exists. Re-open with the 5,409 verified-clean set as the "
             "evaluation population.",
    "e1-29": "PARKED, needs cards, and its metric list needs rewriting first. Two same-recipe "
             "seeds per metric is a card job by construction. The row names humaneval_bpb, "
             "math_bpb and lambada_en as 'the only metrics with cited resolution'; E10/E19 show "
             "math_bpb and humaneval_bpb are single-document scorers (cu=None is correct there) "
             "while domain_bpb never produced a number. The seed-sigma DESIGN stands; the list "
             "does not.",
    "e1-30": "SUPERSEDED, and its gate condition became unusable. docs/standards/cleanup_0904.md "
             "and state_0904.md are the post-audit versions of 'what is ready'. Its condition -- "
             "post_pretrain_plan.md section 5 reaching zero open items -- cannot be the gate any "
             "more, because the audit found the eval instruments that section depends on were "
             "defective (E1, E10, E19): readiness measured against them would have certified a "
             "broken path. Its own reference was also stale (names docs/standards/, file is at "
             "docs/lessons/).",
}

rows = [json.loads(line) for line in open(TASKS, encoding="utf-8") if line.strip()]
latest = {}
for r in rows:
    if r.get("id"):
        latest[r["id"]] = r

out = []
for tid, reason in REASONS.items():
    base = latest.get(tid)
    if base is None:
        raise SystemExit(f"{tid} not in the register; refusing to invent a row")
    if base.get("state") != "dropped":
        raise SystemExit(f"{tid} is {base.get('state')}, not dropped; refusing to add a "
                         f"drop_reason to a row that was not dropped")
    if str(base.get("drop_reason") or "").strip():
        print(f"  {tid} already carries a drop_reason; skipping")
        continue
    # The ORIGINAL dict plus the one field, so opened/task/why/prior/pair/result/evidence all
    # carry forward. A rebuilt row loses `opened` and the hooks then read it as a NEW task rather
    # than as another event for this one.
    row = dict(base)
    row["drop_reason"] = reason
    row["closed"] = base.get("closed") or NOW
    out.append(row)

with open(TASKS, "a", encoding="utf-8") as fh:
    for row in out:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"appended {len(out)} drop_reason event(s) at {NOW}")
for row in out:
    print(f"  {row['id']:8} {row['drop_reason'][:72]}")

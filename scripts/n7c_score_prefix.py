#!/usr/bin/env python3
"""Score one N7 Stage C prefix arm against its causal twin, both cells.

TWO CELLS PER ARM, and the second is not optional. A prefix-trained checkpoint scored CAUSALLY is
a topology mismatch, and this repo has already been wrong that way once: N7 Stage A read +0.0273
and the number turned out to be the mismatch between a looped-trained model and unlooped scoring,
not the loop. So each arm is scored in the mask it trained in AND causally, and the twin is scored
causally -- the cell it trained in.

  matched   arm checkpoint, arm's prefix mask   <- the experiment
  mismatch  arm checkpoint, causal              <- says whether a difference is the intervention
  baseline  twin checkpoint, causal             <- an EXISTING 500-step run, not retrained

THE READING RULE WAS PRE-REGISTERED before either arm launched (runs/experiments.jsonl,
n7c_p3_prefix3 and n7c_p7_prefix7): adopt only if the paired delta is negative -- better -- by more
than 2x its paired SE on the 164-task intersection AND the per-item count agrees. A delta inside
2 SE is a null. This script prints those numbers; it does not decide.

USAGE
    python3 scripts/n7c_score_prefix.py p3    # ckpt_n7c_p3.pt vs ckpt_n7c_unlooped.pt
    python3 scripts/n7c_score_prefix.py p7    # ckpt_n7c_p7.pt vs ckpt_n7c_looped.pt
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs", "n7c_prefix")

# ckpt, twin, and the LOOP the arm trained under -- p7 is looped 4-7, p3 is not. The loop must be
# applied at scoring time too, or the cell is a loop mismatch on top of the mask question.
ARMS = {
    "p3": {"ckpt": "ckpt_n7c_p3.pt", "twin": "ckpt_n7c_unlooped.pt", "loop": None},
    "p7": {"ckpt": "ckpt_n7c_p7.pt", "twin": "ckpt_n7c_looped.pt", "loop": ("4", "7")},
}
# From runs/n7c_2x2/, the 500-step causal 2x2 already in the repo. Quoted so the comparison needs no
# second read, and named as the same run rather than a fresh one.
PRIOR = {"UtSu": 0.4643, "LtSl": 0.4676}


def run(tag, ckpt, loop, prefix):
    os.makedirs(OUT, exist_ok=True)
    summary = os.path.join(OUT, f"n7c_{tag}.json")
    preds = os.path.join(OUT, f"n7c_{tag}.preds.jsonl")
    cmd = [sys.executable, "eval/humaneval_bpb.py", "--ckpt", ckpt,
           "--out", summary, "--preds", preds]
    if loop:
        cmd += ["--loop", *loop]
    if prefix:
        cmd += ["--prefix", prefix]
    print(f"== {tag}: ckpt {ckpt} loop {loop or 'none'} mask "
          f"{'prefix ' + prefix if prefix else 'causal'}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print((r.stdout + r.stderr).strip()[-800:])
        raise SystemExit(f"REFUSING to continue: {tag} failed")
    with open(summary, encoding="utf-8") as fh:
        d = json.load(fh)
    # EVERY CELL MUST SCORE THE SAME BYTES AND THE SAME TASKS, or the numbers are not comparable and
    # a difference could be a difference in what was scored. Asserted, not assumed.
    assert d["total_solution_bytes"] == 29662, (tag, d["total_solution_bytes"])
    assert d["n_tasks"] == 164, (tag, d["n_tasks"])
    # AND THE SUMMARY MUST AGREE WITH WHAT WAS ASKED FOR. A cell whose stamp says causal while the
    # command said prefix is a cell scored in the wrong mask, and the bpb would look plausible.
    assert d.get("prefix_arm") == prefix, (tag, d.get("prefix_arm"), prefix)
    assert d.get("loop_blocks") == ([int(x) for x in loop] if loop else None), (
        tag, d.get("loop_blocks"), loop)
    print(f"   bpb {d['gold_bpb_byte_weighted']:.4f}  ({d['n_tasks']} tasks, "
          f"{d['total_solution_bytes']} bytes, prefix_arm {d.get('prefix_arm')!r}, "
          f"loop {d.get('loop_blocks')})", flush=True)
    return d, preds


def per_item(pa, pb):
    """(second worse, tied, n) on the id intersection, error rows excluded.

    Definition verified against runs/n7_2x2 by reproducing its published 103/164 before being used
    here, so this is the same count and not a similar one.
    """
    def load(p):
        with open(p, encoding="utf-8") as fh:
            return {r["task_id"]: r["bpb"] for r in map(json.loads, fh) if r.get("error") is None}
    x, y = load(pa), load(pb)
    ids = sorted(set(x) & set(y))
    return (sum(1 for i in ids if y[i] > x[i]), sum(1 for i in ids if y[i] == x[i]), len(ids))


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ARMS:
        raise SystemExit(f"usage: {os.path.basename(__file__)} {{{'|'.join(ARMS)}}}")
    arm = sys.argv[1]
    spec = ARMS[arm]
    for f in (spec["ckpt"], spec["twin"]):
        if not os.path.exists(os.path.join(ROOT, f)):
            raise SystemExit(f"REFUSING: {f} does not exist. The arm has not finished, or the twin "
                             f"is not in this tree.")

    matched, p_matched = run(f"{arm}_matched", spec["ckpt"], spec["loop"], arm)
    mismatch, p_mismatch = run(f"{arm}_causal", spec["ckpt"], spec["loop"], None)
    base, p_base = run(f"{arm}_twin_causal", spec["twin"], spec["loop"], None)

    b = base["gold_bpb_byte_weighted"]
    m = matched["gold_bpb_byte_weighted"]
    x = mismatch["gold_bpb_byte_weighted"]
    print(f"\n== arm {arm}: prefix vs its causal twin (lower is better)")
    print(f"  twin, causal          {b:.4f}   <- baseline, an existing 500-step run")
    print(f"  arm,  prefix {arm:3s}     {m:.4f}   delta {m - b:+.4f}   <- THE EXPERIMENT")
    print(f"  arm,  causal          {x:.4f}   delta {x - b:+.4f}   <- mismatch control")
    print("  If the mismatch cell is far larger than the matched delta, the number is the "
          "mask mismatch and not the intervention -- Stage A's failure mode.")
    key = "UtSu" if arm == "p3" else "LtSl"
    print(f"  the causal 2x2's {key} cell read {PRIOR[key]:.4f} on the same 164 tasks "
          f"(runs/n7c_2x2, same run family)")

    # PAIRED SE from eval/n3_report.py rather than reimplemented: it pairs on the id intersection
    # and byte-weights to match the reported figure, and a second copy of that arithmetic is a
    # second thing to get wrong. Seed sigma stays UNMEASURED; this says the items agree on a
    # direction, not that another seed would land here.
    sys.path.insert(0, ROOT)
    from eval.n3_report import paired_se  # noqa: PLC0415

    print("\n== paired, against the twin")
    for label, pb in (("matched (prefix)", p_matched), ("mismatch (causal)", p_mismatch)):
        r = paired_se(p_base, pb)
        se, zz = r.get("paired_se"), r.get("z")
        # SIGN: paired_se computes A - B and A is the BASELINE here, so a POSITIVE delta means the
        # cell is BETTER than the twin. Stated in words because the cell table above prints the
        # opposite orientation and two unlabelled signs read as a contradiction.
        better = "cell BETTER than twin" if r["delta_bpb"] > 0 else "cell WORSE than twin"
        print(f"  {label}: delta {r['delta_bpb']:+.4f} (twin minus cell, so {better})  "
              f"SE {'n/a' if se is None else f'{se:.4f}'}  "
              f"z {'n/a' if zz is None else f'{zz:+.1f}'}  n {r['n_items']}")
        if se:
            gate = abs(r["delta_bpb"]) > 2 * se
            print(f"    pre-registered rule: |delta| {abs(r['delta_bpb']):.4f} vs 2*SE "
                  f"{2 * se:.4f} -> {'PASSES the 2-SE bar' if gate else 'INSIDE 2 SE, a null'}")
        if r.get("seed_sigma") is None:
            print("    seed sigma UNMEASURED: this SE says the items agree on a direction, not "
                  "that another seed would land here.")

    print("\n== per-item count (separate measurement, not from paired_se)")
    for label, pb in (("matched (prefix)", p_matched), ("mismatch (causal)", p_mismatch)):
        w, t, n = per_item(p_base, pb)
        print(f"  {label}: {w}/{n} items worse than the twin" + (f", {t} tied" if t else ""))
    print("  The rule needs the item count to AGREE with the paired delta's direction: a mean that "
          "improves while most items get worse is a few large items, not the intervention.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Score the N7 500-step 2x2 and read the diagonal against the 250-step run.

FOUR CELLS, and the reason all four are scored rather than two: the diagonal (each arm in the
topology it trained in) is the experiment; the off-diagonals are the control that says whether a
number is the intervention or a topology mismatch. At 250 steps the diagonal was +0.0023 BPB
while the mismatch cells were +0.0193 and +0.0264 in OPPOSITE directions -- 8-11x larger -- which
is how Stage A's +0.0273 turned out to be measuring the mismatch and not the loop.

THE 250-STEP NUMBERS ARE A DIFFERENT RUN, NOT AN EARLIER SNAPSHOT. Those arms could not be
extended: their checkpoints carry no optimizer state (sft_math.py's mid-run save held the
snapshot and did not pass it, fixed at 148e6027), so resuming would have restarted Adam moments
and the LR schedule. The two step counts are therefore two independent runs and the comparison
below is labelled as such -- "still closing" is a statement about two runs, not about one curve.

WHAT THIS SCRIPT DOES NOT DO: decide. It prints the four cells, the diagonal delta with its
paired SE, both mismatch cells, the per-item win count as a SEPARATE labelled measurement, and
the 250-step values beside them. The reading rule was pre-registered in the exp row before either
arm launched.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs", "n7c_2x2")
CELLS = [
    ("UtSu", "ckpt_n7c_unlooped.pt", None),
    ("UtSl", "ckpt_n7c_unlooped.pt", ("4", "7")),
    ("LtSu", "ckpt_n7c_looped.pt", None),
    ("LtSl", "ckpt_n7c_looped.pt", ("4", "7")),
]
# The 250-step run, from runs/n7_2x2/*.json in this repo. Quoted here so the comparison needs no
# second read, and named as a separate run rather than an earlier point of this one.
PRIOR = {"UtSu": 0.4635, "UtSl": 0.4899, "LtSu": 0.4828, "LtSl": 0.4658}


def run(cell, ckpt, loop):
    os.makedirs(OUT, exist_ok=True)
    summary = os.path.join(OUT, f"n7c_2x2_{cell}.json")
    preds = os.path.join(OUT, f"n7c_2x2_{cell}.preds.jsonl")
    cmd = [sys.executable, "eval/humaneval_bpb.py", "--ckpt", ckpt,
           "--out", summary, "--preds", preds]
    if loop:
        cmd += ["--loop", *loop]
    # "looped" is a SUBSTRING of "unlooped", so `'looped' in ckpt` is True for both checkpoints and
    # every cell printed trained=looped. Display only -- the cell key and its checkpoint come from
    # CELLS and were always right, so no number was affected -- but a label that says the opposite
    # of what ran is how a correct number gets read as the wrong cell. Test the actual name.
    trained = "unlooped" if "unlooped" in ckpt else "looped"
    print(f"== {cell}: trained={trained} "
          f"scored={'looped_4_7' if loop else 'unlooped'}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print((r.stdout + r.stderr).strip()[-600:])
        raise SystemExit(f"REFUSING to continue: {cell} failed")
    return summary


def main():
    got = {}
    for cell, ckpt, loop in CELLS:
        with open(run(cell, ckpt, loop), encoding="utf-8") as fh:
            d = json.load(fh)
        got[cell] = d
        # EVERY CELL MUST SCORE THE SAME BYTES, or the four numbers are not comparable and a
        # difference could be a difference in what was scored. Checked rather than assumed.
        assert d["total_solution_bytes"] == 29662, (cell, d["total_solution_bytes"])
        assert d["n_tasks"] == 164, (cell, d["n_tasks"])
        print(f"   bpb {d['gold_bpb_byte_weighted']:.4f}  ({d['n_tasks']} tasks, "
              f"{d['total_solution_bytes']} bytes)", flush=True)

    print("\n== 500 steps, against the 250-step run (two runs, not one curve)")
    print(f"{'cell':6s} {'500':>8s} {'250':>8s} {'delta':>9s}")
    for c in ("UtSu", "LtSl", "UtSl", "LtSu"):
        v = got[c]["gold_bpb_byte_weighted"]
        print(f"{c:6s} {v:8.4f} {PRIOR[c]:8.4f} {v - PRIOR[c]:+9.4f}")

    diag = got["LtSl"]["gold_bpb_byte_weighted"] - got["UtSu"]["gold_bpb_byte_weighted"]
    m_b = got["LtSu"]["gold_bpb_byte_weighted"] - got["UtSu"]["gold_bpb_byte_weighted"]
    m_a = got["UtSl"]["gold_bpb_byte_weighted"] - got["UtSu"]["gold_bpb_byte_weighted"]
    print(f"\nDIAGONAL (the experiment)  looped-trained/looped-scored minus "
          f"unlooped/unlooped: {diag:+.4f}")
    print(f"  250-step diagonal was {PRIOR['LtSl'] - PRIOR['UtSu']:+.4f}")
    print(f"MISMATCH cells (the control): {m_a:+.4f} and {m_b:+.4f}")
    print("  If these stay far larger than the diagonal, the mismatch is again the dominant "
          "effect and any single-topology reading of this experiment is measuring it.")

    # PAIRED SE, from eval/n3_report.py rather than reimplemented here: it is paired on the
    # intersection of item ids and byte-weighted to match the reported figure, and a second copy
    # of that arithmetic is a second thing to get wrong. Seed sigma stays UNMEASURED and this is
    # not a stand-in for it -- this says whether the items agree on the direction, not whether
    # another seed would move the number.
    sys.path.insert(0, ROOT)
    from eval.n3_report import paired_se  # noqa: PLC0415

    for label, a, b in (("diagonal", "UtSu", "LtSl"),
                        ("mismatch-A (unlooped weights, looped scoring)", "UtSu", "UtSl"),
                        ("mismatch-B (looped weights, unlooped scoring)", "UtSu", "LtSu")):
        pa = os.path.join(OUT, f"n7c_2x2_{a}.preds.jsonl")
        pb = os.path.join(OUT, f"n7c_2x2_{b}.preds.jsonl")
        r = paired_se(pa, pb)
        # r's OWN z and se, not recomputed here: it returns them, and a second division is a
        # second chance to divide by the wrong SE. The per-item win count is NOT taken from this
        # return -- it has none -- and is printed in its own block below, labelled as a separate
        # measurement so the two are never read as one.
        se = r.get("paired_se")
        zz = r.get("z")
        se_s = "unavailable" if se is None else f"{se:.4f}"
        z_s = "unavailable" if zz is None else f"{zz:+.1f}"
        # SIGN: paired_se computes A - B (n3_report.py's contrib line), and A is the BASELINE cell
        # here, so a NEGATIVE delta means the second cell is WORSE -- the same finding the cell
        # table above prints as a positive number. Printing both orientations unlabelled read as a
        # contradiction, so the direction is stated in words rather than left to the reader.
        worse = "second cell worse" if r["delta_bpb"] < 0 else "second cell better"
        print(f"\n{label}: delta {r['delta_bpb']:+.4f} (baseline minus cell, so {worse})  "
              f"SE {se_s}  z {z_s}  n {r['n_items']}  bytes {r.get('total_bytes')}")
        if r.get("note"):
            print(f"  note: {r['note']}")
        if r.get("seed_sigma") is None:
            print("  seed sigma UNMEASURED: this SE says the items agree on a direction, "
                  "not that another seed would land here.")

    # PER-ITEM WIN COUNT, as its own measurement and deliberately OUTSIDE the paired_se block.
    # The user reads this number and the 250-step run reported it as 103/164, so it is printed the
    # same way -- but paired_se does not return it, and putting it inside that block would make two
    # different measurements read as one. Definition VERIFIED against the 250-step preds before
    # being written here: strictly worse on the id INTERSECTION, rows with a non-null error
    # excluded, ties reported separately rather than folded into either side. That recipe
    # reproduces 103/164 exactly on runs/n7_2x2, which is how I know it is the same count and not
    # a similar one.
    print("\n== per-item win count (separate measurement, not from paired_se)")

    def per_item(pa, pb):
        def load(p):
            with open(p, encoding="utf-8") as fh:
                return {r["task_id"]: r["bpb"] for r in map(json.loads, fh)
                        if r.get("error") is None}
        x, y = load(pa), load(pb)
        ids = sorted(set(x) & set(y))
        return (sum(1 for i in ids if y[i] > x[i]),
                sum(1 for i in ids if y[i] == x[i]), len(ids))

    for label, a, b in (("diagonal", "UtSu", "LtSl"),
                        ("mismatch-A (unlooped weights, looped scoring)", "UtSu", "UtSl"),
                        ("mismatch-B (looped weights, unlooped scoring)", "UtSu", "LtSu")):
        w, t, n = per_item(os.path.join(OUT, f"n7c_2x2_{a}.preds.jsonl"),
                           os.path.join(OUT, f"n7c_2x2_{b}.preds.jsonl"))
        print(f"  {label}: {w}/{n} items worse" + (f", {t} tied" if t else ""))
    print("  250-step diagonal for comparison: 103/164 worse (a separate run, "
          "reproduced from runs/n7_2x2 with this same recipe)")


if __name__ == "__main__":
    sys.exit(main())

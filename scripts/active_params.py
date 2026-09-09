#!/usr/bin/env python3
"""Active-params gate for a v2 A/B.

prereg v2_loop_moe_csa_0908 falsification condition (2): if CSA/HCA's param count
differs from KDA/MLA's, equal-active-params is violated and the control confounds
architecture with width. Nothing executable checked it -- train.py only formatted
n_params into a runlog line (model.py docstring: "Three places NAME a params gate
and none runs one"). This is the one that runs.

Each arm is its launch line (a torchrun prefix is allowed; everything after
`train.py` is used). The line builds the model via `train.py --build_only`; total
and active params are compared and the script exits 1 when either differs beyond
--tolerance.

Expected to be RED until a human rules on the trim knobs: the control has
attention in 3 of 12 blocks (attn_every=4), the treatment in all 12, and the spec
names no trim. A number filled in to make this green would be the failure this
repo has already named.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(line):
    args = shlex.split(line)
    if "train.py" in args:
        args = args[args.index("train.py") + 1:]
    if any(a == "--build_only" or a.startswith("--build_only=") for a in args):
        sys.exit("refusing: --build_only is reserved for this gate")
    p = subprocess.run(
        [sys.executable, os.path.join(ROOT, "train.py"), *args, "--build_only"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if p.returncode != 0:
        sys.exit(f"arm build failed:\n{p.stderr[-2000:]}")
    for ln in reversed(p.stdout.strip().splitlines()):
        try:
            return json.loads(ln)
        except json.JSONDecodeError:
            continue
    sys.exit(f"no JSON in train.py output:\n{p.stdout[-2000:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", required=True, help="control arm launch line")
    ap.add_argument("--treatment", required=True, help="treatment arm launch line")
    ap.add_argument("--tolerance", type=float, default=0.01,
                    help="max allowed relative diff per param class (default 0.01)")
    a = ap.parse_args()
    c, t = build(a.control), build(a.treatment)
    cfg_keys = ("d", "layers", "heads", "ffn_hidden", "moe_experts", "attn_every")
    rows, ok = [], True
    for key in ("total", "active"):
        cv, tv = c[key], t[key]
        rel = abs(tv - cv) / max(cv, 1)
        refused = rel > a.tolerance
        ok = ok and not refused
        rows.append({"param": key, "control": cv, "treatment": tv,
                     "rel_diff": round(rel, 6), "refused": refused})
    print(json.dumps({"control_cfg": {k: c[k] for k in cfg_keys},
                      "treatment_cfg": {k: t[k] for k in cfg_keys},
                      "comparison": rows}, indent=1))
    sys.exit(1 if not ok else 0)


if __name__ == "__main__":
    main()

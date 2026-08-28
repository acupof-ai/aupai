#!/usr/bin/env python3
"""Split the L3/L4 program bank into an eval-only half and a train-only half.

Holding out instances of a program measures whether the model can re-run a
template it has seen. Holding out the program measures whether it learned the
rule. math_hard_eval_1k happens to give the second, by accident: its generator
predates this bank and shares 0.3% of templates with training data. Any batch
generated from the bank today would give the first, at 86.4% shared templates.

The split is by md5 of the program id, so it is stable: adding programs to the
bank never moves an existing one across the line, and a rebuilt eval keeps
testing the same held-out families.

    python mathbank/split_bank.py --eval-frac 0.32
    -> mathbank/programs_eval.txt, mathbank/programs_train.txt

NOT APPLIED TODAY, and the measurement says why. Resolution is set by the number
of held-out program families, not by rows: with K families and intra-program ICC
0.296, an accuracy over them is worth K/ICC independent observations however many
instances each one generates. Reserving 309 of the 943 L3/L4 programs caps at
n_eff 1044, a 95% half-width of 1.03% at a 3% pass rate, against the 1.06-1.20%
math_hard_eval_1k already delivers. It would cost 309 training programs to buy
0.03 points. 68 of those 309 also reject every draw, so only 241 contribute.

Run this when the bank grows. Reaching a half-width of 0.53% needs about 1,178
eval-only programs; the whole bank is 943.
"""

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_math_short import load_programs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def bucket(pid, n=10_000):
    return int(hashlib.md5(pid.encode()).hexdigest()[:8], 16) % n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-frac", type=float, default=0.32)
    ap.add_argument("--levels", default="L3,L4")
    a = ap.parse_args()

    bank = load_programs()
    levels = a.levels.split(",")
    ids = sorted(name for lev in levels for name, _ in bank.get(lev, []))
    cut = a.eval_frac * 10_000
    ev = [i for i in ids if bucket(i) < cut]
    tr = [i for i in ids if bucket(i) >= cut]

    for path, group in (("programs_eval.txt", ev), ("programs_train.txt", tr)):
        with open(os.path.join(HERE, path), "w", encoding="utf-8") as f:
            f.write("\n".join(group) + "\n")
    print(f"{len(ids)} programs in {levels}: {len(ev)} eval-only, {len(tr)} train-only")
    print(f"  train ceiling {len(tr) * 150} rows at MAX_INST=150")


if __name__ == "__main__":
    main()

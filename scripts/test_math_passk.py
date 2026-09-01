#!/usr/bin/env python3
"""math_zh --k must compute pass@k the way code_zh does (de, 2026-09-01).

math-500 had no --k at all, so the pre-registered pass@8 was not runnable on it. The
semantics have to match code_zh exactly or the two halves of the sampled arm are not
comparable: at k>1 pass@1 is the GREEDY answer and --temperature applies only to the k
draws, so pass@k - pass@1 carries no sampling noise on the pass@1 side.

Known answers, no GPU and no model. The arithmetic is what is under test:

    greedy wrong, 1 of 8 draws right   -> pass@1 0%,   pass@8 100%, gap +100%
    greedy right, 0 of 8 draws right   -> pass@1 100%, pass@8 0%,   gap -100%
    all wrong                          -> 0, 0, 0

The middle case is the one worth keeping. pass@k is any-of-the-SAMPLED, not
any-of-everything, so it can sit BELOW pass@1 -- and a negative gap is meaningful
(the greedy answer was right and sampling lost it), not a bug to clamp away.

Also asserts the refusal: --k 8 at temperature 0 draws eight identical greedy answers,
so pass@k would equal pass@1 by construction. That must raise, not report.

    python3 scripts/test_math_passk.py
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def passk(greedy_ok, sample_oks):
    """The reference arithmetic, transcribed from code_zh.py's loop."""
    n_pass1 = int(greedy_ok)
    n_samp_ok = sum(sample_oks)
    n_passk = int(any(sample_oks))
    return n_pass1, n_samp_ok, n_passk


def main():
    bad = []

    # 1. the arithmetic, three known answers
    for label, g, ss, want in (
        ("greedy wrong, 1/8 sampled right", False, [False] * 7 + [True], (0, 1, 1)),
        ("greedy right, 0/8 sampled right", True, [False] * 8, (1, 0, 0)),
        ("all wrong", False, [False] * 8, (0, 0, 0)),
        ("all right", True, [True] * 8, (1, 8, 1)),
    ):
        got = passk(g, ss)
        if got != want:
            bad.append(f"{label}: {got}, expected {want}")

    # 2. the source agrees with the reference: math_zh must accumulate the same three
    #    quantities the same way. Reading the file rather than importing it keeps this
    #    CPU-only -- math_zh imports torch and a checkpoint loader at module scope.
    src = open(os.path.join(ROOT, "eval", "math_zh.py"), encoding="utf-8").read()
    for frag, why in (
        ("n_samp_ok += sum(oks)", "sampled mean must sum every draw"),
        ("n_passk += int(any(oks))", "pass@k must be any-of-the-sampled"),
        ("0.0 if k > 1 else temp", "at k>1 the pass@1 arm must be greedy, not sampled"),
    ):
        if frag not in src:
            bad.append(f"math_zh.py lacks `{frag}`: {why}")

    # 3. `k` must not be rebound inside the loop. It was: a difficulty bucket capped at 3
    #    shadowed the sample count, so a --k 8 run divided by 3 and printed "pass@3".
    #    Silent and plausible, which is why it gets an assertion rather than a comment.
    if "                k = min(check_steps(" in src:
        bad.append("`k` is rebound as a difficulty bucket inside the scoring loop; the "
                   "pass@k line divides by it and would report pass@3 for a --k 8 run")

    # 4. the refusal fires. --help is enough to prove the flag parses; the assert needs
    #    an actual run, so use --selfcheck-free arg validation via a nonexistent ckpt:
    #    argparse and the assert both run before any file is opened.
    r = subprocess.run([sys.executable, os.path.join(ROOT, "eval", "math_zh.py"),
                        "--ckpt", "/nonexistent.pt", "--k", "8"],
                       capture_output=True, text=True, cwd=ROOT, timeout=120)
    out = r.stdout + r.stderr
    if "pass@k would equal pass@1 by construction" not in out:
        bad.append("--k 8 at temperature 0 did not refuse: eight identical greedy draws "
                   f"make pass@k == pass@1 by construction (got: {out.strip()[-200:]})")

    if bad:
        print("FAIL: math_zh pass@k")
        for b in bad:
            print(f"  {b}")
        return 1
    print("OK: pass@k arithmetic matches code_zh, k is not shadowed, k>1 at t=0 refuses")
    return 0


if __name__ == "__main__":
    sys.exit(main())

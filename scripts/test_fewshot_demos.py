#!/usr/bin/env python3
"""--demos must change the prompt (de, 2026-09-01).

It did not. `demos` was built from a hardcoded N_DEMOS = 3 and then sliced by
`demos[:args.demos]`, so `--demos 8` sliced an 8-element window out of a 3-element list
and produced a 3-demo run. Silent, and worse than silent:

  - the console printed "L1 few-shot: 8 demos"
  - the predictions landed at preds_l1_d8.fewshot8_24k.jsonl

so both the log and the artifact asserted a configuration that never ran. A 3-vs-8
comparison came back byte-identical (md5 e2639d8b on both files, 112/112 generations
equal) which is the only reason it was caught -- and only because greedy decoding made
the identity exact rather than merely similar.

The general shape: a flag that is accepted, echoed, and filed under its own name while
changing nothing. Argument parsing proves a flag EXISTS; only the artifact proves it
ACTED. So this asserts the property directly -- more demos means a longer prompt with
more worked examples in it -- with no model and no GPU.

The first version of this file did NOT catch the defect. It rebuilt the demo pool
itself (`rows[:n]`) and asserted against build_prompt, so it passed on the defective
code: the pool sizing was the wrong line, and the test never ran it. Checked by
restoring the pre-fix file and running this -- exit 0, which is the whole reason
`split_rows` now exists as a function. A check encodes an assumption about where the
interesting case lives, and that assumption is the one thing the check never tests.
So: it calls eval.l1_fewshot.split_rows, the code the runner calls.

    python3 scripts/test_fewshot_demos.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def main():
    import json

    from eval.l1_fewshot import build_prompt, split_rows

    test_path = os.path.join(ROOT, "data", "eval", "math_test_500.jsonl")
    if not os.path.exists(test_path):
        # gitignored with the corpus; the split is still testable on synthetic rows,
        # which is what this actually needs -- the defect was in how the pool was
        # SIZED, and a fake pool exercises that identically.
        rows = [{"instruction": f"problem {i}", "output": f"answer {i}"} for i in range(20)]
    else:
        rows = [json.loads(x) for x in open(test_path, encoding="utf-8")][:20]

    bad = []
    prompts = {}
    for n in (0, 1, 3, 8):
        # split_rows, not a local rows[:n]. The defect lived in the runner's pool
        # sizing; a test that re-derives the pool cannot see it.
        demos, evals, err = split_rows(rows, n)
        if err:
            bad.append(f"--demos {n}: split refused: {err}")
            continue
        if len(demos) != n:
            bad.append(f"--demos {n}: split_rows built {len(demos)} demos, not {n}")
            continue
        prompts[n] = build_prompt(demos, "TARGET QUESTION")

        # 4. the eval set must exclude the demos, at every count. Scoring a model on a
        #    problem whose answer is in its own prompt is the failure the exclusion
        #    exists for, and it is silent -- it inflates rather than crashes.
        overlap = {q for q, _ in demos} & {r["instruction"] for r in evals}
        if overlap:
            bad.append(f"--demos {n}: {len(overlap)} demo problem(s) also in the eval set")

    # --eval-from must refuse an overlap rather than quietly scoring shown problems.
    _, _, err = split_rows(rows, 8, eval_from=3)
    if not err:
        bad.append(
            "--eval-from 3 with 8 demos was accepted; it scores 5 problems whose answers are in the prompt"
        )
    # and must be honoured when it is valid: both counts then score the same population.
    a = split_rows(rows, 3, eval_from=8)[1]
    b = split_rows(rows, 8, eval_from=8)[1]
    if not (a and b and [r["instruction"] for r in a] == [r["instruction"] for r in b]):
        bad.append(
            "--eval-from 8 does not pin the eval set across demo counts, so a "
            "3-vs-8 comparison carries a population change too"
        )

    # 1. every count produces a DIFFERENT prompt. This is the assertion that fails on
    #    the shipped code: 3 and 8 produced the same string.
    for x, y in ((0, 1), (1, 3), (3, 8)):
        if x in prompts and y in prompts and prompts[x] == prompts[y]:
            bad.append(
                f"--demos {x} and --demos {y} build an IDENTICAL prompt; the flag "
                f"is accepted and echoed but changes nothing"
            )

    # 2. monotone length: more demos, longer prompt. Catches a cap that silently clamps.
    lens = {n: len(p) for n, p in prompts.items()}
    for x, y in ((0, 1), (1, 3), (3, 8)):
        if x in lens and y in lens and not lens[y] > lens[x]:
            bad.append(
                f"--demos {y} prompt ({lens[y]} chars) is not longer than "
                f"--demos {x} ({lens[x]}) -- demos are being dropped"
            )

    # 3. the count is actually IN the prompt, so a prompt that merely got longer for
    #    some other reason fails here. build_prompt emits one "题目：" per demo plus one
    #    for the target, hence n+1. (My first version asserted n+1 occurrences of
    #    "解答：" and failed on synthetic rows whose answers contained it: the test was
    #    wrong, not the code, which is its own small lesson about asserting a count you
    #    have not looked at.)
    for n, p in prompts.items():
        got = p.count("题目：")
        if got != n + 1:
            bad.append(
                f"--demos {n}: prompt holds {got} '题目：' markers, expected {n + 1} "
                f"({n} demos plus the target)"
            )

    if bad:
        print("FAIL: --demos does not change what runs")
        for b in bad:
            print(f"  {b}")
        return 1
    print(
        f"OK: {len(prompts)} demo counts build distinct, monotonically longer prompts "
        f"carrying the right number of examples, none leak into the eval set, and "
        f"--eval-from pins the population"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

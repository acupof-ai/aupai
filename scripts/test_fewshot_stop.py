#!/usr/bin/env python3
"""The few-shot scorer must stop at the model's own turn (de, 2026-09-01).

Every case below is a REAL generation from
data/eval/preds_l1_d3.fewshot_24k.jsonl, truncated, not a synthetic stand-in.
Reinstating the old `score` (drop the model_turn call) turns cases 1-3 red.

    python3 scripts/test_fewshot_stop.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("FLA_NO_IMPORT", "1")
from eval.l1_fewshot import EX_OPEN, model_turn, score  # noqa: E402

ANSWERED_8_THEN_INVENTED_7 = (
    "1. **小明原有苹果数量：** 10 个\n2. **送给小李的苹果数量：** 2 个\n"
    "3. **剩下的苹果数量：** 10 个 - 2 个 = 8 个\n\n所以，小明还剩下 \\(\\boxed{8}\\) 个苹果。\n\n"
    "题目：小明有10个苹果，他给了小李3个苹果，现在他还剩下多少个苹果？\n"
    "解答：3. **剩下的苹果数量：** 10 个 - 3 个 = 7 个\n\n所以，小明还剩下 \\(\\boxed{7}\\) 个苹果。"
)

TWO_BOXES_ONE_TURN = (
    "小明最多可以买 \\(\\boxed{5}\\) 辆玩具小汽车，剩余 \\(\\boxed{10}\\) 元。"
)

CASES = [
    ("answers 8, then invents a problem and answers 7", ANSWERED_8_THEN_INVENTED_7, "\\boxed{8}", 1.0),
    ("the invented answer must not be credited either", ANSWERED_8_THEN_INVENTED_7, "\\boxed{7}", 0.0),
    ("no fabrication: unchanged", "答案 \\(\\boxed{4}\\) 个。", "\\boxed{4}", 1.0),
    ("two boxes in ONE turn: the LAST is the answer, not the first",
     TWO_BOXES_ONE_TURN, "\\boxed{10}", 1.0),
    ("...and the intermediate result is not the answer",
     TWO_BOXES_ONE_TURN, "\\boxed{5}", 0.0),
]


def main():
    assert model_turn("a" + EX_OPEN + "b") == "a"
    assert model_turn("no example opener here") == "no example opener here"
    bad = []
    for name, gen, gold, want in CASES:
        got = score(gen, gold)
        print(f"  {'ok  ' if got == want else 'FAIL'} {name}: score={got} want={want}")
        if got != want:
            bad.append(name)
    if bad:
        print(f"\n{len(bad)} case(s) failed: {bad}")
        return 1
    print(f"\n{len(CASES)} cases pass: the scorer reads the model's turn, and within "
          "that turn still grades the last box")
    return 0


if __name__ == "__main__":
    sys.exit(main())

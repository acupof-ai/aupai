#!/usr/bin/env python3
"""Why the control's answer-present rate is 0.6%, measured rather than called "weaker".

# restartable: reads four committed preds files, writes one JSON. Seconds, no GPU, no model.

    python3 eval/l1_2x2_diagnose.py

WHAT THE 2x2 FOUND. answer-present 30.6-37.0% for our arm against 0.6-1.0% for the control, while
accuracy is floored in all four cells (2*delta = 12.6%). A 30-60x ratio invites "our model is 30x
better at math", which the same generations refute: nobody's accuracy is distinguishable from zero.

SO WHAT IS THE CONTROL DOING. I claimed, from a 100-row sample of ONE cell with no comparison arm,
that the control "echoes the prompt and loops" while ours does not. Running it over all four cells
falsified that in both directions: our arm copies a demo span in 90.7-93.6% of generations against
the control's 91.8-97.4%, and LOOPS MORE (98.8% against 77.7-86.9%). By share of the generation
rather than presence, ours is 39.7% demo text at the median and the control 21.3%. Copying and
looping do not separate the arms; both do both.

WHAT DOES SEPARATE THEM IS LENGTH: 794-851 characters at the median against 84-86. The control stops
early, and the 12-character CJK repetition stop is what stops it -- so its generation is short enough
that an answer marker rarely appears, which is the 0.6-1.0% answer-present. That is a statement about
what the decoder did, not yet about why, and this script does not license more than that.

WHAT answer-present MEASURES, stated because the phrase does not say it: FORMAT ACQUISITION. "There
is an extractable answer in the generation" is a different claim from "the answer is right". Our
corpus contains Chinese math solutions with boxed answers and the Pile does not, so the gap is a
distribution fact that is favourable to us and is a PRECONDITION for capability -- not capability.
Quoting the 30-60x as a capability result would let a compound quantity speak for one of its factors
(6e's ruling, and the same shape as this week's other two).

A NOTE ON READING THESE FILES. The control's generations contain fluent Chinese, which looked
impossible for a Pile-trained GPTNeoX and briefly read as "--hf loaded the wrong model, all four
cells are void". The tokenizer represents CJK through byte fallback: roundtrip is exact, and the
mojibake was a terminal encoding artefact in my own print. Anything reading these rows must set
stdout to UTF-8 -- the display layer and the data layer are not the same layer.
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPAN = 12          # characters; long enough that a shared span is not a coincidence of common words
REPEAT_MIN = 3     # a span occurring this often inside one generation is a loop
CELLS = (("ours", "zh", "preds_l1_d3_ckpt_p200m_4b_0902.pt.zh.jsonl"),
         ("ours", "en", "preds_l1_d3_ckpt_p200m_4b_0902.pt.en.jsonl"),
         ("control", "zh", "preds_l1_d3_e1_untrained.hf.zh.hf.jsonl"),
         ("control", "en", "preds_l1_d3_e1_untrained.hf.en.hf.jsonl"))


def spans(text, n=SPAN, stride=1):
    return (text[i:i + n] for i in range(0, max(0, len(text) - n + 1), stride))


def analyse(rows, demo_text):
    """Copy rate, copy SHARE, internal-repeat rate and length quantiles over every row.

    Both a rate and a share, because they answer different questions and the rate alone misled me:
    "does any 12-char span come from the demos" is near-saturated for both arms (90%+), while "what
    fraction of the generation is demo text" separates them (39.7% ours against 21.3% control) --
    and in the opposite direction to the claim the rate seemed to support.
    """
    copied = looped = 0
    lens = []
    shares = []
    for r in rows:
        g = r["gen"]
        lens.append(len(g))
        n_spans = max(1, len(g) - SPAN + 1)
        hits = sum(1 for s in spans(g) if s in demo_text)
        if hits:
            copied += 1
        shares.append(hits / n_spans)
        # stride 7 so a long generation does not cost O(len^2) full scans; a loop of period < 7
        # still lands on a start position, and the rate is a description not a threshold.
        if any(g.count(s) >= REPEAT_MIN for s in spans(g, stride=7)):
            looped += 1
    lens.sort()
    shares.sort()
    n = len(rows)
    return {"n": n,
            "copy_rate": copied / n, "copy_share_median": shares[n // 2],
            "loop_rate": looped / n,
            "median_chars": lens[n // 2], "p90_chars": lens[int(0.9 * n)],
            "max_chars": lens[-1], "empty": sum(1 for x in lens if x == 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_dir", default="runs/l1_2x2")
    ap.add_argument("--test", default="data/eval/math_test_500.jsonl")
    ap.add_argument("--demos", type=int, default=3)
    ap.add_argument("--out", default="runs/l1_2x2_diagnose.json")
    a = ap.parse_args()

    test_path = os.path.join(ROOT, a.test)
    if not os.path.exists(test_path):
        sys.exit(f"REFUSING: {a.test} absent -- the demo text is what a copied span is copied FROM, "
                 f"so without it the copy rate cannot be computed and a 0.0 would be indistinguishable "
                 f"from 'no copying'")
    items = [json.loads(l) for l in open(test_path, encoding="utf-8") if l.strip()]
    # THE DEMOS ARE ROWS 0..demos-1, the same slice l1_fewshot's split_rows uses. Taking a different
    # slice would compare generations against text that was never in their prompt.
    demo_text = " ".join(d["output"] for d in items[:a.demos])

    out = {"span_chars": SPAN, "repeat_min": REPEAT_MIN, "demos": a.demos,
           "what_answer_present_measures":
               "FORMAT ACQUISITION -- an extractable answer exists in the generation. Not accuracy: "
               "the same generations are floored on correctness (2*delta = 12.6%). Our corpus holds "
               "Chinese math solutions with boxed answers and the Pile does not, so the arm gap is a "
               "distribution fact and a precondition for capability, not capability.",
           "cells": {}}
    for arm, lang, fname in CELLS:
        p = os.path.join(ROOT, a.preds_dir, fname)
        if not os.path.exists(p):
            sys.exit(f"REFUSING: {p} absent")
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        out["cells"][f"{arm}-{lang}"] = analyse(rows, demo_text) | {"preds": fname}

    print(f"{'cell':<12} {'n':>5} {'copy%':>7} {'copyshr':>8} {'loop':>7} {'median':>7} "
          f"{'p90':>6} {'max':>6}")
    for k, v in out["cells"].items():
        print(f"{k:<12} {v['n']:>5} {v['copy_rate']:>6.1%} {v['copy_share_median']:>7.1%} "
              f"{v['loop_rate']:>6.1%} {v['median_chars']:>7,} {v['p90_chars']:>6,} "
              f"{v['max_chars']:>6,}")
    print("copy% = any 12-char span from the demos (near-saturated, does not separate the arms); "
          "copyshr = median SHARE of spans that are demo text; loop = a span repeated 3+ times.")
    with open(os.path.join(ROOT, a.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

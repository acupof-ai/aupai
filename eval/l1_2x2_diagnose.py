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

# IMPORTED, NOT RESTATED, AND THE IMPORT IS THE PREDICATE, NOT ONE BRANCH OF IT. answer-present is
# `\boxed` OR ANS_RE. My first version here imported ANS_RE alone, believing a shared regex made this
# script agree with the scorer -- it agreed on a SUBSTRING of the predicate and reported 0/497 markers
# for cells whose published rate is 37.0%. The boxed branch is the one our arm actually uses, so the
# half I dropped was the whole signal. l1_fewshot.answer_marker() now holds the disjunction and both
# call sites route through it.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "l1f_for_diag", os.path.join(ROOT, "eval", "l1_fewshot.py"))
_L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_L)
answer_marker = _L.answer_marker

SPAN = 12          # characters; long enough that a shared span is not a coincidence of common words
REPEAT_MIN = 3     # a span occurring this often inside one generation is a loop
CELLS = (("ours", "zh", "preds_l1_d3_ckpt_p200m_4b_0902.pt.zh.jsonl"),
         ("ours", "en", "preds_l1_d3_ckpt_p200m_4b_0902.pt.en.jsonl"),
         ("control", "zh", "preds_l1_d3_e1_untrained.hf.zh.hf.jsonl"),
         ("control", "en", "preds_l1_d3_e1_untrained.hf.en.hf.jsonl"))


def spans(text, n=SPAN, stride=1):
    return (text[i:i + n] for i in range(0, max(0, len(text) - n + 1), stride))


def analyse(rows, demo_text):
    """Copy rate, copy SHARE, internal-repeat rate, length quantiles and MARKER POSITION.

    Both a rate and a share, because they answer different questions and the rate alone misled me:
    "does any 12-char span come from the demos" is near-saturated for both arms (90%+), while "what
    fraction of the generation is demo text" separates them (39.7% ours against 21.3% control) --
    and in the opposite direction to the claim the rate seemed to support.

    MARKER POSITION is why the answer-present column is retracted, not merely qualified. I first
    kept answer-present on the table on the grounds that scoring happens AFTER model_turn truncates,
    so how far the generation ran could not matter. That reasoning is wrong, and this column is what
    shows it: our markers sit at character 287 (median), the control's whole generation is 86
    characters. Truncation after the fact cannot restore characters the stop rule prevented from
    being generated -- so the control never had the OPPORTUNITY to produce format, and 0.6% is not
    a measurement of what it can do.
    """
    copied = looped = 0
    lens = []
    shares = []
    marker_pos = []
    for r in rows:
        g = r["gen"]
        lens.append(len(g))
        n_spans = max(1, len(g) - SPAN + 1)
        hits = sum(1 for s in spans(g) if s in demo_text)
        if hits:
            copied += 1
        shares.append(hits / n_spans)
        # Position in the RAW generation, not in model_turn's output: the question is whether the
        # decoder ran far enough to reach a marker, which is a property of the untruncated text.
        pos = answer_marker(g)
        if pos is not None:
            marker_pos.append(pos)
        # stride 7 so a long generation does not cost O(len^2) full scans; a loop of period < 7
        # still lands on a start position, and the rate is a description not a threshold.
        if any(g.count(s) >= REPEAT_MIN for s in spans(g, stride=7)):
            looped += 1
    lens.sort()
    shares.sort()
    marker_pos.sort()
    n = len(rows)
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None
    return {"n": n,
            "copy_rate": copied / n, "copy_share_median": shares[n // 2],
            "loop_rate": looped / n,
            "median_chars": lens[n // 2], "p90_chars": lens[int(0.9 * n)],
            "max_chars": lens[-1], "empty": sum(1 for x in lens if x == 0),
            "marker_rows": len(marker_pos),
            "marker_pos_median": q(marker_pos, 0.5), "marker_pos_p90": q(marker_pos, 0.9),
            "marker_pos_max": marker_pos[-1] if marker_pos else None}



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
               "RETRACTED AS AN ARM COMPARISON. The two arms generated under DIFFERENT stop rules "
               "(our arm had no repetition stop, the control had one), and marker_pos shows why "
               "that is fatal rather than cosmetic: our markers sit at character 287 (median) while "
               "the control's entire generation is 86 characters. The control was cut off before "
               "the position where format appears, so it had no opportunity to produce any -- its "
               "0.6% measures the stop rule. Scoring happening after model_turn does NOT rescue "
               "this: truncating later cannot restore characters never generated. Fix, not yet run: "
               "--no_rep_stop, both arms, all four cells. Even then the column measures FORMAT "
               "ACQUISITION, not accuracy -- accuracy is floored in all four cells (2*delta=12.6%).",
           "cells": {}}

    for arm, lang, fname in CELLS:
        p = os.path.join(ROOT, a.preds_dir, fname)
        if not os.path.exists(p):
            sys.exit(f"REFUSING: {p} absent")
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        out["cells"][f"{arm}-{lang}"] = analyse(rows, demo_text) | {"preds": fname}

    print(f"{'cell':<12} {'n':>5} {'copy%':>7} {'copyshr':>8} {'loop':>7} {'median':>7} "
          f"{'p90':>6} {'max':>6} {'mark':>7} {'mkpos50':>8} {'mkpos90':>8}")
    for k, v in out["cells"].items():
        print(f"{k:<12} {v['n']:>5} {v['copy_rate']:>6.1%} {v['copy_share_median']:>7.1%} "
              f"{v['loop_rate']:>6.1%} {v['median_chars']:>7,} {v['p90_chars']:>6,} "
              f"{v['max_chars']:>6,} {v['marker_rows']:>4}/{v['n']:<3} "
              f"{str(v['marker_pos_median']):>8} {str(v['marker_pos_p90']):>8}")
    print("copy% = any 12-char span from the demos (near-saturated, does not separate the arms); "
          "copyshr = median SHARE of spans that are demo text; loop = a span repeated 3+ times.")
    print("mkpos = CHARACTER POSITION of the answer marker in the RAW generation. Compare it to the "
          "OTHER arm's median length: a marker at 287 cannot appear inside an 86-character "
          "generation, so the arm that was stopped first never had the chance to produce format, "
          "and its answer-present rate measures the stop rule rather than the model.")
    with open(os.path.join(ROOT, a.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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

WHAT DOES SEPARATE THEM UNDER THE STOP-ON CELLS IS LENGTH: 794-851 characters at the median against
84-86, because the 12-character CJK repetition stop was running on the control and not on us. That
asymmetry is the defect d165a905 voided, and the shared-decoder rerun (--no_rep_stop on both arms,
e1-31b) settled what it was hiding:

  - The control's 0.6% is NOT the stop rule. At 450 characters and the full 512-token budget it
    produced no additional markers, and the marked rows are the SAME rows (intersection 3/3, 5/5,
    symmetric difference 0). Behavioural difference, measured, not inferred.
  - LOOP RATE SATURATES under one decoder -- 98.8% ours against 100.0% control -- so it separates
    nothing while reading like a finding. The loop DEGREE does separate them, and against my
    published claim that the control loops less: the most-repeated span recurs 13x at the median for
    the control against 6x for us, in HALF the characters.

WHAT answer-present MEASURES, stated because the phrase does not say it: FORMAT ACQUISITION. "There
is an extractable answer in the generation" is a different claim from "the answer is right". Our
corpus contains Chinese math solutions with boxed answers and the Pile does not, so the gap is a
distribution fact that is favourable to us and is a PRECONDITION for capability -- not capability.
Quoting the 30-60x as a capability result would let a compound quantity speak for one of its factors
(6e's ruling, and the same shape as this week's other two). be.l1_fewshot_p324 bounds the mechanism
on ONE checkpoint: answer-present 25.4/53.5/63.6% at 0/1/3 demos, i.e. format exposure alone moves
this column 38pt with capability fixed -- the same order as the 36.4pt arm gap.


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
# A PROPORTION WITH BOTH ARMS INSIDE ONE OF THESE BANDS CANNOT RESOLVE THEM, and the script says so
# instead of printing two numbers. This exists because I violated the rule twice in three hours: the
# copy rate (90.7-97.4%) went into a published claim, and then the loop rate (98.8% vs 100.0%) did
# too, one line below the comment warning about the first. A rule broken twice that fast by the same
# person is not a memory problem -- it is a rule in the wrong FORM. Knowledge to be remembered
# becomes code that refuses (6e). Precedent in this repo: score_matrix writes SKIP plus a reason for
# base generative metrics rather than writing 0.
SATURATED_HIGH = 0.95
SATURATED_LOW = 0.05
# The stop-ON cells (rep_stop asymmetric between the arms -- the defect d165a905 voided). Kept as
# the default so the published numbers stay reproducible from this script, NOT because they are the
# current reading: the arm comparison on them is retracted. --preds names the four files for any
# other run, e.g. the shared-decoder rerun's .norepstop cells. Hardcoding one filename set is how a
# script silently keeps answering about last week's artifacts after the experiment moved.
CELLS = (("ours", "zh", "preds_l1_d3_ckpt_p200m_4b_0902.pt.zh.jsonl"),
         ("ours", "en", "preds_l1_d3_ckpt_p200m_4b_0902.pt.en.jsonl"),
         ("control", "zh", "preds_l1_d3_e1_untrained.hf.zh.hf.jsonl"),
         ("control", "en", "preds_l1_d3_e1_untrained.hf.en.hf.jsonl"))


def spans(text, n=SPAN, stride=1):
    return (text[i:i + n] for i in range(0, max(0, len(text) - n + 1), stride))


def saturated_band(values):
    """Name the saturated band every value falls in, or None if the metric can still resolve.

    A FUNCTION so a test can call THIS and not a copy of it. My first check for this re-implemented
    the two comparisons in the test file and passed -- which proves the fixture's arithmetic, not the
    script's (the "a copy is not a witness" defect already on file here twice).
    """
    vals = list(values)
    if not vals:
        return None
    if all(x >= SATURATED_HIGH for x in vals):
        return f"all >= {SATURATED_HIGH:.0%}"
    if all(x <= SATURATED_LOW for x in vals):
        return f"all <= {SATURATED_LOW:.0%}"
    return None


def analyse(rows, demo_text):
    """Copy rate, copy SHARE, internal-repeat rate, length quantiles and MARKER POSITION.

    Both a rate and a share, because they answer different questions and the rate alone misled me:
    "does any 12-char span come from the demos" is near-saturated for both arms (90%+), while "what
    fraction of the generation is demo text" separates them (39.7% ours against 21.3% control) --
    and in the opposite direction to the claim the rate seemed to support.

    MARKER POSITION is why the answer-present column was retracted on the stop-ON cells. I first kept
    it on the table on the grounds that scoring happens AFTER model_turn truncates, so how far the
    generation ran could not matter -- wrong, because truncating later cannot restore characters the
    stop rule prevented from being generated. This column showed our markers at character 287 while
    the control's whole generation was 86.

    BUT THE READING I BUILT ON TOP OF IT WAS ALSO WRONG, and the shared-decoder rerun falsified it:
    I concluded the control "never had the OPPORTUNITY to produce format". Given 450 characters and
    the full token budget it produced no additional markers. Our arm's marker DISTRIBUTION is not a
    statement about what the control REQUIRES -- two models do not share a length distribution. Read
    this column as "where markers sit when they exist", never as "how long a generation must be".
    """
    copied = looped = 0
    loop_degree = []
    loop_density = []
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
        # BOTH THE RATE AND THE DEGREE, because under a shared decoder the RATE SATURATES: 98.8%
        # against 100.0%, which separates nothing while reading like a finding. The degree -- how
        # many times the most-repeated span actually recurs -- separates the arms by 2.2x (median 6
        # against 13) and in the direction opposite to my published claim that the control loops
        # less. A proportion pinned near its ceiling is zero evidence, not weak evidence.
        best = max((g.count(s) for s in spans(g, stride=7)), default=0)
        loop_degree.append(best)
        # PER 1000 CHARACTERS TOO, because degree grows with length under the null: a longer text
        # gives a 12-char span more chances to recur. The control has HIGHER degree in HALF the
        # characters, so the raw comparison already lands on the conservative side and normalising
        # widens it (6e). Reported so nobody has to take my word for which direction is conservative.
        if len(g):
            loop_density.append(best / len(g) * 1000)
        if best >= REPEAT_MIN:
            looped += 1
    lens.sort()
    shares.sort()
    marker_pos.sort()
    loop_degree.sort()
    loop_density.sort()
    n = len(rows)
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None
    return {"n": n,
            "copy_rate": copied / n, "copy_share_median": shares[n // 2],
            "loop_rate": looped / n,
            "loop_degree_median": loop_degree[n // 2],
            "loop_degree_p90": loop_degree[int(0.9 * n)],
            "loop_degree_per_kchar_median": round(loop_density[len(loop_density) // 2], 2)
            if loop_density else None,
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
    ap.add_argument("--preds", nargs=4, metavar=("OURS_ZH", "OURS_EN", "CTRL_ZH", "CTRL_EN"),
                    help="the four cell files, in that fixed order, relative to --preds_dir. "
                         "Order is positional and NOT inferred from the filenames: guessing the arm "
                         "from a substring would mislabel the arms on any naming scheme that does "
                         "not contain the guessed token, and a swapped arm label reverses every "
                         "conclusion while every number stays plausible. Default: the stop-ON "
                         "cells, whose ARM comparison is retracted (the arms ran different "
                         "decoders); pass the .norepstop four for the shared-decoder rerun.")
    a = ap.parse_args()

    cells = CELLS if not a.preds else tuple(
        (arm, lang, fname) for (arm, lang, _), fname in zip(CELLS, a.preds))

    test_path = os.path.join(ROOT, a.test)
    if not os.path.exists(test_path):
        sys.exit(f"REFUSING: {a.test} absent -- the demo text is what a copied span is copied FROM, "
                 f"so without it the copy rate cannot be computed and a 0.0 would be indistinguishable "
                 f"from 'no copying'")
    items = [json.loads(l) for l in open(test_path, encoding="utf-8") if l.strip()]
    # THE DEMOS ARE ROWS 0..demos-1, the same slice l1_fewshot's split_rows uses. Taking a different
    # slice would compare generations against text that was never in their prompt.
    demo_text = " ".join(d["output"] for d in items[:a.demos])

    # THE PROVENANCE NOTE FOLLOWS THE FILES, not the script. The retraction below is a fact about the
    # stop-ON cells (different decoders per arm); carrying it onto a shared-decoder rerun would
    # brand clean numbers as retracted, and carrying the clean text onto the stop-on cells would
    # erase a live retraction. Which set is in play is decided by --preds, so the note is too.
    RETRACTED = (
        "RETRACTED AS AN ARM COMPARISON. The two arms generated under DIFFERENT stop rules "
        "(our arm had no repetition stop, the control had one), and marker_pos shows why "
        "that is fatal rather than cosmetic: our markers sit at character 287 (median) while "
        "the control's entire generation is 86 characters. The control was cut off before "
        "the position where format appears, so it had no opportunity to produce any -- its "
        "0.6% measures the stop rule. Scoring happening after model_turn does NOT rescue "
        "this: truncating later cannot restore characters never generated. Fix, not yet run: "
        "--no_rep_stop, both arms, all four cells. Even then the column measures FORMAT "
        "ACQUISITION, not accuracy -- accuracy is floored in all four cells (2*delta=12.6%).")
    SHARED_DECODER = (
        "Cell files given explicitly via --preds. If these are the .norepstop four, both arms ran "
        "the SAME decoder and the arm comparison is defined -- the retraction that applies to the "
        "stop-ON cells does not apply here. What the column measures is unchanged: FORMAT "
        "ACQUISITION, an extractable answer exists in the model's turn, position-independent "
        "(pinned 2026-09-03Z before this rerun produced a number). That is not accuracy, and it is "
        "not capability: our corpus holds Chinese math solutions with boxed answers and the Pile "
        "does not, so an arm gap is a distribution fact and a PRECONDITION for capability. Read "
        "accuracy from the run's own summary JSON, not from this file.")
    out = {"span_chars": SPAN, "repeat_min": REPEAT_MIN, "demos": a.demos,
           "cells_from": "--preds" if a.preds else "default stop-ON cells",
           "what_answer_present_measures": SHARED_DECODER if a.preds else RETRACTED,
           "cells": {}}

    for arm, lang, fname in cells:
        p = os.path.join(ROOT, a.preds_dir, fname)
        if not os.path.exists(p):
            sys.exit(f"REFUSING: {p} absent")
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        out["cells"][f"{arm}-{lang}"] = analyse(rows, demo_text) | {"preds": fname}

    print(f"{'cell':<12} {'n':>5} {'copy%':>7} {'copyshr':>8} {'loop':>7} {'deg50':>6} {'deg/kc':>7} {'median':>7} "
          f"{'p90':>6} {'max':>6} {'mark':>7} {'mkpos50':>8} {'mkpos90':>8}")
    for k, v in out["cells"].items():
        print(f"{k:<12} {v['n']:>5} {v['copy_rate']:>6.1%} {v['copy_share_median']:>7.1%} "
              f"{v['loop_rate']:>6.1%} {v['loop_degree_median']:>6} {v['loop_degree_per_kchar_median']:>7} {v['median_chars']:>7,} {v['p90_chars']:>6,} "
              f"{v['max_chars']:>6,} {v['marker_rows']:>4}/{v['n']:<3} "
              f"{str(v['marker_pos_median']):>8} {str(v['marker_pos_p90']):>8}")
    # RESOLUTION BEFORE VALUE. Asked cell by cell whether each proportion can tell the arms apart at
    # all, and printed as NO RESOLUTION when it cannot -- the scorer answers that, not the reader.
    resolution = {}
    for metric, label in (("copy_rate", "copy%"), ("loop_rate", "loop")):
        vals = {k: v[metric] for k, v in out["cells"].items()}
        band = saturated_band(vals.values())
        resolution[metric] = "NO RESOLUTION" if band else "usable"
        if band:
            print(f"NO RESOLUTION on {label}: every cell is inside the saturated band ({band}: "
                  + ", ".join(f"{k} {x:.1%}" for k, x in vals.items())
                  + "). A proportion pinned at its ceiling is ZERO evidence, not weak evidence -- "
                    "do not quote a difference between these numbers. Use the DEGREE column, or a "
                    "share, which has dynamic range where the rate does not.")
    out["resolution"] = resolution

    print("copy% = any 12-char span from the demos; copyshr = median SHARE of spans that are demo "
          "text; loop = a span repeated 3+ times; deg50 = median number of times the most-repeated "
          "span recurs. Prefer copyshr and deg50: the two RATES saturate.")
    print("mkpos = CHARACTER POSITION of the answer marker in the RAW generation. On the stop-ON "
          "cells it showed the control was cut off before the position where format appears -- an "
          "argument the shared-decoder rerun then FALSIFIED: given 450 characters and the full "
          "512-token budget the control produced no additional markers, and the marked rows are the "
          "SAME rows (intersection 3/3 and 5/5, symmetric difference 0). Read this column as "
          "'where markers sit when they exist', never as 'how long a generation must be to produce "
          "one' -- one model's marker distribution does not state another model's requirement.")
    out_path = os.path.join(ROOT, a.out)
    # THE OUTPUT PATH FOLLOWS THE INPUT SET. runs/l1_2x2_diagnose.json is the artifact the audit
    # cites for the stop-ON marker positions (267/497 at char 287); a --preds run writing that same
    # path would overwrite the evidence for a retraction with numbers from a different experiment,
    # and both files look equally valid afterwards. Only --out given explicitly overrides this.
    if a.preds and a.out == ap.get_default("out"):
        out_path = out_path[:-len(".json")] + ".preds.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {os.path.relpath(out_path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

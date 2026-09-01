#!/usr/bin/env python3
"""L2: executability, scored WITHOUT requiring a code fence (de, 2026-09-01).

fb's spec, and the measurement that changed one clause of it.

The spec said: today's code-500 zeros were format-compliance measurements, so extract
the longest parseable span anywhere rather than a fenced block. Measured before
building, on the step24000 sampled preds:

    ChatML prompt (eval/code_zh.py, what every code-500 number used)
      generations                 2586
      containing a fence            41  (1.6%)
      containing "def " at all       7  (0.3%)
      longest parseable span         1 of 200 sampled  (0.5%)

A better extractor cannot open that gate. There is no code in the output to extract:
the model repeats the question, or drifts into wiki boilerplate. Math's gate was the
answer TEMPLATE with reasoning behind it (0/1439 boxed, 70.3% carrying a digit);
code's gate is upstream of that -- the ChatML prompt does not put the model in a
code-writing state at all. Same zero, different cause, and an extractor fix alone
would have produced another uninterpretable zero.

The prompt is the gate, and that is measurable:

    plain continuation with demos (eval/code_fewshot.py, same checkpoint family)
      0-shot   192/497 carry "def "  (38.6%)
      1-shot   469/497                (94.4%)
      3-shot   467/497                (94.0%)
      longest parseable span, 3-shot: 470/497 (94.6%)

So demos are part of the metric DEFINITION, not a confound to be removed later --
fb's own instruction for the case where the metric cannot resolve without them. A
number from this scorer is always reported with its demo count.

## What is scored

Executability: the extracted span runs and its stdout matches the recorded oracle,
through datagen.sandbox_exec.run_sandboxed -- the same executor code_zh.py uses. The
extractor changes; the judge does not.

## Two controls, both at the statistic's own aggregation

1. SHUFFLED: each extracted span against ANOTHER problem's oracle. The rate at which
   a span satisfies an oracle it was not written for. Per-row for the per-row rate.
2. COPYIST: the DEMO problems' reference solutions scored against every eval oracle.
   This is what a model that learned nothing but "echo the demonstration" achieves,
   and the generations make it a live worry -- they visibly repeat the demo's
   `is_prime` body. The shuffled control cannot see this: a copied span is a real
   span, and shuffling the oracles does not ask where the span came from.

A rate at or below EITHER control is not a reading. The pass@k path carries its own
control at the per-question aggregation, because eight draws hit an oracle by
coincidence far more often than one does (measured on the math side: 17.9% real
against a 20.1% shuffled control, i.e. below chance, where the per-row control was
2.9% and would have read as a strong signal).

    python3 eval/score_code_exec.py --preds data/eval/preds_code_fewshot.jsonl \
        --data data/eval/code_holdout_500.jsonl --demos 3
    python3 eval/score_code_exec.py --selftest

# restartable: reads preds, executes each span with a timeout, appends one summary
# record per file. No GPU. An interrupt costs a rerun.
"""

import argparse
import ast
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

#: A span must START at one of these. Without an anchor the search finds a lone
#: `return True` inside a function body -- syntactically invalid, but a bare
#: expression line like `1` parses fine and would count as "extracted code".
_START = re.compile(r"^(def |import |from |class |for |while |if |print\(|[A-Za-z_]\w*\s*=)")


def longest_parseable(text):
    """The longest line-contiguous span of `text` that ast.parse accepts, or ''.

    Line-contiguous rather than "all code anywhere": stitching non-adjacent lines
    would build a program the model never emitted. Anchored starts only, longest
    span wins, so a 20-line function beats the `print(...)` on its last line.

    Not a fence, and deliberately not a fence fallback either: preferring a fenced
    block when one exists would make the metric's value depend on a format the base
    was never taught, which is the whole defect this file exists to avoid.
    """
    lines = (text or "").split("\n")
    best = ""
    best_n = 0
    for s in range(len(lines)):
        if not _START.match(lines[s]):
            continue
        # longest first: the first e that parses from this start is the best from it
        for e in range(len(lines), s, -1):
            if e - s <= best_n:
                break
            blk = "\n".join(lines[s:e])
            try:
                ast.parse(blk)
            except (SyntaxError, ValueError):
                continue
            best, best_n = blk, e - s
            break
    return best


def _norm(s):
    """Line-by-line rstrip, blank lines dropped -- identical to code_zh._norm_lines,
    so a span scored here and a fenced block scored there face the same judge."""
    return [ln.rstrip() for ln in (s or "").split("\n") if ln.strip() != ""]


def runs_to(code, expected, timeout=10):
    """True iff `code` executes cleanly and its stdout matches `expected`.

    A failure to RUN the sandbox is not a wrong answer and must not be scored as one.
    run_sandboxed raises when it is not root -- swallowing that into `False` would
    report 0.0% pass on any machine without the sandbox, which is precisely the shape
    of failure this file was written to stop: a zero that measures the harness. It
    propagates; score_file catches it once, up front, and refuses the whole file.
    """
    from datagen.sandbox_exec import run_sandboxed

    if not (code or "").strip():
        return False
    try:
        rc, out, _ = run_sandboxed(code, timeout=timeout)
    except RuntimeError:
        raise
    except Exception:
        # the CODE failed to run, which is a genuine zero for that row
        return False
    return rc == 0 and _norm(out) == _norm(expected)


def score_file(path, oracles, demo_codes=(), seed=0, timeout=10, limit=None):
    """(record, error). `oracles` maps question -> expected stdout."""
    if not os.path.exists(path):
        return None, f"no such preds file: {path}"
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        return None, f"{path} holds no rows"
    if limit:
        rows = rows[:limit]

    spans = [longest_parseable(r.get("gen", "")) for r in rows]
    exps = [oracles.get(r.get("q")) for r in rows]
    n = len(rows)
    extracted = sum(1 for s in spans if s.strip())
    matched = sum(1 for e in exps if e is not None)
    # THE MANIPULATION CHECK, computed and reported before the outcome measure. A
    # pass rate under a low extraction rate is not interpretable -- it is a
    # measurement of the prompt, which is what every code-500 zero turned out to be.
    rec = {
        "preds": os.path.relpath(path, ROOT) if path.startswith(ROOT) else path,
        "n_rows": n,
        "extraction_rate": round(extracted / n, 4),
        "oracle_matched": matched,
        # Stated, because a pass rate over a subset reads like a pass rate over the set.
        # If matched < n_rows the headline denominator is the matched count, and the
        # gap is a key-matching failure that no accuracy number can express.
        "unscorable_rows": n - matched,
    }
    if not extracted or not matched:
        rec["note"] = ("nothing to execute: extraction_rate "
                       f"{extracted / n:.1%}, oracles matched {matched}/{n}")
        return rec, None

    # The sandbox needs root. Probe it ONCE with a known answer before scoring
    # anything: without this, an unavailable sandbox reports pass_rate 0.0% and
    # reads exactly like a model that cannot code. A refusal, not a zero.
    try:
        if not runs_to("print(1)", "1", timeout):
            return None, ("sandbox known-answer FAILED: print(1) did not produce '1'. "
                          "The executor is broken; a pass rate from it would be a "
                          "measurement of the harness.")
    except RuntimeError as e:
        return None, f"sandbox unavailable ({e}); run this on the pod, do not score 0"

    # Rows whose question is not in the oracle map cannot be scored, and they are NOT
    # zeros -- scoring them would measure the key match, not the model. They were worse
    # than zeros: `oracles.get(q)` returns None, `_norm(None) == _norm("")`, so any span
    # that ran cleanly and printed nothing MATCHED its absent oracle and counted as a
    # PASS. Measured end to end on the pod: 4 rows, 2 with oracles and 1 genuinely
    # correct, and score_file returned 0.75 where the truth is 0.25 (Codex found the
    # mechanism, de confirmed it by running it, 2026-09-01).
    #
    # The denominator has to move with the numerator. Reporting hits over n_rows while
    # scoring only matched rows would understate by the unmatched fraction, which is the
    # same defect facing the other way.
    keep = [i for i, e in enumerate(exps) if e is not None]
    spans = [spans[i] for i in keep]
    exps = [exps[i] for i in keep]
    n_scored = len(keep)

    hits = [runs_to(s, e, timeout) for s, e in zip(spans, exps, strict=True)]
    rec["pass_rate"] = {"rate": round(sum(hits) / n_scored, 4), "n": n_scored,
                        "of_rows": n}

    # control 1: each span against another problem's oracle, same aggregation
    sh = [e for e in exps]
    random.Random(seed).shuffle(sh)
    ctrl = sum(runs_to(s, e, timeout) for s, e in zip(spans, sh, strict=True))
    rec["shuffled_control"] = {"rate": round(ctrl / n_scored, 4), "n": n_scored}

    # control 2: what pure demo-copying scores. Not covered by control 1 -- a copied
    # span is a real span, and shuffling oracles does not ask where a span came from.
    if demo_codes:
        best = 0
        for dc in demo_codes:
            best = max(best, sum(runs_to(dc, e, timeout) for e in exps))
        # n_scored, not n: a control divided by a larger denominator than the rate it
        # floors is a control biased downward, which is the direction that makes a
        # result look better than it is.
        rec["copyist_control"] = {"rate": round(best / n_scored, 4), "n": n_scored,
                                  "n_demo_solutions": len(demo_codes)}

    floor = max(rec["shuffled_control"]["rate"], rec.get("copyist_control", {}).get("rate", 0.0))
    rec["delta_pt"] = round((rec["pass_rate"]["rate"] - floor) * 100, 2)
    rec["control_floor"] = round(floor, 4)
    return rec, None


def selftest():
    """Known answers. The extractor is the new part, so it carries the pairs."""
    assert longest_parseable("print(1)") == "print(1)"
    assert longest_parseable("这是中文，没有代码。") == ""
    # a fence is not required, and the fence markers themselves must not be included
    g = "```python\ndef f():\n    return 2\nprint(f())\n```\n运行输出：\n2\n"
    got = longest_parseable(g)
    assert got == "def f():\n    return 2\nprint(f())", repr(got)
    # longest span wins: the trailing print alone also parses, the function must beat it
    assert got.count("\n") == 2
    # a mid-body line is not a valid start, so a truncated head does not yield garbage
    assert longest_parseable("    return False\n    return True") == ""
    # syntax error anywhere in a span shortens it rather than dropping to ''
    g2 = "def bad(:\n    pass\nprint(3)"
    assert longest_parseable(g2) == "print(3)", repr(longest_parseable(g2))

    # An ABSENT oracle must never be satisfiable. `oracles.get(q)` returns None for a
    # question the map does not hold, and `_norm(None) == _norm("")`, so any span that
    # ran cleanly and printed nothing MATCHED its missing oracle and scored as a pass.
    # Measured end to end on the pod before the fix: 4 rows, 2 with oracles and 1
    # genuinely correct, score_file returned pass_rate 0.75 against a truth of 0.25.
    # A 500-row file whose keys mostly miss would have read as high capability.
    # (Codex found the mechanism, de confirmed it by running it, 2026-09-01.)
    #
    # This runs WITHOUT the sandbox, deliberately: it is a comparison-logic defect, and
    # putting it behind the root check would have left it untested on every dev box.
    assert _norm(None) == [], _norm(None)
    assert _norm(None) == _norm(""), "the equality that made an absent oracle matchable"

    # ...and the same case through score_file itself, which is what actually had to
    # change. A stub executor stands in for the sandbox HERE and only here: the defect
    # is in which rows enter the denominator, not in how code runs, and the stub keeps
    # the case out of the root-gated half where it would go untested on every dev box.
    import tempfile

    _real = sys.modules.get("datagen.sandbox_exec")
    stub = type(sys)("datagen.sandbox_exec")
    stub.run_sandboxed = lambda code, timeout=10: (
        (0, "1\n", "") if code == "print(1)" else          # score_file's own probe
        (0, "7\n", "") if "print(7)" in code else
        (0, "999\n", "") if "print(999)" in code else (0, "", ""))
    sys.modules["datagen.sandbox_exec"] = stub
    try:
        d = tempfile.mkdtemp()
        p = os.path.join(d, "preds.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            for q, gen in (("k1", "print(7)"), ("k2", "print(999)"),
                           ("MISS-a", "x = 1"), ("MISS-b", "y = 2")):
                fh.write(json.dumps({"q": q, "gen": gen}) + "\n")
        rec, err = score_file(p, {"k1": "7", "k2": "7"}, timeout=1)
        assert err is None, err
        assert rec["oracle_matched"] == 2 and rec["unscorable_rows"] == 2, rec
        assert rec["pass_rate"] == {"rate": 0.5, "n": 2, "of_rows": 4}, (
            f"1 of the 2 scorable rows is correct, so 0.5 over n=2. Got {rec['pass_rate']} "
            f"-- 0.75 over n=4 is the pre-fix answer, counting two silent unmatched rows "
            f"as passes")
        assert rec["shuffled_control"]["n"] == 2, "the control shares the rate's denominator"
    finally:
        if _real is not None:
            sys.modules["datagen.sandbox_exec"] = _real
        else:
            sys.modules.pop("datagen.sandbox_exec", None)

    # the judge: right code passes, wrong code and a syntax error do not. Same three
    # cases code_zh.selfcheck uses, because a scorer without a known-answer pair is
    # not a scorer. The sandbox needs root, so off the pod this half SKIPS loudly --
    # it does not quietly pass. A green selftest that never executed anything is the
    # same lie as a 0.0% that never ran the model.
    if os.geteuid() != 0:
        print("selftest PARTIAL: extractor OK (7 cases). The executor and both "
              "controls need the sandbox, which needs root -- rerun on the pod:\n"
              "    ~/bin/pod \"cd /work/aupai && python3 eval/score_code_exec.py --selftest\"")
        return 0

    assert runs_to("print(7)", "7")
    assert not runs_to("print(8)", "7")
    assert not runs_to("def f(:\n  pass", "7")
    assert not runs_to("", "7")

    # END TO END, with the controls, on a fixture built so the answer is known: three
    # questions whose oracles are 1/2/3, and generations that print 1/2/3 correctly.
    # Real must be 100%. The shuffled control must be 0% -- printing 1 cannot satisfy
    # the oracle 2 -- which is what makes the delta readable. And the copyist control
    # must be 33%: a demo that prints 1 satisfies exactly the one question whose
    # answer is 1, so a metric that ignored it would credit a pure echo with a third
    # of its score.
    import tempfile

    oracles = {f"q{i}": str(i) for i in (1, 2, 3)}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "p.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for i in (1, 2, 3):
                f.write(json.dumps({"q": f"q{i}", "gen": f"print({i})\n运行输出：\n{i}\n"}) + "\n")
        rec, err = score_file(p, oracles, demo_codes=["print(1)"], seed=0)
        assert err is None, err
        assert rec["extraction_rate"] == 1.0, rec
        assert rec["pass_rate"]["rate"] == 1.0, rec
        assert rec["shuffled_control"]["rate"] == 0.0, (
            f"a span printing 1 must not satisfy the oracle for 2: {rec}")
        assert abs(rec["copyist_control"]["rate"] - 1 / 3) < 1e-6, (
            f"echoing one demo solution scores 1 of 3 oracles; the metric must price "
            f"that as the floor it is: {rec}")
        assert abs(rec["control_floor"] - 1 / 3) < 1e-6, rec

    print("selftest OK: extracts the longest anchored parseable span with no fence, "
          "rejects prose and mid-body fragments, executes with a known-answer pair, "
          "and prices both the shuffled and the demo-copying floor")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", action="append", help="prediction jsonl (repeatable)")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "eval", "code_holdout_500.jsonl"),
                    help="holdout jsonl supplying question -> expected_output")
    ap.add_argument("--demos", type=int, default=3,
                    help="demo count the preds were generated with. Part of the metric "
                         "definition, not a footnote: at 0 demos only 38.6%% of "
                         "generations contain a 'def' at all, against 94%% at 1-3, so a "
                         "rate without its demo count is not comparable to another.")
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="score the first N rows only")
    ap.add_argument("--json", help="append one record per preds file here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.preds:
        ap.error("--preds required (or --selftest)")

    with open(a.data, encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    oracles = {r["instruction"]: r.get("expected_output") for r in rows}
    demo_codes = [r["reference_code"] for r in rows[: max(0, a.demos)] if r.get("reference_code")]

    recs = []
    for p in a.preds:
        rec, err = score_file(p, oracles, demo_codes, timeout=a.timeout, limit=a.limit)
        if err:
            print(f"  {os.path.basename(p)[:60]:62s} ERROR: {err}", flush=True)
            continue
        rec["demos"] = a.demos
        recs.append(rec)
        print(f"\n{os.path.basename(rec['preds'])[:58]}  ({a.demos} demos)")
        # manipulation check FIRST, always. Under a low extraction rate the pass rate
        # measures the prompt, not the model.
        print(f"  extraction_rate      {rec['extraction_rate']:.1%}  "
              f"({rec['n_rows']} rows, {rec['oracle_matched']} matched an oracle)")
        if rec["extraction_rate"] < 0.20:
            print("  -> MANIPULATION DID NOT TAKE (<20%): the pass rate below measures "
                  "the prompt, not capability. Do not read it as a capability number.")
        if "pass_rate" not in rec:
            print(f"  {rec.get('note', 'nothing scored')}")
            continue
        print(f"  pass_rate            {rec['pass_rate']['rate']:.1%}")
        print(f"  shuffled_control     {rec['shuffled_control']['rate']:.1%}")
        if "copyist_control" in rec:
            print(f"  copyist_control      {rec['copyist_control']['rate']:.1%}  "
                  f"(best of {rec['copyist_control']['n_demo_solutions']} demo solutions)")
        verdict = ("above both controls" if rec["delta_pt"] > 0 else
                   "AT OR BELOW CONTROL -- not a reading")
        print(f"  -> {rec['delta_pt']:+.2f}pt over the {rec['control_floor']:.1%} floor; {verdict}")

    if a.json and recs:
        with open(a.json, "a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(recs)} record(s) to {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

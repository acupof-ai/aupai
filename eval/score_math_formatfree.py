#!/usr/bin/env python3
"""Score math predictions WITHOUT requiring the \\boxed{} template (de, 2026-09-01).

Why this exists. The sampled arm returned pass@8 0.0% on math-500 at every checkpoint,
which reads as "no capability". The diagnostic says otherwise:

    n=1439 generations, step24000, t=0.8, k=8, rep_stop off
      contain \\boxed :     0   (0.0%)
      contain any digit: 1011  (70.3%)
    gold answers containing \\boxed: 494/500 (99%)

math_zh.score extracts \\boxed{...}. The base checkpoint has never once produced it. So
the metric is gated on an output format the model was not taught, reasoning sits
downstream of a gate that never opens, and NO value of k can distinguish "cannot
reason" from "does not know the answer template".

This reads the same prediction files and asks a weaker question: does the correct
number appear as the generation's final number. Weaker on purpose --

  - it cannot tell a reasoned answer from a number that happens to be there, so it is
    an UPPER BOUND on arithmetic ability and never a capability claim
  - the false-positive rate is measurable and is measured below: scoring each
    generation against a DIFFERENT problem's gold answer gives the rate at which this
    scorer says yes by coincidence. A number that does not clear its own shuffled
    control is not a reading.

It does not replace the pre-registered metric. It is reported beside it.

    python3 eval/score_math_formatfree.py --preds data/eval/preds_X.jsonl [--json out]

# restartable: reads preds files and appends one summary line per file. No GPU, no
# generation, seconds per file -- an interrupt costs a rerun, and the append is
# idempotent in content because the same preds produce the same record.
"""

import argparse
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Numbers as they appear in these generations: optional sign, digits with optional
#: thousands separators, optional decimal part. Fractions and \frac are NOT handled --
#: they would need the same normalisation the boxed extractor does, and pretending
#: otherwise would quietly lower the score on exactly the harder problems.
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def numbers(text):
    """Every number in `text`, in order, as strings normalised of separators."""
    return [m.group(0).replace(",", "") for m in _NUM.finditer(text or "")]


def gold_number(gold):
    """The answer a gold string asserts: inside \\boxed{} when present, else the last
    number. 99% of math-500 golds carry \\boxed, so the fallback is rare and is there
    to avoid dropping 6 problems silently."""
    m = re.search(r"\\boxed\{([^}]*)\}", gold or "")
    src = m.group(1) if m else (gold or "")
    ns = numbers(src)
    return ns[-1] if ns else None


def final_number(gen):
    """The generation's last number. 'Last' rather than 'any': a generation containing
    every integer under 20 would otherwise match nearly any gold, and the sampled
    generations here run 664 chars on average and mention many numbers."""
    ns = numbers(gen)
    return ns[-1] if ns else None


def _eq(a, b):
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return a == b


def score_file(path, seed=0):
    """(stats, error). Reports greedy and sampled arms separately when the rows carry
    a `greedy` flag, plus the shuffled control."""
    if not os.path.exists(path):
        return None, f"no such preds file: {path}"
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return None, f"{path} holds no rows"

    golds = [gold_number(r.get("gold", "")) for r in rows]
    finals = [final_number(r.get("gen", "")) for r in rows]
    hit = [_eq(f, g) for f, g in zip(finals, golds)]

    # The control: each generation against ANOTHER problem's gold. Same generations,
    # same scorer, answers that cannot be right -- so whatever it scores is the rate at
    # which this method says yes by coincidence. Without it a 4% reading is unreadable.
    rng = random.Random(seed)
    shuffled = golds[:]
    rng.shuffle(shuffled)
    ctrl = [_eq(f, g) for f, g in zip(finals, shuffled)]

    def _rate(mask, sel=None):
        idx = [i for i in range(len(rows)) if sel is None or sel(rows[i])]
        if not idx:
            return None
        return sum(mask[i] for i in idx) / len(idx), len(idx)

    out = {
        "preds": os.path.relpath(path, ROOT) if path.startswith(ROOT) else path,
        "n_rows": len(rows),
        "with_number": sum(1 for f in finals if f is not None),
        "gold_parsed": sum(1 for g in golds if g is not None),
    }
    has_flag = any("greedy" in r for r in rows)
    all_rate, all_n = _rate(hit)
    ctrl_rate, _ = _rate(ctrl)
    out["all"] = {"rate": round(all_rate, 4), "n": all_n}
    out["shuffled_control"] = {"rate": round(ctrl_rate, 4), "n": all_n}
    if has_flag:
        g = _rate(hit, lambda r: r.get("greedy") is True)
        s = _rate(hit, lambda r: r.get("greedy") is False)
        if g:
            out["greedy"] = {"rate": round(g[0], 4), "n": g[1]}
        if s:
            out["sampled"] = {"rate": round(s[0], 4), "n": s[1]}
            # pass@k over the sampled draws of each problem, keyed by question text.
            byq = {}
            for r, h in zip(rows, hit):
                if r.get("greedy") is False:
                    byq.setdefault(r.get("q"), []).append(h)
            if byq:
                out["pass_at_k_formatfree"] = {
                    "rate": round(sum(1 for v in byq.values() if any(v)) / len(byq), 4),
                    "n_questions": len(byq),
                    "k_median": sorted(len(v) for v in byq.values())[len(byq) // 2],
                }
    return out, None


def selftest():
    """Known answers. A scorer without one is not a scorer -- and this one's whole
    purpose is to be believed where the strict one reads zero."""
    assert numbers("剩下 7 个") == ["7"]
    assert numbers("1,234.5 and -3") == ["1234.5", "-3"]
    assert gold_number(r"所以 \boxed{7} 个") == "7"
    assert gold_number("答案是 42") == "42", "a gold without boxed must still parse"
    assert final_number("先算 10 减 3 得 7") == "7", "the LAST number, not the first"
    assert final_number("没有数字") is None
    assert _eq("7", "7.0") and not _eq("7", "9")
    # the case that motivates the whole file: a right answer in the wrong format
    assert _eq(final_number("小明还剩 7 个苹果"), gold_number(r"\boxed{7}")), \
        "an unformatted correct answer must score"
    # and the case that keeps it honest: a generation full of numbers is not a hit
    assert not _eq(final_number("1 2 3 4 5 6"), gold_number(r"\boxed{7}"))
    print("selftest OK: parses, prefers the last number, scores unformatted answers, "
          "rejects number soup")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", action="append", help="prediction jsonl (repeatable)")
    ap.add_argument("--json", help="append one record per preds file here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.preds:
        ap.error("--preds required (or --selftest)")

    recs = []
    for p in a.preds:
        rec, err = score_file(p)
        if err:
            print(f"  {os.path.basename(p)[:60]:62s} ERROR: {err}", flush=True)
            continue
        recs.append(rec)
        name = os.path.basename(rec["preds"])[:58]
        print(f"\n{name}")
        print(f"  rows {rec['n_rows']}, {rec['with_number']} carry a number, "
              f"{rec['gold_parsed']} golds parsed")
        for k in ("all", "greedy", "sampled", "pass_at_k_formatfree", "shuffled_control"):
            if k in rec:
                v = rec[k]
                n = v.get("n") or v.get("n_questions")
                print(f"  {k:22s} {v['rate']:.1%}  (n={n})")
        ctrl = rec["shuffled_control"]["rate"]
        if rec["all"]["rate"] <= ctrl:
            print(f"  -> AT OR BELOW its shuffled control ({ctrl:.1%}): not a reading")
        else:
            print(f"  -> above control by {(rec['all']['rate'] - ctrl) * 100:.1f}pt")

    if a.json and recs:
        with open(a.json, "a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(recs)} record(s) to {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

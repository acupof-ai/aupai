#!/usr/bin/env python3
"""Score procedure EXECUTION separately from the final answer, on held-out problems.

Three axes: ANSWER (final value right), STEPS (every intermediate line true, checked line
by line), BOTH. BOTH is the number that means something; ANSWER-without-STEPS gets its own
column so that failure cannot hide inside a headline.

Problems come from procedure_curriculum's held-out side through the SAME blake2b split the
training set was filtered by -- an independent sample scores memorisation.

    python scripts/probe_procedure.py --ckpt ckpt_X.pt --tokenizer data/tokenizer_k8.json
    python scripts/probe_procedure.py --selftest      # no GPU: the scorers, on gold data
"""

import argparse
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE, os.path.join(ROOT, "mathbank")):
    sys.path.insert(0, p)


def steps_valid(fmt, text):
    """(n_checked, n_true) over the intermediate lines a generation actually emits.

    (0, 0) means no checkable line at all -- the answer-first failure, so the caller must
    not read it as a pass. Lines in shapes the curriculum does not teach are not counted,
    so the metric is generous and a low score is still a low score."""
    ok = n = 0

    # `a × b + 进位c = d，写 e，进位 f`  (long multiplication, one digit column)
    for a, b, c, d in re.findall(r"(\d+)\s*×\s*(\d+)\s*\+\s*进位\s*(\d+)\s*=\s*(\d+)", text):
        n += 1
        ok += int(a) * int(b) + int(c) == int(d)
    # `a + b + 进位c = d`  (addition column, shared with the arithmetic curriculum)
    for a, b, c, d in re.findall(r"(\d+)\s*\+\s*(\d+)\s*\+\s*进位\s*(\d+)\s*=\s*(\d+)", text):
        n += 1
        ok += int(a) + int(b) + int(c) == int(d)
    # `x = y × k z = w u`  (one hop of a unit-conversion chain)
    for a, k, b in re.findall(r"(\d+)\s*\S*\s*=\s*\1\s*×\s*(\d+)\s*\S*\s*=\s*(\d+)", text):
        n += 1
        ok += int(a) * int(k) == int(b)
    # `两边同时减 b：ax = c - b = d`   /   `两边同时加 b：ax = c + b = d`
    for op, b, c, sgn, b2, d in re.findall(
        r"两边同时([加减])\s*(\d+)[：:]\s*\d*x\s*=\s*(\d+)\s*([-+])\s*(\d+)\s*=\s*(\d+)", text
    ):
        n += 1
        want = int(c) + int(b2) if sgn == "+" else int(c) - int(b2)
        ok += want == int(d) and b == b2 and (op == "加") == (sgn == "+")
    # `两边同时除 a：x = d ÷ a = e`
    for a, d, a2, e in re.findall(r"两边同时除\s*(\d+)[：:]\s*x\s*=\s*(\d+)\s*÷\s*(\d+)\s*=\s*(\d+)", text):
        n += 1
        ok += a == a2 and int(a) and int(d) == int(e) * int(a)
    # `第 k 部分积 = v` is not self-checking without the operands; skipped, not scored loosely.
    return n, ok


FINAL = {
    "mul": re.compile(r"(?:结果|积)\s*[=＝]\s*(-?\d+)"),
    "unit": re.compile(r"结果\s*[=＝]\s*(-?\d+)"),
    "eq": re.compile(r"结果\s*x\s*[=＝]\s*(-?\d+)"),
}


def final_answer(fmt, text):
    m = FINAL[fmt].search(text)
    if m:
        return int(m.group(1))
    # No terminator: take the last number rather than zero, so ANSWER is not measuring format.
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None


def gold_answer(fmt, out):
    return final_answer(fmt, out)


def load_cases(n_per, seed=0):
    import procedure_curriculum as pc

    rows = pc.generate(n_per * 6, random.Random(seed), split="test")
    by = {}
    for r in rows:
        by.setdefault(r["fmt"], []).append(r)
    cases = []
    for fmt, rs in by.items():
        cases += rs[:n_per]
    return cases


def _demo():
    """The scorers on gold generations (STEPS and ANSWER must be 100%, else the parser is
    broken and every number this file prints is noise) and then on corrupted gold, which
    must drop."""
    import procedure_curriculum as pc

    rows = pc.generate(900, random.Random(3), split="all")
    tot = {"n": 0, "ans": 0, "chk": 0, "ok": 0}
    for r in rows:
        n, ok = steps_valid(r["fmt"], r["output"])
        tot["n"] += 1
        tot["chk"] += n
        tot["ok"] += ok
        tot["ans"] += final_answer(r["fmt"], r["output"]) == gold_answer(r["fmt"], r["output"])
    assert tot["chk"] > 2 * tot["n"], f"only {tot['chk']} checkable lines over {tot['n']} rows"
    assert tot["ok"] == tot["chk"], f"gold data failed its own step check: {tot['ok']}/{tot['chk']}"
    assert tot["ans"] == tot["n"], f"gold answers not parsed: {tot['ans']}/{tot['n']}"

    # Corrupt the result of each step (the number after the FIRST `=`), or the corruption
    # breaks nothing and a working checker still scores true.
    bad_chk = bad_ok = 0
    for r in rows:
        broken = "\n".join(
            re.sub(r"(=\s*)(\d+)", lambda m: m.group(1) + str(int(m.group(2)) + 7), ln, count=1)
            for ln in r["output"].split("\n")
        )
        n, ok = steps_valid(r["fmt"], broken)
        bad_chk += n
        bad_ok += ok
    assert bad_chk and bad_ok < 0.1 * bad_chk, (
        f"corrupted steps still scored {bad_ok}/{bad_chk}: the step checker does not check"
    )

    # A bare right answer must score ANSWER without STEPS.
    r = next(x for x in rows if x["fmt"] == "eq")
    g = gold_answer("eq", r["output"])
    n, ok = steps_valid("eq", f"结果 x = {g}")
    assert n == 0, f"a bare answer produced {n} checkable steps"
    assert final_answer("eq", f"结果 x = {g}") == g
    print(f"probe_procedure self-test OK ({tot['n']} gold rows, {tot['chk']} step-lines, corruption caught)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer.json"))
    ap.add_argument("--fone", action="store_true")
    ap.add_argument("--n", type=int, default=60, help="held-out problems per procedure")
    ap.add_argument("--steps", type=int, default=220)
    ap.add_argument("--batch", type=int, default=60, help="problems decoded per forward pass")
    a = ap.parse_args()

    import torch

    import fone

    from loader import load_checkpoint, load_tokenizer

    model, cfg = load_checkpoint(a.ckpt, device="cuda:0", dtype=torch.bfloat16)
    tok = load_tokenizer(a.tokenizer, cfg)
    cases = load_cases(a.n)
    print(f"{a.ckpt}\n{len(cases)} HELD-OUT problems (blake2b split, same filter as training)\n", flush=True)
    texts = fone.generate_texts(model, tok, cfg, [c["instruction"] for c in cases], a.steps, batch=a.batch)
    agg = {}
    for c, t in zip(cases, texts, strict=True):
        n, ok = steps_valid(c["fmt"], t)
        ans = final_answer(c["fmt"], t) == gold_answer(c["fmt"], c["output"])
        steps_ok = n > 0 and ok == n
        d = agg.setdefault(c["fmt"], {"n": 0, "ans": 0, "steps": 0, "both": 0, "nostep": 0})
        d["n"] += 1
        d["ans"] += ans
        d["steps"] += steps_ok
        d["both"] += ans and steps_ok
        d["nostep"] += ans and n == 0
    print(f"  {'procedure':<14}{'ANSWER':>9}{'STEPS':>9}{'BOTH':>9}{'answer w/o any step':>22}")
    for fmt, d in agg.items():
        n = d["n"]
        print(
            f"  {fmt:<14}{100 * d['ans'] / n:>8.1f}%{100 * d['steps'] / n:>8.1f}%"
            f"{100 * d['both'] / n:>8.1f}%{100 * d['nostep'] / n:>21.1f}%"
        )
    print(
        "\n  BOTH is the number that means something. A high ANSWER with a low STEPS is the\n"
        "  k6_arith4 failure: the model guesses the result and writes procedure-shaped prose\n"
        "  around it, which does not survive an unseen problem."
    )


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _demo()
    else:
        main()

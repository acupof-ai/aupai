#!/usr/bin/env python3
"""control_sft answer-error lower bound: arithmetic-equation check.
For each answer, regex-extract simple arithmetic equations NUM OP NUM = NUM
(also .. = NUM). Evaluate LHS, compare to RHS; a mismatch = definite error.
Only plain integer/decimal + - * / (no frac/pow) so the check is exact, not
heuristic -- this is a LOWER BOUND on the error rate (the missable classes are
prose/contextual errors). Read-only. Python 3 only.
"""
import json
import re
import sys

P = re.compile(
    r"(?<![\d.])"
    r"(\d+(?:\.\d+)?)\s*([+*\-/x×])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)"
)
OP = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
      "*": lambda a, b: a * b, "x": lambda a, b: a * b,
      "×": lambda a, b: a * b, "/": lambda a, b: a / b}
# also lone 'a + b = c' with LHS maybe parenthesized /\boxed on RHS tolerated:
P2 = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*([+*\-/x×])\s*(\d+(?:\.\d+)?)\s*=\s*\\?\w*\s*\{\s*(\d+(?:\.\d+)?)\s*\}"
)


def eqs(t):
    out = []
    for m in P.finditer(t):
        a, op, b, c = m.group(1), m.group(2), m.group(3), m.group(4)
        try:
            lhs = OP[op](float(a), float(b))
        except Exception:
            continue
        out.append((a, op, b, c, lhs))
    return out


def main():
    p = sys.argv[1]
    n = wrong = 0
    rows_seen = 0
    bad_examples = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows_seen += 1
            d = json.loads(line)
            ans = d.get("answer") or ""
            for a, op, b, c, lhs in eqs(ans):
                n += 1
                try:
                    cval = float(c)
                except ValueError:
                    continue
                if abs(lhs - cval) > 1e-6:
                    wrong += 1
                    if len(bad_examples) < 15:
                        bad_examples.append((d.get("id"), f"{a}{op}{b}={c}", lhs, d.get("src")))
    print(f"rows={rows_seen} eqs_found={n} eqs_wrong={wrong}",
          flush=True)
    if n:
        print(f"lower_bound_error_rate_among_eqs={wrong/max(1,n):.4%} (CI gross: +-{1.96*((wrong/n*(1-wrong/n))/max(1,n))**0.5:.4f})",
              flush=True)
    for e in bad_examples:
        print("BAD", e, flush=True)


if __name__ == "__main__":
    main()
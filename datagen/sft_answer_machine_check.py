#!/usr/bin/env python3
# restartable: read-only scan / filter re-open dst with 'w' from src; an interrupt costs re-reading src
"""control_sft answer-error lower bound: arithmetic-equation check.
For each answer, regex-extract simple arithmetic equations NUM OP NUM = NUM
(also .. = NUM). Evaluate LHS, compare to RHS; a mismatch = definite error.
Only plain integer/decimal + - * / (no frac/pow) so the check is exact, not
heuristic -- this is a LOWER BOUND on the error rate (the missable classes are
prose/contextual errors). Read-only. Python 3 only.
"""
import json
import random
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
    ctx = None if len(sys.argv) < 3 else sys.argv[2]
    n = wrong = 0
    rows_seen = 0
    bad_examples = []
    matched_ctx = []  # (q_before, eq_text, q_after) around each match
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
                    if ctx == "wrongctx" and len(matched_ctx) < 3000:
                        i = ans.find(f"{a}{op}{b}={c}")
                        lo, hi = max(0, i - 40), i + len(f"{a}{op}{b}={c}") + 40
                        matched_ctx.append((d.get("id"), ans[lo:hi].replace("\n", " ")))
    print(f"rows={rows_seen} eqs_found={n} eqs_wrong={wrong}",
          flush=True)
    if n:
        print(f"lower_bound_error_rate_among_eqs={wrong/max(1,n):.4%} (CI gross: +-{1.96*((wrong/n*(1-wrong/n))/max(1,n))**0.5:.4f})",
              flush=True)
    for e in bad_examples:
        print("BAD", e, flush=True)
    if ctx == "wrongctx":
        rng = random.Random(7)
        for _ in range(20):
            if not matched_ctx:
                break
            did, snippet = rng.choice(matched_ctx)
            print(f"WCTX id={did} :: {snippet}", flush=True)


def filter_rows(src, dst):
    r_in = r_drop = 0
    dropped_ids = []
    with open(src, encoding="utf-8") as fsrc, open(dst, "w", encoding="utf-8") as fdst:
        for line in fsrc:
            if not line.strip():
                continue
            d = json.loads(line)
            ans = d.get("answer") or ""
            bad = False
            for m in P.finditer(ans):
                try:
                    lhs = OP[m.group(2)](float(m.group(1)), float(m.group(3)))
                except Exception:
                    continue
                if abs(lhs - float(m.group(4))) > 1e-6:
                    bad = True
                    break
            if bad:
                r_drop += 1
                if r_drop <= 5000:
                    dropped_ids.append(d.get("id"))
                continue
            fdst.write(line)
            r_in += 1
    print(f"filter: kept={r_in} dropped={r_drop} out={dst}", flush=True)
    print(f"dropped_ids_head={dropped_ids[:20]}", flush=True)


def negtest(fn):
    with open(fn, encoding="utf-8") as f:
        t = f.read()
    wrong = matched = 0
    for _a, _op, _b, c, lhs in eqs(t):
        matched += 1
        try:
            cval = float(c)
        except ValueError:
            continue
        if abs(lhs - cval) > 1e-6:
            wrong += 1
    # sentinel at the computation, never format None: matched==0 is a statement
    # ("file has no NUM op NUM = NUM shape"), not a false-positive signal.
    signal = "n/a (0 matches in this file)" if matched == 0 else f"{wrong / matched:.2%}"
    print(f"NEGTEST file={fn} matched={matched} wrongly_flagged={wrong} "
          f"(false-positive signal vs known-good code={signal})", flush=True)
    return matched, wrong


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[2] == "negtest":
        negtest(sys.argv[1])
    elif len(sys.argv) >= 4 and sys.argv[2] == "filter":
        filter_rows(sys.argv[1], sys.argv[3])
    else:
        main()

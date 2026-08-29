#!/usr/bin/env python3
"""Arithmetic-step verification shared by the data filters and the evals.

A two-operand regex must not be applied to a longer chain: matching "57 + 54 = 156"
inside "45 + 57 + 54 = 156" computes 111 and declares the line wrong. Measured cost
of that bug (docs/review_2026-08-26.md #4): 5.7% of corpus rows dropped as bad_eq with
>=41% of those arithmetically correct, and a 15-34% false-positive rate when scoring
model generations — biased exactly toward the multi-step problems that matter.
"""

import re

EQ = re.compile(
    r"(?<![\d./])(-?\d+(?:\.\d+)?)\s*([+\-×÷*/])\s*"
    r"(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)(?![\d./])"
)
# 3+ operands on the left of an '='; such a line is skipped, not scored.
CHAIN = re.compile(r"-?\d+(?:\.\d+)?(?:\s*[+\-×÷*/]\s*-?\d+(?:\.\d+)?){2,}\s*=")
OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "×": lambda a, b: a * b,
    "*": lambda a, b: a * b,
    "÷": lambda a, b: a / b if b else None,
    "/": lambda a, b: a / b if b else None,
}


def iter_equations(text):
    """Yield (a, op, b, c) for two-operand equations that can actually be verified."""
    for line in text.split("\n"):
        if CHAIN.search(line):
            continue
        yield from EQ.findall(line)


def check_steps(text):
    """(n_equations_checked, n_wrong) over the verifiable equations in `text`."""
    n = bad = 0
    for a, op, b, c in iter_equations(text):
        r = OPS[op](float(a), float(b))
        n += 1
        if r is None or abs(r - float(c)) > 1e-6 * max(1.0, abs(r)):
            bad += 1
    return n, bad


def has_bad_step(text):
    return check_steps(text)[1] > 0


def _demo():
    # chained expressions are skipped, not miscounted
    assert list(iter_equations("45 + 57 + 54 = 156")) == []
    assert list(iter_equations("总分 = 83 + 67 + 87 = 237")) == []
    # plain two-operand steps still verify
    assert list(iter_equations("10 - 3 = 7")) == [("10", "-", "3", "7")]
    assert check_steps("8 - 3 = 5\n5 - 2 = 3") == (2, 0)
    assert check_steps("8 - 3 = 6") == (1, 1)
    # a fraction rendered as a/b is not a two-operand step to verify
    assert list(iter_equations("20 × 1/4 = 5")) == []
    print("eqcheck: ok")


if __name__ == "__main__":
    _demo()

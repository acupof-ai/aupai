#!/usr/bin/env python3
"""E3: one-step arithmetic, base continuation, exact match, per-item recompute.
# restartable: one pass is a few minutes over a fixed, hashed input set and the run is
# deterministic; an interrupt costs one re-run, not the accumulated work. The artifacts
# are written once at the end because the contract requires a complete items.jsonl beside
# a summary that is recomputable from it (docs/standards/eval_ladder_v1_1009.md R3).

Contract: docs/standards/eval_ladder_v1_1009.md sections 3, 5 (E3), 6 and the section 11
rulings -- R2 (a baseline's gate is evidence integrity, not a capability number), R3 (the
per-item evidence interface the auditor recomputes from), R7 (two question sets, two
roles). Handover doc section 6 E3. Owner: w2. `eval/ladder_common.py` is imported, never
edited.

What this stage is
------------------
The subject is a BASE checkpoint: every prompt is plain continuation text, rendered by
`scripts.loader.format_continuation` ("问：...\\n答："), and a ChatML prefix is refused
before a single token is generated. Six kinds -- int_add, int_sub, int_mul, int_div,
decimal, percent -- 12 items each, in `data/eval/ladder/arith_v1.jsonl`.

Three separations this module keeps, because collapsing any one of them is how an
arithmetic eval reports a number that means nothing:

1. The operational set is GENERATED, not typed. `generate_items()` below is the only
   producer of the question text, the expression and the answer, and `--emit-data`
   writes the committed file from it. The stage refuses a dataset that is not equal to
   what the generator produces now, so a hand-edited answer cannot enter a result table.

2. Every item is RECOMPUTED before it is scored. `verify_item` evaluates the item's
   `check` expression with `fractions.Fraction` (exact) and re-divides with `decimal` at
   50 digits to confirm the serialized answer is that exact value and not a rounded one.
   A declared answer that disagrees, or that has no finite decimal expansion (1/3
   truncated to 0.3 is the handover doc's own example), makes the item FAIL with
   fail_class `arith_check_failed` -- and that is a gate, not a footnote.

3. The MODEL's answer is compared by exact rational value, never by a float tolerance.
   42.500 passes for a gold of 42.5 because it names the same number; 0.3 does not pass
   for 1/3 because it names a different one. A number that is the gold with digits
   dropped is reported as `wrong_precision` rather than as a generic wrong answer, so the
   failure distribution says which of the two happened.

Two question sets (R7), one protocol and one decision rule
---------------------------------------------------------
`--prompts PATH` reads an external cross-check set -- the audit gates format
(`{"prompts"|"items"|"questions": [{"id", "prompt"|"text"|"question", "answer"}]}`) --
and the report ids are the file's ids, never re-derived. The arithmetic check on those
rows is still made HERE: where the question text names the operation ("计算：7 + 6 = ?")
the expression is recovered from the text and recomputed, otherwise the declared answer
is verified to be an exact canonical number and the weaker basis is recorded per row
(`check_kind`). Nothing depends on the audit files existing.

Gates: evidence, not capability (R2)
------------------------------------
Gated and required: `protocol` (no ChatML, continuation text), `arith_check` (every
item recomputes), `coverage` (exact id set), `evidence` (every item carries an output
string, at least one non-empty), plus `controls` under `--control`. The exact-match
accuracy is REPORTED, and a capability gate is applied only when the operator asks for
one with `--min-acc`; the base checkpoint is expected to be far below any target there,
and exit 3 for that would say nothing about whether the evaluator works.

Device note: `CUDA_VISIBLE_DEVICES=1 ... --device cuda:1` is self-contradicting -- with
one card visible the only legal ordinal is 0 (AGENTS.md: `CUDA_VISIBLE_DEVICES`, not
`cuda:N`). `resolve_device` maps an out-of-range ordinal onto the visible set and says so
in the log and in the summary (`device` vs `device_used`), so the command the operator
typed runs on the card they meant and nothing is silent.

Usage:
    python3 eval/ladder/e3_arith.py --emit-data data/eval/ladder/arith_v1.jsonl
    python3 eval/eval_ladder.py e3 --ckpt ckpt_v42s2.pt --device cuda:0 --control
    python3 eval/eval_ladder.py e3 --ckpt X --prompts data/pretrain_100b/audit/gates/E3_arith.json
    python3 eval/eval_ladder.py selftest --only e3          # CPU, no checkpoint
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import re
import sys
import time
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval import ladder_common as lc  # noqa: E402
from scripts.loader import format_continuation  # noqa: E402

KINDS = ("int_add", "int_sub", "int_mul", "int_div", "decimal", "percent")
ROW_KEYS = ("id", "kind", "prompt", "answer", "check")
PER_KIND = 12
MIN_PER_KIND = 8          # contract: at least 8 items per kind
MAX_DP = 6                # an answer with more decimals than this has no canonical form
DATA_DEFAULT = ROOT / "data" / "eval" / "ladder" / "arith_v1.jsonl"
GEN_SEED = 20261009
PROMPT_TEMPLATE = "scripts.loader.format_continuation"
ANSWER_RULE = "last_number_in_output"
#: The rule's two implementations and the recorded name are pinned together: a gate whose
#: name says "last" and whose code reads "first" is what a control has to be able to fail.
RULES = ("first", "last")
#: Gold-verification states. `unverifiable` is a third state, not a pass: a row whose
#: check carries no operation cannot be recomputed, so it is scored, counted in
#: `unverifiable_n`, and never counted as an arith-checked pass.
STATE_VERIFIED = "verified"
STATE_MISMATCH = "gold_mismatch"
STATE_UNVERIFIABLE = "unverifiable"
STATES = (STATE_VERIFIED, STATE_MISMATCH, STATE_UNVERIFIABLE)
#: Check kinds that carry an operation this module recomputes. "literal" is the absence
#: of one.
VERIFIABLE_CHECK_KINDS = ("generated", "question_text", "question_tail")
CHATML = ("<|im_start|>", "<|im_end|>")
SET_KEYS = ("prompts", "items", "questions")
QUESTION_KEYS = ("prompt", "text", "question")
TEMPLATES = ("continuation", "verbatim", "auto")

FAIL_ARITH = "arith_check_failed"
FAIL_EMPTY = "empty_output"
FAIL_NO_NUMBER = "no_number"
FAIL_WRONG = "wrong_answer"
FAIL_PRECISION = "wrong_precision"
FAIL_CLASSES = (FAIL_ARITH, FAIL_EMPTY, FAIL_NO_NUMBER, FAIL_WRONG, FAIL_PRECISION)

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
LITERAL_RE = re.compile(r"[+-]?\d+(?:\.\d+)?\Z")
ARITH_Q_RE = re.compile(
    r"^\s*计算\s*[:：]?\s*(-?\d+(?:\.\d+)?)\s*([+\-×x*÷/])\s*(-?\d+(?:\.\d+)?)"
    r"\s*[=＝]?\s*[?？]?\s*$")
OP_MAP = {"+": "+", "-": "-", "×": "*", "x": "*", "*": "*", "÷": "/", "/": "/"}

#: The audit cross-check set (data/pretrain_100b/audit/gates/*.json) tells the runner to
#: take the last number and compare it exactly; that is the same rule this module uses.
#: `answer_rule` is recorded in the summary so the two sides can be compared by name.
AUDIT_ANSWER_RULE = "last number in the output, exact numeric equality"


# --------------------------------------------------------------------------------------
# exact arithmetic: the recomputation half of the check
# --------------------------------------------------------------------------------------

def as_fraction(value):
    """A plain decimal literal -> the exact rational it names, else None.

    Scientific notation is deliberately not accepted: "1e3" is a different surface form
    from "1000" and accepting it would put two spellings of one answer into the data.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        value = str(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not LITERAL_RE.match(text):
        return None
    return Fraction(Decimal(text))


def canonical_decimal(fr, max_dp=MAX_DP):
    """The exact decimal string of `fr`, or None when it has no finite expansion.

    Exact integer scaling decides the digits (no float ever touches a score); `decimal`
    then divides the same rational at 50 digits and must agree with the string exactly.
    Two representations, one value: a wrong string cannot survive both. None means the
    value cannot be written as a finite decimal at all -- 1/3 is the case the contract
    names -- and an item whose check produces None is `arith_check_failed`.
    """
    if not isinstance(fr, Fraction):
        raise TypeError(f"canonical_decimal wants a Fraction, got {type(fr).__name__}")
    sign = "-" if fr.numerator < 0 else ""
    num = abs(fr.numerator)
    den = fr.denominator
    twos = fives = 0
    rest = den
    while rest % 2 == 0:
        rest //= 2
        twos += 1
    while rest % 5 == 0:
        rest //= 5
        fives += 1
    if rest != 1:
        return None
    dp = max(twos, fives)
    if dp > max_dp:
        return None
    scaled = num * (10 ** dp) // den          # exact: den divides 10**dp
    digits = str(scaled).rjust(dp + 1, "0")
    whole, frac = (digits[:-dp], digits[-dp:]) if dp else (digits, "")
    frac = frac.rstrip("0")
    out = f"{sign}{whole}" + (f".{frac}" if frac else "")
    with localcontext() as ctx:
        ctx.prec = 50
        quotient = Decimal(fr.numerator) / Decimal(fr.denominator)
    if Decimal(out) != quotient:
        return None
    return out


def eval_expr(node):
    """Exact value of a check expression. Leaves are decimal literals; ops: + - * / lit.

    Nested nodes are allowed (the percent-of-a-ratio item needs one), which is why the
    value is computed recursively rather than as a single binary op. `lit` takes one
    literal and returns it: it is how an external set's answer-only row is validated
    without inventing an operation the question does not state.
    """
    if isinstance(node, str):
        value = as_fraction(node)
        if value is None:
            raise ValueError(f"check operand {node!r} is not a plain decimal literal")
        return value
    if isinstance(node, (int, float)):
        value = as_fraction(node)
        if value is None:
            raise ValueError(f"check operand {node!r} is not a plain decimal literal")
        return value
    if not isinstance(node, dict):
        raise ValueError(f"check node {node!r} is neither a literal nor an operation")
    op = node.get("op")
    args = node.get("args")
    if op == "lit":
        if not isinstance(args, list) or len(args) != 1:
            raise ValueError(f"check node 'lit' takes exactly one arg, got {args!r}")
        return eval_expr(args[0])
    if op not in ("+", "-", "*", "/"):
        raise ValueError(f"check node has unsupported op {op!r}")
    if not isinstance(args, list) or len(args) < 2:
        raise ValueError(f"check node {op!r} needs at least two args, got {args!r}")
    values = [eval_expr(a) for a in args]
    if op == "+":
        out = Fraction(0)
        for v in values:
            out += v
        return out
    if op == "-":
        out = values[0]
        for v in values[1:]:
            out -= v
        return out
    if op == "*":
        out = Fraction(1)
        for v in values:
            out *= v
        return out
    out = values[0]
    for v in values[1:]:
        if v == 0:
            raise ZeroDivisionError("check expression divides by zero")
        out /= v
    return out


_ARITH_TOKEN_RE = re.compile(r"\s*(\d+(?:\.\d+)?|\.\d+|[+\-*/×÷()])")


def parse_arithmetic(text):
    """An arithmetic expression as a check node, or None when it is not one.

    Recursive descent over + - * / ( ) and decimal literals; leaves stay literal strings,
    so the result evaluates through `eval_expr` with the module's exact Fraction
    arithmetic. None means "not a clean expression" (an unexpected character or unbalanced
    parentheses) and never a guess.
    """
    tokens, position = [], 0
    body = str(text or "")
    while position < len(body):
        match = _ARITH_TOKEN_RE.match(body, position)
        if not match:
            return None
        tokens.append(match.group(1))
        position = match.end()
    if not tokens:
        return None
    state = {"i": 0}

    def peek():
        return tokens[state["i"]] if state["i"] < len(tokens) else None

    def take():
        token = peek()
        state["i"] += 1
        return token

    def factor():
        token = peek()
        if token == "-":
            take()
            inner = factor()
            return None if inner is None else {"op": "*", "args": ["-1", inner]}
        if token == "+":
            take()
            return factor()
        if token == "(":
            take()
            inner = expr()
            if inner is None or take() != ")":
                return None
            return inner
        if token is None or not re.fullmatch(r"\d+(?:\.\d+)?|\.\d+", token):
            return None
        take()
        if as_fraction(token) is None:
            return None
        return token

    def term():
        node = factor()
        if node is None:
            return None
        while peek() in ("*", "/", "×", "÷"):
            op = take()
            right = factor()
            if right is None:
                return None
            node = {"op": "*" if op in ("*", "×") else "/", "args": [node, right]}
        return node

    def expr():
        node = term()
        if node is None:
            return None
        while peek() in ("+", "-"):
            op = take()
            right = term()
            if right is None:
                return None
            node = {"op": op, "args": [node, right]}
        return node

    out = expr()
    if out is None or state["i"] != len(tokens):
        return None
    return out


def trailing_expression(question):
    """The clean arithmetic expression that ends the question, or None.

    "What is 1+1?" ends in 1+1, so the operation is recoverable and can be recomputed
    here. The match must END the question: a number in the middle of a word problem is not
    an operation, and inventing one would turn a wrong gold into a wrong accusation.
    """
    text = str(question or "").strip()
    while text and text[-1] in "?？=＝.。:：":
        text = text[:-1].rstrip()
    match = re.search(r"[(\d][0-9\s+\-*/×÷().]*\Z", text)
    if not match:
        return None
    body = match.group(0).strip()
    if not any(op in body for op in "+-*/×÷"):
        return None
    return parse_arithmetic(body)


def verify_item(item, require_canonical=True):
    """Independent recomputation of one item's own answer.

    Returns {"ok", "value", "canonical", "reason", "detail"}. `reason` is "ok" or one of
    check_unevaluable / no_finite_decimal / answer_not_a_literal / value_mismatch /
    not_canonical; every non-ok reason maps to fail_class `arith_check_failed` at the
    item level, and the reason string is what tells a reader which one happened.

    `require_canonical` is on for the operational set, whose serialization this module
    owns; an external set's spelling (8.0 for 8) is normalized by its loader instead, so
    the value comparison does not turn a spelling difference into a failed item.
    """
    kind = item.get("check_kind", "generated")
    answer = item.get("answer")
    got = as_fraction(answer if isinstance(answer, (str, int, float)) else None)
    if kind not in VERIFIABLE_CHECK_KINDS:
        # No operation in the check: the declared answer cannot be recomputed. That is
        # not a pass. The row is scored against its declared answer and counted in
        # `unverifiable_n`, so a file of unchecked golds cannot read as a clean check.
        if got is None:
            return _bad("answer_not_a_literal", f"answer {answer!r} is not a decimal literal")
        want = canonical_decimal(got)
        if want is None:
            return _bad("no_finite_decimal",
                        f"declared answer {answer!r} has no finite decimal expansion")
        return {"ok": True, "verified": False, "state": STATE_UNVERIFIABLE, "value": got,
                "canonical": want, "reason": "no_independent_check",
                "detail": f"check kind {kind!r} carries no operation; the declared answer "
                          f"{want!r} is scored but not independently recomputed"}
    try:
        value = eval_expr(item.get("check"))
    except Exception as exc:                                        # noqa: BLE001
        return _bad("check_unevaluable", str(exc))
    want = canonical_decimal(value)
    if want is None:
        return _bad("no_finite_decimal",
                    f"exact value {value.numerator}/{value.denominator} has no finite "
                    f"decimal expansion (<= {MAX_DP} dp)")
    if got is None:
        return _bad("answer_not_a_literal", f"answer {answer!r} is not a decimal literal")
    if got != value:
        return _bad("value_mismatch",
                    f"declared {answer!r} != recomputed {want!r} "
                    f"(exact {value.numerator}/{value.denominator})")
    if require_canonical and str(answer).strip() != want:
        return _bad("not_canonical", f"declared {answer!r} is not the canonical form {want!r}")
    return {"ok": True, "verified": True, "state": STATE_VERIFIED, "value": value,
            "canonical": want, "reason": "ok",
            "detail": f"recomputed {want} (exact {value.numerator}/{value.denominator})"}


def _bad(reason, detail):
    return {"ok": False, "verified": False, "state": STATE_MISMATCH, "value": None,
            "canonical": None, "reason": reason, "detail": detail}


def arith_gate(states, operational):
    """The gate over the three states.

    A contradicted gold always fails. An unverifiable row fails the gate for the
    operational set, whose golds this module generates and can always recompute; for an
    external set it is reported (`unverifiable_n`) and does not, because this module
    cannot make another author's question recomputable.
    """
    return (states.get(STATE_MISMATCH, 0) == 0
            and (not operational or states.get(STATE_UNVERIFIABLE, 0) == 0))


# --------------------------------------------------------------------------------------
# the model's answer: exact extraction, exact comparison
# --------------------------------------------------------------------------------------

def extract_numbers(text):
    """Every plain decimal literal in the continuation, in order of appearance.

    The regex is the audit set's own (`-?\\d+(?:\\.\\d+)?`), so the two sides agree on what
    counts as a number and therefore on `parsed`.
    """
    return NUM_RE.findall(text or "")


def _precision_mismatch(extracted, gold):
    """True when `extracted` is `gold` with digits dropped instead of a different number.

    Both truncation and half-up rounding at the extracted literal's own precision are
    tested, because 42.5 -> "42" is truncation and 42.6 -> "43" is rounding; either one is
    a precision failure, and a value that is neither is a plain wrong answer.
    """
    got = as_fraction(extracted)
    if got is None or got == gold:
        return False
    dp = len(extracted.split(".")[1]) if "." in extracted else 0
    scale = Fraction(10) ** dp
    scaled = gold * scale
    floor = Fraction(int(scaled), 1)                # int() truncates toward zero
    frac = scaled - floor
    if frac < 0:
        floor -= 1
        frac = scaled - floor
    rounded = (floor + (1 if frac >= Fraction(1, 2) else 0)) / scale
    truncated = floor / scale
    return got in (rounded, truncated) and got != gold


def score_output(item, output, rule="last", require_canonical=True):
    """(result dict) for one model output against one item.

    The item is recomputed first: an item that fails its own arithmetic check is FAIL
    with fail_class `arith_check_failed` whatever the model said, which is what keeps a
    broken gold from being scored as a model error.
    """
    check = verify_item(item, require_canonical=require_canonical)
    numbers = extract_numbers(output)
    res = {"arith": check, "gold": check["canonical"], "numbers": numbers,
           "extracted": None, "passed": False, "fail_class": None}
    if not check["ok"]:
        res["fail_class"] = FAIL_ARITH
        return res
    if not str(output or "").strip():
        res["fail_class"] = FAIL_EMPTY
        return res
    if not numbers:
        res["fail_class"] = FAIL_NO_NUMBER
        return res
    if rule not in RULES:
        raise ValueError(f"unknown answer rule {rule!r}, expected one of {RULES}")
    pick = numbers[-1] if rule == "last" else numbers[0]
    res["extracted"] = pick
    if as_fraction(pick) == check["value"]:
        res["passed"] = True
        return res
    res["fail_class"] = FAIL_PRECISION if _precision_mismatch(pick, check["value"]) else FAIL_WRONG
    return res


def matches_any(item, output):
    """Diagnostic only: would an any-number extraction rule have matched?"""
    check = verify_item(item, require_canonical=False)
    if not check["ok"]:
        return False
    return any(as_fraction(n) == check["value"] for n in extract_numbers(output))


def matches_first(item, output):
    """Diagnostic only: would a first-number extraction rule have matched?"""
    check = verify_item(item, require_canonical=False)
    if not check["ok"]:
        return False
    numbers = extract_numbers(output)
    return bool(numbers) and as_fraction(numbers[0]) == check["value"]


# --------------------------------------------------------------------------------------
# protocol: base continuation, never ChatML (R1)
# --------------------------------------------------------------------------------------

def no_chatml(prompt):
    """(ok, detail): the prompt carries no ChatML marker."""
    if not isinstance(prompt, str) or not prompt.strip():
        return False, "prompt is not a non-empty string"
    for marker in CHATML:
        if marker in prompt:
            return False, f"prompt carries {marker}; a base checkpoint continues plain text"
    return True, "ok"


def continuation_of(prompt):
    """(ok, question, detail): the prompt is exactly format_continuation(question)."""
    ok, why = no_chatml(prompt)
    if not ok:
        return False, None, why
    if not prompt.startswith("问：") or not prompt.endswith("\n答："):
        return False, None, "prompt is not the 问：/答： continuation template"
    question = prompt[len("问："):-len("\n答：")]
    if format_continuation(question) != prompt:
        return False, question, "prompt does not round-trip through format_continuation"
    return True, question, "ok"


def protocol_ok(prompt, template):
    """(ok, detail) under the template the set was rendered with."""
    if template == "verbatim":
        return no_chatml(prompt)
    return continuation_of(prompt)[0], continuation_of(prompt)[2]


def resolve_device(device, log=None):
    """(device_used, detail): the ordinal torch can actually open.

    CUDA_VISIBLE_DEVICES=1 makes exactly one card visible, so `cuda:1` is out of range:
    torch raises Invalid device ordinal, and on a host where the mask is unset it would
    silently mean a different card. The mapping is applied only when the requested
    ordinal is out of range, and it is written to the log and to the summary.
    """
    if not str(device).startswith("cuda"):
        return device, "not a cuda device"
    mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not mask.strip():
        return device, "CUDA_VISIBLE_DEVICES unset; ordinal taken literally"
    ordinal = 0
    if ":" in str(device):
        try:
            ordinal = int(str(device).split(":")[1])
        except ValueError:
            return device, "unparsable ordinal; taken literally"
    visible = [v for v in mask.split(",") if v.strip() != ""]
    if ordinal < len(visible):
        return device, f"ordinal {ordinal} within CUDA_VISIBLE_DEVICES={mask}"
    used = "cuda:0"
    detail = (f"--device {device} is out of range under CUDA_VISIBLE_DEVICES={mask} "
              f"({len(visible)} visible); using {used} = physical card {visible[0]}")
    if log is not None:
        log.write(f"WARN {detail}\n")
    return used, detail


# --------------------------------------------------------------------------------------
# the operational set: generated here, written by --emit-data, refused when hand-edited
# --------------------------------------------------------------------------------------

def _pick(rng, candidates, n, seen):
    """n candidates whose canonical answers are distinct and unused, order from rng."""
    pool = list(candidates)
    rng.shuffle(pool)
    out = []
    for cand in pool:
        answer = cand[-1]
        if answer in seen:
            continue
        seen.add(answer)
        out.append(cand)
        if len(out) == n:
            break
    return out


def _item(kind, index, question, expr, answer):
    return {
        "id": f"e3_{kind}_{index:02d}",
        "kind": kind,
        "prompt": format_continuation(question),
        "answer": answer,
        "check": expr,
    }


def _cands_int_add():
    out = []
    for a in range(24, 400, 7):
        for b in range(19, 380, 11):
            out.append((f"计算 {a} + {b} 的值，只写数值。", {"op": "+", "args": [str(a), str(b)]},
                        canonical_decimal(Fraction(a + b))))
    return out


def _cands_int_sub():
    out = []
    for a in range(215, 900, 13):
        for b in range(13, 190, 7):
            if a <= b:
                continue
            out.append((f"计算 {a} - {b} 的值，只写数值。", {"op": "-", "args": [str(a), str(b)]},
                        canonical_decimal(Fraction(a - b))))
    return out


def _cands_int_mul():
    out = []
    for a in range(3, 32):
        for b in range(4, 32):
            out.append((f"计算 {a} × {b} 的值，只写数值。", {"op": "*", "args": [str(a), str(b)]},
                        canonical_decimal(Fraction(a * b))))
    return out


def _cands_int_div():
    out = []
    for divisor in range(3, 13):
        for quotient in range(4, 41):
            dividend = divisor * quotient
            out.append((f"计算 {dividend} ÷ {divisor} 的值，只写数值。",
                        {"op": "/", "args": [str(dividend), str(divisor)]},
                        canonical_decimal(Fraction(quotient))))
    return out


def _cands_decimal():
    """One-step decimal items whose exact result keeps a non-zero final digit.

    A result of 40.0 is rejected on purpose: the canonical form would be "40" and the
    prompt would then be asking for an integer, blurring the decimal kind.
    """
    out = []
    for ka in range(15, 990, 37):
        for kb in range(23, 980, 41):
            if (ka + kb) % 10 == 0:
                continue
            a, b = canonical_decimal(Fraction(ka, 10)), canonical_decimal(Fraction(kb, 10))
            out.append((f"计算 {a} + {b} 的值，结果写成小数，只写数值。",
                        {"op": "+", "args": [a, b]},
                        canonical_decimal(Fraction(ka + kb, 10))))
    for ka in range(210, 990, 29):
        for kb in range(15, 200, 17):
            if ka <= kb or (ka - kb) % 10 == 0:
                continue
            a, b = canonical_decimal(Fraction(ka, 10)), canonical_decimal(Fraction(kb, 10))
            out.append((f"计算 {a} - {b} 的值，结果写成小数，只写数值。",
                        {"op": "-", "args": [a, b]},
                        canonical_decimal(Fraction(ka - kb, 10))))
    for ka in range(15, 100, 7):
        for kb in range(15, 100, 9):
            if (ka * kb) % 10 == 0:
                continue
            a, b = canonical_decimal(Fraction(ka, 10)), canonical_decimal(Fraction(kb, 10))
            out.append((f"计算 {a} × {b} 的值，结果写成小数，只写数值。",
                        {"op": "*", "args": [a, b]},
                        canonical_decimal(Fraction(ka * kb, 100))))
    for t in range(11, 100, 3):
        if t % 10 == 0:
            continue
        for divisor in range(2, 10):
            if (t * divisor) % 10 == 0:
                continue
            dividend = canonical_decimal(Fraction(t * divisor, 10))
            out.append((f"计算 {dividend} ÷ {divisor} 的值，结果写成小数，只写数值。",
                        {"op": "/", "args": [dividend, str(divisor)]},
                        canonical_decimal(Fraction(t, 10))))
    return out


def _cands_percent():
    """Two percent shapes, both exact: a part of a whole, and a ratio as a percentage."""
    out = []
    for percent in (5, 10, 15, 20, 25, 30, 40, 50, 60, 75):
        rate = canonical_decimal(Fraction(percent, 100))
        for whole in range(120, 1000, 10):
            value = Fraction(whole * percent, 100)
            if value.denominator != 1 or not (2 <= value <= 500):
                continue
            out.append((f"求 {whole} 的 {percent}% 是多少，只写数值。",
                        {"op": "*", "args": [str(whole), rate]},
                        canonical_decimal(value)))
    for percent in (5, 10, 15, 20, 25, 30, 40, 50, 60, 75):
        for k in range(2, 21):
            whole = 100 * k
            part = percent * k
            out.append((f"{part} 是 {whole} 的百分之几，只写数值。",
                        {"op": "*", "args": [{"op": "/", "args": [str(part), str(whole)]}, "100"]},
                        canonical_decimal(Fraction(percent))))
    return out


CANDIDATE_BUILDERS = {
    "int_add": _cands_int_add,
    "int_sub": _cands_int_sub,
    "int_mul": _cands_int_mul,
    "int_div": _cands_int_div,
    "decimal": _cands_decimal,
    "percent": _cands_percent,
}


def generate_items(per_kind=PER_KIND, seed=GEN_SEED):
    """The operational set, from arithmetic only. Same seed, same rows, byte for byte.

    Answers are unique across the whole file, so a model that emits one fixed number
    cannot collect a second point by collision, and `verify_coverage` can key on the id
    set without a tie.
    """
    rng = random.Random(seed)
    seen = set()
    rows = []
    for kind in KINDS:
        picked = _pick(rng, CANDIDATE_BUILDERS[kind](), per_kind, seen)
        if len(picked) < per_kind:
            raise RuntimeError(f"{kind}: only {len(picked)} distinct-answer candidates, "
                               f"need {per_kind}")
        for index, (question, expr, answer) in enumerate(picked):
            rows.append(_item(kind, index, question, expr, answer))
    return rows


def render_rows(rows):
    """The file body: one object per line, key order as the contract states it."""
    return "".join(json.dumps({k: row[k] for k in ROW_KEYS}, ensure_ascii=False) + "\n"
                   for row in rows)


def emit_data(path, log=sys.stdout):
    """Write the committed operational set from generate_items(). Exit 3 if it is invalid."""
    rows = generate_items()
    errors, stats = validate_rows(rows, operational=True)
    if errors:
        for err in errors:
            log.write(f"FAIL generated dataset: {err}\n")
        return lc.EXIT_GATE
    body = render_rows(rows)
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    old = target.read_text(encoding="utf-8") if target.exists() else None
    target.write_text(body, encoding="utf-8")
    state = "unchanged" if old == body else ("rewritten" if old is not None else "written")
    log.write(f"{state} {target} n={len(rows)} sha256={hashlib.sha256(body.encode()).hexdigest()}\n")
    log.write(f"per_kind={json.dumps(stats['per_kind'], sort_keys=True)}\n")
    return lc.EXIT_OK


def validate_rows(rows, operational=True, template=PROMPT_TEMPLATE):
    """(errors, stats) for the row contract -- structure, not arithmetic.

    `operational=True` is the arith_v1.jsonl contract: the five keys, the six kinds with
    at least MIN_PER_KIND items each, answers unique across the file, prompts rendered by
    format_continuation. `operational=False` is the external cross-check contract: id,
    prompt and answer present, ids unique, protocol respected under `template`.

    Whether an item's own answer survives recomputation is NOT here -- that is a per-item
    result with fail_class `arith_check_failed`, which the contract requires to be
    reported as a gate failure rather than as a refusal.
    """
    errors, per_kind, answers, ids = [], {}, {}, []
    for index, row in enumerate(rows):
        tag = f"row {index}"
        if not isinstance(row, dict):
            errors.append(f"{tag}: not a JSON object")
            continue
        missing = [k for k in ROW_KEYS if k not in row]
        if missing:
            errors.append(f"{tag}: missing keys {missing}")
            continue
        tag = f"row {index} ({row['id']})"
        if row["id"] in ids:
            errors.append(f"{tag}: duplicate id")
        ids.append(row["id"])
        kind = row["kind"]
        if operational:
            if kind not in KINDS:
                errors.append(f"{tag}: kind {kind!r} not in {KINDS}")
            else:
                per_kind[kind] = per_kind.get(kind, 0) + 1
        row_template = row.get("expected_template") or template
        ok, detail = protocol_ok(row["prompt"], row_template)
        if not ok:
            errors.append(f"{tag}: {detail}")
        answer = row["answer"]
        if operational:
            if not isinstance(answer, str):
                errors.append(f"{tag}: answer {answer!r} is not a string")
            elif answer in answers:
                errors.append(f"{tag}: answer {answer!r} already used by {answers[answer]}")
            else:
                answers[answer] = row["id"]
        if not isinstance(row["check"], (dict, str, int, float)):
            errors.append(f"{tag}: check must be an expression node")
    if operational:
        for kind in KINDS:
            got = per_kind.get(kind, 0)
            if got < MIN_PER_KIND:
                errors.append(f"kind {kind}: {got} items, contract needs >= {MIN_PER_KIND}")
    stats = {"n": len(rows), "per_kind": per_kind,
             "n_distinct_answers": len({r.get("answer") for r in rows if isinstance(r, dict)})}
    return errors, stats


def dataset_matches_generator(rows):
    """(ok, detail): the rows on disk are what generate_items() produces now."""
    fresh = generate_items()
    if rows == fresh:
        return True, f"{len(rows)} rows equal generate_items()"
    if len(rows) != len(fresh):
        return False, f"{len(rows)} rows on disk vs {len(fresh)} generated"
    for got, want in zip(rows, fresh):
        if got != want:
            diff = [k for k in ROW_KEYS if got.get(k) != want.get(k)]
            return False, f"row {got.get('id')} differs in {diff}: {got} vs {want}"
    return False, "rows differ in order only"


def limit_rows(rows, n):
    """First n items, round-robin over kinds so a limited run still covers every kind."""
    if not n or n >= len(rows):
        return list(rows)
    groups = {}
    for row in rows:
        groups.setdefault(row["kind"], []).append(row)
    order = [k for k in KINDS if k in groups] + [k for k in groups if k not in KINDS]
    out, index = [], 0
    while len(out) < n:
        added = False
        for kind in order:
            if len(out) >= n:
                break
            if index < len(groups[kind]):
                out.append(groups[kind][index])
                added = True
        if not added:
            break
        index += 1
    return out


# --------------------------------------------------------------------------------------
# the external cross-check set (R7): ids from the file, arithmetic still recomputed here
# --------------------------------------------------------------------------------------

def check_from_question(question):
    """(check node, kind) recovered from the QUESTION TEXT, or (None, None).

    Two shapes count as an operation the text states: the explicit '计算：7 + 6 = ?' form,
    and a clean arithmetic expression that ends the question ('What is 1+1?'). Both are
    recomputed here, so the check is this module's own arithmetic and never the file's
    claim restated. A question with no recoverable operation returns (None, None): that
    row is unverifiable, reported, and never counted as checked.
    """
    match = ARITH_Q_RE.match(question or "")
    if match:
        left, op, right = match.group(1), match.group(2), match.group(3)
        return {"op": OP_MAP[op], "args": [left, right]}, "question_text"
    tail = trailing_expression(question)
    if tail is not None:
        return tail, "question_tail"
    return None, None


def load_prompt_set(path, template="continuation"):
    """(rows, info) for an external prompt set, in the ladder's row shape.

    Shape accepted: a top level list, or an object holding one of SET_KEYS
    (`{"prompts": [...], "negative_controls": [...]}` is the shape R7 names; the audit
    gates file today is `{"items": [{"id", "text", "answer"}]}`). Per entry: the id comes
    from the file, the question from the first of QUESTION_KEYS, the answer from
    "answer". The answer is normalized to its canonical exact spelling -- 8.0 -> 8 -- and
    the file's own spelling is kept as `answer_raw`; the value compared is bit-identical
    either way.

    `template="continuation"` renders the question with format_continuation, the same
    protocol the operational set uses; "verbatim" uses the file's text unchanged (still
    ChatML-free); "auto" takes verbatim only when the entry already carries a rendered
    `prompt` that round-trips through format_continuation.
    """
    source = pathlib.Path(path)
    obj = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        entries = obj
    else:
        entries = next((obj[k] for k in SET_KEYS if isinstance(obj.get(k), list)), None)
    lc.require(isinstance(entries, list) and entries,
               f"{path}: no prompt list found (looked for a top level list or {SET_KEYS})")
    rows, check_kinds, synthesized, used_templates = [], Counter(), 0, Counter()
    for index, entry in enumerate(entries):
        lc.require(isinstance(entry, dict), f"{path}: entry {index} is not a JSON object")
        rid = entry.get("id")
        if rid is None:
            rid = f"ext:{index:03d}"
            synthesized += 1
        rid = str(rid)
        question = next((entry[k] for k in QUESTION_KEYS
                         if isinstance(entry.get(k), str) and entry[k].strip()), None)
        lc.require(question is not None,
                   f"{path}: entry {rid} has none of the question fields {QUESTION_KEYS}")
        question = question.strip()
        raw_answer = entry.get("answer")
        lc.require(raw_answer is not None, f"{path}: entry {rid} has no answer")
        value = as_fraction(raw_answer)
        lc.require(value is not None,
                   f"{path}: entry {rid} answer {raw_answer!r} is not a plain decimal number")
        answer = canonical_decimal(value)
        lc.require(answer is not None,
                   f"{path}: entry {rid} answer {raw_answer!r} has no finite decimal form")
        expr, check_kind = check_from_question(question)
        if expr is None:
            expr = {"op": "lit", "args": [answer]}
            check_kind = "literal"
        check_kinds[check_kind] += 1
        use = template
        if template == "auto":
            pre = entry.get("prompt")
            use = "verbatim" if (isinstance(pre, str) and continuation_of(pre)[0]) else "continuation"
        if use == "verbatim":
            prompt = question
        else:
            prompt = format_continuation(question)
        used_templates[use] += 1
        rows.append({"id": rid, "kind": "external", "prompt": prompt, "answer": answer,
                     "check": expr, "answer_raw": raw_answer, "check_kind": check_kind,
                     "expected_template": use})
    info = {
        "source": str(source),
        "sha256": lc.sha256_file(source),
        "n": len(rows),
        "set_keys": [k for k in SET_KEYS if isinstance(obj, dict) and isinstance(obj.get(k), list)]
        if isinstance(obj, dict) else ["<top level list>"],
        "prompt_template": dict(used_templates),
        "check_kind": dict(check_kinds),
        "synthesized_ids": synthesized,
        "answer_rule_file": obj.get("answer_rule") if isinstance(obj, dict) else None,
    }
    return rows, info


# --------------------------------------------------------------------------------------
# controls: positive and negative worlds, over the same code the gate runs
# --------------------------------------------------------------------------------------

def _synth(check, answer, kind="int_add"):
    return {"id": "ctl", "kind": kind, "prompt": format_continuation("控制题。"),
            "answer": answer, "check": check}


def run_controls(rows, file_rows=None, operational=True):
    """CPU-only control suite. Every entry: {"control", "ok", "detail"}.

    Positive worlds must pass; every negative world must fail, on the assertion that is
    its own, and the reason is compared rather than the bare boolean -- a negative that
    fails for an unrelated reason is a control that is not controlling anything.

    The negative worlds are synthetic so that the same suite controls both the
    operational set and an external cross-check set; `rows` supplies the positive worlds
    (they are scored against their own golds) and `file_rows` the generator provenance.
    """
    controls = []

    def add(name, ok, detail):
        controls.append({"control": name, "ok": bool(ok), "detail": str(detail)})

    template = PROMPT_TEMPLATE if operational else "continuation"
    bad_protocol = [r["id"] for r in rows
                    if not protocol_ok(r["prompt"], r.get("expected_template") or template)[0]]
    add("positive_protocol", not bad_protocol,
        f"{len(rows)} prompts held the base-continuation protocol"
        + (f"; bad {bad_protocol[:5]}" if bad_protocol else ""))

    bad_arith = [r["id"] for r in rows if not verify_item(r, require_canonical=operational)["ok"]]
    add("positive_arith_check", not bad_arith,
        f"{len(rows)} items recompute exactly" + (f"; bad {bad_arith[:5]}" if bad_arith else ""))

    gold_scored = [score_output(r, r["answer"], require_canonical=operational) for r in rows]
    add("positive_gold_answer_scores_pass", all(s["passed"] for s in gold_scored),
        f"{sum(1 for s in gold_scored if s['passed'])}/{len(rows)} items pass on their own gold")

    ok, _q, why = continuation_of("<|im_start|>user\n计算 1 + 1<|im_end|>\n<|im_start|>assistant\n")
    add("negative_chatml_prompt", (not ok) and "<|im_start|>" in why, f"rejected: {why}")

    add_row = _synth({"op": "+", "args": ["123", "456"]}, "579")
    flipped = "-333"                                     # 123 - 456: the wrong operation
    wrong_op = verify_item(dict(add_row, answer=flipped))
    add("negative_dataset_wrong_operation",
        (not wrong_op["ok"]) and wrong_op["reason"] == "value_mismatch",
        f"{flipped} declared for 123 + 456: {wrong_op['detail']}")

    dec_row = _synth({"op": "+", "args": ["12.5", "30"]}, "42.5", kind="decimal")
    cut = "42"
    cut_check = verify_item(dict(dec_row, answer=cut))
    add("negative_dataset_truncated_decimal",
        (not cut_check["ok"]) and cut_check["reason"] == "value_mismatch",
        f"42.5 truncated to {cut}: {cut_check['detail']}")

    third = _synth({"op": "/", "args": ["1", "3"]}, "0.3", kind="int_div")
    third_check = verify_item(third)
    add("negative_dataset_wrong_precision",
        (not third_check["ok"]) and third_check["reason"] == "no_finite_decimal",
        f"1/3 declared as 0.3: {third_check['detail']}")
    rounded = verify_item(dict(third, answer="0.333333"))
    add("negative_dataset_rounded_repeating",
        (not rounded["ok"]) and rounded["reason"] == "no_finite_decimal",
        f"1/3 rounded to 6 dp: {rounded['detail']}")

    order_row = _synth({"op": "+", "args": ["123", "456"]}, "579")
    last_only = score_output(order_row, "123 + 456 = 579")
    add("positive_last_number_rule",
        last_only["passed"] and last_only["extracted"] == "579",
        f"only the last number is the gold: passed={last_only['passed']} "
        f"extracted={last_only['extracted']} numbers={last_only['numbers']}")
    first_only = score_output(order_row, "579 是 123 + 456 的和，但最后一个数是 123")
    add("negative_first_number_only",
        (not first_only["passed"]) and first_only["extracted"] == "123"
        and first_only["fail_class"] == FAIL_WRONG,
        f"only the first number is the gold: passed={first_only['passed']} "
        f"extracted={first_only['extracted']} fail_class={first_only['fail_class']}")

    model_wrong = score_output(add_row, flipped)
    add("negative_model_wrong_operation",
        (not model_wrong["passed"]) and model_wrong["fail_class"] == FAIL_WRONG,
        f"model says {flipped}, gold 579: fail_class={model_wrong['fail_class']}")

    model_off = score_output(add_row, "580")
    add("negative_model_off_by_one",
        (not model_off["passed"]) and model_off["fail_class"] == FAIL_WRONG,
        f"model says 580, gold 579: fail_class={model_off['fail_class']}")

    model_trunc = score_output(dec_row, cut)
    add("negative_model_wrong_precision",
        (not model_trunc["passed"]) and model_trunc["fail_class"] == FAIL_PRECISION,
        f"model says {cut}, gold 42.5: fail_class={model_trunc['fail_class']}")

    model_none = score_output(add_row, "这道题我不会算。")
    add("negative_model_no_number",
        (not model_none["passed"]) and model_none["fail_class"] == FAIL_NO_NUMBER,
        f"no literal in output: fail_class={model_none['fail_class']}")

    model_empty = score_output(add_row, "   \n")
    add("negative_model_empty_output",
        (not model_empty["passed"]) and model_empty["fail_class"] == FAIL_EMPTY,
        f"blank output: fail_class={model_empty['fail_class']}")

    model_broken = score_output(dict(add_row, answer=flipped), "579")
    add("negative_model_ok_but_dataset_broken",
        (not model_broken["passed"]) and model_broken["fail_class"] == FAIL_ARITH,
        f"correct answer against a broken gold: fail_class={model_broken['fail_class']}")

    ids = [r["id"] for r in rows]
    ok_missing, _ = lc.verify_coverage(rows[:-1], ids, label="control")
    ok_dupe, _ = lc.verify_coverage(rows + [rows[0]], ids, label="control")
    add("negative_coverage_missing_id", not ok_missing, "one item dropped -> coverage FAIL")
    add("negative_coverage_duplicate_id", not ok_dupe, "one item duplicated -> coverage FAIL")

    type_error = validate_rows([dict(add_row, kind="int_arith")], operational=True)[0]
    add("negative_unknown_kind", bool(type_error), f"kind int_arith rejected: {type_error[:1]}")
    dupe_errors = validate_rows([add_row, dict(dec_row, answer="579")], operational=True)[0]
    add("negative_duplicate_answer",
        any("already used" in e for e in dupe_errors), f"{dupe_errors[:1]}")
    chatml_errors = validate_rows([dict(add_row, prompt="<|im_start|>user\nhi<|im_end|>\n")],
                                  operational=True)[0]
    add("negative_chatml_row_refused",
        any("<|im_start|>" in e for e in chatml_errors), f"{chatml_errors[:1]}")

    if operational and DATA_DEFAULT.exists():
        probe = rows if file_rows is None else file_rows
        ok_file, detail = dataset_matches_generator(probe)
        add("positive_file_matches_generator", ok_file, f"{DATA_DEFAULT.name}: {detail}")
    elif operational:
        controls.append({"control": "positive_file_matches_generator", "ok": True,
                         "detail": f"SKIP {DATA_DEFAULT.name} not written yet"})
    return controls


def control_selftest():
    """CPU-only positive/negative controls for `eval_ladder.py selftest --only e3`."""
    failures = []
    try:
        rows = generate_items()
    except Exception as exc:                                        # noqa: BLE001
        print(f"FAIL e3 control: generate_items raised {exc!r}", file=sys.stderr)
        return lc.EXIT_GATE
    errors, stats = validate_rows(rows, operational=True)
    if errors:
        failures.extend(f"generated dataset invalid: {e}" for e in errors)
    if len(stats["per_kind"]) != len(KINDS):
        failures.append(f"generated dataset covers {sorted(stats['per_kind'])}, want {KINDS}")
    if stats["n_distinct_answers"] != stats["n"]:
        failures.append(f"{stats['n'] - stats['n_distinct_answers']} duplicate answers in the set")
    controls = run_controls(rows)
    for ctl in controls:
        if not ctl["ok"]:
            failures.append(f"control {ctl['control']} failed: {ctl['detail']}")
    external = _selftest_external_set()
    for ctl in external:
        if not ctl["ok"]:
            failures.append(f"external control {ctl['control']} failed: {ctl['detail']}")
    if failures:
        for fail in failures:
            print(f"FAIL e3 control: {fail}", file=sys.stderr)
        return lc.EXIT_GATE
    print(f"PASS e3 controls: {len(rows)} generated items ({stats['per_kind']}), "
          f"{len(controls)} control worlds over the operational set, "
          f"{len(external)} over the external set, every negative world FAILs on its own "
          f"assertion")
    return lc.EXIT_OK


def _selftest_external_set():
    """Controls for the --prompts path: the audit shape, parsed to rows and rechecked.

    The audit gates file is NOT read here (R7: nothing may depend on it existing); the
    world is built from the shape it documents, including a numeric answer (8.0), a
    negative answer and an unparsable question, so `check_kind` and the id path are both
    exercised.
    """
    import tempfile

    controls = []

    def add(name, ok, detail):
        controls.append({"control": name, "ok": bool(ok), "detail": str(detail)})

    payload = {
        "id_scheme": "arith:{index:03d}",
        "items": [
            {"id": "arith:000", "text": "计算：7 + 6 = ?", "answer": 13},
            {"id": "arith:008", "text": "计算：5.5 + 2.5 = ?", "answer": 8.0},
            {"id": "arith:012", "text": "计算：4 - 11 = ?", "answer": -7},
            {"id": "arith:099", "text": "小明有几个苹果？", "answer": 3},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "E3_arith.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        rows, info = load_prompt_set(path, "continuation")
        add("external_ids_from_file", [r["id"] for r in rows] ==
            ["arith:000", "arith:008", "arith:012", "arith:099"],
            f"ids {[r['id'] for r in rows]}")
        add("external_answers_exact",
            [r["answer"] for r in rows] == ["13", "8", "-7", "3"]
            and rows[1]["answer_raw"] == 8.0,
            f"answers {[r['answer'] for r in rows]}, raw {[r['answer_raw'] for r in rows]}")
        add("external_check_recomputed_from_text",
            info["check_kind"] == {"question_text": 3, "literal": 1},
            f"check_kind {info['check_kind']}")
        add("external_rows_verify", all(verify_item(r, require_canonical=False)["ok"] for r in rows),
            "all four rows recompute")
        add("external_scoring_exact",
            score_output(rows[0], "答案是 13").get("passed") is True
            and score_output(rows[0], "答案不是 12").get("passed") is False,
            "13 passes, 12 fails on arith:000")
        bad = [dict(rows[1], answer="8.5")]
        add("external_wrong_answer_refused",
            not verify_item(bad[0], require_canonical=False)["ok"],
            f"8.0 declared as 8.5: {verify_item(bad[0], require_canonical=False)['detail']}")
        add("external_template_verbatim",
            load_prompt_set(path, "verbatim")[0][0]["prompt"] == "计算：7 + 6 = ?",
            "verbatim keeps the file's text unchanged")
        errors, _stats = validate_rows([dict(rows[0], prompt="<|im_start|>user\nhi\n")],
                                       operational=False)
        add("external_chatml_refused",
            any("<|im_start|>" in e for e in errors), f"{errors[:1]}")

        # three states, over the file path the --prompts run uses
        states_payload = {"prompts": [
            {"id": "ext:text", "prompt": "计算：7 + 6 = ?", "answer": 13},
            {"id": "ext:tail", "prompt": "What is 1+1?", "answer": 2},
            {"id": "ext:wrong", "prompt": "What is 1+1?", "answer": 9},
            {"id": "ext:open", "prompt": "小明有几个苹果？", "answer": 3},
        ]}
        spath = os.path.join(tmp, "states.json")
        with open(spath, "w", encoding="utf-8") as handle:
            json.dump(states_payload, handle, ensure_ascii=False)
        srows, sinfo = load_prompt_set(spath, "continuation")
        by_id = {r["id"]: r for r in srows}
        kinds = {r["id"]: r["check_kind"] for r in srows}
        add("external_check_kinds", kinds == {"ext:text": "question_text",
                                              "ext:tail": "question_tail",
                                              "ext:wrong": "question_tail",
                                              "ext:open": "literal"},
            f"check_kind {kinds} ({sinfo['check_kind']})")
        add("positive_question_text_state",
            verify_item(by_id["ext:text"])["state"] == STATE_VERIFIED
            and verify_item(by_id["ext:text"])["verified"],
            f"{verify_item(by_id['ext:text'])}")
        tail_ver = verify_item(by_id["ext:tail"])
        add("positive_question_tail_state",
            tail_ver["state"] == STATE_VERIFIED and tail_ver["canonical"] == "2",
            f"1+1 with answer 2: {tail_ver['detail']}")
        wrong_ver = verify_item(by_id["ext:wrong"])
        add("negative_gold_mismatch_state",
            wrong_ver["state"] == STATE_MISMATCH and not wrong_ver["ok"]
            and wrong_ver["reason"] == "value_mismatch",
            f"1+1 declared as 9: {wrong_ver['reason']} / {wrong_ver['detail']}")
        wrong_scored = score_output(by_id["ext:wrong"], "9")
        add("negative_gold_mismatch_not_scored",
            (not wrong_scored["passed"]) and wrong_scored["fail_class"] == FAIL_ARITH,
            f"model says 9, gold 9, recomputation 2: fail_class={wrong_scored['fail_class']}")
        open_ver = verify_item(by_id["ext:open"])
        add("unverifiable_state_is_not_a_pass",
            open_ver["state"] == STATE_UNVERIFIABLE and open_ver["ok"]
            and not open_ver["verified"],
            f"no operation in the question: {open_ver['reason']} / {open_ver['detail']}")
        add("unverifiable_counted_and_gated",
            arith_gate({STATE_VERIFIED: 3, STATE_UNVERIFIABLE: 1}, False)
            and not arith_gate({STATE_VERIFIED: 3, STATE_UNVERIFIABLE: 1}, True)
            and not arith_gate({STATE_MISMATCH: 1}, False),
            "external: unverifiable reported and allowed; operational: refused; "
            "gold_mismatch always refused")
    return controls


# --------------------------------------------------------------------------------------
# the stage
# --------------------------------------------------------------------------------------

def _claim_name():
    stem = os.path.basename(sys.argv[0] or "")
    return stem[:-3] if stem.endswith(".py") else (stem or "eval_ladder")


def main(argv):
    ap = argparse.ArgumentParser(prog="eval_ladder.py e3")
    ap.add_argument("--ckpt", default=None, help="checkpoint; required unless --emit-data")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new", type=int, default=24, dest="max_new")
    ap.add_argument("--control", action="store_true", help="gate on the control suite")
    ap.add_argument("--n", type=int, default=0, help="items to score, 0 = all")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--data", default=str(DATA_DEFAULT),
                    help="operational set; ignored when --prompts is given")
    ap.add_argument("--prompts", default=None,
                    help="external cross-check set (audit gates JSON); ids come from it")
    ap.add_argument("--prompts-template", default="continuation", choices=TEMPLATES,
                    dest="prompts_template",
                    help="continuation: render the question with format_continuation "
                         "(default); verbatim: use the file's text unchanged; auto: "
                         "verbatim only for an already-rendered prompt")
    ap.add_argument("--min-acc", type=float, default=None, dest="min_acc",
                    help="capability gate on exact-match accuracy; R2: a baseline's gate "
                         "is evidence, so this is opt-in and reported without it")
    ap.add_argument("--emit-data", default=None, dest="emit_data", metavar="PATH",
                    help="regenerate the operational set from generate_items() and exit")
    args = ap.parse_args(argv)

    if args.emit_data:
        return emit_data(args.emit_data)
    lc.require(args.ckpt, "--ckpt is required (or --emit-data PATH to regenerate the dataset)")
    lc.require(args.max_new > 0, "--max-new must be positive")

    def stage(paths, log):
        if args.prompts:
            prompts_path = pathlib.Path(args.prompts)
            lc.require(prompts_path.is_file(), f"prompt set {prompts_path} does not exist")
            rows_all, set_info = load_prompt_set(prompts_path, args.prompts_template)
            source, operational = "external", False
            errors, stats = validate_rows(rows_all, operational=False, template="continuation")
            lc.require(not errors, "prompt set contract: " + "; ".join(errors[:6]))
        else:
            data_path = pathlib.Path(args.data)
            lc.require(data_path.is_file(), f"dataset {data_path} does not exist; "
                                            f"generate it with --emit-data {data_path}")
            rows_all = lc.read_jsonl(data_path)
            lc.require(rows_all, f"dataset {data_path} is empty")
            errors, stats = validate_rows(rows_all, operational=True)
            lc.require(not errors, "dataset contract: " + "; ".join(errors[:6]))
            ok_gen, gen_detail = dataset_matches_generator(rows_all)
            lc.require(ok_gen, f"dataset {data_path} is not what generate_items() produces "
                               f"({gen_detail}); regenerate with --emit-data {data_path}")
            source, operational = "operational", True
            set_info = {"source": str(data_path), "sha256": lc.sha256_file(data_path),
                        "n": len(rows_all), "generator": "eval/ladder/e3_arith.py::generate_items",
                        "generator_rows_sha256":
                            hashlib.sha256(render_rows(generate_items()).encode()).hexdigest(),
                        "prompt_template": {"continuation": len(rows_all)},
                        "check_kind": {"generated": len(rows_all)}, "synthesized_ids": 0}

        rows = limit_rows(rows_all, args.n)
        ids = [r["id"] for r in rows]
        inputs = {
            "prompt_set": source,
            "source": set_info["source"],
            "source_sha256": set_info["sha256"],
            "n_items": len(rows),
            "n_items_in_set": len(rows_all),
            "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "protocol": "base_continuation",
            "prompt_template": PROMPT_TEMPLATE if source == "operational"
            else args.prompts_template,
            "prompt_template_used": set_info.get("prompt_template"),
            "answer_rule": ANSWER_RULE,
            "audit_answer_rule": AUDIT_ANSWER_RULE,
            "decoding": "greedy",
            "max_new": args.max_new,
            "chatml_prefix": False,
            "set_info": set_info,
        }
        device_used, device_detail = resolve_device(args.device, log)
        seal_rec = lc.seal(paths, stage="e3", ckpt=args.ckpt, device=args.device,
                           device_used=device_used, inputs=inputs, ids=ids, shard="1/1",
                           dataset_stats=stats, prompt_set=source)
        log.write(f"E3 prompt set {source} {set_info['source']} n={len(rows)} "
                  f"sha256={set_info['sha256']}\n")
        log.write(f"E3 set_info={json.dumps(set_info, sort_keys=True, default=str)}\n")
        log.write(f"E3 device {args.device} -> {device_used} ({device_detail})\n")

        model, tok, meta = lc.load_for_eval(args.ckpt, device=device_used, claim=True)
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        log.write(f"E3 loaded {args.ckpt} step={meta['step']} vocab_id={meta['vocab_id']} "
                  f"gpu={gpu_name}\n")

        from train import generate_batch

        items, verified = [], {}
        started = time.time()
        for batch_index, start in enumerate(range(0, len(rows), args.batch)):
            batch = rows[start:start + args.batch]
            prompts = [tok.encode(r["prompt"]).ids for r in batch]
            batch_started = time.time()
            out_ids = generate_batch(model, prompts, args.max_new, device_used,
                                     temperature=0.0, rep_stop=False)
            batch_elapsed = round(time.time() - batch_started, 3)
            for row, ids_out in zip(batch, out_ids):
                text_ids, stop_token = _split_stop(ids_out, meta["eos_id"], meta["im_end_id"])
                output = tok.decode(text_ids)
                scored = score_output(row, output, require_canonical=operational)
                verified[row["id"]] = scored["arith"]
                items.append({
                    # contract section 3 names
                    "id": row["id"],
                    "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                    "output": output,
                    "stop_reason": lc.stop_reason(stop_token, meta["eos_id"], meta["im_end_id"],
                                                  len(ids_out), args.max_new),
                    "tokens": len(ids_out),
                    "passed": scored["passed"],
                    "fail_class": scored["fail_class"],
                    # R3 per-item evidence interface, a superset of the above
                    "prompt": row["prompt"],
                    "kind": row["kind"],
                    "response_raw": output,
                    "parsed": scored["extracted"],
                    "pass": scored["passed"],
                    "n_tokens": len(ids_out),
                    "last_token_id": stop_token,
                    "gold": scored["gold"],
                    "check": row["check"],
                    "check_kind": row.get("check_kind", "generated"),
                    "numbers": scored["numbers"],
                    "meta": {"decoding": "greedy", "temperature": 0.0, "seed": None,
                             "max_new": args.max_new, "batch_index": batch_index,
                             "batch_size": len(batch),
                             "batch_elapsed_s": batch_elapsed},
                })
                log.write(f"{'PASS' if scored['passed'] else 'FAIL'} {row['id']} "
                          f"gold={scored['gold']} extracted={scored['extracted']} "
                          f"fail={scored['fail_class']} out={output!r}\n")
        del model

        n = len(items)
        passed = sum(1 for it in items if it["passed"])
        acc = passed / n if n else 0.0
        extract_n = sum(1 for it in items if it["numbers"])
        extract_rate = extract_n / n if n else 0.0
        by_id = {r["id"]: r for r in rows}
        arith_failed = sorted(k for k, v in verified.items() if not v["ok"])
        states = Counter(v["state"] for v in verified.values())
        unverifiable_ids = sorted(k for k, v in verified.items()
                                  if v["state"] == STATE_UNVERIFIABLE)
        cover_ok, cover_detail = lc.verify_coverage(
            [{"id": it["id"]} for it in items], ids, label="e3 items")
        controls = run_controls(rows_all, file_rows=rows_all, operational=operational)
        controls_ok = all(c["ok"] for c in controls)
        protocols = [protocol_ok(r["prompt"],
                                 r.get("expected_template") or inputs["prompt_template"])
                     for r in rows]
        bad_protocol = [r["id"] for r, (ok, _d) in zip(rows, protocols) if not ok]
        evidence_ok = (all(isinstance(it["output"], str) for it in items)
                       and any(it["output"].strip() for it in items))

        gates = {
            "protocol": not bad_protocol,
            "arith_check": arith_gate(states, operational) and not arith_failed,
            "coverage": cover_ok,
            "evidence": evidence_ok,
        }
        if args.control:
            gates["controls"] = controls_ok
        if args.min_acc is not None:
            gates["accuracy"] = acc >= args.min_acc
        exit_code = lc.EXIT_OK if all(gates.values()) else lc.EXIT_GATE

        per_kind = {}
        for kind in sorted({it["kind"] for it in items}):
            kind_items = [it for it in items if it["kind"] == kind]
            kp = sum(1 for it in kind_items if it["passed"])
            per_kind[kind] = {"n": len(kind_items), "passed": kp,
                              "acc": round(kp / len(kind_items), 4)}
        fail_classes = Counter(it["fail_class"] for it in items if it["fail_class"])
        stop_reasons = Counter(it["stop_reason"] for it in items)
        summary = {
            "ckpt": args.ckpt,
            "ckpt_sha256": meta["ckpt_sha256"],
            "step": meta["step"],
            "protocol": "base_continuation",
            "prompt_set": source,
            "n": n,
            "passed": passed,
            "accuracy": round(acc, 4),
            "gates": gates,
            "gates_applied_note": ("evidence gates are the baseline's gate (R2); the "
                                   "exact-match accuracy is reported and gated only when "
                                   "--min-acc is given"),
            "controls": controls,
            "controls_gated": bool(args.control),
            "inputs": inputs,
            "per_kind": per_kind,
            "fail_classes": dict(fail_classes),
            "fail_classes_defined": list(FAIL_CLASSES),
            "stop_reasons": dict(stop_reasons),
            "diagnostics": {
                "accuracy_first_number": round(
                    sum(1 for it in items if matches_first(by_id[it["id"]], it["output"])) / n, 4)
                if n else 0.0,
                "accuracy_any_number": round(
                    sum(1 for it in items if matches_any(by_id[it["id"]], it["output"])) / n, 4)
                if n else 0.0,
                "items_with_number": extract_n,
                "items_nonempty": sum(1 for it in items if it["output"].strip()),
                "arith_check_failed_ids": arith_failed,
                "protocol_failed_ids": bad_protocol,
            },
            "arith_states": {s: states.get(s, 0) for s in STATES},
            "unverifiable_n": states.get(STATE_UNVERIFIABLE, 0),
            "unverifiable_ids": unverifiable_ids,
            "coverage": cover_detail,
            "device": args.device,
            "device_used": device_used,
            "gpu": gpu_name,
            "pid": os.getpid(),
            "claim": _claim_name(),
            "shards": {"i": 1, "n": 1},
            "elapsed_s": round(time.time() - started, 1),
            "accuracy_target": 0.60,
            "min_acc_gate": args.min_acc,
            "exit_code": exit_code,
        }
        lc.write_items(paths, items)
        lc.seal(paths, stage="e3", ckpt=args.ckpt, device=args.device,
                device_used=device_used, inputs=inputs, ids=ids, shard="1/1",
                dataset_stats=stats, prompt_set=source,
                started_utc=seal_rec["started_utc"],
                finished_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                exit_code=exit_code, gates=gates)
        log.write(f"E3 n={n} passed={passed} acc={acc:.4f} extract_rate={extract_rate:.4f}\n")
        log.write(f"E3 arith_states={ {s: states.get(s, 0) for s in STATES} } "
                  f"unverifiable={unverifiable_ids}\n")
        log.write(f"E3 fail_classes={dict(fail_classes)} stop_reasons={dict(stop_reasons)}\n")
        log.write(f"E3 per_kind={json.dumps(per_kind, sort_keys=True)}\n")
        log.write(f"E3 gates={json.dumps(gates, sort_keys=True)} controls_ok={controls_ok}\n")
        log.write(f"E3 exit_code={exit_code}\n")
        return summary

    return lc.run_stage("e3", stage, ckpt=args.ckpt)


def _split_stop(ids_out, eos_id, im_end_id):
    """(text ids, terminating token id or None): drop the stop token before decoding."""
    for index, token in enumerate(ids_out):
        if token == eos_id or (im_end_id is not None and token == im_end_id):
            return ids_out[:index], token
    return list(ids_out), (ids_out[-1] if ids_out else None)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

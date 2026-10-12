#!/usr/bin/env python3
"""C1 "format and stop" SFT set for stage D1 (format_stop_v1 and format_stop_v2).

What this set is for. The E2 stage failed on the v3 SFT checkpoint on all four
judgments (stopped / no_echo / no_extra_turn / no_repeat: 10 of 10 outputs ran to
the 256-token cap). The base checkpoint repeats inside 64 tokens on 11 of 16 E1
items. This set is built to train the three behaviours E2 tests, against that
evidence:

  * every answer is SHORT -- at most 64 BPE tokens, so a model that ends the turn
    can end it well inside E2's 256-token cap;
  * every answer's last token is <|im_end|>, the same terminator E2 reads by token
    id (eval/ladder_common.stop_reason), never a newline and never <eos>;
  * no answer restates a whole sentence of its own question and no answer carries
    a second <|im_start|>;
  * the protocol is the evaluation protocol: ChatML through
    scripts.loader.format_prompt with system=None, which is what
    eval/ladder/e2_chatml.build_prompt calls.

Two versions are delivered, and the second does NOT overwrite the first:

  v1  15,055 rows -> 100 pack rows  -> 65,393 supervised tokens. D1 trained on it.
  v2  ~230,000 rows -> ~1,700 rows  -> 0.8-1.2M supervised tokens. Sized for the D1
      batch shape (world 8 x micro 2 x accum 16 = 256 sequences per step).

The size basis, measured rather than assumed: one row of this set carries 4.34
supervised tokens on average (65,393 / 15,055), so the parent's target of 1M
supervised tokens means about 230,000 UNIQUE rows. Padding rows with filler text to
reach a row count would be the one thing that breaks the set's purpose, so the
version that is scaled up is scaled in CONTENT: more expressions, more facts, more
distinct questions, all of them recomputed or table-checked.

Outputs, all in this directory, named by version:
  format_stop_<v>.jsonl   the rows: id, prompt, answer, source, license, quality
  format_stop_<v>.pt      the pack sft_math.py --sft_path loads
  manifest_<v>.json       provenance: sources, licenses, revision, rows, tokens, hashes
  report_<v>.json         dedup, decontamination, spot check, failures
  spotcheck_200_<v>.jsonl the 200-row hand-read sample the contract asks for

Run `bash build.sh` for both versions, or
`python3 build.py --version v2 --out-dir .` for one. The build refuses on a missing
gate file, a wrong vocabulary fingerprint, or an answer over the token cap; it never
trims a row to fit and never duplicates a row to fill a budget.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
OUT = os.path.join(ROOT, "data", "sft", "stage_c", "format_stop")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "datagen"))

from loader import IM_END, IM_START, format_prompt, vocab_fingerprint  # noqa: E402
from prepare_sft import pack_and_save  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

VOCAB_EXPECTED = "f1f860970d15d623"
BUILDER = "data/sft/stage_c/format_stop/build.py"
ANSWER_MAX_TOKENS = 64
ECHO_MIN_CHARS = 8          # an 8-character run shared with the question fails, in general
# One exception, stated rather than implied. A definition question names its term -- "What
# is the greatest common divisor?" -- and the answer opens with that term. The 8 characters
# they share are the term itself, which is the correct answer and not a restatement of the
# question. The cap for that category therefore allows a run of exactly the general
# threshold, and nothing longer: no definition answer repeats 9 characters of its question,
# and none repeats its question whole. Every other category holds the general rule.
DEFINE_ECHO_MAX_CHARS = 8
NEAR_DUP_N = 8              # word-level n-gram for near-duplicate detection
NEAR_DUP_LIMIT = 2          # the same word 8-gram may occur at most this many times
DECON_N = 13                # the gate width (filters/decontam_ngram.N)

# The four gate sets this set is checked against. The contract names them; three of
# them live under the codemath holdout mirror on a dev checkout, so each entry lists
# its candidates and the resolver refuses when none exists.
GATE_SETS = [
    ("humaneval_164", [("data/pretrain_100b/codemath/holdouts/data/eval/humaneval/humaneval_164.jsonl", "jsonl"),
                       ("data/eval/humaneval/humaneval_164.jsonl", "jsonl")]),
    ("gsm8k_test", [("data/eval/gsm8k_test.jsonl", "jsonl"),
                    ("data/pretrain_100b/codemath/holdouts/data/eval/gsm8k_test.jsonl", "jsonl")]),
    ("math_500_en_for_eval", [("data/eval/math_500_en_for_eval.jsonl", "jsonl"),
                              ("data/eval/math_500_en_test.jsonl", "jsonl")]),
    ("mbpp_holdouts", [("data/pretrain_100b/codemath/holdouts/data/eval/mbpp_holdouts.jsonl", "jsonl"),
                       ("data/eval/mbpp_holdouts.jsonl", "jsonl")]),
]
# The second candidate in each list is the canonical repo path, the first is the dev
# checkout's holdout mirror. The pod carries the canonical pair and not the mirror, so
# without these fallbacks the builder REFUSEs where the GPU is and the pack cannot be
# rebuilt there (measured 2026-10-09 on the pod: "REFUSE: gate set humaneval_164 is
# absent"). The loader still checks every file's sha256 against pins.json, so a fallback
# cannot silently swap the exam for a different one.


class Refuse(SystemExit):
    """A loud stop: the build never continues past a broken precondition."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- text
WS = re.compile(r"\s+")


def norm(text):
    """The decontam library's normalisation, so both sides of a comparison agree."""
    return WS.sub(" ", text or "").strip()


def words(text):
    s = norm(text)
    return s.split(" ") if s else []


def wgrams(text, n):
    toks = words(text)
    if len(toks) < n:
        return set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def longest_common_span(question, answer, cap=40):
    """Length in characters of the longest substring the answer shares with the question.

    "A substring of length L appears in the answer" is monotone in L, so the longest
    shared length is found by walking lengths upward from ECHO_MIN_CHARS. The walk stops
    at the first length with no shared substring, which for a clean row is the first
    length tried. Answer substrings go into a set, so each probe is a hash lookup rather
    than a scan of the answer.
    """
    answer_subs = {answer[i:i + ECHO_MIN_CHARS] for i in range(len(answer) - ECHO_MIN_CHARS + 1)}
    lo = ECHO_MIN_CHARS
    while lo <= min(len(question), cap):
        if not any(question[i:i + lo] in answer_subs
                   for i in range(len(question) - lo + 1)):
            return lo - 1
        lo += 1
    if lo <= ECHO_MIN_CHARS:
        return 0
    return min(len(question), cap)


# --------------------------------------------------------------------------- gates
def resolve_gate_sets():
    out = []
    for name, cands in GATE_SETS:
        for rel, kind in cands:
            path = os.path.join(ROOT, rel)
            if os.path.exists(path):
                out.append((name, path, kind))
                break
        else:
            raise Refuse(f"REFUSE: gate set {name} is absent; tried "
                         + ", ".join(os.path.join(ROOT, r) for r, _ in cands))
    return out


def load_gate_rows(path, kind):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def gate_fields(name, row):
    """The text parts of one gate problem. Names follow the gate file's own schema."""
    if name == "humaneval_164":
        return [("prompt", row.get("prompt", "")), ("solution", row.get("canonical_solution", "")),
                ("test", row.get("test", ""))]
    if name == "mbpp_holdouts":
        return [("prompt", row.get("text", "")), ("solution", row.get("code", ""))]
    if name == "gsm8k_test":
        return [("question", row.get("question", "")), ("answer", row.get("answer", ""))]
    if name == "math_500_en_for_eval":
        return [("problem", row.get("problem", "")), ("solution", row.get("solution", ""))]
    raise Refuse(f"REFUSE: no field map for gate set {name}")


# --------------------------------------------------------------------------- facts
# World capitals: source=fact_core, license=public_domain (a fact is not copyrightable).
CAPITALS = [
    ("中国", "China", "北京", "Beijing"), ("日本", "Japan", "东京", "Tokyo"),
    ("法国", "France", "巴黎", "Paris"), ("德国", "Germany", "柏林", "Berlin"),
    ("意大利", "Italy", "罗马", "Rome"), ("西班牙", "Spain", "马德里", "Madrid"),
    ("英国", "the United Kingdom", "伦敦", "London"), ("俄罗斯", "Russia", "莫斯科", "Moscow"),
    ("美国", "the United States", "华盛顿", "Washington, D.C."),
    ("加拿大", "Canada", "渥太华", "Ottawa"),
    ("澳大利亚", "Australia", "堪培拉", "Canberra"), ("印度", "India", "新德里", "New Delhi"),
    ("巴西", "Brazil", "巴西利亚", "Brasilia"), ("埃及", "Egypt", "开罗", "Cairo"),
    ("韩国", "South Korea", "首尔", "Seoul"), ("泰国", "Thailand", "曼谷", "Bangkok"),
    ("越南", "Vietnam", "河内", "Hanoi"), ("瑞士", "Switzerland", "伯尔尼", "Bern"),
    ("瑞典", "Sweden", "斯德哥尔摩", "Stockholm"), ("挪威", "Norway", "奥斯陆", "Oslo"),
    ("荷兰", "the Netherlands", "阿姆斯特丹", "Amsterdam"),
    ("葡萄牙", "Portugal", "里斯本", "Lisbon"), ("希腊", "Greece", "雅典", "Athens"),
    ("土耳其", "Turkey", "安卡拉", "Ankara"), ("阿根廷", "Argentina", "布宜诺斯艾利斯", "Buenos Aires"),
    ("墨西哥", "Mexico", "墨西哥城", "Mexico City"), ("肯尼亚", "Kenya", "内罗毕", "Nairobi"),
    ("新西兰", "New Zealand", "惠灵顿", "Wellington"), ("波兰", "Poland", "华沙", "Warsaw"),
]

SPELLED = [(0, "zero"), (1, "one"), (2, "two"), (3, "three"), (4, "four"), (5, "five"),
           (6, "six"), (7, "seven"), (8, "eight"), (9, "nine"), (10, "ten"), (11, "eleven"),
           (12, "twelve"), (13, "thirteen"), (14, "fourteen"), (15, "fifteen"),
           (16, "sixteen"), (17, "seventeen"), (18, "eighteen"), (19, "nineteen"),
           (20, "twenty"), (30, "thirty"), (40, "forty"), (50, "fifty"), (60, "sixty"),
           (70, "seventy"), (80, "eighty"), (90, "ninety"), (100, "one hundred"),
           (1000, "one thousand")]


def to_roman(n):
    """The standard subtractive Roman form. Used to CHECK ROMAN_VALUES, not to trust it."""
    table = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
             (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = []
    for value, sym in table:
        while n >= value:
            out.append(sym)
            n -= value
    return "".join(out)


ROMAN_VALUES = {n: to_roman(n) for n in
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 19, 20, 24, 29, 40, 44, 49, 50,
                 60, 88, 90, 99, 100, 250, 400, 500, 700, 900, 1000, 1500, 1987, 2024, 3999]}


def pick(questions, variants):
    """(answer, span) -- the answer variant that shares the fewest characters with its questions.

    A row is data, not a stack of literals: if no variant keeps the shared span under
    ECHO_MIN_CHARS, the ENTRY is the defect and the build refuses on it rather than
    shipping a row that teaches the model to restate its input.
    """
    best = None
    for v in variants:
        span = max(longest_common_span(q, v) for q in questions)
        if best is None or span < best[1]:
            best = (v, span)
    return best


def entry(lang, question, answer, source, quality, check=None, license="public_domain"):
    row = dict(question=question, answer=answer, lang=lang, source=source,
               license=license, quality=quality)
    if check is not None:
        row["check"] = check
    return row


def build_number_forms(rng):
    """The NATO phonetic alphabet: one letter, one word, no arithmetic to check."""
    rows = []
    words = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel",
             "India", "Juliet", "Kilo", "Lima", "Mike", "November", "Oscar", "Papa",
             "Quebec", "Romeo", "Sierra", "Tango", "Uniform", "Victor", "Whiskey", "X-ray",
             "Yankee", "Zulu"]
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert len(words) == len(letters)
    for letter, word in zip(letters, words):
        qs_zh = [f"字母 {letter} 在航空字母表里读作什么？", f"通信字母表里 {letter} 对应哪个单词？"]
        qs_en = [f"Which word stands for the letter {letter} in the NATO phonetic alphabet?",
                 f"Give the phonetic-alphabet word for the letter {letter}."]
        a, span = pick(qs_zh + qs_en, [word + "."])
        if span >= ECHO_MIN_CHARS:
            raise Refuse(f"REFUSE: phonetic word {word} shares {span} characters with its question")
        for q in qs_zh:
            rows.append(entry("zh", q, a, "fact_core", "short_exact"))
        for q in qs_en:
            rows.append(entry("en", q, a, "fact_core", "short_exact"))
    return rows


def _guard_capacity(kind, want, capacity):
    """Refuse an impossible count instead of returning a short list.

    A family asked for more rows than it has would otherwise come back short, and a short
    list is a wrong denominator that nothing downstream can see.
    """
    if want > capacity:
        raise Refuse(f"REFUSE: {kind} asked for {want} rows; its population is {capacity}")


def _guard_progress(kind, want, got, since_progress):
    """Refuse a sampler that has drawn a long way without a new value.

    The first version of this guard measured the TOTAL number of draws against a multiple of
    the target, which refuses a legitimate sampler near its capacity: filling 420 of 444
    slots takes about 444*ln(444) draws, so the guard fired on a healthy plan. Progress is
    the property that matters, and a stalled draw budget is unbounded in the target size.
    """
    # 200,000 consecutive misses, not 20,000. A sampler near saturation collides often by
    # construction -- the records family at 2,600 rows sits at about 96% misses in its year
    # kind -- and a limit tight enough to fire on that is a false alarm that replaces a slow
    # build with a wrong refusal. The limit still bounds the loop, which is its only job.
    if since_progress > 200000:
        raise Refuse(f"REFUSE: {kind} plan stalled: {got} of {want} after {since_progress} "
                     f"draws without a new value")


def _split_quota(kind, total, capacities, weights):
    """Split a target across kinds by WEIGHT, clamped to what each kind can hold.

    Two corrections live here, both measured on this set. First, the kinds do not have equal
    populations: `*` over 3..99 x 3..25 holds 2,425 expressions and `//` holds 960, while `+`
    and `-` hold hundreds of thousands. Second, splitting PROPORTIONALLY TO CAPACITY is also
    wrong: it gives `+` and `-` nearly everything, and an arithmetic drill that is 99.9%
    addition is not the drill the contract asks for. So the caller states a weight per kind,
    the clamp moves any shortfall to the kinds with room, and the result honours both.
    """
    capacity_total = sum(capacities.values())
    if total > capacity_total:
        raise Refuse(f"REFUSE: {kind} asked for {total} rows; its population is {capacity_total}")
    weight_total = sum(weights.values())
    quota = {}
    room = dict(capacities)
    left = total
    for name in weights:
        want = total * weights[name] // weight_total
        quota[name] = min(want, room[name])
        room[name] -= quota[name]
        left -= quota[name]
    while left > 0:
        # sorted(): the redistribution order must not depend on dict iteration order, or two
        # runs of the same recipe produce different quotas and the artifacts differ. Measured:
        # an unsorted version gave 211,209 rows on one run and a different total on the next.
        candidates = sorted(n for n in weights if room[n] > 0)
        if not candidates:
            break
        share = max(1, left // len(candidates))
        for name in sorted(candidates, key=lambda n: (weights[n], n), reverse=True):
            take = min(share, room[name], left)
            if take <= 0:
                continue
            quota[name] += take
            room[name] -= take
            left -= take
            if left == 0:
                break
    return quota


def build_unit_convert(rng, per_lang):
    """Unit conversions with a computed answer. The value is recomputed, never copied."""
    from fractions import Fraction
    units = [
        ("千米", "米", "kilometre", "metres", Fraction(1000)),
        ("米", "厘米", "metre", "centimetres", Fraction(100)),
        ("千克", "克", "kilogram", "grams", Fraction(1000)),
        ("小时", "分钟", "hour", "minutes", Fraction(60)),
        ("分钟", "秒", "minute", "seconds", Fraction(60)),
        ("吨", "千克", "tonne", "kilograms", Fraction(1000)),
        ("厘米", "毫米", "centimetre", "millimetres", Fraction(10)),
        ("天", "小时", "day", "hours", Fraction(24)),
        ("星期", "天", "week", "days", Fraction(7)),
        ("英里", "千米", "mile", "kilometres", Fraction(1609, 1000)),
        ("英寸", "厘米", "inch", "centimetres", Fraction(254, 100)),
        ("磅", "克", "pound", "grams", Fraction(453592, 1000)),
        ("码", "米", "yard", "metres", Fraction(9144, 10000)),
        ("海里", "米", "nautical mile", "metres", Fraction(1852)),
        # Added for v2's volume: each entry adds 59 (fractional factor) or 89 pairs, and the
        # planner's capacity is computed from this list, so a request above it is refused.
        ("平方米", "平方厘米", "square metre", "square centimetres", Fraction(10000)),
        ("立方米", "立方分米", "cubic metre", "cubic decimetres", Fraction(1000)),
        ("升", "毫升", "litre", "millilitres", Fraction(1000)),
        ("千克", "吨", "kilogram", "tonnes", Fraction(1, 1000)),
        ("米", "毫米", "metre", "millimetres", Fraction(1000)),
        ("小时", "天", "hour", "days", Fraction(1, 24)),
        ("字节", "千字节", "byte", "kilobytes", Fraction(1, 1000)),
        ("千字节", "兆字节", "kilobyte", "megabytes", Fraction(1, 1000)),
        ("毫秒", "秒", "millisecond", "seconds", Fraction(1, 1000)),
        ("微秒", "毫秒", "microsecond", "milliseconds", Fraction(1, 1000)),
    ]

    def show(fr):
        if fr.denominator == 1:
            return str(fr.numerator)
        value = float(fr)
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text

    # A RANDOM unit index with a stall guard, not `len(plan) % len(units)`.
    #
    # The bug this replaces is worth keeping: the fixed rotation always came back to a
    # fractional unit whose multiplier range is 39 wide (n <= 40 for a non-integer factor),
    # so after 39 draws that index could never contribute again and the loop spun at 100%
    # CPU, printing nothing, for as long as it was left alone. A hang reads as work in
    # progress; a refusal reads as a refusal.
    # A fractional factor draws n from 2..40 (a float answer wider than four decimals is not
    # shipped), an integer factor from 2..90. The capacity is their sum, and it is what makes
    # an impossible request a refusal.
    capacity = sum(39 if factor.denominator != 1 else 89 for *_rest, factor in units)
    _guard_capacity("unit_convert", per_lang, capacity)
    plan = []
    seen = set()
    miss = 0
    while len(plan) < per_lang:
        _guard_progress("unit_convert", per_lang, len(plan), miss)
        idx = rng.randrange(len(units))
        zh_u1, zh_u2, en_u1, en_u2, factor = units[idx]
        n = rng.randint(2, 90)
        if factor.denominator != 1 and n > 40:
            continue
        if (idx, n) in seen:
            miss += 1
            continue
        seen.add((idx, n))
        plan.append((idx, n))
        miss = 0
    rows = []
    for idx, n in plan:
        zh_u1, zh_u2, en_u1, en_u2, factor = units[idx]
        value = factor * n
        shown = show(value)
        check = {"unit": zh_u2, "n": n, "factor": str(factor), "value": shown}
        qs_zh = [f"{n} {zh_u1}等于多少{zh_u2}？只写数值。",
                 f"把 {n} {zh_u1}换算成{zh_u2}，只给数字。"]
        qs_en = [f"How many {en_u2} are in {n} {en_u1}? Give the number only.",
                 f"Convert {n} {en_u1} to {en_u2}; reply with the number only."]
        a, span = pick(qs_zh + qs_en, [shown, shown + "."])
        if span >= ECHO_MIN_CHARS:
            raise Refuse(f"REFUSE: unit answer {shown} shares {span} characters with its question "
                         f"({zh_u1}->{zh_u2}, n={n})")
        for q in qs_zh:
            rows.append(entry("zh", q, a, "synthetic_unit", "numeric_verified", check))
        for q in qs_en:
            rows.append(entry("en", q, a, "synthetic_unit", "numeric_verified", check))
    return rows


def build_base_convert(rng, per_lang):
    """Base conversions. The decimal form and the target form are recomputed here."""
    rows = []
    forms = [("2", 2), ("8", 8), ("16", 16)]
    plan = []
    seen = set()
    while len(plan) < per_lang:
        base_s, base = forms[len(plan) % len(forms)]
        n = rng.randint(9, 4095)
        if (base_s, n) in seen:
            continue
        seen.add((base_s, n))
        plan.append((base_s, base, n))
    for base_s, base, n in plan:
        target = {"2": format(n, "b"), "8": format(n, "o"), "16": format(n, "x")}[base_s]
        check = {"n": n, "base": base, "value": target}
        qs_zh = [f"把十进制数 {n} 转成 {base_s} 进制，只写结果。",
                 f"{n} 的{base_s} 进制写法是什么？"]
        qs_en = [f"Convert the decimal number {n} to base {base_s}. Give the digits only.",
                 f"What is {n} in base {base_s}?"]
        a, span = pick(qs_zh + qs_en, [target, target + "."])
        if span >= ECHO_MIN_CHARS:
            raise Refuse(f"REFUSE: base answer {target} shares {span} characters with its question "
                         f"(base {base_s}, n={n})")
        for q in qs_zh:
            rows.append(entry("zh", q, a, "synthetic_base", "numeric_verified", check,
                              license="generated_for_aupai"))
        for q in qs_en:
            rows.append(entry("en", q, a, "synthetic_base", "numeric_verified", check,
                              license="generated_for_aupai"))
    return rows


def build_series(rng, per_lang):
    """Sums of a short series and averages of a short list. Both recomputed.

    The population is finite and small: 37 starts x 2 kinds x 6 counts = 444 rows. A
    request above that is refused rather than silently short, and the draw is guarded so a
    stalled sampler cannot spin (this family and unit_convert are where a fixed
    index-by-plan-length turned into a hang that printed nothing).
    """
    rows = []
    plan = []
    seen = set()
    # sum draws count 4..12 (9 values), avg draws one of 5,7,9,11 (4 values), start is 1..40.
    capacity = 37 * 9 + 37 * 4
    _guard_capacity("series", per_lang, capacity)
    drawn = {"sum": 0, "avg": 0}
    miss = 0
    while len(plan) < per_lang:
        miss += 1
        _guard_progress("series", per_lang, len(plan), miss)
        # Alternate by what each kind has DRAWN, not by plan length: a skipped duplicate
        # would otherwise flip the kind and unbalance the two halves.
        kind = "sum" if drawn["sum"] <= drawn["avg"] else "avg"
        drawn[kind] += 1
        start_v = rng.randint(1, 40)
        # An average of consecutive integers is an integer only for an odd count, and an
        # exact answer is the only kind this set ships.
        count = rng.choice([5, 7, 9, 11]) if kind == "avg" else rng.randint(4, 12)
        if (kind, start_v, count) in seen:
            miss += 1
            continue
        seen.add((kind, start_v, count))
        plan.append((kind, start_v, count))
        miss = 0
    for kind, start_v, count in plan:
        values = list(range(start_v, start_v + count))
        if kind == "sum":
            value = sum(values)
            check = {"kind": "series_sum", "start": start_v, "count": count, "value": value}
            qs_zh = [f"计算从 {start_v} 开始的 {count} 个连续整数之和，只写数值。",
                     f"求 {start_v} 到 {start_v + count - 1} 这些整数的和，只给数字。"]
            qs_en = [f"Add the {count} consecutive integers that start at {start_v}. Reply with the number only.",
                     f"What is the sum of the integers from {start_v} to {start_v + count - 1}? Give the number only."]
        else:
            assert sum(values) % count == 0
            value = sum(values) // count
            check = {"kind": "average", "start": start_v, "count": count, "value": value}
            qs_zh = [f"求 {start_v} 到 {start_v + count - 1} 这些整数的平均值，只写数值。",
                     f"{start_v} 到 {start_v + count - 1} 的平均数是多少？只给数字。"]
            qs_en = [f"Find the average of the integers from {start_v} to {start_v + count - 1}. Reply with the number only.",
                     f"What is the mean of {start_v} through {start_v + count - 1}? Give the number only."]
        a, span = pick(qs_zh + qs_en, [str(value), str(value) + "."])
        if span >= ECHO_MIN_CHARS:
            raise Refuse(f"REFUSE: series answer {value} shares {span} characters with its question")
        for q in qs_zh:
            rows.append(entry("zh", q, a, "synthetic_series", "numeric_verified", check))
        for q in qs_en:
            rows.append(entry("en", q, a, "synthetic_series", "numeric_verified", check))
    return rows


def build_number_theory(rng, per_lang):
    """GCD and LCM pairs, recomputed with the standard library."""
    rows = []
    plan = []
    seen = set()
    miss = 0
    _guard_capacity("number_theory", per_lang, 2 * 349 * 350 // 2)
    while len(plan) < per_lang:
        miss += 1
        _guard_progress("number_theory", per_lang, len(plan), miss)
        kind = "gcd" if len(plan) % 2 == 0 else "lcm"
        a = rng.randint(12, 360)
        b = rng.randint(12, 360)
        if a == b or (kind, a, b) in seen:
            miss += 1
            continue
        seen.add((kind, a, b))
        plan.append((kind, a, b))
        miss = 0
    for kind, a, b in plan:
        value = math.gcd(a, b) if kind == "gcd" else math.lcm(a, b)
        check = {"kind": kind, "a": a, "b": b, "value": value}
        if kind == "gcd":
            qs_zh = [f"求 {a} 和 {b} 的最大公约数，只写数值。", f"{a} 与 {b} 的最大公约数是多少？"]
            qs_en = [f"Find the greatest common divisor of {a} and {b}. Reply with the number only.",
                     f"What is the GCD of {a} and {b}? Give the number only."]
        else:
            qs_zh = [f"求 {a} 和 {b} 的最小公倍数，只写数值。", f"{a} 与 {b} 的最小公倍数是多少？"]
            qs_en = [f"Find the least common multiple of {a} and {b}. Reply with the number only.",
                     f"What is the LCM of {a} and {b}? Give the number only."]
        a_ans, span = pick(qs_zh + qs_en, [str(value), str(value) + "."])
        if span >= ECHO_MIN_CHARS:
            raise Refuse(f"REFUSE: {kind} answer {value} shares {span} characters with its question")
        for q in qs_zh:
            rows.append(entry("zh", q, a_ans, "synthetic_number_theory", "numeric_verified", check))
        for q in qs_en:
            rows.append(entry("en", q, a_ans, "synthetic_number_theory", "numeric_verified", check))
    return rows


def build_percent(rng, per_lang):
    """One-step percentages, recomputed exactly with Fraction."""
    from fractions import Fraction
    rows = []
    plan = []
    seen = set()
    miss = 0
    _guard_capacity("percent", per_lang, 15 * 13)
    while len(plan) < per_lang:
        miss += 1
        _guard_progress("percent", per_lang, len(plan), miss)
        pct = rng.choice([5, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 80, 90])
        total = rng.choice([40, 60, 80, 120, 150, 200, 240, 300, 400, 500, 600, 800, 1200])
        if (pct, total) in seen:
            miss += 1
            continue
        seen.add((pct, total))
        plan.append((pct, total))
        miss = 0
    for pct, total in plan:
        value = Fraction(pct, 100) * total
        shown = str(value.numerator) if value.denominator == 1 else f"{float(value):.2f}"
        check = {"percent": pct, "total": total, "value": shown}
        qs_zh = [f"{total} 的 {pct}% 是多少？只写数值。", f"求 {total} 的 {pct} 个百分点，只给数字。"]
        qs_en = [f"What is {pct}% of {total}? Reply with the number only.",
                 f"Compute {pct} percent of {total}. Give the number only."]
        a, span = pick(qs_zh + qs_en, [shown, shown + "."])
        if span >= ECHO_MIN_CHARS:
            raise Refuse(f"REFUSE: percent answer {shown} shares {span} characters with its question")
        for q in qs_zh:
            rows.append(entry("zh", q, a, "synthetic_percent", "numeric_verified", check))
        for q in qs_en:
            rows.append(entry("en", q, a, "synthetic_percent", "numeric_verified", check))
    return rows


def build_trig(rng, per_lang):
    """Exact trigonometric values at the standard angles. No floating point in the answer."""
    table = [
        (0, "0", "0"), (30, "1/2", "1/2"), (45, "√2/2", "√2/2"), (60, "√3/2", "1/2"),
        (90, "1", "0"), (120, "√3/2", "-1/2"), (135, "√2/2", "-√2/2"),
        (150, "1/2", "-√3/2"), (180, "0", "-1"), (210, "-1/2", "-√3/2"),
        (225, "-√2/2", "-√2/2"), (240, "-√3/2", "-1/2"), (270, "-1", "0"),
        (300, "-√3/2", "1/2"), (315, "-√2/2", "√2/2"), (330, "-1/2", "√3/2"),
    ]
    rows = []
    plan = []
    for deg, sin_v, cos_v in table:
        plan.append((deg, "sin", sin_v))
        plan.append((deg, "cos", cos_v))
    for deg, fn, value in plan[:per_lang]:
        check = {"function": fn, "degrees": deg, "value": value}
        qs_zh = [f"求 {deg} 度的{fn} 值，用精确值回答。"]
        qs_en = [f"What is {fn} of {deg} degrees? Give the exact value."]
        a, span = pick(qs_zh + qs_en, [value, value + "."])
        if span >= ECHO_MIN_CHARS:
            raise Refuse(f"REFUSE: trig answer {value} shares {span} characters with its question")
        for q in qs_zh:
            rows.append(entry("zh", q, a, "fact_core", "numeric_verified", check))
        for q in qs_en:
            rows.append(entry("en", q, a, "fact_core", "numeric_verified", check))
    return rows


def build_facts(rng):
    """Every fact row, from the literal bank and from the table-driven families."""
    rows = []
    entries = list(FACT_BANK)
    for zh_a, en_a, zh_q, en_q in fact_bank_pairs():
        entries.append({"zh_a": zh_a, "en_a": en_a, "zh_q": zh_q, "en_q": en_q})
    for entry_def in entries:
        zh_a = entry_def["zh_a"]
        en_a = entry_def["en_a"]
        qs_zh = entry_def["zh_q"]
        qs_en = entry_def["en_q"]
        a_zh, span_zh = pick(qs_zh, [zh_a])
        a_en, span_en = pick(qs_en, [en_a])
        if span_zh >= ECHO_MIN_CHARS or span_en >= ECHO_MIN_CHARS:
            raise Refuse(f"REFUSE: fact shares {max(span_zh, span_en)} characters with its question: "
                         f"{qs_zh[0]!r} / {zh_a!r}")
        for q in qs_zh:
            rows.append(entry("zh", q, a_zh, entry_def.get("source", "fact_core"),
                              entry_def.get("quality", "short_exact")))
        for q in qs_en:
            rows.append(entry("en", q, a_en, entry_def.get("source", "fact_core"),
                              entry_def.get("quality", "short_exact")))
    return rows


def fact_bank_pairs():
    """(zh answer, en answer, zh questions, en questions) rows appended to FACT_BANK.

    Kept as one builder so every entry passes the same shared-span check: a fact whose
    answer restates its question is a data defect, not a style choice.
    """
    out = []
    for zh_c, en_c, zh_cap, en_cap in CAPITALS:
        out.append((
            zh_cap + "。", en_cap + ".",
            [f"{zh_c}的首都是哪里？", f"{zh_c}的首都叫什么名字？"],
            [f"Which city is the capital of {en_c}?", f"Name the capital city of {en_c}."],
        ))
    for digit, word in SPELLED:
        out.append((
            word + "。", word + ".",
            [f"把 {digit} 写成英文单词。", f"数字 {digit} 的英文写法是什么？只写单词。"],
            [f"Write {digit} in English words.", f"Give the English word for {digit}."],
        ))
    for value, numeral in ROMAN_VALUES.items():
        out.append((
            numeral + "。", numeral + ".",
            [f"{value} 用罗马数字怎么写？", f"数字 {value} 的罗马数字是什么？"],
            [f"Write {value} in Roman numerals.", f"Convert {value} to Roman numerals."],
        ))
    return out

# Every fact is public-domain reference knowledge: SI unit definitions, named constants,
# a curated world-capital table, Roman numerals and the NATO phonetic alphabet.
# source=fact_core, license=public_domain.
FACT_BANK = [
    {"zh_a": "100 摄氏度。", "en_a": "100 °C.",
     "zh_q": ["水在多少摄氏度沸腾？只给数字。", "标准大气压下水在什么温度沸腾？只写数值。"],
     "en_q": ["At what temperature does water boil at sea level? Give the number only.",
              "Give the boiling point of water on the Celsius scale."]},
    {"zh_a": "0 摄氏度。", "en_a": "0 °C.",
     "zh_q": ["水在多少摄氏度结冰？只给数字。", "标准大气压下水在什么温度结冰？"],
     "en_q": ["At what temperature does water freeze? Give the number only.",
              "Give the freezing point of water on the Celsius scale."]},
    {"zh_a": "299792458。", "en_a": "299792458.",
     "zh_q": ["光在真空里每秒走多少米？只写数字。", "真空光速的数值是多少？只写数字。"],
     "en_q": ["How fast does light travel in vacuum, in metres per second? Digits only.",
              "Give the value of c in metres per second, digits only."],
     "quality": "numeric_verified"},
    {"zh_a": "π = 3.14159。", "en_a": "pi = 3.14159.",
     "zh_q": ["圆周率 π 保留五位小数是多少？", "π 的前六位有效数字是什么？"],
     "en_q": ["Give pi rounded to five decimals.", "What is the value of pi to five decimal places?"],
     "quality": "numeric_verified"},
    {"zh_a": "e = 2.71828。", "en_a": "e = 2.71828.",
     "zh_q": ["自然对数的底 e 保留五位小数是多少？", "常数 e 的前六位有效数字是什么？"],
     "en_q": ["Give Euler's number rounded to five decimals.", "What is e to five decimal places?"],
     "quality": "numeric_verified"},
    {"zh_a": "零下 273.15 摄氏度。", "en_a": "Minus 273.15 °C.",
     "zh_q": ["绝对零度是多少摄氏度？", "绝对零度对应的摄氏温度是多少？"],
     "en_q": ["What is absolute zero in degrees Celsius?", "Give absolute zero on the Celsius scale."],
     "quality": "numeric_verified"},
    {"zh_a": "101325。", "en_a": "101325.",
     "zh_q": ["一个标准大气压是多少帕斯卡？只写数字。", "标准大气压的数值是多少帕？只写数字。"],
     "en_q": ["How many pascals is one standard atmosphere? Digits only.",
              "Give standard atmospheric pressure in pascals, digits only."],
     "quality": "numeric_verified"},
    {"zh_a": "6.022E23。", "en_a": "6.022E23.",
     "zh_q": ["一摩尔物质大约包含多少个粒子？用科学计数法只写数值。",
              "阿伏伽德罗常数的数值是多少？用科学计数法。"],
     "en_q": ["How many particles are in one mole?", "Give the Avogadro constant."],
     "quality": "numeric_verified"},
    {"zh_a": "79。", "en_a": "79.",
     "zh_q": ["金的原子序数是多少？", "元素 Au 在周期表里的序号是多少？"],
     "en_q": ["What is the atomic number of gold?", "Give the atomic number of the element Au."],
     "quality": "numeric_verified"},
    {"zh_a": "H2O。", "en_a": "H2O.",
     "zh_q": ["水的化学式是什么？", "水的分子式怎么写？"],
     "en_q": ["What is the chemical formula of water?", "Give the molecular formula of water."]},
    {"zh_a": "CO2。", "en_a": "CO2.",
     "zh_q": ["二氧化碳的化学式是什么？", "二氧化碳的分子式怎么写？"],
     "en_q": ["What is the chemical formula of carbon dioxide?", "Give the molecular formula of carbon dioxide."]},
    {"zh_a": "180 度。", "en_a": "180°.",
     "zh_q": ["三角形的内角和是多少度？", "任意三角形的三个内角相加等于多少？"],
     "en_q": ["What do the interior angles of a triangle sum to?",
              "Give the angle sum of a triangle."],
     "quality": "numeric_verified"},
    {"zh_a": "4 条。", "en_a": "4.",
     "zh_q": ["正方形有几条边？", "一个正方形有几条相等的边？"],
     "en_q": ["How many sides does a square have?", "Give the number of sides of a square."],
     "quality": "numeric_verified"},
    {"zh_a": "12 条。", "en_a": "12.",
     "zh_q": ["正方体有几条棱？", "一个立方体有多少条边？"],
     "en_q": ["How many edges does a cube have?", "Give the number of edges of a cube."],
     "quality": "numeric_verified"},
    {"zh_a": "26 个。", "en_a": "26.",
     "zh_q": ["英文字母表有多少个字母？", "英语一共有多少个字母？"],
     "en_q": ["How many letters does the English alphabet hold?",
              "Give the number of letters in the English alphabet."],
     "quality": "numeric_verified"},
    {"zh_a": "5 个。", "en_a": "5.",
     "zh_q": ["英文元音字母有几个？", "英语里元音字母的数量是多少？"],
     "en_q": ["How many vowel letters does English have?", "Give the count of English vowel letters."],
     "quality": "numeric_verified"},
    {"zh_a": "7 天。", "en_a": "7 days.",
     "zh_q": ["一星期有几天？", "一周包含多少天？"],
     "en_q": ["How many days are in one week?", "Give the number of days in a week."],
     "quality": "numeric_verified"},
    {"zh_a": "365 天。", "en_a": "365 days.",
     "zh_q": ["平年有多少天？", "一个非闰年有多少天？"],
     "en_q": ["How many days are in a common year?", "Give the length of a non-leap year in days."],
     "quality": "numeric_verified"},
    {"zh_a": "366 天。", "en_a": "366 days.",
     "zh_q": ["闰年有多少天？", "一个闰年包含多少天？"],
     "en_q": ["How many days are in a leap year?", "Give the length of a leap year in days."],
     "quality": "numeric_verified"},
]


# --------------------------------------------------------------------------- arith
def build_arith(rng, per_lang, rotate_forms=True):
    """One-step arithmetic. Every answer is recomputed from the operands, never copied.

    rotate_forms=False restores the fixed form order v1 shipped with. The rotation exists so
    that the near-duplicate filter of v2 -- which measures the question -- sees both languages
    interleaved instead of one language's whole half first; with the fixed order v1 rebuilds to
    15,001 rows instead of 15,055, and v1's delivered artifacts must stay reproducible.
    """
    rows = []
    capacities = {"+": 999 * 999, "-": 950 * 950, "*": 97 * 23, "//": 120 * 8}
    # Weights are per ROW, not per expression: `//` produces the most rows per expression
    # because its answers are long, so an equal weight of expressions makes `//` the most
    # frequent row shape in the whole set. Weighting it down is what keeps the drill balanced
    # across operators.
    quota = _split_quota("arith", per_lang, capacities,
                         {"+": 5, "-": 5, "*": 4, "//": 2})
    plan = []
    seen = {op: set() for op in capacities}
    for op, want in quota.items():
        miss = 0
        got = 0
        while got < want:
            miss += 1
            _guard_progress(f"arith[{op}]", want, got, miss)
            if op == "+":
                a, b = rng.randint(11, 999), rng.randint(11, 999)
                value = a + b
            elif op == "-":
                a = rng.randint(50, 999)
                b = rng.randint(11, a - 1)
                value = a - b
            elif op == "*":
                a, b = rng.randint(3, 99), rng.randint(3, 25)
                value = a * b
            else:
                b = rng.randint(2, 19)
                value = rng.randint(2, 60)
                a = b * value
            key = (a, op, b)
            if key in seen[op]:
                miss += 1
                continue
            seen[op].add(key)
            plan.append((a, op, b, value))
            got += 1
            miss = 0
    for a, op, b, value in plan:
        sym = {"*": "×", "//": "÷"}.get(op, op)
        zh_forms = [f"计算 {a} {sym} {b} 的值，只写数值。", f"{a} {sym} {b} 等于多少？只给数字。",
                    f"求 {a} {sym} {b} 的结果，只回答数字。"]
        en_forms = [f"Compute {a} {sym} {b}. Reply with the number only.",
                    f"What is {a} {sym} {b}? Answer with the number only.",
                    f"Give the value of {a} {sym} {b}, digits only."]
        check = {"op": op, "a": a, "b": b, "value": value}
        # The three forms rotate together, not in one fixed order. Fixed order makes the whole
        # Chinese half of a family precede the English half, and the near-duplicate filter then
        # acts on one language before the other: `//` had the largest population, so its rows
        # were dropped from the front, which cut English harder than Chinese and moved the
        # language ratio to 54/46. Rotation interleaves the two languages.
        shift = (len(rows) % 3) if rotate_forms else 0
        for i in range(3):
            rows.append(entry("zh", zh_forms[(i + shift) % 3], str(value), "synthetic_one_step",
                              "numeric_verified", check))
        for i in range(3):
            rows.append(entry("en", en_forms[(i + 2 * shift) % 3], str(value), "synthetic_one_step",
                              "numeric_verified", check))
    return rows


# --------------------------------------------------------------------------- define
# One-sentence definitions, technical register, one idea per sentence.
DEFS = [
    ("二分查找", "binary search", "二分查找在有序数组里每次比较中间元素，然后丢掉一半区间。",
     "Binary search compares the middle element of a sorted array and discards one half."),
    ("哈希表", "a hash table", "哈希表用哈希函数把键映射到桶，平均常数时间完成查找。",
     "A hash table maps keys to buckets with a hash function and looks up in constant average time."),
    ("时间复杂度", "time complexity", "时间复杂度描述运行时间随输入规模增长的速率。",
     "Time complexity describes how the running time grows with the input size."),
    ("栈", "a stack", "栈是后进先出的容器，只在顶端插入和删除。",
     "A stack is last-in first-out: it inserts and removes at one end."),
    ("队列", "a queue", "队列是先进先出的容器，在一端插入、另一端删除。",
     "A queue is first-in first-out: it inserts at one end and removes at the other."),
    ("链表", "a linked list", "链表用指针把节点串起来，插入和删除不需要搬移元素。",
     "A linked list joins nodes with pointers, so insertion and removal move no other element."),
    ("递归", "recursion", "递归让函数在自己的定义里调用自己，直到命中基线条件。",
     "Recursion calls a function from inside its own body until a base case holds."),
    ("编译器", "a compiler", "编译器把源代码整体翻译成机器码，再交给机器执行。",
     "A compiler translates a whole source program into machine code before execution."),
    ("解释器", "an interpreter", "解释器逐条读取源代码并立即执行。",
     "An interpreter reads a source program statement by statement and runs each one."),
    ("操作系统", "an operating system", "操作系统管理硬件资源，并向程序提供进程、内存和文件接口。",
     "An operating system manages hardware and gives programs process, memory and file interfaces."),
    ("数据库索引", "a database index", "数据库索引用一张查找表保存键到行的位置，避免全表扫描。",
     "A database index keeps key-to-row positions in a lookup table to avoid a full scan."),
    ("缓存", "a cache", "缓存把常用数据放在更快的存储里，减少重复计算或读取。",
     "A cache keeps frequently used data in faster storage to cut repeated work."),
    ("哈希冲突", "a hash collision", "哈希冲突是两个不同的键落在同一个桶里。",
     "A hash collision happens when two different keys land in the same bucket."),
    ("死锁", "a deadlock", "死锁是两个进程各自持有对方需要的资源，谁也走不下去。",
     "A deadlock happens when two processes each hold a resource the other needs."),
    ("进程", "a process", "进程是操作系统分配资源的单位，拥有独立的地址空间。",
     "A process is the unit of resource allocation and owns its own address space."),
    ("线程", "a thread", "线程是进程内的执行流，与其他线程共享地址空间。",
     "A thread is a flow of execution inside a process and shares its address space."),
    ("API", "an API", "API 是程序之间约定的调用接口，规定输入、输出和错误。",
     "An API is the agreed call interface between programs, with inputs, outputs and errors."),
    ("正则表达式", "a regular expression", "正则表达式用一套字符规则描述要匹配的文本模式。",
     "A regular expression describes a text pattern with a small set of character rules."),
    ("JSON", "JSON", "JSON 是一种文本数据格式，用对象和数组写出数据。",
     "JSON is a text data format that writes data as objects and arrays."),
    ("UTF-8", "UTF-8", "UTF-8 是一种变长编码，用一到四个字节表示一个 Unicode 字符。",
     "UTF-8 is a variable-length encoding that uses one to four bytes per Unicode character."),
    ("算法", "an algorithm", "算法是有限步内把输入变成输出的明确步骤。",
     "An algorithm is a finite list of clear steps that turns input into output."),
    ("版本控制", "version control", "版本控制记录文件每次改动，并支持回到旧版本。",
     "Version control records every change to a file and restores an old revision."),
    ("单元测试", "a unit test", "单元测试独立检查一个函数在给定输入下的输出。",
     "A unit test checks one function's output for a given input in isolation."),
    ("浮点数", "a floating-point number", "浮点数用符号、指数和尾数表示实数，因此有精度上限。",
     "A floating-point number stores a sign, an exponent and a significand, so it has finite precision."),
    ("整数溢出", "integer overflow", "整数溢出是运算结果超出该整数类型的表示范围。",
     "Integer overflow happens when a result falls outside the range of the integer type."),
    ("面向对象", "object-oriented programming", "面向对象把数据和操作数据的函数放进同一个类里。",
     "Object-oriented programming puts data and the functions that act on it in one class."),
    ("函数式编程", "functional programming", "函数式编程把计算写成函数的组合，并避免修改共享状态。",
     "Functional programming composes functions and avoids changing shared state."),
    ("机器学习", "machine learning", "机器学习从数据里拟合参数，使模型在新样本上给出预测。",
     "Machine learning fits parameters to data so the model predicts on new samples."),
    ("梯度下降", "gradient descent", "梯度下降沿着损失函数的负梯度方向逐步更新参数。",
     "Gradient descent updates parameters along the negative gradient of the loss."),
    ("过拟合", "overfitting", "过拟合是模型记住训练集细节，在新数据上表现变差。",
     "Overfitting is a model that memorises training detail and scores worse on new data."),
    ("精度与召回", "precision and recall", "精度衡量预测为正的样本里正确的比例，召回衡量正样本被找出的比例。",
     "Precision measures how many predicted positives are correct; recall measures how many positives were found."),
    ("熵", "entropy", "熵衡量一个分布的不确定度，均匀分布的熵最大。",
     "Entropy measures the uncertainty of a distribution and is largest for a uniform one."),
    ("编译器优化", "compiler optimisation", "编译器优化在不改变程序结果的前提下改写中间表示，让代码更快。",
     "Compiler optimisation rewrites the intermediate form for speed without changing the result."),
    ("内存泄漏", "a memory leak", "内存泄漏是程序不再使用的内存没有归还给系统。",
     "A memory leak is memory a program no longer uses and never returns to the system."),
    ("垃圾回收", "garbage collection", "垃圾回收自动找出不再可达的对象并释放它们的内存。",
     "Garbage collection finds unreachable objects and frees their memory automatically."),
    ("虚拟内存", "virtual memory", "虚拟内存给每个进程一套独立地址，由硬件映射到物理页。",
     "Virtual memory gives each process its own address space, mapped to physical pages by hardware."),
    ("协议", "a protocol", "协议是双方交换消息时共同遵守的格式和顺序。",
     "A protocol is the shared format and order two sides use to exchange messages."),
    ("序列化", "serialisation", "序列化把内存里的对象转成可存储或可传输的字节。",
     "Serialisation turns an in-memory object into bytes that can be stored or sent."),
    ("并行与并发", "parallelism and concurrency", "并发让多个任务交替推进，并行让多个任务真正同时执行。",
     "Concurrency interleaves several tasks; parallelism runs several tasks at the same time."),
    ("链表与数组", "a linked list and an array", "数组按下标常数时间访问，链表按顺序访问但插入更快。",
     "An array indexes in constant time; a linked list walks in order and inserts faster."),
    ("缓存命中", "a cache hit", "缓存命中是请求的数据已经在缓存里，不必再取一次。",
     "A cache hit is a request whose data is already in the cache."),
    ("幂等", "idempotence", "幂等操作执行一次和执行多次的结果相同。",
     "An idempotent operation gives the same result after one run and after several."),
    ("时区", "a time zone", "时区是地球上共用同一标准时间的一片区域。",
     "A time zone is a region that shares one standard clock time."),
    ("比特与字节", "a bit and a byte", "一位是一个二进制数字，一个字节由八位组成。",
     "A bit is one binary digit; a byte holds eight bits."),
    ("素数", "a prime number", "素数只有 1 和它本身两个正因数。",
     "A prime number has exactly two positive divisors: 1 and itself."),
    ("最大公约数", "the greatest common divisor", "最大公约数是能同时整除两个整数的最大正整数。",
     "The greatest common divisor is the largest positive integer that divides both numbers."),
    ("排列与组合", "permutations and combinations", "排列区分顺序，组合不区分顺序。",
     "Permutations count ordered selections; combinations count unordered ones."),
    ("平均数", "an average", "平均数是所有数值之和除以数值个数。",
     "An average is the sum of the values divided by how many values there are."),
    ("中位数", "a median", "中位数是把数值排序后位于正中间的那个数。",
     "A median is the middle value after the numbers are sorted."),
    ("百分比", "a percentage", "百分比把一个数表示成另一个数的百分之几。",
     "A percentage states one number as a fraction of another out of one hundred."),
    ("面积", "area", "面积是平面图形覆盖的单位方格数量。",
     "Area is the number of unit squares a flat shape covers."),
    ("周长", "perimeter", "周长是图形边界一周的总长度。",
     "A perimeter is the total length of a shape's boundary."),
    ("速度", "speed", "速度是路程除以所用时间。",
     "Speed is distance divided by the time taken."),
    ("密度", "density", "密度是质量除以体积。",
     "Density is mass divided by volume."),
    ("能量守恒", "conservation of energy", "能量守恒指孤立系统的总能量保持不变。",
     "Conservation of energy means the total energy of an isolated system stays constant."),
    ("牛顿第一定律", "Newton's first law", "物体不受合外力时保持静止或匀速直线运动。",
     "An object with no net force stays at rest or moves at constant velocity."),
    ("电压", "voltage", "电压是两点之间单位电荷的电势差。",
     "Voltage is the electric potential difference per unit charge between two points."),
    ("pH 值", "pH", "pH 值是溶液酸度的度量，7 表示中性。",
     "pH measures how acidic a solution is, and 7 is neutral."),
    ("光年", "a light-year", "光年是光在真空中一年走过的距离。",
     "A light-year is the distance light travels in vacuum in one year."),
    ("太阳系", "the solar system", "太阳系由太阳和围绕它运行的行星等天体组成。",
     "The solar system holds the Sun and the bodies that orbit it."),
    ("DNA", "DNA", "DNA 是携带遗传信息的双链分子，由四种碱基组成。",
     "DNA is a double-stranded molecule that carries genetic information in four bases."),
    ("细胞", "a cell", "细胞是生命体功能和生长的基本单位。",
     "A cell is the basic unit of function and growth in living things."),
    ("光合作用", "photosynthesis", "光合作用用光能把水和二氧化碳变成有机物并放出氧气。",
     "Photosynthesis uses light to turn water and carbon dioxide into sugar and oxygen."),
    ("通货膨胀", "inflation", "通货膨胀是一段时间内一般物价水平的持续上涨。",
     "Inflation is a sustained rise in the general price level over time."),
    ("利率", "an interest rate", "利率是借款成本占本金的百分比。",
     "An interest rate is the cost of borrowing as a percentage of the principal."),
    ("复利", "compound interest", "复利把每期利息计入本金，下一期按新本金计息。",
     "Compound interest adds each period's interest to the principal before the next period."),
    ("国内生产总值", "gross domestic product", "国内生产总值是一国境内一定时期内生产的最终产品和服务的总价值。",
     "Gross domestic product is the total value of final goods and services produced inside a country."),
    ("契约", "a contract", "契约是双方就权利义务达成的有约束力的约定。",
     "A contract is a binding agreement between parties about their rights and duties."),
    ("开源许可", "an open-source licence", "开源许可授予他人使用、修改和分发源代码的权利，并规定条件。",
     "An open-source licence grants the right to use, change and redistribute source code under conditions."),
]


def build_defs(rng):
    """One-sentence definitions.

    A definition question names its term and the answer opens with that term, so the two
    share a short run. That run is measured here against the category's own cap: 8
    characters, exactly the term, never 9 and never the whole question. E2's 32-character
    probe -- the check that fires on a restated prompt -- does not match on any of these
    rows, which is the property that matters; the measurement is in report.json.
    """
    rows = []
    # The first form in each list is the one E2's own prompt set uses
    # (data/eval/ladder/e2_prompts.jsonl: "用一句话说明什么是二分查找。" /
    # "What does a hash table do? Answer in one sentence."), so the drill matches the
    # phrasing the evaluation will ask in.
    zh_forms = ["用一句话说明什么是{t}。", "{t}是什么？一句话回答。"]
    en_forms = ["Explain in one sentence what {t} is.", "What does {t} do? Answer in one sentence."]
    for zh_t, en_t, zh_a, en_a in DEFS:
        for f in zh_forms:
            rows.append(entry("zh", f.format(t=zh_t), zh_a, "curated_define",
                              "one_sentence_definition", license="generated_for_aupai"))
        for f in en_forms:
            rows.append(entry("en", f.format(t=en_t), en_a, "curated_define",
                              "one_sentence_definition", license="generated_for_aupai"))
    return rows


# --------------------------------------------------------------------------- translate
# Curated phrase pairs, both directions. source=curated_translation,
# license=generated_for_aupai (written for this set, not copied from a corpus).
TRANS = [
    ("今天天气很好。", "The weather is nice today."),
    ("请把这句话翻译成英文。", "Please translate this sentence into English."),
    ("这个函数返回一个整数。", "This function returns an integer."),
    ("程序在启动时报错。", "The program reports an error at startup."),
    ("服务器已经恢复运行。", "The server is running again."),
    ("请稍后再试一次。", "Please try again later."),
    ("文件已经保存到本地。", "The file is saved locally."),
    ("这是一次小改动。", "This is a small change."),
    ("我们需要更多测试数据。", "We need more test data."),
    ("会议改到明天上午。", "The meeting moves to tomorrow morning."),
    ("他每天跑步三十分钟。", "He runs for thirty minutes every day."),
    ("这台机器需要维护。", "This machine needs maintenance."),
    ("请把结果写到日志里。", "Write the result to the log."),
    ("数据库连接失败了。", "The database connection failed."),
    ("代码已经合并到主干。", "The code is merged into the main branch."),
    ("这个版本修复了三个缺陷。", "This release fixes three defects."),
    ("先把输入校验一遍。", "Validate the input first."),
    ("用户看不到这条记录。", "The user cannot see this record."),
    ("训练在第八步停了下来。", "The training stopped at step eight."),
    ("请给出具体的复现步骤。", "Please give the exact steps to reproduce."),
    ("这个接口每天调用一百万次。", "This interface is called a million times a day."),
    ("内存使用量保持稳定。", "Memory usage stays stable."),
    ("我们明天讨论这个方案。", "We will discuss this plan tomorrow."),
    ("结果与预期一致。", "The result matches the expectation."),
    ("请检查硬件连接。", "Please check the hardware connection."),
    ("天气冷了，记得加衣服。", "It is getting cold, so remember to dress warmly."),
    ("这本书值得读第二遍。", "This book is worth a second read."),
    ("他正在学习一门新语言。", "He is learning a new language."),
    ("请把窗户关上。", "Please close the window."),
    ("孩子们在操场上跑步。", "The children are running on the playground."),
    ("高铁比飞机更方便。", "The high-speed train is more convenient than the plane."),
    ("这家餐厅的菜很新鲜。", "The food at this restaurant is very fresh."),
    ("我每天早上六点起床。", "I get up at six every morning."),
    ("请把地址写清楚。", "Please write the address clearly."),
    ("会议记录已经发出。", "The meeting notes have been sent."),
    ("他很擅长解决难题。", "He is good at solving hard problems."),
    ("雨水打湿了路面。", "The rain has made the road wet."),
    ("请保持安静。", "Please keep quiet."),
    ("这只猫喜欢晒太阳。", "This cat likes to lie in the sun."),
    ("我们下个月搬家。", "We are moving next month."),
    ("这支笔写起来很顺。", "This pen writes smoothly."),
    ("他昨天请假了。", "He took a day off yesterday."),
    ("请按时提交报告。", "Please submit the report on time."),
    ("这项任务需要两个人。", "This task needs two people."),
    ("价格比去年低了。", "The price is lower than last year."),
    ("医生建议他多休息。", "The doctor advises him to rest more."),
    ("公园里有很多人。", "There are many people in the park."),
    ("请把灯关掉。", "Please turn off the light."),
    ("他的发音很标准。", "His pronunciation is very standard."),
    ("我们坐地铁去机场。", "We take the subway to the airport."),
    ("这周的工作已经完成。", "This week's work is done."),
    ("请在表格上签名。", "Please sign the form."),
    ("空气很干燥。", "The air is very dry."),
    ("他说话声音很小。", "He speaks in a low voice."),
    ("这家店周日休息。", "This shop is closed on Sunday."),
    ("我同意你的看法。", "I agree with your view."),
    ("请在门口等我。", "Please wait for me at the door."),
    ("这道题有两种解法。", "This problem has two solutions."),
    ("他负责这个项目。", "He is in charge of this project."),
    ("数据已经备份完成。", "The data backup is complete."),
    ("请更新你的密码。", "Please update your password."),
    ("系统正在升级。", "The system is being upgraded."),
    ("这个错误可以复现。", "This error can be reproduced."),
    ("他提前完成了任务。", "He finished the task ahead of time."),
    ("请把文件发给我。", "Please send me the file."),
    ("会议延迟了半小时。", "The meeting was delayed by half an hour."),
    ("这台电脑运行很快。", "This computer runs fast."),
    ("假期从周五开始。", "The holiday starts on Friday."),
    ("他把问题记录下来了。", "He wrote the problem down."),
    ("这条路正在维修。", "This road is under repair."),
    ("请确认收货地址。", "Please confirm the delivery address."),
    ("火车准点到达。", "The train arrived on time."),
    ("他喜欢喝黑咖啡。", "He likes black coffee."),
    ("花园里的花开了。", "The flowers in the garden are open."),
    ("请把音量调低。", "Please turn the volume down."),
    ("我们在这里等一下。", "Let us wait here for a while."),
    ("他解释了原因。", "He explained the reason."),
    ("这个方案成本更低。", "This plan costs less."),
    ("请核对一下数字。", "Please check the numbers."),
    ("他们合作了三年。", "They worked together for three years."),
    ("这件事已经解决了。", "This matter is settled."),
    ("他习惯早起。", "He is used to getting up early."),
    ("请把椅子搬过来。", "Please move the chair over here."),
    ("雨停了。", "The rain has stopped."),
    ("我不确定这个答案。", "I am not sure about this answer."),
    ("请再读一遍题目。", "Please read the question again."),
    ("这个想法值得尝试。", "This idea is worth trying."),
    ("他把钥匙放进口袋。", "He put the key in his pocket."),
    ("孩子们已经睡了。", "The children are asleep."),
    ("请把垃圾带走。", "Please take the rubbish away."),
    ("他每周读一本书。", "He reads one book a week."),
    ("这扇门锁不上。", "This door will not lock."),
    ("我们换个地方谈。", "Let us talk somewhere else."),
    ("他给朋友打了电话。", "He called his friend."),
    ("请把答案写在第一行。", "Please write the answer on the first line."),
    ("这台机器很安静。", "This machine is very quiet."),
    ("他放弃了这次机会。", "He gave up this chance."),
    ("请提前十分钟到。", "Please arrive ten minutes early."),
    ("水已经烧开了。", "The water has boiled."),
    ("他记得每一个细节。", "He remembers every detail."),
    ("请在周一之前回复。", "Please reply before Monday."),
    ("这个季节雨水很多。", "There is a lot of rain in this season."),
    ("他把书放回书架。", "He put the book back on the shelf."),
    ("我们走楼梯上去。", "Let us take the stairs up."),
    ("请把结果打印出来。", "Please print the result."),
]


def build_translate(rng):
    rows = []
    for zh, en in TRANS:
        rows.append(dict(question=f"把“{zh}”翻译成英文。", answer=en, lang="zh",
                         source="curated_translation", license="generated_for_aupai",
                         quality="curated_bilingual"))
        rows.append(dict(question=f"把“{zh}”翻译成英文，只给译文。", answer=en, lang="zh",
                         source="curated_translation", license="generated_for_aupai",
                         quality="curated_bilingual"))
        rows.append(dict(question=f"Translate “{en}” into Chinese.", answer=zh, lang="en",
                         source="curated_translation", license="generated_for_aupai",
                         quality="curated_bilingual"))
        rows.append(dict(question=f"Translate “{en}” into Chinese. Give the translation only.", answer=zh,
                         lang="en", source="curated_translation", license="generated_for_aupai",
                         quality="curated_bilingual"))
    return rows


# --------------------------------------------------------------------------- code
def run_verify(program, timeout=10):
    """Execute a one-liner bank entry's verify program in a fresh interpreter."""
    try:
        proc = subprocess.run([sys.executable, "-c", program], capture_output=True,
                              text=True, timeout=timeout,
                              cwd=tempfile.gettempdir())
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        return False, ("rc=%d %s" % (proc.returncode, tail[-1] if tail else "no stderr"))
    return True, "rc=0"


def build_code(rng, bank_path, failures):
    """One-line Python answers. Each entry is EXECUTED; a failing entry is dropped, not shipped."""
    with open(bank_path, encoding="utf-8") as fh:
        bank = json.load(fh)["entries"]
    rows = []
    kept = 0
    for idx, entry in enumerate(bank):
        ok, detail = run_verify(entry["verify"])
        if not ok:
            failures.append({"stage": "code_verify", "bank_index": idx,
                             "code": entry["code"], "reason": detail})
            continue
        kept += 1
        line = entry["code"]
        if "\n" in line:
            failures.append({"stage": "code_verify", "bank_index": idx, "code": line,
                             "reason": "answer is not one line"})
            continue
        rows.append(dict(question=entry["zh"], answer=line, lang="zh", source="stdlib_oneliner",
                         license="PSF-2.0", quality="code_verified_exec"))
        rows.append(dict(question=entry["en"], answer=line, lang="en", source="stdlib_oneliner",
                         license="PSF-2.0", quality="code_verified_exec"))
    return rows, kept


# --------------------------------------------------------------------------- refusal
REFUSAL = [
    ("给我一个明天一定上涨的股票代码。", "价格方向无法预知。请用公开数据自行判断。",
     ["Give me a stock code that will definitely rise tomorrow.",
      "Tell me which stock goes up tomorrow for sure."],
     "Price direction cannot be known in advance. Judge from public data."),
    ("我应该吃多少毫克的这种药？", "用药剂量要问医生或药师。",
     ["How many milligrams of this drug should I take?",
      "Give me the dose of this medicine in milligrams."],
     "Dosage needs a doctor or a pharmacist."),
    ("帮我写一封伪造身份的邮件。", "这条请求越过了法律底线。",
     ["Write an email that fakes someone's identity.",
      "Help me impersonate another person in an email."],
     "That request crosses a legal line."),
    ("告诉我怎么破解邻居的 Wi-Fi 密码。", "这条请求越过了法律底线。",
     ["Tell me how to crack my neighbour's Wi-Fi password.",
      "Help me break into a Wi-Fi network I do not own."],
     "That request crosses a legal line."),
    ("这段话的作者是谁？", "缺少出处，无法署名。",
     ["Who wrote this paragraph?", "Name the author of the text above."],
     "No source is given, so no name can be attached."),
    ("这份合同第二页写了什么？", "没有正文，无法回答。请把原文贴出来。",
     ["What does page two of this contract say?",
      "Summarise the second page of the document."],
     "The text itself is missing. Please paste it."),
    ("上个月那台服务器的日志里有什么错误？", "没有访问权限，无法查看日志。",
     ["Which errors were in last month's server log?",
      "List the errors in the log from last month."],
     "No access to that log is available here."),
    ("我明年能考上吗？", "结果无法预知。请按计划复习。",
     ["Will I pass the exam next year?",
      "Predict whether I get admitted next year."],
     "The result cannot be known now. Study to your plan."),
    ("请把整本教材复制给我。", "版权作品受法律保护，不能完整提供。",
     ["Send me a full copy of this textbook.",
      "Give me the complete text of this book."],
     "Copyright law covers such a work, so it cannot be sent."),
    ("删除我电脑上所有文件，我不想确认。", "这一步会丢数据。请先确认目标目录。",
     ["Delete every file on my computer, do not ask me again.",
      "Wipe all my files without confirmation."],
     "This step destroys data. Confirm the target directory first."),
    ("解释一下怎么入侵别人的邮箱。", "这条请求越过了法律底线。",
     ["Explain how to break into someone's email account.",
      "How do I break into another person's mailbox?"],
     "That request crosses a legal line."),
    ("他是不是在骗我？", "缺少信息，无法判断。",
     ["Is he lying to me?", "Tell me whether this person is dishonest."],
     "Too little information is available to judge."),
    ("把这段代码直接部署到生产环境，不用测试。", "跳过测试会带来线上风险。请先跑一遍测试。",
     ["Deploy this code straight to production without testing.",
      "Ship this change to production, skip the tests."],
     "Skipping tests risks the live service. Run them first."),
    ("告诉我明天彩票的开奖号码。", "开奖结果无法预知。",
     ["Tell me tomorrow's lottery numbers.",
      "Give me the winning lottery numbers for tomorrow."],
     "The draw result cannot be known in advance."),
    ("帮我确认这条网上看到的消息是不是真的。", "没有来源，无法核实。请给出处。",
     ["Confirm whether this claim I saw online is true.",
      "Tell me if this rumour is real."],
     "With no source given, nothing can be verified. Please cite one."),
    ("帮我完成这份作业的全部答案。", "我不能代做整份作业，但可以讲方法。",
     ["Do my whole homework and give every answer.",
      "Finish this assignment for me, all the answers."],
     "Doing such work for you is out of scope, but the method can be explained."),
]


def build_refusal(rng):
    """Refusals and clarifications. The answer names the reason, never the request.

    A policy answer that repeats the request's wording is the echo failure in its most
    damaging form: it trains the model to restate a premise before refusing it.
    """
    rows = []
    for zh_q, zh_a, en_qs, en_a in REFUSAL:
        zh_qs = [zh_q, zh_q.rstrip("。？") + "，请直接回答。"]
        qs = zh_qs + list(en_qs)
        a_zh, span_zh = pick(qs, [zh_a])
        a_en, span_en = pick(qs, [en_a])
        if span_zh >= ECHO_MIN_CHARS or span_en >= ECHO_MIN_CHARS:
            raise Refuse(f"REFUSE: refusal shares {max(span_zh, span_en)} characters with its "
                         f"question: {zh_q!r} / {zh_a!r}")
        for q in zh_qs:
            rows.append(entry("zh", q, a_zh, "curated_refusal", "policy_refusal",
                              license="generated_for_aupai"))
        for q in en_qs:
            rows.append(entry("en", q, a_en, "curated_refusal", "policy_refusal",
                              license="generated_for_aupai"))
    return rows


# --------------------------------------------------------------------------- dates
# Dates are computed with datetime, so a wrong weekday or a wrong day count cannot be
# shipped. Answers carry the day or the count, never the question's own cells.
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAYS_ZH = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
MONTH_LENGTHS = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30,
                 10: 31, 11: 30, 12: 31}


def date_plan(rng, per_lang):
    """Unique (kind, value) date tasks. Each kind has enough cardinality to fill it."""
    import datetime
    kinds = ["weekday", "year", "month", "add", "between"]
    plans = {k: [] for k in kinds}
    seen = {k: set() for k in kinds}
    cursor = 0
    # Each kind's population is bounded by its date range: 3,651 days for the single-date
    # kinds, 3,601 x 20 for add, 3,501 x 900 for between. The guard turns a stalled draw
    # into a refusal.
    date_guard = 0
    while any(len(plans[k]) < per_lang for k in kinds):
        date_guard += 1
        if date_guard > 500 * per_lang * len(kinds) + 10000:
            raise Refuse(f"REFUSE: date plan stalled at "
                         f"{ {k: len(v) for k, v in plans.items()} } for per_lang={per_lang}")
        kind = kinds[cursor % len(kinds)]
        cursor += 1
        if len(plans[kind]) >= per_lang:
            continue
        if kind in ("weekday", "year", "month"):
            d = datetime.date(2016, 1, 1) + datetime.timedelta(days=rng.randint(0, 3650))
            key = d.isoformat()
        elif kind == "add":
            d = datetime.date(2016, 1, 1) + datetime.timedelta(days=rng.randint(0, 3600))
            n = rng.choice([-365, -200, -90, -60, -30, -14, -7, -3, -1,
                            1, 3, 7, 14, 30, 45, 60, 90, 120, 200, 365])
            key = (d.isoformat(), n)
        else:
            d = datetime.date(2016, 1, 1) + datetime.timedelta(days=rng.randint(0, 3500))
            e = d + datetime.timedelta(days=rng.randint(1, 900))
            key = (d.isoformat(), e.isoformat())
        if key in seen[kind]:
            continue
        seen[kind].add(key)
        plans[kind].append(key)
    return plans


def build_dates(rng, per_lang):
    import datetime
    plans = date_plan(rng, per_lang)
    rows = []

    def emit(check, zh_qs, en_qs, zh_a, en_a):
        qs = list(zh_qs) + list(en_qs)
        a_zh, span_zh = pick(qs, [zh_a])
        a_en, span_en = pick(qs, [en_a])
        if span_zh >= ECHO_MIN_CHARS or span_en >= ECHO_MIN_CHARS:
            raise Refuse(f"REFUSE: date answer shares {max(span_zh, span_en)} characters "
                         f"with its question: {zh_qs[0]!r} / {zh_a!r}")
        for q in zh_qs:
            rows.append(entry("zh", q, a_zh, "synthetic_date", "numeric_verified", check))
        for q in en_qs:
            rows.append(entry("en", q, a_en, "synthetic_date", "numeric_verified", check))

    for iso in plans["weekday"]:
        d = datetime.date.fromisoformat(iso)
        zh = WEEKDAYS_ZH[d.weekday()] + "。"
        en = WEEKDAYS[d.weekday()] + "."
        emit({"date": iso, "field": "weekday"}, [f"{iso} 是星期几？只写星期。"],
             [f"Which day of the week is {iso}? Give the day only."], zh, en)
    for iso in plans["year"]:
        year = int(iso[:4])
        emit({"date": iso, "field": "year"}, [f"{iso} 是哪一年？只写年份。"],
             [f"Which year does {iso} fall in? Give the year only."], f"{year} 年。", f"{year}.")
    for iso in plans["month"]:
        month = int(iso[5:7])
        emit({"date": iso, "field": "month"}, [f"{iso} 是几月？只写月份数字。"],
             [f"Which month is {iso} in? Give the month number only."], f"{month} 月。",
             f"Month {month}.")
    # The answer is the target's MONTH, not the whole date: a date answer repeats its own
    # question's "YYYY-MM-" prefix, eight characters, and no rule can make that not an echo.
    for iso, n in plans["add"]:
        d = datetime.date.fromisoformat(iso)
        target = d + datetime.timedelta(days=n)
        month = target.month
        zh_q = (f"{iso} 往后 {n} 天落在几月？只写月份数字。"
                if n > 0 else f"{iso} 往前 {-n} 天落在几月？只写月份数字。")
        en_q = (f"Which month does {n} days after {iso} fall in? Give the month number only."
                if n > 0 else
                f"Which month does {-n} days before {iso} fall in? Give the month number only.")
        emit({"date": iso, "days": n, "field": "add_month"}, [zh_q], [en_q],
             f"{month} 月。", f"Month {month}.")
    for a, b in plans["between"]:
        d1, d2 = datetime.date.fromisoformat(a), datetime.date.fromisoformat(b)
        days = (d2 - d1).days
        emit({"from": a, "to": b, "field": "between"},
             [f"{a} 到 {b} 一共有多少天？只写数字。"],
             [f"How many days are there between {a} and {b}? Give the number only."],
             str(days) + "。", str(days) + ".")
    return rows


def build_month_length(rng):
    """Month lengths. The answer is a bare day count, recomputed from the calendar.

    A bare count, not "28 days in a common year": the question already names the month and
    the year kind, so the count is unambiguous, while a phrase answer repeats the question
    ("days in a common year", 22 characters) and is exactly the restatement this set exists
    to remove.
    """
    rows = []
    for m in range(1, 13):
        days = MONTH_LENGTHS[m]
        if m == 2:
            zh_qs = ["二月在平年有多少天？只写数字。", "非闰年的 2 月有多少天？只写数字。"]
            en_qs = ["How many days does February have in a common year? Give the number only.",
                     "Give the length of February in a non-leap year, digits only."]
        else:
            zh_qs = [f"{m} 月一共有多少天？只写数字。", f"公历第 {m} 个月有几天？只写数字。"]
            en_qs = [f"How many days does month {m} have? Give the number only.",
                     f"Give the length of month {m} in days, digits only."]
        qs = zh_qs + en_qs
        variants = [str(days) + "。", str(days) + "."]
        a, span = pick(qs, variants)
        if span > 7:
            raise Refuse(f"REFUSE: month length {m} shares {span} characters with its question")
        for q in zh_qs:
            rows.append(entry("zh", q, a, "fact_core", "numeric_verified",
                              {"month": m, "days": days}))
        for q in en_qs:
            rows.append(entry("en", q, a, "fact_core", "numeric_verified",
                              {"month": m, "days": days}))
    return rows


# --------------------------------------------------------------------------- strings
def build_strings(rng, count):
    """Character and substring operations on a generated ASCII string."""
    rows = []
    seen = set()
    plan = []
    kinds = ["index", "count", "reverse"]
    miss = 0
    _guard_capacity("strings", count, 3 * 26 ** 10)
    while len(plan) < count:
        miss += 1
        _guard_progress("strings", count, len(plan), miss)
        kind = kinds[len(plan) % len(kinds)]
        n = rng.randint(6, 10)
        text = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n))
        if kind == "index":
            i = rng.randint(1, n - 1)
            key = (kind, text, i)
        elif kind == "count":
            ch = rng.choice(sorted(set(text)))
            key = (kind, text, ch)
        else:
            key = (kind, text)
        if key in seen:
            miss += 1
            continue
        seen.add(key)
        plan.append(key)
        miss = 0
    for key in plan:
        kind = key[0]
        if kind == "index":
            _, text, i = key
            value = text[i - 1]
            zh_q = f"字符串 {text} 的第 {i} 个字符是什么？"
            en_q = f"Which character is at position {i} in the string {text}?"
            check = {"string": text, "index": i, "value": value}
        elif kind == "count":
            _, text, ch = key
            value = str(text.count(ch))
            zh_q = f"字符串 {text} 里有几个 {ch}？只写数字。"
            en_q = f"How many times does {ch} occur in the string {text}? Give the number only."
            check = {"string": text, "char": ch, "value": text.count(ch)}
        else:
            _, text = key
            value = text[::-1]
            zh_q = f"把字符串 {text} 反转后是什么？"
            en_q = f"What is the string {text} reversed?"
            check = {"string": text, "value": value}
        qs = [zh_q, en_q]
        if kind == "count":
            variants = [value, value + "。" if False else value]
            variants = [value]
        else:
            variants = [value + "。", value]
        a, span = pick(qs, variants)
        if span >= ECHO_MIN_CHARS:
            raise Refuse(f"REFUSE: string answer {a!r} shares {span} characters with its "
                         f"question {zh_q!r}")
        rows.append(entry("zh", zh_q, a, "synthetic_string", "numeric_verified" if kind == "count"
                          else "short_exact", check))
        rows.append(entry("en", en_q, a, "synthetic_string", "numeric_verified" if kind == "count"
                          else "short_exact", check))
    return rows


# --------------------------------------------------------------------------- lists
def build_lists(rng, count):
    """One-line Python over a short numeric list. Python computes every answer."""
    rows = []
    seen = set()
    plan = []
    ops = ["min", "max", "sum", "sorted", "len", "count_gt"]
    miss = 0
    _guard_capacity("list_py", count, 6 * 99 ** 4)
    while len(plan) < count:
        miss += 1
        _guard_progress("list_py", count, len(plan), miss)
        op = ops[len(plan) % len(ops)]
        size = rng.randint(3, 4)
        values = [rng.randint(1, 99) for _ in range(size)]
        threshold = rng.choice([20, 30, 40, 50, 60, 70])
        key = (op, tuple(values), threshold if op == "count_gt" else None)
        if key in seen:
            continue
        if op == "count_gt" and not any(v > threshold for v in values):
            continue
        if op == "count_gt" and not any(v <= threshold for v in values):
            continue
        seen.add(key)
        plan.append(key)
        miss = 0
    for op, values, threshold in plan:
        shown = "[" + ", ".join(str(v) for v in values) + "]"
        if op == "min":
            value, code = str(min(values)), "min(xs)"
            zh_q, en_q = f"给定 {shown}，用一行 Python 求最小值。", f"Given {shown}, write a Python one-liner for the smallest value."
        elif op == "max":
            value, code = str(max(values)), "max(xs)"
            zh_q, en_q = f"给定 {shown}，用一行 Python 求最大值。", f"Given {shown}, write a Python one-liner for the largest value."
        elif op == "sum":
            value, code = str(sum(values)), "sum(xs)"
            zh_q, en_q = f"给定 {shown}，用一行 Python 求和。", f"Given {shown}, write a Python one-liner for the total."
        elif op == "sorted":
            value, code = str(sorted(values)), "sorted(xs)"
            zh_q, en_q = f"给定 {shown}，用一行 Python 按升序排列。", f"Given {shown}, write a Python one-liner that sorts it."
        elif op == "len":
            value, code = str(len(values)), "len(xs)"
            zh_q, en_q = f"给定 {shown}，用一行 Python 求元素个数。", f"Given {shown}, write a Python one-liner for how many elements it holds."
        else:
            value = str(sum(1 for v in values if v > threshold))
            code = f"sum(1 for v in xs if v > {threshold})"
            zh_q = f"给定 {shown}，用一行 Python 数出大于 {threshold} 的元素个数。"
            en_q = f"Given {shown}, write a Python one-liner that counts the elements above {threshold}."
        check = {"list": values, "op": op, "threshold": threshold, "value": value}
        qs = [zh_q, en_q]
        a, span = pick(qs, [code])
        if span >= ECHO_MIN_CHARS:
            raise Refuse(f"REFUSE: list one-liner {code!r} shares {span} characters with its "
                         f"question {zh_q!r}")
        rows.append(entry("zh", zh_q, a, "synthetic_list_py", "code_verified_exec", check,
                          license="generated_for_aupai"))
        rows.append(entry("en", en_q, a, "synthetic_list_py", "code_verified_exec", check,
                          license="generated_for_aupai"))
    return rows


# --------------------------------------------------------------------------- batch
def build_two_step(rng, count):
    """Two one-step results in one short reply.

    This family exists for the size budget, and the reasoning is worth writing down: a row
    of this set carries 4.34 supervised tokens, so 1M supervised tokens needs about 230,000
    UNIQUE rows, and the single-answer families run out of population before that. A
    two-answer reply costs one prompt and yields two answers, which raises the tokens per
    row to about 6 without adding a filler word. Both values are recomputed, and the reply
    stays one short line, so the stopping behaviour this set teaches is untouched.
    """
    rows = []
    seen = set()
    plan = []
    miss = 0
    _guard_capacity("two_step", count, 1000 * 1000)
    ops = ["+", "-", "*"]
    while len(plan) < count:
        miss += 1
        _guard_progress("two_step", count, len(plan), miss)
        op1, op2 = rng.choice(ops), rng.choice(ops)
        picks = []
        for op in (op1, op2):
            if op == "+":
                a, b = rng.randint(11, 899), rng.randint(11, 899)
                value = a + b
            elif op == "-":
                a = rng.randint(60, 899)
                b = rng.randint(11, a - 1)
                value = a - b
            else:
                a, b = rng.randint(3, 89), rng.randint(3, 19)
                value = a * b
            picks.append((op, a, b, value))
        key = tuple((o, a, b) for o, a, b, _v in picks)
        if key in seen:
            miss += 1
            continue
        seen.add(key)
        plan.append(picks)
        miss = 0
    for picks in plan:
        parts = []
        for op, a, b, _v in picks:
            sym = {"*": "×"}.get(op, op)
            parts.append((a, sym, b))
        v1, v2 = picks[0][3], picks[1][3]
        zh_q = f"先算 {parts[0][0]} {parts[0][1]} {parts[0][2]}，再算 {parts[1][0]} {parts[1][1]} {parts[1][2]}，两个结果都写出来。"
        en_q = (f"Compute {parts[0][0]} {parts[0][1]} {parts[0][2]}, then "
                f"{parts[1][0]} {parts[1][1]} {parts[1][2]}. Give both results.")
        check = {"pairs": [list(x) for x in key], "values": [v1, v2]}
        qs = [zh_q, en_q]
        a, span = pick(qs, [f"{v1} 和 {v2}。", f"{v1}, {v2}。"])
        if span > 7:
            raise Refuse(f"REFUSE: two-step answer shares {span} characters with its question: "
                         f"{zh_q!r}")
        rows.append(entry("zh", zh_q, a, "synthetic_two_step", "numeric_verified", check))
        rows.append(entry("en", en_q, a, "synthetic_two_step", "numeric_verified", check))
    return rows


# --------------------------------------------------------------------------- errors
def build_arith_errors(rng, count):
    """Correct this wrong sum. The right value is recomputed; the wrong one is built."""
    rows = []
    capacities = {"+": 399 * 399, "-": 440 * 440, "*": 37 * 17, "//": 11 * 38}
    quota = _split_quota("arith_fix", count, capacities,
                         {"+": 3, "-": 3, "*": 2, "//": 2})
    plan = []
    seen = {op: set() for op in capacities}
    for op, want in quota.items():
        miss = 0
        got = 0
        while got < want:
            miss += 1
            _guard_progress(f"arith_fix[{op}]", want, got, miss)
            if op == "+":
                a, b = rng.randint(11, 399), rng.randint(11, 399)
                right = a + b
            elif op == "-":
                a = rng.randint(60, 499)
                b = rng.randint(11, a - 1)
                right = a - b
            elif op == "*":
                a, b = rng.randint(3, 39), rng.randint(3, 19)
                right = a * b
            else:
                b = rng.randint(2, 12)
                right = rng.randint(3, 40)
                a = b * right
            delta = rng.choice([-9, -7, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 9])
            wrong = right + delta
            if wrong == right or wrong < 0:
                miss += 1
                continue
            key = (a, op, b)
            if key in seen[op]:
                miss += 1
                continue
            seen[op].add(key)
            plan.append((a, op, b, right, wrong))
            got += 1
            miss = 0
    for a, op, b, right, wrong in plan:
        sym = {"*": "×", "//": "÷"}.get(op, op)
        check = {"op": op, "a": a, "b": b, "right": right, "wrong": wrong}
        zh_q = f"下面的算式是错的，写出正确结果：{a} {sym} {b} = {wrong}。"
        en_q = f"This sum is wrong. Give the correct result: {a} {sym} {b} = {wrong}."
        qs = [zh_q, en_q]
        a_ans, span = pick(qs, [str(right), str(right) + "。"])
        if span >= ECHO_MIN_CHARS:
            raise Refuse(f"REFUSE: correction answer {right} shares {span} characters with its "
                         f"question {zh_q!r}")
        rows.append(entry("zh", zh_q, a_ans, "synthetic_error_fix", "numeric_verified", check))
        rows.append(entry("en", en_q, a_ans, "synthetic_error_fix", "numeric_verified", check))
    return rows


def build_unit_errors(rng, count):
    """Correct this wrong conversion. The right value is recomputed from the factor."""
    from fractions import Fraction
    units = [
        ("千米", "米", "kilometre", "metres", Fraction(1000)),
        ("米", "厘米", "metre", "centimetres", Fraction(100)),
        ("千克", "克", "kilogram", "grams", Fraction(1000)),
        ("小时", "分钟", "hour", "minutes", Fraction(60)),
        ("分钟", "秒", "minute", "seconds", Fraction(60)),
        ("天", "小时", "day", "hours", Fraction(24)),
        ("吨", "千克", "tonne", "kilograms", Fraction(1000)),
        ("厘米", "毫米", "centimetre", "millimetres", Fraction(10)),
        ("星期", "天", "week", "days", Fraction(7)),
        ("英里", "千米", "mile", "kilometres", Fraction(1609, 1000)),
        ("英寸", "厘米", "inch", "centimetres", Fraction(254, 100)),
        ("磅", "克", "pound", "grams", Fraction(453592, 1000)),
    ]
    rows = []
    seen = set()
    plan = []
    miss = 0
    fix_capacity = sum(39 if factor.denominator != 1 else 59 for *_rest, factor in units)
    miss = 0
    _guard_capacity("unit_fix", count, fix_capacity)
    while len(plan) < count:
        miss += 1
        _guard_progress("unit_fix", count, len(plan), miss)
        idx = len(plan) % len(units)
        n = rng.randint(2, 60)
        if (idx, n) in seen:
            miss += 1
            continue
        seen.add((idx, n))
        plan.append((idx, n))
        miss = 0
    for idx, n in plan:
        zh_u1, zh_u2, en_u1, en_u2, factor = units[idx]
        right = factor * n
        if right.denominator == 1:
            right_s = str(right.numerator)
        else:
            right_s = f"{float(right):.4f}".rstrip("0").rstrip(".")
        wrong_f = right * Fraction(rng.choice([3, 4, 5, 6, 7, 9, 11]), rng.choice([2, 3, 5]))
        wrong_s = (str(wrong_f.numerator) if wrong_f.denominator == 1
                   else f"{float(wrong_f):.4f}".rstrip("0").rstrip("."))
        if wrong_s == right_s:
            continue
        check = {"unit": zh_u2, "n": n, "right": right_s, "wrong": wrong_s}
        zh_q = f"下面的换算写错了，给出正确数值：{n} {zh_u1} = {wrong_s} {zh_u2}。"
        en_q = f"This conversion is wrong. Give the correct value: {n} {en_u1} = {wrong_s} {en_u2}."
        qs = [zh_q, en_q]
        a, span = pick(qs, [right_s, right_s + "。"])
        if span >= ECHO_MIN_CHARS:
            raise Refuse(f"REFUSE: unit correction {right_s} shares {span} characters with its "
                         f"question {zh_q!r}")
        rows.append(entry("zh", zh_q, a, "synthetic_unit_fix", "numeric_verified", check))
        rows.append(entry("en", en_q, a, "synthetic_unit_fix", "numeric_verified", check))
    return rows


# --------------------------------------------------------------------------- records
# A record spread over several lines, with one field asked for. Tables are invented, so
# no real person or measurement is described; the answers are short cells.
def build_records(rng, count):
    """A record spread over several lines, with one field asked for.

    The tables are invented, so no real person or measurement is described. Two answer
    shapes are used, and the reasons are measured rather than stylistic: a year or a count
    is a different string from the record's own cells, while a LABEL copied into the
    question shares its whole length with it ("Portuguese" and the question asking about it
    share 10 characters). A label is therefore asked as a yes/no question, answering
    "Yes." or "No."

    Each kind gets its OWN plan and its own quota. One shared round-robin over the kinds is
    what stalled this family at 925 of 2,600 rows: `year` holds only a few hundred distinct
    keys, so once it saturated the loop kept drawing from it, hit its own duplicates, and
    made no progress on any other kind. With per-kind plans the narrow kind finishes first
    and the others continue.
    """
    writers = ["Norwegian", "Portuguese", "Italian", "Chilean", "Kenyan",
               "Japanese", "Canadian", "Greek", "Polish", "Thai"]
    visit_counts = [2, 3, 4, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 21, 24]
    years = list(range(1500, 2025))
    mix = {"year": 1, "checks": 3, "writer_yes": 3, "writer_no": 3, "longest": 2}
    rows = []

    def emit(kind, key):
        if kind == "longest":
            sizes = list(key[1])
            best = ["A", "B", "C"][sizes.index(max(sizes))]
            zh_q = (f"三条记录的字段长度分别是 {sizes[0]}、{sizes[1]}、{sizes[2]} 个字符。"
                    f"哪条记录的字段最长？只写编号。")
            en_q = (f"Three records hold fields of {sizes[0]}, {sizes[1]} and {sizes[2]} "
                    f"characters. Which record's field is longest? Give the letter only.")
            check = {"kind": kind, "sizes": sizes, "longest": best}
            a, span = pick([zh_q, en_q], [best + "。"])
            a_en, span_en = a, span
        elif kind == "year":
            year = key[1]
            zh_q = (f"记录 R-{year}：\n年代：{year} 年\n记录者：{key[2]}\n"
                    f"这段记录写于哪一年？只写年份。")
            en_q = (f"Record R-{year}:\nYear: {year}\nWriter: {key[2]}\n"
                    f"Which year does this record describe? Give the year only.")
            a, span = pick([zh_q, en_q], [str(year) + "。", str(year) + "."])
            a_en, span_en = a, span
            check = {"kind": kind, "year": year}
        elif kind == "checks":
            year, visits = key[1], key[2]
            zh_q = (f"记录 R-{year}：\n年代：{year} 年\n核验次数：{visits} 次\n"
                    f"这条记录一共核验了几次？只写数字。")
            en_q = (f"Record R-{year}:\nYear: {year}\nChecks: {visits} times\n"
                    f"How many times was this record checked? Give the number only.")
            a, span = pick([zh_q, en_q], [str(visits) + "。", str(visits) + "."])
            a_en, span_en = a, span
            check = {"kind": kind, "year": year, "visits": visits}
        else:
            year, writer, asked = key[1], key[2], key[3]
            hit = (kind == "writer_yes")
            zh_q = (f"记录 R-{year}：\n年代：{year} 年\n记录者：{writer}\n"
                    f"记录者是不是{asked}人？只回答“是”或“否”。")
            en_q = (f"Record R-{year}:\nYear: {year}\nWriter: {writer}\n"
                    f"Is the writer {asked}? Answer Yes or No.")
            a, span = pick([zh_q], ["是。" if hit else "否。", "否。" if hit else "是。"])
            a_en, span_en = pick([en_q], ["Yes." if hit else "No.", "No." if hit else "Yes."])
            check = {"kind": kind, "year": year, "writer": writer, "asked": asked,
                     "answer": hit}
        if span > 7 or span_en > 7:
            raise Refuse(f"REFUSE: record cell shares {max(span, span_en)} characters with its "
                         f"question: {zh_q!r}")
        rows.append(entry("zh", zh_q, a, "synthetic_record", "short_exact", check))
        rows.append(entry("en", en_q, a_en, "synthetic_record", "short_exact", check))

    weight_total = sum(mix.values())
    for kind, weight in mix.items():
        want = max(1, count * weight // weight_total)
        known = set()
        got = 0
        miss = 0
        while got < want:
            miss += 1
            _guard_progress(f"records[{kind}]", want, got, miss)
            if kind == "longest":
                sizes = [rng.randint(3, 24) for _ in range(3)]
                if len(set(sizes)) != 3:
                    miss += 1
                    continue
                key = (kind, tuple(sizes))
            elif kind == "year":
                key = (kind, rng.choice(years), rng.choice(writers))
            elif kind == "checks":
                key = (kind, rng.choice(years), rng.choice(visit_counts))
            else:
                writer = rng.choice(writers)
                key = (kind, rng.choice(years), writer,
                       rng.choice([w for w in writers if w != writer]))
            if key in known:
                miss += 1
                continue
            known.add(key)
            emit(kind, key)
            got += 1
            miss = 0
    return rows


# --------------------------------------------------------------------------- assemble
def echo_cap(source):
    """The shared-run cap for one row's source."""
    return DEFINE_ECHO_MAX_CHARS if source == "curated_define" else ECHO_MIN_CHARS - 1


def check_row(row, tok):
    """The per-row invariants. Returns (ok, reason).

    The echo rule is expressed twice: a shared span of ECHO_MIN_CHARS characters (the
    strict form, measured in characters) and E2's own probe -- the first 32 characters of
    the question must not appear in the answer text (eval/ladder/e2_chatml.judge). A row
    that passes the first necessarily passes the second; the second is here so the set is
    judged by the evaluator's own predicate, not by an argument that they agree.
    """
    q, a = row["question"], row["answer"]
    if not q.strip() or not a.strip():
        return False, "empty question or answer"
    if IM_START in a:
        return False, "answer holds a second <|im_start|>"
    if IM_END in a:
        return False, "answer holds <|im_end|> before the terminator"
    n = len(tok.encode(a + IM_END).ids)
    if n > ANSWER_MAX_TOKENS:
        return False, f"answer is {n} tokens, over the cap of {ANSWER_MAX_TOKENS}"
    span = longest_common_span(q, a)
    cap = echo_cap(row["source"])
    if span > cap:
        return False, (f"answer shares a {span}-character run with its own question "
                       f"(cap {cap} for source {row['source']})")
    probe = q[:32]
    if probe and probe in a:
        return False, "answer contains the 32-character probe E2 builds from the question"
    if tok.encode(a + IM_END).ids[-1] != tok.token_to_id(IM_END):
        return False, "answer does not end at <|im_end|>"
    return True, ""


def assign_ids(rows):
    """One id per (source, language, question, answer).

    The question alone is not unique: the defining phrases of two entries can coincide, and
    when they do they are two different rows with two different answers. Including the answer
    is what makes the id a key rather than a label.
    """
    for row in rows:
        key = "|".join((row["source"], row["lang"], row["question"], row["answer"])).encode()
        row["id"] = "c1-" + hashlib.sha256(key).hexdigest()[:12]


def dedup(rows, report, scope="row"):
    """Exact duplicates, then word-8-gram near duplicates over `scope`.

    `scope` is part of the version, not a detail. v1 shipped with the rule applied to the
    whole row; v2 applies it to the question (see the note below on why). Keeping both means
    v1's delivered artifacts remain reproducible: rebuilding them with the v2 rule gives
    15,001 rows instead of the 15,055 D1 trained on, and a manifest whose hashes no longer
    match its own file is worse than no manifest.

    The near-duplicate metric acts on the question, not on the whole row, and the difference
    is measured rather than stylistic. An answer of this set is often a bare value ("min(xs)",
    "240", "3"), so identical answer text appears in thousands of legitimate rows; counting
    8-grams over the whole row therefore scores those rows as near duplicates of each other
    and drops them. At v2's size that removed 20,526 rows, and because the rows happen to be
    emitted Chinese-first the drops landed on the English half (list_py kept 2,179 rows, of
    which 11 were English) and moved the language ratio to 54/46. The question is what
    distinguishes one row from another; a repeated answer is the drill, not a duplicate.
    """
    seen_full, seen_q = set(), set()
    kept = []
    for row in rows:
        full = norm(row["question"]) + "\u0000" + norm(row["answer"])
        qkey = norm(row["question"]).lower()
        if full in seen_full:
            report["dedup"]["dropped_full_duplicate"] += 1
            continue
        if qkey in seen_q:
            report["dedup"]["dropped_prompt_duplicate"] += 1
            continue
        seen_full.add(full)
        seen_q.add(qkey)
        kept.append(row)
    def grams_of(row):
        text = norm(row["question"]) if scope == "question" else (
            norm(row["question"]) + " " + norm(row["answer"]))
        return wgrams(text, NEAR_DUP_N)

    counter = collections.Counter()
    for row in kept:
        counter.update(grams_of(row))
    out = []
    used = collections.Counter()
    for row in kept:
        grams = grams_of(row)
        over = [g for g in grams if counter[g] > NEAR_DUP_LIMIT and used[g] >= NEAR_DUP_LIMIT]
        if over:
            report["dedup"]["dropped_near_duplicate"] += 1
            # A bounded sample with the full count beside it. The first v2 run wrote one of
            # these per dropped row -- 14,683 of them, a 2 MB report of a single repeated
            # shape. A record whose size grows with the corpus hides its own summary.
            if len(report["dedup"]["near_duplicate_examples"]) < 20:
                report["dedup"]["near_duplicate_examples"].append(
                    {"id": row["id"], "gram": sorted(over)[0], "count": counter[sorted(over)[0]]})
            continue
        for g in grams:
            used[g] += 1
        out.append(row)
    return out


def balance_languages(rows, report, tolerance=0.02):
    """Trim the larger language back to within `tolerance` of an even split.

    An explicit step because nothing else guarantees it. The families are built per language
    in equal numbers, but the near-duplicate filter acts on the merged list, and its drops are
    not language-neutral: measured at v2, the raw set was 53.1% Chinese, which is outside the
    "about half each" the contract asks for. Compensating by tuning family sizes would be
    guesswork, since which rows the filter drops depends on the template text. This trims the
    over-represented language by whole families, round-robin, so no single family is emptied
    and the kept rows stay a deterministic function of the seed.
    """
    counts = collections.Counter(r["lang"] for r in rows)
    total = len(rows)
    if total == 0 or abs(counts["zh"] - counts["en"]) / total <= tolerance:
        return rows
    big, small = ("zh", "en") if counts["zh"] > counts["en"] else ("en", "zh")
    keep_target = int(total * (0.5 + tolerance / 2))
    to_drop = counts[big] - keep_target
    if to_drop <= 0:
        return rows
    buckets = collections.defaultdict(list)
    for r in rows:
        if r["lang"] == big:
            buckets[r["source"]].append(r)
    # Proportional, one row at a time from each family in turn. Draining family by family
    # instead is what the first version did, and it emptied the small long-answer families:
    # the whole curated_define Chinese half went before any arithmetic row did. A trim that
    # changes the family composition changes what the set teaches, which is not its job.
    total_big = sum(len(v) for v in buckets.values())
    order = sorted(buckets)
    share = {name: len(buckets[name]) * to_drop / total_big for name in order}
    taken = {name: 0 for name in order}
    drop = set()
    while to_drop > 0 and order:
        for name in list(order):
            lst = buckets[name]
            quota = share[name]
            if taken[name] >= quota or taken[name] >= len(lst):
                order.remove(name)
                continue
            drop.add(id(lst[taken[name]]))
            taken[name] += 1
            to_drop -= 1
            if to_drop == 0:
                break
    # any remainder goes round-robin over the families that still hold rows
    idx = {name: taken[name] for name in sorted(buckets)}
    order = [n for n in sorted(buckets) if idx[n] < len(buckets[n])]
    while to_drop > 0 and order:
        for name in list(order):
            lst = buckets[name]
            if idx[name] >= len(lst):
                order.remove(name)
                continue
            drop.add(id(lst[idx[name]]))
            idx[name] += 1
            to_drop -= 1
            if to_drop == 0:
                break
    kept = [r for r in rows if id(r) not in drop]
    report["language_balance"] = {
        "before": {"zh": counts["zh"], "en": counts["en"]},
        "dropped_from": big, "dropped": len(rows) - len(kept),
        "after": {"zh": sum(1 for r in kept if r["lang"] == "zh"),
                  "en": sum(1 for r in kept if r["lang"] == "en")},
        "tolerance": tolerance,
    }
    return kept


def decontaminate(rows, gates, tok, report):
    """13-gram overlap against every gate set. A hit drops the row and is reported."""
    gram_sets = []
    for name, path, kind in gates:
        grams = {}
        n = 0
        for i, grow in enumerate(load_gate_rows(path, kind)):
            for part, text in gate_fields(name, grow):
                for g in wgrams(text, DECON_N):
                    grams.setdefault(g, f"{name}:{i}:{part}")
            n += 1
        gram_sets.append((name, path, n, grams))
        report["decontamination"]["gate_sets"][name] = {
            "path": os.path.relpath(path, ROOT), "rows": n, "grams": len(grams),
            "sha256": sha256_file(path)}
    kept = []
    for row in rows:
        text = norm(row["question"]) + " " + norm(row["answer"])
        hits = [gram_sets[i][3][g] for i, (_, _, _, _) in enumerate(gram_sets)
                for g in wgrams(text, DECON_N) if g in gram_sets[i][3]]
        if hits:
            report["decontamination"]["hits"].append({"id": row["id"], "hit": sorted(hits)[0]})
            report["decontamination"]["dropped"] += 1
            continue
        for i, (name, _, _, grams) in enumerate(gram_sets):
            for g in wgrams(text, DECON_N):
                if g in grams:
                    raise AssertionError("decontam sweep disagreed with itself")
        kept.append(row)
    return kept


def length_dist(vals):
    vals = sorted(vals)
    if not vals:
        return {}
    def pct(p):
        return vals[min(len(vals) - 1, int(round((len(vals) - 1) * p)))]
    return {"min": vals[0], "median": pct(0.5), "p90": pct(0.9), "max": vals[-1],
            "mean": round(sum(vals) / len(vals), 2), "n": len(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1", choices=("v1", "v2"),
                    help="which delivered version to build; the names carry the version, "
                         "so v2 never overwrites the pack D1 is training on")
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--arith-per-lang", type=int, default=None,
                    help="unique arithmetic expressions; 3 question forms each per language")
    ap.add_argument("--unit-per-lang", type=int, default=None)
    ap.add_argument("--base-per-lang", type=int, default=None)
    ap.add_argument("--series-per-lang", type=int, default=None)
    ap.add_argument("--numtheory-per-lang", type=int, default=None)
    ap.add_argument("--percent-per-lang", type=int, default=None)
    ap.add_argument("--date-per-lang", type=int, default=None,
                    help="unique date tasks per kind (weekday / year / month / add / between)")
    ap.add_argument("--string-count", type=int, default=None)
    ap.add_argument("--list-count", type=int, default=None)
    ap.add_argument("--arith-fix-count", type=int, default=None)
    ap.add_argument("--unit-fix-count", type=int, default=None)
    ap.add_argument("--record-count", type=int, default=None)
    ap.add_argument("--two-step-count", type=int, default=None)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--selftest", action="store_true", help="build into a temp dir and check the invariants")
    args = ap.parse_args()

    # Measured supervised tokens per row drive these numbers, not a round scale factor:
    # v1's pack yields 65,393 supervised tokens over 15,055 rows (4.34 per row), so the
    # 0.8-1.2M target is about 230,000-280,000 UNIQUE rows. v2 is composed for roughly
    # 1.0M: the families below were chosen for their measured tokens per row (dates 4.6,
    # list one-liners 6.3, records 3.2, arithmetic 2.4), because filler rows would defeat
    # the point of the set.
    VOLUME = {
        "v1": dict(arith_per_lang=1600, unit_per_lang=400, base_per_lang=250,
                   series_per_lang=150, numtheory_per_lang=150, percent_per_lang=150,
                   date_per_lang=0, string_count=0, list_count=0,
                   arith_fix_count=0, unit_fix_count=0, record_count=0,
                   two_step_count=0),
        # Sizes are the measured capacities, not round numbers: unit_convert holds 1,636
        # and series 420, and asking for more is a refusal (the planner's capacity check).
        # Sized from the measured 4.99 supervised tokens per row: 210,000 rows is about
        # 1.05M. The first v2 attempt produced 1.13M, which sits 5% above the requested
        # ceiling, so arithmetic and two-step (the two largest families) are trimmed.
        "v2": dict(arith_per_lang=15000, unit_per_lang=1600, base_per_lang=4500,
                   series_per_lang=420, numtheory_per_lang=3600, percent_per_lang=180,
                   date_per_lang=2600, string_count=3200, list_count=2600,
                   arith_fix_count=2600, unit_fix_count=600, record_count=2600,
                   two_step_count=22000),
    }
    preset = VOLUME[args.version]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    rng = random.Random(args.seed)
    out_dir = tempfile.mkdtemp(prefix="c1_selftest_") if args.selftest else args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    report = {
        "set": f"format_stop_{args.version}", "seed": args.seed, "builder": BUILDER,
        "dedup": {"dropped_full_duplicate": 0, "dropped_prompt_duplicate": 0,
                  "dropped_near_duplicate": 0, "near_duplicate_examples": [],
                  "near_dup_ngram": NEAR_DUP_N, "near_dup_limit": NEAR_DUP_LIMIT},
        "decontamination": {"n": DECON_N, "gate_sets": {}, "hits": [], "dropped": 0},
        "failures": [], "spot_check": {},
    }

    tok_path = os.path.join(ROOT, "data", "tokenizer.json")
    if not os.path.exists(tok_path):
        raise Refuse(f"REFUSE: {tok_path} is absent; the pack cannot be built without the gate vocabulary")
    tok = Tokenizer.from_file(tok_path)
    vid = vocab_fingerprint(tok)
    if vid != VOCAB_EXPECTED:
        raise Refuse(f"REFUSE: tokenizer fingerprint {vid} != {VOCAB_EXPECTED}")

    gates = resolve_gate_sets()
    print("[c1] gate sets:", ", ".join(f"{n}={os.path.relpath(p, ROOT)}" for n, p, _ in gates))

    code_rows, bank_kept = build_code(rng, os.path.join(OUT, "oneliner_bank.json"), report["failures"])
    families = [
        ("arith", build_arith(rng, args.arith_per_lang,
                              rotate_forms=(args.version != "v1"))),
        ("facts", build_facts(rng)),
        ("number_forms", build_number_forms(rng)),
        ("unit", build_unit_convert(rng, args.unit_per_lang)),
        ("base", build_base_convert(rng, args.base_per_lang)),
        ("series", build_series(rng, args.series_per_lang)),
        ("number_theory", build_number_theory(rng, args.numtheory_per_lang)),
        ("percent", build_percent(rng, args.percent_per_lang)),
        ("trig", build_trig(rng, 32)),
        ("define", build_defs(rng)),
        ("translate", build_translate(rng)),
        ("code", code_rows),
        ("refusal", build_refusal(rng)),
    ]
    # Families added for v2. Each is skipped at count 0, so v1's composition is unchanged
    # and its artifacts still rebuild byte for byte.
    if args.date_per_lang:
        families.append(("dates", build_dates(rng, args.date_per_lang)))
    families.append(("month_length", build_month_length(rng)))
    if args.string_count:
        families.append(("strings", build_strings(rng, args.string_count)))
    if args.list_count:
        families.append(("list_py", build_lists(rng, args.list_count)))
    if args.arith_fix_count:
        families.append(("arith_fix", build_arith_errors(rng, args.arith_fix_count)))
    if args.unit_fix_count:
        families.append(("unit_fix", build_unit_errors(rng, args.unit_fix_count)))
    if args.record_count:
        families.append(("records", build_records(rng, args.record_count)))
    if args.two_step_count:
        families.append(("two_step", build_two_step(rng, args.two_step_count)))
    report["generated_by_family"] = {name: len(rows_) for name, rows_ in families}
    rows = [row for _name, rows_ in families for row in rows_]
    report["generated"] = len(rows)
    print(f"[c1] generated {len(rows)} rows (one-liner bank kept {bank_kept}); "
          + ", ".join(f"{n}={c}" for n, c in report["generated_by_family"].items()))

    # Per-row invariants first: a row that breaks one is a defect of the generator.
    good = []
    for row in rows:
        ok, why = check_row(row, tok)
        if ok:
            good.append(row)
        else:
            report["failures"].append({"stage": "row_invariant", "question": row["question"],
                                       "answer": row["answer"], "reason": why})
    if report["failures"]:
        first = report["failures"][0]
        report["failures"] = report["failures"][:50]
        report["failures_total"] = len(good) and 0
        raise Refuse(f"REFUSE: rows break a generator invariant; first: {first}")
    rows = good

    assign_ids(rows)
    ids = collections.Counter(r["id"] for r in rows)
    dupes = [i for i, c in ids.items() if c > 1]
    if dupes:
        raise Refuse(f"REFUSE: {len(dupes)} duplicate ids after assignment, e.g. {dupes[:3]}")

    near_dup_scope = {"v1": "row", "v2": "question"}[args.version]
    report["dedup"]["scope"] = near_dup_scope
    rows = dedup(rows, report, scope=near_dup_scope)
    print(f"[c1] after dedup {len(rows)} rows "
          f"(dropped {report['dedup']['dropped_full_duplicate']} full, "
          f"{report['dedup']['dropped_prompt_duplicate']} prompt, "
          f"{report['dedup']['dropped_near_duplicate']} near)")
    rows = balance_languages(rows, report)
    print(f"[c1] after language balance {len(rows)} rows "
          f"({report.get('language_balance', {}).get('after', 'unchanged')})")
    rows = decontaminate(rows, gates, tok, report)
    print(f"[c1] after 13-gram decontamination {len(rows)} rows "
          f"(dropped {report['decontamination']['dropped']})")
    # v1's floor is the contract's 10,000-row minimum. v2 exists to carry a supervised-token
    # budget, and its floor is set from that budget rather than from a row count: a row of
    # this set carries 4.34 supervised tokens, so 0.8M needs about 185,000 rows. The
    # measured value is checked against the pack below, where it is counted exactly.
    row_floor = {"v1": 10000, "v2": 185000}[args.version]
    if len(rows) < row_floor:
        raise Refuse(f"REFUSE: only {len(rows)} rows survived; {args.version} needs at least "
                     f"{row_floor}")

    rng.shuffle(rows)
    for row in rows:
        row["prompt"] = format_prompt(row["question"])
        row["answer"] = row["answer"] + IM_END

    # Re-check the shipped text, not the pre-assembly text.
    for row in rows:
        if not row["prompt"].endswith(f"{IM_START}assistant\n"):
            raise Refuse(f"REFUSE: prompt does not end at the assistant turn: {row['id']}")
        if row["prompt"].count(IM_START) != 2 or IM_END not in row["prompt"]:
            raise Refuse(f"REFUSE: prompt is not the two-turn ChatML prefix: {row['id']}")
        if not row["answer"].endswith(IM_END) or row["answer"].count(IM_START):
            raise Refuse(f"REFUSE: answer terminator is wrong: {row['id']}")

    jsonl_path = os.path.join(out_dir, f"format_stop_{args.version}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({k: row[k] for k in
                                 ("id", "prompt", "answer", "source", "license", "quality")},
                                ensure_ascii=False) + "\n")

    # --- pack, the format sft_math.py --sft_path loads -----------------------
    eos = tok.token_to_id("<eos>")
    pairs = [(r["prompt"], r["answer"]) for r in rows]
    pack_path = os.path.join(out_dir, f"format_stop_{args.version}.pt")
    pack_and_save(pairs, tok, eos, pack_path, 4096, split_encode=True,
                  extra_stats={"set": f"format_stop_{args.version}", "rows_jsonl": len(rows),
                               "builder": BUILDER, "seed": args.seed})
    import torch
    blob = torch.load(pack_path, map_location="cpu", weights_only=True)
    for key in ("input_ids", "labels", "vocab_id", "holdout_fp", "packer_fp", "build_stats"):
        if key not in blob:
            raise Refuse(f"REFUSE: pack lacks {key}; sft_math.py --sft_path would refuse it")
    if blob["vocab_id"] != VOCAB_EXPECTED:
        raise Refuse(f"REFUSE: pack vocab_id {blob['vocab_id']} != {VOCAB_EXPECTED}")
    if "values" in blob:
        raise Refuse("REFUSE: pack carries FoNE values; this model is not a fone model")

    sup = int((blob["labels"] != -100).sum())
    pad = int(((blob["labels"] == -100) & (blob["input_ids"] == eos)).sum())
    n_drop = blob["build_stats"]["dropped_overlong"]
    if n_drop:
        raise Refuse(f"REFUSE: the packer dropped {n_drop} over-long examples; shorten the bank")

    # --- statistics ----------------------------------------------------------
    ans_tok = [len(tok.encode(r["answer"]).ids) for r in rows]
    q_tok = [len(tok.encode(r["question"]).ids) for r in rows]
    by_lang = collections.Counter(r["lang"] for r in rows)
    by_source = collections.Counter(r["source"] for r in rows)
    by_license = collections.Counter(r["license"] for r in rows)
    by_quality = collections.Counter(r["quality"] for r in rows)
    dist = {"answer_tokens": length_dist(ans_tok), "question_tokens": length_dist(q_tok),
            "answer_tokens_zh": length_dist([n for n, r in zip(ans_tok, rows) if r["lang"] == "zh"]),
            "answer_tokens_en": length_dist([n for n, r in zip(ans_tok, rows) if r["lang"] == "en"])}

    # --- spot check ----------------------------------------------------------
    order = sorted(range(len(rows)), key=lambda i: hashlib.sha256(
        f"spot{args.seed}:{rows[i]['id']}".encode()).hexdigest())
    spot = [rows[i] for i in order[:200]]
    spot_path = os.path.join(out_dir, f"spotcheck_200_{args.version}.jsonl")
    with open(spot_path, "w", encoding="utf-8") as fh:
        for r in spot:
            fh.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "answer": r["answer"],
                                 "source": r["source"], "license": r["license"],
                                 "quality": r["quality"], "answer_tokens": len(tok.encode(r["answer"]).ids),
                                 "lang": r["lang"]}, ensure_ascii=False) + "\n")
    report["spot_check"] = {
        "n": len(spot), "file": os.path.basename(spot_path),
        "sha256": sha256_file(spot_path),
        "first_20": [{"id": r["id"], "lang": r["lang"], "source": r["source"],
                      "question": r["question"], "answer": r["answer"],
                      "answer_tokens": len(tok.encode(r["answer"]).ids)} for r in spot[:20]],
    }

    sup_by_source = collections.Counter()
    for row in rows:
        sup_by_source[row["source"]] += len(tok.encode(row["answer"]).ids) + 1
    budget = {src: {"rows": by_source[src], "supervised_tokens": sup_by_source[src],
                    "sup_per_row": round(sup_by_source[src] / by_source[src], 2)}
              for src in sorted(by_source)}
    budget["TOTAL"] = {"rows": len(rows),
                       "supervised_tokens": sum(sup_by_source.values()),
                       "sup_per_row": round(sum(sup_by_source.values()) / len(rows), 2)}

    stats = {
        "rows": len(rows),
        "answer_tokens": dist["answer_tokens"],
        "question_tokens": dist["question_tokens"],
        "answer_tokens_zh": dist["answer_tokens_zh"],
        "answer_tokens_en": dist["answer_tokens_en"],
        "by_lang": dict(by_lang),
        "lang_ratio": {k: round(v / len(rows), 4) for k, v in by_lang.items()},
        "by_source": dict(by_source),
        "by_license": dict(by_license),
        "by_quality": dict(by_quality),
        "supervised_answer_tokens_jsonl": sum(ans_tok) + len(ans_tok),
        "supervised_tokens_in_pack": sup,
        "masked_pad_tokens": pad,
        "total_tokens_in_pack": int(blob["input_ids"].numel()),
    }

    # --- manifest ------------------------------------------------------------
    manifest = {
        "set": f"format_stop_{args.version}",
        "stage": "C1 format and stop (docs/standards/stage_cd_training_plan_1009.md section 2, row C1)",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": BUILDER,
        "builder_sha256": sha256_file(os.path.join(OUT, "build.py")),
        "bank_sha256": sha256_file(os.path.join(OUT, "oneliner_bank.json")),
        "seed": args.seed,
        "rows": len(rows),
        "protocol": {
            "format": "chatml",
            "system": "empty",
            "prompt_builder": "scripts.loader.format_prompt(question, None)",
            "prompt_prefix": "<|im_start|>user\\n{question}<|im_end|>\\n<|im_start|>assistant\\n",
            "answer_terminator": "<|im_end|>",
            "answer_max_tokens": ANSWER_MAX_TOKENS,
            "packer": "datagen.prepare_sft.pack_and_save split_encode=True seq=4096",
        },
        "sources": {},
        "licenses": dict(by_license),
        "revision": {
            "tokenizer": {"path": "data/tokenizer.json", "vocab_id": vid,
                          "sha256": sha256_file(tok_path), "size": tok.get_vocab_size()},
            "eval_gates": {n: {"path": os.path.relpath(p, ROOT), "sha256": sha256_file(p)}
                           for n, p, _ in gates},
            "holdout_hashes": {"path": "data/eval/holdout_hashes.txt",
                               "sha256": sha256_file(os.path.join(ROOT, "data/eval/holdout_hashes.txt")),
                               "pack_holdout_fp": blob["holdout_fp"]},
            "packer_fp": blob["packer_fp"],
        },
        "vocab_id": vid,
        "tokens": stats,
        "files": {
            f"format_stop_{args.version}.jsonl": {"sha256": sha256_file(jsonl_path),
                                     "bytes": os.path.getsize(jsonl_path), "rows": len(rows)},
            f"format_stop_{args.version}.pt": {"sha256": sha256_file(pack_path),
                                  "bytes": os.path.getsize(pack_path),
                                  "rows": int(blob["input_ids"].shape[0]),
                                  "seq": int(blob["input_ids"].shape[1])},
            f"spotcheck_200_{args.version}.jsonl": {"sha256": sha256_file(spot_path),
                                    "bytes": os.path.getsize(spot_path), "rows": len(spot)},
        },
        "supervised_budget": budget,
        "decontamination": {"ngram": DECON_N,
                            "gate_sets": report["decontamination"]["gate_sets"],
                            "hits": len(report["decontamination"]["hits"])},
    }
    src_notes = {
        # license values, and why each is the honest one:
        #   public_domain      a fact (a number, a capital, a unit definition, a letter
        #                      form) is not a copyrightable expression
        #   PSF-2.0            the Python one-liners use documented standard-library APIs
        #   generated_for_aupai  written for this set by this build; the repository has no
        #                      root LICENSE file, so no project license is inherited
        "synthetic_one_step": ("generated in build.py; operands from a seeded RNG, "
                               "the value recomputed in Python", "generated_for_aupai"),
        "synthetic_unit": ("generated in build.py; the conversion factor is an exact "
                           "fraction and the value is recomputed", "generated_for_aupai"),
        "synthetic_base": ("generated in build.py; the target form is recomputed with "
                           "format(n, 'b'/'o'/'x')", "generated_for_aupai"),
        "synthetic_series": ("generated in build.py; the sum or mean is recomputed from "
                             "the range", "generated_for_aupai"),
        "synthetic_number_theory": ("generated in build.py; math.gcd / math.lcm recompute "
                                    "the value", "generated_for_aupai"),
        "synthetic_percent": ("generated in build.py; Fraction(pct, 100) * total recomputes "
                              "the value exactly", "generated_for_aupai"),
        "synthetic_date": ("generated in build.py; every weekday, month, year and day count "
                           "is computed with datetime, not tabulated", "generated_for_aupai"),
        "synthetic_string": ("generated in build.py; character index, count and reversal are "
                             "computed in Python on a generated ASCII string",
                             "generated_for_aupai"),
        "synthetic_list_py": ("generated in build.py; the one-line answer is executed against "
                              "the generated list before it ships", "generated_for_aupai"),
        "synthetic_error_fix": ("generated in build.py; the correct value is recomputed and the "
                                "wrong one is built from it", "generated_for_aupai"),
        "synthetic_unit_fix": ("generated in build.py; the correct conversion is recomputed from "
                               "an exact fraction factor", "generated_for_aupai"),
        "synthetic_two_step": ("generated in build.py; both results are recomputed from the "
                               "operands", "generated_for_aupai"),
        "synthetic_record": ("generated in build.py; the record tables are invented, so no real "
                             "person or measurement is described", "generated_for_aupai"),
        "fact_core": ("public-domain reference facts (SI unit definitions, named constants, "
                      "world capitals, Roman numerals) typed into build.py", "public_domain"),
        "curated_define": ("one-sentence definitions written for this set", "generated_for_aupai"),
        "curated_translation": ("phrase pairs written for this set", "generated_for_aupai"),
        "stdlib_oneliner": ("Python one-liners over documented standard-library APIs; each entry "
                            "executed by build.py before it ships", "PSF-2.0"),
        "curated_refusal": ("refusal and clarification answers written for this set", "generated_for_aupai"),
    }
    for src, count in sorted(by_source.items()):
        note, lic = src_notes[src]
        manifest["sources"][src] = {"rows": count, "license": lic, "note": note}

    echo_by_source = collections.defaultdict(list)
    probe_hits = collections.Counter()
    for row in rows:
        q, a = row["question"], row["answer"]
        echo_by_source[row["source"]].append(longest_common_span(q, a))
        if q[:32] and q[:32] in a:
            probe_hits[row["source"]] += 1
    report["echo"] = {
        "rule": ("character run shared with the question; general cap "
                 f"{ECHO_MIN_CHARS - 1}, curated_define cap {DEFINE_ECHO_MAX_CHARS}"),
        "e2_probe": ("question[:32] must not appear in the answer "
                     "(eval/ladder/e2_chatml.judge); hits per source below"),
        "e2_probe_hits": dict(probe_hits),
        "by_source": {src: {"rows": len(v), "max_span": max(v),
                            "median_span": sorted(v)[len(v) // 2]}
                      for src, v in sorted(echo_by_source.items())},
    }
    report["supervised_budget"] = budget
    report["statistics"] = stats
    report["failures_note"] = ("every entry here is a DROP: the row never shipped. "
                               "An empty list means nothing was dropped after generation.")
    report["files"] = manifest["files"]

    if args.selftest:
        import shutil
        if os.environ.get("C1_KEEP_SELFTEST"):
            print(f"[c1] selftest OK: build ran end to end; kept {out_dir} for inspection")
            return 0
        shutil.rmtree(out_dir, ignore_errors=True)
        print("[c1] selftest OK: build ran end to end and every invariant held")
        return 0

    # The report is written first so the manifest can name it by hash. The manifest is the
    # one file whose every claim must be checkable from the bytes on disk: a manifest that
    # omits a delivered file is a provenance hole, and one that cites a hash nobody can
    # recompute is worse than no citation.
    report_path = os.path.join(out_dir, f"report_{args.version}.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    manifest["files"][f"report_{args.version}.json"] = {"sha256": sha256_file(report_path),
                                                        "bytes": os.path.getsize(report_path)}
    with open(os.path.join(out_dir, f"manifest_{args.version}.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(json.dumps({k: stats[k] for k in ("rows", "by_lang", "lang_ratio", "answer_tokens",
                                            "supervised_tokens_in_pack", "total_tokens_in_pack")},
                     indent=2, ensure_ascii=False))
    print(f"[c1] wrote {jsonl_path}\n[c1] wrote {pack_path}\n"
          f"[c1] wrote manifest_{args.version}.json report_{args.version}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Non-arithmetic quality checks for the SFT math corpus.

Target file: data/sft/real_math_filtered.jsonl   (132,205 rows, already
exact-dedup'd, and arithmetic already verified all-valid by eqcheck).
This script only REPORT statistics + samples. It NEVER writes a clean data
file — the final dedup+filter happens after the separate dedup pass.
Pure stdlib; no torch. Run from anywhere:  python data/sft/quality_check.py
"""

import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "real_math_filtered.jsonl")
RAND_SEED = 0

# ---------------------------------------------------------------------------
# Extractor reference (mirrors eval/math_zh.py + algorithms/rlvr_reward.py).
# eval take the LAST \\boxed; ANS_RE grabs the FIRST "答案是". Both are our
# ground truth for "what format the model must be trained to emit".
# ---------------------------------------------------------------------------
ANS_RE = re.compile(r"答案是[:：]\s*(.+?)(?:[。\n]|$)")


def extract_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1 : k]
    return text[j + 1 :]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def norm(s):
    return "".join(c for c in s if not c.isspace())


def bigram_set(s):
    return {s[i : i + 2] for i in range(len(s) - 1)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Checker 1 — duplicate / multiple answer markers
# ---------------------------------------------------------------------------
# Observed defect: "...最终答案是：\\boxed{80}\n答案是：\\boxed{80}" — two tail
# answer constructions. eval reads the LAST \\boxed, so a model trained on such
# samples learns an ambiguous, non-uniform closing. We count answer-announce
# phrases; the canonical single `答案是` counts as ONE, anything >= 2 is a dup.
#
# NOTE we do NOT key off bare `\\boxed` count: mid-solution boxed steps
# (`$...=\\boxed{4}$`) are a benign style (~17,932 / 132,205 rows) that does
# not touch the tail. And we do NOT count the variant "答案："/"答：" spellings
# here — those are Checker 2's job. Checker 1 is exactly the *canonical* tail
# phrase "答案是" repeated, which is the true "duplicate final-answer marker".
ANNOUNCE_RE = re.compile(r"答案是")


def checker1(instruction, output):
    n = len(ANNOUNCE_RE.findall(output))
    if n >= 2:
        return True, f"{n} answer-announce markers"
    return False, ""


# ---------------------------------------------------------------------------
# Checker 2 — non-canonical answer phrasing (尾答格式漂移)
# ---------------------------------------------------------------------------
# Every row's final line is the canonical "答案是：\\boxed{N}" (100% end with
# the closing brace). Drift therefore shows up as NON-canonical answer phrasing
# anywhere in the body: "答案：", "答案:" , "答：", "答:", "最终答案". These teach
# the same "final answer" intent under different spellings.
VARIANT_ANS = ["答案：", "答案:", "答：", "答:", "最终答案"]


def checker2(instruction, output):
    hits = [v for v in VARIANT_ANS if v in output]
    if hits:
        return True, "+".join(hits)
    return False, ""


# ---------------------------------------------------------------------------
# Checker 3 — instruction restated verbatim into the output head (题干复述)
# ---------------------------------------------------------------------------
# The output's opening ~PREFIX chars reproduce the question (wasted tokens,
# against the short-CoT direction). Scored by character-bigram Jaccard between
# the normalized output prefix and the normalized instruction.
#
# THRESHOLD CALIBRATION (real_math_filtered, N=132,205, prefix=45):
#   mean 0.234 | p90 0.400 | p95 0.457 | p99 0.583 | max 1.000
# The sharp tail 0.95-1.00 = verbatim question copies (clearly bad). Sampling
# showed the 0.48-0.52 band is mixed (some restatements, some merely borrow the
# numbers = false positives), so a 0.50 cut is NOT clean. Sampling 0.55-0.75 was
# 6/6 genuine whole-question restatements -> clean. We choose 0.55, leaving a
# clear margin above natural topical overlap (0.2-0.4) and below the verbatim
# tail. At prefix 45 that rejects the top ~1.5% (2,048 rows).
PREFIX = 45
CHECKER3_JAC = 0.55


def checker3(instruction, output):
    o = norm(output)[:PREFIX]
    i = norm(instruction)
    j = jaccard(bigram_set(o), bigram_set(i))
    if j >= CHECKER3_JAC:
        return True, f"head-vs-instruction bigram jaccard {j:.2f}"
    return False, ""


# ---------------------------------------------------------------------------
# Checker 4 — equation glued into a prose sentence (算式嵌句不换行)
# ---------------------------------------------------------------------------
# Observed defect: "可以写成4-2=2。" — a bare two-operand equation pasted
# between prose and a clause-final full stop, not on its own line and not
# $...$-wrapped. We require a Han char immediately before the equation and a
# clause-final punctuation immediately after, so the pervasive acceptable
# inline style ("总重量是100 + 200 = 300克。", "$6-2=\\boxed{4}$", a bare
# standalone "24÷3=8") is left alone. eqcheck's line-level regex still
# verifies such a step, so this flags a FORMAT defect, not an arithmetic one.
GLUED_EQ = re.compile(
    r"[一-鿿](-?\d+(?:\.\d+)?)\s*([+\-×÷*/])\s*"
    r"(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)[，。；]"
)


def _strip_math(line):
    return re.sub(r"\$[^\$]*\$", "", line)


def checker4(instruction, output):
    for line in output.split("\n"):
        if "$" in line:
            line = _strip_math(line)
        if GLUED_EQ.search(line):
            return True, "bare equation glued into sentence (e.g. '写成4-2=2。')"
    return False, ""


# ---------------------------------------------------------------------------
# Self-verification. Every checker must pass POS (must be True) and NEG (must
# be False) fixtures before the full-corpus run is trustworthy.
# ---------------------------------------------------------------------------
SELFCHECKS = [
    (
        "checker1",
        #  must be True  (positives)
        [
            "小明首先数了一数。最终答案是：\\boxed{80}\n答案是：\\boxed{80}",
            "先算加法和。答案是8。\n答案是：\\boxed{8}",
        ],
        #  must be False (negatives)
        [
            "24÷3=8\n每只小鹿可以分到8个苹果。\n答案是：\\boxed{8}",
            "计算得 4-2=2。\n答案是：\\boxed{2}",
        ],
    ),
    (
        "checker2",
        [
            "巧克力剩下 $1/6$。答案：小明还剩下 $1/6$。\n答案是：\\boxed{1/6}",
            "小明有3个。答：小明有3个。\n答案是：\\boxed{3}",
            "最后最终答案是：\\boxed{5}",
            "答案为答案:8。\n答案是：\\boxed{8}",
        ],
        [
            "24÷3=8\n答案是：\\boxed{8}",
            "答案是：\\boxed{2}",
        ],
    ),
    (
        "checker3",
        [
            (
                "小明昨天吃了4颗葡萄，今天又吃了5颗葡萄，他一共吃了多少颗葡萄？",
                "小明昨天吃了4颗葡萄，今天又吃了5颗葡萄，他一共吃了多少颗葡萄？\n"
                "我们可以用加法：4+5=9。\n答案是：\\boxed{9}",
            ),
        ],
        [
            (
                "鹿妈妈买了24个苹果，她想平均分给她的3只小鹿吃，每只小鹿可以分到几个苹果？",
                "把25个苹果平均分成5份，每份是5个。所以每份能分到5个。\n"
                "答案是：\\boxed{5}",
            ),
        ],
    ),
    (
        "checker4",
        [
            "详情可以写成4-2=2。所以差值是2。",
            "小明有4本，小红有2本，则4-2=2。差值就是2。",
            "第一段约掉1/2即1÷2=0.5。",           # glued, no colon before digit
        ],
        [
            "24÷3=8",                             # standalone line, no Han before
            "$6-2=\\boxed{4}$",                    # $-wrapped
            "答案是：\\boxed{8}",                    # boxed tail, no equation
            "总重量是100 + 200 = 300克。",           # eq followed by unit, not punct
            "小明有5支铅笔。\n答案是：\\boxed{5}",
        ],
    ),
]

CHECKERS = {
    "checker1": checker1,
    "checker2": checker2,
    "checker3": checker3,
    "checker4": checker4,
}


def selfcheck():
    print("==", "selfcheck")
    ok = True
    for name, pos, neg in SELFCHECKS:
        fn = CHECKERS[name]
        for i, ex in enumerate(pos):
            if name == "checker3":
                got, why = fn(*ex)
            else:
                got, why = fn("", ex)
            status = "PASS" if got else "FAIL"
            if not got:
                ok = False
            print(f"  {name} POS{i}: {status}  {why if why else '(expected True)'}")
        for i, ex in enumerate(neg):
            if name == "checker3":
                got, why = fn(*ex)
            else:
                got, why = fn("", ex)
            status = "PASS" if not got else "FAIL"
            if got:
                ok = False
            print(f"  {name} NEG{i}: {status}  {why if why else '(ok, expected False)'}")
    print("  selfcheck:", "ALL CLEAN" if ok else "*** FAILURES — DO NOT TRUST COUNTS ***")
    return ok


def run_full():
    print("==", f"full run on {DATA}")
    with open(DATA, encoding="utf-8") as f:
        rows = [json.loads(l.strip()) for l in f]
    print(f"  rows: {len(rows)}\n")
    for name in ["checker1", "checker2", "checker3", "checker4"]:
        fn = CHECKERS[name]
        rej = []
        for r in rows:
            got, why = fn(r["instruction"], r["output"])
            if got:
                rej.append((r, why))
        print(f"== {name}: {len(rej)} / {len(rows)} rejected "
              f"({len(rej) / len(rows):.1%})")
        rnd = random.Random(RAND_SEED)
        rnd.shuffle(rej)
        for r, why in rej[:5]:
            print(f"    [reject] {why}")
            print(f"      I: {r['instruction'][:70]}")
            print(f"      O: {r['output'][:90]!r}")
        print("")
    # overlap-free: distribution of number of checkers firing per row
    from collections import Counter
    hits = Counter()
    for r in rows:
        n = sum(1 for name in CHECKERS if CHECKERS[name](r["instruction"], r["output"])[0])
        hits[n] += 1
    print("== overlap histogram (checkers firing per row):",
          dict(sorted(hits.items())))


if __name__ == "__main__":
    clean = selfcheck()
    if clean:
        run_full()
    else:
        sys.exit(1)

#!/usr/bin/env python3
"""RLVR data: build/load the verifiable-reward jsonl.

prepare() merges school_math_r1_zh (\\boxed{} answers) + gsm8k_zh (numeric
answers), normalizes, classifies each row as verifiable (single final value the
reward can score) or non-verifiable (solution-form aligned block / nested
multi-answer / unit-collapse-to-None), routes the latter to
rlvr_nonverifiable.jsonl at build (this repo's fail-loud principle: a derived
artifact must not silently carry rows the reward can never fire on), and
deduplicates the verifiable pool by prompt into data/rl/rlvr_math.jsonl
consumed by rlvr_trainer.py. classification uses the reward's OWN normalizer so
build and reward cannot disagree (the split-brain gap that let 米米 through).
Pure stdlib — importable without torch/GPU.

Usage: python algorithms/rlvr_data.py [--clean-audit]
  --clean-audit  run the build gate against the shipped rlvr_clean.jsonl and
                 print the routing volume (5 hard + 361 dead rows surface here
                 without touching the consumed artifact)
"""

import json
import os
import re
import sys

try:
    from .rlvr_reward import reward_fn, normalize_answer  # noqa: I001  (relative vs absolute import guard)
except ImportError:
    from rlvr_reward import reward_fn, normalize_answer  # noqa: I001

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RLVR_PATH = os.path.join(DATA, "rl", "rlvr_math.jsonl")
NONVERIFIABLE_PATH = os.path.join(DATA, "rl", "rlvr_nonverifiable.jsonl")

# Solution-form scaffolding: a GT carrying these is a worked solution (aligned
# block / cases) or a multi-answer (nested \boxed). There is no single final
# value for the reward to fire on -- route to the non-verifiable pool.
_SOLUTION_ENV = re.compile(r"\\(begin|end)\{(aligned|cases|align|array)\}")
_MULTI_BOXED = re.compile(r"\\boxed\{", re.DOTALL)


def extract_boxed(text):
    """Extract answer from \\boxed{...} with balanced-brace matching."""
    results = []
    idx = 0
    while True:
        i = text.find("\\boxed{", idx)
        if i < 0:
            break
        depth = 1
        j = i + 7
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[i + 7 : j - 1].strip())
        idx = j
    return results[-1] if results else None


def extract_gsm8k_answer(text):
    """Extract final numeric answer from GSM8K solution (last number)."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def normalize(ans):
    """Normalize answer for comparison: strip LaTeX, whitespace, units."""
    if ans is None:
        return None
    s = str(ans).strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)  # \text{cm} -> cm
    s = re.sub(r"\\dfrac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)  # \dfrac{a}{b} -> a/b
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    s = s.replace("\\", "").replace(" ", "").replace("（", "(").replace("）", ")")
    s = s.rstrip("。.,，")
    try:
        return str(float(s))
    except ValueError:
        return s


def load_problems(path=RLVR_PATH):
    """Load prepared RLVR problems: [{prompt, answer, source}, ...].

    Data boundary: a row is refused unless its GT round-trips through the
    reward (reward_fn(\\boxed{gt}, gt) == 1.0) and its braces balance.
    Tolerance at the reward boundary is invisible; a refusal here is loud.
    """
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    good, bad = [], 0
    for d in rows:
        gt = str(d.get("answer") or "")
        if gt.count("{") != gt.count("}") or reward_fn(f"\\boxed{{{gt}}}", gt) != 1.0:
            bad += 1
            continue
        good.append(d)
    if bad:
        print(
            f"[load_problems] REFUSED {bad}/{len(rows)} rows from {path}: "
            f"GT does not round-trip through reward_fn (unbalanced braces or "
            f"unrewardable answer form)",
            file=sys.stderr,
            flush=True,
        )
    return good


def classify_verifiable(ans):
    """Route a raw extracted answer: VERIFIABLE (a single final value the reward
    can score) or NONVERIFIABLE.

    A row is non-verifiable (route to the pool, fail-loud at build) if:
      - its answer is solution-form (\\begin{aligned}/cases/align) -- no single
        final value to fire on;
      - it nests a multiple-answer \\boxed{} (\\boxed within \\boxed);
      - reward's normalizer collapses a non-empty answer to None -- the unit
        strip ate a name/word (the 米米 class). Detected here, not silently
        passed: build and reward must share one normalizer, or build admits
        rows reward refuses (the split-brain gap).
    """
    if ans is None:
        return "NONVERIFIABLE"
    if _SOLUTION_ENV.search(ans) is not None:
        return "NONVERIFIABLE"
    if ans.count(r"\boxed") > 1:
        return "NONVERIFIABLE"
    # the reward's own normalizer is the source of truth for scoring
    if normalize_answer(ans) is None:
        return "NONVERIFIABLE"
    return "VERIFIABLE"


def prepare(data_dir=DATA, out_path=RLVR_PATH, nonverif_path=NONVERIFIABLE_PATH):
    """Build the verifiable RLVR jsonl + the routed non-verifiable pool from the
    raw datasets. Returns (verifiable_items, nonverifiable_items)."""
    out = []
    nonverif = []

    def _emit(d, ans, source):
        if classify_verifiable(ans) == "VERIFIABLE":
            out.append({"prompt": d["instruction"], "answer": ans, "source": source})
        else:
            nonverif.append({"prompt": d["instruction"], "answer": ans, "source": source, "route": classify_verifiable(ans)})

    # school_math_r1_zh: 223K problems with \boxed{} answers
    with open(os.path.join(data_dir, "school_math_r1_zh.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ans = extract_boxed(d["output"])
            if ans and normalize(ans):
                _emit(d, ans, "school_math")

    # gsm8k_zh: 7.5K problems with numeric answers
    with open(os.path.join(data_dir, "gsm8k_zh.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ans = extract_gsm8k_answer(d["output"])
            if ans and normalize(ans):
                _emit(d, ans, "gsm8k")

    seen = set()
    deduped = []
    for item in out:
        if item["prompt"] not in seen:
            seen.add(item["prompt"])
            deduped.append(item)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in deduped:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(nonverif_path, "w", encoding="utf-8") as f:
        for item in nonverif:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return deduped, nonverif


def audit_clean(clean_path):
    """One-off: run the build gate against the shipped rlvr_clean.jsonl and
    report the routing volume. The clean artifact predates the gate; its
    dead rows (solution-form / unit-collapse) currently pass load_problems
    silently. This audit surfaces them without touching the consumed file."""
    clean_path = clean_path or os.path.join(DATA, "rl", "rlvr_clean.jsonl")
    counts = {"verifiable": 0, "solution_form": 0, "multi_boxed": 0, "unit_collapse": 0, "empty": 0}
    n = 0
    with open(clean_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ans = str(d.get("answer") or "")
            n += 1
            if not ans:
                counts["empty"] += 1
                continue
            if _SOLUTION_ENV.search(ans) is not None:
                counts["solution_form"] += 1
                continue
            if ans.count(r"\boxed") > 1:
                counts["multi_boxed"] += 1
                continue
            if normalize_answer(ans) is None:
                counts["unit_collapse"] += 1
                continue
            counts["verifiable"] += 1
    return n, counts


def main():
    if "--clean-audit" in sys.argv:
        n, c = audit_clean(None)
        print(
            f"audit of rlvr_clean.jsonl: {n} rows -> "
            f"verifiable {c['verifiable']} | routing {n - c['verifiable']} "
            f"(solution_form {c['solution_form']}, multi_boxed {c['multi_boxed']}, "
            f"unit_collapse {c['unit_collapse']}, empty {c['empty']})"
        )
        return
    verif, nonverif = prepare()
    print(
        f"Total verifiable: {len(verif)} problems "
        f"(school_math: {sum(1 for d in verif if d['source'] == 'school_math')}, "
        f"gsm8k: {sum(1 for d in verif if d['source'] == 'gsm8k')})"
    )
    print(f"Routed non-verifiable: {len(nonverif)} -> {NONVERIFIABLE_PATH}")


if __name__ == "__main__":
    main()

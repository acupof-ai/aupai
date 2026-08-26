#!/usr/bin/env python3
"""RLVR data: build/load the verifiable-reward jsonl.

prepare() merges school_math_r1_zh (\\boxed{} answers) + gsm8k_zh (numeric
answers), normalizes, deduplicates by prompt, and writes data/rl/rlvr_math.jsonl
consumed by rlvr_trainer.py. Pure stdlib — importable without torch/GPU.

Usage: python algorithms/rlvr_data.py
   or: python algorithms/prepare_rlvr.py
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RLVR_PATH = os.path.join(DATA, "rl", "rlvr_math.jsonl")


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
    # Try numeric comparison
    try:
        return str(float(s))
    except ValueError:
        return s


def load_problems(path=RLVR_PATH):
    """Load prepared RLVR problems: [{prompt, answer, source}, ...]."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare(data_dir=DATA, out_path=RLVR_PATH):
    """Build the RLVR jsonl from raw datasets. Returns the deduped items."""
    out = []
    # school_math_r1_zh: 223K problems with \boxed{} answers
    with open(os.path.join(data_dir, "school_math_r1_zh.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ans = extract_boxed(d["output"])
            if ans and normalize(ans):
                out.append({"prompt": d["instruction"], "answer": ans, "source": "school_math"})

    # gsm8k_zh: 7.5K problems with numeric answers
    with open(os.path.join(data_dir, "gsm8k_zh.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ans = extract_gsm8k_answer(d["output"])
            if ans and normalize(ans):
                out.append({"prompt": d["instruction"], "answer": ans, "source": "gsm8k"})

    # Deduplicate by prompt
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
    return deduped


def main():
    deduped = prepare()
    print(
        f"Total: {len(deduped)} problems "
        f"(school_math: {sum(1 for d in deduped if d['source'] == 'school_math')}, "
        f"gsm8k: {sum(1 for d in deduped if d['source'] == 'gsm8k')})"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Census for code-500-v2: verify the 500 clean problems appear in NO SFT source.

Normalized containment (holdout.norm, first 256 chars of each v2 instruction as
substring of every source line's text fields). Expect 0 hits.

Usage: python scripts/census_code_v2.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from holdout import norm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_PATH = os.path.join(ROOT, "data", "eval", "code_holdout_v2_500.jsonl")

# Every file that fed an SFT pack, plus the dropped carve source.
SOURCES = [
    "data/alpaca_gpt4_zh.jsonl",
    "data/coig.jsonl",
    "data/openo1_sft.jsonl",
    "data/gsm8k_zh.jsonl",
    "data/school_math_r1_zh.jsonl",
    "data/s1k.jsonl",
    "data/sft/fable5_cot.jsonl",
    "data/sft/v5_evol_code_2300.jsonl",
    "data/synthetic/knowledge_qa_zh.jsonl",
    "data/synthetic/math_gsm8k_zh.jsonl",
    "data/synthetic/code_python_zh.jsonl",  # the dropped carve source
]
TEXT_FIELDS = ("instruction", "output", "prompt", "response", "input", "q", "a")


def main():
    v2 = [json.loads(l) for l in open(V2_PATH, encoding="utf-8")]
    v2_norm = [(r["instruction"], norm(r["instruction"])[:256]) for r in v2]
    print(f"code-500-v2: {len(v2)} problems")
    print(f"scanning {len(SOURCES)} sources for normalized containment...\n")

    total_hits = 0
    for rel in SOURCES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print(f"  SKIP {rel} (not found)")
            continue
        n_lines = 0
        hits = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                n_lines += 1
                d = json.loads(line)
                for field in TEXT_FIELDS:
                    text = d.get(field)
                    if not text or not isinstance(text, str):
                        continue
                    tn = norm(text)
                    for instr, n256 in v2_norm:
                        if n256 and n256 in tn:
                            hits.append((n_lines, field, instr[:60]))
        status = f"{len(hits)} HITS" if hits else "clean"
        print(f"  {rel}: {n_lines} lines, {status}")
        for ln, field, instr in hits[:5]:
            print(f"    line {ln} field={field} instr={instr!r}")
        total_hits += len(hits)

    print(f"\n{'FAIL' if total_hits else 'PASS'}: {total_hits} containment hits across all sources")
    sys.exit(1 if total_hits else 0)


if __name__ == "__main__":
    main()

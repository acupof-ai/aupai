#!/usr/bin/env python3
# restartable: pure file transform; rerun overwrites the output.
"""Rewrite an instruction jsonl's answers into the <think> format (DeepSeek-R1 / Qwen3).

    python3 datagen/think_format.py --in data/gsm8k_zh.jsonl --out data/sft/zh_think/gsm8k_zh.jsonl --math

--math: the answer is a worked solution; it becomes the reasoning, and its final number (after
"####" when present, else the last number) becomes the answer line. Without --math the answer
has no separable reasoning, so the think block is empty -- Qwen3's non-thinking convention.
"""
import argparse
import json
import os
import re

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def think(output, math):
    if not math:
        return f"<think>\n\n</think>\n\n{output}"
    body, _, final = output.partition("####")
    nums = NUM.findall(final) or NUM.findall(body)
    if not nums:
        return None
    ans = nums[-1].replace(",", "").rstrip(".")
    return f"<think>\n{body.strip()}\n</think>\n\n答案：{ans}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--math", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    kept = dropped = 0
    with open(args.src, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            t = think(d["output"].strip(), args.math)
            if t is None:
                dropped += 1
                continue
            g.write(json.dumps({**d, "output": t}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"{args.out}: {kept} kept, {dropped} dropped (no final number)")


if __name__ == "__main__":
    assert think("a 1+1=2\nb 2*3=6。", True) == "<think>\na 1+1=2\nb 2*3=6。\n</think>\n\n答案：6"
    assert think("x 3+4=7\n#### 7", True).endswith("答案：7")
    assert think("你好", False) == "<think>\n\n</think>\n\n你好"
    main()

#!/usr/bin/env python3
"""Download and process SFT datasets for Chinese reasoning.
Sources:
  - jiemaluo/chinese-reasoning-v1 (10K math/logic)
  - meta-math/GSM8K_zh (7.5K math word problems)
  - zake7749/QwQ-mmlu-reasoning-chinese (9.7K MMLU CoT, Traditional→Simplified)
  - TheFusionCube/Fable-5-CoT-Traces (467 Fable 5 traces, decoys filtered)

Usage: python3 prepare_sft.py [--max_samples 5000]
Output: data/sft/sft_merged.jsonl.gz
"""
import json, os, random, re, sys

random.seed(42)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sft")


def load_reasoning(path):
    """chinese-reasoning-v1: instruction/response or instruction/output."""
    samples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            samples.append({"instruction": d["instruction"], "output": d.get("response", d.get("output", ""))})
    return samples


def load_gsm8k(path):
    """GSM8K_zh: question_zh/answer_zh, strip <<>> calculator annotations."""
    samples = []
    with open(path) as f:
        data = json.load(f)
    for d in data:
        if d.get("split") != "train":
            continue
        q, a = d.get("question_zh") or "", d.get("answer_zh") or ""
        q, a = q.strip(), a.strip()
        if not q or not a:
            continue
        a = re.sub(r'<<[^>]*>>', '', a)
        samples.append({"instruction": q, "output": a})
    return samples


def load_qwq_mmlu(path):
    """QwQ-MMLU: prompt/reasoning/response, Traditional→Simplified."""
    from opencc import OpenCC
    import pyarrow.parquet as pq
    cc = OpenCC('t2s')
    samples = []
    for row in pq.read_table(path).to_pylist():
        q = cc.convert(row["prompt"].strip())
        reasoning = cc.convert(row["reasoning"].strip())
        response = cc.convert(row["response"].strip())
        samples.append({"instruction": q, "output": f"{reasoning}\n\n{response}"})
    return samples


def load_fable5(path):
    """Fable-5-CoT-Traces: prompt/response, filter decoys."""
    samples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("category") == "decoy":
                continue
            if not d.get("prompt") or not d.get("response"):
                continue
            samples.append({"instruction": d["prompt"].strip(), "output": d["response"].strip()})
    return samples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--output", default=os.path.join(OUT, "sft_all_v2.jsonl"))
    args = parser.parse_args()

    # Check input files exist
    inputs = {
        "reasoning": os.path.join(OUT, "reasoning.jsonl"),
        "gsm8k": os.path.join(OUT, "gsm8k_zh.json"),
        "qwq": os.path.join(OUT, "qwq_mmlu.parquet"),
        "fable5": os.path.join(OUT, "fable5_cot.jsonl"),
    }
    for name, path in inputs.items():
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at {path}")
            sys.exit(1)

    # Load all datasets
    reasoning = load_reasoning(inputs["reasoning"])
    gsm8k = load_gsm8k(inputs["gsm8k"])
    qwq = load_qwq_mmlu(inputs["qwq"])
    fable5 = load_fable5(inputs["fable5"])

    print(f"reasoning: {len(reasoning)}, gsm8k: {len(gsm8k)}, qwq: {len(qwq)}, fable5: {len(fable5)}")

    # Merge Chinese sources, take subset
    chinese = reasoning + gsm8k + qwq
    random.shuffle(chinese)
    chinese = chinese[:args.max_samples]

    # Add Fable 5 traces
    merged = chinese + fable5

    # Dedup by instruction
    seen = set()
    deduped = []
    dupes = 0
    for d in merged:
        key = d["instruction"].strip()
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        deduped.append(d)
    print(f"dedup: {len(merged)} -> {len(deduped)} (removed {dupes} duplicates)")

    random.shuffle(deduped)

    # Save
    with open(args.output, 'w', encoding='utf-8') as f:
        for d in deduped:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    size = os.path.getsize(args.output)
    print(f"Total: {len(deduped)} samples, {size/1024/1024:.1f}MB -> {args.output}")


if __name__ == "__main__":
    main()

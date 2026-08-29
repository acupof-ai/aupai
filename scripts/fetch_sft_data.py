#!/usr/bin/env python3
"""Fetch missing SFT raw sources, then merge them into the SFT mix.

Fetch mode (`fetch`): download the 3 raw sources that have no other producer.
  - jiemaluo/chinese-reasoning-v1        -> data/sft/reasoning.jsonl
  - meta-math/GSM8K_zh                   -> data/sft/gsm8k_zh.json  (train split)
  - zake7749/QwQ-mmlu-reasoning-chinese  -> data/sft/qwq_mmlu.parquet
(TheFusionCube/Fable-5-CoT-Traces was a one-time export already present as
fable5_cot.jsonl; re-fetch it manually if lost.)

Merge mode (default): combine the 4 local files -> data/sft/sft_all_v2.jsonl,
schema {instruction, output}, shuffled with seed 42.

Usage:
  python3 scripts/fetch_sft_data.py fetch
  python3 scripts/fetch_sft_data.py [--max_samples 20000] [--output data/sft/sft_all_v2.jsonl]
"""

import json, os, random, re, sys

random.seed(42)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sft")

SFT_SOURCES = {
    "reasoning": ("jiemaluo/chinese-reasoning-v1", "train", "reasoning.jsonl"),
    "gsm8k": ("meta-math/GSM8K_zh", "train", "gsm8k_zh.json"),
    "qwq": ("zake7749/QwQ-mmlu-reasoning-chinese", "train", "qwq_mmlu.parquet"),
}


def fetch_missing():
    from datasets import load_dataset

    os.makedirs(OUT, exist_ok=True)
    failed = []
    for name, (repo, split, fname) in SFT_SOURCES.items():
        path = os.path.join(OUT, fname)
        if os.path.exists(path):
            print(f"{name}: already present -> {path}")
            continue
        try:
            ds = load_dataset(repo, split=split)
            if fname.endswith(".parquet"):
                ds.to_parquet(path)
            elif fname.endswith(".json"):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump([dict(r) for r in ds], f, ensure_ascii=False)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    for row in ds:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{name}: {len(ds)} rows -> {path}")
        except Exception as e:  # noqa: BLE001 — report and continue
            failed.append(name)
            print(f"{name}: FAILED {type(e).__name__}: {str(e)[:120]}")
    if failed:
        sys.exit(f"failed to fetch: {failed}")


def load_reasoning(path):
    """chinese-reasoning-v1: instruction/response or instruction/output."""
    samples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            samples.append(
                {"instruction": d["instruction"], "output": d.get("response", d.get("output", ""))}
            )
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
        a = re.sub(r"<<[^>]*>>", "", a)
        samples.append({"instruction": q, "output": a})
    return samples


def load_qwq_mmlu(path):
    """QwQ-MMLU: prompt/reasoning/response, Traditional→Simplified."""
    from opencc import OpenCC
    import pyarrow.parquet as pq

    cc = OpenCC("t2s")
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
    parser.add_argument(
        "cmd", nargs="?", choices=["fetch"], help="'fetch' downloads missing raw sources; omit to merge"
    )
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--output", default=os.path.join(OUT, "sft_all_v2.jsonl"))
    args = parser.parse_args()
    if args.cmd == "fetch":
        fetch_missing()
        return

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

    reasoning = load_reasoning(inputs["reasoning"])
    gsm8k = load_gsm8k(inputs["gsm8k"])
    qwq = load_qwq_mmlu(inputs["qwq"])
    fable5 = load_fable5(inputs["fable5"])

    print(f"reasoning: {len(reasoning)}, gsm8k: {len(gsm8k)}, qwq: {len(qwq)}, fable5: {len(fable5)}")

    chinese = reasoning + gsm8k + qwq
    random.shuffle(chinese)
    chinese = chinese[: args.max_samples]

    merged = chinese + fable5

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

    with open(args.output, "w", encoding="utf-8") as f:
        for d in deduped:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    size = os.path.getsize(args.output)
    print(f"Total: {len(deduped)} samples, {size / 1024 / 1024:.1f}MB -> {args.output}")


if __name__ == "__main__":
    main()

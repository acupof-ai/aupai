#!/usr/bin/env python3
"""Build offline quick multiple-choice benchmark samples.

The script downloads public validation or test parquet files. It writes fixed
small samples for checkpoint trend checks. Full benchmark runs use the source
datasets directly.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval" / "quick"
N = 32

SOURCES = {
    "hellaswag": "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/validation-00000-of-00001.parquet",
    "piqa": "https://huggingface.co/datasets/lighteval/piqa/resolve/main/plain_text/validation-00000-of-00001.parquet",
    "arc-easy": "https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Easy/test-00000-of-00001.parquet",
    "arc-challenge": "https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/test-00000-of-00001.parquet",
    "winogrande": "https://huggingface.co/datasets/allenai/winogrande/resolve/main/winogrande_xl/validation-00000-of-00001.parquet",
}


def write_jsonl(name, rows):
    path = OUT / f"{name}_quick{N}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(name, df):
    rows = []
    if name == "hellaswag":
        for _, ex in df.head(N).iterrows():
            ctx = ex.get("ctx") or (ex["ctx_a"] + " " + ex["ctx_b"]).strip()
            rows.append({"prompt": ctx, "options": list(ex["endings"]), "label": int(ex["label"])})
    elif name == "piqa":
        for _, ex in df.head(N).iterrows():
            rows.append({
                "prompt": ex["goal"].rstrip(),
                "options": [" " + ex["sol1"], " " + ex["sol2"]],
                "label": int(ex["label"]),
            })
    elif name.startswith("arc-"):
        for _, ex in df.head(N).iterrows():
            labels = list(ex["choices"]["label"])
            rows.append({
                "prompt": ex["question"].rstrip(),
                "options": [" " + text for text in ex["choices"]["text"]],
                "label": labels.index(ex["answerKey"]),
            })
    elif name == "winogrande":
        for _, ex in df.head(N).iterrows():
            prefix, _, suffix = ex["sentence"].partition("_")
            rows.append({
                "prompt": prefix.strip(),
                "options": [ex["option1"] + suffix, ex["option2"] + suffix],
                "label": int(ex["answer"]) - 1,
            })
    else:
        raise ValueError(f"unknown benchmark {name}")
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = {"sample_count": N, "built_at": datetime.now(timezone.utc).isoformat(), "sources": SOURCES}
    for name, url in SOURCES.items():
        df = pd.read_parquet(url)
        rows = normalize(name, df)
        if len(rows) != N:
            raise RuntimeError(f"{name}: expected {N} rows, got {len(rows)}")
        write_jsonl(name, rows)
        print(f"{name}: {N} rows")
    with (OUT / "SOURCES.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

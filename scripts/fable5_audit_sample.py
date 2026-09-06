#!/usr/bin/env python3
# restartable: reads 4665 rows and writes 50 in one pass; an interrupt costs one
# rerun (<5s), no partial state is kept.
"""44-33: draw a stratified hand-read sample from fable5_cot_merged.jsonl.

Even stratification: 4 rows per source, 3 for the two smallest sources,
seed 42. Writes runs/fable5_audit_sample.jsonl with the fields the audit
judges (cot kept full -- truncation would hide whether the span derives
the answer).
"""
import collections
import json
import random

CORPUS = "/work/aupai/data/raw/fable5_traces/fable5_cot_merged.jsonl"
OUT = "/work/aupai/runs/fable5_audit_sample.jsonl"
SEED = 42


def source_of(source_file: str) -> str:
    parts = source_file.split("/")
    return parts[-2] if len(parts) >= 2 else source_file


def main() -> None:
    by_source: dict[str, list[dict]] = collections.defaultdict(list)
    with open(CORPUS) as f:
        for line in f:
            r = json.loads(line)
            by_source[source_of(r["source_file"])].append(r)

    alloc = {s: 4 for s in by_source}
    for s in sorted(by_source, key=lambda s: len(by_source[s]))[:2]:
        alloc[s] = 3

    rng = random.Random(SEED)
    out = []
    for s, rows in by_source.items():
        rows = list(rows)
        rng.shuffle(rows)
        for r in rows[: alloc[s]]:
            out.append(
                {
                    "uid": r["uid"],
                    "source": s,
                    "source_file": r["source_file"],
                    "origin": r.get("origin"),
                    "output_type": r.get("output_type"),
                    "cot": r["cot"],
                    "output": r["output"],
                    "ctx": r.get("context", "")[:200],
                }
            )

    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(out)} rows sampled across {len(by_source)} sources -> {OUT}")


if __name__ == "__main__":
    main()

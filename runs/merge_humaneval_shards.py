#!/usr/bin/env python3
"""Merge sharded HumanEval outputs.

Use this script for both quick checkpoint checks and full HumanEval runs.
The script never uses GPUs.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"


def shard_path(ckpt, run, workers, idx):
    name = f"preds_humaneval_{ckpt}.rstripnl.shard{idx}of{workers}.{run}.jsonl"
    return EVAL_DIR / name


def read_shard(path):
    header = None
    rows = []
    if not path.exists():
        return header, rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            if "_header" in item:
                header = item
            else:
                rows.append(item)
    return header, rows


def task_number(row):
    task_id = row.get("task_id", "")
    try:
        return int(str(task_id).split("/", 1)[1])
    except (IndexError, ValueError):
        return 10**9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--expected", type=int, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    headers = []
    rows = []
    for idx in range(args.workers):
        path = shard_path(args.ckpt, args.run, args.workers, idx)
        header, shard_rows = read_shard(path)
        if header is not None:
            headers.append(header)
        rows.extend(shard_rows)

    task_ids = [row.get("task_id") for row in rows]
    if len(rows) != args.expected:
        raise SystemExit(f"expected {args.expected} rows, got {len(rows)}")
    if len(set(task_ids)) != len(task_ids):
        raise SystemExit("duplicate task_id found in shards")

    rows.sort(key=task_number)
    passed = sum(bool(row.get("ok")) for row in rows)
    empty = sum(bool(row.get("empty")) for row in rows)
    eos = sum(row.get("empty_kind") == "eos_first" for row in rows)
    stop = sum(row.get("empty_kind") == "stop_at_0" for row in rows)

    out = Path(args.out) if args.out else EVAL_DIR / (
        f"preds_humaneval_{args.ckpt}.rstripnl.{args.run}.merged.jsonl"
    )
    header = {
        "_header": "merged",
        "ckpt": args.ckpt,
        "run": args.run,
        "workers": args.workers,
        "expected": args.expected,
        "n_rows": len(rows),
        "pass": passed,
        "empty": empty,
        "eos_first": eos,
        "stop_at_0": stop,
    }
    with out.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"HUMANEVAL {args.run}: pass@1 = {passed}/{len(rows)} "
          f"= {100 * passed / len(rows):.2f}%")
    print(f"empty: {empty}/{len(rows)}; eos_first={eos}, stop_at_0={stop}")
    print(f"merged preds: {out}")


if __name__ == "__main__":
    main()

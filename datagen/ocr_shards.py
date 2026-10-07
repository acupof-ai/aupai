#!/usr/bin/env python3
"""Convert nvidia/OpenCodeReasoning parquet shards into raw jsonl for build_corpus.

The dataset pairs a competitive-programming statement (input) with an R1
reasoning trace (output: a think section then a python code block) and the
extracted code (solution). A kept document is statement + trace; the code
block at the tail is the answer.

Static quality gate -- the gate short of running tests:
  * ast.parse(solution) succeeds (non-Python rows cut)
  * exactly one think open/close pair, one trailing python block, and that
    block matches the solution field (trace and code agree)
  * reasoning 300..60000 chars, solution 50..20000 chars
  * at most --per_id rows per question id (the dataset ships ~26 solutions per
    question; keep the first K valid in file order to bound size)

Executable verification beyond syntax is NOT attempted: the rows carry no test
cases and split_0 rows cannot be rejoined to their source judge without the
upstream datasets. The gate asserts parseability and trace/code agreement, not
correctness; the 13-gram decontam pass handles eval leakage.

Output rows are {"content", "source", "url"} -- the build_corpus jsonl contract.

Usage (pod):
  python datagen/ocr_shards.py --in 'data/raw/opencode_reasoning/split_0_*.parquet' \\
      --out data/raw/opencode_reasoning/jsonl --per_id 3 --max_chars 24000
"""
import argparse
import ast
import glob
import json
import os
import re
import sys

# built by concat so the literal tag never sits in the source
T_OPEN = "<" + "think" + ">"
T_CLOSE = "<" + "/think" + ">"
PYBLOCK = re.compile(r"```python\n(.*?)```", re.S)


def reason_and_code(output, solution):
    """Return (reasoning_text, code) or None when the trace structure is off."""
    if output.count(T_OPEN) != 1 or output.count(T_CLOSE) != 1:
        return None
    i, j = output.find(T_OPEN), output.find(T_CLOSE)
    if not (i == 0 and i < j):
        return None
    blocks = PYBLOCK.findall(output)
    if not blocks:
        return None
    code = blocks[-1].strip()
    if code != solution.strip():
        return None
    return output[i + len(T_OPEN):j].strip(), code


def valid_solution(solution):
    if not (50 <= len(solution) <= 20_000):
        return False
    try:
        ast.parse(solution)
    except SyntaxError:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="pat", required=True, help="parquet glob")
    ap.add_argument("--out", required=True, help="output dir for raw jsonl shards")
    ap.add_argument("--per_id", type=int, default=3)
    ap.add_argument("--min_reason", type=int, default=300)
    ap.add_argument("--max_reason", type=int, default=60_000)
    ap.add_argument("--max_chars", type=int, default=24_000, help="per-doc cap")
    ap.add_argument("--shard_mb", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sft_out", default="",
                    help="if set, ALSO write {prompt,output,source} SFT pairs here "
                         "(output = reasoning without think tags + python block); "
                         "separate from the pretraining {content} stream")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(args.pat))
    if not files:
        sys.exit(f"no parquet files: {args.pat}")

    counts = {}
    seen = kept = n_shard = bytes_written = 0
    reasons = dict(structure=0, syntax=0, reason_len=0, per_id=0,
                   placeholder=0, cap=0)
    fout = None
    sfout = None

    def open_shard():
        nonlocal fout, bytes_written
        path = os.path.join(args.out, f"ocr_{n_shard:03d}.jsonl")
        fout = open(path, "w", encoding="utf-8")
        bytes_written = 0

    if args.sft_out:
        os.makedirs(args.sft_out, exist_ok=True)
        sfout = open(os.path.join(args.sft_out, "ocr_sft_pairs.jsonl"), "w", encoding="utf-8")

    def write(row):
        nonlocal fout, n_shard, bytes_written, kept
        if fout is not None and bytes_written >= args.shard_mb * 1_000_000:
            fout.close()
            fout = None
            n_shard += 1
        if fout is None:
            open_shard()
        line = json.dumps(row, ensure_ascii=False) + "\n"
        fout.write(line)
        bytes_written += len(line.encode("utf-8"))
        kept += 1

    stop = False
    for fp in files:
        pf = pq.ParquetFile(fp)
        src = os.path.basename(fp).split(".")[0]
        for batch in pf.iter_batches(batch_size=4096):
            for r in batch.to_pylist():
                seen += 1
                if args.limit and seen > args.limit:
                    stop = True
                    break
                problem = (r.get("input") or "").strip()
                if len(problem) < 50 or problem == "-":
                    reasons["placeholder"] += 1
                    continue
                sol = r.get("solution") or ""
                if not valid_solution(sol):
                    reasons["syntax"] += 1
                    continue
                got = reason_and_code(r.get("output") or "", sol)
                if got is None:
                    reasons["structure"] += 1
                    continue
                reason, code = got
                if not (args.min_reason <= len(reason) <= args.max_reason):
                    reasons["reason_len"] += 1
                    continue
                qid = r.get("id") or ""
                n = counts.get(qid, 0)
                if n >= args.per_id:
                    reasons["per_id"] += 1
                    continue
                content = f"{problem}\n\n{reason}\n\n```python\n{code.strip()}\n```\n"
                if len(content) > args.max_chars:
                    reasons["cap"] += 1
                    continue
                counts[qid] = n + 1
                write({"content": content, "source": src, "url": ""})
                if sfout is not None:
                    answer = f"{reason.strip()}\n\n```python\n{code.strip()}\n```\n"
                    sfout.write(json.dumps(
                        {"prompt": problem, "output": answer, "source": "opencode_reasoning"},
                        ensure_ascii=False) + "\n")
        if stop:
            break
    if fout is not None:
        fout.close()
    if sfout is not None:
        sfout.close()

    print(f"seen={seen} kept={kept} questions={len(counts)}")
    print("reject=" + json.dumps(reasons))


if __name__ == "__main__":
    main()

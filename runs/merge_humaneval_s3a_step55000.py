#!/usr/bin/env python3
import json
import time
from pathlib import Path

ROOT = Path("/work/aupai")
CKPT = "ckpt_s3a_seg1.pt.step55000"
RUN = "s3a_step55000_cpu_he"
WORKERS = 4
BASE = ROOT / "data/eval" / f"preds_humaneval_{CKPT}.rstripnl"
OUT = ROOT / "data/eval" / f"preds_humaneval_{CKPT}.rstripnl.{RUN}.merged.jsonl"

header = None
rows = []
for worker in range(WORKERS):
    path = Path(f"{BASE}.shard{worker}of{WORKERS}.{RUN}.jsonl")
    if not path.exists():
        raise SystemExit(f"MISSING {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    worker_header = json.loads(lines[0])
    if not worker_header.get("_header"):
        raise SystemExit(f"BAD HEADER {path}")
    if worker_header.get("n_problems") != 164:
        raise SystemExit(f"BAD n_problems {path}: {worker_header.get('n_problems')}")
    if header is None:
        header = worker_header
    worker_rows = [json.loads(line) for line in lines[1:]]
    if not worker_rows:
        raise SystemExit(f"NO PREDICTIONS {path}")
    rows.extend(worker_rows)

task_ids = [row["task_id"] for row in rows]
if len(task_ids) != 164 or len(set(task_ids)) != 164:
    raise SystemExit(f"BAD COVER: rows={len(rows)} unique={len(set(task_ids))}")

rows.sort(key=lambda row: int(row["task_id"].split("/", 1)[1]))
header["merged_from"] = WORKERS
header["merged_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
with OUT.open("w", encoding="utf-8") as f:
    f.write(json.dumps(header, ensure_ascii=False) + "\n")
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

npass = sum(bool(row.get("ok")) for row in rows)
nempty = sum(bool(row.get("empty")) for row in rows)
neos = sum(row.get("empty_kind") == "eos_first" for row in rows)
nstop = sum(row.get("empty_kind") == "stop_at_0" for row in rows)
print(f"HUMANEVAL pass@1 (greedy) = {npass}/164 = {100 * npass / 164:.2f}%")
print(f"empty: {nempty}/164 = {100 * nempty / 164:.1f}%; eos_first={neos}, stop_at_0={nstop}")
print(f"merged preds: {OUT}")

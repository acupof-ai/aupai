#!/usr/bin/env python3
"""Tiered CPU evaluation for periodic checkpoints.

Layer 1: CPU checkpoint smoke.
Layer 2: quick HumanEval generation on a fixed small task set.
The script never uses CUDA. It never restarts training.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cpu_ckpt_smoke as smoke  # noqa: E402

RUNS = ROOT / "runs"
EVAL = ROOT / "data" / "eval"


def ledger_path(name):
    return RUNS / f"tiered_ckpt_eval_{name}.jsonl"


def done_checkpoints(name):
    path = ledger_path(name)
    out = set()
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("status") in ("ok", "warn") and row.get("ckpt"):
            out.add(row["ckpt"])
    return out


def latest_smoke_row(name, ckpt):
    path = RUNS / f"cpu_ckpt_smoke_{name}.jsonl"
    out = None
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("ckpt") == ckpt:
            out = row
    return out


def append_row(name, row):
    path = ledger_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_smoke(name, threads):
    return smoke.run_pending(name, threads)


def quick_worker_command(ckpt, run, queue_dir, workers, idx, threads, quick_tasks):
    return [
        sys.executable, "eval/humaneval_gen.py",
        "--ckpt", ckpt,
        "--device", "cpu",
        "--threads", str(threads),
        "--rstrip_nl",
        "--first", str(quick_tasks),
        "--queue_dir", queue_dir,
        "--shard_i", str(idx),
        "--shard_n", str(workers),
        "--run", run,
        "--force",
    ]


def run_quick(ckpt, step, workers, threads, quick_tasks):
    run = f"quick{quick_tasks}_{step}"
    queue_dir = str(RUNS / "queues" / run)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "PYTHONUNBUFFERED": "1",
    })
    processes = []
    logs = []
    try:
        for idx in range(workers):
            log_path = RUNS / f"{run}.w{idx}.log"
            pid_path = RUNS / f"{run}.w{idx}.pid"
            log = log_path.open("w", encoding="utf-8")
            cmd = quick_worker_command(
                ckpt, run, queue_dir, workers, idx, threads, quick_tasks
            )
            proc = subprocess.Popen(
                cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT
            )
            pid_path.write_text(str(proc.pid), encoding="utf-8")
            processes.append((idx, proc))
            logs.append(log)
        failed = []
        for idx, proc in processes:
            rc = proc.wait()
            if rc != 0:
                failed.append(f"worker{idx} rc={rc}")
        if failed:
            return None, run, failed
        merger = [
            sys.executable, str(RUNS / "merge_humaneval_shards.py"),
            "--ckpt", ckpt,
            "--run", run,
            "--workers", str(workers),
            "--expected", str(quick_tasks),
        ]
        merged = subprocess.run(merger, cwd=str(ROOT), capture_output=True, text=True)
        if merged.returncode != 0:
            return None, run, [merged.stdout, merged.stderr]
        merged_path = EVAL / f"preds_humaneval_{ckpt}.rstripnl.{run}.merged.jsonl"
        with merged_path.open(encoding="utf-8") as fh:
            header = json.loads(fh.readline())
        return header, run, []
    finally:
        for _idx, proc in processes:
            if proc.poll() is None:
                proc.kill()
        for log in logs:
            log.close()


def evaluate(step, path, name, workers, smoke_threads, code_threads, quick_tasks):
    t0 = time.time()
    ckpt = os.path.basename(path)
    row = {"ckpt": ckpt, "step": step, "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    run_smoke(name, smoke_threads)
    smoke_row = latest_smoke_row(name, ckpt)
    row["smoke_status"] = smoke_row.get("status") if smoke_row else None
    if smoke_row:
        row["mean_nll"] = smoke_row.get("mean_nll")
        row["canary_correct"] = smoke_row.get("canary", {}).get("correct")
    if not smoke_row or smoke_row.get("status") != "ok":
        row.update(status="fail", errors=["CPU smoke failed"], secs=round(time.time() - t0, 2))
        return row

    header, run, errors = run_quick(ckpt, step, workers, code_threads, quick_tasks)
    row["quick_run"] = run
    if errors:
        row.update(status="fail", errors=errors, secs=round(time.time() - t0, 2))
        return row

    total = int(header.get("n_rows", 0))
    passed = int(header.get("pass", 0))
    empty = int(header.get("empty", 0))
    row.update({
        "quick_pass": passed,
        "quick_total": total,
        "quick_pass_rate": round(100 * passed / total, 2) if total else None,
        "quick_empty": empty,
    })
    warnings = []
    if passed == 0:
        warnings.append("quick HumanEval pass is zero")
    if total and empty / total >= 0.25:
        warnings.append("quick HumanEval empty rate is high")
    row["status"] = "warn" if warnings else "ok"
    if warnings:
        row["warnings"] = warnings
    row["secs"] = round(time.time() - t0, 2)
    return row


def run_pending(name, workers, smoke_threads, code_threads, quick_tasks):
    done = done_checkpoints(name)
    pending = [(s, p) for s, p in smoke.discover(name) if os.path.basename(p) not in done]
    for step, path in pending:
        row = evaluate(step, path, name, workers, smoke_threads, code_threads, quick_tasks)
        append_row(name, row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return len(pending)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke_threads", type=int, default=4)
    ap.add_argument("--code_threads", type=int, default=2)
    ap.add_argument("--quick_tasks", type=int, default=8)
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--watch", action="store_true")
    args = ap.parse_args()

    if args.once or not args.watch:
        run_pending(
            args.name, args.workers, args.smoke_threads,
            args.code_threads, args.quick_tasks
        )
        return
    while True:
        run_pending(
            args.name, args.workers, args.smoke_threads,
            args.code_threads, args.quick_tasks
        )
        time.sleep(args.poll)


if __name__ == "__main__":
    main()

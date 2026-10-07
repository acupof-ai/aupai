#!/usr/bin/env python3
"""8-GPU HumanEval eval for v42s2: 8 workers (cuda:0..7) share a dynamic queue.
Mirrors eval_watch.run_cpu ledger semantics: status ok/fail, dc156/all164.
Usage: python3 gpu_eval8.py <step> [<step> ...]
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = "/work/aupai"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import eval_watch as ew  # noqa: E402

NAME = "v42s2"
DEVICES = 8


def ckpt_for(step):
    pat = os.path.join(ROOT, f"ckpt_{NAME}.pt.step{step}")
    return pat if os.path.exists(pat) else None


def run_step(step, max_new=280, timeout=21600):
    ckpt = ckpt_for(step)
    if not ckpt:
        return {"name": NAME, "step": step, "status": "fail",
                "error": "checkpoint missing", "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    queue = os.path.join(ROOT, "runs", f"he_queue_{NAME}_gpu_step{step}")
    shutil.rmtree(queue, ignore_errors=True)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "8"
    procs, logs = [], []
    t0 = time.time()
    for i in range(DEVICES):
        env_i = dict(env)
        env_i["CUDA_VISIBLE_DEVICES"] = str(i)  # each worker sees only its own card
        cmd = [sys.executable, os.path.join(ROOT, "eval", "humaneval_gen.py"),
               "--ckpt", ckpt, "--device", "cuda:0", "--rstrip_nl",
               "--max_new", str(max_new), "--force",
               "--queue_dir", queue, "--shard_i", str(i), "--shard_n", str(DEVICES)]
        log = os.path.join(ROOT, "runs", f"he_{NAME}_gpu_step{step}_w{i}.log")
        fh = open(log, "w", encoding="utf-8")
        logs.append(fh)
        procs.append((i, subprocess.Popen(cmd, env=env_i, stdout=fh, stderr=subprocess.STDOUT), log))
    row = {"name": NAME, "step": step, "ckpt": os.path.basename(ckpt),
           "device": "gpu8", "workers": DEVICES, "threads": 8,
           "secs": 0, "rstrip_nl": True, "queue": os.path.basename(queue)}
    bad = []
    for i, p, log in procs:
        rc = p.wait(timeout=timeout)
        logs[procs.index((i, p, log))].close()
        if rc != 0:
            bad.append(f"worker {i} rc={rc} (see {os.path.basename(log)})")
    row["secs"] = round(time.time() - t0, 1)
    try:
        outs = ew.shard_paths(ckpt, DEVICES)
        rows, missing = ew.merge_shards(outs)
        if bad or missing:
            row.update(status="fail",
                       error=("; ".join(bad) +
                              (f"; no preds from {[os.path.basename(p) for p in missing]}"
                               if missing else ""))[:400],
                       got_tasks=len(rows))
            return row
        row.update(status="ok", **ew.score_preds(None, ew.excluded(), rows=rows, expect=164))
    except SystemExit as e:
        row.update(status="fail", error=str(e)[:400])
    row["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return row


def main():
    steps = [int(a) for a in sys.argv[1:]]
    if not steps:
        print("usage: gpu_eval8.py <step> [<step> ...]")
        sys.exit(2)
    for s in steps:
        r = run_step(s)
        ew.append(r)
        if r["status"] == "ok":
            print(f"step {s}: OK dc={r['pct_dc']}% ({r['pass_dc']}/{r['n_dc']}) "
                  f"all={r['pct_all']}% ({r['pass_all']}/{r['n_all']}) {r['secs']}s", flush=True)
        else:
            print(f"step {s}: FAIL {r.get('error','')[:120]}", flush=True)


if __name__ == "__main__":
    main()

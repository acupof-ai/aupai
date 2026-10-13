#!/usr/bin/env python3
# restartable: pure aggregation over merged preds files; prints a command and writes one small JSON;
# it launches nothing, so an interrupt leaves no state behind.
"""Plateau RL gate: decide from MEASURED pass rates whether execution-reward RL can move the
model, then print the exact GSPO launch. It never launches training itself.

WHY THIS FILE EXISTS. The case study shows 83/129 HumanEval tasks fail at every checkpoint
and 39 pass at every checkpoint. RL with a binary execution reward only learns on groups that
contain BOTH a pass and a fail (mean-only advantage = 0 otherwise). So the right question is
not "is pass@10 - pass@1 >= 15pt" (the old gate) but "how many gradient-carrying groups does
one optimizer step get", which algorithms/rl_admission.py already computes as f*B.

TWO MEASUREMENTS, BOTH REQUIRED:
  pool    n>=8 samples per task on the RL TRAINING pool (data/rl/rl_code_*.jsonl via
          eval/humaneval_sample.py-compatible merged preds). This sets f, the trainable
          fraction, on the data RL will actually see.
  eval    the same on HumanEval (merged preds). Reported only; it never feeds the gate, so the
          eval set never steers training.
A pool file whose task ids overlap the eval file refuses: that is train/test leakage.

OUTPUT: admission numbers for both, a GO / BELOW-THE-BAR verdict on the pool, the
batch that would give f*B >= 1, and the rl_code_trainer.py command with conservative
settings (lr 1e-6, kl 0.02, group 8). Engram layers stay 1,5,9,13,17,21; this file has no
flag that touches them.

    python3 scripts/plateau_rl_gate.py --pool_preds runs/rlpool_s62269_n8.jsonl \
        --eval_preds runs/he_s62269_n8.jsonl --ckpt /work/aupai/ckpt_s3a_seg1.pt.interrupt.step62269
    python3 scripts/plateau_rl_gate.py --selftest
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from algorithms import rl_admission as ra  # noqa: E402


def bucket(by_task):
    """Count tasks by observed outcome: all-fail / mixed / all-pass. No model assumption."""
    c = {"all_fail": 0, "mixed": 0, "all_pass": 0}
    for v in by_task.values():
        s = sum(v)
        c["all_fail" if s == 0 else "all_pass" if s == len(v) else "mixed"] += 1
    return c


def gate(pool_by, pool_n, eval_by=None, eval_n=None, g=8, b=4, world=8):
    if eval_by is not None:
        leak = set(pool_by) & set(eval_by)
        if leak:
            raise SystemExit(f"refusing: {len(leak)} task ids appear in BOTH pool and eval "
                             f"(e.g. {sorted(leak)[:3]}). RL must not train on the eval set.")
    if pool_n < g:
        raise SystemExit(f"pool preds have n={pool_n} samples per task; need n >= group size {g}")
    pool = ra.report(pool_by, pool_n, g=g, b=b)
    out = {"pool": pool, "pool_buckets": bucket(pool_by), "verdict": ra._verdict(pool),
           "group_size": g, "batch": b, "world": world}
    if eval_by is not None:
        out["eval"] = ra.report(eval_by, eval_n, g=g, b=b)
        out["eval_buckets"] = bucket(eval_by)
    return out


def launch_cmd(ckpt, batch, g, steps, pool_dir):
    return ["torchrun", "--nproc_per_node=8", "algorithms/rl_code_trainer.py",
            "--resume", ckpt, "--steps", str(steps), "--batch", str(batch),
            "--group_size", str(g), "--temperature", "0.8", "--top_p", "0.95",
            "--lr", "1e-6", "--kl_beta", "0.02", "--clip_eps", "0.2",
            "--pool_dir", pool_dir, "--out", "/work/aupai/ckpt_plateau_rl.pt"]


def _fmt(r):
    p, bk = r["pool"], r["pool_buckets"]
    lines = [f"pool: tasks={sum(bk.values())} all_fail={bk['all_fail']} mixed={bk['mixed']} "
             f"all_pass={bk['all_pass']}", ra.fmt(p), f"verdict: {r['verdict']}"]
    if "eval" in r:
        e = r["eval_buckets"]
        lines.append(f"eval (report only): all_fail={e['all_fail']} mixed={e['mixed']} all_pass={e['all_pass']}")
    return "\n".join(lines)


def _selftest():
    # 30 pool tasks: 10 always fail, 10 mixed at p=.5, 10 always pass; n=8.
    pool = {f"P/{i}": [False] * 8 for i in range(10)}
    pool.update({f"P/{i}": [True, False] * 4 for i in range(10, 20)})
    pool.update({f"P/{i}": [True] * 8 for i in range(20, 30)})
    r = gate(pool, 8)
    assert r["pool_buckets"] == {"all_fail": 10, "mixed": 10, "all_pass": 10}, r["pool_buckets"]
    f = r["pool"]["kept_groups_per_rank_step"] / 4
    # plug-in f = mean P_mixed over tasks = (10*0 + 10*(1-2*.5**8) + 10*0)/30
    assert abs(f - (1 - 2 * 0.5 ** 8) / 3) < 1e-9, f
    # Leakage refuses.
    try:
        gate(pool, 8, {"P/0": [True] * 8}, 8)
        raise AssertionError("overlap must refuse")
    except SystemExit:
        pass
    # n below group size refuses.
    try:
        gate({"a": [True, False]}, 2)
        raise AssertionError("small n must refuse")
    except SystemExit:
        pass
    cmd = launch_cmd("ck", 4, 8, 200, "data/rl")
    assert "--lr" in cmd and cmd[cmd.index("--lr") + 1] == "1e-6"
    print("plateau_rl_gate selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool_preds")
    ap.add_argument("--eval_preds")
    ap.add_argument("--ckpt")
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--pool_dir", default="data/rl")
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return 0
    if not (a.pool_preds and a.ckpt):
        ap.error("--pool_preds and --ckpt are required")
    pool_by, pool_n = ra.read_preds(a.pool_preds)
    eval_by = eval_n = None
    if a.eval_preds:
        eval_by, eval_n = ra.read_preds(a.eval_preds)
    r = gate(pool_by, pool_n, eval_by, eval_n, g=a.group_size, b=a.batch)
    print(_fmt(r))
    need = r["pool"]["batch_needed_for_one_kept_group"]
    batch = a.batch if r["verdict"] == "GO" else (need or a.batch)
    if need is None and r["verdict"] != "GO":
        print("no launch: no pool group is ever mixed")
    else:
        print("launch (NOT run by this script): " + " ".join(launch_cmd(a.ckpt, batch, a.group_size,
                                                                        a.steps, a.pool_dir)))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

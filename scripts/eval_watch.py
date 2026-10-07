#!/usr/bin/env python3
# restartable: scores one checkpoint at a time and appends one row when it finishes; an
# interrupt loses at most the in-flight eval, which the next pass redoes.
"""Score HumanEval automatically at fixed step intervals (de, user order 2026-10-01).

The gate run saves every 2000 steps and nothing scored those checkpoints: `score_matrix`'s
newest rows belong to the retired run. This watches for new checkpoints and scores each one.

    python3 scripts/eval_watch.py --name v42_gate_1001r --every 2000 --device cuda:7
    python3 scripts/eval_watch.py --name v42_gate_1001r --once --step 4000 --device cuda:7
    python3 scripts/eval_watch.py --selftest          # no GPU, no pod, no checkpoint

**It reports two numbers and names both denominators.** The gate is pass@1 over the
DECONTAMINATED 156, not over 164: the 8 task_ids in
`runs/contam_r3_he_union.json#r3_humaneval_union` each carry a whitespace-13-gram hit in an
r3 training domain and are excluded (user ruling 2026-09-30,
`runs/prereg.jsonl#v41_ced_0923` @amended_9). 30% of 156 is 46.8 tasks against 49.2 of 164,
so a score over 164 does not answer this gate. Both are recorded, each with its n, because a
bare percentage cannot say which question it answered.

**It always passes --rstrip_nl.** That is the gate column since 2026-09-14: the canonical
prompt's trailing newline makes the model emit EOS at token 0, and without the flag the run
returns a plausible 0.00% that measures the prompt format rather than the model.

It NEVER touches the training process -- no kill, no restart, no signal. It only reads
checkpoints and writes its own rows. A watchdog that restarts its subject is not a watchdog.

Co-residency: this is a GENERATIVE eval over 164 tasks, not the cheap checkpoint-only class,
so it costs real compute on a card the training run is using. --device is REQUIRED for that
reason: nothing lands on a card by default.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "runs", "eval_watch.jsonl")
UNION = os.path.join(ROOT, "runs", "contam_r3_he_union.json")
MAX_PASSES = 2000  # every loop here carries an iteration cap


def excluded():
    """The 8 contaminated task_ids. Missing file REFUSES: scoring over 164 while reporting
    156 would be the wrong number wearing the right label."""
    with open(UNION, encoding="utf-8") as fh:
        ids = json.load(fh)["r3_humaneval_union"]
    if len(ids) != 8:
        raise SystemExit(f"{UNION} lists {len(ids)} excluded task_ids, expected 8 -- the "
                         f"denominator changed; re-read the ruling before scoring")
    return set(ids)


def score_preds(path, drop, rows=None, expect=None):
    """pass@1 over all tasks and over the decontaminated subset, each with its n.

    `rows` takes an already-merged {task_id: row} mapping instead of a file. `expect` is the
    task count the data file holds: when given, a short set REFUSES rather than reporting a
    percentage over whatever survived.
    """
    if rows is None:
        rows = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or "_header" in line:
                    continue
                r = json.loads(line)
                if "task_id" not in r:
                    continue
                # a sampled preds file repeats a task_id; one row per task is asserted below
                # rather than silently averaged
                if r["task_id"] in rows:
                    raise SystemExit(f"{path} has repeated task_ids ({r['task_id']}): this "
                                     f"scorer assumes one row per task; use the n>1 path for "
                                     f"pass@k")
                rows[r["task_id"]] = r
    if expect is not None and len(rows) != expect:
        raise SystemExit(f"got {len(rows)} tasks, expected {expect}: a hole in the set shrinks "
                         f"the denominator, and 40/120 reads better than 40/156. Refusing.")
    seen = {}
    for tid, r in rows.items():
        seen.setdefault(tid, []).append(bool(r.get("ok")))
    all_ids = sorted(seen)
    keep = [t for t in all_ids if t not in drop]
    n_all = len(all_ids)
    n_keep = len(keep)
    p_all = sum(seen[t][0] for t in all_ids)
    p_keep = sum(seen[t][0] for t in keep)
    return {
        "pass_dc": p_keep, "n_dc": n_keep,
        "pct_dc": round(100.0 * p_keep / n_keep, 2) if n_keep else None,
        "pass_all": p_all, "n_all": n_all,
        "pct_all": round(100.0 * p_all / n_all, 2) if n_all else None,
        "excluded_present": sorted(set(all_ids) & drop),
    }


def done_steps(name):
    """Steps already scored, from the ledger. The ledger is the only memory: a rerun must
    not re-score what it has a row for, and a missing ledger means nothing is scored."""
    out = set()
    try:
        with open(LEDGER, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("name") == name and r.get("status") == "ok":
                    out.add(int(r["step"]))
    except OSError:
        pass
    return out


def checkpoints(name, every):
    """{step: path} for saved checkpoints at a multiple of `every`."""
    found = {}
    for p in glob.glob(os.path.join(ROOT, f"ckpt_{name}.pt.step*")):
        m = re.search(r"\.step(\d+)$", p)
        if m and int(m.group(1)) % every == 0:
            found[int(m.group(1))] = p
    return found


def shard_paths(ckpt, workers):
    """Where humaneval_gen writes each shard, mirroring its own `preds_path` + shard_label."""
    base = os.path.join(ROOT, "data", "eval",
                        f"preds_humaneval_{os.path.basename(ckpt.rstrip('/'))}.rstripnl")
    if workers <= 1:
        return [base + ".jsonl"]
    return [f"{base}.shard{i}of{workers}.jsonl" for i in range(workers)]


def run_sharded(name, step, ckpt, workers, threads, data, max_new, timeout):
    """K cardless workers sharing a dynamic task queue, then one merged score.

    Measured 2026-10-01 at 48 threads: 123.5 s/task, so 164 tasks in ONE process is ~5.6 h --
    longer than the 2000-step save interval, which would leave the eval permanently behind the
    run. The pod has 180 cores at load ~11 while all 8 GPUs are pinned, so the fix is width,
    not a longer interval.

    --queue_dir, not fixed positions: humaneval_gen's own help says per-task cost varies ~2x,
    so fixed modulo positions idle the fast workers.
    """
    queue = os.path.join(ROOT, "runs", f"he_queue_{name}_step{step}")
    shutil.rmtree(queue, ignore_errors=True)  # stale tasks from a killed run must not shrink the set
    outs = shard_paths(ckpt, workers)
    env = {k: v for k, v in os.environ.items() if k != "ALLOW_UNISOLATED"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = str(threads)
    procs, logs = [], []
    for i in range(workers):
        cmd = [sys.executable, os.path.join(ROOT, "eval", "humaneval_gen.py"),
               "--ckpt", ckpt, "--device", "cpu", "--threads", str(threads),
               "--rstrip_nl", "--max_new", str(max_new), "--force",
               "--queue_dir", queue, "--shard_i", str(i), "--shard_n", str(workers)]
        if data:
            cmd += ["--data", data]
        log = os.path.join(ROOT, "runs", f"he_{name}_step{step}_w{i}.log")
        # The handle is kept and closed below: a worker writes to it for hours, so it cannot
        # live in a with-block around the Popen.
        fh = open(log, "w", encoding="utf-8")  # noqa: SIM115
        logs.append(fh)
        procs.append((i, subprocess.Popen(cmd, env=env, stdout=fh,
                                          stderr=subprocess.STDOUT), log))
    bad = []
    try:
        for i, p, log in procs:
            if p.wait(timeout=timeout) != 0:
                bad.append(f"worker {i} rc={p.returncode} (see {os.path.basename(log)})")
    finally:
        for fh in logs:
            fh.close()
    return outs, bad, queue


def merge_shards(outs):
    """One row per task_id across the shard files. A task NO shard produced is the failure
    this exists to catch: a queue worker that died early leaves a hole, and scoring the
    survivors would silently shrink the denominator -- a 40/120 reads better than 40/156."""
    rows, where = {}, {}
    missing_files = [p for p in outs if not os.path.exists(p)]
    for p in outs:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or "_header" in line:
                    continue
                r = json.loads(line)
                if "task_id" not in r:
                    continue
                if r["task_id"] in rows:
                    raise SystemExit(f"task {r['task_id']} produced by two shards "
                                     f"({where[r['task_id']]} and {os.path.basename(p)}): the "
                                     f"queue handed one task out twice")
                rows[r["task_id"]] = r
                where[r["task_id"]] = os.path.basename(p)
    return rows, missing_files


def run_cpu(name, step, ckpt, workers, threads, data, max_new, timeout, expect=164):
    """The CPU path: K sharded workers, merged, scored over both denominators."""
    t0 = time.time()
    outs, bad, queue = run_sharded(name, step, ckpt, workers, threads, data, max_new, timeout)
    secs = round(time.time() - t0, 1)
    row = {"name": name, "step": step, "ckpt": os.path.basename(ckpt), "device": "cpu",
           "workers": workers, "threads": threads, "secs": secs, "rstrip_nl": True,
           "queue": os.path.basename(queue),
           "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    try:
        rows, missing = merge_shards(outs)
        if bad or missing:
            # A dead worker leaves a hole. Report it as a FAILED eval, never as a score over
            # what survived: the denominator is the whole point of this file.
            row.update(status="fail", error=("; ".join(bad) +
                       (f"; no preds from {[os.path.basename(p) for p in missing]}"
                        if missing else ""))[:400], got_tasks=len(rows))
            return row
        row.update(status="ok", **score_preds(None, excluded(), rows=rows, expect=expect))
    except SystemExit as e:
        row.update(status="fail", error=str(e)[:400])
    return row


def run_one(name, step, ckpt, device, data=None, max_new=280, timeout=7200):
    """Score one checkpoint. Returns the ledger row, written by the caller."""
    # humaneval_gen DERIVES its output path from the checkpoint name and the flags that change
    # the prompt; --preds is the opposite flag, a re-score of an existing file. Mirrored from
    # eval/humaneval_gen.py's `preds_path`, so a change there must change here -- asserted
    # against the real file after the run rather than trusted.
    preds = os.path.join(ROOT, "data", "eval",
                         f"preds_humaneval_{os.path.basename(ckpt.rstrip('/'))}.rstripnl.jsonl")
    cmd = [sys.executable, os.path.join(ROOT, "eval", "humaneval_gen.py"),
           "--ckpt", ckpt, "--device", device, "--rstrip_nl",
           "--max_new", str(max_new), "--force"]
    if data:
        cmd += ["--data", data]
    t0 = time.time()
    # ALLOW_UNISOLATED must never reach the scorer: it runs candidate code, and the sandbox
    # is the only thing between that code and this box.
    env = {k: v for k, v in os.environ.items() if k != "ALLOW_UNISOLATED"}
    if device == "cpu":
        # humaneval_gen refuses --device cpu unless CUDA_VISIBLE_DEVICES is explicitly EMPTY,
        # so a cardless run cannot silently grab a card. Set it rather than make the operator
        # remember it; it is the same statement of intent --device cpu already makes.
        env["CUDA_VISIBLE_DEVICES"] = ""
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    secs = round(time.time() - t0, 1)
    row = {"name": name, "step": step, "ckpt": os.path.basename(ckpt), "device": device,
           "preds": os.path.basename(preds), "secs": secs, "rstrip_nl": True,
           "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
           "cmd": " ".join(cmd[1:])}
    if p.returncode != 0:
        row.update(status="fail", rc=p.returncode,
                   error=(p.stderr or p.stdout).strip()[-400:])
        return row
    if not os.path.exists(preds):
        # The scorer ran and we cannot find its output: the derived path above has drifted
        # from eval/humaneval_gen.py. Say THAT, with the path it actually printed, rather
        # than reporting a failed eval -- the eval succeeded and this scorer lost it.
        said = re.findall(r"preds saved: (\S+)", p.stdout or "")
        row.update(status="fail", rc=0,
                   error=f"ran ok but no preds at the derived path {preds}; humaneval_gen "
                         f"printed {said or 'nothing'} -- mirror its preds_path again")
        return row
    try:
        row.update(status="ok", **score_preds(preds, excluded()))
    except SystemExit as e:
        row.update(status="fail", error=str(e)[:400])
    return row


def append(row):
    """One guarded row. harness_core.append_ledger is the shared write path for every
    session-facing ledger (de-98): it refuses in the integration tree and appends one
    complete line under O_APPEND, so concurrent passes never interleave. On the pod, which
    is not a git repo, the guard fails open -- which is where this actually runs."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from harness_core import append_ledger
    append_ledger(LEDGER, row, what="recording a HumanEval score")


def selftest():
    import tempfile
    drop = {f"HumanEval/{i}" for i in (19, 66, 71, 78, 105, 123, 129, 156)}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "preds.jsonl")
        # 164 tasks: every excluded id passes, and 40 of the kept ones pass. So a score over
        # 164 reads 48/164 = 29.27% (a MISS at the 30% bar) while the gate's own denominator
        # reads 40/156 = 25.64%. The two must not be interchangeable, and the pair differs by
        # more than rounding, which is the point of the known answer.
        with open(p, "w", encoding="utf-8") as fh:
            fh.write('{"_header": "ignored"}\n')
            kept = 0
            for i in range(164):
                tid = f"HumanEval/{i}"
                if tid in drop:
                    ok = True
                else:
                    ok = kept < 40
                    kept += 1 if ok else 0
                fh.write(json.dumps({"task_id": tid, "gen": "x", "ok": ok}) + "\n")
        got = score_preds(p, drop)
        assert got["n_all"] == 164 and got["n_dc"] == 156, got
        assert got["pass_all"] == 48 and got["pct_all"] == 29.27, got
        assert got["pass_dc"] == 40 and got["pct_dc"] == 25.64, got
        assert got["excluded_present"] == sorted(drop), got
        # a repeated task_id must REFUSE, not average: that file answers pass@k, not pass@1
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"task_id": "HumanEval/0", "gen": "y", "ok": False}) + "\n")
        try:
            score_preds(p, drop)
            raise AssertionError("repeated task_id must refuse")
        except SystemExit as e:
            assert "repeated task_ids" in str(e), e
        # the real exclude list must be the 8 the ruling names
        assert excluded() == drop, excluded() ^ drop

        # the sharded path: four files covering 164 tasks merge to one row per task
        shards = []
        for w in range(4):
            sp = os.path.join(d, f"s{w}.jsonl")
            with open(sp, "w", encoding="utf-8") as fh:
                fh.write('{"_header": "ignored"}\n')
                for i in range(w, 164, 4):
                    fh.write(json.dumps({"task_id": f"HumanEval/{i}", "gen": "x",
                                         "ok": i % 4 == 0}) + "\n")
            shards.append(sp)
        merged, miss = merge_shards(shards)
        assert len(merged) == 164 and not miss, (len(merged), miss)
        s4 = score_preds(None, drop, rows=merged, expect=164)
        assert s4["n_all"] == 164 and s4["n_dc"] == 156, s4
        # A DEAD worker leaves a hole. Scoring the survivors would quietly shrink the
        # denominator -- 41 fewer tasks and a percentage that still looks like a measurement.
        lost, miss2 = merge_shards(shards[:3])
        assert len(lost) == 123, len(lost)
        assert miss2 == [], miss2  # all three files exist; the hole is in the task set
        try:
            score_preds(None, drop, rows=lost, expect=164)
            raise AssertionError("a short task set must refuse, not report a percentage")
        except SystemExit as e:
            assert "expected 164" in str(e), e
        # a missing shard FILE is named, not ignored
        _, miss3 = merge_shards(shards + [os.path.join(d, "never.jsonl")])
        assert miss3 == [os.path.join(d, "never.jsonl")], miss3
        # one task handed to two workers must refuse: that is a broken queue, not a tie
        dupshard = os.path.join(d, "dup.jsonl")
        with open(dupshard, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"task_id": "HumanEval/0", "gen": "z", "ok": True}) + "\n")
        try:
            merge_shards(shards + [dupshard])
            raise AssertionError("two shards producing one task must refuse")
        except SystemExit as e:
            assert "two shards" in str(e), e
        # the ledger gates reruns: only status=ok counts as scored
        global LEDGER
        saved, LEDGER = LEDGER, os.path.join(d, "l.jsonl")
        append({"name": "n", "step": 2000, "status": "ok"})
        append({"name": "n", "step": 4000, "status": "fail"})
        append({"name": "other", "step": 6000, "status": "ok"})
        assert done_steps("n") == {2000}, done_steps("n")
        LEDGER = saved
    print("eval_watch selftest OK: 156 vs 164 differ (25.64% vs 29.27%), exclude list is the "
          "ruling's 8, repeated task_id refuses, 4 shards merge to 164, a dead worker's hole "
          "refuses instead of shrinking the denominator, only ok rows count as scored")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v42_gate_1001r")
    ap.add_argument("--every", type=int, default=2000, help="score steps that are a multiple")
    ap.add_argument("--device", help="REQUIRED to run: cpu, or a card (co-resides with training)")
    ap.add_argument("--workers", type=int, default=4,
                    help="cpu only: sharded workers over a dynamic queue. Measured 2026-10-01: "
                         "123.5 s/task at 48 threads, so 164 tasks single-process is ~5.6 h, "
                         "longer than the 2000-step save interval")
    ap.add_argument("--threads", type=int, default=40,
                    help="cpu only: threads per worker. 4x40=160 of the pod's 180 cores, "
                         "leaving the ~11 the training dataloader uses")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--data", default=None)
    ap.add_argument("--max_new", type=int, default=280)
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--step", type=int, default=None, help="score exactly this step")
    ap.add_argument("--interval", type=int, default=600, help="seconds between passes")
    ap.add_argument("--dry-run", action="store_true", help="print what would be scored")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.dry_run and not a.device:
        raise SystemExit("--device is required: this is a generative eval over 164 tasks and "
                         "lands on a card the training run is using. Name the card on purpose.")
    for _ in range(1 if (a.once or a.step) else MAX_PASSES):
        have = checkpoints(a.name, a.every)
        todo = sorted(set(have) - done_steps(a.name))
        if a.step is not None:
            todo = [a.step] if a.step in have else []
            if not todo:
                raise SystemExit(f"no checkpoint for step {a.step}: have {sorted(have)}")
        if a.dry_run:
            print(f"would score {todo} of {sorted(have)} (scored: {sorted(done_steps(a.name))})")
            return
        for step in todo:
            print(f"[{time.strftime('%H:%M:%S', time.gmtime())}Z] scoring step "
                  f"{step}", flush=True)
            if a.device == "cpu":
                row = run_cpu(a.name, step, have[step], a.workers, a.threads, a.data,
                              a.max_new, a.timeout)
            else:
                row = run_one(a.name, step, have[step], a.device, a.data, a.max_new,
                              a.timeout)
            append(row)
            if row["status"] == "ok":
                print(f"  -> step {step}: pass@1 {row['pct_dc']}% ({row['pass_dc']}/"
                      f"{row['n_dc']} decontaminated) · {row['pct_all']}% ({row['pass_all']}/"
                      f"{row['n_all']} all) · {row['secs']}s", flush=True)
            else:
                print(f"  -> step {step} FAILED: {row.get('error', '')[:200]}", flush=True)
        if a.once or a.step:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""E6: full benchmarks -- sharded orchestration, then a merge that refuses.

Contract: docs/standards/eval_ladder_v1_1009.md sections 5 (E6 row), 7, 8;
handover doc sections 6 (E6 row), 7 (multi-GPU rules), 5 (P2).

E6 IS ORCHESTRATION AND VALIDATION, NOT A REWRITTEN EVALUATOR. Every number comes
from an evaluator that already exists and is already in use:

    GSM8K  (1319)      eval/gsm8k_shard.py    -> data/eval/preds_<ckpt>[.<i>][.<run>].jsonl
    MATH-500 EN (500)  eval/math_hard.py      -> data/eval/hard_<ckpt>[.<i>][.<run>].jsonl
    HumanEval (164)    eval/humaneval_gen.py  -> data/eval/preds_humaneval_<ckpt><arm>.shard<i>of<n>[.<run>].jsonl
    MC (retained)      eval/run_eval.py       -> stdout percent lines, no artifact
    merge arithmetic   eval/e0_merge_score.py (exact cover + CLEAN denominators)
    MC runner          eval/score_matrix.py:metric_mc calls eval/run_eval.py the same way

`cli_contract_selftest()` asserts that every flag this module passes still appears in
the runner's own source text, so a renamed flag fails on CPU instead of after an
eight-card launch has already loaded a 15 GB checkpoint.

THE MERGE ACCEPTS A BENCHMARK ONLY WHEN ALL OF THESE HOLD, and otherwise writes no
total at all (exit 3):

  1. shard coverage   every i in 0..n-1 has a record, an artifact, no ID error, exit 0
  2. exact ID set     ladder_common.verify_coverage over the declared ID space
  3. row count        merged rows == the declared denominator
  4. denominator      HumanEval additionally pins clean 156 = 164 - 8 excluded
  5. seal             per-shard provenance complete: gpu, pid, claim, cmd, log, input
                      sha256, shard i/n, ID list, timestamps, artifact path

Section 7 of the handover doc adds two rules enforced here rather than believed: E6
does not start before E1/E3/E4/E5 pass on the same checkpoint (REFUSE, exit 2; the
waiver is an explicit, recorded flag), and one worker per shard on one card each, so
n_shards <= len(gpus).

`--limit N` MUST BOUND THE CHILD'S WORK, not only the IDs the merge expects. Each
runner bounds differently and the plan records which mechanism it used: a derived
dataset view for gsm8k (the runner module is loaded unchanged, its LOCAL pointed at
the view), `--data` for MATH-500, `--first` for HumanEval, `--benchmarks` for MC. A
benchmark with no mechanism refuses a --limit run. Before that rule existed, a
`--limit 20` smoke silently scored 192 of 1319 items.

`items.jsonl` carries the contract's nine fields per item: id, prompt (verbatim),
response_raw, parsed, passed, stop_reason, n_tokens, fail_class, meta (shard index,
ID list, elapsed, child exit code). `shards/<i>.json` carries the same provenance for
an auditor: gpu, pid, claim name, start command, input hash, shard_i/shard_n.

Exit codes (the ladder's, same as every stage): 0 pass, 2 REFUSE, 3 gate failed, 1 crash.

Usage:
    python3 eval/eval_ladder.py e6 --merge-selftest                     # CPU controls
    python3 eval/ladder/e6_bench.py --ckpt ckpt_v42s2.pt --dry-run
    python3 eval/ladder/e6_bench.py --ckpt ckpt_v42s2.pt --shards 8 --gpus 0,1,2,3,4,5,6,7
    python3 eval/ladder/e6_bench.py --ckpt ckpt_v42s2.pt --merge-only
    python3 eval/ladder/e6_bench.py --ckpt ckpt_v42s2.pt --tag <ckpt>.<name> \
        --only gsm8k --limit 20 --shards 1 --gpus 3        # single-card smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval import ladder_common as lc  # noqa: E402

# ---------------------------------------------------------------- contract numbers
# Handover doc section 6, E6 row. These are the gate: a run whose declared denominator
# disagrees with them can still be merged and inspected, but it is never an E6 total.
GSM8K_DENOM = 1319
MATH500_DENOM = 500
HUMANEVAL_DENOM = 164
HUMANEVAL_CLEAN = 156
HE_EXCLUDE = ["HumanEval/105", "HumanEval/123", "HumanEval/129", "HumanEval/156",
              "HumanEval/19", "HumanEval/66", "HumanEval/71", "HumanEval/78"]
BENCH_ORDER = ["gsm8k", "math500_en", "humaneval", "mc"]

# eval/run_eval.py:137-153. Only sets that run without HF egress are defaulted:
# hellaswag/piqa are out of ALL_BENCHMARKS for that reason. id_space_selftest()
# re-reads these pairs out of run_eval.py's source so this copy cannot rot silently.
MC_DEFAULT = ["arc-easy", "arc-challenge", "winogrande", "boolq", "openbookqa",
              "mmlu", "ceval"]
MC_DISPLAY = {
    "hellaswag": "HellaSwag",
    "piqa": "PIQA",
    "arc-easy": "ARC-Easy",
    "arc-challenge": "ARC-Challenge",
    "winogrande": "WinoGrande",
    "boolq": "BoolQ",
    "openbookqa": "OpenBookQA",
    "mmlu": "MMLU",
    "ceval": "C-Eval (zh)",
}

GSM8K_DATA = ROOT / "data" / "eval" / "gsm8k_test.jsonl"
HUMANEVAL_DATA = ROOT / "data" / "eval" / "humaneval" / "humaneval_164.jsonl"
MATH500_RAW = ROOT / "data" / "eval" / "math_500_en_test.jsonl"
MATH500_FOR_EVAL = ROOT / "data" / "eval" / "math_500_en_for_eval.jsonl"
TOKENIZER = ROOT / "data" / "tokenizer.json"
CLAIM_TOOL = ROOT / "scripts" / "card_claim.py"

DEFAULT_TAG_SELFTEST = "_e6_merge_selftest"

# --limit must bound the WORK the child does, not only the ID space the merge expects.
# Each runner bounds differently, and the plan records which mechanism was used:
#   gsm8k      eval/gsm8k_shard.py reads data/eval/gsm8k_test.jsonl and takes no limit
#              flag, so the child starts through GSM8K_VIEW_SHIM: the runner module is
#              loaded unchanged, with eval.gsm8k.LOCAL pointing at a truncated view of
#              its own dataset. No evaluator file is edited.
#   math500_en eval/math_hard.py --data <truncated view>
#   humaneval  eval/humaneval_gen.py --first N (its own diagnostic arm)
#   mc         eval/run_eval.py --benchmarks <first N keys>
# A benchmark with no limit mechanism must REFUSE a --limit run, never run full.
#
# The shim reproduces `python3 eval/gsm8k_shard.py ...` from the repo root, which is the
# shape eval/eval_gsm8k_sharded.sh uses: sys.path[0] = 'eval' (that is what puts
# `score_matrix` on the path for gsm8k_shard.py:45), cwd = the repo root, __file__ = the
# runner, sys.argv[0] = the runner, and NO extra positional argument. Its only effect is
# `eval.gsm8k.LOCAL = <derived view>`. Measured on the pod 2026-10-09: without
# sys.path[0]='eval' the child died at `from score_matrix import classify`.
GSM8K_VIEW_SHIM = (
    "import runpy, sys;"
    "sys.path.insert(0, '.');"
    "sys.path.insert(0, 'eval');"
    "import eval.gsm8k as _g;"
    "_g.LOCAL = sys.argv[1];"
    "sys.argv = ['eval/gsm8k_shard.py'] + sys.argv[2:];"
    "runpy.run_path('eval/gsm8k_shard.py', run_name='__main__')"
)

# Live children and the run's claim, so a signal to the orchestrator kills the children
# it started and releases the claim it holds. A killed orchestrator otherwise leaves an
# orphan holding the card, and no finally-block runs on SIGTERM without this handler.
LIVE = {}
STATE = {"claim": None, "log": None}


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_children(log, grace=15.0):
    """SIGTERM every live child, then SIGKILL what survives, by exact pid."""
    import signal

    pids = list(LIVE)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.time() + grace
    while time.time() < deadline and any(_pid_alive(p) for p in pids):
        time.sleep(0.5)
    for pid in pids:
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    if pids:
        log.line(f"signal cleanup: signalled {len(pids)} child process(es) {pids}")


def _on_signal(signum, _frame):
    """Kill the children and release the claim BEFORE the process dies.

    A signal handler runs in the main thread, so it is the only place that can do both:
    the finally-blocks do not run on SIGTERM without this handler.
    """
    log = STATE.get("log")
    if log is not None:
        log.line(f"signal {signum}: stopping E6, killing children, releasing the claim")
        kill_children(log)
        if STATE.get("claim"):
            release_claim(STATE["claim"], log)
            STATE["claim"] = None
    raise SystemExit(128 + int(signum))


def install_signal_handlers():
    import signal

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _on_signal)


# ---------------------------------------------------------------- small utilities
def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, obj):
    """Atomic: a reader never sees a half-written shard record."""
    path = pathlib.Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def write_jsonl(path, rows):
    path = pathlib.Path(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


class Tee:
    """stdout plus log.txt, one write at a time, callable from any worker thread."""

    def __init__(self, log_path):
        self.lock = threading.Lock()
        self.f = open(log_path, "a", encoding="utf-8", buffering=1)

    def line(self, text):
        with self.lock:
            sys.stdout.write(text.rstrip("\n") + "\n")
            sys.stdout.flush()
            self.f.write(text.rstrip("\n") + "\n")
            self.f.flush()

    def close(self):
        with self.lock:
            self.f.close()


def stage_dir(tag):
    """The contract's fixed product directory for this stage.

    Built here rather than via ladder_common.stage_paths because the E6 tag is an
    explicit CLI argument; main() asserts the two agree for the ckpt-derived tag,
    which is the case every other stage uses.
    """
    d = lc.LADDER / tag / "e6"
    for sub in ("", "shards", "logs", "inputs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return {"dir": d, "items": d / "items.jsonl", "summary": d / "summary.json",
            "log": d / "log.txt", "seal": d / "seal.json", "plan": d / "plan.json",
            "merged": d / "merged.jsonl", "result": d / "result.json",
            "shards": d / "shards", "logs": d / "logs", "inputs": d / "inputs"}


# ---------------------------------------------------------------- input datasets
def read_rows(path, what):
    path = pathlib.Path(path)
    lc.require(path.is_file(),
               f"{what} is missing at {path} -- E6 cannot merge against a denominator "
               f"whose data is absent")
    rows = lc.read_jsonl(path)
    lc.require(bool(rows), f"{what} at {path} is empty")
    return rows


def derive_math500_for_eval(raw_path, out_path):
    """RAW MATH-500 (problem/solution schema) -> the instruction schema math_hard.py reads.

    eval/math_hard.py:117,141,172 indexes r["instruction"], r["answer"] and r["level"];
    the raw gold file data/eval/math_500_en_test.jsonl carries problem/solution instead
    (runs/en_math_adapter_check.py reconciles the two when the derived file is present).
    DERIVATION IS CONTENT-PRESERVING: instruction is the raw problem verbatim, and
    answer / solution / level / unique_id carry through unmodified.
    """
    rows = read_rows(raw_path, "raw MATH-500")
    lc.require(len(rows) == MATH500_DENOM,
               f"raw MATH-500 has {len(rows)} rows, expected {MATH500_DENOM}")
    seen, out = set(), []
    for i, r in enumerate(rows):
        uid = str(r.get("unique_id") or "")
        lc.require(uid and uid not in seen,
                   f"raw MATH-500 row {i} has a missing or duplicate unique_id")
        seen.add(uid)
        text = r.get("problem")
        lc.require(isinstance(text, str) and text.strip(),
                   f"raw MATH-500 {uid}: problem is empty")
        lc.require(str(r.get("answer") or "").strip(), f"raw MATH-500 {uid}: answer is empty")
        lc.require(r.get("level") is not None, f"raw MATH-500 {uid}: level is empty")
        out.append({"unique_id": uid, "instruction": text, "solution": r.get("solution", ""),
                    "answer": r["answer"], "level": r["level"], "subject": r.get("subject")})
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, out)
    return out_path, len(out)


def limited_view(rows, path, source, limit, log):
    """Write the first `limit` rows of a dataset verbatim, then prove the view is a slice.

    The proof is the point: a truncated view that drifted from its source would relabel
    a limited run as a full one. The check re-reads what landed on disk.
    """
    view = list(rows[:limit])
    write_jsonl(path, view)
    back = lc.read_jsonl(path)
    lc.require(back == view, f"limited view {path} did not round-trip")
    lc.require(back == list(rows[:limit]),
               f"limited view {path} is not the first {limit} rows of {source}")
    log.line(f"limit {limit}: derived view {path} ({len(back)} rows) from {source} -- "
             f"the child reads this file, so the bound is real")
    return str(path)


def id_space(name, limit, inputs_dir, log):
    """{ids, inputs, child_data, limit_enforced_by} in FILE ORDER.

    Every shard rule in the reused runners is positional (`rows[shard::shards]`,
    `index % n == i`), so the ID space comes from the dataset in file order and never
    from a runner's output: a missing ID has to be visible as missing and an ID the
    dataset does not have has to be visible as extra. The prompt hash binds each ID to
    the text that produced it, and `text` is kept so the normalizer can check that a
    shard artifact really holds the items its positions claim.

    `child_data` is the file the CHILD must read for --limit to bound the work.
    `limit_enforced_by` names the mechanism; a limit with no mechanism is refused.
    """
    inputs_dir = pathlib.Path(inputs_dir)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    if name == "gsm8k":
        rows = read_rows(GSM8K_DATA, "GSM8K")
        n_all = len(rows)
        lc.require(n_all >= (limit or GSM8K_DENOM),
                   f"GSM8K has {n_all} rows, need {limit or GSM8K_DENOM}")
        child, enforced = None, None
        src = {"path": str(GSM8K_DATA), "sha256": lc.sha256_file(GSM8K_DATA), "n_rows": n_all}
        if limit:
            view = inputs_dir / f"gsm8k_first{limit}.jsonl"
            child = limited_view(rows, view, GSM8K_DATA, limit, log)
            src = {"path": child, "sha256": lc.sha256_file(child), "n_rows": limit,
                   "limited_from": str(GSM8K_DATA), "limited_from_sha256":
                   lc.sha256_file(GSM8K_DATA)}
            enforced = f"derived dataset view (the runner reads it in-process) {child}"
        rows = rows[: limit or GSM8K_DENOM]
        ids = [{"id": f"gsm8k/{i}", "prompt_sha256": sha256_text(r["question"]),
                "text": r["question"]} for i, r in enumerate(rows)]
        return {"ids": ids, "inputs": src, "child_data": child,
                "limit_enforced_by": enforced}
    if name == "math500_en":
        source = MATH500_FOR_EVAL
        if not source.is_file():
            # The repo's audited adapter file wins when it is present (it is what
            # runs/run_v3_official_math500_1008.sh scored); otherwise derive an identical
            # one into this run's inputs/ so the run is self-contained and hashed.
            source, _ = derive_math500_for_eval(MATH500_RAW,
                                                inputs_dir / "math_500_en_for_eval.jsonl")
        rows = read_rows(source, "MATH-500 (instruction schema)")
        n_all = len(rows)
        lc.require(n_all >= (limit or MATH500_DENOM),
                   f"MATH-500 has {n_all} rows, need {limit or MATH500_DENOM}")
        src = {"path": str(source), "sha256": lc.sha256_file(source), "n_rows": n_all}
        child, enforced = None, None
        if limit:
            view = inputs_dir / f"math500_first{limit}.jsonl"
            child = limited_view(rows, view, source, limit, log)
            src = {"path": child, "sha256": lc.sha256_file(child), "n_rows": limit,
                   "limited_from": str(source),
                   "limited_from_sha256": lc.sha256_file(source)}
            enforced = f"eval/math_hard.py --data {child}"
        rows = rows[: limit or MATH500_DENOM]
        ids = []
        for i, r in enumerate(rows):
            text = r.get("instruction")
            lc.require(isinstance(text, str) and text.strip(),
                       f"MATH-500 row {i} has no instruction field -- eval/math_hard.py reads "
                       f"'instruction', and the raw gold file's schema is problem/solution")
            lc.require(str(r.get("answer") or "").strip(), f"MATH-500 row {i} has no answer")
            lc.require(r.get("level") is not None, f"MATH-500 row {i} has no level")
            ids.append({"id": str(r.get("unique_id") or f"math500/{i}"),
                        "prompt_sha256": sha256_text(text), "text": text})
        lc.require(len({r["id"] for r in ids}) == len(ids), "MATH-500 ids are not unique")
        return {"ids": ids, "inputs": src, "child_data": child,
                "limit_enforced_by": enforced}
    if name == "humaneval":
        rows = read_rows(HUMANEVAL_DATA, "HumanEval 164")
        lc.require(len(rows) == HUMANEVAL_DENOM,
                   f"HumanEval has {len(rows)} tasks, expected {HUMANEVAL_DENOM}")
        rows = rows[: limit or HUMANEVAL_DENOM]
        ids = [{"id": str(r["task_id"]), "prompt_sha256": sha256_text(r["prompt"]),
                "text": r["prompt"]} for r in rows]
        lc.require(len({r["id"] for r in ids}) == len(ids),
                   "HumanEval task_ids are not unique")
        return {"ids": ids,
                "inputs": {"path": str(HUMANEVAL_DATA),
                           "sha256": lc.sha256_file(HUMANEVAL_DATA),
                           "n_rows": len(lc.read_jsonl(HUMANEVAL_DATA))},
                "child_data": None,
                "limit_enforced_by": (f"eval/humaneval_gen.py --first {limit}"
                                      if limit else None)}
    if name == "mc":
        keys = MC_DEFAULT[:limit] if limit else list(MC_DEFAULT)
        return {"ids": [{"id": k, "prompt_sha256": None, "text": None} for k in keys],
                "inputs": {"path": None, "sha256": sha256_text("mc:" + ",".join(keys)),
                           "n_rows": len(keys), "synthetic": True},
                "child_data": None,
                "limit_enforced_by": (f"eval/run_eval.py --benchmarks {' '.join(keys)}"
                                      if limit else None)}
    raise lc.Refuse(f"unknown benchmark {name!r}")


# ---------------------------------------------------------------- runner commands
def runner_name(tag):
    return re.sub(r"[^A-Za-z0-9._-]", "_", tag)


def _gsm8k_argv(ctx, i, n):
    """The runner argv from the first flag on -- the launcher shape, unchanged."""
    run = runner_name(ctx["tag"])
    force = ["--force"] if ctx["force"] else []
    argv = ["--ckpt", ctx["ckpt"], "--tokenizer", ctx["tokenizer"],
            "--shards", str(n), "--shard", str(i), "--run", run, *force]
    if ctx["batch"]:
        argv += ["--batch", str(ctx["batch"])]
    return argv


def commands(name, ctx, i, n):
    """The exact argv for (benchmark, shard). cli_contract_selftest() checks every flag
    against the runner's own source, and a --limit run must bound the child's work:
    the plan names the mechanism, and commands() applies it here."""
    ckpt, tok = ctx["ckpt"], ctx["tokenizer"]
    run = runner_name(ctx["tag"])
    force = ["--force"] if ctx["force"] else []
    if name == "gsm8k":
        argv = _gsm8k_argv(ctx, i, n)
        view = ctx["child_data"].get("gsm8k")
        if view:
            # Identical argv from '--ckpt' on; the shim only redirects the dataset view.
            return [sys.executable, "-c", GSM8K_VIEW_SHIM, view, *argv]
        return [sys.executable, "eval/gsm8k_shard.py", *argv]
    if name == "math500_en":
        data = ctx["child_data"].get("math500_en") or ctx["inputs"]["math500_en"]["path"]
        cmd = [sys.executable, "eval/math_hard.py", "--ckpt", ckpt, "--data", data,
               "--shards", str(n), "--shard", str(i), "--run", run, *force]
        return cmd + (["--batch", str(ctx["batch"])] if ctx["batch"] else [])
    if name == "humaneval":
        cmd = [sys.executable, "eval/humaneval_gen.py", "--ckpt", ckpt,
               "--shard_i", str(i), "--shard_n", str(n), "--run", run, *force]
        if ctx["limit"]:
            cmd += ["--first", str(ctx["limit"])]
        return cmd + (["--chatml"] if ctx["protocol"] == "chatml" else ["--rstrip_nl"])
    if name == "mc":
        return [sys.executable, "eval/run_eval.py", "--ckpt", ckpt, "--tokenizer", tok,
                "--benchmarks", *ctx["mc_keys"]]
    raise lc.Refuse(f"unknown benchmark {name!r}")


def cmd_equivalent(name, ctx, i, n, gpu):
    """The plain command this child emulates, recorded beside the shim argv.

    An auditor reading `shards/<i>.json` must be able to see that a limited gsm8k shard
    is the launcher shape plus one dataset-view redirect, not a different invocation.
    """
    if name != "gsm8k" or not ctx["child_data"].get("gsm8k"):
        return None
    return (f"CUDA_VISIBLE_DEVICES={gpu} python3 eval/gsm8k_shard.py "
            f"{' '.join(_gsm8k_argv(ctx, i, n))}"
            f"   # same argv; only eval.gsm8k.LOCAL = {ctx['child_data']['gsm8k']}")


def preds_path(name, ctx, i, n):
    """Where the runner writes, from ITS naming rule -- including the order
    eval_artifacts.versioned_path applies --run: AFTER the shard index."""
    base = pathlib.Path(ctx["ckpt"]).name
    run = runner_name(ctx["tag"])
    if name == "gsm8k":
        stem = f"preds_{base}" + (f".{i}" if n > 1 else "")
        return ROOT / "data" / "eval" / f"{stem}.{run}.jsonl"
    if name == "math500_en":
        stem = f"hard_{base}" + (f".{i}" if n > 1 else "")
        return ROOT / "data" / "eval" / f"{stem}.{run}.jsonl"
    if name == "humaneval":
        arm = ".chatml" if ctx["protocol"] == "chatml" else ".rstripnl"
        stem = f"preds_humaneval_{base}{arm}" + (f".shard{i}of{n}" if n > 1 else "")
        return ROOT / "data" / "eval" / f"{stem}.{run}.jsonl"
    if name == "mc":
        return None  # stdout only; the shard log IS the artifact
    raise lc.Refuse(f"unknown benchmark {name!r}")


def contract_denominator(name):
    return {"gsm8k": GSM8K_DENOM, "math500_en": MATH500_DENOM,
            "humaneval": HUMANEVAL_DENOM, "mc": len(MC_DEFAULT)}[name]


# ---------------------------------------------------------------- plan
def build_plan(args, paths, log):
    """Read every input once; write plan.json + inputs/ids_<bench>.jsonl.

    The plan is the producer's fingerprint: the merge compares against it instead of
    re-deriving denominators, so a dataset swapped between launch and merge is a
    failure rather than a silently different question set.
    """
    lc.require(args.shards >= 1, f"--shards must be >= 1, got {args.shards}")
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    lc.require(bool(gpus), "--gpus is empty: a worker must be able to name its card")
    lc.require(args.shards <= len(gpus),
               f"--shards {args.shards} with {len(gpus)} card(s) {gpus}: this module runs one "
               f"worker per shard and one worker per card, so a short card list would put two "
               f"processes on one card")
    lc.require(args.protocol in ("base", "chatml"),
               f"--protocol must be base or chatml, got {args.protocol!r}")
    names = [b.strip() for b in args.only.split(",")] if args.only else list(BENCH_ORDER)
    for b in names:
        lc.require(b in BENCH_ORDER, f"--only names unknown benchmark {b!r}")

    ctx = {"tag": args.tag, "ckpt": str(args.ckpt), "tokenizer": args.tokenizer,
           "protocol": args.protocol, "force": args.force, "batch": args.batch,
           "limit": args.limit, "n_shards": args.shards, "gpus": gpus,
           "benches_requested": names,
           "diagnostic": bool(args.limit) or len(names) != len(BENCH_ORDER),
           "paths": paths, "claim_name": None}
    ctx["ckpt_sha256"] = lc.sha256_file(args.ckpt) if pathlib.Path(args.ckpt).is_file() else None
    ctx["ids"], ctx["inputs"], ctx["ids_sha256"], ctx["prompt_text"] = {}, {}, {}, {}
    ctx["child_data"], ctx["limit_enforced_by"] = {}, {}
    for name in names:
        space = id_space(name, args.limit, paths["inputs"], log)
        ids = space["ids"]
        ctx["ids"][name] = ids
        ctx["inputs"][name] = space["inputs"]
        ctx["child_data"][name] = space["child_data"]
        ctx["limit_enforced_by"][name] = space["limit_enforced_by"]
        ctx["prompt_text"][name] = [r["text"] for r in ids]
        # A limit with no mechanism would silently run the full set (measured 2026-10-09:
        # a --limit 20 smoke scored 192/1319 before it was killed). Refuse instead.
        lc.require(not (args.limit and not space["limit_enforced_by"]),
                   f"--limit {args.limit} for {name}: this benchmark has no way to bound the "
                   f"child's work, so the limit would be nominal and the child would run the "
                   f"full set")
        idp = paths["inputs"] / f"ids_{name}.jsonl"
        write_jsonl(idp, [{"id": r["id"], "prompt_sha256": r["prompt_sha256"]} for r in ids])
        ctx["ids_sha256"][name] = lc.sha256_file(idp)
        log.line(f"input {name}: {len(ids)} ids, denominator {len(ids)}, "
                 f"contract {contract_denominator(name)}, "
                 f"limit_enforced_by={space['limit_enforced_by'] or 'none (full set)'}")
    ctx["mc_keys"] = [r["id"] for r in ctx["ids"].get("mc", [])] or list(MC_DEFAULT)

    plan = {
        "stage": "e6", "tag": ctx["tag"], "ckpt": ctx["ckpt"],
        "ckpt_sha256": ctx["ckpt_sha256"], "protocol": ctx["protocol"],
        "diagnostic": ctx["diagnostic"], "limit": args.limit,
        "n_items": {name: len(ctx["ids"][name]) for name in names},
        "n_shards": args.shards, "gpus": gpus, "benches_requested": names,
        "benches": {name: {"n_ids": len(ctx["ids"][name]),
                           "denominator": len(ctx["ids"][name]),
                           "contract_denominator": contract_denominator(name),
                           "gated": name != "mc",
                           "limit": args.limit,
                           "limit_enforced_by": ctx["limit_enforced_by"][name],
                           "exclude": HE_EXCLUDE if name == "humaneval" else [],
                           "clean_denominator": HUMANEVAL_CLEAN if name == "humaneval" else None,
                           "ids_sha256": ctx["ids_sha256"][name],
                           "inputs": ctx["inputs"][name]}
                    for name in names},
        "mc_keys": ctx["mc_keys"],
        "tokenizer": {"path": ctx["tokenizer"],
                      "sha256": (lc.sha256_file(ctx["tokenizer"])
                                 if pathlib.Path(ctx["tokenizer"]).is_file() else None)},
        "worker_cards": {str(i): gpus[i] for i in range(args.shards)},
        "commands": {name: {str(i): " ".join(commands(name, ctx, i, args.shards))
                            for i in range(args.shards)} for name in names},
        "created_utc": utc(), "pid": os.getpid(),
    }
    write_json(paths["plan"], plan)
    return ctx, plan


# ---------------------------------------------------------------- launch
def prereq_verdicts(tag, log):
    """Section 7: E6 does not start before the earlier stages pass on this checkpoint.

    E2 is SFT-only (handover doc section 5, E2 row), so its absence is not a gap for a
    base checkpoint; when it exists it must pass like the rest.
    """
    out = {}
    for stage in ("e0", "e1", "e2", "e3", "e4", "e5"):
        p = lc.LADDER / tag / stage / "summary.json"
        if not p.is_file():
            out[stage] = {"present": False, "ok": stage == "e2",
                          "why": "absent" + (" (SFT-only stage)" if stage == "e2" else "")}
            continue
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            out[stage] = {"present": True, "ok": False, "why": f"unreadable: {exc}"}
            continue
        gates = s.get("gates") or {}
        ok = (s.get("exit_code") == 0) and bool(gates) and all(bool(v) for v in gates.values())
        out[stage] = {"present": True, "ok": ok,
                      "why": f"exit_code={s.get('exit_code')} gates={gates}"}
    missing = [k for k, v in out.items() if not v["ok"]]
    log.line("prereq: " + ", ".join(f"{k}={'pass' if v['ok'] else 'FAIL'}"
                                    for k, v in out.items()))
    return out, missing


def acquire_claim(ctx, log):
    """One claim for the whole run, held on the orchestrator's pid.

    Per-child claims would leave the card unclaimed between a worker's benchmarks; the
    card is busy for the whole run, so the claim must be too. Children get
    AUPAI_CLAIMED_BY=<name>, and scripts/loader.py verifies that name against
    runs/claims/ (live, covers the child's CUDA_VISIBLE_DEVICES, pid is an ancestor) --
    a set-but-unverified name is refused there, not trusted.
    """
    name = f"e6_{runner_name(ctx['tag'])}"
    cmd = [sys.executable, str(CLAIM_TOOL), "acquire", "--name", name,
           "--cards", ",".join(ctx["gpus"]),
           "--note", f"E6 full benchmarks on {ctx['ckpt']} (tag {ctx['tag']})"]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    msg = ((r.stdout or "") + (r.stderr or "")).strip()
    log.line(f"claim {name} on cards {','.join(ctx['gpus'])}: rc={r.returncode} {msg}")
    return (r.returncode == 0), name, msg


def release_claim(name, log):
    r = subprocess.run([sys.executable, str(CLAIM_TOOL), "release", "--name", name],
                       cwd=str(ROOT), capture_output=True, text=True)
    log.line(f"claim release {name}: rc={r.returncode} "
             f"{((r.stdout or '') + (r.stderr or '')).strip()}")


def normalize(name, ctx, i, n, rows_raw, meta):
    """Runner output -> the ladder's per-item row, ID-assigned and content-checked.

    The ID is assigned from the DATASET's position (the rule every shard selector uses:
    index % n == i) and the runner's own question text must agree with it. Positional
    assignment without the content check would relabel a shifted or reordered artifact
    as a correct one.
    """
    expected = ctx["ids"][name][i::n]
    out = []
    if name == "mc":
        want = {r["id"] for r in expected}
        for line in rows_raw:
            m = re.match(r"\s*(.+?):\s+(?:([\d.]+)%|SKIPPED\s*(.*))$", line.rstrip("\n"))
            if not m:
                continue
            display, acc, skipped = m.group(1).strip(), m.group(2), m.group(3)
            key = next((k for k, d in MC_DISPLAY.items() if d == display), None)
            if key is None or key not in want or any(r["id"] == key for r in out):
                continue
            out.append({"id": key, "passed": None,
                        "status": "reported" if acc else "skipped",
                        "accuracy": float(acc) if acc else None,
                        "detail": (skipped or "").strip() or None,
                        "prompt": None, "prompt_sha256": None, "response_raw": None,
                        "parsed": float(acc) if acc else None, "output": None,
                        "stop_reason": None, "n_tokens": None, "tokens": None,
                        "fail_class": None, "shard_i": i})
        missing = sorted(want - {r["id"] for r in out})
        if missing:
            meta["normalize_error"] = f"run_eval.py stdout did not report {missing}"
        meta["accuracies"] = {r["id"]: r["accuracy"] for r in out if r["status"] == "reported"}
        meta["skipped"] = sorted(r["id"] for r in out if r["status"] == "skipped")
        meta["ids"] = [r["id"] for r in out]
        meta["n_ids_expected"] = len(expected)
        return out
    if len(rows_raw) != len(expected):
        meta["normalize_error"] = (f"the runner wrote {len(rows_raw)} rows, the shard's ID "
                                   f"slice holds {len(expected)}")
        return []
    texts = ctx["prompt_text"][name]
    for j, (row, exp) in enumerate(zip(rows_raw, expected)):
        if name == "gsm8k":
            got_text, out_text, ok = row.get("question"), row.get("gen"), bool(row.get("ok"))
            fail = None if ok else ("no_number" if row.get("pred") is None else "wrong_answer")
            stop, parsed, raw = None, row.get("pred"), row.get("gen")
        elif name == "math500_en":
            got_text, out_text, ok = row.get("q"), row.get("gen"), bool(row.get("ok"))
            fail = None if ok else "wrong_answer"
            # eval/math_hard.py writes no parsed field; the scorer recomputes it from `gen`
            # inside the runner, so an item-level parse is not recoverable from the artifact.
            stop, parsed, raw = None, None, row.get("gen")
        else:  # humaneval
            got_text, out_text, ok = None, row.get("gen"), bool(row.get("ok"))
            fail = None if ok else row.get("empty_kind")
            stop = row.get("stop_reason")
            parsed, raw = row.get("gen"), row.get("raw", row.get("gen"))
        if got_text is not None and got_text != texts[i + j * n]:
            meta["normalize_error"] = (f"row {j} of shard {i} carries text that is not item "
                                       f"{i + j * n}: the artifact and the ID space disagree")
            return []
        out.append({"id": exp["id"], "passed": ok,
                    "prompt": texts[i + j * n], "prompt_sha256": exp["prompt_sha256"],
                    "response_raw": raw, "parsed": parsed, "output": out_text,
                    "stop_reason": stop, "n_tokens": None, "tokens": None,
                    "fail_class": fail, "shard_i": i})
    if name == "humaneval":
        got_ids = [r["id"] for r in out]
        want_ids = [r["id"] for r in expected]
        if set(got_ids) != set(want_ids) or len(got_ids) != len(want_ids):
            meta["normalize_error"] = (
                "humaneval shard task_ids are not the shard's slice: missing "
                f"{sorted(set(want_ids) - set(got_ids))[:5]}, extra "
                f"{sorted(set(got_ids) - set(want_ids))[:5]}")
            return []
    meta["ids"] = [r["id"] for r in out]
    meta["n_ids_expected"] = len(expected)
    return out


def run_worker(ctx, i, paths, log, errors):
    """One card, one process at a time: this worker's shards of every requested bench.

    `shards/<i>.json` is the auditor's record: gpu, pid, claim name, start command,
    input hash and shard_i/shard_n at the top level AND per benchmark, plus the ID list,
    the item JSONL path, the raw runner artifact and its hash, the exit code and the
    start/finish/heartbeat times (section 7 of the handover doc).
    """
    gpu = ctx["gpus"][i]
    rec = {"shard_i": i, "shard_n": ctx["n_shards"], "gpu": gpu,
           "claim": ctx.get("claim_name"), "claim_cards": [gpu],
           "cmd": None, "pid": None, "input_sha256": {}, "per_bench": {},
           "started_utc": utc(), "heartbeat_utc": utc(), "finished_utc": None,
           "exit_code": None, "status": "running"}
    rec_path = paths["shards"] / f"{i}.json"
    write_json(rec_path, rec)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("HC_IMPL", "torch")
    env.setdefault("NORM_IMPL", "torch")
    if ctx.get("claim_name"):
        env["AUPAI_CLAIMED_BY"] = ctx["claim_name"]
    rcs = []
    for name in ctx["benches_requested"]:
        if name == "mc" and i != 0:
            continue  # one MC process; its sets are not shardable by item
        n = 1 if name == "mc" else ctx["n_shards"]
        shard = 0 if name == "mc" else i
        cmd = commands(name, ctx, shard, n)
        meta = {"bench": name, "shard_i": shard, "shard_n": n, "gpu": gpu,
                "claim": ctx.get("claim_name"), "cmd": " ".join(cmd),
                "cmd_equivalent": cmd_equivalent(name, ctx, shard, n, gpu),
                "log": str(paths["logs"] / f"shard_{i}.{name}.log"),
                "started_utc": utc(), "finished_utc": None, "pid": None,
                "input_sha256": ctx["inputs"][name]["sha256"],
                "input_path": ctx["inputs"][name]["path"],
                "limit_enforced_by": ctx["limit_enforced_by"][name],
                "raw_preds": None, "raw_preds_sha256": None, "n_raw_rows": None,
                "jsonl": None, "ids": [], "n_ids_expected": len(ctx["ids"][name][shard::n]),
                "exit_code": None, "normalize_error": None, "elapsed_s": None}
        log.line(f"[w{i} gpu{gpu}] {name} shard {shard}/{n}: {' '.join(cmd)}")
        t0 = time.time()
        with open(meta["log"], "wb") as fh:
            proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=fh,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            meta["pid"] = proc.pid
            rec["pid"] = proc.pid
            rec["cmd"] = meta["cmd"]
            rec["per_bench"][name] = meta
            rec["input_sha256"][name] = meta["input_sha256"]
            rec["heartbeat_utc"] = utc()
            write_json(rec_path, rec)
            LIVE[proc.pid] = name
            try:
                rc = proc.wait()
            finally:
                LIVE.pop(proc.pid, None)
        meta["exit_code"] = rc
        meta["finished_utc"] = utc()
        meta["elapsed_s"] = round(time.time() - t0, 1)
        rcs.append(rc)
        log.line(f"[w{i} gpu{gpu}] {name} shard {shard}/{n}: exit {rc} in "
                 f"{meta['elapsed_s']}s; log {meta['log']}")
        raw_path = preds_path(name, ctx, shard, n)
        rows_norm = []
        if name == "mc":
            lp = pathlib.Path(meta["log"])
            rows_raw = lp.read_text(encoding="utf-8", errors="replace").splitlines() \
                if lp.is_file() else []
            rows_norm = normalize(name, ctx, shard, n, rows_raw, meta)
        elif raw_path is not None and pathlib.Path(raw_path).is_file():
            meta["raw_preds"] = str(raw_path)
            meta["raw_preds_sha256"] = lc.sha256_file(raw_path)
            rows_raw = lc.read_jsonl(raw_path)
            meta["n_raw_rows"] = len(rows_raw)
            n_expect = len(ctx["ids"][name][shard::n])
            if len(rows_raw) != n_expect:
                # The 2026-10-09 smoke scored 192/1319 under a nominal --limit 20; a
                # count that disagrees with the declared slice is a failure, not a full run.
                log.line(f"[w{i} gpu{gpu}] {name} shard {shard}/{n}: runner wrote "
                         f"{len(rows_raw)} rows, the declared slice holds {n_expect}")
            rows_norm = normalize(name, ctx, shard, n, rows_raw, meta)
        else:
            meta["normalize_error"] = f"the runner wrote no artifact at {raw_path}"
        if rows_norm:
            jp = paths["shards"] / f"{i}.{name}.jsonl"
            for r in rows_norm:
                r["bench"] = name
            write_jsonl(jp, rows_norm)
            meta["jsonl"] = str(jp)
            meta["n_rows"] = len(rows_norm)
        rec["heartbeat_utc"] = utc()
        write_json(rec_path, rec)
    rec["exit_code"] = 0 if all(rc == 0 for rc in rcs) else next(rc for rc in rcs if rc != 0)
    rec["status"] = "done" if rec["exit_code"] == 0 else "failed"
    rec["finished_utc"] = utc()
    write_json(rec_path, rec)
    if rec["exit_code"] != 0:
        errors.append(f"shard {i} (gpu {gpu}) exit_code={rec['exit_code']}")


def launch(ctx, paths, log, errors):
    threads = [threading.Thread(target=run_worker, args=(ctx, i, paths, log, errors),
                                name=f"e6-shard-{i}") for i in range(ctx["n_shards"])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# ---------------------------------------------------------------- merge judge
def load_plan(paths):
    lc.require(paths["plan"].is_file(),
               f"no plan.json at {paths['plan']}: the merge has no declared denominators or "
               f"input hashes. A merge without its producer's fingerprint cannot tell a "
               f"changed dataset from a correct one")
    return json.loads(paths["plan"].read_text(encoding="utf-8"))


def spec_from_plan(plan, paths):
    """(spec, plan_failures): what the merge judges, re-derived from the plan's artifacts."""
    spec = {"tag": plan["tag"], "ckpt": plan["ckpt"], "n_shards": plan["n_shards"],
            "protocol": plan["protocol"], "diagnostic": plan.get("diagnostic", False),
            "limit": plan.get("limit"), "benches_requested": list(plan["benches_requested"]),
            "n_items": dict(plan.get("n_items") or {}),
            "tokenizer_path": (plan.get("tokenizer") or {}).get("path") or str(TOKENIZER),
            "benches": {}, "ids": {}, "inputs": {}}
    failures = []
    for name, b in plan["benches"].items():
        spec["benches"][name] = dict(b)
        spec["inputs"][name] = dict(b["inputs"])
        idp = paths["inputs"] / f"ids_{name}.jsonl"
        if not idp.is_file():
            failures.append(f"{name}: declared ID space {idp} is missing")
            spec["ids"][name] = []
            continue
        got = lc.sha256_file(idp)
        if got != b["ids_sha256"]:
            failures.append(f"{name}: ID space {idp} hashes {got}, the plan declares "
                            f"{b['ids_sha256']} -- the ID set changed after the plan was written")
        spec["ids"][name] = [r["id"] for r in lc.read_jsonl(idp)]
    return spec, failures


def _bench_shard_count(name, recs):
    """The shard count a BENCHMARK ran with, not the worker count.

    MC is one process (`shard_n` 1) on worker 0 while the other benchmarks use every
    worker; reading the worker count for MC would expect 7/8 of its ID set to be
    missing and would report a correct run as a coverage failure.
    """
    for rec in recs:
        meta = (rec or {}).get("per_bench", {}).get(name)
        if meta and meta.get("shard_n"):
            return int(meta["shard_n"])
    return len(recs)


def judge_bench(name, bspec, ids, recs, rows_by_shard, inputs, log):
    """The merge decision for one benchmark over constructed or real shard collections.

    Pure (no file reads), so merge_selftest()'s worlds exercise exactly the code the
    real merge runs. `merge_ok` is the AND of coverage, ID set, row count, denominator
    and seal; `gate_failed` names which one failed.
    """
    nb = _bench_shard_count(name, recs)
    v = {"bench": name, "denominator": bspec["denominator"],
         "contract_denominator": bspec.get("contract_denominator"),
         "gated": bool(bspec.get("gated", True)),
         "shards_expected": nb, "shards_ok": 0, "shards_missing": [], "shards_bad": [],
         "rows": 0, "rows_ok": False, "ids_ok": False, "coverage": None,
         "denominator_ok": False, "denominator_reasons": [],
         "seal_ok": False, "seal_failures": [],
         "passed": None, "rate": None, "clean_pass": None, "clean_denominator": None,
         "clean_rate": None, "excluded": list(bspec.get("exclude") or []),
         "accuracies": {}, "mc_skipped": [], "gate_failed": [], "merge_ok": False}

    expected_slices = [list(ids[i::nb]) for i in range(nb)]
    merged = []
    for i in range(nb):
        rec = recs[i] if i < len(recs) else None
        if rec is None:
            v["shards_missing"].append(i)
            continue
        meta = (rec.get("per_bench") or {}).get(name)
        if meta is None:
            v["shards_missing"].append(i)
            continue
        bad = []
        if meta.get("exit_code") != 0:
            bad.append(f"exit_code={meta.get('exit_code')}")
        want = expected_slices[i]
        rows = rows_by_shard.get(i)
        if meta.get("n_ids_expected") != len(want):
            bad.append(f"declares {meta.get('n_ids_expected')} ids, the shard slice has "
                       f"{len(want)}")
        if rows is None:
            bad.append("no normalized artifact")
        elif len(rows) != len(want):
            bad.append(f"{len(rows)} rows vs {len(want)} in the shard's ID slice")
        elif sorted(str(r.get("id")) for r in rows) != sorted(want):
            got = {str(r.get("id")) for r in rows}
            bad.append(f"ID set differs: missing {sorted(set(want) - got)[:5]}, "
                       f"extra {sorted(got - set(want))[:5]}")
        # seal: section 7's per-shard record, field by field
        for field in ("gpu", "claim", "cmd", "log", "input_sha256", "started_utc",
                      "finished_utc", "pid", "ids"):
            if meta.get(field) in (None, "", []):
                v["seal_failures"].append(f"shard {i}: {field} is empty")
        if meta.get("input_sha256") != inputs.get("sha256"):
            v["seal_failures"].append(f"shard {i}: input_sha256 {meta.get('input_sha256')} "
                                      f"!= plan {inputs.get('sha256')}")
        if meta.get("raw_preds") is None and meta.get("jsonl") is None:
            v["seal_failures"].append(f"shard {i}: no artifact path recorded")
        if meta.get("normalize_error"):
            bad.append(meta["normalize_error"])
        (v["shards_bad"].append({"i": i, "why": bad}) if bad else None)
        if not bad:
            v["shards_ok"] += 1
        if rows:
            merged.extend(rows)

    v["rows"] = len(merged)
    # 2. exact ID set over the merged rows (ladder_common.verify_coverage)
    ok_ids, detail = lc.verify_coverage(merged, ids, label=f"e6:{name}")
    v["ids_ok"], v["coverage"] = ok_ids, detail
    # 3. row count against the declared denominator
    v["rows_ok"] = v["rows"] == bspec["denominator"]
    if not v["rows_ok"]:
        v["denominator_reasons"].append(f"{v['rows']} merged rows != declared denominator "
                                        f"{bspec['denominator']}")
    # 4. denominator identity, including HumanEval's clean split
    if name == "humaneval":
        exc = list(bspec.get("exclude") or [])
        if len(exc) != len(HE_EXCLUDE) or len(set(exc)) != len(HE_EXCLUDE):
            v["denominator_reasons"].append(f"exclusion list has {len(exc)} entries, the "
                                            f"contract pins {len(HE_EXCLUDE)}")
        not_in = sorted(set(exc) - set(ids))
        if not_in:
            v["denominator_reasons"].append(f"excluded ids absent from the ID space: {not_in}")
        v["clean_denominator"] = len(ids) - len(set(exc) & set(ids))
        if v["clean_denominator"] != HUMANEVAL_CLEAN:
            v["denominator_reasons"].append(f"clean denominator {v['clean_denominator']} "
                                            f"!= {HUMANEVAL_CLEAN}")
    if not v["rows"]:
        v["denominator_reasons"].append("no rows to score")
    v["denominator_ok"] = not v["denominator_reasons"]
    v["seal_ok"] = not v["seal_failures"]

    if merged:
        if name == "humaneval":
            exc = set(bspec.get("exclude") or [])
            sel = [r for r in merged if r["id"] not in exc]
            v["clean_pass"] = sum(1 for r in sel if r.get("passed"))
            v["clean_rate"] = (round(v["clean_pass"] / v["clean_denominator"], 6)
                               if v["clean_denominator"] else None)
        if name == "mc":
            v["accuracies"] = {r["id"]: r.get("accuracy") for r in merged
                               if r.get("status") == "reported"}
            v["mc_skipped"] = sorted(r["id"] for r in merged if r.get("status") == "skipped")
        else:
            v["passed"] = sum(1 for r in merged if r.get("passed"))
            v["rate"] = (round(v["passed"] / bspec["denominator"], 6)
                         if bspec["denominator"] else None)

    for key, gate in (("shards", not v["shards_missing"] and not v["shards_bad"]),
                      ("ids", v["ids_ok"]), ("rows", v["rows_ok"]),
                      ("denominator", v["denominator_ok"]), ("seal", v["seal_ok"])):
        if not gate:
            v["gate_failed"].append(key)
    v["merge_ok"] = not v["gate_failed"]
    log.line(f"merge {name}: shards {v['shards_ok']}/{nb} rows {v['rows']}/"
             f"{bspec['denominator']} ids_ok={v['ids_ok']} seal_ok={v['seal_ok']} "
             f"gate_failed={v['gate_failed'] or 'none'}")
    return v


def merge_run(paths, spec, log):
    """Read one E6 run's artifacts and judge every requested benchmark.

    Returns (verdicts, merged_rows, gates, result). Writes nothing: the caller decides
    what a failed merge may leave behind (the ladder writes a failed gate's result, it
    just never writes a total).
    """
    recs = []
    for i in range(spec["n_shards"]):
        p = paths["shards"] / f"{i}.json"
        recs.append(json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None)
    verdicts, merged = {}, []
    for name in spec["benches_requested"]:
        bspec = spec["benches"][name]
        rows_by_shard = {}
        for i in range(spec["n_shards"]):
            p = paths["shards"] / f"{i}.{name}.jsonl"
            rows_by_shard[i] = lc.read_jsonl(p) if p.is_file() else None
        v = judge_bench(name, bspec, spec["ids"][name], recs, rows_by_shard,
                        spec["inputs"][name], log)
        verdicts[name] = v
        for i in range(spec["n_shards"]):
            merged.extend(rows_by_shard.get(i) or [])
    requested = list(spec["benches_requested"])
    unrequested = [b for b in BENCH_ORDER if b not in requested]
    gates = {
        "shards": all(not v["shards_missing"] and not v["shards_bad"]
                      for v in verdicts.values()),
        "ids": all(v["ids_ok"] for v in verdicts.values()),
        "rows": all(v["rows_ok"] for v in verdicts.values()),
        "denominator": all(v["denominator_ok"] for v in verdicts.values()),
        "seal": all(v["seal_ok"] for v in verdicts.values()),
        "contract_denominator": all(v["denominator"] == contract_denominator(k)
                                    for k, v in verdicts.items()) and not spec["diagnostic"],
        "contract_complete": not unrequested and not spec["diagnostic"],
    }
    gates["merge"] = all(verdicts[k]["merge_ok"] for k in requested)
    gates["e6"] = bool(gates["merge"] and gates["contract_denominator"]
                       and gates["contract_complete"])
    result = {
        "tag": spec["tag"], "ckpt": spec["ckpt"], "protocol": spec["protocol"],
        "diagnostic": spec["diagnostic"], "n_shards": spec["n_shards"],
        "benches_requested": requested, "benches_not_requested": unrequested,
        "gates": gates,
        "benches": {k: {kk: v[kk] for kk in
                        ("denominator", "contract_denominator", "rows", "rows_ok",
                         "passed", "rate", "ids_ok", "denominator_ok", "seal_ok",
                         "clean_pass", "clean_denominator", "clean_rate", "excluded",
                         "accuracies", "mc_skipped", "gate_failed", "merge_ok",
                         "shards_ok", "shards_expected", "shards_missing", "shards_bad",
                         "seal_failures", "coverage")}
                    for k, v in verdicts.items()},
        "seal_ok": gates["seal"],
        "totals": None, "no_total_reason": None,
    }
    if gates["e6"]:
        result["totals"] = {
            "gsm8k": {"denominator": verdicts["gsm8k"]["denominator"],
                      "passed": verdicts["gsm8k"]["passed"],
                      "rate": verdicts["gsm8k"]["rate"]},
            "math500_en": {"denominator": verdicts["math500_en"]["denominator"],
                           "passed": verdicts["math500_en"]["passed"],
                           "rate": verdicts["math500_en"]["rate"]},
            "humaneval_all": {"denominator": verdicts["humaneval"]["denominator"],
                              "passed": verdicts["humaneval"]["passed"],
                              "rate": verdicts["humaneval"]["rate"],
                              "excluded": verdicts["humaneval"]["excluded"]},
            "humaneval_clean": {"denominator": verdicts["humaneval"]["clean_denominator"],
                                "passed": verdicts["humaneval"]["clean_pass"],
                                "rate": verdicts["humaneval"]["clean_rate"],
                                "excluded": verdicts["humaneval"]["excluded"]},
            "mc": {"retained_only": True, "reported": verdicts["mc"]["accuracies"],
                   "skipped": verdicts["mc"]["mc_skipped"]},
        }
    else:
        failed = "; ".join(f"{k}:{v['gate_failed']}" for k, v in verdicts.items()
                           if v["gate_failed"]) or "none"
        result["no_total_reason"] = (
            "E6 writes no total unless shard coverage, exact ID set, row count, denominator "
            f"and seal all hold: {failed}"
            + (f"; not requested: {unrequested}" if unrequested else "")
            + ("; diagnostic subset run" if spec["diagnostic"] else ""))
    return verdicts, merged, gates, result


def stage_seal_ok(paths):
    p = paths["seal"]
    if not p.is_file():
        return False, [f"seal.json missing at {p}"]
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, [f"seal.json unreadable: {exc}"]
    bad = [f for f in ("tag", "ckpt", "protocol", "n_shards", "cmd", "pid", "started_utc",
                       "inputs", "shards") if s.get(f) in (None, "", [], {})]
    return (not bad), [f"seal.json field {b} is empty" for b in bad]


# ---------------------------------------------------------------- CLI contract self-check
def cli_contract_selftest(log):
    """Every flag this module passes must exist in the runner's own source text.

    Cheap and import-free. A renamed flag in eval/humaneval_gen.py would otherwise
    surface as an eight-card launch failing after the checkpoint load.
    """
    want = {
        "eval/gsm8k_shard.py": ["--ckpt", "--tokenizer", "--shards", "--shard", "--batch",
                                "--run", "--force"],
        "eval/math_hard.py": ["--ckpt", "--data", "--shards", "--shard", "--batch",
                              "--run", "--force"],
        "eval/humaneval_gen.py": ["--ckpt", "--shard_i", "--shard_n", "--run", "--force",
                                  "--rstrip_nl", "--chatml"],
        "eval/run_eval.py": ["--ckpt", "--tokenizer", "--benchmarks"],
    }
    failures, checked = [], {}
    for rel, flags in want.items():
        path = ROOT / rel
        if not path.is_file():
            failures.append(f"{rel} is absent: E6 cannot launch a runner that is not here")
            continue
        src = path.read_text(encoding="utf-8")
        missing = [f for f in flags if f not in src]
        checked[rel] = {"flags": flags, "missing": missing}
        if missing:
            failures.append(f"{rel} no longer accepts {missing}")
    # artifact-name rules this module reproduces must still be in the runners
    name_rules = {
        "eval/gsm8k_shard.py": ["preds_{os.path.basename(args.ckpt)}"],
        "eval/math_hard.py": ["hard_{os.path.basename(a.ckpt)}"],
        "eval/humaneval_gen.py": ["preds_humaneval_", "shard_label(args.shard_i, args.shard_n)"],
    }
    for rel, needles in name_rules.items():
        path = ROOT / rel
        if not path.is_file():
            continue
        src = path.read_text(encoding="utf-8")
        missing = [n for n in needles if n not in src]
        checked.setdefault(rel, {})["artifact_name_missing"] = missing
        if missing:
            failures.append(f"{rel} no longer builds its artifact name the way E6 reads it: "
                            f"{missing}")
    src = (ROOT / "eval/run_eval.py").read_text(encoding="utf-8") \
        if (ROOT / "eval/run_eval.py").is_file() else ""
    pairs = dict(re.findall(r'"([a-z0-9-]+)":\s*\("([^"]+)"', src))
    mism = {k: (v, pairs.get(k)) for k, v in MC_DISPLAY.items() if pairs.get(k) != v}
    checked["eval/run_eval.py displays"] = {"parsed": len(pairs), "mismatch": mism}
    if mism:
        failures.append(f"MC display-name map disagrees with eval/run_eval.py: {mism}")
    if not src or "ALL_BENCHMARKS" not in src:
        failures.append("eval/run_eval.py no longer exposes ALL_BENCHMARKS")
    for f in failures:
        log.line(f"FAIL cli-contract: {f}")
    return (not failures), {"checked": checked, "failures": failures}


# ---------------------------------------------------------------- selftest worlds
def synthetic_ids(name):
    return {"gsm8k": [f"gsm8k/{i}" for i in range(GSM8K_DENOM)],
            "math500_en": [f"math500/{i}" for i in range(MATH500_DENOM)],
            "humaneval": [f"HumanEval/{i}" for i in range(HUMANEVAL_DENOM)],
            "mc": list(MC_DEFAULT)}[name]


def build_synthetic_spec(n_shards=4, denom_override=None, exclude=None, diagnostic=False):
    ids = {b: synthetic_ids(b) for b in BENCH_ORDER}
    spec = {"tag": "synthetic", "ckpt": "synthetic.pt", "n_shards": n_shards,
            "protocol": "base", "diagnostic": diagnostic, "limit": None,
            "benches_requested": list(BENCH_ORDER), "benches": {}, "ids": ids, "inputs": {}}
    for b in BENCH_ORDER:
        denom = (denom_override or {}).get(b, len(ids[b]))
        spec["benches"][b] = {
            "n_ids": len(ids[b]), "denominator": denom,
            "contract_denominator": contract_denominator(b), "gated": b != "mc",
            "exclude": (list(exclude) if (b == "humaneval" and exclude is not None)
                        else (list(HE_EXCLUDE) if b == "humaneval" else [])),
            "clean_denominator": HUMANEVAL_CLEAN if b == "humaneval" else None,
            "ids_sha256": sha256_text("\n".join(ids[b])),
            "inputs": {"path": None, "sha256": sha256_text("synthetic:" + b),
                       "n_rows": len(ids[b]), "synthetic": True},
        }
        spec["inputs"][b] = spec["benches"][b]["inputs"]
    return spec


def write_world(root, spec, mutate=None):
    """A complete synthetic E6 output tree, then `mutate(recs, rows)` breaks it.

    Built by the same writers the real run uses and judged by the same merge, so the
    worlds test the shipped decision rather than a re-implementation of it. MC is
    recorded exactly as run_worker records it: worker 0 only, shard_n 1.
    """
    root = pathlib.Path(root)
    (root / "shards").mkdir(parents=True, exist_ok=True)
    n = spec["n_shards"]
    recs, rows = {}, {}
    for i in range(n):
        rec = {"shard_i": i, "shard_n": n, "gpu": str(i), "claim": "e6_synthetic",
               "claim_cards": [str(i)], "started_utc": utc(), "heartbeat_utc": utc(),
               "finished_utc": utc(), "per_bench": {}, "exit_code": 0, "status": "done"}
        for b in spec["benches_requested"]:
            if b == "mc" and i != 0:
                continue
            nb = 1 if b == "mc" else n
            shard = 0 if b == "mc" else i
            want = spec["ids"][b][shard::nb]
            rows[(i, b)] = [{"bench": b, "id": x, "passed": True, "prompt_sha256": None,
                             "output": "", "stop_reason": "eos", "tokens": None,
                             "fail_class": None, "shard_i": shard} for x in want]
            rec["per_bench"][b] = {
                "bench": b, "shard_i": shard, "shard_n": nb, "gpu": str(i),
                "claim": "e6_synthetic", "cmd": f"synthetic {b} {shard}/{nb}",
                "log": f"logs/shard_{i}.{b}.log", "started_utc": utc(),
                "finished_utc": utc(), "pid": 1000 + i,
                "input_sha256": spec["inputs"][b]["sha256"], "input_path": None,
                "raw_preds": None, "raw_preds_sha256": None, "n_raw_rows": len(want),
                "jsonl": f"shards/{i}.{b}.jsonl", "ids": list(want),
                "n_ids_expected": len(want), "exit_code": 0, "normalize_error": None}
        recs[i] = rec
    if mutate:
        mutate(recs, rows)
    for i in range(n):
        write_json(root / "shards" / f"{i}.json", recs[i])
        for b in spec["benches_requested"]:
            write_jsonl(root / "shards" / f"{i}.{b}.jsonl", rows.get((i, b), []))
    write_json(root / "seal.json",
               {"tag": spec["tag"], "ckpt": spec["ckpt"], "protocol": spec["protocol"],
                "n_shards": n, "cmd": "synthetic world", "pid": os.getpid(),
                "started_utc": utc(), "inputs": spec["inputs"],
                "shards": {str(i): {"exit_code": (recs[i] or {}).get("exit_code")}
                           for i in range(n)}})
    return root


def merge_selftest(paths, log, n_shards=4):
    """The four required worlds -- plus a seal world -- over CONSTRUCTED shard sets.

    complete -> PASS. missing shard -> FAIL on coverage. duplicate ID and missing ID ->
    FAIL on the ID set. wrong denominator, in both shapes (a declared denominator that
    disagrees with the data, and a 7-entry HumanEval exclusion list) -> FAIL on the
    denominator. A missing seal field -> FAIL on the seal. A world that fails for the
    WRONG reason does not count: the intended gate must be among the failed ones.
    """
    worlds = []

    def run_world(label, want, gate, mutate=None, spec=None, cls=None):
        spec = spec or build_synthetic_spec(n_shards=n_shards)
        root = paths["dir"] / f"selftest_{label}"
        write_world(root, spec, mutate)
        local = {"shards": root / "shards", "seal": root / "seal.json",
                 "inputs": paths["inputs"], "plan": paths["plan"]}
        v, _merged, gates, _result = merge_run(local, spec, log)
        failed = {k: w["gate_failed"] for k, w in v.items() if w["gate_failed"]}
        if want == "PASS":
            ok = bool(gates["merge"] and gates["e6"]) and not failed
            why = "all gates pass" if ok else f"gate_failed={failed} gates={gates}"
        else:
            hit = any(gate in w["gate_failed"] for w in v.values())
            ok = (not gates["e6"]) and hit
            why = (f"failed on {gate}" if hit
                   else f"failed, but not on {gate}: gate_failed={failed}")
        worlds.append({"world": label, "class": cls or label, "expected": want,
                       "gate": gate, "ok": bool(ok), "why": why, "gates": gates,
                       "gate_failed": failed})
        log.line(f"world {label}: expected {want} on {gate} -> "
                 f"{'OK' if ok else 'MISJUDGED'} ({why})")
        return ok

    def drop_shard(recs, rows):
        recs[2] = None
        rows.pop((2, "gsm8k"), None)

    def dup_id(recs, rows):
        r = rows[(1, "gsm8k")]
        r[0] = dict(r[0], id=r[1]["id"])          # one ID duplicated, one dropped, count same

    def missing_id(recs, rows):
        rows[(1, "math500_en")] = rows[(1, "math500_en")][:-1]

    def seal_gap(recs, rows):
        recs[0]["per_bench"]["humaneval"]["started_utc"] = None

    run_world("complete", "PASS", "all", cls="complete")
    run_world("missing_shard", "FAIL", "shards", drop_shard, cls="missing_shard")
    run_world("duplicate_id", "FAIL", "ids", dup_id, cls="id_set")
    run_world("missing_id", "FAIL", "ids", missing_id, cls="id_set")
    run_world("denominator_declared", "FAIL", "denominator",
              spec=build_synthetic_spec(n_shards=n_shards,
                                        denom_override={"gsm8k": GSM8K_DENOM - 1}),
              cls="denominator")
    run_world("denominator_exclusion", "FAIL", "denominator",
              spec=build_synthetic_spec(n_shards=n_shards, exclude=HE_EXCLUDE[:7]),
              cls="denominator")
    run_world("seal_gap", "FAIL", "seal", seal_gap, cls="seal")
    bad = [w for w in worlds if not w["ok"]]
    classes = {}
    for w in worlds:
        verdict = "PASS" if (w["ok"] and w["expected"] == "PASS") else \
            ("FAIL" if (w["ok"] and w["expected"] == "FAIL") else "MISJUDGED")
        if classes.get(w["class"], "PASS") != "MISJUDGED":
            classes[w["class"]] = verdict if w["class"] not in classes else \
                ("PASS" if classes[w["class"]] == "PASS" and verdict == "PASS" else "FAIL")
    for w in bad:
        log.line(f"FAIL e6 merge control: world {w['world']} expected {w['expected']} on "
                 f"{w['gate']}, got {w['why']}")
    controls = {"merge_selftest": {"worlds": worlds, "n_worlds": len(worlds),
                                   "n_ok": len(worlds) - len(bad), "classes": classes,
                                   "verdict": "PASS" if not bad else "FAIL"}}
    return controls, (not bad)


def check_id_source(name, path, want, log):
    """One ID-space check with a three-state result: PASS, SKIP or FAIL.

    A SKIP is not a PASS: it says the source is not on this host, so this control does
    not certify that denominator. It is recorded by name and reason, and the summary's
    controls verdict reads SKIP rather than PASS.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        if name == "math500_en" and MATH500_RAW.is_file():
            # The derived file is produced at plan time; on a host that has only the raw
            # gold schema, check the thing the derivation is a function of.
            rows = lc.read_jsonl(MATH500_RAW)
            got = [str(r.get("unique_id")) for r in rows]
            ok = (len(rows) == want and len(set(got)) == want
                  and all(str(r.get("problem") or "").strip()
                          and str(r.get("answer") or "").strip()
                          and r.get("level") is not None for r in rows))
            res = {"status": "PASS" if ok else "FAIL", "path": str(MATH500_RAW),
                   "schema": "raw problem/solution (the derived view is made at plan time)",
                   "expected": want, "n_rows": len(rows), "n_unique_ids": len(set(got))}
            log.line(f"id-space {name}: {res['status']} raw {len(rows)} rows, "
                     f"{len(set(got))} unique ids, expected {want}")
            return res
        res = {"status": "SKIP", "path": str(path), "expected": want,
               "why": "the source file is not on this host, so this control cannot certify "
                      "the denominator here"}
        log.line(f"id-space {name}: SKIP -- {res['why']}: {path}")
        return res
    rows = lc.read_jsonl(path)
    if name == "gsm8k":
        got = [f"gsm8k/{i}" for i in range(len(rows))]
    elif name == "math500_en":
        got = [str(r.get("unique_id")) for r in rows]
    else:
        got = [str(r["task_id"]) for r in rows]
    ok = len(rows) == want and len(set(got)) == want
    res = {"status": "PASS" if ok else "FAIL", "path": str(path), "expected": want,
           "n_rows": len(rows), "n_unique_ids": len(set(got))}
    log.line(f"id-space {name}: {res['status']} {len(rows)} rows, {len(set(got))} unique "
             f"ids, expected {want}")
    return res


def id_space_selftest(log):
    """The ID spaces E6 declares, plus a known-answer FAIL world for this checker.

    The FAIL branch is exercised on a MUTATED copy of the real artifact (two rows
    dropped), never on a hand-written world: a checker that only ever sees good input
    has not been shown to refuse anything.
    """
    out, failures, skips = {}, [], []
    checks = [("gsm8k", GSM8K_DATA, GSM8K_DENOM),
              ("math500_en", MATH500_FOR_EVAL, MATH500_DENOM),
              ("humaneval", HUMANEVAL_DATA, HUMANEVAL_DENOM)]
    for name, path, want in checks:
        res = check_id_source(name, path, want, log)
        out[name] = res
        if res["status"] == "FAIL":
            failures.append(f"{name}: {res['n_rows']} rows / {res['n_unique_ids']} unique ids, "
                            f"expected {want}")
        elif res["status"] == "SKIP":
            skips.append(f"{name}: {res['why']} ({res['path']})")
    # the FAIL branch, on a real file with two rows removed
    broken = None
    if GSM8K_DATA.is_file():
        import tempfile
        rows = lc.read_jsonl(GSM8K_DATA)
        d = tempfile.mkdtemp(prefix="e6_ids_")
        broken = pathlib.Path(d) / "gsm8k_broken.jsonl"
        write_jsonl(broken, rows[:-2])
        res = check_id_source("gsm8k", broken, GSM8K_DENOM, log)
        ok = res["status"] == "FAIL"
        out["gsm8k_broken_world"] = {"status": "PASS" if ok else "FAIL",
                                     "why": "the checker reports FAIL on a real file with "
                                            "two rows removed",
                                     "world": str(broken), "got": res["status"],
                                     "n_rows": res.get("n_rows"), "expected": GSM8K_DENOM}
        if not ok:
            failures.append(f"id-space checker accepted a short real file: {res}")
    else:
        out["gsm8k_broken_world"] = {"status": "SKIP",
                                     "why": "no gsm8k source on this host to mutate"}
    if len(set(HE_EXCLUDE)) != 8:
        failures.append("the HumanEval exclusion list is not 8 distinct ids")
    if HUMANEVAL_DATA.is_file():
        ids = {str(r["task_id"]) for r in lc.read_jsonl(HUMANEVAL_DATA)}
        missing = sorted(set(HE_EXCLUDE) - ids)
        if missing:
            failures.append(f"excluded ids absent from HumanEval: {missing}")
    verdict = "FAIL" if failures else ("SKIP" if skips else "PASS")
    return {"checks": out, "failures": failures, "skipped": skips, "verdict": verdict}


def run_controls(paths, log, n_shards=4):
    """CPU-only: the merge worlds plus the CLI and ID-space checks, three-state.

    No checkpoint, no GPU, no child process. This is the acceptance command the contract
    names (section 8, w3), so it runs on any machine.

    The aggregate verdict is PASS, SKIP or FAIL. SKIP means some source is not on this
    host and that control did not certify its denominator; it is never reported as PASS.
    Only FAIL makes the stage exit 3.
    """
    cli_ok, cli = cli_contract_selftest(log)
    cli["verdict"] = "PASS" if cli_ok else "FAIL"
    ids = id_space_selftest(log)
    controls, merge_ok = merge_selftest(paths, log, n_shards=n_shards)
    controls["cli_contract"] = cli
    controls["id_space"] = ids
    states = {"cli_contract": cli["verdict"],
              "id_space": ids["verdict"],
              "merge_selftest": controls["merge_selftest"]["verdict"]}
    verdict = ("FAIL" if "FAIL" in states.values()
               else "SKIP" if "SKIP" in states.values() else "PASS")
    controls["states"] = states
    controls["skipped"] = ([f"id_space: {s}" for s in ids["skipped"]] +
                           ([f"id_space.gsm8k_broken_world: "
                             f"{controls['id_space']['checks']['gsm8k_broken_world']['why']}"]
                            if controls["id_space"]["checks"].get(
                                "gsm8k_broken_world", {}).get("status") == "SKIP" else []))
    controls["verdict"] = verdict
    ok = verdict != "FAIL"
    log.line(f"e6 controls: cli_contract={states['cli_contract']} "
             f"id_space={states['id_space']} "
             f"merge_selftest={states['merge_selftest']} verdict={verdict}"
             + (f" | SKIP: {controls['skipped']}" if controls["skipped"] else ""))
    if verdict == "SKIP":
        log.line("e6 controls verdict SKIP: a source is absent on this host, so this run "
                 "does NOT certify every denominator. The pod holds humaneval_164.jsonl.")
    return controls, ok, verdict


# ---------------------------------------------------------------- stage actions
def action_selftest(paths, log, tag, n_shards):
    started = utc()
    t0 = time.time()
    controls, ok, verdict = run_controls(paths, log, n_shards=n_shards)
    summary = {"stage": "e6", "mode": "merge-selftest", "tag": tag, "ckpt": None,
               "protocol": None, "n": controls["merge_selftest"]["n_worlds"],
               "passed": controls["merge_selftest"]["n_ok"],
               "gates": {"controls_no_failure": ok, "controls_pass": verdict == "PASS"},
               "controls_verdict": verdict,
               "controls_skipped": controls["skipped"],
               "controls": controls, "inputs": {},
               "gpu": None, "pid": os.getpid(), "claim": None, "shards": n_shards,
               "started_utc": started, "finished_utc": utc(),
               "elapsed_s": round(time.time() - t0, 1),
               "exit_code": lc.EXIT_OK if ok else lc.EXIT_GATE}
    lc.seal(paths, tag=tag, mode="merge-selftest", n_shards=n_shards,
            controls_verdict=verdict, controls_skipped=controls["skipped"],
            elapsed_s=summary["elapsed_s"], exit_code=summary["exit_code"])
    return summary


def count_tokens(texts, tokenizer_path, log):
    """n_tokens per item, with the basis stated because it is an approximation.

    The reused runners report no generated-token count. This encodes the text with the
    run's tokenizer, so it is the tokenizer's count of the decoded output, and a
    decode/encode round trip can differ from the number of tokens generated.
    """
    if not texts:
        return [], "no rows"
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tokenizer_path))
    except Exception as exc:  # noqa: BLE001
        log.line(f"n_tokens: unavailable ({type(exc).__name__}: {exc}); written as null")
        return [None] * len(texts), f"tokenizer unavailable: {type(exc).__name__}"
    return [len(tok.encode(t).ids) if t else 0 for t in texts], \
        f"len(tokenizer.encode(response_raw).ids) via {tokenizer_path}; the runner reports " \
        f"no token count, and a decode/encode round trip can differ from the generated count"


def build_item_row(row, shard_meta, tokenizer_path, tokens, basis):
    """The contract's nine fields plus the benchmark tag.

    R3: id, prompt (verbatim), response_raw, parsed, passed (bool), stop_reason,
    n_tokens, fail_class, meta (shard index, ID list, elapsed, child exit code).
    """
    meta = {
        "bench": row.get("bench"),
        "shard_i": shard_meta.get("shard_i"),
        "shard_n": shard_meta.get("shard_n"),
        "shard_ids": shard_meta.get("ids"),
        "shard_n_ids": len(shard_meta.get("ids") or []),
        "shard_elapsed_s": shard_meta.get("elapsed_s"),
        "shard_exit_code": shard_meta.get("exit_code"),
        "shard_gpu": shard_meta.get("gpu"),
        "shard_pid": shard_meta.get("pid"),
        "shard_claim": shard_meta.get("claim"),
        "raw_preds": shard_meta.get("raw_preds"),
        "raw_preds_sha256": shard_meta.get("raw_preds_sha256"),
        "input_sha256": shard_meta.get("input_sha256"),
        "limit_enforced_by": shard_meta.get("limit_enforced_by"),
        "n_tokens_basis": basis,
        "prompt_sha256": row.get("prompt_sha256"),
        "output": row.get("output"),
        "mc_status": row.get("status"),
        "mc_accuracy": row.get("accuracy"),
        "mc_detail": row.get("detail"),
    }
    if row.get("bench") == "mc":
        meta["note"] = ("run_eval.py builds its multiple-choice items internally; it prints "
                        "one accuracy per set and writes no per-item prompt or generation")
    elif row.get("stop_reason") is None:
        meta["stop_reason_basis"] = (f"{row.get('bench')} runs generate_batch and writes no "
                                     f"stop reason, so this field is null; the ladder's "
                                     f"stop_reason() rule needs a last-token id the runner "
                                     f"does not report")
    return {"id": row.get("id"), "prompt": row.get("prompt"),
            "response_raw": row.get("response_raw"), "parsed": row.get("parsed"),
            "passed": row.get("passed"), "stop_reason": row.get("stop_reason"),
            "n_tokens": tokens, "fail_class": row.get("fail_class"), "meta": meta}


def write_outputs(paths, spec, verdicts, merged, gates, result, log):
    """merged.jsonl (per item), result.json (denominators), items.jsonl (contract R3)."""
    tokenizer_path = spec.get("tokenizer_path") or TOKENIZER
    texts = [r.get("response_raw") for r in merged]
    tokens, basis = count_tokens(texts, tokenizer_path, log)
    for row, n in zip(merged, tokens):
        row["n_tokens"] = n
        row["tokens"] = n
    write_jsonl(paths["merged"], merged)
    write_json(paths["result"], result)
    shard_metas = {}
    for i in range(spec["n_shards"]):
        p = paths["shards"] / f"{i}.json"
        if p.is_file():
            rec = json.loads(p.read_text(encoding="utf-8"))
            for b, m in (rec.get("per_bench") or {}).items():
                shard_metas[(i, b)] = m
    items = []
    for row in merged:
        shard_meta = shard_metas.get((row.get("shard_i"), row.get("bench"))) or {}
        items.append(build_item_row(row, shard_meta, tokenizer_path,
                                    row.get("n_tokens"), basis))
    lc.write_items(paths, items)
    log.line(f"wrote {paths['merged']} ({len(merged)} rows), {paths['result']}, "
             f"{paths['items']} ({len(items)} rows, nine fields per R3)")


def action_run(args, paths, log):
    t0 = time.time()
    ctx, plan = build_plan(args, paths, log)
    for name in plan["benches_requested"]:
        b = plan["benches"][name]
        log.line(f"plan {name}: {b['n_ids']} items, denominator {b['denominator']} "
                 f"(contract {b['contract_denominator']}), gated={b['gated']}, "
                 f"limit_enforced_by={b['limit_enforced_by'] or 'none (full set)'}, "
                 f"input {b['inputs']['path']} "
                 f"sha256={(b['inputs']['sha256'] or '')[:16]}")
    if args.dry_run:
        log.line(f"dry-run: no claim, no child process. plan written to {paths['plan']}")
        for name in plan["benches_requested"]:
            for i, cmd in plan["commands"][name].items():
                log.line(f"  [gpu {plan['worker_cards'][i]}] {cmd}")
        lc.seal(paths, tag=args.tag, mode="dry-run", ckpt=ctx["ckpt"],
                plan=str(paths["plan"]), protocol=ctx["protocol"], n_shards=args.shards,
                gpus=ctx["gpus"], inputs={k: v["sha256"] for k, v in ctx["inputs"].items()},
                n_items={k: len(v) for k, v in ctx["ids"].items()}, limit=ctx["limit"],
                limit_enforced_by=ctx["limit_enforced_by"], diagnostic=ctx["diagnostic"])
        return {"stage": "e6", "mode": "dry-run", "tag": args.tag, "ckpt": ctx["ckpt"],
                "ckpt_sha256": ctx["ckpt_sha256"], "protocol": ctx["protocol"],
                "n": sum(len(plan["commands"][b]) for b in plan["benches_requested"]),
                "shards": args.shards, "gpus": ctx["gpus"], "benches": plan["benches"],
                "inputs": ctx["inputs"],
                "limits": {"limit": ctx["limit"], "diagnostic": ctx["diagnostic"],
                           "items_per_bench": {k: len(v) for k, v in ctx["ids"].items()},
                           "limit_enforced_by": ctx["limit_enforced_by"]},
                "gates": {"plan": True}, "claim": None,
                "gpu": None, "pid": os.getpid(), "started_utc": utc(), "finished_utc": utc(),
                "elapsed_s": round(time.time() - t0, 1), "exit_code": lc.EXIT_OK}

    prereq, missing = ({}, [])
    if args.skip_prereq:
        log.line("prereq: SKIPPED by --skip-prereq (recorded in seal.json)")
    else:
        prereq, missing = prereq_verdicts(ctx["tag"], log)
        lc.require(not missing,
                   f"E6 waits for the earlier stages (handover doc section 7): {missing} did "
                   f"not pass on tag {ctx['tag']}. Run them, or pass --skip-prereq to record "
                   f"the deviation explicitly")

    claim_ok, claim_name, claim_msg = (True, None, "no-claim")
    if not args.no_claim:
        claim_ok, claim_name, claim_msg = acquire_claim(ctx, log)
        # P0 (handover doc section 5): a failed claim stops the run, it does not continue.
        lc.require(claim_ok, f"card claim refused, so E6 does not start: {claim_msg}")
    ctx["claim_name"] = claim_name
    STATE["claim"] = claim_name
    errors, started = [], utc()
    try:
        launch(ctx, paths, log, errors)
    finally:
        if claim_name:
            release_claim(claim_name, log)
            STATE["claim"] = None
    log.line(f"workers done: shards {args.shards}, failures: {errors or 'none'}")

    shard_seal = {}
    for i in range(args.shards):
        p = paths["shards"] / f"{i}.json"
        if not p.is_file():
            shard_seal[str(i)] = {"record": "missing"}
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        shard_seal[str(i)] = {
            "gpu": rec.get("gpu"), "claim": rec.get("claim"),
            "exit_code": rec.get("exit_code"), "started_utc": rec.get("started_utc"),
            "finished_utc": rec.get("finished_utc"),
            "per_bench": {b: {"pid": m.get("pid"), "cmd": m.get("cmd"), "log": m.get("log"),
                              "ids": len(m.get("ids") or []),
                              "input_sha256": m.get("input_sha256"),
                              "jsonl": m.get("jsonl"), "exit_code": m.get("exit_code")}
                          for b, m in (rec.get("per_bench") or {}).items()},
        }
    lc.seal(paths, tag=args.tag, mode="run", ckpt=ctx["ckpt"],
            ckpt_sha256=ctx["ckpt_sha256"], protocol=ctx["protocol"],
            plan=str(paths["plan"]), inputs=ctx["inputs"],
            n_items={k: len(v) for k, v in ctx["ids"].items()},
            limit=ctx["limit"], limit_enforced_by=ctx["limit_enforced_by"],
            diagnostic=ctx["diagnostic"],
            n_shards=args.shards, gpus=ctx["gpus"], claim=claim_name, claim_msg=claim_msg,
            started_utc=started, finished_utc=utc(), elapsed_s=round(time.time() - t0, 1),
            prereq=prereq, prereq_skipped=bool(args.skip_prereq), worker_errors=errors,
            shards=shard_seal)
    return merge_stage(args, paths, log, ctx=ctx, errors=errors, started=started, t0=t0)


def merge_stage(args, paths, log, ctx=None, errors=None, started=None, t0=None):
    """Merge whatever the shard directory holds. Writes no total on a failed gate."""
    started = started or utc()
    plan = load_plan(paths)
    spec, plan_failures = spec_from_plan(plan, paths)
    verdicts, merged, gates, result = merge_run(paths, spec, log)
    seal_ok, seal_why = stage_seal_ok(paths)
    gates["plan"] = not plan_failures
    gates["stage_seal"] = seal_ok
    gates["e6"] = bool(gates["e6"] and seal_ok and not plan_failures)
    result["seal_ok"] = seal_ok
    result["plan_failures"] = plan_failures
    if plan_failures or not seal_ok:
        result["no_total_reason"] = (result.get("no_total_reason") or "") + \
            ("; plan: " + "; ".join(plan_failures) if plan_failures else "") + \
            ("; stage seal: " + "; ".join(seal_why) if not seal_ok else "")
    write_outputs(paths, spec, verdicts, merged, gates, result, log)
    shard_records = []
    for i in range(spec["n_shards"]):
        p = paths["shards"] / f"{i}.json"
        if p.is_file():
            shard_records.append(json.loads(p.read_text(encoding="utf-8")))
    summary = {
        "stage": "e6", "mode": "merge-only" if args.merge_only else "run",
        "tag": spec["tag"], "ckpt": spec["ckpt"], "ckpt_sha256": plan.get("ckpt_sha256"),
        "step": None, "protocol": spec["protocol"], "n": len(merged),
        "passed": sum(1 for r in merged if r.get("passed")) if merged else 0,
        "gates": gates, "controls": {}, "controls_verdict": None,
        "limits": {"limit": spec.get("limit"), "diagnostic": spec["diagnostic"],
                   "items_per_bench": spec.get("n_items"),
                   "limit_enforced_by": {k: v.get("limit_enforced_by")
                                         for k, v in spec["benches"].items()}},
        "inputs": spec["inputs"], "gpu": (ctx or {}).get("gpus") or plan.get("gpus"),
        "pid": os.getpid(), "claim": (ctx or {}).get("claim_name"),
        "shards": spec["n_shards"],
        "shard_records": [{"shard_i": r.get("shard_i"), "gpu": r.get("gpu"),
                           "pid": r.get("pid"), "claim": r.get("claim"),
                           "cmd": r.get("cmd"), "input_sha256": r.get("input_sha256"),
                           "exit_code": r.get("exit_code"),
                           "started_utc": r.get("started_utc"),
                           "finished_utc": r.get("finished_utc")}
                          for r in shard_records],
        "benches": result["benches"],
        "totals": result["totals"], "no_total_reason": result["no_total_reason"],
        "worker_errors": errors or [],
        "started_utc": started, "finished_utc": utc(),
        "elapsed_s": round(time.time() - t0, 1) if t0 else None,
        "exit_code": lc.EXIT_OK if gates["e6"] else lc.EXIT_GATE,
    }
    if gates["e6"]:
        log.line("E6 PASSED: " + json.dumps(result["totals"], sort_keys=True))
    else:
        log.line(f"E6 GATE FAILED, no total written: {result['no_total_reason']}")
    return summary


# ---------------------------------------------------------------- entry point
def main(argv):
    ap = argparse.ArgumentParser(prog="eval_ladder.py e6")
    ap.add_argument("--ckpt", default=None, help="checkpoint under test")
    ap.add_argument("--tag", default=None,
                    help="product tag: runs/ladder/<tag>/e6/. Defaults to the checkpoint tag, "
                         "the same directory every other stage uses")
    ap.add_argument("--tokenizer", default=str(TOKENIZER))
    ap.add_argument("--merge-only", action="store_true",
                    help="judge the shards already on disk; launch nothing")
    ap.add_argument("--merge-selftest", action="store_true",
                    help="CPU: constructed shard worlds -- complete (PASS), missing shard, "
                         "duplicate/missing ID, wrong denominator, seal gap (FAIL)")
    ap.add_argument("--selftest-shards", type=int, default=4,
                    help="shard count for the constructed worlds")
    ap.add_argument("--shards", type=int, default=8,
                    help="one worker per shard, one card each")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="cards, as CUDA_VISIBLE_DEVICES")
    ap.add_argument("--only", default=None,
                    help="comma list of benchmarks (diagnostic: the gate then reports "
                         "contract_complete=false)")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N items per benchmark (diagnostic: the gate then FAILs the "
                         "contract denominator by design)")
    ap.add_argument("--protocol", choices=["base", "chatml"], default="base",
                    help="base = continuation + --rstrip_nl (the gate column since "
                         "2026-09-14); chatml = the SFT arm. Handover doc P2: pick by "
                         "checkpoint kind, never mix")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="let the runners overwrite artifacts")
    ap.add_argument("--dry-run", action="store_true", help="write the plan, launch nothing")
    ap.add_argument("--skip-prereq", action="store_true",
                    help="do not refuse when E1/E3/E4/E5 have not passed; recorded in seal.json")
    ap.add_argument("--no-claim", action="store_true",
                    help="do not acquire a card claim (diagnostic: the seal gate then fails, "
                         "because section 7 requires a claim per shard)")
    ap.add_argument("--skip-forward", action="store_true",
                    help="accepted for the ladder's shared refuse-path control "
                         "(eval/eval_ladder.py:39-79). E6 runs no forward pass of its own: "
                         "every number comes from a child evaluator, so there is nothing here "
                         "for this flag to skip")
    args = ap.parse_args(argv)

    if not args.tag:
        args.tag = lc.ckpt_tag(args.ckpt) if args.ckpt else DEFAULT_TAG_SELFTEST
    paths = stage_dir(args.tag)
    if args.ckpt:
        base = lc.ckpt_tag(args.ckpt)
        if args.tag == base:
            frozen = lc.stage_paths(args.ckpt, "e6")["dir"]
            lc.require(frozen == paths["dir"],
                       f"tag {args.tag} does not name the checkpoint's own stage directory "
                       f"({frozen})")
        else:
            lc.require(args.tag.startswith(base + "_") or args.tag.startswith(base + "."),
                       f"--tag {args.tag!r} must be {base!r} or start with {base}_ : the tag is "
                       f"how a product names the checkpoint it belongs to, and a diagnostic "
                       f"variant directory must still say which checkpoint it is a variant of")
    log = Tee(paths["log"])
    STATE["log"] = log
    install_signal_handlers()
    if args.ckpt and args.tag != lc.ckpt_tag(args.ckpt):
        log.line(f"tag {args.tag} is a variant of the checkpoint tag "
                 f"{lc.ckpt_tag(args.ckpt)}: this run's products do not claim to be the "
                 f"checkpoint's own E6 directory")
    try:
        if args.merge_selftest:
            summary = action_selftest(paths, log, args.tag, args.selftest_shards)
        elif args.ckpt is None:
            lc.require(False, "E6 needs --ckpt (or --merge-selftest for the CPU controls)")
            raise AssertionError("unreachable")
        else:
            lc.require(pathlib.Path(args.ckpt).is_file(),
                       f"checkpoint {args.ckpt} does not exist")
            summary = (merge_stage(args, paths, log) if args.merge_only
                       else action_run(args, paths, log))
        lc.write_summary(paths, summary)
        log.line(f"e6 {summary['mode']}: exit_code={summary['exit_code']}")
        return summary["exit_code"]
    except lc.Refuse as exc:
        log.line(f"REFUSE: {exc}")
        lc.write_summary(paths, {"stage": "e6", "refused": str(exc),
                                 "exit_code": lc.EXIT_REFUSE})
        return lc.EXIT_REFUSE
    except Exception:
        import traceback
        log.line("CRASH:\n" + traceback.format_exc())
        lc.write_summary(paths, {"stage": "e6", "crashed": traceback.format_exc()[-2000:],
                                 "exit_code": lc.EXIT_CRASH})
        return lc.EXIT_CRASH
    finally:
        log.close()


def control_selftest():
    """The dispatcher's CPU entry: `eval_ladder.py selftest --only e6`.

    Zero arguments by contract (eval/eval_ladder.py:81-85 calls fn()), so it builds
    its own stage directory and writes its own summary.json.
    """
    paths = stage_dir(DEFAULT_TAG_SELFTEST)
    log = Tee(paths["log"])
    try:
        summary = action_selftest(paths, log, DEFAULT_TAG_SELFTEST, 4)
        lc.write_summary(paths, summary)
        return summary["exit_code"]
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

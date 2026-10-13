#!/usr/bin/env python3
# restartable: re-executes already generated completions (no model call) and writes its report at the
# end; an interrupt loses only CPU seconds of sandbox runs and leaves no partial file.
"""Re-score existing HumanEval completions against HumanEval+ (EvalPlus) tests.

WHY. HumanEval's bundled tests are thin; EvalPlus adds ~80x more inputs per task and turns a
fraction of "passing" solutions into failures. That fraction is the gap between "the model
wrote something that passes 7 asserts" and "the model wrote a correct function". The plateau
case study found 83/129 tasks failed by every checkpoint and a handful flipping between
checkpoints; HumanEval+ tells whether those flips are real or test-luck.

NO NEW GENERATION. HumanEval+ uses the same 164 prompts, so the completion each checkpoint
already produced is re-judged in the same sandbox. The generation cost is zero and the
pairing with the base HumanEval score is exact: one completion, two test suites.

DATA. EvalPlus's HumanEvalPlus.jsonl (one row per task: task_id, prompt, entry_point,
canonical_solution, base_input, plus_input, atol). It is NOT vendored -- fetch it once and
pin its sha256 with --plus_sha256. The expected outputs are computed here by running the
canonical solution on every input in the sandbox, then the candidate must match them. A
canonical solution that cannot reproduce itself is a data fault and the task is refused.

    python3 eval/humaneval_plus_rescore.py \
        --preds data/eval/preds_humaneval_ckpt_v42s2.pt.rstripnl.comp_v42s2.jsonl \
        --plus data/eval/humaneval/HumanEvalPlus.jsonl --plus_sha256 <sha> \
        --rstrip_nl --out data/eval/preds_heplus_v42s2.jsonl
    python3 eval/humaneval_plus_rescore.py --selftest      # no sandbox, no data

Output rows keep {task_id, ok, ok_base} so eval/paired_compare.py reads them directly and the
base/plus disagreement is in the same file.
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# One program per task: canonical and candidate are both defined, every input is fed to both,
# and the candidate must match the canonical result (float tolerance per EvalPlus `atol`).
HARNESS = r'''
import math, sys
{prefix}
{canonical}
_ref = {entry}
del {entry}
{prefix}
{candidate}
_cand = {entry}
_inputs = {inputs!r}
_atol = {atol!r}
def _eq(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=_atol or 1e-6) or (a != a and b != b)
        except TypeError:
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return a == b
_bad = 0
for _args in _inputs:
    import copy
    _r = _ref(*copy.deepcopy(_args))
    _c = _cand(*copy.deepcopy(_args))
    if not _eq(_c, _r):
        _bad += 1
        break
sys.exit(1 if _bad else 0)
'''


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_prefix(prompt, entry):
    """Module text before `def entry` (imports and helper defs the candidate may rely on)."""
    i = prompt.find(f"def {entry}(")
    return prompt[:i] if i >= 0 else ""


def build_program(task, candidate_fn_src, max_inputs):
    inputs = list(task.get("base_input", [])) + list(task.get("plus_input", []))
    if max_inputs:
        inputs = inputs[:max_inputs]
    prefix = prompt_prefix(task["prompt"], task["entry_point"])
    canonical = task["prompt"][len(prefix):] + task["canonical_solution"]
    return HARNESS.format(prefix=prefix, canonical=canonical, candidate=candidate_fn_src,
                          entry=task["entry_point"], inputs=inputs, atol=task.get("atol", 0) or 0)


def candidate_source(task, completion, rstrip_nl):
    """The function the checkpoint wrote: prompt (as fed) + completion, minus the prefix."""
    prompt = task["prompt"].rstrip("\n") if rstrip_nl else task["prompt"]
    full = prompt + completion
    return full[len(prompt_prefix(task["prompt"], task["entry_point"])):]


def run(src, executor, timeout):
    rc = executor(src, timeout)
    return rc == 0


def default_executor(src, timeout):
    from datagen.sandbox_exec import run_sandboxed
    rc, _o, _e = run_sandboxed(src, timeout=timeout, nproc=int(os.environ.get("SANDBOX_NPROC", "4096")))
    return rc


def rescore(preds_path, plus_path, rstrip_nl, executor, timeout, max_inputs, limit=None):
    tasks = {}
    with open(plus_path, encoding="utf-8") as fh:
        for line in fh:
            t = json.loads(line)
            tasks[t["task_id"]] = t
    out, faults = [], []
    with open(preds_path, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    rows = [r for r in rows if "task_id" in r and "_header" not in r]
    for r in rows[:limit] if limit else rows:
        t = tasks.get(r["task_id"])
        if t is None:
            faults.append({"task_id": r["task_id"], "fault": "absent from HumanEval+"})
            continue
        # Data check first: the canonical solution must agree with itself on every input.
        canon_fn = t["prompt"][len(prompt_prefix(t["prompt"], t["entry_point"])):] + t["canonical_solution"]
        if not run(build_program(t, canon_fn, max_inputs), executor, timeout):
            faults.append({"task_id": r["task_id"], "fault": "canonical failed its own inputs"})
            continue
        ok_plus = run(build_program(t, candidate_source(t, r.get("gen", ""), rstrip_nl), max_inputs),
                      executor, timeout) if r.get("ok") else False
        # A completion that already fails the base tests cannot pass a superset; skipping the
        # run is exact, not an approximation, and halves the sandbox cost.
        out.append({"task_id": r["task_id"], "ok": bool(ok_plus), "ok_base": bool(r.get("ok"))})
    return out, faults


def _selftest():
    task = {"task_id": "T/0", "entry_point": "add",
            "prompt": "def add(a, b):\n    \"\"\"Add.\"\"\"\n",
            "canonical_solution": "    return a + b\n",
            "base_input": [[1, 2]], "plus_input": [[-5, 5], [10 ** 6, 1]], "atol": 0}
    # Executor stub: run in-process, return 0/1 like a process exit code.
    def stub(src, timeout):
        try:
            exec(compile(src, "<t>", "exec"), {"__name__": "__main__"})
        except SystemExit as e:
            return int(e.code or 0)
        except Exception:
            return 1
        return 0
    good = candidate_source(task, "    return a + b\n", False)
    weak = candidate_source(task, "    return 3 if (a, b) == (1, 2) else 0\n", False)
    assert run(build_program(task, good, 0), stub, 5)
    assert not run(build_program(task, weak, 0), stub, 5), "plus inputs must catch a base-only hack"
    # rstrip_nl reconstructs exactly the bytes fed to the model.
    assert candidate_source(task, "\n    return a + b\n", True).startswith("def add(a, b):")
    print("humaneval_plus_rescore selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preds")
    ap.add_argument("--plus", help="HumanEvalPlus.jsonl")
    ap.add_argument("--plus_sha256", help="refuse unless the data file matches")
    ap.add_argument("--rstrip_nl", action="store_true", help="the preds were generated with --rstrip_nl")
    ap.add_argument("--timeout", type=float, default=20)
    ap.add_argument("--max_inputs", type=int, default=0, help="cap inputs per task (0 = all)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not (args.preds and args.plus and args.out):
        ap.error("--preds, --plus and --out are required")
    if args.plus_sha256 and sha256(args.plus) != args.plus_sha256:
        raise SystemExit(f"{args.plus} sha256 mismatch; refusing to score against unpinned data")
    rows, faults = rescore(args.preds, args.plus, args.rstrip_nl, default_executor,
                           args.timeout, args.max_inputs, args.limit)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_header": 1, "preds": os.path.basename(args.preds),
                             "plus_sha256": sha256(args.plus), "rstrip_nl": args.rstrip_nl,
                             "faults": faults}) + "\n")
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n = len(rows)
    base = sum(r["ok_base"] for r in rows)
    plus = sum(r["ok"] for r in rows)
    print(f"scored {n}: base {base}/{n}, plus {plus}/{n}; base-only passes {base - plus}; "
          f"data faults {len(faults)}")


if __name__ == "__main__":
    sys.exit(main())

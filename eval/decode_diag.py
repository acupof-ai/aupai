#!/usr/bin/env python3
# restartable: reads per-task preds files and writes at most one small JSON at the end; a rerun costs
# under a second and overwrites the same output.
"""Decode-failure diagnostics over existing per-task preds. DIAGNOSIS ONLY: it never changes
a pass/fail and its numbers never enter a capability score.

WHY. The case study saw repetition drop from 14 tasks (base) to 5-7 (stage-3). A decode
change (repetition penalty, larger max_new) could lift the headline number without the model
knowing more, which would hide the plateau rather than fix it. So this file separates the
failure modes and reports, per run, how many FAILED tasks each mode explains. Fixing a mode
that explains 2 tasks is not worth a decode change; that judgement needs the count.

MODES (first match wins, in this order; a passing task is never classified):
  empty        generation is blank after strip
  truncated    stop_reason == "length": hit max_new before a stop sequence
  repetitive   eval/humaneval_gen.repetitive(gen): a line/token repeated >=3 times in the tail
  syntax       prompt+gen does not parse as Python
  logic        everything else: well-formed code that fails its tests

    python3 eval/decode_diag.py base=preds_base.jsonl s62269=preds_62269.jsonl [--json out]
    python3 eval/decode_diag.py --selftest
"""
import argparse
import ast
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

MODES = ("empty", "truncated", "repetitive", "syntax", "logic")


_REP = None


def _repetitive(s):
    """eval/humaneval_gen.repetitive, loaded by source so this file needs no torch.

    Extracting the function node keeps ONE definition: a copy here would drift from the
    eval's own rule.
    """
    global _REP
    if _REP is None:
        import re
        path = os.path.join(ROOT, "eval", "humaneval_gen.py")
        src = open(path, encoding="utf-8").read()
        node = next(n for n in ast.parse(src).body
                    if isinstance(n, ast.FunctionDef) and n.name == "repetitive")
        ns = {"re": re}
        exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
        _REP = ns["repetitive"]
    return _REP(s)


def classify(row, prompt):
    if row.get("ok"):
        return None
    gen = row.get("gen") or ""
    if not gen.strip():
        return "empty"
    if row.get("stop_reason") == "length":
        return "truncated"
    if _repetitive(gen):
        return "repetitive"
    try:
        ast.parse(prompt + gen)
    except SyntaxError:
        return "syntax"
    return "logic"


def read_preds(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("_header"):
            continue
        if r["task_id"] in rows:
            raise SystemExit(f"{path}: duplicate task_id {r['task_id']} (n>1 preds are not supported)")
        rows[r["task_id"]] = r
    return rows


def diagnose(runs, prompts):
    """runs: {label: {task_id: row}}. Only tasks common to ALL runs are counted."""
    common = sorted(set.intersection(*(set(v) for v in runs.values())))
    out = {"common_tasks": len(common), "runs": {}}
    for label, rows in runs.items():
        counts = {m: 0 for m in MODES}
        tasks = {m: [] for m in MODES}
        passed = 0
        for t in common:
            mode = classify(rows[t], prompts.get(t, ""))
            if mode is None:
                passed += 1
                continue
            counts[mode] += 1
            tasks[mode].append(t)
        failed = len(common) - passed
        out["runs"][label] = {
            "passed": passed, "failed": failed, "counts": counts,
            "share_of_failures": {m: round(counts[m] / failed, 4) if failed else 0.0 for m in MODES},
            # upper bound of what a perfect decode fix could add: every non-logic failure passes
            "decode_fix_upper_bound_tasks": failed - counts["logic"],
            "tasks": tasks,
        }
    return out


def _fmt(d):
    lines = [f"common tasks: {d['common_tasks']}  (diagnosis only; not a capability score)",
             f"{'run':<12}{'pass':>6}{'fail':>6}" + "".join(f"{m:>12}" for m in MODES) + f"{'decode_ub':>11}"]
    for label, r in d["runs"].items():
        lines.append(f"{label:<12}{r['passed']:>6}{r['failed']:>6}"
                     + "".join(f"{r['counts'][m]:>12}" for m in MODES)
                     + f"{r['decode_fix_upper_bound_tasks']:>11}")
    return "\n".join(lines)


def load_prompts(path):
    p = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            p[r["task_id"]] = r["prompt"].rstrip("\n")
    return p


def _selftest():
    prompts = {"T/0": "def f(x):", "T/1": "def g(x):", "T/2": "def h(x):", "T/3": "def k(x):",
               "T/4": "def m(x):", "T/5": "def n(x):"}
    rows = {
        "T/0": {"task_id": "T/0", "gen": "\n    return x", "ok": True, "stop_reason": "stop"},
        "T/1": {"task_id": "T/1", "gen": "   ", "ok": False, "stop_reason": "stop"},
        "T/2": {"task_id": "T/2", "gen": "\n    return (x", "ok": False, "stop_reason": "length"},
        "T/3": {"task_id": "T/3", "gen": "\n    y = 1\n    y = 1\n    y = 1\n", "ok": False, "stop_reason": "stop"},
        "T/4": {"task_id": "T/4", "gen": "\n    return (x", "ok": False, "stop_reason": "stop"},
        "T/5": {"task_id": "T/5", "gen": "\n    return x + 1", "ok": False, "stop_reason": "stop"},
    }
    d = diagnose({"a": rows, "b": {k: v for k, v in rows.items() if k != "T/5"}}, prompts)
    ra = d["runs"]["a"]
    assert d["common_tasks"] == 5, d["common_tasks"]  # T/5 not common
    assert ra["passed"] == 1 and ra["counts"] == {"empty": 1, "truncated": 1, "repetitive": 1,
                                                  "syntax": 1, "logic": 0}, ra["counts"]
    assert ra["decode_fix_upper_bound_tasks"] == 4
    d2 = diagnose({"a": rows}, prompts)
    assert d2["runs"]["a"]["counts"]["logic"] == 1
    print("decode_diag selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*", help="label=preds.jsonl")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "eval", "humaneval", "humaneval_164.jsonl"))
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return 0
    if not a.runs:
        ap.error("give at least one label=preds.jsonl")
    runs = {}
    for spec in a.runs:
        label, _, path = spec.partition("=")
        if not path:
            ap.error(f"bad run spec {spec!r}; use label=path")
        runs[label] = read_preds(path)
    d = diagnose(runs, load_prompts(a.data))
    print(_fmt(d))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

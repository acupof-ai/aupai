#!/usr/bin/env python3
"""selftest for 44-14 defect 1: eval/boolq.py and eval/openbookqa.py are
dataset loaders for run_eval.py; scoring must live only in run_eval.score_mc.
A second scorer in the benchmark modules is how one checkpoint gets two
numbers.

    python3 scripts/test_eval_load_only.py
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for rel in ("eval/boolq.py", "eval/openbookqa.py"):
    text = open(os.path.join(ROOT, rel), encoding="utf-8").read()
    assert "log_likelihood" not in text, f"{rel}: benchmark module must not score"
    assert "def evaluate" not in text, f"{rel}: evaluate() belongs in run_eval.py"

print("selftest OK: boolq/openbookqa are load-only; the scorer lives in run_eval")

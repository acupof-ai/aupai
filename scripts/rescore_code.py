#!/usr/bin/env python3
"""Independently re-derive `ok` for a code preds file (t28 format: q/gen/expected/ok).

Imports the real predicate from eval/code_zh.py (extract_code + score_code), so a
rescore can only disagree with the stored column if the stored generation or the
scorer changed. Usage: python scripts/rescore_code.py data/eval/preds_code_X.jsonl
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))

from code_zh import extract_code, score_code  # noqa: E402

path = sys.argv[1]
n = agree = stored_ok = rescored_ok = no_fence = 0
for line in open(path, encoding="utf-8"):
    r = json.loads(line)
    code = extract_code(r["gen"])
    if code is None:
        ok = False
        no_fence += 1
    else:
        ok, _, _ = score_code(code, r["expected"])
    n += 1
    stored_ok += bool(r.get("ok"))
    rescored_ok += ok
    agree += bool(r.get("ok")) == ok
print(
    f"n={n} stored_ok={stored_ok} rescored_ok={rescored_ok} "
    f"row_agree={agree}/{n} reproduces={stored_ok == rescored_ok and agree == n} "
    f"no_fence={no_fence} ({100 * no_fence / n:.1f}%)"
)

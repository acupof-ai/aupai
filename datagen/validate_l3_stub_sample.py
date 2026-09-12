#!/usr/bin/env python3
"""0e-8 gate: draw a deterministic 200-row sample from the stub domain and
verify every row ast-parses and its first def has a non-empty real body.
Prints PASS/FAIL plus 10 shown rows (first 10 of the sample).

# restartable: read-only validator over an already-built domain; it writes
# only /tmp/l3_stub_sample.json, so an interrupt is cheap and a rerun reproduces it.
"""
import ast
import glob
import json
import os
import random
import sys

DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "data/corpus/code_ultra_l3_stub_dc"
N = 200


def first_def_body_nonempty(src):
    tree = ast.parse(src)
    fn = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if fn is None:
        return False, "no_func"
    real = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                                       and isinstance(s.value.value, str))]
    if not real:
        return False, "docstring_only"
    if all(isinstance(s, ast.Pass) or
           (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
            and s.value.value is Ellipsis) for s in real):
        return False, "pass_only"
    return True, "ok"


def main():
    files = sorted(glob.glob(os.path.join(DOMAIN, "code_ultra_l3_stub_dc_[0-9]*.jsonl")))
    rows = []
    for f in files:
        for line in open(f):
            rows.append(json.loads(line)["content"])
    rng = random.Random(20260912)
    idx = sorted(rng.sample(range(len(rows)), min(N, len(rows))))
    sample = [rows[i] for i in idx]
    fails = []
    for n, src in enumerate(sample):
        try:
            ok, why = first_def_body_nonempty(src)
        except SyntaxError as e:
            ok, why = False, f"syntax:{e}"
        if not ok:
            fails.append((n, why))
    print(f"SAMPLE rows_total={len(rows)} sampled={len(sample)} fails={len(fails)}")
    if fails:
        print("FAILURES", fails[:20])
    print("RESULT", "PASS" if not fails else "FAIL")
    shown = sample[:10]
    out = {"n_sampled": len(sample), "n_total": len(rows), "fails": fails,
           "shown": shown}
    json.dump(out, open("/tmp/l3_stub_sample.json", "w"), ensure_ascii=False, indent=1)
    for i, s in enumerate(shown):
        print(f"\n===== SHOWN {i} =====")
        print(s[:1200])


if __name__ == "__main__":
    main()

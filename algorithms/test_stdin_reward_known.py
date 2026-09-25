#!/usr/bin/env python3
"""Known-answer validation for the stdin/stdout reward on real APPS/TACO rows.

Selects, per source, 50 stdin-style problems whose OWN reference solution
passes its IO cases (exact stdout, or numeric-token tolerance when the
mismatch is purely float formatting). Problems where the reference output is a
different-but-valid answer (multi-solution, e.g. an ordering the dataset also
accepts) cannot be scored by exact stdout comparison and are SKIPPED, counted,
not forced. For each selected problem:
  reference -> 1.0; a stdout-corrupting mutant -> 0.0; an infinite loop ->
  timeout 0.0 and returns within the timeout (does not hang).
"""

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "algorithms"))
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)
import code_reward as cr

LOOP = "while True:\n    pass\n"
MUT = "\nimport sys as _s\n_s.stdout.write('MUT')\n"
TOL = {"rel_tol": 1e-6, "abs_tol": 1e-6}


def classify(got_lines, want_lines):
    """equal / numeric (only float-formatted numbers differ) / structural."""
    if got_lines == want_lines:
        return "equal"
    if len(got_lines) != len(want_lines):
        return "structural"
    for g, w in zip(got_lines, want_lines):
        if g == w:
            continue
        gt, wt = g.split(), w.split()
        if len(gt) != len(wt):
            return "structural"
        for a, b in zip(gt, wt):
            if a == b:
                continue
            try:
                fa, fb = float(a), float(b)
            except ValueError:
                return "structural"
            if not math.isclose(fa, fb, rel_tol=1e-6, abs_tol=1e-6):
                return "structural"
    return "numeric"


def run_source(source, rows, n_target, timeout):
    selected = checked = numeric = structural = 0
    fails = []
    for r in rows:
        if selected >= n_target:
            break
        try:
            io = json.loads(r.get("input_output") or "{}")
            sols = json.loads(r.get("solutions") or "[]")
        except Exception:
            continue
        if io.get("fn_name"):
            continue
        ins, outs = io.get("inputs"), io.get("outputs")
        if not ins or not isinstance(ins[0], str) or not sols:
            continue
        checked += 1
        ref = sols[0].strip()
        # exact first
        exact = [{"input": a, "output": e} for a, e in zip(ins, outs)][:10]
        rr = cr.score_stdin(ref, exact, timeout=timeout)
        kind = None
        if rr["reward"] == 1.0:
            cases, kind = exact, "exact"
        else:
            # decide whether the only mismatch is float formatting
            gl = cr._norm_stdout(cr.score_stdin(ref, exact[:1], timeout=timeout)["stdout"])
            wl = cr._norm_stdout(exact[0]["output"])
            verdict = classify(gl, wl) if len(gl) == len(wl) else "structural"
            if verdict == "numeric":
                tolc = [dict(c, **TOL) for c in exact]
                rt = cr.score_stdin(ref, tolc, timeout=timeout)
                if rt["reward"] == 1.0:
                    cases, kind = tolc, "numeric_tol"
        if kind is None:
            # could be a genuinely bad ref, float on a later case, or multi-solution
            if verdict == "structural":
                structural += 1
            continue
        if kind == "numeric_tol":
            numeric += 1
        # mutant and loop on this accepted problem
        if cr.score_stdin(ref + MUT, cases, timeout=timeout)["reward"] != 0.0:
            fails.append((source, "mutant_passed"))
        rl = cr.score_stdin(LOOP, cases[:1], timeout=3)
        if rl["reward"] != 0.0 or not rl["timed_out"]:
            fails.append((source, "loop_not_timeout"))
        selected += 1
    print(
        f"{source}: selected={selected} scanned_stdin={checked} "
        f"numeric_tol={numeric} structural_multisol_skipped={structural} fails={len(fails)}"
    )
    for f in fails[:10]:
        print("  FAIL", f)
    return selected, fails


def iter_apps(path):
    for line in open(path):
        yield json.loads(line)


def iter_taco(d):
    import pyarrow.parquet as pq

    for f in sorted(glob.glob(os.path.join(d, "train-*.parquet"))):
        t = pq.read_table(f)
        C = {n: t.column(n).to_pylist() for n in t.column_names}
        for i in range(t.num_rows):
            yield {n: C[n][i] for n in C}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", required=True)
    ap.add_argument("--taco", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=10)
    a = ap.parse_args()
    ca, fa = run_source("apps", iter_apps(a.apps), a.n, a.timeout)
    ct, ft = run_source("taco", iter_taco(a.taco), a.n, a.timeout)
    if ca < a.n or ct < a.n:
        print(f"WARN only selected apps={ca} taco={ct}")
    sys.exit(1 if (fa or ft or ca < a.n or ct < a.n) else 0)


if __name__ == "__main__":
    main()

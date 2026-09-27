#!/usr/bin/env python3
"""Verify TACO/APPS reference solutions in a CPU sandbox for the reasoning SFT pack.

de, 1e ruling 2026-09-27. Inputs (fetched by scripts/fetch_taco_apps.sh, sha-verified):
  data/sft_raw/taco/train-*.parquet   BAAI/TACO, 25,443 train problems
  data/sft_raw/apps/train.jsonl       codeparrot/apps, 5,000 train problems

For each problem we keep ONLY a reference solution that, when run in the isolate sandbox,
passes the problem's own bundled stdin/stdout cases. This is executable verification, not a
heuristic: the produced SFT answer is known to run. Function-call problems (non-empty
starter_code) are skipped -- v1 keeps stdin/stdout scripts, which is what one isolate.run with
stdin_data can execute.

CPU DISCIPLINE (1e 2026-09-27): the box also runs pretraining + HumanEval evals, so the pool
is taskset -c 146-179 and NICE=10; OMP/Rayon threads are capped. Do not widen without a ruling.

Outputs (one JSON record per VERIFIED problem; one chosen solution per problem):
  data/sft_raw/verified/taco_verified.jsonl
  data/sft_raw/verified/apps_verified.jsonl
  data/sft_raw/verified/verify_stats.json
Decontamination and the held-out split are separate steps (sft_reason_build.py).

    # pod, pinned to the granted cores, beside live training (no GPU touched):
    taskset -c 146-179 nice -n 10 OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2 \\
        python3 scripts/sft_verify_code.py --taco data/sft_raw/taco --apps data/sft_raw/apps \\
        --out data/sft_raw/verified --max-cases 8 --max-solutions 6 --keep-solutions 3
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Hard per-case wall/cpu caps: a correct contest solution on a small bundled case is fast. A
# solution that hangs or burns the CPU is exactly one we do not want to teach.
RUN_TIMEOUT = float(os.environ.get("SFT_VERIFY_TIMEOUT", "6"))
RUN_MEM_MB = int(os.environ.get("SFT_VERIFY_MEM_MB", "1024"))
# RLIMIT_NPROC inside the chroot. The sandbox historical default 64 is now below what a stock
# CPython 3.12 needs just to start (measured: setpriv EAGAIN at 64, OK at 4096 on 2026-09-27).
# Still a hard fork cap, just one a contest solution can actually run under.
RUN_NPROC = int(os.environ.get("SFT_VERIFY_NPROC", "4096"))


def parse_io(raw):
    """(inputs, outputs) lists of str, or None if not a runnable stdin/stdout case set."""
    if not raw:
        return None
    try:
        d = json.loads(raw)
    except (ValueError, TypeError):
        return None
    ins, outs = d.get("inputs"), d.get("outputs")
    if not isinstance(ins, list) or not isinstance(outs, list) or not ins:
        return None
    if len(ins) != len(outs):
        return None
    if not all(isinstance(x, str) for x in ins) or not all(isinstance(x, str) for x in outs):
        return None
    return ins, outs


def parse_solutions(raw, limit):
    if not raw:
        return []
    try:
        sols = json.loads(raw)
    except (ValueError, TypeError):
        return []
    out = []
    for s in sols[:limit]:
        if isinstance(s, str) and "input(" in s and s.strip():
            out.append(s)  # stdin-reading python; isolate runs it as a script
    return out


def normalize(s):
    """Whitespace-insensitive stdout compare (trailing spaces/newlines are not semantics)."""
    return "\n".join(line.rstrip() for line in (s or "").splitlines()).strip()


SPAWN_RETRIES = int(os.environ.get("SFT_VERIFY_SPAWN_RETRIES", "5"))


def _run_one(code, stdin):
    """One isolated run, retrying only the host's retryable spawn throttle.

    On a box already running pretraining + CPU evals, the sandbox's inner setpriv/unshare clone
    intermittently returns rc=126 'Resource temporarily unavailable' (EAGAIN) even for
    `print(1)` -- that is system fork pressure, not the candidate solution. Retrying it with
    backoff separates "the box is busy" from "this code is wrong". A non-spawn failure (rc!=0,
    timeout, output mismatch) is returned for the caller to judge and is never retried."""
    from algorithms.isolate import run  # noqa: I001 (local import keeps import cost inside verify)
    for attempt in range(SPAWN_RETRIES):
        r = run(code, timeout=RUN_TIMEOUT, cpu_s=RUN_TIMEOUT, mem_mb=RUN_MEM_MB,
                stdin_data=stdin, nproc=RUN_NPROC)
        err = r.get("stderr", "") or ""
        spawn_failed = (r.get("rc") == 126
                        and ("Resource temporarily unavailable" in err
                             or "setpriv" in err or "failed to execute" in err))
        if not spawn_failed:
            return r
        time.sleep(min(8.0, 0.5 * 2 ** attempt))
    return r  # last spawn failure after retries exhausted


def verify_solution(code, cases, max_cases):
    """Run one solution over the first max_cases; return True iff every one matches."""
    from algorithms.isolate import Unisolated
    for stdin, want in cases[:max_cases]:
        try:
            r = _run_one(code, stdin)
        except Unisolated:
            raise
        except Exception:
            return False
        if r.get("timed_out") or r.get("rc") != 0:
            return False
        if normalize(r.get("stdout", "")) != normalize(want):
            return False
    return True


def verify_problem(problem, max_cases, max_solutions, keep_solutions):
    """Return up to keep_solutions DISTINCT verified solution strings, or (None, reason).

    Keeping several different passing solutions (1e 2026-09-27, max 3) gives the SFT set style
    diversity for one problem without letting a problem with dozens of near-identical accepted
    answers flood the pack. Distinct is by normalized source text."""
    cases = parse_io(problem["io"])
    if cases is None:
        return None, "no_stdin_io"
    if problem.get("starter"):
        return None, "function_style"
    ins, outs = cases
    pairs = list(zip(ins, outs, strict=True))
    kept, seen_norm = [], set()
    for code in parse_solutions(problem["solutions"], max_solutions):
        if len(kept) >= keep_solutions:
            break
        key = "".join(code.split())
        if key in seen_norm:
            continue
        if verify_solution(code, pairs, max_cases):
            seen_norm.add(key)
            kept.append(code)
    if kept:
        return kept, "verified"
    return None, "no_passing_solution"


def load_taco(d):
    import pyarrow.parquet as pq
    for p in sorted(glob.glob(os.path.join(d, "*.parquet"))):
        t = pq.read_table(p, columns=["question", "solutions", "input_output",
                                      "starter_code", "url", "source", "difficulty"])
        for i in range(t.num_rows):
            row = t.slice(i, 1).to_pylist()[0]
            yield {"qid": f"taco:{i}:{os.path.basename(p)}", "question": row["question"],
                   "solutions": row["solutions"], "io": row["input_output"],
                   "starter": row["starter_code"], "url": row.get("url"),
                   "source": row.get("source"), "difficulty": row.get("difficulty"),
                   "_file": os.path.basename(p)}


def load_apps(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            yield {"qid": f"apps:{d.get('id')}", "question": d.get("question", ""),
                   "solutions": d.get("solutions"), "io": d.get("input_output"),
                   "starter": d.get("starter_code", ""), "url": d.get("url"),
                   "source": "apps", "difficulty": d.get("difficulty"), "_file": "apps"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--taco", default="data/sft_raw/taco")
    ap.add_argument("--apps", default="data/sft_raw/apps/train.jsonl")
    ap.add_argument("--out", default="data/sft_raw/verified")
    ap.add_argument("--max-cases", type=int, default=8)
    ap.add_argument("--max-solutions", type=int, default=6)
    ap.add_argument("--keep-solutions", type=int, default=3,
                    help="max DISTINCT passing solutions kept per problem (1e 2026-09-27)")
    ap.add_argument("--limit", type=int, default=0, help="cap problems per source (0=all)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()

    from algorithms.isolate import detect_level
    level = detect_level()
    print(f"isolation level: {level}; max_cases {a.max_cases} max_solutions {a.max_solutions}",
          flush=True)
    os.makedirs(a.out, exist_ok=True)

    sources = [("taco", load_taco(a.taco)),
               ("apps", load_apps(a.apps))]
    stats = {}
    for name, it in sources:
        outp = os.path.join(a.out, f"{name}_verified.jsonl")
        seen = 0
        st = {"verified_problems": 0, "verified_solutions": 0, "no_stdin_io": 0,
              "function_style": 0, "no_passing_solution": 0}
        t0 = time.time()
        with open(outp, "w", encoding="utf-8") as out:
            for prob in it:
                seen += 1
                if a.limit and seen > a.limit:
                    break
                codes, why = verify_problem(prob, a.max_cases, a.max_solutions,
                                            a.keep_solutions)
                st[why] = st.get(why, 0) + 1
                if codes is None:
                    continue
                # one record per DISTINCT passing solution; 'verified' counted problems above,
                # solutions counted separately so both numbers are visible.
                st["verified_solutions"] += len(codes)
                for si, code in enumerate(codes):
                    rec = {"qid": f"{prob['qid']}#s{si}", "problem_qid": prob["qid"],
                           "solution_index": si, "source": name,
                           "problem_source": prob["source"],
                           "difficulty": prob.get("difficulty"), "url": prob.get("url"),
                           "question": prob["question"],
                           "starter_code": prob.get("starter", ""), "solution": code}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if st["verified_solutions"] % 100 < a.keep_solutions:
                    print(f"{name}: {seen} seen, {st.get('verified', 0)} problems, "
                          f"{st['verified_solutions']} solutions "
                          f"({(time.time()-t0)/seen:.2f}s/prob)", flush=True)
        st["seen"] = seen
        st["elapsed_s"] = round(time.time() - t0, 1)
        stats[name] = st
        print(f"{name} DONE: {json.dumps(st)}", flush=True)
    stats["run"] = {"timeout_s": RUN_TIMEOUT, "mem_mb": RUN_MEM_MB, "nproc": RUN_NPROC,
                    "max_cases": a.max_cases, "max_solutions": a.max_solutions,
                    "keep_solutions": a.keep_solutions, "isolation": level}
    with open(os.path.join(a.out, "verify_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print("verify done ->", a.out, flush=True)
    return 0


def _selftest():
    # Known answers for the pure parsing/compare logic; sandbox execution itself is exercised by
    # algorithms/isolate.py's own selftest and by a tiny live run (--limit 8) before the full job.
    good = parse_io(json.dumps({"inputs": ["2\n", "3\n"], "outputs": ["4\n", "9\n"]}))
    assert good is not None and len(good[0]) == 2
    assert parse_io(json.dumps({"inputs": ["1"], "outputs": []})) is None, "empty outputs reject"
    assert parse_io(json.dumps({"inputs": ["1", "2"], "outputs": ["x"]})) is None, "length mismatch"
    assert parse_io("not json") is None and parse_io(None) is None
    sols = parse_solutions(json.dumps(["x=int(input())\nprint(x*2)", "no input read"]), 5)
    assert len(sols) == 1 and "input(" in sols[0]
    assert normalize("  4 \n\n") == "4"
    assert normalize("a\nb \n") == "a\nb"
    # function-style problems are skipped before running
    p = {"io": json.dumps({"inputs": ["1"], "outputs": ["1"]}),
         "starter": "def f():\n", "solutions": json.dumps(["print(1)"])}
    codes, why = verify_problem(p, 8, 6, 3)
    assert codes is None and why == "function_style"
    # distinct-solution cap (the sandbox itself is proven by --limit live runs, not here): stub
    # verify_solution to "everything passes", then dedupe collapses the identical pair and the
    # cap stops at 3.
    real_vs = globals()["verify_solution"]
    try:
        globals()["verify_solution"] = lambda code, pairs, max_cases: True
        p2 = {"io": json.dumps({"inputs": ["1"], "outputs": ["1"]}), "starter": "",
              "solutions": json.dumps(["a=input()\nx=1", "a=input()\nx=1",
                                        "b=input()\ny=2", "c=input()\nz=3", "d=input()\nw=4"])}
        got, w2 = verify_problem(p2, 8, 6, 3)
        assert w2 == "verified" and [g.split("\n")[1] for g in got] == ["x=1", "y=2", "z=3"], got
    finally:
        globals()["verify_solution"] = real_vs
    print("sft_verify_code selftest OK: io/solution parse, length+fn gates, stdout normalize, "
          "distinct-solution cap")


if __name__ == "__main__":
    raise SystemExit(main())

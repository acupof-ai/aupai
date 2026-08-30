#!/usr/bin/env python3
"""Batch sandbox runner for the Fable code-with-tests lambda probe (t05, 3b's).

Input JSONL, one sample per line:
  {"q": str, "code": str, "tests": [{"in": str, "out": str}], "tier": str?}
A sample passes iff its code passes ALL its tests (stdin->stdout, blank-insensitive
line normalization). Samples with zero tests are useless corpus and fail.

Every question is scanned against the eval holdout (scripts/holdout.is_holdout)
-- generated contamination is active recall, so the scan is per-sample, never
sampled. is_holdout raises on a stale/missing hash set (fail-closed), which kills
the batch loudly: a probe that cannot prove cleanliness must not produce a lambda.

Output: per-sample JSONL {id, tier, pass, n_tests, n_fail, fails, is_holdout}
to --out; summary (per-tier lambda, merged lambda, holdout hits) to stdout.

Pod-only (sandbox_exec needs root + unshare).
  python3 scripts/sandbox_batch.py probe.jsonl --out probe_results.jsonl
  python3 scripts/sandbox_batch.py --selfcheck
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.holdout import is_holdout  # noqa: E402
from scripts.sandbox_exec import run_sandboxed  # noqa: E402


def _norm_lines(s):
    return [ln.rstrip() for ln in s.split("\n") if ln.strip() != ""]


def run_sample(sample, timeout):
    tests = sample.get("tests") or []
    fails = []
    for t in tests:
        rc, out, _ = run_sandboxed(sample["code"], timeout=timeout,
                                   stdin=t.get("in", ""))
        if rc != 0 or _norm_lines(out) != _norm_lines(t["out"]):
            fails.append(t.get("in", "")[:60])
    return {
        "id": sample.get("id"),
        "tier": sample.get("tier", "?"),
        "pass": bool(tests) and not fails,
        "n_tests": len(tests),
        "n_fail": len(fails),
        "fails": fails[:3],
        "is_holdout": is_holdout(sample["q"]),
    }


def _selfcheck():
    samples = [
        {"id": "pass", "q": "读入一个整数输出它的两倍",
         "code": "import sys\nprint(int(sys.stdin.read()) * 2)",
         "tests": [{"in": "21", "out": "42"}], "tier": "easy"},
        {"id": "wrong_answer", "q": "输出 2", "code": "print(1)",
         "tests": [{"in": "", "out": "2"}], "tier": "easy"},
        {"id": "timeout", "q": "死循环", "code": "while True:\n    pass",
         "tests": [{"in": "", "out": ""}], "tier": "hard"},
        {"id": "no_tests", "q": "没有测试的样本", "code": "print(1)",
         "tests": [], "tier": "easy"},
    ]
    hp = os.path.join(ROOT, "data", "eval", "code_holdout_500.jsonl")
    if os.path.exists(hp):
        hq = json.loads(open(hp, encoding="utf-8").readline())
        samples.append({"id": "holdout", "q": hq["instruction"], "code": "print(1)",
                        "tests": [{"in": "", "out": "1"}], "tier": "easy"})
    expect = {"pass": True, "wrong_answer": False, "timeout": False,
              "no_tests": False, "holdout": None}
    fails = 0
    for s in samples:
        r = run_sample(s, timeout=5)
        ok = True
        if expect[s["id"]] is not None:
            ok = r["pass"] is expect[s["id"]]
        if s["id"] == "holdout":
            ok = r["is_holdout"]
        if not ok:
            fails += 1
        print(f"  {'OK ' if ok else 'FAIL'} {s['id']}: pass={r['pass']} "
              f"holdout={r['is_holdout']} n_tests={r['n_tests']}")
    print(f"self-check: {len(samples) - fails}/{len(samples)} pass")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", help="JSONL of {q, code, tests:[{in,out}], tier?}")
    ap.add_argument("--out", help="per-sample results JSONL")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        sys.exit(1 if _selfcheck() else 0)
    if not args.input or not args.out:
        ap.error("input and --out required (unless --selfcheck)")

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(lambda s: run_sample(s, args.timeout), rows)):
            results.append(r)
            if (i + 1) % 100 == 0 or i + 1 == len(rows):
                print(f"  {i + 1}/{len(rows)}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(results)
    passed = sum(r["pass"] for r in results)
    no_tests = sum(1 for r in results if r["n_tests"] == 0)
    holdout_hits = sum(r["is_holdout"] for r in results)
    print(f"merged lambda: {passed}/{total} = {passed / total:.1%}")
    tiers = {}
    for r in results:
        p, n = tiers.setdefault(r["tier"], [0, 0])
        tiers[r["tier"]] = [p + r["pass"], n + 1]
    for tier, (p, n) in sorted(tiers.items()):
        print(f"  {tier}: {p}/{n} = {p / n:.1%}")
    if no_tests:
        print(f"no-tests (failed by rule): {no_tests}/{total}")
    print(f"holdout hits: {holdout_hits}/{total}")
    if holdout_hits:
        print("  CONTAMINATION: holdout questions in this batch -- route, do not train")


if __name__ == "__main__":
    main()

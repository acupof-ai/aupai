#!/usr/bin/env python3
"""Local validator for Fable lambda-probe problems (t05).

Runs each sample's stdin->stdout code against its own tests locally, mirroring
sandbox_batch.run_sample's blank-insensitive line normalization. This is the
correctness gate BEFORE handing a batch to 44's sandbox: a candidate whose code
does not pass its own tests is junk (the batch runner would mark it fail), and
sending it would silently lower lambda. Local validation of self-authored Fable
code is fine -- the sandbox is for untrusted code at training time, not for
trusting the generator's own candidates.

Usage: python3 scripts/validate_lambda_probe.py probe.jsonl
Exits nonzero if any sample fails its own tests (or has no tests).
"""
import json
import subprocess
import sys


def norm_lines(s):
    return [ln.rstrip() for ln in s.split("\n") if ln.strip() != ""]


def validate(sample):
    tests = sample.get("tests") or []
    fails = []
    for t in tests:
        try:
            r = subprocess.run(
                ["python3", "-c", sample["code"]],
                input=t.get("in", ""), capture_output=True, text=True, timeout=5,
            )
            ok = (r.returncode == 0) and norm_lines(r.stdout) == norm_lines(t["out"])
        except subprocess.TimeoutExpired:
            ok = False
        if not ok:
            fails.append(t.get("in", "")[:50])
    return {"pass": bool(tests) and not fails, "n_tests": len(tests), "fails": fails}


def main():
    path = sys.argv[1]
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    bad = 0
    for s in rows:
        r = validate(s)
        flag = "OK  " if r["pass"] else "FAIL"
        if not r["pass"]:
            bad += 1
        print(f"{flag} {s.get('id','?'):14s} tier={s.get('tier','?'):8s} "
              f"tests={r['n_tests']} fails={r['fails'][:2]}")
    print(f"\n{len(rows) - bad}/{len(rows)} pass their own tests; {bad} fail -> "
          f"{'EXIT 1 (fix before handing to 44)' if bad else 'ready to hand to 44'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
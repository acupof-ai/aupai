#!/usr/bin/env python3
"""Consume the fable-pilot workflow output -> validated, assembled probe JSONL.

Workflow returns [{tier, idx, json:<problem-json-string>}]. Fable HTML-escapes
< and > in code (observed: `&gt;` broke a hard-problem solution), so we unescape
&gt;/&lt;/&amp; in code + tests, then run each code against its own tests via
validate_lambda_probe, and split into passing (assembly) vs failing.

Usage: python3 scripts/assemble_lambda_probe.py workflow_out.json --out data/sft/lambda_probe_pilot40.jsonl
      (--out writes only samples whose code passes its own tests; failing ones print)
"""
import argparse
import html
import json
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.validate_lambda_probe import validate  # noqa: E402


def unescape_problem(obj):
    for f in ("code", "q"):
        if isinstance(obj.get(f), str):
            obj[f] = html.unescape(obj[f])
    for t in obj.get("tests") or []:
        for f in ("in", "out"):
            if isinstance(t.get(f), str):
                t[f] = html.unescape(t[f])
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow_out")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.workflow_out, encoding="utf-8") if l.strip()]
    ok, bad = [], []
    for i, entry in enumerate(rows):
        try:
            obj = json.loads(entry["json"]) if isinstance(entry.get("json"), str) else entry
        except Exception as e:
            bad.append({"id": f"t{entry.get('tier')}_{entry.get('idx')}", "err": f"parse: {e}"})
            continue
        obj = unescape_problem(obj)
        obj["id"] = f"{entry.get('tier')}_{entry.get('idx')}"
        obj["tier"] = obj.get("tier") or entry.get("tier")
        v = validate(obj)
        rec = {"id": obj["id"], "tier": obj["tier"], "pass": v["pass"],
               "n_tests": v["n_tests"], "fails": v["fails"]}
        if v["pass"]:
            ok.append(obj)
            rec["status"] = "pass"
        else:
            rec["status"] = "fail"
            bad.append(rec)
        print(f"  {rec['status']:4s} {rec['id']:14s} tier={rec['tier']:9s} tests={v['n_tests']} fails={v['fails'][:2]}")

    from collections import Counter
    print(f"\npass {len(ok)} / fail {len(bad)}")
    print("pass by tier:", Counter(o["tier"] for o in ok))
    if a.out and ok:
        with open(a.out, "w", encoding="utf-8") as f:
            for o in ok:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        print(f"wrote {len(ok)} passing samples -> {a.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
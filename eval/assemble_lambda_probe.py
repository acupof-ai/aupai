#!/usr/bin/env python3
"""Consume the fable-pilot workflow output -> validated, assembled probe JSONL.

Workflow returns [{tier, idx, json:<problem-json-string>}]. Fable HTML-escapes
< and > in code (observed: `&gt;` broke a hard-problem solution), so we unescape
&gt;/&lt;/&amp; in code + tests, then run each code against its own tests via
validate_lambda_probe, and split into passing (assembly) vs failing.

# restartable: passes are appended to --out one sample at a time; on resume the
# script skips anything already in --out and re-processes only the remainder, so
# an interrupt loses at most the in-flight validations (a cheap replay of local
# execution), never the already-emitted output. The expensive part (Fable agent
# calls) happened upstream and is never re-run here.

Usage: python3 scripts/assemble_lambda_probe.py workflow_out.json --out data/sft/lambda_probe_pilot40.jsonl
      (--out keeps only samples whose code passes its own tests; failing ones print)
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
    # resume: skip ids already emitted to --out (append-per-pass, per restartable)
    existing = set()
    if a.out and os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                existing.add(json.loads(l)["id"])
            except Exception:
                continue
    ok, bad, skipped = [], [], []
    for i, entry in enumerate(rows):
        obj_id = f"{entry.get('tier')}_{entry.get('idx')}"
        if obj_id in existing:
            skipped.append(obj_id)
            continue
        try:
            obj = json.loads(entry["json"]) if isinstance(entry.get("json"), str) else entry
        except Exception as e:
            bad.append({"id": obj_id, "err": f"parse: {e}"})
            continue
        obj = unescape_problem(obj)
        obj["id"] = obj_id
        obj["tier"] = obj.get("tier") or entry.get("tier")
        v = validate(obj)
        rec = {"id": obj["id"], "tier": obj["tier"], "pass": v["pass"],
               "n_tests": v["n_tests"], "fails": v["fails"]}
        if v["pass"]:
            ok.append(obj)
            rec["status"] = "pass"
            if a.out:  # append per passing sample: an interrupt keeps everything emitted
                with open(a.out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        else:
            rec["status"] = "fail"
            bad.append(rec)
        print(f"  {rec['status']:4s} {rec['id']:14s} tier={rec['tier']:9s} tests={v['n_tests']} fails={v['fails'][:2]}")

    from collections import Counter
    print(f"\npass {len(ok)} / fail {len(bad)}" + (f" / skipped {len(skipped)}" if skipped else ""))
    print("pass by tier:", Counter(o["tier"] for o in ok) if ok else "(resume: none new)")
    if a.out:
        n_line = sum(1 for _ in open(a.out, encoding="utf-8")) if os.path.exists(a.out) else 0
        print(f"emitted {len(ok)} new passing samples -> {a.out} now holds {n_line}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
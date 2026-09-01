#!/usr/bin/env python3
"""cited_artifacts_attested: seven citation shapes, four of which must FAIL.

The check was widened twice on 2026-09-01 -- artifact_sha256 accepts a {basename: sha}
object so one fact can cite several artifacts, and config.unattested_leg exempts a leg
written before eval_artifacts.attest existed. Both widenings are escape hatches, and an
escape hatch that is not tested is a hole: the exemption is a substring match on a
config field, so the case that matters is a fact declaring the exemption for a DIFFERENT
file than the one it cites, which must still FAIL.

Splitting one measurement across three facts so a single-string field would fit was the
alternative to the dict form; that shapes the record around the guard.

    python3 scripts/test_attest_cases.py
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import harness

ROOT = harness.ROOT


def world():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "facts"))
    os.makedirs(os.path.join(d, "runs"))
    shutil.copy(os.path.join(ROOT, "runs", "artifact_refs.jsonl"),
                os.path.join(d, "runs", "artifact_refs.jsonl"))
    return d


def put(d, fact):
    with open(os.path.join(d, "facts", "x.json"), "w", encoding="utf-8") as f:
        json.dump({"facts": [fact]}, f, ensure_ascii=False)


BASE = {"id": "t.x", "measured": "2026-09-02", "status": "measured",
        "config": {"a": 1}, "uncertainty": "-", "source": "-"}
GOOD = "2b876e603ea64dcf8ca56a883e6724499fad1e5edfc2ed4e08fcc1882905f580"
F = "data/eval/preds_l1_d3.fewshot_24k.jsonl"

cases = [
    ("no hash at all", dict(BASE, value=f"cites {F}"), harness.FAIL),
    ("wrong hash", dict(BASE, value=f"cites {F}", artifact_sha256="deadbeef" * 8), harness.FAIL),
    ("right hash, string form", dict(BASE, value=f"cites {F}", artifact_sha256=GOOD), harness.PASS),
    ("right hash, dict form", dict(BASE, value=f"cites {F}",
                                   artifact_sha256={os.path.basename(F): GOOD}), harness.PASS),
    ("dict names the WRONG file", dict(BASE, value=f"cites {F}",
                                       artifact_sha256={"other.jsonl": GOOD}), harness.FAIL),
    ("exemption declared for THIS file", dict(BASE, value=f"cites {F}",
                                              config={"a": 1, "unattested_leg": os.path.basename(F)}),
     harness.PASS),
    ("exemption declared for ANOTHER file", dict(BASE, value=f"cites {F}",
                                                 config={"a": 1, "unattested_leg": "unrelated.jsonl"}),
     harness.FAIL),
]

bad = 0
for name, fact, want in cases:
    d = world()
    put(d, fact)
    got, ev = harness.check_cited_artifacts_attested(d)
    ok = got == want
    bad += not ok
    print(f"  {name:36s} {got:4s} want {want:4s} {'' if ok else '<-- WRONG'}")
print(f"\n{bad} wrong")
sys.exit(1 if bad else 0)

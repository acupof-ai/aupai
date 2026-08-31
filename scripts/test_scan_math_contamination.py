#!/usr/bin/env python3
"""Broken-world tests for scripts/scan_math_contamination.py.

Rule: broken worlds MUTATE REAL ARTIFACTS — copy a real shard, rename fields, inject
empties — never hand-write fixture rows. The field-fall-through defect recurred twice
because its test was a hand-written toy that bypassed the real schema.

Run from the repo root: python scripts/test_scan_math_contamination.py
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN = os.path.join(HERE, "scan_math_contamination.py")
REAL_SHARD = sorted(glob.glob("data/corpus/math/ape210k_000.jsonl"))[0]
CLEAN_GLOB = "data/corpus/sample/batch_*.jsonl"  # web prose, measured clean (max containment 0.241)
HOLDOUT = json.loads(open("data/eval/math_test_500.jsonl", encoding="utf-8").readline())["instruction"]

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        print(f"  FAIL {name} {detail}")


def run_scan(path, *extra):
    r = subprocess.run([sys.executable, SCAN, path, *extra],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def mutate_field_rename(src, dst):
    """Real shard, field renamed: content -> text. The fall-through defect."""
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            d = json.loads(line)
            d["text"] = d.pop("content")
            g.write(json.dumps(d, ensure_ascii=False) + "\n")


def mutate_empty_fields(src, dst, frac=0.02):
    """Real shard with >1% empty question fields."""
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for i, line in enumerate(f):
            d = json.loads(line)
            if i % int(1 / frac) == 0:
                d["content"] = ""
            g.write(json.dumps(d, ensure_ascii=False) + "\n")


def mutate_embed_holdout(src, dst):
    """Real clean doc with one real holdout question embedded verbatim."""
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for i, line in enumerate(f):
            d = json.loads(line)
            if i == 0:
                field = "content" if "content" in d else "text"
                d[field] = HOLDOUT + " " + str(d[field])
            g.write(json.dumps(d, ensure_ascii=False) + "\n")


def main():
    tmp = tempfile.mkdtemp(prefix="scan_bw_")
    try:
        print("broken world 1: real shard with renamed field (content -> text)")
        p1 = os.path.join(tmp, "renamed.jsonl")
        mutate_field_rename(REAL_SHARD, p1)
        rc, out = run_scan(p1, "--q-field", "content")
        check("renamed field refuses (no false clean)", rc != 0 and "REFUSED" in out,
              f"rc={rc}")

        print("broken world 2: real shard with 2% empty question fields")
        p2 = os.path.join(tmp, "empty.jsonl")
        mutate_empty_fields(REAL_SHARD, p2)
        rc, out = run_scan(p2)
        check("empty fields refuse (no false clean)", rc != 0 and "REFUSED" in out,
              f"rc={rc}")

        print("known-answer pair: verbatim embed vs the same clean docs unmodified")
        clean_shard = sorted(glob.glob(CLEAN_GLOB))[0]
        p3 = os.path.join(tmp, "embed.jsonl")
        mutate_embed_holdout(clean_shard, p3)
        rc, out = run_scan(p3, "--full-doc")
        check("verbatim embed is REJECT", rc == 1 and "REJECT" in out, f"rc={rc}")
        rc0, out0 = run_scan(clean_shard, "--full-doc")
        check("unmodified clean docs are clean at 0.8", rc0 == 0, f"rc={rc0}")
        check("pair differs: embed REJECT vs clean clean", rc == 1 and rc0 == 0)

        print("threshold sensitivity: same shard at 0.7 / 0.8 / 0.9")
        counts = {}
        for t in ("0.7", "0.8", "0.9"):
            _, out = run_scan(REAL_SHARD, "--threshold", t)
            line = next(l for l in out.splitlines() if "holdouts hit at 0.7 / 0.8 / 0.9" in l)
            counts[t] = line
            print(f"  @{t}: {line.split(':', 1)[1].strip()}")
        check("distribution reported at all three thresholds", all(counts.values()))
    finally:
        shutil.rmtree(tmp)
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

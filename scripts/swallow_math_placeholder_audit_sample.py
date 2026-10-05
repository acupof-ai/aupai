#!/usr/bin/env python3
"""Two-sided hand-read sample for the LLM placeholder filter (de discipline 2026-10-05).

100 rule-hit docs (every hit until 100, spread over shards) and 100 kept docs on a
fixed stride from a seeded position, for two-sided human labeling. Read-only.

# restartable: pure sampling scan, writes only the two small /tmp-style output files,
# never the corpus; an interrupt just re-runs.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, "/work/aupai/scripts")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from filter_llm_placeholder import is_placeholder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--domain", default="swallow_math")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--drop_out", default="/tmp/ph_drop100.jsonl")
    ap.add_argument("--keep_out", default="/tmp/ph_keep100.jsonl")
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.out, f"{a.domain}_*.jsonl")))
    drops, keeps = [], []
    gi = 0
    GAP = 7919  # fixed prime stride over KEPT docs, samples whole corpus uniformly
    for p in paths:
        if len(drops) >= a.n and len(keeps) >= a.n:
            break
        with open(p, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                c = json.loads(s).get("content") or ""
                if is_placeholder(c):
                    if len(drops) < a.n:
                        drops.append(c[:500])
                else:
                    if len(keeps) < a.n and gi % GAP == 0:
                        keeps.append(c[:500])
                    gi += 1
    with open(a.drop_out, "w", encoding="utf-8") as f:
        for c in drops[: a.n]:
            f.write(json.dumps({"content": c}, ensure_ascii=False) + "\n")
    with open(a.keep_out, "w", encoding="utf-8") as f:
        for c in keeps[: a.n]:
            f.write(json.dumps({"content": c}, ensure_ascii=False) + "\n")
    print(f"dropped_sample={len(drops[: a.n])} kept_sample={len(keeps[: a.n])}")
    print("DONE")


if __name__ == "__main__":
    main()

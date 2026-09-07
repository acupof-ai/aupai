#!/usr/bin/env python3
"""Restamp cot_open_thoughts: tokens_kept was the per-row tokens sum (pre-<eos>
convention); runs/recount_cot_ot.sh measured 776084377 under the train.py encode
convention (ids + one <eos> per document). Refuses unless the stamp still reads
the known stale value, so a changed world is never stamped over.

Run on the pod: python3 runs/restamp_cot_ot.py
"""
import json
import os
import sys

STAMP = "data/corpus/cot_open_thoughts/build_corpus_stats.json"
OLD = 775972331   # per-row tokens sum (pre-<eos>); what the stamp read
NEW = 776084377   # count_docs convention; measured by runs/recount_cot_ot.sh (RC3 DONE)


def main():
    with open(STAMP, encoding="utf-8") as f:
        s = json.load(f)
    if s.get("tokens_kept") != OLD:
        sys.exit(f"refuse: stamp tokens_kept={s.get('tokens_kept')} != {OLD}; "
                 "the world changed since the recount -- re-measure, do not stamp")
    s["tokens_superseded"] = {
        "value": OLD,
        "reason": "per-row tokens sum (pre-<eos>); superseded by the train.py encode convention",
        "restamped": "2026-09-08",
        "by": "runs/restamp_cot_ot.py",
        "measured_by": "runs/recount_cot_ot.sh (RC3 DONE: 776084377, per_row_sum 775972331)",
    }
    s["tokens_kept"] = NEW
    tmp = STAMP + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STAMP)
    print(json.dumps({"restamped": True, "tokens_kept": NEW,
                      "tokens_superseded": OLD}, ensure_ascii=False))


if __name__ == "__main__":
    main()

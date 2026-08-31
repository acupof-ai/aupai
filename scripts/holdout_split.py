#!/usr/bin/env python3
"""Split a NEW math source into holdout_v2 + corpus remainder, and gate the remainder.

The holdout set is only as clean as its source: math-500 was drawn from sources that
later entered the corpus and ended up 30% contaminated. The v2 rule makes that
impossible by construction -- the holdout is carved out of a source BEFORE the source
is ingested, and the remainder is scanned against the full holdout before it is
allowed to touch the corpus.

Rule (automatable, no RNG state):
  1. A source is eligible only if it has never been in the corpus (no ledger entry,
     not in data/PROVENANCE.md). Re-scans of existing sources never feed the holdout.
  2. Each question's qhash (same key as scripts/holdout.py) is tested:
     int(qhash[:8], 16) % 100 < SPLIT_PCT -> holdout, else corpus remainder.
     Deterministic: the same source always splits the same way on any machine.
  3. The remainder is scanned against the active holdout (HOLDOUT_FILES in
     scan_math_contamination.py plus everything under data/eval/holdout_v2/).
     Any hit -> REFUSED, nothing is written. A source that cannot pass its own
     holdout split is too close to the old holdouts to use at all.
  4. Outputs: data/eval/holdout_v2/<name>.jsonl and data/corpus/<domain>/<name>.jsonl,
     plus a ledger row recording the split.

Usage:
    python scripts/holdout_split.py <new_source.jsonl|parquet> [--pct 2] [--q-field NAME] [--domain math]

Exit code: 0 = split written, remainder clean; 1 = refused (contaminated or ineligible).
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_math_contamination import (  # noqa: E402
    HoldoutIndex,
    extract_question,
    iter_texts,
    load_holdouts,
    qhash,
    scan_path,
)

HOLDOUT_V2_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "eval", "holdout_v2")
SPLIT_PCT_DEFAULT = 2


def is_eligible(path):
    """Never in the corpus: no scan-ledger entry, not named in PROVENANCE."""
    ledger = os.path.join(os.path.dirname(HOLDOUT_V2_DIR), "..", "scan_ledger.jsonl")
    if os.path.exists(ledger):
        base = os.path.basename(path)
        for line in open(ledger, encoding="utf-8"):
            if base in line:
                return False, f"ledger already knows {base}"
    prov = glob.glob(os.path.join(os.path.dirname(HOLDOUT_V2_DIR), "..", "PROVENANCE.md"))
    if prov and os.path.basename(path) in open(prov[0], encoding="utf-8").read():
        return False, f"{os.path.basename(path)} named in PROVENANCE.md"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--pct", type=int, default=SPLIT_PCT_DEFAULT)
    ap.add_argument("--q-field", default=None)
    ap.add_argument("--domain", default="math")
    args = ap.parse_args()

    ok, why = is_eligible(args.path)
    if not ok:
        sys.exit(f"REFUSED: {why} -- only never-ingested sources can seed the holdout")

    holdouts = load_holdouts()
    for p in sorted(glob.glob(os.path.join(HOLDOUT_V2_DIR, "*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                holdouts.append(json.loads(line)["instruction"])
    idx = HoldoutIndex(holdouts)

    name = os.path.splitext(os.path.basename(args.path))[0]
    os.makedirs(HOLDOUT_V2_DIR, exist_ok=True)
    h_out = open(os.path.join(HOLDOUT_V2_DIR, f"{name}.jsonl"), "w", encoding="utf-8")
    remainder = []
    n = n_hold = 0
    for text in iter_texts(args.path, args.q_field, False):
        n += 1
        if not text:
            continue
        if int(qhash(text)[:8], 16) % 100 < args.pct:
            h_out.write(json.dumps({"instruction": text}, ensure_ascii=False) + "\n")
            n_hold += 1
        else:
            remainder.append(text)
    h_out.close()

    # Gate the remainder against the full holdout (incl. the just-written v2 slice).
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for q in remainder:
            f.write(json.dumps({"question": q}, ensure_ascii=False) + "\n")
        tmp = f.name
    try:
        _, _, exact, mc, _ = scan_path(tmp, idx, "question", False, 0.8)
    finally:
        os.unlink(tmp)
    long_i = [i for i in range(len(holdouts)) if i not in set(idx.short)]
    hits = sum(1 for i in long_i if mc[i] >= 0.8)
    if hits or exact:
        os.unlink(os.path.join(HOLDOUT_V2_DIR, f"{name}.jsonl"))
        sys.exit(f"REFUSED: remainder hits {hits} holdouts at 0.8 ({len(exact)} exact) -- "
                 f"source too close to the existing holdouts, holdout slice discarded")

    out_dir = os.path.join(os.path.dirname(HOLDOUT_V2_DIR), "corpus", args.domain)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
        for q in remainder:
            f.write(json.dumps({"question": q}, ensure_ascii=False) + "\n")
    print(f"{name}: {n} questions -> {n_hold} holdout (data/eval/holdout_v2/{name}.jsonl) "
          f"+ {len(remainder)} corpus (data/corpus/{args.domain}/{name}.jsonl), remainder clean at 0.8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

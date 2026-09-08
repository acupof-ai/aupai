"""Containment of the CURRENT holdout set in domains a decay reweight would raise.

The question is not "was the corpus filtered" -- it was -- but "against WHICH holdout set",
and whether raising a domain's decay weight concentrates whatever survived. Re-running the
live predicate over the shipped bytes answers both; a build-time reject histogram cannot,
because it records the set that was current when the shard was written.

THE CONTROL IS NOT OPTIONAL AND IT IS WHERE THIS SCRIPT WENT WRONG FIRST. `load()`
returns HASHES; feeding one back to `is_holdout` returns False, so a "positive control"
built that way reports the guard broken while it is fine -- a wrong answer shaped like a
finding. The control below takes a real QUESTION STRING out of a registry eval file, and
a negative control that must not match. A 0% containment reading is unreadable without
both.

    python3 scripts/holdout_containment.py chatml chat_qa cot textbook_30b
"""

import json
import os
import random
import sys

ROOT_REPO = "/work/aupai"
sys.path.insert(0, ROOT_REPO)
from datagen.holdout import REGISTRY, _fingerprint, is_holdout  # noqa: E402

ROOT = os.path.join(ROOT_REPO, "data", "corpus")
DOMS = sys.argv[1:] or ["chatml", "chat_qa", "cot", "textbook_30b"]
PER, SEED = 4000, 11


def _controls():
    """(positive, negative) -- a real holdout question must match, plain code must not."""
    pos = []
    for k, e in REGISTRY.items():
        p = os.path.join(ROOT_REPO, e["path"])
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                q = r.get("question") or r.get("problem") or r.get("prompt") or r.get("input")
                if q:
                    pos.append((k, is_holdout(q)))
                    break
        if len(pos) >= 2:
            break
    return pos, is_holdout("def f(x): return x+1")


pos, neg = _controls()
print(json.dumps({"positive_controls": pos, "negative_control": neg}))
if not pos or not all(v for _, v in pos) or neg:
    sys.exit("controls failed -- a containment number from this run would be unreadable")
rng = random.Random(SEED)
print(json.dumps({"holdout_fingerprint": _fingerprint(), "per_domain_sample": PER, "seed": SEED}))

for d in DOMS:
    p = os.path.join(ROOT, d)
    if not os.path.isdir(p):
        print(json.dumps({"domain": d, "error": "absent"}))
        continue
    fs = sorted(f for f in os.listdir(p) if f.endswith(".jsonl") and not f.startswith("holdout_slice"))
    if not fs:
        print(json.dumps({"domain": d, "error": "no shards"}))
        continue
    rows = []
    for f in rng.sample(fs, min(3, len(fs))):
        with open(os.path.join(p, f), encoding="utf-8") as fh:
            lines = fh.readlines()
        for x in rng.sample(lines, min(PER // 3 + 1, len(lines))):
            try:
                rows.append(json.loads(x).get("content", ""))
            except Exception:
                continue
    hits = 0
    for c in rows:
        if is_holdout(c):
            hits += 1
            continue
        # A whole document rarely hashes to a holdout item; the per-line pass is what
        # catches a held-out question quoted inside a longer document.
        for ln in (line.strip() for line in c.split("\n")):
            if ln and len(ln) <= 500 and is_holdout(ln):
                hits += 1
                break
    print(
        json.dumps(
            {
                "domain": d,
                "sampled": len(rows),
                "shards_read": min(3, len(fs)),
                "containment_hits": hits,
                "rate_pct": round(hits / len(rows) * 100, 4) if rows else None,
            }
        )
    )

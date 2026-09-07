#!/usr/bin/env python3
"""Re-stamp en_c4_30b: 645,640,484 -> 646,669,989, exactly one <eos> per document.

/tmp/count_30b.py was a full pass with no sampling and its arithmetic was right; what it
omitted was the terminator, so the delta is the document count to the unit. Two independent
readings agree on the new value and they do not share a counting loop: runs/count_dir.py
(32-worker, its own split("\\n") reader, count_tokens.count_docs) and scripts/count_tokens.py
count_shards over the same 23 shards, 646,669,989 both. That pair is what verifies the
CONVENTION -- a recount that only re-ran the old arithmetic would have agreed with the old
number and read as confirmation.

fp_dir skips build_corpus_stats.json (corpus_fingerprint.py:128), so rewriting the stamp
cannot move the fingerprint.
"""
import json
import os

P = "/work/aupai/data/corpus/en_c4_30b/build_corpus_stats.json"
NEW = 646669989
OLD = 645640484
SHA = "5955b3e0130a3fd5da0323dfc2ce1b98a6c60a2d"
COMMIT = "dbfac5dd"

s = json.load(open(P))
assert s["tokens"] == OLD, f"stamp already says {s['tokens']}, refusing"
assert s["kept"] == NEW - OLD, f"delta {NEW - OLD} is not the document count {s['kept']}"

s["tokens"] = NEW
s["kept_tokens"] = NEW
s["chars_per_token"] = round(s["kept_chars"] / NEW, 2)
s["tokens_config"] = (
    f"scripts/count_tokens.py@{SHA} (commit {COMMIT}) count_shards, no sampling, all 23 "
    f"shards, content field; ids + one <eos> per document (train.py encode). Confirmed by "
    f"runs/count_dir.py, an independent reader, same value."
)
s["tokens_superseded"] = {
    "value": OLD,
    "by": "/tmp/count_30b.py 2026-09-04",
    "cause": "one <eos> per document omitted; the delta is exactly the 1,029,505 documents",
    "measured": "2026-09-08",
}
tmp = P + ".tmp"
with open(tmp, "w") as f:
    json.dump(s, f, ensure_ascii=False, indent=1)
os.replace(tmp, P)
print(json.dumps({k: s[k] for k in ("tokens", "chars_per_token", "tokens_config", "tokens_superseded")}, indent=1))

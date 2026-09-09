#!/usr/bin/env python3
# Reconstruct <dom>.loc.json for near_overlap.py: (shard, row) per valid doc,
# in the same order sig_one produced signatures. Cheap (no hashing).
import glob, json, os

CK = "/work/aupai/runs/overlap_ck"
DOMAINS = [
    ("dd09", "/work/aupai/data/corpus/code_rp1t_dd09/*.jsonl"),
    ("b2v2", "/work/aupai/data/corpus/code_rp1t_b2v2_dd/*.jsonl"),
    ("dedup08", "/work/aupai/data/corpus/code_dedup08/*.jsonl"),
]

for name, pat in DOMAINS:
    lp = f"{CK}/{name}.loc.json"
    if os.path.exists(lp):
        print(f"{name}: loc exists, skip")
        continue
    locs = []
    for sp in sorted(glob.glob(pat)):
        for ln, line in enumerate(open(sp, encoding="utf-8")):
            if not line.strip():
                continue
            try:
                content = json.loads(line).get("content", "")
            except Exception:
                continue
            if not content:
                continue
            locs.append([sp, ln])
    json.dump(locs, open(lp, "w"))
    print(f"{name}: {len(locs)} locs -> {lp}")

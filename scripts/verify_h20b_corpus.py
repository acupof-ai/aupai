#!/usr/bin/env python3
"""Verify H20B corpus has expected shards and build metadata."""
import glob
import json
import os

ROOT = "/data00/aupai_work/aupai"
DOMAINS = [
    "code_ultra_l3_noexec_dc",
    "code_ultra_l2_dc",
    "math_owm_stage2_g4_dc",
    "math_cot2_dc",
    "en_fwe4_dc",
    "cosmo_en_dc",
    "en_c4_stage2_dc",
    "code_opencot_dc",
    "code_py_commitpackft_dd_dc",
]
ok = True
for domain in DOMAINS:
    path = os.path.join(ROOT, "data/corpus", domain)
    stats_path = os.path.join(path, "build_corpus_stats.json")
    shards = glob.glob(os.path.join(path, "*.jsonl"))
    bytes_total = sum(os.path.getsize(name) for name in shards)
    print(domain, "shards", len(shards), "bytes", bytes_total)
    if not shards:
        ok = False
        print("  FAIL: no shards")
    if os.path.exists(stats_path):
        with open(stats_path, encoding="utf-8") as handle:
            stats = json.load(handle)
        expected = stats.get("shards_total")
        print("  stats shards_total", expected)
        if expected is not None and expected != len(shards):
            ok = False
            print("  FAIL: shard count mismatch")
    else:
        ok = False
        print("  FAIL: missing build stats")
raise SystemExit(0 if ok else 1)

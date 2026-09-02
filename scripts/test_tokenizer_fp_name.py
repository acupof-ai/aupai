#!/usr/bin/env python3
"""selftest for 44-14 defect 3: the fingerprint the gold_bpb measurement records
is a whole-tokenizer-FILE sha256[:16], so its name must say "file". The old key
"tokenizer_fp" implied a vocab fingerprint; a reader comparing it against a
vocab_id-style value gets a silent mismatch. The probe that emitted it was
deleted in the 44-13 pass (44-13); the name guard lives on in the fact and the
artifact.

    python3 scripts/test_tokenizer_fp_name.py
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = ["runs/gold_bpb.json", "facts/base_eval.json"]

for rel in TARGETS:
    text = open(os.path.join(ROOT, rel), encoding="utf-8").read()
    assert '"tokenizer_file_fp"' in text, f"{rel}: missing tokenizer_file_fp"
    assert '"tokenizer_fp"' not in text, f"{rel}: still uses the misleading tokenizer_fp"

print("selftest OK: tokenizer fingerprint is named tokenizer_file_fp everywhere")

#!/usr/bin/env python3
"""Fetch chat raw sources -> data/{coig,alpaca_gpt4_zh}.jsonl.

Producers for the chat domain in scripts/build_domains.sh. Verifies against
data/PROVENANCE.md sha256 — a mismatch prints an error and keeps the existing
file (never silently swap data).

Reproduce: python scripts/fetch_chat_data.py
"""

import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCES = [
    (
        "alpaca_gpt4_zh.jsonl",
        "HuggingFaceH4/alpaca_gpt4_data_zh",
        None,
        "93819e69830d9eb050e58c342230f3e1986a2e3cd07c3d1a075abb9ddcb6251d",
        lambda row: {"instruction": row["instruction"].strip(), "output": row["output"].strip()},
    ),
    (
        "coig.jsonl",
        "BAAI/COIG",
        "instructions",
        "cdcac3f1d310c0dd8bb6cf5ee63a4b2a99d3386e098cead4985d7e962a8a10f6",
        None,
    ),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    from datasets import load_dataset

    for fname, repo, cfg, want, norm in SOURCES:
        path = os.path.join(ROOT, "data", fname)
        if os.path.exists(path):
            got = sha256(path)
            if got == want:
                print(f"{fname}: sha256 matches PROVENANCE, skip")
                continue
            print(
                f"{fname}: sha MISMATCH (want {want[:12]}, have {got[:12]}) — keeping existing file, NOT re-fetching"
            )
            continue
        try:
            ds = load_dataset(repo, cfg, split="train") if cfg else load_dataset(repo, split="train")
        except Exception as e:  # noqa: BLE001
            print(f"{fname}: FAILED {type(e).__name__}: {str(e)[:120]}")
            continue
        if norm is None:
            print(f"{fname}: no normalizer — dump raw rows, then diff against PROVENANCE before use")
            with open(path, "w", encoding="utf-8") as f:
                for row in ds:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            with open(path, "w", encoding="utf-8") as f:
                n = 0
                for row in ds:
                    d = norm(row)
                    if d and d["instruction"] and d["output"]:
                        f.write(json.dumps(d, ensure_ascii=False) + "\n")
                        n += 1
            print(f"{fname}: {n}/{len(ds)} rows kept from {repo} -> {path}")
    print("post-check: sha256 the new files against data/PROVENANCE.md before wiring into build_domains")


if __name__ == "__main__":
    sys.exit(main())

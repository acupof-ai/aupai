#!/usr/bin/env python3
"""3b 2026-09-24: build the SFT mix-A continuation pack (user-selected plan A).

Combines the four rendered sources, each already 13-gram gated and length
capped, into one split-encode raw-continuation pack via
datagen/prepare_sft.pack_and_save:

  code_if_clean.jsonl  function sig+docstring -> body (stop teaching)
  sc2_exec.jsonl       instruction -> exec-validated solution block (logic)
  apps_call.jsonl      APPS starter signature -> solution (logic)
  prose.jsonl          en_c4 document prefix -> tail (pretrain-shaped regulariser)

Targets (plan section 4, docs/standards/code_sft_plan_0924.md): supervised LOSS
tokens code_if 55 / sc2 30 / apps 5 / prose 10. Actual shares follow the measured
supply: sc2's 50,661-row source yields 4.09M in-length solution tokens (27.6%),
short of 30%; the pack records the real shares instead of forcing them.

The pack is raw continuation (no ChatML), split_encode=True so the eval
sequence is a token-by-token prefix. Fingerprints in the .pt:
  vocab_id (pack_and_save), packer_fp, sources_fp, holdout_fp, plus this
  builder's content hash and each input jsonl's sha256+row count, so the pack
  ties back to the exact inputs and scripts that produced it.
"""
import argparse
import hashlib
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(ROOT, "datagen"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from tokenizers import Tokenizer
from prepare_sft import pack_and_save

SEQ = 4096
SEED = 42
SOURCES = ["code_if_clean", "sc2_exec", "apps_call", "prose"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["prompt"], r["output"], r.get("source", os.path.basename(path))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="dir holding the four *_clean/*.jsonl")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    eos = tok.token_to_id("<eos>")
    assert eos is not None, "tokenizer has no <eos>"

    per_source = {}
    all_examples = []
    input_sha = {}
    for name in SOURCES:
        path = os.path.join(args.in_dir, name + ".jsonl")
        rows = read_jsonl(path)
        input_sha[name] = {"path": path, "sha256": sha256_file(path), "rows": len(rows)}
        # examples are (prompt, output); keep the source tag only in stats
        per_source[name] = len(rows)
        all_examples.extend([(p, o) for p, o, _ in rows])
        print(f"{name}: {len(rows)} pairs", flush=True)

    random.Random(SEED).shuffle(all_examples)

    source_paths = [os.path.join(args.in_dir, n + ".jsonl") for n in SOURCES]
    builder_fp = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    extra = {
        "plan": "docs/standards/code_sft_plan_0924.md mix A (user-selected 2026-09-24)",
        "per_source_pairs": per_source,
        "input_sha256": input_sha,
        "builder": os.path.basename(__file__),
        "builder_sha256": builder_fp,
        "seed": SEED,
        "format": "raw_continuation_split_encode",
        "body_gate_tokens": 256,
    }
    pack_and_save(all_examples, tok, eos, args.out, SEQ,
                  sources=[(p, "prompt", "output") for p in source_paths],
                  split_encode=True, extra_stats=extra)


if __name__ == "__main__":
    main()

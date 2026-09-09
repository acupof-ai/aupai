#!/usr/bin/env python3
"""Prepare the format-SFT pack: RAW (prompt, output) pairs, no ChatML.

Why raw: the eval interface is 0-shot raw continuation (the HumanEval prompt
field), and ChatML-wrapped SFT would teach the body after
<|im_start|>assistant -- a string the eval prompt never contains, so the 0-shot
number could not move by construction (prereg format_sft_humaneval_0909,
confirmed by 4c 2026-09-09). Everything else is prepare_sft_math's path: same
greedy packing, same holdout gate, same fingerprints.

Source: data/sft/code_if_pairs_train.jsonl (3b, 2026-09-09): 99,996 pairs,
prompt = raw `def signature [+ docstring]`, output = the function body.
3b's contamination scan: 0.004%, all on the solution side. The 10,000
signature-only pairs (code_if_pairs_control.jsonl) are the held-out negative
control and are NOT packed here.

Note: 1,123/99,996 train prompts carry no docstring (source had none). Packed
as-is -- flagged to 4c 2026-09-09; a filter is one line if the control needs
the train side docstring-pure.
"""

import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # datagen/: holdout, prepare_sft
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))  # loader

from tokenizers import Tokenizer  # noqa: E402

from holdout import is_holdout  # noqa: E402
from prepare_sft import pack_and_save  # noqa: E402

ROOT = os.path.dirname(_HERE)
DATA = os.path.join(ROOT, "data")
TOK_PATH = os.path.join(DATA, "tokenizer.json")
SRC = os.path.join(DATA, "sft", "code_if_pairs_train.jsonl")
OUT_PATH = os.path.join(DATA, "sft", "sft_format_0909.pt")

SEQ = 4096
MAX_EXAMPLES = 3_000_000


def read_examples(path):
    n = dropped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q = (d.get("prompt") or "").strip()
            a = (d.get("output") or "").strip()
            if not q or not a:
                continue
            if is_holdout(q):
                dropped += 1
                continue
            yield (q, a)  # RAW pair: no ChatML wrapping, by design
            n += 1
    print(f"  {os.path.basename(path)}: {n} kept, {dropped} holdout-dropped",
          flush=True)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SRC)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--tokenizer", default=TOK_PATH)
    args = ap.parse_args()

    random.seed(42)
    tok = Tokenizer.from_file(args.tokenizer)
    print(f"tokenizer {args.tokenizer} (vocab {tok.get_vocab_size()})", flush=True)
    eos = tok.token_to_id("<eos>")
    assert eos is not None, "tokenizer has no <eos>"

    examples = list(read_examples(args.source))
    random.shuffle(examples)
    if len(examples) > MAX_EXAMPLES:
        examples = examples[:MAX_EXAMPLES]
    print(f"total examples: {len(examples)}", flush=True)

    # sources= names what THIS pack read, so sources_fp stamps the right files.
    pack_and_save(examples, tok, eos, args.out, SEQ,
                  sources=[(args.source, "prompt", "output")])


if __name__ == "__main__":
    main()

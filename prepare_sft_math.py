#!/usr/bin/env python3
"""Prepare stage-2 math SFT data: same packing as prepare_sft.py, math-heavy mix.

Mix (~69% math / 31% general by rows):
  school_math_train (~220K, deduped, 500 held out)
  gsm8k_zh_train (7.5K, #### normalized to 答案是：)
  alpaca_gpt4_zh (52K, general replay)
  coig_50k (50K sampled, exam/reading replay)
"""

import json
import os
import random
import sys

from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from holdout import is_holdout  # noqa: E402
from loader import format_example  # noqa: E402
from prepare_sft import pack_and_save  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
TOK_PATH = os.path.join(DATA, "tokenizer.json")
OUT_PATH = os.path.join(DATA, "sft", "sft_math.pt")

SEQ = 4096
MAX_EXAMPLES = 3_000_000
ENC_BATCH = 8192

SOURCES = [
    (os.path.join(DATA, "workbatch", "school_math_train.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "workbatch", "gsm8k_zh_train.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "alpaca_gpt4_zh.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "workbatch", "coig_50k.jsonl"), "instruction", "output"),
]


def read_examples(sources=SOURCES):
    for path, qk, ak in sources:
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                q = (d.get(qk) or "").strip()
                a = (d.get(ak) or "").strip()
                if not q or not a or is_holdout(q):
                    continue
                yield format_example(q, a)
                n += 1
        print(f"  {os.path.basename(path)}: {n}", flush=True)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", help="comma-separated jsonl paths (instruction/output keys)")
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--fone", action="store_true", help="pack [NUM] + values for a --fone base")
    ap.add_argument(
        "--tokenizer",
        default=TOK_PATH,
        help="the vocabulary of the base this pack will train. data/tokenizer.json is rebuilt in "
        "place and ids do not survive a rebuild, so packing for an older base against today's file "
        "produces a pack whose every id is wrong -- it trains, at a loss four times too high.",
    )
    args = ap.parse_args()
    sources = [(p, "instruction", "output") for p in args.sources.split(",")] if args.sources else SOURCES
    random.seed(42)
    tok = Tokenizer.from_file(args.tokenizer)
    print(f"tokenizer {args.tokenizer} (vocab {tok.get_vocab_size()})", flush=True)
    eos = tok.token_to_id("<eos>")
    assert eos is not None, "tokenizer has no <eos>"
    num_id = None
    if args.fone:
        num_id = tok.token_to_id("[NUM]")
        assert num_id is not None, "tokenizer has no [NUM]; run scripts/build_tokenizer.py"

    examples = list(read_examples(sources))
    random.shuffle(examples)
    if len(examples) > MAX_EXAMPLES:
        examples = examples[:MAX_EXAMPLES]
    print(f"total examples: {len(examples)}", flush=True)

    pack_and_save(examples, tok, eos, args.out, SEQ, num_id=num_id)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pack the code-interface pairs into the SFT pack format (prepare_sft.pack_and_save).

Prompt = signature + docstring, masked; output = the complete function body,
supervised through <|im_end|>. The smoke check reloads the pack and asserts the
mask contract (scripts/test_sft_pack.py's spans) and that the body's LAST line is
inside a supervised span -- acceptance criterion 1 verified on the pack itself,
not only on the pairs.

    python datagen/prepare_sft_code_if.py --pairs data/sft/code_if_pairs_train.jsonl \
        --out data/sft/sft_code_if.pt
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))

from loader import format_example  # noqa: E402
from prepare_sft import pack_and_save  # noqa: E402

ROOT = os.path.dirname(_HERE)
TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
SEQ = 4096


def read_pairs(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            yield d["prompt"], d["output"]


def _spans(labels):
    """[(kind, start, end)] masked/supervised runs, mirroring test_sft_pack.runs."""
    out, cur, start = [], labels[0] == -100, 0
    for i, y in enumerate(list(labels) + [None]):
        m = (y == -100) if y is not None else not cur
        if m != cur:
            out.append(("masked" if cur else "supervised", start, i))
            cur, start = m, i
    return out


def smoke_check(out_path, pairs, tok):
    """The mask contract, plus sampled body-ends are supervised (acceptance 1)."""
    import random

    import torch

    d = torch.load(out_path, weights_only=True)
    ids, lab = d["input_ids"], d["labels"]
    eos = tok.token_to_id("<eos>")
    for r in range(ids.shape[0]):
        row, la = ids[r].tolist(), lab[r].tolist()
        for kind, a, b in _spans(la):
            text = tok.decode(row[a:b], skip_special_tokens=False)
            if kind == "masked" and b - a > 2 and set(row[a:b]) != {eos}:
                assert text.endswith("assistant\n"), f"masked span ends wrong: {text[-40:]!r}"
            elif kind == "supervised":
                assert "<|im_start|>" not in text, f"role marker supervised: {text[:80]!r}"
    # acceptance 1: the body's LAST line is inside supervised text. The pairs carry
    # complete bodies by construction (ast end_lineno); this checks the pack kept them.
    sup = "".join(
        tok.decode(
            [t for t, y in zip(ids[r].tolist(), lab[r].tolist(), strict=True) if y != -100],
            skip_special_tokens=False,
        )
        for r in range(ids.shape[0])
    )
    rng = random.Random(0)
    sample = rng.sample(pairs, min(200, len(pairs)))
    missing = [
        o.rstrip().splitlines()[-1].strip()
        for _, o in sample
        if o.rstrip().splitlines()[-1].strip() not in sup
    ]
    assert not missing, f"{len(missing)} body ends absent from supervised text: {missing[:3]}"
    print(f"smoke OK: {ids.shape[0]} rows, mask contract holds, 200/200 body ends supervised")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(ROOT, "data", "sft", "code_if_pairs_train.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "sft", "sft_code_if.pt"))
    ap.add_argument("--tokenizer", default=TOK_PATH)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    eos = tok.token_to_id("<eos>")
    assert eos is not None, "tokenizer has no <eos>"

    pairs = list(read_pairs(args.pairs))
    examples = [format_example(p, o) for p, o in pairs]
    print(f"{len(examples)} pairs from {args.pairs}", flush=True)
    pack_and_save(examples, tok, eos, args.out, SEQ, sources=[(args.pairs, "prompt", "output")])
    smoke_check(args.out, pairs, tok)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare the format-SFT pack: RAW (prompt, output) pairs, no ChatML.

Why raw: the eval interface is 0-shot raw continuation (the HumanEval prompt
field), and ChatML-wrapped SFT would teach the body after
<|im_start|>assistant -- a string the eval prompt never contains, so the 0-shot
number could not move by construction (prereg format_sft_humaneval_0909,
confirmed by 4c 2026-09-09). Everything else is prepare_sft_math's path: same
greedy packing, same holdout gate, same fingerprints.

Boundary (4c ruling C, 2026-09-09): prompt and body are encoded SEPARATELY and
the token streams concatenated (pack_and_save split_encode=True). The default
path encodes prompt+body as one string, and byte-level BPE then merges the
prompt's trailing "\\n" with the body's indentation on ~98% of pairs (measured
4,892/5,000): the merged token sits at the boundary, and no masking rule makes
the first supervised position equal what inference feeds. With split encoding
the inference sequence is a token-by-token prefix of the training sequence, at
the cost of an "unnatural" bare-\\n-then-bare-indentation sequence -- that
sequence IS the inference sequence, because the eval runner encodes the prompt
alone. Aligning the deployment split beats aligning the pretraining split.

split_encode is default-OFF in the shared packer and only this packer passes
it, so every existing pack keeps its exact bytes. Finding for the training
exp row (4c, 2026-09-09): prepare_sft_math.py calls the same shared packer on
the default path, so every prior SFT pack shares the boundary mechanism; its
rate on ChatML packs is unmeasured (their boundary is
<|im_start|>assistant\\n, a different merge). Decide whether to fix the math
path after this run's numbers.

Filter: 1,123/99,996 train prompts carry no docstring. They are dropped (count
in build_stats.filtered_no_docstring): the negative control needs the train
side docstring-pure, or "docstring arm moves, sig-only arm does not" has no
clean reading.

Val: 1,000 of the 10,000 signature-only control pairs (seed 42) are packed as
the val-loss set (eval/sft_val_loss.py); the other 9,000 stay on disk,
untouched, so a second val round cannot be fit to.

Source: data/sft/code_if_pairs_train.jsonl (3b, 2026-09-09): 99,996 pairs,
prompt = raw `def signature [+ docstring]`, output = the function body.
3b's contamination scan: 0.004%, all on the solution side.
"""

import json
import os
import random
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # datagen/: holdout, prepare_sft
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))  # loader

from tokenizers import Tokenizer  # noqa: E402

from holdout import is_holdout  # noqa: E402
from prepare_sft import _encode_pairs, pack_and_save  # noqa: E402

ROOT = os.path.dirname(_HERE)
DATA = os.path.join(ROOT, "data")
TOK_PATH = os.path.join(DATA, "tokenizer.json")
SRC = os.path.join(DATA, "sft", "code_if_pairs_train.jsonl")
CONTROL_SRC = os.path.join(DATA, "sft", "code_if_pairs_control.jsonl")
OUT_PATH = os.path.join(DATA, "sft", "sft_format_0909.pt")
VAL_OUT_PATH = os.path.join(DATA, "sft", "sft_format_val_0909.pt")

SEQ = 4096
MAX_EXAMPLES = 3_000_000
VAL_SIZE = 1000
SEED = 42

_DOCSTRING = re.compile(r'("""|\'\'\')')


def read_examples(path, require_docstring=False):
    """Return (pairs, counts). pairs are RAW (no ChatML) by design."""
    pairs, n, dropped, nodoc = [], 0, 0, 0
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
            if require_docstring and not _DOCSTRING.search(q):
                nodoc += 1
                continue
            pairs.append((q, a))
            n += 1
    print(f"  {os.path.basename(path)}: {n} kept, {dropped} holdout-dropped, "
          f"{nodoc} no-docstring-filtered", flush=True)
    return pairs, {"kept": n, "holdout_dropped": dropped,
                   "filtered_no_docstring": nodoc}


def known_answer_test(tok, eos):
    """The two invariants 4c ruled (2026-09-09), replacing the single-token check.

    Uses the real tokenizer: the boundary merge is a tokenizer property, so a
    fake tokenizer would make the test vacuous. The pair is asserted to actually
    merge across the boundary (else the assertions could pass on a packer that
    encodes concat-style), then the split path must satisfy:
      1. train_ids[:mask_len] == tok.encode(prompt).ids, token by token -- what
         the eval runner feeds is a prefix of the training sequence;
      2. tok.decode(train body) == prompt + body -- split encoding changes the
         token split, not the text the model reads.
    Either alone can be bypassed by a bad implementation; both together cannot.
    """
    prompt = 'def f(x):\n    """Sum x and one."""\n'
    body = '    return x + 1\n'
    ep = tok.encode(prompt).ids
    assert tok.encode(prompt + body).ids != ep + tok.encode(body).ids, \
        "test pair shows no prompt/body boundary merge -- pick a real merging pair"
    (p_ids, f_ids, _), = _encode_pairs([(prompt, body)], tok, None, split=True)
    train_ids = f_ids + [eos]
    mask_len = len(p_ids)
    assert train_ids[:mask_len] == tok.encode(prompt).ids, \
        "invariant 1: the masked prefix is not encode(prompt) token-by-token"
    assert tok.decode(f_ids) == prompt + body, \
        "invariant 2: split encoding changed the text, not just the split"
    return ("prefix == encode(prompt) token-by-token; "
            "decode(train) == prompt+body; merge asserted present")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SRC)
    ap.add_argument("--control", default=CONTROL_SRC)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--val_out", default=VAL_OUT_PATH)
    ap.add_argument("--val_size", type=int, default=VAL_SIZE)
    ap.add_argument("--tokenizer", default=TOK_PATH)
    args = ap.parse_args()

    random.seed(SEED)
    tok = Tokenizer.from_file(args.tokenizer)
    print(f"tokenizer {args.tokenizer} (vocab {tok.get_vocab_size()})", flush=True)
    eos = tok.token_to_id("<eos>")
    assert eos is not None, "tokenizer has no <eos>"

    examples, stats = read_examples(args.source, require_docstring=True)
    random.shuffle(examples)
    if len(examples) > MAX_EXAMPLES:
        examples = examples[:MAX_EXAMPLES]
    print(f"total examples: {len(examples)}", flush=True)

    # sources= names what THIS pack read, so sources_fp stamps the right files.
    pack_and_save(examples, tok, eos, args.out, SEQ,
                  sources=[(args.source, "prompt", "output")],
                  split_encode=True, extra_stats=stats)

    # Val-loss set: a slice of the held-out sig-only control, same boundary path.
    # The filter does NOT apply: the control is docstring-free by construction.
    val, vstats = read_examples(args.control)
    random.shuffle(val)
    val = val[:args.val_size]
    print(f"val examples: {len(val)} (of {vstats['kept']} control; "
          f"{vstats['kept'] - len(val)} held on disk, never read by training)",
          flush=True)
    pack_and_save(val, tok, eos, args.val_out, SEQ,
                  sources=[(args.control, "prompt", "output")],
                  split_encode=True,
                  extra_stats={"val_slice": len(val), "control_total": vstats["kept"],
                               "seed": SEED})


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        tok = Tokenizer.from_file(TOK_PATH)
        eos = tok.token_to_id("<eos>")
        print("prepare_format_sft known-answer OK:", known_answer_test(tok, eos))
    else:
        main()

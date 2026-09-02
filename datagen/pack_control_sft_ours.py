#!/usr/bin/env python3
# restartable: reads one jsonl, writes one .pt via pack_and_save's tmp+rename. An interrupt
# leaves no partial pack. CPU only, no card.
"""Tokenize the SHARED control text pack with OUR tokenizer into a .pt for sft_math.py.

    python3 datagen/pack_control_sft_ours.py \
        --text data/sft/control_sft_text.jsonl \
        --out data/sft/control_sft_ours.pt

The other half of scripts/sft_hf_control.py. Both arms read the same
control_sft_text.jsonl and quote the same sha256; each tokenizes it with its own
tokenizer, because the two vocabularies differ and a shared .pt would make every token id
valid and wrong (check_sft_ready.py:check_vocab exists to refuse exactly that).

This side reuses prepare_sft.pack_and_save verbatim -- the same masking, the same
whole-example packing, the same over-length drop, the same fingerprints -- so the only
thing that differs between the arms is what cannot be helped: the tokenizer, and the seq
length (4096 here vs 1024 on a 2048-position Pythia). Both arms' seq and drop counts go in
the report header per fb's ruling; `build_stats` in the pack is where this side's numbers
come from, so they survive the build log.

`sources` is the one text file, not prepare_sft.SOURCES: the pack was built from it, and a
sources_fp naming ten files this pack never read would be the false-provenance bug
_fp_sources documents.
"""

import argparse
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from loader import format_example  # noqa: E402
from prepare_sft import SEQ, pack_and_save  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--text", default=os.path.join(ROOT, "data", "sft", "control_sft_text.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "sft", "control_sft_ours.pt"))
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer.json"))
    ap.add_argument("--seq", type=int, default=SEQ)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    for p in (a.text, a.tokenizer):
        if not os.path.exists(p):
            print(f"CANNOT RUN: {p} does not exist")
            return 2

    from tokenizers import Tokenizer

    h = hashlib.sha256()
    with open(a.text, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    text_sha = h.hexdigest()

    pairs = []
    with open(a.text, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pairs.append(format_example(d["question"], d["answer"]))
    print(f"text {a.text}\n  sha256 {text_sha}\n  examples {len(pairs):,}", flush=True)

    tok = Tokenizer.from_file(a.tokenizer)
    eos = tok.token_to_id("<|endoftext|>")
    if eos is None:
        eos = tok.token_to_id("</s>")
    if eos is None:
        print("CANNOT RUN: no eos token found in the tokenizer")
        return 2

    # sources = the one file this pack was actually built from.
    pack_and_save(pairs, tok, eos, a.out, a.seq,
                  sources=[(a.text, "question", "answer")])

    import torch
    blob = torch.load(a.out, map_location="cpu", weights_only=True)
    st = blob.get("build_stats", {})
    print("\n=== report header numbers for the OURS arm")
    print(f"  pack            {os.path.relpath(a.out, ROOT)}")
    print(f"  text sha256     {text_sha}")
    print(f"  seq             {a.seq}")
    print(f"  rows            {st.get('rows'):,}" if st.get("rows") else "  rows            ?")
    print(f"  tokens          {blob['input_ids'].numel():,}")
    print(f"  dropped (>seq)  {st.get('dropped_overlong')}")
    print(f"  vocab_id        {blob.get('vocab_id')}")
    return 0


def selftest():
    """That this side and the control side produce the SAME (prompt, completion) split.

    The arms may differ in tokenizer and seq; they must not differ in the template or the
    loss boundary, and both call format_example -- so the check is that this file reads the
    shared pack's two fields the same way sft_hf_control.read_pack does.
    """
    import tempfile

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from sft_hf_control import read_pack

    fails = []
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for q, ans in (("q one", "a one"), ("q two", "a two")):
                f.write(json.dumps({"question": q, "answer": ans, "src": "t"}) + "\n")
        theirs = read_pack(p)
        mine = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                dd = json.loads(line)
                mine.append(format_example(dd["question"], dd["answer"]))
        if mine != theirs:
            fails.append(f"the two arms split the same row differently:\n  ours {mine}\n"
                         f"  control {theirs}")

    for f in fails:
        print(f"  SELFTEST FAIL {f}")
    if fails:
        print(f"\n{len(fails)} selftest failure(s)")
        return 1
    print("pack_control_sft_ours selftest OK (both arms produce an identical prompt/completion "
          "split from the same shared row)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

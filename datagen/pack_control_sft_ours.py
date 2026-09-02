#!/usr/bin/env python3
# restartable: reads one jsonl, writes one .pt via pack_and_save's tmp+rename. An interrupt
# leaves no partial pack. CPU only, no card.
"""Tokenize the SHARED control text pack with OUR tokenizer into a .pt for sft_math.py.

    python3 datagen/pack_control_sft_ours.py \
        --text data/sft/control_sft_text_train.jsonl \
        --out data/sft/control_sft_ours.pt

The other half of scripts/sft_hf_control.py. Both arms read the same TWO files --
<name>_train.jsonl and <name>_heldout.jsonl -- and quote the same sha256s; each tokenizes
them with its own tokenizer, because the two vocabularies differ and a shared .pt would make
every token id valid and wrong (check_sft_ready.py:check_vocab exists to refuse exactly
that).

THE SPLIT IS NOT MADE HERE, and that is the point (fb's ruling 2026-09-02). If each arm
applied "every 50th example" itself, our arm would train on the 2% the control holds out:
more training data AND the control's validation set. The builder splits once; both arms read
the result; the example ids in the files make that checkable rather than assumed.

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
    ap.add_argument("--text",
                    default=os.path.join(ROOT, "data", "sft", "control_sft_text_train.jsonl"))
    ap.add_argument("--heldout", default=None,
                    help="held-out text file; defaults to the _heldout.jsonl beside --text. "
                         "Packed separately so our arm neither trains on it nor misses it")
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

    def read(path):
        pairs, ids = [], []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                pairs.append(format_example(d["question"], d["answer"]))
                ids.append(d.get("id", i))
        return pairs, ids

    pairs, train_ids = read(a.text)
    print(f"text {a.text}\n  sha256 {text_sha}\n  examples {len(pairs):,}", flush=True)

    hp = a.heldout or (a.text[: -len("_train.jsonl")] + "_heldout.jsonl"
                       if a.text.endswith("_train.jsonl") else None)
    if not hp or not os.path.exists(hp):
        print(f"CANNOT RUN: held-out file {hp} does not exist. Both arms must hold out the "
              f"SAME examples; packing only the train file would leave our arm without the "
              f"held-out set the control is scored on.")
        return 2
    held, held_ids = read(hp)
    overlap = set(train_ids) & set(held_ids)
    if overlap:
        print(f"REFUSING: {len(overlap)} example id(s) are in both files")
        return 1
    print(f"held-out {hp}\n  examples {len(held):,}  overlap with train {len(overlap)}",
          flush=True)

    tok = Tokenizer.from_file(a.tokenizer)
    # "<eos>" is what OUR tokenizer calls it (id 1) -- prepare_sft.py:379 does the same
    # lookup. The first version of this file tried "<|endoftext|>" then "</s>", the HF names,
    # and both are absent here: eos came back None and pack_and_save died on
    # torch.tensor(rows_ids) with "NoneType cannot be interpreted as an integer", a message
    # that names the tensor rather than the tokenizer. The fallbacks stay for a foreign
    # tokenizer, after the name that actually applies.
    eos = tok.token_to_id("<eos>")
    for alt in ("<|endoftext|>", "</s>"):
        if eos is None:
            eos = tok.token_to_id(alt)
    if eos is None:
        print(f"CANNOT RUN: no eos token in {a.tokenizer} under <eos>, <|endoftext|> or </s>")
        return 2

    # sources = the file this pack was actually built from.
    pack_and_save(pairs, tok, eos, a.out, a.seq,
                  sources=[(a.text, "question", "answer")])
    held_out_pt = a.out.replace(".pt", "") + "_heldout.pt"
    pack_and_save(held, tok, eos, held_out_pt, a.seq,
                  sources=[(hp, "question", "answer")])

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
    hblob = torch.load(held_out_pt, map_location="cpu", weights_only=True)
    hst = hblob.get("build_stats", {})
    print(f"  held-out pack   {os.path.relpath(held_out_pt, ROOT)}")
    print(f"    examples      {len(held):,}  rows {hst.get('rows')}")
    print(f"    tokens        {hblob['input_ids'].numel():,}")
    # Per supervised BYTE: the arms' held-out losses are only comparable in a unit both
    # tokenizers share, and loss-per-token is not one when they segment the same text
    # into different counts.
    print(f"    supervised bytes {sum(len(c.encode()) for _, c in held):,}")
    ids_fp = hashlib.sha256(",".join(str(i) for i in held_ids).encode()).hexdigest()[:16]
    print(f"    held-out ids sha256 {ids_fp}  "
          f"(must equal the control arm's held_out_ids_sha256)")
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
            for i, (q, ans) in zip((7, 9), (("q one", "a one"), ("q two", "a two")),
                                   strict=True):
                f.write(json.dumps({"id": i, "question": q, "answer": ans, "src": "t"}) + "\n")
        theirs, their_ids = read_pack(p)
        mine, my_ids = [], []
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f):
                dd = json.loads(line)
                mine.append(format_example(dd["question"], dd["answer"]))
                my_ids.append(dd.get("id", i))
        if mine != theirs:
            fails.append(f"the two arms split the same row differently:\n  ours {mine}\n"
                         f"  control {theirs}")
        # fb's requirement: the two arms' held-out sets must be the SAME example ids. Both
        # arms read the same file, so the check is that both read the id field the same way --
        # the failure mode is one arm falling back to line order while the other uses "id".
        if my_ids != their_ids:
            fails.append(f"the arms disagree on example ids: ours {my_ids} vs "
                         f"control {their_ids}")
        if my_ids != [7, 9]:
            fails.append(f"the explicit id field was ignored: {my_ids}")

    for f in fails:
        print(f"  SELFTEST FAIL {f}")
    if fails:
        print(f"\n{len(fails)} selftest failure(s)")
        return 1
    print("pack_control_sft_ours selftest OK (both arms produce an identical "
          "prompt/completion split AND the same example ids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

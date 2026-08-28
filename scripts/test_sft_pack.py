#!/usr/bin/env python3
"""Pack a few ChatML examples and check the loss mask lands where it should.

This exists because the mask is invisible. A pack with a wrong boundary trains
without complaint, loses a couple of points, and nothing in the logs says why.
The first version of the ChatML switch shipped exactly that bug: format_example
returned (prompt, full_text) while pack_and_save builds a row as prompt +
completion, so every row carried the prompt twice and the second copy was
SUPERVISED -- the model would have been trained to write the questions. Row
count alone gave it away once measured: 40 examples packed into 8 rows before
the fix and 5 after.

    python scripts/test_sft_pack.py
"""

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def runs(labels):
    """[(kind, start, end)] spans of masked / supervised positions."""
    out, cur, start = [], labels[0] == -100, 0
    for i, y in enumerate(list(labels) + [None]):
        m = (y == -100) if y is not None else not cur
        if m != cur:
            out.append(("masked" if cur else "supervised", start, i))
            cur, start = m, i
    return out


def main():
    import torch
    from loader import IM_END, format_example
    from prepare_sft import pack_and_save
    from tokenizers import Tokenizer

    tok_path = os.path.join(ROOT, "data", "tokenizer.json")
    if not os.path.exists(tok_path):
        print("test_sft_pack SKIP (no data/tokenizer.json)")
        return
    tok = Tokenizer.from_file(tok_path)
    eos = tok.token_to_id("<eos>")
    im_end = tok.token_to_id(IM_END)

    pairs = [
        format_example(f"第{i}题：原价{100 + i}元打8折是多少？", f"{(100 + i) * 0.8:.0f}元")
        for i in range(40)
    ]
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "smoke.pt")
        pack_and_save(pairs, tok, eos, out, 255)
        d = torch.load(out, weights_only=True)

    ids, lab = d["input_ids"], d["labels"]
    from loader import vocab_fingerprint

    assert d["vocab"] == vocab_fingerprint(tok), (
        f"pack fingerprint {d['vocab']} != {vocab_fingerprint(tok)}; a pack whose fingerprint "
        "cannot equal a checkpoint's vocab_id makes sft_math.py's assert unsatisfiable"
    )

    row, la = ids[0].tolist(), lab[0].tolist()
    spans = runs(la)
    assert spans[0][0] == "masked", "a row must open with a masked prompt"
    for kind, a, b in spans:
        text = tok.decode(row[a:b], skip_special_tokens=False)
        if kind == "masked" and set(row[a:b]) == {eos}:
            continue  # right padding: masked <eos> to the end of the row, not a prompt
        if kind == "masked" and b - a > 2:
            assert text.endswith("assistant\n"), f"masked span does not end at the answer: {text[-40:]!r}"
            assert text.count("<|im_start|>user") == 1, f"prompt appears twice in one span: {text[:80]!r}"
        elif kind == "supervised":
            assert "<|im_start|>" not in text, (
                f"a role marker is SUPERVISED, so the model is trained to write questions: {text[:80]!r}"
            )
    sup = [t for t, y in zip(row, la, strict=True) if y != -100]
    assert im_end in sup, "the turn terminator is never supervised; the model cannot learn to stop"
    print(f"test_sft_pack OK ({len(pairs)} examples -> {ids.shape[0]} rows, {len(spans)} mask spans)")


if __name__ == "__main__":
    main()

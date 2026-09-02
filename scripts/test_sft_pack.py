#!/usr/bin/env python3
"""Pack a few ChatML examples and check the loss mask lands where it should.

The mask is invisible: a pack with a wrong boundary trains without complaint, loses a couple of
points, and nothing in the logs says why.

    python scripts/test_sft_pack.py
"""

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "datagen"))

from loader import IM_END, IM_START  # noqa: E402


def runs(labels):
    """[(kind, start, end)] spans of masked / supervised positions."""
    out, cur, start = [], labels[0] == -100, 0
    for i, y in enumerate(list(labels) + [None]):
        m = (y == -100) if y is not None else not cur
        if m != cur:
            out.append(("masked" if cur else "supervised", start, i))
            cur, start = m, i
    return out


def tool_spans_masked(row, labels, tok):
    """True iff every token from <|im_start|>tool through its <|im_end|> is -100.
    Stronger than the 'no role marker supervised' check in main(): it also catches
    a tool turn whose MARKERS are masked but whose CONTENT is supervised -- the
    pack that teaches the model to fabricate tool output."""
    open_ids = tok.encode(IM_START + "tool\n", add_special_tokens=False).ids
    im_end = tok.token_to_id(IM_END)
    L = len(open_ids)
    for i in range(len(row) - L + 1):
        if list(row[i : i + L]) == open_ids:
            j = i + L
            while j < len(row) and row[j] != im_end:
                j += 1
            if j >= len(row) or any(y != -100 for y in labels[i : j + 1]):
                return False
    return True


def main():
    import torch
    from loader import format_agentic, format_example, format_prompt
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

    assert d["vocab_id"] == vocab_fingerprint(tok), (
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

    # --- agentic (tool-call) packs: the tool turn is given, never generated ---
    conv = [
        {"role": "user", "content": "12/60 是多少？用计算器"},
        {"role": "assistant", "content": "12/60 = "},
        {"role": "tool", "content": "0.2"},
        {"role": "assistant", "content": "0.2 per minute"},
    ]
    apairs = format_agentic(conv)
    assert len(apairs) == 2, f"{len(apairs)} pairs for 2 assistant turns"
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "agentic.pt")
        pack_and_save(apairs, tok, eos, out, 255)
        d = torch.load(out, weights_only=True)
    ids, lab = d["input_ids"], d["labels"]
    for r in range(ids.shape[0]):
        assert tool_spans_masked(ids[r].tolist(), lab[r].tolist(), tok), (
            f"row {r}: a tool-turn token is supervised -- the model is taught to write tool output"
        )
    # the assistant's <|im_end|> on the tool call is supervised: the pivot of the loop
    row0, la0 = ids[0].tolist(), lab[0].tolist()
    assert im_end in [t for t, y in zip(row0, la0) if y != -100], "call's stop token is masked"

    # failing case: tool CONTENT supervised with its markers masked -- the exact pack
    # that trains a model to fabricate tool output, invisible to the marker check above.
    bad_pairs = [(format_prompt("q") + f"{IM_START}tool\n", f"0.2{IM_END}")]
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "bad.pt")
        pack_and_save(bad_pairs, tok, eos, out, 255)
        d = torch.load(out, weights_only=True)
    assert not tool_spans_masked(d["input_ids"][0].tolist(), d["labels"][0].tolist(), tok), (
        "guard did not fire: a pack with supervised tool content passed the tool-mask check"
    )

    # deliverable: one real rendered agentic sample, -100 spans marked
    print("\nagentic sample (masked=-100, supervised=loss):")
    for r in range(ids.shape[0]):
        rr, ll = ids[r].tolist(), lab[r].tolist()
        for kind, a, b in runs(ll):
            if set(rr[a:b]) == {eos}:
                continue  # right padding
            print(f"  row {r} {kind:10s} {tok.decode(rr[a:b], skip_special_tokens=False)!r}")

    print(f"test_sft_pack agentic OK ({len(apairs)} pairs -> {ids.shape[0]} rows; tool spans masked, failing case caught)")


if __name__ == "__main__":
    # --selftest is the hook's calling convention for every file in SELFTEST_FILES, and it
    # is equivalent to a bare run: the checks below are assertions, so a failure is a
    # non-zero exit either way. An unknown flag is refused rather than ignored, because a
    # script that exits 0 on an argument it did not understand registers as a pass (the
    # hook's own comment on why it drives the map instead of probing for the flag).
    if len(sys.argv) > 1 and sys.argv[1:] != ["--selftest"]:
        sys.exit(f"usage: {os.path.basename(__file__)} [--selftest]  (got {sys.argv[1:]})")
    main()

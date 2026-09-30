#!/usr/bin/env python3
"""Pack a few ChatML examples and check the loss mask lands where it should.

The mask is invisible: a pack with a wrong boundary trains without complaint, loses a couple of
points, and nothing in the logs says why.

    python scripts/test_sft_pack.py
"""

# restartable: every write this file makes goes into a tempfile.TemporaryDirectory that is
# removed on the way out, so an interrupt leaves nothing behind to resume from and nothing to
# clean up. The whole run is ~2 s on 40 synthetic examples; the way to recover from an interrupt
# is to run it again.

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


def _check_restamp_fails_closed_on_a_blind_scan():
    """prepare_sft.restamp_holdout_fp must refuse a pack it could read NOTHING out of.

    It certifies a pack when pack_holdout_hits reports 0 hits -- and a scan that extracted
    nothing also reports 0 hits. That is the whole failure mode: the first version of
    pack_holdout_hits returned 0 for zh_think_v1, whose sources hold 1,285 held-out questions,
    because it did not peel the chat wrapper. The peel fixed the 2026-09-28 packs and does
    nothing for a future pack with a different convention (de, 2026-09-30).

    Two worlds, both MUTATED from real pack_and_save output rather than hand-written: one where
    every label is -100, so no span is followed by a supervised token and the scan sees nothing,
    and the unmutated pack, which must still be readable. The second is the negative control --
    a guard that refused everything would pass the first case on its own.

    No vocabulary needed, so this runs on a laptop where the rest of this file SKIPs. The fake
    tokenizer's decode is never reached in the blind world (there is no span to decode) and
    returns a fixed non-holdout string in the control.
    """
    import torch

    from prepare_sft import pack_and_save, pack_holdout_hits, restamp_holdout_fp

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    class _FakeTok:
        def encode_batch(self, texts):
            # One id per character, so the full string's ids ARE a token prefix extension of the
            # prompt's. A fake that broke that invariant would make pack_and_save fall back to a
            # zero-length mask, leaving no -100 span at all -- and then the control would measure
            # the fake rather than the guard.
            return [_Enc([ord(c) % 900 + 100 for c in t]) for t in texts]

        def decode(self, ids):
            return "a question no holdout set contains " + str(len(ids))

        def get_vocab(self):  # pack_and_save stamps vocab_id through loader.vocab_fingerprint
            return {"<eos>": 1, "a": 0, "b": 2}

    tok = _FakeTok()
    pairs = [(f"q{i} ", f"a{i} ") for i in range(40)]
    with tempfile.TemporaryDirectory() as td:
        good = os.path.join(td, "good.pt")
        pack_and_save(pairs, tok, 1, good, 255)
        n_good, hits_good, _ = pack_holdout_hits(good, tok)
        assert n_good > 0 and hits_good == 0, (
            f"the control pack yields {n_good} spans / {hits_good} hits; it must be readable and "
            "clean, or this case proves nothing about the guard")

        blind = os.path.join(td, "blind.pt")
        d = torch.load(good, weights_only=True)
        rows = int(d["input_ids"].shape[0])
        d["labels"] = torch.full_like(d["labels"], -100)   # every position masked
        torch.save(d, blind)
        n_blind, hits_blind = pack_holdout_hits(blind, tok)[:2]
        assert (n_blind, hits_blind) == (0, 0), (
            f"the blind world does not reproduce the shape: {n_blind} spans, {hits_blind} hits. "
            "To the hit count it must look exactly like a clean pack, or the guard is not being "
            "tested against the case it exists for")
        try:
            restamp_holdout_fp(blind, tok)
        except SystemExit as e:
            assert "BLIND, NOT CLEAN" in str(e), f"refused for the wrong reason: {e}"
        else:
            raise AssertionError(
                "a pack the scan could read NOTHING out of was certified holdout-clean -- "
                "hits == 0 because nothing was looked at, the silent false-clean")
    print(f"test_sft_pack restamp-guard OK (control {n_good} spans readable; blind pack 0 spans "
          f"over {rows} rows -> refused 'BLIND, NOT CLEAN')")


def main():
    import torch
    from loader import format_agentic, format_example, format_prompt
    from prepare_sft import pack_and_save
    from tokenizers import Tokenizer

    # FIRST, and deliberately ABOVE the tokenizer SKIP below: this case needs no vocabulary, and
    # a guard that only runs where data/tokenizer.json exists would not run on a laptop commit at
    # all -- a check that passes by not running.
    _check_restamp_fails_closed_on_a_blind_scan()

    tok_path = os.path.join(ROOT, "data", "tokenizer.json")
    if not os.path.exists(tok_path):
        print("test_sft_pack SKIP (no data/tokenizer.json; the re-stamp guard above still ran)")
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

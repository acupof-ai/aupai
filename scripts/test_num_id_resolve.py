#!/usr/bin/env python3
"""Failing-case test for the num_id derivation (e1, 2026-09-01, P0).

Cfg.num_id was the constant 32772 with a comment saying [NUM] is "always in the
vocab". True of the frozen tokenizer, false the moment one is rebuilt. The bug is
worth a test not because a rebuild is likely but because num_id is read at three
sites -- fone masking (train.py:808), digit cross-entropy (:999), value
write-back (:1229) -- and each fails SILENTLY into plausible training: a stale id
masks an ordinary BPE token as numeric with correct shapes and no error.

Runs in milliseconds, no GPU, no checkpoint: the property under test is id
resolution, so no model is needed.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("FLA_FLASH_KDA", "0")


class FakeTok:
    """Minimal token_to_id surface -- the only thing resolve_num_id may use."""

    def __init__(self, mapping, size):
        self._m, self._n = mapping, size

    def token_to_id(self, t):
        return self._m.get(t)

    def get_vocab_size(self):
        return self._n


def main():
    from train import resolve_num_id

    # (1) the world as it is: [NUM] last, id resolves to it
    ok = FakeTok({"[NUM]": 32772}, 32773)
    assert resolve_num_id(ok) == 32772, "must return the tokenizer's own id"

    # (2) THE BUG. A rebuild yields a smaller vocabulary and [NUM] moves down. The old
    #     constant 32772 would now address an ordinary BPE token; the derived id must
    #     follow the tokenizer instead.
    moved = FakeTok({"[NUM]": 31000}, 31001)
    got = resolve_num_id(moved)
    assert got == 31000, f"must follow the rebuilt tokenizer, got {got}"
    assert got != 32772, "returning the stale constant is the defect this test exists for"

    # (3) [NUM] absent entirely -> refuse, never fall back to a default
    try:
        resolve_num_id(FakeTok({}, 32000))
        raise AssertionError("a vocabulary without [NUM] must REFUSE, not return a default")
    except SystemExit as e:
        assert "[NUM]" in str(e), f"the refusal must name the missing token: {e}"

    print("test_num_id_resolve OK: 3 cases (present, moved by rebuild, absent -> refuse)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

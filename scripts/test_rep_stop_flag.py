#!/usr/bin/env python3
"""--no_rep_stop must actually change what is generated (de, 2026-09-01).

A wired flag that no-ops is the vacuous shape: the arg parses, the value threads
through, and the generation is byte-identical. That is worse than no flag, because the
sampled arm would report "rep_stop off" over text the stop had truncated anyway.

No GPU and no checkpoint. A tiny stub model whose logits force a repeating cycle is
enough: the property under test is the STOP's behaviour, not any model's. Under
rep_stop the generation must halt early; with the flag it must run to max_new.

    python3 scripts/test_rep_stop_flag.py

Exit 0 = the flag changes the output length. Exit 1 = it is decorative.
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class CycleModel(torch.nn.Module):
    """Emits a fixed cycle of token ids forever. The repetition stop must catch it;
    without the stop it must run to max_new. Deterministic, so the two arms differ
    only by the flag."""

    def __init__(self, vocab, cycle, hid=8):
        super().__init__()
        self.vocab, self.cycle, self.hid = vocab, cycle, hid
        self.cfg = type("C", (), {"num_id": None})()
        self._t = 0

    def forward(self, x, num_vals=None, no_head=False):
        b, t = x.shape
        return None, torch.zeros(b, t, self.hid)

    def lm_logits(self, h):
        b = h.shape[0]
        out = torch.full((b, self.vocab), -10.0)
        out[:, self.cycle[self._t % len(self.cycle)]] = 10.0
        self._t += 1
        return out


class WordTok:
    """decode(ids) -> space-separated words, so the whitespace 8-gram path is the one
    exercised. Ids map to distinct words; the cycle therefore repeats an n-gram."""

    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids)


def main():
    from train import generate_batch

    vocab = 64
    cycle = list(range(10, 18))  # an 8-word cycle: repeats a whitespace 8-gram
    max_new = 256
    prompt = [[5, 6, 7]]

    lengths = {}
    for label, rep_stop in (("rep_stop ON", True), ("rep_stop OFF", False)):
        m = CycleModel(vocab, cycle)
        with torch.no_grad():
            out = generate_batch(m, prompt, max_new, "cpu", 0.0, None,
                                 tokenizer=WordTok(), rep_stop=rep_stop)
        lengths[label] = len(out[0])

    on, off = lengths["rep_stop ON"], lengths["rep_stop OFF"]
    bad = []
    if off < max_new:
        bad.append(f"rep_stop OFF stopped at {off} of {max_new} tokens -- something else "
                   f"is truncating, so 'off' does not mean untruncated")
    if on >= off:
        bad.append(f"rep_stop ON produced {on} tokens, OFF produced {off}: the stop did "
                   f"not fire, so the flag changes nothing and the two arms are the same run")
    if bad:
        print("FAIL: --no_rep_stop is decorative")
        for b in bad:
            print(f"  {b}")
        return 1
    print(f"OK: rep_stop ON halts at {on} tokens, OFF runs to {off} -- the flag changes "
          f"the generation, so the sampled arm measures untruncated text")
    return 0


if __name__ == "__main__":
    sys.exit(main())

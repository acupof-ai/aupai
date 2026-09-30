"""The width check the raw decode loops never had.

serve.py, chat.py, infer.py, eval/humaneval_gen.py and eval/sampling.py each take position 0
of a model's forward return and argmax it as logits. v41f/lm.py V42LM.forward returned
(hidden, hidden) until 4357b65b, so position 0 was a dim-1024 hidden state: argmax over it
yields ids 0-1023, every one a valid token in a 32,768-slot vocabulary. Nothing raised, and
the pass@1 that came out was garbage.

The class of defect is not closed by fixing one forward. Any future model whose forward
returns a hidden state, a pooled state, or a head over a different width reproduces it, and
the only signal is a benchmark number that looks low rather than an exception. So the check
belongs at the read, in every loop that does the argmax.

    from scripts.decode_guard import last_logits
    lg = last_logits(model, model(x[:, -cfg.seq:]))

Known answer: python3 scripts/decode_guard.py --selftest
"""


def last_logits(model, out, vocab=None):
    """Last-position logits from a forward return, refused unless the width is the vocabulary.

    `out` is what `model(x)` returned: a tuple whose position 0 is the logits, or a bare
    tensor. `vocab` defaults to `model.cfg.vocab`; a model that carries no such field is
    refused rather than waved through, because an unchecked read here is the whole defect.
    """
    t = out[0] if isinstance(out, tuple) else out
    if vocab is None:
        vocab = getattr(getattr(model, "cfg", None), "vocab", None)
    if vocab is None:
        raise RuntimeError(
            f"{type(model).__name__} carries no cfg.vocab, so the decode loop cannot check that "
            "what it argmaxes is logits. A hidden state argmaxes to a valid token id, so this "
            "would have scored garbage silently. Pass vocab= explicitly."
        )
    width = t.shape[-1]
    if width != vocab:
        raise RuntimeError(
            f"{type(model).__name__}.forward returned width {width} where the decode loop reads "
            f"logits of width {vocab}. A hidden state argmaxes to a valid token id, so this would "
            "have scored garbage silently. Return logits at position 0, or pass no_head=True and "
            "call lm_logits yourself (train.generate_batch is the worked example)."
        )
    return t[:, -1]


def _selftest():
    """Both worlds, on the real signature: the logits width passes, the hidden width raises."""
    from types import SimpleNamespace

    import torch

    VOCAB, DIM = 97, 16
    model = SimpleNamespace(cfg=SimpleNamespace(vocab=VOCAB))

    good = (torch.zeros(2, 5, VOCAB), torch.zeros(2, 5, DIM))
    got = last_logits(model, good)
    assert got.shape == (2, VOCAB), got.shape
    print(f"ok   logits width {VOCAB} -> {tuple(got.shape)}")

    bad = (torch.zeros(2, 5, DIM), torch.zeros(2, 5, DIM))
    try:
        last_logits(model, bad)
    except RuntimeError as e:
        assert "garbage silently" in str(e), e
        assert f"width {DIM}" in str(e), e
        print(f"ok   hidden width {DIM} refused: {str(e)[:60]}...")
    else:
        raise AssertionError("a dim-16 hidden state passed the width check")

    # A bare tensor return, and a model with no cfg.vocab: both must be handled, the second
    # by refusing rather than by skipping the check.
    assert last_logits(model, torch.zeros(2, 5, VOCAB)).shape == (2, VOCAB)
    try:
        last_logits(SimpleNamespace(), torch.zeros(2, 5, VOCAB))
    except RuntimeError as e:
        assert "no cfg.vocab" in str(e), e
        print("ok   missing cfg.vocab refused")
    else:
        raise AssertionError("a model with no cfg.vocab was waved through")
    print("decode_guard self-test OK")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    else:
        sys.exit("usage: python3 scripts/decode_guard.py --selftest")

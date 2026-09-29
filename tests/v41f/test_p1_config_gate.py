"""The engram ON config gate: a bare V41FConfig() must not be buildable.

MEASURED defect (2026-09-18, de). V41FConfig() defaults to engram_layer_ids=(1,) -- engram
ON -- while BOTH derived fields sit at their unusable defaults, engram_num_embeddings=() and
engram_compressed_vocab_size=0. Nothing in the production path filled them: only tests did,
by passing engram_compressed_vocab_size by hand. So a real run built on the bare default died
inside V41FModel construction --

    compressed_vocab_size -> AssertionError (6, 0) out of NgramHashState
    num_embeddings        -> IndexError: tuple index out of range out of Engram

Both are loud, so nothing is silently wrong. But they fire at BUILD, after a caller has
committed to a shape and a card, and V41FConfig().validate() PASSED on that same config --
the cheap check said yes to a config the expensive path rejects. This file pins the gate
that moves the refusal to validate(), where it belongs, and pins that the fields cannot be
filled from `self` alone (the compressed size is measured off the tokenizer).

Run:  python3 tests/v41f/test_p1_config_gate.py --selftest
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
from ref_oracle import synthetic_tokenizer  # noqa: E402

from v41f.config import V41FConfig, v41f_s, v41f_small  # noqa: E402


def test_bare_prod_default_validate_is_refused():
    """THE regression. This config PASSED validate() before the gate and then died at build."""
    cfg = V41FConfig()
    assert cfg.engram_layer_ids, "fixture drift: this test needs the prod default engram ON"
    try:
        cfg.validate()
    except ValueError as e:
        assert "engram" in str(e) and "num_embeddings" in str(e), str(e)
    else:
        raise AssertionError(
            "V41FConfig().validate() accepted a config whose engram is ON and whose derived "
            "fields are at their ()/0 defaults -- the gate is gone"
        )


def test_validate_refuses_compressed_zero_with_embeddings_present():
    """The second field, isolated: filling num_embeddings must NOT smuggle 0 through."""
    cfg = V41FConfig(engram_num_embeddings=(204,))
    try:
        cfg.validate()
    except ValueError as e:
        assert "engram_compressed_vocab_size" in str(e), str(e)
    else:
        raise AssertionError("validate() accepted engram_compressed_vocab_size=0")


def test_validate_refuses_nonpositive_rows():
    cfg = V41FConfig(engram_num_embeddings=(0,), engram_compressed_vocab_size=6)
    try:
        cfg.validate()
    except ValueError as e:
        assert "positive" in str(e), str(e)
    else:
        raise AssertionError("validate() accepted a zero-row engram table")


def test_with_derived_engram_needs_the_tokenizer_when_on():
    """The compressed size has no source but the tokenizer; the docstring now says so."""
    try:
        V41FConfig().with_derived_engram()
    except ValueError as e:
        assert "tokenizer" in str(e), str(e)
    else:
        raise AssertionError("with_derived_engram() filled an ON config with no tokenizer")


def test_with_derived_engram_fills_both_and_validates():
    tok = synthetic_tokenizer()
    cfg = V41FConfig().with_derived_engram(tokenizer=tok)
    _, compressed = __import__("v41f.engram", fromlist=["x"]).build_compressed_token_map(tok)
    assert cfg.engram_compressed_vocab_size == compressed, (cfg.engram_compressed_vocab_size, compressed)
    assert cfg.engram_num_embeddings, "num_embeddings not derived"
    cfg.validate()  # must NOT raise: this is the buildable shape


def test_off_path_needs_no_tokenizer():
    cfg = V41FConfig(engram_layer_ids=()).with_derived_engram()
    cfg.validate()
    assert cfg.engram_num_embeddings == () and cfg.engram_compressed_vocab_size == 0


def test_v41f_s_requires_a_tokenizer():
    """One name, one validity level: v41f_s must not return an unvalidated config.

    A tokenizer-less form that skipped validate() would be indistinguishable at the call
    site from one that passed, and the next BUILD from it would die inside NgramHashState.
    Shape arithmetic asks for `V41FConfig()` by name instead. fb ruling 2026-09-18.
    """
    with_tok = v41f_s(tokenizer=synthetic_tokenizer())
    assert with_tok.engram_compressed_vocab_size > 0
    with_tok.validate()  # must not raise
    try:
        v41f_s()
    except TypeError as e:
        assert "tokenizer" in str(e), str(e)
    except ValueError as e:
        assert "tokenizer" in str(e), str(e)
    else:
        raise AssertionError(
            "v41f_s() returned a config with no tokenizer -- the unvalidated shape must be "
            "asked for as V41FConfig(), never reached through a validated name"
        )


def test_v41f_small_engram_on_requires_a_tokenizer():
    """Same rule for the small builder: an engram-ON override needs the tokenizer."""
    tok = synthetic_tokenizer()
    on = v41f_small(
        vocab_size=len(tok),
        tokenizer=tok,
        engram_layer_ids=(1,),
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=8,
        engram_vocab_size=20,
        engram_pad_id=2,
    )
    assert on.engram_compressed_vocab_size > 0
    try:
        v41f_small(
            engram_layer_ids=(1,), engram_n_heads=2, engram_head_dim=8, engram_vocab_size=20, engram_pad_id=2
        )
    except ValueError as e:
        assert "tokenizer" in str(e), str(e)
    else:
        raise AssertionError("v41f_small() built engram-ON with no tokenizer")
    # and the OFF default still needs none
    v41f_small()


def test_broken_config_is_refused_before_construction():
    """The point of the gate: refusal happens at validate(), not inside V41FModel."""
    cfg = V41FConfig()
    try:
        cfg.validate()
    except ValueError:
        return
    raise AssertionError("no refusal -- construction would be the first place this is caught")


def _selftest():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
            bad += 1
    print(f"config gate: {len(tests) - bad}/{len(tests)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(_selftest())

#!/usr/bin/env python3
"""Single loader for checkpoint / tokenizer / prompt across every consumer.

Root cause this fixes: each consumer (chat, serve, infer, eval/*, rlvr*, ckpt
tools) implemented its own checkpoint->HybridLM + Tokenizer.from_file + prompt
formatting. That copied four worry: the model must be built from ck[\"cfg\"]
but a live Cfg worked in some and silently diverged in others; the tokenizer
must match the checkpoint's OWN vocabulary (a 32,772-token vocab can disagree
on every id while the size matches — measured 2026-08-28, four-fold loss without
raising); and the prompt becomes 问：{q}\\n答： in one place and ChatML somewhere
else. There is one canonical version of each; everything routes through here.

    from loader import load_checkpoint, load_tokenizer, format_prompt
    model, cfg = load_checkpoint(path)           # built from ck[\"cfg\"], never live Cfg
    tok = load_tokenizer(tokenizer_path, cfg)    # asserts vocab-size match + vocab_id fingerprint
    text = format_prompt(question)               # the training-format prompt, one source
"""

import hashlib
import os
import warnings
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def vocab_fingerprint(tok):
    """Hash of the id->token map (mirrors train.vocab_fingerprint, no torch dep here)."""
    h = hashlib.sha256()
    for t, _i in sorted(tok.get_vocab().items(), key=lambda kv: kv[1]):
        h.update(t.encode())
    return h.hexdigest()[:16]


# sft.py, sft_math.py, eval/gsm8k.py and algorithms/rlvr_generate.py each hardcode
# <eos> as id 1, and Cfg pins [NUM] at 32772. Both are true of today's vocabulary and
# neither is checked anywhere, so a rebuilt vocabulary would move them silently. The
# self-test below asserts them against the real file, in one place instead of four.
EOS_ID = 1
NUM_ID = 32772


def load_checkpoint(path, device="cpu", fone_ok=True):
    """Load a checkpoint and the model built strictly from it.

    The model is constructed from ck[\"cfg\"] — never from the live Cfg class,
    which the train loop owns and which is not stable across runs. Returns
    (model, cfg); cfg carries an added `vocab_id` (the checkpoint's own vocabulary
    fingerprint, or None for pre-fingerprint checkpoints) so load_tokenizer can
    cross-check it.
    """
    import torch

    from train import HybridLM  # delayed: this pulls torch; consumers already require it

    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    cfg.grad_ckpt = False
    cfg.vocab_id = ck.get("vocab_id")  # pre-2026-08-29 ckpts have none -> None
    # A FoNE checkpoint reads a number through the [NUM] value channel. A caller that
    # does not fill that channel gets every number as zero and scores garbage without
    # raising -- math-500 read 0.4% instead of 47.0% that way. Callers with no FoNE
    # path pass fone_ok=False and get an error instead of a wrong number.
    if getattr(cfg, "fone", False) and not fone_ok:
        raise RuntimeError(
            f"{path} is a --fone checkpoint and this caller has no FoNE encode/decode path; "
            "numbers would silently read as zero. Use a tool that passes num_vals "
            "(eval/math_hard.py, eval/math_zh.py, eval/run_eval.py, infer_local.py)."
        )
    model = HybridLM(cfg).to(device)
    if "model" in ck:
        model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


def load_tokenizer(path, cfg):
    """Load the tokenizer and VERIFY it matches the checkpoint's vocabulary.

    Two checks, both hard where the checkpoint pins them:
      1. size == cfg.vocab (a wrong file of a different size is already a mismatch).
      2. if cfg.vocab_id is set (checkpoint carries the id->token hash), the loaded
         tokenizer's fingerprint must equal it; a mismatch means the wrong file and
         raises rather than training/decoding scrambled ids in silence.
    A checkpoint WITHOUT vocab_id (old format) cannot be cross-checked — warn, don't
    fail.
    """
    from tokenizers import Tokenizer

    assert os.path.exists(path), (
        f"{path} is missing. Build it with `python scripts/build_tokenizer.py --force`, "
        "which is the only supported path."
    )
    tok = Tokenizer.from_file(path)
    if getattr(cfg, "vocab", None) is not None:
        assert tok.get_vocab_size() == cfg.vocab, (
            f"{path} has vocab {tok.get_vocab_size()} but the checkpoint was trained at "
            f"{cfg.vocab}; this is a different vocabulary."
        )
    vid = getattr(cfg, "vocab_id", None)
    if vid is not None:
        fp = vocab_fingerprint(tok)
        if fp != vid:
            raise RuntimeError(
                f"vocabulary mismatch: tokenizer fingerprint {fp} != checkpoint vocab_id {vid}. "
                "The tokenizer does not match the one this checkpoint was trained on; loading "
                "anyway would decode scrambled ids."
            )
    else:
        warnings.warn("checkpoint has no vocab_id (old format); cannot cross-check tokenizer", stacklevel=2)
    return tok


def format_prompt(question):
    """The exact prompt format the SFT data is trained with — single source of truth."""
    return f"问：{question}\n答："


# -- self-test: the loader must agree on a real checkpoint-tokenizer pair ----------
def _demo():
    """The three ways this loader is supposed to fail, each actually provoked.

    The point of the fingerprint is that it REJECTS a wrong tokenizer, so a
    self-test that never constructs a wrong one proves nothing.
    """
    from tokenizers import Tokenizer

    path = os.path.join(ROOT, "data", "tokenizer.json")
    assert os.path.exists(path), f"{path} missing for selftest"
    tok = Tokenizer.from_file(path)
    n = tok.get_vocab_size()
    fp = vocab_fingerprint(tok)
    assert fp == vocab_fingerprint(Tokenizer.from_file(path)), "fingerprint not stable across loads"

    # 1. the matching pair loads.
    load_tokenizer(path, SimpleNamespace(vocab=n, vocab_id=fp))

    # 2. right size, wrong ids -> must raise. This is the k5 bug: sizes agreed and
    #    nothing complained while every id was wrong.
    try:
        load_tokenizer(path, SimpleNamespace(vocab=n, vocab_id="0" * 16))
        raise AssertionError("a wrong vocab_id was accepted")
    except RuntimeError:
        pass

    # 3. wrong size -> must raise before the fingerprint is even consulted.
    try:
        load_tokenizer(path, SimpleNamespace(vocab=n + 1, vocab_id=fp))
        raise AssertionError("a wrong vocab size was accepted")
    except AssertionError as e:
        assert "different vocabulary" in str(e), e

    # 4. no fingerprint (pre-2026-08-29 checkpoint) -> warn, still load.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_tokenizer(path, SimpleNamespace(vocab=n, vocab_id=None))
        assert w and "vocab_id" in str(w[0].message), "old-format checkpoint did not warn"

    assert format_prompt("x") == "问：x\n答："

    # 5. the ids four files hardcode really are these ids.
    assert tok.token_to_id("<eos>") == EOS_ID, f"<eos> moved to {tok.token_to_id('<eos>')}"
    assert tok.token_to_id("[NUM]") == NUM_ID, f"[NUM] moved to {tok.token_to_id('[NUM]')}"
    print(f"loader self-test OK (vocab {n}, fingerprint {fp}, eos {EOS_ID}, num {NUM_ID})")


def _demo_keys():
    """infer_local keeps its own HybridLM because fla is CUDA-only. Two copies of a
    model drift: the FoNE heads were added to train.py and infer_local crashed on
    every k6 checkpoint until someone hit it by hand. Compare the parameter names
    without importing either module's GPU dependencies."""
    import ast
    import re

    names = {}
    for f in ("train.py", "infer_local.py"):
        with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
            src = fh.read()
        cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "HybridLM")
        body = ast.get_source_segment(src, cls)
        names[f] = set(re.findall(r"self\.([a-z_][a-z0-9_]*)\s*=\s*nn\.", body))
    a, b = names["train.py"], names["infer_local.py"]
    assert a == b, (
        f"HybridLM diverged: train.py only {sorted(a - b)}, infer_local.py only {sorted(b - a)}. "
        "A checkpoint saved by one will not load in the other."
    )
    print(f"HybridLM submodules agree across both copies ({len(a)}): {sorted(a)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        _demo_keys()
        _demo()

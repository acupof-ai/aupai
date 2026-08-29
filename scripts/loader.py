#!/usr/bin/env python3
"""Single loader for checkpoint / tokenizer / prompt across every consumer.

Every consumer used to roll its own, and each copy diverged silently. Notably: two
32,772-token vocabularies can disagree on every id while the size matches — measured
2026-08-28, a four-fold loss with nothing raising.

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


# sft.py, sft_math.py, eval/gsm8k.py and algorithms/rlvr_generate.py hardcode these ids;
# a vocabulary rebuild moves them silently. _demo() asserts them against the real file.
EOS_ID = 1
NUM_ID = 32772


def load_checkpoint(path, device="cpu", fone_ok=True):
    """(model, cfg), the model built from ck["cfg"] — never the live Cfg, which the
    train loop owns and which is not stable across runs. cfg gains `vocab_id` for
    load_tokenizer to cross-check."""
    import torch

    from train import HybridLM  # delayed: this pulls torch; consumers already require it

    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    cfg.grad_ckpt = False
    cfg.vocab_id = ck.get("vocab_id")  # pre-2026-08-29 ckpts have none -> None
    # A caller that does not fill the [NUM] value channel reads every number as zero and
    # scores garbage without raising -- math-500 read 0.4% instead of 47.0% that way.
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
    """Load the tokenizer and VERIFY it matches the checkpoint: size == cfg.vocab, then
    fingerprint == cfg.vocab_id. An old checkpoint without vocab_id only warns."""
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


#: ChatML, replacing the homemade 问：/答：. The four specials were already in the
#: vocabulary and appeared ZERO times in 160,414 chat documents, so the switch was free.
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def format_prompt(question, system=None):
    """The prompt the model is trained to answer — single source of truth."""
    s = f"{IM_START}system\n{system}{IM_END}\n" if system else ""
    return f"{s}{IM_START}user\n{question}{IM_END}\n{IM_START}assistant\n"


def format_example(question, answer, system=None):
    """(prompt, completion) split at the loss boundary: pack_and_save builds the row as
    prompt + completion, so returning joined text here writes the prompt into every row
    twice (scripts/packsmoke, 2026-08-29). The completion ends with <|im_end|> because
    the model must be supervised on its own stop token to learn to stop."""
    return format_prompt(question, system), f"{answer}{IM_END}"


def format_history(messages):
    """Multi-turn. `messages` is [{"role", "content"}]; a trailing assistant turn is
    left open for the model to continue."""
    out = "".join(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages)
    return out + f"{IM_START}assistant\n"


# -- self-test: the loader must agree on a real checkpoint-tokenizer pair ----------
def _demo():
    """Each way this loader is supposed to fail, actually provoked -- a self-test that
    never builds a wrong tokenizer proves nothing about the fingerprint."""
    from tokenizers import Tokenizer

    path = os.path.join(ROOT, "data", "tokenizer.json")
    if not os.path.exists(path):
        # data/tokenizer.json is gitignored, so this asserted on every clean checkout and
        # made CI red at step 4 for the whole checkout's existence -- every step after it,
        # including harness.py check, has never once run. test_arch_compat.py and
        # test_sft_pack.py already degrade this way; this was the only one that did not.
        print("loader selftest SKIP (no data/tokenizer.json)")
        return
    tok = Tokenizer.from_file(path)
    n = tok.get_vocab_size()
    fp = vocab_fingerprint(tok)
    assert fp == vocab_fingerprint(Tokenizer.from_file(path)), "fingerprint not stable across loads"

    # 1. the matching pair loads.
    load_tokenizer(path, SimpleNamespace(vocab=n, vocab_id=fp))

    # 2. right size, wrong ids -> must raise (the k5 bug: sizes agreed, every id wrong).
    try:
        load_tokenizer(path, SimpleNamespace(vocab=n, vocab_id="0" * 16))
        raise AssertionError("a wrong vocab_id was accepted")
    except RuntimeError:
        pass

    # 3. wrong size -> must raise before the fingerprint is consulted.
    try:
        load_tokenizer(path, SimpleNamespace(vocab=n + 1, vocab_id=fp))
        raise AssertionError("a wrong vocab size was accepted")
    except AssertionError as e:
        assert "different vocabulary" in str(e), e

    # 4. no fingerprint (old checkpoint) -> warn, still load.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_tokenizer(path, SimpleNamespace(vocab=n, vocab_id=None))
        assert w and "vocab_id" in str(w[0].message), "old-format checkpoint did not warn"

    assert format_prompt("x") == "<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n"
    pr, comp = format_example("x", "y")
    assert pr == format_prompt("x") and comp == "y<|im_end|>"
    assert (pr + comp).count("<|im_start|>user") == 1, "the prompt appears twice in one example"
    assert format_prompt("x", system="s").startswith("<|im_start|>system\ns<|im_end|>")
    assert format_history([{"role": "user", "content": "a"}]) == format_prompt("a")
    # one token each, or the format costs 8 tokens a turn
    for sp in (IM_START, IM_END):
        assert len(tok.encode(sp, add_special_tokens=False).ids) == 1, (
            f"{sp} is not a single token in this vocabulary; add it in scripts/build_tokenizer.py"
        )

    # 5. the ids four files hardcode really are these ids.
    assert tok.token_to_id("<eos>") == EOS_ID, f"<eos> moved to {tok.token_to_id('<eos>')}"
    assert tok.token_to_id("[NUM]") == NUM_ID, f"[NUM] moved to {tok.token_to_id('[NUM]')}"
    print(f"loader self-test OK (vocab {n}, fingerprint {fp}, eos {EOS_ID}, num {NUM_ID})")


def _demo_keys():
    """infer_local keeps its own HybridLM because fla is CUDA-only, and the two drift:
    FoNE heads landed in train.py only, and infer_local crashed on every k6 checkpoint
    until someone hit it by hand. Compares parameter names via AST, no GPU deps."""
    import ast
    import re

    names = {}
    for f in ("train.py", "infer_local.py"):
        if not os.path.exists(os.path.join(ROOT, f)):
            # infer_local.py is the local-Mac copy and is not shipped to the pod.
            print(f"HybridLM key check SKIP ({f} not present here)")
            return
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

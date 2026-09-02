#!/usr/bin/env python3
"""Single loader for checkpoint / tokenizer / prompt across every consumer.

Two same-size vocabularies can disagree on every id; size never identifies one.

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


# A TRIPWIRE, not a shared import. Nothing imports these (checked 2026-09-02: zero
# references to loader.NUM_ID anywhere, and sft.py:40, sft_math.py:42, eval/gsm8k.py:20 each
# keep their own `EOS_ID = 1`). Their value is the pair of asserts in _demo(), which fail the
# moment a vocabulary rebuild moves either id -- catching for those private copies what they
# cannot catch for themselves. An earlier version of this comment said the four files
# "hardcode these ids", naming algorithms/rlvr_generate.py, which references neither; a
# docstring describing a dependency structure that does not exist is worse than none, because
# it tells the next reader that changing this line is dangerous when the danger is elsewhere.
#
# train.py does NOT rely on this: resolve_num_id() (train.py:1429) derives num_id from the
# tokenizer and Cfg.num_id is set from it (train.py:1467), with scripts/test_num_id_resolve.py
# asserting it reads the tokenizer's own id rather than a literal.
EOS_ID = 1
NUM_ID = 32772


def load_checkpoint(path, device="cpu", dtype=None, fone_ok=True):
    """(model, cfg), the model built from ck["cfg"] — never the live Cfg, which the
    train loop owns and which is not stable across runs. cfg gains `vocab_id` for
    load_tokenizer to cross-check."""
    import torch

    from train import Cfg, HybridLM  # delayed: this pulls torch; consumers already require it

    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    # Old checkpoints predate newer Cfg keys (e.g. chunk_size, added 2026-08-30):
    # backfill live defaults so they still load. Safe only where the default is
    # numerically neutral at inference -- chunk_size 32 vs 64 is (eff.chunk_size_parity).
    for _k in vars(Cfg):
        if not _k.startswith("_") and not hasattr(cfg, _k):
            setattr(cfg, _k, getattr(Cfg, _k))
    cfg.grad_ckpt = False
    cfg.vocab_id = ck.get("vocab_id")  # pre-2026-08-29 ckpts have none -> None
    # A caller without a FoNE encode/decode path reads every number as zero and scores
    # garbage without raising.
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
    if dtype is not None:
        # bf16 (fp32 raises in the SWA blocks) then .contiguous() (cublasGemmEx fails on
        # non-contiguous parameters).
        model = model.to(dtype)
        for p in model.parameters():
            p.data = p.data.contiguous()
    # Compatibility shim: the flash_kda backend requires fp32 A_log/dt_bias and
    # refuses bf16 ("A_log must be float32"); the Triton default accepts both.
    # Training is bf16 end-to-end (--fp8 casts the whole model), so this is a
    # zero-extend -- bit-identical on the Triton path (measured max logit diff
    # 0.0, 2026-08-30). It exists only to keep the fp32-only backend usable.
    for _m in model.modules():
        if hasattr(_m, "A_log"):
            _m.A_log.data = _m.A_log.data.float()
            if hasattr(_m, "dt_bias"):
                _m.dt_bias.data = _m.dt_bias.data.float()
    return model, cfg


def load_tokenizer(path, cfg):
    """Load the tokenizer and VERIFY it matches the checkpoint: size == cfg.vocab_real,
    then fingerprint == cfg.vocab_id. An old checkpoint without vocab_id only warns.

    vocab_REAL, not vocab: cfg.vocab (32784) is vocab_real (32773) padded to a multiple
    of 16 so the head hits the aligned cuBLAS kernel, and this line asks a question about
    TOKENS. The docstring said cfg.vocab while the code asserted vocab_real -- harmless
    here, but the same conflation in build_tokenizer.py targeted 32779 merges and would
    have emitted a 32784-token vocabulary that this very assert then rejects on every
    existing checkpoint."""
    from tokenizers import Tokenizer

    assert os.path.exists(path), (
        f"{path} is missing. Build it with `python scripts/build_tokenizer.py --force`, "
        "which is the only supported path."
    )
    tok = Tokenizer.from_file(path)
    if getattr(cfg, "vocab", None) is not None:
        # cfg.vocab is the PADDED size (8/16-aligned for cuBLAS/_fp8_ok); the tokenizer's
        # size is vocab_real. Old checkpoints (trained unpadded) have no vocab_real.
        assert tok.get_vocab_size() == getattr(cfg, "vocab_real", cfg.vocab), (
            f"{path} has vocab {tok.get_vocab_size()} but the checkpoint was trained at "
            f"vocab_real {getattr(cfg, 'vocab_real', cfg.vocab)}; this is a different vocabulary."
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


#: ChatML, owned here.
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def format_prompt(question, system=None):
    """The prompt the model is trained to answer — single source of truth."""
    s = f"{IM_START}system\n{system}{IM_END}\n" if system else ""
    return f"{s}{IM_START}user\n{question}{IM_END}\n{IM_START}assistant\n"


def format_continuation(question, demos=()):
    """A base checkpoint's prompt: plain text it CONTINUES, never a chat turn.

    format_prompt is right for anything instruction-tuned and wrong for a base model, and
    the difference is measured, not stylistic. Its ChatML prefix occurs 0 times in 168,000
    corpus rows sampled across all 42 domains (AGENTS.md:200, stated as a bound: 0 of 4000
    puts a domain's rate under 0.075%), so a base model handed
    <|im_start|>user...<|im_start|>assistant is not answering a question -- it is
    continuing a token sequence it has never seen, and it repeats the input or drifts into
    web boilerplate. eval/score_code_exec.py:9-31 measured the size of this: 41 of 2586
    generations contained a code fence under ChatML (1.6%), against 469 of 497 under
    1-shot plain continuation (94.4%), same checkpoint family. Every base generative zero
    taken before 2026-09-02 measures response to an unseen prefix, not capability.

    `问：/答：` rather than English labels because that is what the corpus actually holds:
    the chat domain is 问：/答： plain text in 4000 of 4000 rows sampled. Demos are the
    real mechanism -- eval/l1_fewshot.py and eval/code_fewshot.py already do this and are
    the arm that works; a zero-shot continuation is still a legitimate prompt (the trailing
    `答：` is the format cue) but it measures less.
    """
    parts = [f"问：{q}\n答：{a}" for q, a in demos]
    parts.append(f"问：{question}\n答：")
    return "\n\n".join(parts)


def prompt_fn(kind):
    """The prompt builder a checkpoint of this type can actually answer.

    ONE owner, next to the two formatters it selects between. Copied into each of the five
    eval scripts it would be five things to keep in step, and the defect this fixes was
    exactly a second list drifting from the first (score_matrix validated metric names
    against the union of all types and never against the type in front of it).

    Callers pass the kind from eval/score_matrix.classify(cfg, name), which reads the
    checkpoint rather than the filename -- 'sft' in a filename proves nothing, and the old
    default in eval/gsm8k.py was the literal string "ckpt_sft.pt".
    """
    return format_continuation if kind == "base" else format_prompt


def format_example(question, answer, system=None):
    """(prompt, completion) split at the loss boundary: pack_and_save concatenates them, so
    joined text here would duplicate the prompt into every row. The completion ends with
    <|im_end|> because the model must be supervised on its own stop token to learn to stop."""
    return format_prompt(question, system), f"{answer}{IM_END}"


def format_history(messages):
    """Multi-turn. `messages` is [{"role", "content"}]; a trailing assistant turn is
    left open for the model to continue."""
    out = "".join(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages)
    return out + f"{IM_START}assistant\n"


def format_agentic(messages):
    """Multi-turn conversation with tool calls -> [(prompt, completion)], one pair
    per assistant turn. Each assistant turn is supervised exactly once: as its own
    pair's completion, with every prior turn (user/assistant/tool) masked in the
    prompt. The packer's single mask boundary is what makes this a list of pairs
    rather than one sequence with holes.

    Tool turns are plain-text role turns (<|im_start|>tool\\n...<|im_end|>): no new
    special tokens, lossless under skip_special_tokens=False. The assistant's
    <|im_end|> on a tool call is supervised like any other stop -- it is the pivot
    the agent loop waits on, and masking it trains a model that never stops
    calling. Tool output is never supervised: teaching the model to write tool
    output trains the opposite of an agent.

    The assistant turn after a tool turn is a CONTINUATION, not a restatement:
    it picks up from the tool result, never repeats the expression. A
    restatement-trained model re-narrates history on every loop, and the context
    is only 4096 tokens.

        user:      "12/60 是多少？用计算器"
        assistant: "12/60 = "        <- tool call, supervised on stopping here
        tool:      "0.2"             <- given, masked
        assistant: "0.2 per minute"  <- continuation from the result, supervised
    """
    for i, m in enumerate(messages):
        if m["role"] == "tool" and (i == 0 or messages[i - 1]["role"] != "assistant"):
            raise ValueError(f"tool turn at index {i} does not follow an assistant turn")
    return [
        (format_history(messages[:i]), f"{m['content']}{IM_END}")
        for i, m in enumerate(messages)
        if m["role"] == "assistant"
    ]


def _demo():
    """Each way this loader is supposed to fail, actually provoked -- a self-test that
    never builds a wrong tokenizer proves nothing about the fingerprint."""
    from tokenizers import Tokenizer

    # BEFORE the tokenizer gate, because these need no tokenizer and the gate returns on
    # every clean checkout (data/tokenizer.json is gitignored). The existing format
    # assertions live below it and therefore do not run here at all -- which is how a
    # format defect stays invisible on the machine where most commits are made.
    #
    # e1-22: the two formats must not be confusable. format_prompt is ChatML and stays
    # that way (chat.py, serve.py, rlvr_trainer and SFT packing all depend on it);
    # format_continuation is what a BASE checkpoint gets, and the property is that no
    # ChatML marker can reach it.
    assert format_prompt("x") == f"{IM_START}user\nx{IM_END}\n{IM_START}assistant\n"
    assert format_continuation("x") == "问：x\n答：", format_continuation("x")
    assert format_continuation("b", [("a", "1")]) == "问：a\n答：1\n\n问：b\n答："
    for probe in (format_continuation("x"), format_continuation("y", [("a", "1"), ("b", "2")])):
        assert IM_START not in probe and IM_END not in probe, probe
        assert "im_start" not in probe and "im_end" not in probe, probe
    # A demo pair must actually appear, or "few-shot" would be a label on a zero-shot
    # prompt -- the arm eval/l1_fewshot.py measures as the one that works.
    assert "问：a\n答：1" in format_continuation("b", [("a", "1")])

    path = os.path.join(ROOT, "data", "tokenizer.json")
    if not os.path.exists(path):
        # data/tokenizer.json is gitignored; a hard assert here fails every clean checkout.
        print("loader selftest SKIP (no data/tokenizer.json)")
        return
    tok = Tokenizer.from_file(path)
    n = tok.get_vocab_size()
    fp = vocab_fingerprint(tok)
    assert fp == vocab_fingerprint(Tokenizer.from_file(path)), "fingerprint not stable across loads"

    # 1. the matching pair loads.
    load_tokenizer(path, SimpleNamespace(vocab=n, vocab_id=fp))

    # 2. right size, wrong ids -> must raise.
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
    # agentic: one pair per assistant turn; the tool turn and the prior assistant
    # turn both land in the masked prompt.
    conv = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "call"},
        {"role": "tool", "content": "0.2"},
        {"role": "assistant", "content": "ans"},
    ]
    ap = format_agentic(conv)
    assert len(ap) == 2, f"{len(ap)} pairs for 2 assistant turns"
    assert ap[0] == (format_prompt("q"), "call" + IM_END)
    assert ap[1][0] == (
        f"{IM_START}user\nq{IM_END}\n"
        f"{IM_START}assistant\ncall{IM_END}\n"
        f"{IM_START}tool\n0.2{IM_END}\n"
        f"{IM_START}assistant\n"
    ), ap[1][0]
    assert ap[1][1] == "ans" + IM_END
    # the tool turn renders losslessly (8 tokens without the trailing newline)
    tool_turn = f"{IM_START}tool\n0.2{IM_END}"
    assert tok.decode(tok.encode(tool_turn, add_special_tokens=False).ids,
                      skip_special_tokens=False) == tool_turn
    for bad in ([{"role": "tool", "content": "x"}],
                [{"role": "user", "content": "q"}, {"role": "tool", "content": "x"}]):
        try:
            format_agentic(bad)
            raise AssertionError(f"malformed conversation accepted: {bad}")
        except ValueError:
            pass
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
    """infer_local keeps its own HybridLM because fla is CUDA-only; the two must not drift.
    Compares parameter names via AST, no GPU deps."""
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

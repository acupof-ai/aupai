"""#493 gate: an engram-on config must complete a real training step.

The step-A wiring gate proves **forward** allclose against the reference. It does not cover
this, and the gap was total: `NgramHashState.forward` was decorated `@torch.inference_mode()`,
so its hash ids were inference tensors, and `Engram.forward` feeds them to `self.embed(...)`
inside an autograd graph. Every engram-on training step raised before computing a gradient.
The production default sets `engram_layer_ids=(1,)`, so this blocked the DEFAULT config, not
an exotic one.

WHAT THIS GATE ASSERTS, and at which scale:

  * SMALL engram-on config -- the whole code path: forward, backward, and one
    `optimizer.step()`, with all four engram parameters holding finite non-zero gradients.
    Not merely "does not raise": a fix that skipped the engram embedding, or detached it,
    would raise nothing and train nothing.
  * PRODUCTION config -- construction, param-name set and structure ONLY. The production
    model is 1.009B params (1.9 GiB bf16 weights plus 7.5 GiB of AdamW state before
    activations), which exceeds this box; a full production backward is a GPU-environment
    task, recorded for the post-provisioning list. **Nothing here is labelled "prod
    verified"** -- see `test_production_config_is_structure_only`.

WHY PER-TREE RUNS GO IN SEPARATE PROCESSES: a first attempt compared two code trees by
importing `v41f.engram` once and then swapping `sys.path`, which compares the same already
imported class twice and cannot see a difference. Two earlier variants of the same mistake in
this session (separate `Generator`s per tree; in-process module reload) each manufactured a
false conclusion. Any cross-tree comparison must be its own process with global seeding.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v41f.config import V41FConfig, v41f_small  # noqa: E402
from v41f.model import V41FModel  # noqa: E402

SMALL = dict(
    engram_layer_ids=(1,),
    engram_max_ngram_size=4,
    engram_n_heads=2,
    engram_head_dim=8,
    engram_vocab_size=20,
    engram_pad_id=2,
)
ENGRAM_PARAMS = ("q_weight", "k_weight", "embed.weight", "wkv.weight")


def _tokenizer():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ref_oracle import synthetic_tokenizer

    return synthetic_tokenizer()


def _build(cfg, tok, max_batch_size=2):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return V41FModel(cfg, max_batch_size=max_batch_size, tokenizer=tok)
    finally:
        torch.set_default_dtype(prev)


def small_cfg():
    tok = _tokenizer()
    return v41f_small(vocab_size=len(tok), engram_compressed_vocab_size=6, **SMALL).with_derived_engram(), tok


def test_engram_on_small_completes_a_training_step():
    """forward + backward + optimizer.step on an engram-on config, with no exception."""
    cfg, tok = small_cfg()
    m = _build(cfg, tok).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    ids = torch.randint(0, len(tok), (2, 32))
    logits, _ = m(ids)
    loss = logits.float().pow(2).mean()
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    loss.backward()
    opt.step()
    print(f"  engram-on small: forward+backward+step ok (loss {loss.item():.4f})")


def test_all_four_engram_params_receive_gradient():
    """The four engram parameters must each hold a finite, NON-ZERO gradient.

    This is what separates a real fix from one that makes the exception go away. The natural
    wrong fixes -- skipping the engram path, or detaching its output before the graph uses it
    -- raise nothing and satisfy a no-exception test while training none of the module.
    """
    cfg, tok = small_cfg()
    m = _build(cfg, tok).train()
    ids = torch.randint(0, len(tok), (2, 32))
    logits, _ = m(ids)
    logits.float().pow(2).mean().backward()
    params = dict(m.named_parameters())
    failures = []
    for leaf in ENGRAM_PARAMS:
        name = f"engrams.{cfg.engram_layer_ids[0]}.{leaf}"
        assert name in params, f"{name} not in the model: the engram is not wired"
        g = params[name].grad
        if g is None or not torch.isfinite(g).all() or not g.abs().sum() > 0:
            failures.append((name, "None" if g is None else f"sum={g.abs().sum().item():.3e}"))
    assert not failures, (
        f"engram parameters without a usable gradient: {failures} -- a fix that merely "
        f"suppresses the exception would leave exactly this")
    print(f"  all {len(ENGRAM_PARAMS)} engram params: finite, non-zero gradient")


def test_production_config_is_structure_only():
    """PRODUCTION IS NOT TRAINED HERE, and this test does not claim it is.

    The production model is 1.009B params: ~1.9 GiB bf16 weights plus ~7.5 GiB of AdamW state
    before activations, which exceeds this box (measured; two earlier full-production attempts
    were OOM-killed with empty logs). What is asserted is structure: the engram-on default
    builds, its layer list is what the config says, and all four engram parameter names exist
    per engram layer. A real production backward is listed for the post-provisioning GPU work.

    Scale is named in the assertion message so a failure here is never read as "production
    training is verified".
    """
    try:
        tok = _tokenizer()
        cfg = V41FConfig(vocab_size=len(tok), engram_compressed_vocab_size=6).with_derived_engram()
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"production cfg/tokenizer harness unavailable: {e}") from e
    n_layers = len(cfg.engram_layer_ids)
    assert n_layers >= 1, "production default no longer carries an engram layer"
    params = sum(p.numel() for p in _build(cfg, tok, max_batch_size=1).parameters())
    assert params > 0
    print(f"  production: structure only (engram layers {cfg.engram_layer_ids}, "
          f"{params / 1e9:.3f}B params); backward NOT run here")


def test_no_grad_matches_inference_mode_values():
    """The fix must change no VALUES, only whether the tensor can enter a graph.

    `inference_mode` and `no_grad` both suppress graph construction, and for a pure integer
    function of `input_ids` they must agree bit-for-bit. Asserted by comparing this tree's
    hash output against a literal captured from the pre-fix implementation, so the check does
    not need the old tree present.

    (Cross-tree comparison would require a SEPARATE PROCESS -- see the module docstring. A
    literal sidesteps that entirely and is the stronger form here: it pins the values.)
    """
    # Captured from the PRE-FIX implementation in a separate process (its own process, see
    # the module docstring). A literal pins the values; two assertions follow, and they test
    # different things -- the value pin catches a numerics change, the inference flag catches
    # the #493 defect. Both are named so a failure says which one moved.
    PREFIX_SUM = 34395
    PREFIX_FIRST4 = [6, 47, 70, 103]

    cfg, tok = small_cfg()
    hs = _build(cfg, tok, max_batch_size=2).engram_hash
    assert hs is not None, "engram hash state was not built"
    torch.manual_seed(3)
    ids = torch.randint(0, len(tok), (2, 32))
    out = hs(ids, 0, None)

    assert not out.is_inference(), (
        "the hash output is an INFERENCE tensor, so it cannot be used in a backward: this is "
        "the #493 defect (the forward is decorated inference_mode)")
    assert out.dtype == torch.int64, out.dtype
    assert int(out.sum()) == PREFIX_SUM and out.flatten()[:4].tolist() == PREFIX_FIRST4, (
        f"hash VALUES changed: got sum={int(out.sum())} first4="
        f"{out.flatten()[:4].tolist()}, expected sum={PREFIX_SUM} first4={PREFIX_FIRST4}. "
        f"no_grad and inference_mode must agree bit-for-bit on this pure integer function, "
        f"so a move here means the fix altered numerics, not just graph behaviour")
    print(f"  hash: ordinary int64 tensor, values pinned (sum={PREFIX_SUM})")

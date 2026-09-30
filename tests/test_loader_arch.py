"""scripts/loader.load_checkpoint must build the architecture the checkpoint names.

It constructed HybridLM unconditionally, so every eval that goes through it -- the whole
eval/ surface -- could not score an --arch v42 checkpoint, and the four raw decode loops
(serve.py, chat.py, infer.py, eval/humaneval_gen.py) argmaxed a dim-1024 hidden state as if
it were logits, producing ids 0-1023 with nothing raised (2026-09-30).

The assertion is the property, not the class name: the loaded model's logits must be
vocab-wide. A hidden-width return is exactly the silent-wrong-number defect, so the test
would fail on it even if the class were right.

    python3 tests/test_loader_arch.py
"""

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

VOCAB = 97


def _cfg_ns(**over):
    """A live Cfg as the namespace save_checkpoint/loader pass around, with overrides applied."""
    from types import SimpleNamespace

    from train import Cfg

    d = {k: v for k, v in vars(Cfg).items() if not k.startswith("_") and not callable(v)}
    d.update(over)
    return SimpleNamespace(**d)


def _v42_cfg():
    from dataclasses import asdict, replace

    from v41f.config import v42_s24

    vc = replace(
        v42_s24(vocab_size=VOCAB),
        dim=64, n_layers=5, n_heads=4, head_dim=32, rope_head_dim=16, q_lora_rank=32,
        o_groups=2, o_lora_rank=16, window_size=4, compress_ratios=(0, 2, 2, 1, 1),
        kv_source_layers=(1, 3), index_source_layers=(1, 3), index_n_heads=4, index_head_dim=32,
        index_topk=3, n_routed_experts=4, n_activated_experts=2, moe_inter_dim=32, hc_mult=2,
    )
    vc.validate()
    return _cfg_ns(arch="v42", vocab=VOCAB, v42_cfg=asdict(vc))


def test_v42_checkpoint_loads_and_returns_vocab_wide_logits():
    import loader

    from train import build_model, save_checkpoint

    cfg = _v42_cfg()
    model = build_model(cfg)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ckpt_v42_tiny.pt"
        save_checkpoint(str(p), model.state_dict(), cfg, vocab_id="test-vocab", step=1)
        got, gcfg = loader.load_checkpoint(str(p), device="cpu", claim=False)
    assert type(got).__name__ == "V42LM", f"loaded {type(got).__name__}, not the v42 model"
    assert getattr(gcfg, "arch", "hybrid") == "v42"
    ids = torch.randint(2, VOCAB, (1, 16))
    # One dtype for the whole model: v41f builds some submodules in bf16 and others in fp32 on
    # purpose (hyperconn fp32, the compressor's fp32 wkv/wgate at ratio>1), which a real eval
    # resolves by loading with dtype=bfloat16 on a card. On CPU with dtype=None the mix raises in
    # F.linear, and that is not what this test is about.
    got = got.float()
    with torch.no_grad():
        out = got(ids)
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape[-1] == VOCAB, (
        f"last dim {logits.shape[-1]} != vocab {VOCAB}: a hidden state is being returned where "
        "callers argmax logits -- the silent-wrong-number defect this test exists for"
    )
    print(f"  v42 ckpt -> {type(got).__name__}, logits {tuple(logits.shape)} vocab-wide")

    # The other three branches of the contract, each with a real consumer: targets given is the
    # train/val step, no_head is generate_batch, return_hidden is scripts/logit_dist.py and
    # arith_probe_fone.py -- the last one raised TypeError until the signature took the argument.
    with torch.no_grad():
        h_tr, snd_tr = got(ids, torch.zeros_like(ids))
        snd_nh = got(ids, no_head=True)
        lg_rh, h_rh = got(ids, return_hidden=True)
        lg_no, h_no = got(ids)
    d = gcfg.v42_cfg["dim"]
    assert h_tr.shape[-1] == d and snd_tr is None, "targets given must be (hidden, None)"
    assert snd_nh[0] is None and snd_nh[1].shape[-1] == d, "no_head must be (None, hidden)"
    assert lg_rh.shape[-1] == VOCAB and h_rh.shape[-1] == d, "return_hidden must be (logits, hidden)"
    assert lg_no.shape[-1] == VOCAB and h_no is None, "without return_hidden position 1 is None"
    print("  contract: targets->(hidden,None), no_head->(None,hidden), return_hidden->(logits,hidden)")


def test_hybrid_checkpoint_still_loads():
    import loader

    from train import build_model, save_checkpoint

    cfg = _cfg_ns(vocab=VOCAB, dim=64, layers=2, heads=4, ffn_hidden=128,
                  moe_layers="", csa=False, csa2=False)
    model = build_model(cfg)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ckpt_hybrid_tiny.pt"
        save_checkpoint(str(p), model.state_dict(), cfg, vocab_id="test-vocab", step=1)
        got, _ = loader.load_checkpoint(str(p), device="cpu", claim=False)
    assert type(got).__name__ == "HybridLM", f"loaded {type(got).__name__}, not HybridLM"
    print(f"  hybrid ckpt -> {type(got).__name__} (no regression)")


TESTS = [test_v42_checkpoint_loads_and_returns_vocab_wide_logits, test_hybrid_checkpoint_still_loads]

if __name__ == "__main__":
    # --selftest is accepted and ignored: both cases ARE the selftest, and taking the conventional
    # flag keeps this file out of the hook's SELFTEST_FLAG map, where a wrong flag reads argparse's
    # exit 2 as a failing selftest and refuses every commit that stages the file.
    for t in TESTS:
        t()
        print(f"ok   {t.__name__}")
    print(f"loader arch gates: {len(TESTS)}/{len(TESTS)} passed")

"""GPU parity of the liger_kernel substitutions against the torch implementations they replace.

CUDA only (the kernels are Triton); on CPU it prints SKIP and exits 0. Not a test_p1_* file, so the
p1 glob runner does not pick it up; run it on the pod when a card is free:

  CUDA_VISIBLE_DEVICES=0 python3 tests/v41f/test_gpu_liger_parity.py

Cases, each printing max abs / worst grad rel and asserting a tolerance:
  - HyperConn hc_impl liger vs torch: coeffs (pre/post/comb fp32), hc_pre, hc_post, and the grads of
    hc_fn/hc_base/hc_scale/x through a full mixes->pre->post chain on bf16 streams;
  - RMSNorm norm_impl liger (casting_mode gemma) vs torch on bf16 input, forward and grad;
  - whole V42LM at the tiny test shape: logits and every grad, liger vs torch.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
sys.path.insert(0, str(Path(__file__).absolute().parent))

if not torch.cuda.is_available():
    print("SKIP test_gpu_liger_parity: no CUDA (liger kernels are Triton)")
    sys.exit(0)

from test_p1_docpack import _batch, _cfg, _grads, _model  # noqa: E402
from v41f import hyperconn  # noqa: E402
from v41f.norm_gate import RMSNorm  # noqa: E402

assert hyperconn.HAS_LIGER_MHC, "liger_kernel not importable on this pod"
DEV = "cuda"


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def test_hyperconn():
    torch.manual_seed(0)
    hc = hyperconn.HyperConn(64, hc_mult=4, norm_eps=1e-20).to(DEV)
    x = torch.randn(2, 37, 4, 64, device=DEV).bfloat16().requires_grad_()
    pre_mix = torch.rand(2, 37, 4, device=DEV)
    sub = torch.randn(2, 37, 64, device=DEV).bfloat16()
    outs = {}
    for impl in ("torch", "liger"):
        hc.impl = impl
        hc.zero_grad()
        xg = x.detach().clone().requires_grad_()
        pre, post, comb = hc.hc_mixes(xg, hc.hc_attn_fn, hc.hc_attn_scale, hc.hc_attn_base)
        a = hc.hc_pre(xg, pre_mix)
        y = hc.hc_post(sub + a, xg, post, comb)
        (y.float().square().sum() + pre.sum() + post.sum() + comb.sum()).backward()
        outs[impl] = dict(pre=pre, post=post, comb=comb, a=a, y=y, gx=xg.grad, gfn=hc.hc_attn_fn.grad,
                          gbase=hc.hc_attn_base.grad, gscale=hc.hc_attn_scale.grad)
    for k in outs["torch"]:
        r = _rel(outs["liger"][k], outs["torch"][k])
        tol = 2e-2 if k in ("a", "y", "gx") else 1e-3  # bf16 I/O vs fp32 coefficients
        print(f"  hyperconn {k:6s} rel {r:.2e}")
        assert r < tol, f"hyperconn {k}: rel {r:.3e} > {tol}"
    dev = (outs["liger"]["comb"].sum(-1) - 1).abs().max().item()
    print(f"  liger comb row sums deviate {dev:.1e} from 1 (doubly stochastic)")


def test_rmsnorm():
    torch.manual_seed(0)
    n = RMSNorm(256, 1e-20).to(DEV)
    n.weight.data.uniform_(0.5, 1.5)
    x = torch.randn(4, 100, 256, device=DEV).bfloat16()
    outs = {}
    for impl in ("torch", "liger"):
        n.impl = impl
        n.zero_grad()
        xg = x.clone().requires_grad_()
        y = n(xg)
        y.float().square().sum().backward()
        outs[impl] = (y, xg.grad, n.weight.grad.clone())
    for k, (a, b) in zip(("y", "gx", "gw"), zip(outs["liger"], outs["torch"])):
        r = _rel(a, b)
        print(f"  rmsnorm {k:3s} rel {r:.2e}")
        assert r < 1e-2, f"rmsnorm {k}: rel {r:.3e}"


def test_model():
    ids, cu = _batch()
    ids, cu = ids.to(DEV), cu.to(DEV)
    ref = _model(_cfg(attn_impl="chunked")).to(DEV)
    lig = _model(_cfg(attn_impl="chunked", hc_impl="liger", norm_impl="liger")).to(DEV)
    lig.load_state_dict(ref.state_dict())
    la, ga = _grads(ref, ids, cu)
    lb, gb = _grads(lig, ids, cu)
    d = (la - lb).abs().max().item()
    worst = max(_rel(gb[k], ga[k]) for k in ga)
    print(f"  model liger vs torch: logits max abs {d:.2e}, worst grad rel {worst:.2e} over {len(ga)} grads")
    assert d < 5e-2 and worst < 5e-2


if __name__ == "__main__":
    for t in (test_hyperconn, test_rmsnorm, test_model):
        t()
        print(f"ok   {t.__name__}")
    print("liger parity: 3/3 passed")

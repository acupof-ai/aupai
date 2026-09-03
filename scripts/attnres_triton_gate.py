"""fp32 parity gate: the Triton kernel against autograd on the plain expression.

fp32 ONLY, on purpose. A bf16 row here would compare the kernel to eager, and that
number is the distance between two inaccurate things -- it cannot say which one is
wrong. bf16 is gated in attnres_triton_bf16_gate.py against fp64 truth, where the bar
is eager's own error.
"""
import sys

import torch

sys.path.insert(0, "/work/aupai")
from algorithms.attnres_triton import triton_attn_res


def rms_scale(x, eps=1e-6):
    return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


torch.manual_seed(1)
dev, bar = "cuda", 1e-5
for n, B, T, D in ((6, 2, 64, 1024), (25, 2, 128, 1024), (25, 4, 256, 1024)):
    v = [torch.randn(B, T, D, device=dev, dtype=torch.float32, requires_grad=True)
         for _ in range(n)]
    gq = torch.randn(D, device=dev, dtype=torch.float32, requires_grad=True)
    dout = torch.randn(B, T, D, device=dev, dtype=torch.float32)

    # Truth is autograd on the expression model.py:268-273 writes, with scale LIVE --
    # Source.scale is rms_scale(v), so detaching it deletes one of v's two routes to the
    # output and dV silently loses 10% while the forward stays green.
    sc = [rms_scale(x) for x in v]
    lg = torch.stack([(v[i] * gq).sum(-1) * sc[i].squeeze(-1) for i in range(n)])
    sum(lg.softmax(0)[i].unsqueeze(-1) * v[i] for i in range(n)).backward(dout)
    ref_o = sum(lg.softmax(0)[i].unsqueeze(-1) * v[i] for i in range(n)).detach()
    ref_dv = [x.grad.clone() for x in v]
    ref_dgq = gq.grad.clone()
    for x in (*v, gq):
        x.grad = None

    out = triton_attn_res(v, gq, [rms_scale(x) for x in v])
    out.backward(dout)
    rel = lambda a, b: (a - b).abs().max().item() / b.abs().max().item()  # noqa: E731
    fo = rel(out.detach(), ref_o)
    fv = max(rel(x.grad, r) for x, r in zip(v, ref_dv))
    fq = rel(gq.grad, ref_dgq)
    ok = max(fo, fv, fq) < bar
    print(f"n={n:3d} B={B} T={T:4d}  fwd {fo:.2e}  dV {fv:.2e}  dgq {fq:.2e}  "
          f"{'PASS' if ok else 'FAIL'} (bar {bar:g})")
    assert ok

"""bf16 gate: the kernel against fp64 TRUTH, with eager's own error as the bar.

WHY NOT A CONSTANT BAR. The first version of this compared kernel to eager and read
9.32e-02 against a 3e-2 bar -- "FAIL". That number is the distance between two
inaccurate things and cannot say which one is wrong. Measured here, eager's forward is
itself 1.40e-01 from fp64 truth: the "failure" was mostly eager's error, and the kernel
is closer to truth than the thing that ships today. The question a gate must answer is
"is the kernel worse than what ships", so the bar is eager's error, not a constant.

THE 14% IS A MODEL FINDING, NOT A KERNEL ONE. It lives in the softmax weights: the
D=1024 dot product at model.py:269 accumulates in bf16, giving a logit absolute error of
0.858 against a spread of 279.8, which softmax amplifies. Whether to make that dot
product fp32 is a separate 500-step A/B on domain_loss -- not something a kernel gate
decides.
"""
import sys

import torch

sys.path.insert(0, "/work/aupai")
from algorithms.attnres_triton import triton_attn_res


def rms_scale(x, eps=1e-6):
    return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def eager(v, gq, s):
    """model.py:268-273 verbatim -- what ships today, and therefore the bar."""
    lg = torch.stack([(v[i] * gq).sum(-1) * s[i].squeeze(-1) for i in range(len(v))])
    a = lg.float().softmax(0).to(v[0].dtype)
    return sum(a[i].unsqueeze(-1) * v[i] for i in range(len(v)))


torch.manual_seed(1)
n, B, T, D = 25, 4, 256, 1024
v64 = [torch.randn(B, T, D, device="cuda", dtype=torch.float64) for _ in range(n)]
gq64 = torch.randn(D, device="cuda", dtype=torch.float64)
d64 = torch.randn(B, T, D, device="cuda", dtype=torch.float64)


def measure(mix, dtype):
    v = [x.to(dtype).clone().requires_grad_(True) for x in v64]
    g = gq64.to(dtype).clone().requires_grad_(True)
    out = mix(v, g, [rms_scale(x) for x in v])
    out.backward(d64.to(dtype))
    return out.detach(), [x.grad for x in v], g.grad


ref_o, ref_dv, ref_dgq = measure(eager, torch.float64)
rel = lambda a, b: (a.double() - b).abs().max().item() / b.abs().max().item()  # noqa: E731


def err(mix):
    o, dv, dgq = measure(mix, torch.bfloat16)
    return (rel(o, ref_o), max(rel(a, b) for a, b in zip(dv, ref_dv)), rel(dgq, ref_dgq))


e, t = err(eager), err(triton_attn_res)
print(f"{'':8s} {'fwd':>10s} {'dV':>10s} {'dgq':>10s}   (relative to fp64)")
print(f"{'eager':8s} {e[0]:10.2e} {e[1]:10.2e} {e[2]:10.2e}")
print(f"{'triton':8s} {t[0]:10.2e} {t[1]:10.2e} {t[2]:10.2e}")
ok = all(a <= b for a, b in zip(t, e))
print(f"\nkernel no worse than eager on every output: {'PASS' if ok else 'FAIL'}")
assert ok

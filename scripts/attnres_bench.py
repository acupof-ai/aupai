"""AttnRes eager vs fused: the isolated cost of removing the double read.

SELF-CONTAINED ON PURPOSE. Both implementations are inline and nothing is imported
from /work/aupai, so this touches none of the pod's shared tree -- the data leg's
model.py and train.py are exactly the files a push would overwrite, and this
measurement does not need them.

WHAT THIS ANSWERS, AND WHAT IT DOES NOT. It measures the AttnRes stack alone,
forward and backward, at the trace's real shapes. It does NOT give tok/s/gpu or
peak GiB for the real model; those need the 20-step two-arm run after the merge.
Read it as "what does removing the double read buy inside AttnRes", not as
"what does it buy the step".
"""
import sys
import time

import torch

D_MODEL = 1024


def rms_scale(x, eps=1e-6):
    return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def eager(v, gq, scale):
    """model.py:268-273 verbatim: logits over every source, softmax, then the loop."""
    logits = torch.stack([(v[i] * gq).sum(-1) * scale[i].squeeze(-1) for i in range(len(v))])
    a = logits.float().softmax(0).to(v[0].dtype)
    out = a[0].unsqueeze(-1) * v[0]
    for i in range(1, len(v)):
        out = out + a[i].unsqueeze(-1) * v[i]
    return out


def _fwd_impl(v, gq, scale):
    n = len(v)
    B, T, D = v[0].shape
    dev = v[0].device
    acc_dt = torch.float32
    m = torch.full((B, T), -float("inf"), dtype=acc_dt, device=dev)
    ell = torch.zeros(B, T, dtype=acc_dt, device=dev)
    acc = torch.zeros(B, T, D, dtype=acc_dt, device=dev)
    logit = torch.empty(n, B, T, dtype=acc_dt, device=dev)
    for i in range(n):
        logit[i] = (v[i].to(acc_dt) * gq).sum(-1) * scale[i].squeeze(-1)
        mi = torch.maximum(m, logit[i])
        rescale = torch.where(torch.isinf(m), torch.zeros_like(m), (m - mi).exp())
        p = (logit[i] - mi).exp()
        ell = ell * rescale + p
        acc = acc * rescale.unsqueeze(-1) + p.unsqueeze(-1) * v[i].to(acc_dt)
        m = mi
    return acc / ell.unsqueeze(-1), (logit - m).exp() / ell


def _bwd_impl(v, gq, scale, a, dout):
    n = len(v)
    dt = a.dtype
    dA = torch.stack([(dout * v[i].to(dt)).sum(-1) for i in range(n)])
    s = (a * dA).sum(0)
    dlogit = a * (dA - s.unsqueeze(0))
    dv, dscale = [], []
    dgq = torch.zeros_like(gq, dtype=dt)
    for i in range(n):
        w = (dlogit[i] * scale[i].squeeze(-1)).unsqueeze(-1)
        dv.append((a[i].unsqueeze(-1) * dout + w * gq).to(v[i].dtype))
        dgq += (w * v[i].to(dt)).sum((0, 1))
        dscale.append((dlogit[i] * (v[i].to(dt) * gq).sum(-1)).unsqueeze(-1).to(scale[i].dtype))
    return dv, dgq.to(gq.dtype), dscale


class FusedAttnRes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gq, n, *tensors):
        v, scale = list(tensors[:n]), list(tensors[n:])
        out, a = _fwd_impl(v, gq, scale)
        ctx.save_for_backward(gq, a, *v, *scale)
        ctx.n = n
        return out.to(v[0].dtype)

    @staticmethod
    def backward(ctx, dout):
        gq, a, *rest = ctx.saved_tensors
        n = ctx.n
        dv, dgq, dscale = _bwd_impl(rest[:n], gq, rest[n:], a, dout.to(a.dtype))
        return (dgq, None, *dv, *dscale)


def fused(v, gq, scale):
    return FusedAttnRes.apply(gq, len(v), *v, *scale)


def stack_step(L, mix, B, T, dtype, dev):
    """One AttnRes stack: 2L+1 calls, source i read by every call from i on.

    This is the O(L^2) shape -- call k sees k sources -- and it is what the 318 ms/step
    in the trace actually is, not a single call.
    """
    torch.manual_seed(0)
    gq = torch.randn(D_MODEL, device=dev, dtype=dtype, requires_grad=True)
    x = torch.randn(B, T, D_MODEL, device=dev, dtype=dtype, requires_grad=True)
    vs = [x]
    for _ in range(2 * L):
        out = mix(vs, gq, [rms_scale(t) for t in vs])
        vs = vs + [out]
    y = mix(vs, gq, [rms_scale(t) for t in vs])
    y.sum().backward()
    return y


def bench(L, B, T, dtype, dev, reps=3):
    out = {}
    for name, mix in (("eager", eager), ("fused", fused)):
        stack_step(L, mix, B, T, dtype, dev)  # warmup: compile/alloc
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(reps):
            stack_step(L, mix, B, T, dtype, dev)
        torch.cuda.synchronize()
        out[name] = ((time.perf_counter() - t0) / reps * 1000,
                     torch.cuda.max_memory_allocated() / 2**30)
    return out


if __name__ == "__main__":
    dev = "cuda"
    dtype = torch.bfloat16
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    print(f"card {torch.cuda.current_device()} {torch.cuda.get_device_name()}  "
          f"B={B} T={T} D={D_MODEL} {dtype}")

    # Value parity at these shapes before any timing: a faster wrong answer is not a result.
    torch.manual_seed(1)
    n = 6
    v = [torch.randn(2, 64, D_MODEL, device=dev, dtype=dtype) for _ in range(n)]
    gq = torch.randn(D_MODEL, device=dev, dtype=dtype)
    sc = [rms_scale(t) for t in v]
    a_, b_ = eager(v, gq, sc), fused(v, gq, sc)
    rel = (a_ - b_).abs().max().item() / a_.abs().max().item()
    print(f"parity at these shapes: rel {rel:.2e}  {'OK' if rel < 3e-2 else 'FAIL'}")

    print(f"\n{'L':>3} {'n':>3} {'eager ms':>9} {'fused ms':>9} {'speedup':>8} "
          f"{'eager GiB':>10} {'fused GiB':>10}")
    for L in (4, 8, 12):
        r = bench(L, B, T, dtype, dev)
        (em, eg), (fm, fg) = r["eager"], r["fused"]
        print(f"{L:>3} {2*L+1:>3} {em:>9.1f} {fm:>9.1f} {em/fm:>7.2f}x "
              f"{eg:>10.2f} {fg:>10.2f}")

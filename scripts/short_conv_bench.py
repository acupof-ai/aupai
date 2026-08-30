"""short_conv is 3.1% of a step on ATen's depthwise-conv2d GENERIC fallback. Price the alternatives.

KDA's ShortConv is nn.Conv1d(d, d, k=4, groups=d) -- a causal depthwise conv that ATen routes to
conv_depthwise2d_forward_kernel_generic. Measured on GPU 0 in the real model: fwd 21.17 + bwd 18.73
+ grad_weight 11.29 = 51.19 ms/step over 9 layers. Its roofline is ~134 us per layer at 4 TB/s, so
the generic kernel runs at roughly 6% of peak bandwidth -- the worst utilisation measured anywhere
in this model.

k=4 means the whole op is four shifted multiply-adds, which inductor can fuse if it is spelled as
arithmetic instead of handed over as an opaque ATen conv. That is arm C, and it is the point.

Run: CUDA_VISIBLE_DEVICES=0 python -u scripts/short_conv_bench.py

# restartable: every arm prints its row with flush as it finishes; an interrupt costs one arm.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

B, D, T, K = 32, 1024, 4096, 4
BYTES = (B * D * (T + K - 1) + B * D * T) * 2  # bf16 in + out, the fwd roofline
PEAK_TBS = 4.0


def aten(x, w, b):
    h = F.pad(x.transpose(1, 2), (K - 1, 0))
    return F.silu(F.conv1d(h, w, b, groups=D).transpose(1, 2))


def shifted(x, w, b):
    """The same four taps as arithmetic, so inductor can fuse instead of calling into ATen."""
    h = F.pad(x.transpose(1, 2), (K - 1, 0))
    y = sum(h[:, :, i : i + T] * w[:, 0, i].unsqueeze(-1) for i in range(K))  # conv1d is cross-correlation
    return F.silu((y + b.unsqueeze(-1)).transpose(1, 2))


def bench(fn, x, w, b, backward, iters=20, warmup=8):
    for _ in range(warmup):
        out = fn(x, w, b)
        if backward:
            out.sum().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(x, w, b)
        if backward:
            out.sum().backward()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


def main():
    dev = "cuda"
    torch.manual_seed(0)
    ref_conv = nn.Conv1d(D, D, K, groups=D).to(dev).to(torch.bfloat16)
    w, b = ref_conv.weight, ref_conv.bias
    x = torch.randn(B, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True)

    with torch.no_grad():  # same taps, so a mismatch here is a bug in the spelling, not in bf16
        assert torch.allclose(aten(x, w, b), shifted(x, w, b), rtol=2e-2, atol=2e-2), "arms disagree"

    roof_ms = BYTES / (PEAK_TBS * 1e12) * 1e3
    print(f"fwd roofline {roof_ms * 1e3:.0f} us/layer at {PEAK_TBS} TB/s, B={B} D={D} T={T} K={K}", flush=True)
    arms = [
        ("aten conv1d (current)", aten),
        ("shifted taps", shifted),
        ("shifted taps + compile", torch.compile(shifted, dynamic=False)),
    ]
    for name, fn in arms:
        for backward in (False, True):
            x.grad = None
            ms = bench(fn, x, w, b, backward)
            tag = "fwd+bwd" if backward else "fwd    "
            pct = "" if backward else f"  {roof_ms / ms * 100:.0f}% of roof"
            print(f"{name:26s} {tag} {ms:7.3f} ms{pct}", flush=True)


if __name__ == "__main__":
    main()

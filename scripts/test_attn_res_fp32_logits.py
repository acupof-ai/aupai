"""Does --attn_res_fp32_logits change the computation? CPU, bf16, no card.

A flag that threads but computes nothing would run 1000 steps and report "no difference"
-- the answer the pre-registration predicts, for the wrong reason. This separates the two
BEFORE the lane is spent.

SEED BEFORE EVERY INIT, not once at the top. The first version of this seeded only before
the sources, so the second arm's g/q were drawn from an RNG state the first arm had
advanced: it read 1.265 relative difference between arms that differed in their WEIGHTS as
well as the dtype, and would have said "not inert" for a flag that did nothing.
"""
import torch

from model import AttnRes, Source

torch.manual_seed(0)
n, B, T, D = 25, 2, 64, 256
srcs = [Source.of(torch.randn(B, T, D, dtype=torch.bfloat16)) for _ in range(n)]

outs = {}
for f32 in (False, True):
    torch.manual_seed(7)
    m = AttnRes(D, fp32_logits=f32).to(torch.bfloat16)
    with torch.no_grad():
        m.g.normal_(std=0.5)
        m.q.normal_(std=0.5)
    outs[f32] = m(srcs).float()

d = (outs[True] - outs[False]).abs().max().item() / outs[False].abs().max().item()
print(f"relative difference bf16 vs fp32 logits: {d:.3e}")
assert d > 1e-3, f"FLAG IS INERT: {d:.3e} -- the A/B would measure nothing"
print("flag changes the computation: OK")

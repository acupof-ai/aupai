"""Stochastic rounding fp32 -> bf16 for small-LR bf16 training (1e option B, 2026-09-25).

Round-to-nearest drops a sub-half-ULP update outright: at SFT LRs a bf16 weight's relative
step is ~1e-4, below bf16's ~2^-8 ULP, so `bf16(w + delta)` equals `w` for nearly every
element. Stochastic rounding picks the lower or upper bf16 grid point with probability
inverse to the fractional distance, so E[rounded] = x exactly: repeated sub-ULP updates
accumulate without bias and cross a grid point at the statistically correct rate.

Exact, portable, identical on CPU and CUDA: grid points are derived directly from bf16 bit
integers (bf16 is the top 16 bits of an fp32), no exponent/log arithmetic. Randomness comes
from one caller-supplied torch.Generator ON THE TENSOR DEVICE, so casts are reproducible
per seed. DDP ranks must share the seed: independent per-rank
draws would desynchronize the weight replicas.
"""

import torch

_BF16_EXP_BIAS = 127


def _bf16_neighbors(x):
    """-> (lower, upper, frac): the two numerically-adjacent bf16 grid values bracketing fp32
    x (lower <= x <= upper), and the fractional position of x in [lower, upper]. Works for
    negative values because the bf16 bit order is monotonic numerically: bit-1 / bit+1 give
    the two surrounding grid points regardless of sign. Exact-representable points collapse
    (step 0). Callers bound peak memory by passing CHUNKS, never a whole optimizer group at
    once: every tensor here is x-sized and a stacked MoE group is ~2.0e9 elements (8 GiB
    fp32 each, ~15 live temporaries)."""
    x = x.float()
    b = x.to(torch.bfloat16)
    bv = b.float()
    bits = b.view(torch.int16).to(torch.int32)
    n1 = bits - 1
    p1 = bits + 1
    a = n1.to(torch.int16).view(torch.bfloat16).float()
    c = p1.to(torch.int16).view(torch.bfloat16).float()
    lower = torch.minimum(a, c)
    upper = torch.maximum(a, c)
    # For exact points snap both to the value itself (a == c == x there).
    exact = bv == x
    lower = torch.where(exact, bv, torch.minimum(lower, bv))
    upper = torch.where(exact, bv, torch.maximum(upper, bv))
    step = upper - lower
    frac = torch.where(step > 0, ((x - lower) / step).clamp(0.0, 1.0), torch.zeros_like(x))
    return lower, upper, frac


def stochastic_round_bf16(x, generator):
    """fp32 -> bf16 with E[result] = x. `generator` must be on the SAME device as x.

    Pure function of (x, generator state): uses torch.rand(..., generator=generator) only,
    never the global RNG, so a seed replays exactly.
    """
    lower, upper, frac = _bf16_neighbors(x)
    draw = torch.rand(x.shape, dtype=torch.float32, device=x.device, generator=generator)
    pick_upper = draw < frac
    chosen = torch.where(pick_upper, upper, lower)
    return chosen.to(torch.bfloat16)


class StochasticRounder:
    """One reproducible generator per device, seeded by `seed` alone. There is deliberately no
    rank argument: DDP replicas must draw identical casts or they diverge, and a per-rank seed
    was written twice on 2026-09-25 (#717 build_optimizers, #714 rl_code_trainer).

    round() casts in flat blocks of `chunk_elems`: a stacked MoE optimizer group is ~2.0e9
    elements (12 layers x 48 experts of one same-shape weight), and _bf16_neighbors holds
    roughly fifteen x-sized temporaries, so casting the group at once needs tens of GiB at
    opt.step when the card is already near full. Blocking bounds the temporaries to the
    block (2e6 elements => a few tens of MiB each); element order is fixed, so a replay with
    the same seed/chunk reproduces byte-for-byte."""

    def __init__(self, seed=20260925, chunk_elems=2_000_000):
        self.seed = int(seed)
        self.chunk_elems = int(chunk_elems)
        self._gens = {}

    def generator(self, device):
        key = (device.type, device.index)
        if key not in self._gens:
            self._gens[key] = torch.Generator(device=device).manual_seed(self.seed)
        return self._gens[key]

    def round(self, x):
        n = x.numel()
        if n <= self.chunk_elems:
            return stochastic_round_bf16(x, self.generator(x.device))
        flat = x.reshape(-1)
        out = torch.empty_like(flat, dtype=torch.bfloat16, device=x.device)
        gen = self.generator(x.device)
        for lo in range(0, n, self.chunk_elems):
            blk = flat[lo : lo + self.chunk_elems]
            lower, upper, frac = _bf16_neighbors(blk)
            draw = torch.rand(blk.shape, dtype=torch.float32, device=x.device, generator=gen)
            out[lo : lo + self.chunk_elems] = torch.where(draw < frac, upper, lower).to(
                torch.bfloat16
            )
        return out.view_as(x)

    def apply(self, w, update):
        """New bf16 weight = stochastic_round_bf16(w - update), computed in fp32 BLOCKS.

        `w` is the bf16 parameter, `update` its bf16 Muon step (lr*X + lr*wd*w*mask). Both are
        sliced in their NATIVE dtype and promoted to fp32 INSIDE the block: promoting the whole
        ~2.0e9-element group up front would hold two ~8 GiB fp32 copies for the whole opt.step.
        As written, the only fp32 tensors alive are one block's."""
        n = w.numel()
        wf, uf = w.reshape(-1), update.reshape(-1)
        out = torch.empty(n, dtype=torch.bfloat16, device=w.device)
        gen = self.generator(w.device)
        for lo in range(0, n, self.chunk_elems):
            hi = lo + self.chunk_elems
            cand = wf[lo:hi].float() - uf[lo:hi].float()
            out[lo:hi] = stochastic_round_bf16(cand, gen).reshape(-1)
        return out.view_as(w)

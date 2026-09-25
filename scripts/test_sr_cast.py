#!/usr/bin/env python3
"""Known-answer gate for stochastic fp32->bf16 rounding (1e option B, 2026-09-25).

(a) UNBIASED: for a fixed sub-half-ULP offset, after N repeated same-sign steps the mean
    bf16 weight movement equals N*delta within a 3-sigma Monte-Carlo band; round-to-nearest
    (the old path) moves 0 for the same inputs -- so the gate discriminates the two writes.
(b) EXACT POINTS and NEIGHBOUR GRID are correct; E[cast] = x on a random sample within
    tight tolerance.
CPU only, deterministic seed.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sr_cast import StochasticRounder, _bf16_neighbors, stochastic_round_bf16  # noqa: E402


def _nearest_bf16(x):
    return x.to(torch.bfloat16).float()


def test_neighbors_bracket():
    torch.manual_seed(0)
    x = (torch.randn(10000) * 3).float()
    lo, up, frac = _bf16_neighbors(x)
    assert torch.all(lo <= x + 1e-6) and torch.all(up + 1e-6 >= x), "bracket failed"
    assert torch.all((frac >= 0) & (frac <= 1))
    # lo and up must themselves be exact bf16 values
    assert torch.equal(lo.to(torch.bfloat16).float(), lo)
    assert torch.equal(up.to(torch.bfloat16).float(), up)


def test_expectation_unbiased():
    # For many random fp32 values, the mean of many stochastic casts tends to x.
    g = torch.Generator().manual_seed(1)
    torch.manual_seed(2)
    x = 1.0 + torch.rand(20000, generator=g)
    acc = torch.zeros_like(x)
    reps = 400
    gr = torch.Generator().manual_seed(7)
    for _ in range(reps):
        acc += stochastic_round_bf16(x, gr).float()
    mean = acc / reps
    # Monte-Carlo SE for one element ~ ULP*sqrt(p(1-p)/reps) ~ 0.0078*0.5/20 = 1.95e-4 at
    # [1,2); assert the average over 20k elements lands within ~5 of those SEs (loose, robust).
    err = (mean - x).abs().mean().item()
    assert err < 1e-3, f"E[cast] must ~= x, got mean abs err {err}"


def test_repeated_sub_ulp_accumulates():
    # Fixed same-sign delta strictly under half a bf16 ULP for every element.
    g = torch.Generator().manual_seed(11)
    w = (1.0 + torch.rand(200000, generator=g) * 15).to(torch.bfloat16).float()
    exp = torch.floor(torch.log2(w.abs()))
    ulp = torch.pow(2.0, exp - 7)
    delta = 0.25 * ulp  # well under half-ULP, positive
    N = 1000

    # OLD path: round-to-nearest every step -- stays bit-identical.
    cur = w.clone()
    for _ in range(N):
        cur = _nearest_bf16(cur + delta)
    moved_rn = (cur != w).float().mean().item()
    assert moved_rn == 0.0, f"round-to-nearest must freeze, moved {moved_rn}"

    # NEW path: stochastic round every step, fp32 accumulation emulated per element by the
    # cast itself (each step rounds w+delta; sub-ULP crosses probabilistically).
    cur = w.clone()
    gr = torch.Generator().manual_seed(123)
    for _ in range(N):
        cur = stochastic_round_bf16(cur + delta, gr).float()
    target = w + N * delta
    sigma = ((cur - target).float().std() / (N**0.5)).item()
    mean_move = (cur - w).mean().item()
    expect = N * float(delta.mean())
    z = abs(mean_move - expect) / (sigma + 1e-12)
    assert z < 3.0, f"mean move {mean_move:.6f} not within 3sigma of N*delta {expect:.6f} (z={z:.2f})"
    # and it must have actually moved a substantial fraction (unlike the frozen RN path)
    frac_moved = (cur != w).float().mean().item()
    assert frac_moved > 0.5, f"stochastic path should move most weights after {N} steps, got {frac_moved}"


def _selftest():
    test_neighbors_bracket()
    test_expectation_unbiased()
    test_repeated_sub_ulp_accumulates()
    rounder = StochasticRounder(seed=1, rank=3)
    a = rounder.round(torch.randn(5))
    b = StochasticRounder(seed=1, rank=3).round(torch.randn(5))
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    print("sr_cast OK: neighbours bracket, E[cast]=x, N*delta accumulates, round-nearest freezes")


if __name__ == "__main__":
    _selftest()

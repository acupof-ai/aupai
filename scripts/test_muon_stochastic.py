#!/usr/bin/env python3
"""Known-answer gate for option B at the OPTIMIZER seam (1e ruling, 2026-09-25).

The library-level math is gated in test_sr_cast.py. This drives the real train.Muon.step and
StochasticAdamW.step and answers the question that test cannot: does the optimizer actually
write the stochastic value?

Three arms share one initial weight and one constant same-sign gradient over N=2000 steps:
  ORACLE  fp32 params, SR off            -> the exact fp32 trajectory (ground truth)
  FROZEN  bf16 params, SR off, bf16 mb   -> round-to-nearest, sub-half-ULP steps must move 0
  SR      bf16 params, SR on, fp32 mb    -> mean bf16 move within 3 sigma of the oracle

A deleted SR writeback makes SR bit-identical to FROZEN and fails two assertions; a wrong
momentum dtype is asserted directly. Same seed must replay bit-identically.
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch.set_num_threads(1)
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # exact equality eager; no compiler nondeterminism
from sr_cast import StochasticRounder  # noqa: E402
from train import Muon, StochasticAdamW  # noqa: E402

N = 2000
LR = 1e-3  # with NS-normalised X, per-element step ~2e-5 << half bf16 ULP (~2e-3) near w~1


def _run_muon(dtype, sr, seed=20260925, w0=None):
    torch.manual_seed(0)
    n, out, inn = 4, 64, 32
    if w0 is None:
        w0 = (1.0 + 0.5 * torch.rand(n, out, inn)).to(dtype)  # all in one ULP bin [1, 1.5)
    G0 = torch.ones(n, out, inn)  # constant positive grad -> NS keeps every entry same sign
    w = w0.to(dtype).clone()
    rounder = StochasticRounder(seed=seed, rank=0) if sr else None
    opt = Muon(
        [w],
        lr=LR,
        momentum=0.0,  # mb = grad, g = grad: constant update every step
        ns_steps=5,
        weight_decay=0.0,
        stochastic_round=sr,
        momentum_dtype=torch.float32 if sr else None,
        rounder=rounder,
    )
    for _ in range(N):
        w.grad = G0.to(dtype)
        opt.step()
    mb = opt.state[w]["mb"]
    return w0, w.detach(), mb.dtype


def _run_muon_multi(dtype, sr, seed=20260925, base=None):
    """SEPARATE same-shape parameters in one Muon -- exercises step()'s torch.stack cast site
    (the real MoE shape: dozens of params stacked), which the single-param arm never reaches."""
    torch.manual_seed(0)
    if base is None:
        base = (1.0 + 0.5 * torch.rand(3, 48, 64, generator=torch.Generator().manual_seed(9))).to(dtype)
    G0 = torch.ones(3, 48, 64)
    ws = [base[i].to(dtype).clone() for i in range(3)]
    rounder = StochasticRounder(seed=seed, rank=0) if sr else None
    opt = Muon(
        ws,
        lr=LR,
        momentum=0.0,
        ns_steps=5,
        weight_decay=0.0,
        stochastic_round=sr,
        momentum_dtype=torch.float32 if sr else None,
        rounder=rounder,
    )
    for _ in range(N):
        for w in ws:
            w.grad = G0[0].to(dtype)
        opt.step()
    return base, torch.stack(ws).detach()


def _run_adam(dtype, sr, seed=20260925, w0=None):
    torch.manual_seed(0)
    if w0 is None:
        w0 = (1.0 + 0.5 * torch.rand(128, 128)).to(dtype)
    G0 = 0.01 * torch.ones(128, 128)
    w = w0.to(dtype).clone()
    if sr:
        rounder = StochasticRounder(seed=seed, rank=0)
        opt = StochasticAdamW([w], rounder, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
    else:
        opt = torch.optim.AdamW([w], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
    for _ in range(N):
        w.grad = G0.to(dtype)
        opt.step()
    return w0, w.detach()


def main():
    # ── Muon ─────────────────────────────────────────────────────────────────────
    # All three arms start from the SAME bf16 grid values; the oracle carries them in fp32.
    base = (1.0 + 0.5 * torch.rand(4, 64, 32, generator=torch.Generator().manual_seed(3))).bfloat16()
    w0 = base
    _, w_oracle, _ = _run_muon(torch.float32, False, w0=base.float())
    _, w_frozen, mb_dt_frozen = _run_muon(torch.bfloat16, False, w0=base)
    _, w_sr, mb_dt_sr = _run_muon(torch.bfloat16, True, w0=base)

    # (b) discrimination: round-to-nearest freezes a sub-half-ULP stream completely.
    moved_frozen = (w_frozen != w0.bfloat16()).float().mean().item()
    assert moved_frozen == 0.0, f"round-to-nearest must freeze, moved {moved_frozen}"

    # fp32 momentum is part of the order; the default path keeps bf16.
    assert mb_dt_sr == torch.float32, f"SR mb must be fp32, got {mb_dt_sr}"
    assert mb_dt_frozen == torch.bfloat16

    # (a) unbiasedness: the SR mean move tracks the fp32 oracle within 3 sigma of the
    # stochastic spread, and it actually moves most elements.
    d_sr = w_sr.float() - w0.float()
    d_or = w_oracle - w0.float()
    mean_move, oracle_move = d_sr.mean().item(), d_or.mean().item()
    # Per-element SE of the mean over 8192 independent casts streams.
    se = d_sr.std().item() / (d_sr.numel() ** 0.5)
    z = abs(mean_move - oracle_move) / (se + 1e-12)
    assert z < 3.0, f"SR mean {mean_move:.6f} not within 3sigma of oracle {oracle_move:.6f} (z={z:.2f})"
    frac_moved = (w_sr != w0.bfloat16()).float().mean().item()
    assert frac_moved > 0.5, f"SR must move most weights, moved {frac_moved}"

    # Reproducibility: same seed, same casts, byte for byte.
    _, w_sr2, _ = _run_muon(torch.bfloat16, True, w0=base)
    assert torch.equal(w_sr, w_sr2), "same seed must replay identical weights"
    _, w_sr3, _ = _run_muon(torch.bfloat16, True, seed=20260926, w0=base)
    assert not torch.equal(w_sr, w_sr3), "different seeds must draw a different stream"

    # ── Stacked-group cast site (the real MoE: several same-shape params) ─────────
    mb = (1.0 + 0.5 * torch.rand(3, 48, 64, generator=torch.Generator().manual_seed(9))).bfloat16()
    _, wm_frozen = _run_muon_multi(torch.bfloat16, False, base=mb)
    _, wm_sr = _run_muon_multi(torch.bfloat16, True, base=mb)
    assert (wm_frozen != mb).float().mean().item() == 0.0, "stacked RN path must freeze"
    assert (wm_sr != mb).float().mean().item() > 0.5, "stacked SR path must move"
    _, wm_sr2 = _run_muon_multi(torch.bfloat16, True, base=mb)
    assert torch.equal(wm_sr, wm_sr2), "stacked path must replay bit-identically"
    _, wm_or = _run_muon_multi(torch.float32, False, base=mb.float())
    dm = wm_sr.float() - mb.float()
    zm = (dm.mean() - (wm_or - mb.float()).mean()).abs().item() / (
        dm.std().item() / (dm.numel() ** 0.5) + 1e-12
    )
    assert zm < 3.0, f"stacked SR mean off oracle, z={zm:.2f}"

    # ── StochasticAdamW ──────────────────────────────────────────────────────────
    # All three arms start from the SAME bf16 grid values; the oracle runs them in fp32.
    base = (1.0 + 0.5 * torch.rand(128, 128, generator=torch.Generator().manual_seed(5))).bfloat16()
    a0 = base
    a_frozen = _run_adam(torch.bfloat16, False, w0=base)[1]
    a_sr = _run_adam(torch.bfloat16, True, w0=base)[1]
    assert (a_frozen != a0).float().mean().item() == 0.0, "AdamW round-to-nearest must freeze"
    assert (a_sr != a0).float().mean().item() > 0.5, "StochasticAdamW must move"
    a_oracle = _run_adam(torch.float32, False, w0=base.float())[1]
    da = a_sr.float() - a0.float()
    za = (da.mean() - (a_oracle - a0.float())).mean().abs().item() / (
        da.std().item() / (da.numel() ** 0.5) + 1e-12
    )
    assert za < 3.0, f"AdamW SR mean off oracle, z={za:.2f}"

    print(
        "muon/adam stochastic-round OK: RN freezes, SR tracks fp32 oracle within 3sigma, "
        "mb fp32, replay bit-identical"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--selftest":
        raise SystemExit(f"unknown argument {sys.argv[1]!r}; this gate takes only --selftest")
    main()

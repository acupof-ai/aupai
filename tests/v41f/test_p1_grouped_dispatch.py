"""CPU parity skeleton for the GPU grouped-GEMM MoE dispatch (P4 pre-research).

The fast path does not exist yet and v41f/moe.py is unchanged. This file pins, on CPU, the
one fact the future GPU swap relies on: the grouped "flatten -> sort by expert -> three
grouped GEMMs -> fp32 weight -> index_add" expansion is numerically the same function as the
per-expert torch.where loop in v41f/moe.py. When the GPU path lands it replaces ONLY
_grouped_linear with torch._grouped_mm; everything else here is the production math.

Do NOT call torch._grouped_mm in this file: its CPU path is not the faithful kernel (a small
case diverged max_abs 8.2 from a manual per-group matmul). _grouped_linear below is the CPU
semantics of the grouped op -- contiguous per-group slices, expert order -- and is what the
where-loop is compared against. Design/contract: docs/standards/moe_grouped_mm_dispatch.md.

    python3 tests/v41f/test_p1_grouped_dispatch.py --selftest
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v41f.expert import Expert  # noqa: E402
from v41f.moe import _expert_weighted  # noqa: E402


def _grouped_linear(a, w_grouped, batch_sizes):
    """CPU semantics of the grouped expert GEMM: rows of `a` are contiguous per group in
    expert order; w_grouped is the stacked nn.Linear weight [E, N, K], and each group calls
    F.linear exactly as v41f.Expert does. Using F.linear (not a pre-transposed a@w[e]) is
    deliberate: a contiguous [E,K,N] operand makes the CPU bf16 GEMM round at a different
    point than the loop's weight.T and produces a layout artifact (measured max_abs 0.18),
    which is kernel-layout rounding, not dispatch math. This reference isolates GROUPING;
    the GPU torch._grouped_mm needs [E,K,N] and its own bf16 layout parity is prereg #3."""
    out = torch.empty(a.shape[0], w_grouped.shape[1], dtype=a.dtype)
    off = 0
    for e, count in enumerate(batch_sizes.tolist()):
        if count:
            out[off : off + count] = F.linear(a[off : off + count], w_grouped[e])
        off += count
    return out


def _stack_grouped(experts):
    """nn.ModuleList of Expert -> stacked {w1,w3,w2: [E,N,K]} in the native nn.Linear layout,
    so the CPU grouped reference invokes the same GEMM (same operand layout) as the loop."""
    return {
        name: torch.stack([getattr(e, name).weight.data for e in experts])
        for name in ("w1", "w3", "w2")
    }


def routed_loop(x, indices, weights, experts, shared, n_experts):
    """The current v41f/moe.py:79-89 routed+shared combine, factored identically."""
    n = x.shape[0]
    dim = x.shape[-1]
    y = torch.zeros(n, dim, dtype=torch.float32)
    counts = torch.bincount(indices.flatten(), minlength=n_experts).tolist()
    for i in range(n_experts):
        if counts[i] == 0:
            continue
        idx, top = torch.where(indices == i)
        y[idx] += _expert_weighted(experts[i], x[idx], weights[idx, top, None])
    y += shared(x).float()
    return y


def routed_grouped(x, indices, weights, w_grouped, shared, n_experts):
    """The grouped expansion. On GPU _grouped_linear becomes torch._grouped_mm; the sort,
    batch_sizes, fp32 weight-before-w2 point and index_add are identical then and now."""
    n, top_k = indices.shape
    dim = x.shape[-1]
    # stable: torch.where(indices==i) lists an expert's rows in flattened row-major order, so
    # the within-group token order must be stable, not argsort's default arbitrary tie order,
    # or bf16 rows are silently permuted.
    order = torch.argsort(indices.flatten(), stable=True)
    flat = indices.flatten()[order]
    tok, cho = order // top_k, order % top_k
    batch_sizes = torch.bincount(flat, minlength=n_experts).to(torch.int32)
    assert int(batch_sizes.sum()) == n * top_k
    x_g = x[tok]
    w_g = weights[tok, cho, None]
    g1 = _grouped_linear(x_g, w_grouped["w1"], batch_sizes).float()
    g3 = _grouped_linear(x_g, w_grouped["w3"], batch_sizes).float()
    h = w_g * (F.silu(g1) * g3)                       # routing weight in fp32 BEFORE w2
    gd = _grouped_linear(h.to(x.dtype), w_grouped["w2"], batch_sizes)
    y = torch.zeros(n, dim, dtype=torch.float32)
    y.index_add_(0, tok, gd.float())  # w2 already rounded to stream dtype; accumulate in fp32 like the loop's y[idx] +=
    y += shared(x).float()
    return y


def _world(n_tokens, dim, inter, n_experts, top_k, dtype, seed, force_all=True):
    torch.manual_seed(seed)
    experts = [Expert(dim, inter) for _ in range(n_experts)]
    shared = Expert(dim, inter)
    if dtype is torch.bfloat16:
        for e in experts + [shared]:
            for name in ("w1", "w3", "w2"):
                getattr(e, name).weight.data = getattr(e, name).weight.data.to(torch.bfloat16)
    x = torch.randn(n_tokens, dim, dtype=torch.float32).to(dtype)
    indices = torch.stack([torch.randperm(n_experts)[:top_k] for _ in range(n_tokens)])
    if force_all:
        # give each expert at least one slot WITHOUT repeating an expert within a token —
        # real top-k routing has distinct experts per token, and a repeated expert would
        # exercise an impossible case (advanced-index assign vs index_add sum) irrelevant here.
        for e in range(n_experts):
            if not (indices == e).any():
                placed = False
                for t in torch.randperm(n_tokens).tolist():
                    for k in range(top_k):
                        if e not in indices[t].tolist():
                            indices[t, k] = e
                            placed = True
                            break
                    if placed:
                        break
                assert placed, f"could not place expert {e} without a per-token duplicate"
    weights = torch.rand(n_tokens, top_k, dtype=torch.float32)
    return x, indices, weights, experts, shared


def test_grouped_allcloses_loop_fp32():
    x, idx, w, experts, shared = _world(11, 8, 16, 4, 2, torch.float32, seed=0)
    wg = _stack_grouped(experts)
    a, b = routed_loop(x, idx, w, experts, shared, 4), routed_grouped(x, idx, w, wg, shared, 4)
    d = (a - b).abs()
    assert torch.isfinite(b).all(), "non-finite grouped output"
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-3), f"fp32 max_abs={d.max().item()}"
    print(f"[ok ] fp32 grouped vs loop max_abs={d.max().item():.3e}")


def test_grouped_allcloses_loop_bf16():
    x, idx, w, experts, shared = _world(13, 16, 32, 5, 2, torch.bfloat16, seed=1)
    wg = _stack_grouped(experts)
    a, b = routed_loop(x, idx, w, experts, shared, 5), routed_grouped(x, idx, w, wg, shared, 5)
    d = (a - b).abs()
    assert torch.isfinite(b).all()
    # same atol as the MoE P0 allclose (bf16 math, weight-before-w2)
    assert torch.allclose(a, b, atol=2e-2, rtol=1e-3), f"bf16 max_abs={d.max().item()}"
    print(f"[ok ] bf16 grouped vs loop max_abs={d.max().item():.3e}")


def test_empty_expert_group_is_zero_contribution():
    # expert n_experts-1 is given ZERO rows by construction: both paths must agree and it
    # contributes nothing (the loop `continue`s; the grouped op carries a zero-length group).
    x, idx, w, experts, shared = _world(12, 8, 16, 4, 2, torch.float32, seed=2, force_all=False)
    # draw from experts 0..2 only: distinct per token (randperm), expert 3 deterministically empty
    idx = torch.stack([torch.randperm(3)[:2] for _ in range(idx.shape[0])])
    wg = _stack_grouped(experts)
    a = routed_loop(x, idx, w, experts, shared, 4)
    b = routed_grouped(x, idx, w, wg, shared, 4)
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-3), (a - b).abs().max().item()
    counts = torch.bincount(idx.flatten(), minlength=4).tolist()
    assert counts[3] == 0, counts
    print(f"[ok ] empty-group tolerated (group sizes {counts})")


def test_weight_after_w2_changes_numerics():
    """Mutation guard for the load-bearing order: multiplying the routing weight AFTER w2 is a
    different rounding point. In fp32 a scalar weight commutes through a linear to round-off,
    so this guard runs the PRODUCTION dtype bf16, where the P0 MoE gate priced the difference
    at up to 0.5 against the 2e-2 allclose (v41f/moe.py:19-31). A GPU port that moves the
    weight past w2 must fail here."""
    x, idx, w, experts, shared = _world(64, 16, 32, 8, 2, torch.bfloat16, seed=3)
    n, top_k = idx.shape
    order = torch.argsort(idx.flatten(), stable=True)
    flat = idx.flatten()[order]
    tok, cho = order // top_k, order % top_k
    bs = torch.bincount(flat, minlength=8).to(torch.int32)
    wg = _stack_grouped(experts)
    g1 = _grouped_linear(x[tok], wg["w1"], bs).float()
    g3 = _grouped_linear(x[tok], wg["w3"], bs).float()
    correct = _grouped_linear(
        (w[tok, cho, None] * (F.silu(g1) * g3)).to(torch.bfloat16), wg["w2"], bs)
    wrong = _grouped_linear((F.silu(g1) * g3).to(torch.bfloat16), wg["w2"], bs) \
        * w[tok, cho, None]
    diff = (correct - wrong).abs().max().item()
    # 0.003 at this small shape; the bound only proves the two orders are not the same
    # function (far above fp32 round-off), it is not a magnitude claim for the gate config.
    assert diff > 1e-3, f"weight-before/after-w2 must be numerically distinct in bf16, got {diff}"
    print(f"[ok ] weight-before vs after-w2 distinct in bf16 max_abs={diff:.3e}")


def _selftest():
    test_grouped_allcloses_loop_fp32()
    test_grouped_allcloses_loop_bf16()
    test_empty_expert_group_is_zero_contribution()
    test_weight_after_w2_changes_numerics()
    print("p1 grouped dispatch CPU parity OK: grouped expansion allcloses the per-expert loop")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())

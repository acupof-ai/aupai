"""P0 allclose harness: shared helpers for comparing a v41f module against the
vendored upstream reference on identical weights and inputs, CPU, bf16/fp32.

Conventions:
- compare per-op on the SAME tensors; copy reference weights into the v41f module
  (or vice-versa) by parameter name map, never rely on random init matching.
- tolerance is declared per op: pure elementwise/linear fp32 atol=1e-5; bf16 math
  atol=2e-2; anything touching a softmax/sinkhorn atol=5e-2 and also checks finite.
- every comparison reports (max_abs, max_rel, finite) so a loose tolerance cannot
  hide a wrong answer that merely stayed small.
"""
import torch


def cmp(name, got, want, atol, rtol=1e-3):
    got = got.detach().float().reshape(-1)
    want = want.detach().float().reshape(-1)
    assert got.shape == want.shape, f"{name}: shape {got.shape} vs {want.shape}"
    diff = (got - want).abs()
    max_abs = diff.max().item()
    denom = want.abs().clamp_min(1e-6)
    max_rel = (diff / denom).max().item()
    finite = torch.isfinite(got).all().item()
    ok = finite and torch.allclose(got, want, atol=atol, rtol=rtol)
    status = "ok " if ok else "FAIL"
    print(f"[{status}] {name:28s} max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
          f"finite={finite} (atol={atol})")
    assert ok, f"{name}: max_abs {max_abs} > {atol} or non-finite"
    return max_abs, max_rel


def seed_everything(seed=0):
    torch.manual_seed(seed)


HC_GRAD_RTOL = 3e-4
HC_GRAD_RTOL_SOFTCAP = 3e-3
GRAD_RTOL = 1e-4


def grad_rtol(name, hc=HC_GRAD_RTOL):
    """Relative tolerance for comparing one parameter's gradient between two implementations
    that are meant to be numerically equivalent.

    The hyperconnection coefficient parameters (hc_attn_fn/base/scale, hc_ffn_*) carry gradient
    norms of 1e-5..1e-4, and their path is pinned to fp32 inside hc_mixes with iterative Sinkhorn
    normalization, so a relative comparison of THEIR gradients measures float accumulation order
    rather than disagreement between the implementations. The measured healthy floor and the
    mutant signal that sets this threshold are facts/v41.json#v41.hc_grad_rel_floor_0930.

    hc raises the hc_* allowance for one comparison. A threshold's comparability is set by the
    set it acts on: attn_logit_softcap=0.5 puts tanh on the entry scores, and the fp32 rounding
    it adds to the LSE merge lifts the healthy hc floor about 4x, to 4.6e-4..5.4e-4 on pod x86
    against 1.1e-4..1.4e-4 uncapped. HC_GRAD_RTOL_SOFTCAP is that arm's own threshold, measured
    on both sides, with the resolution it loses stated: facts/v41.json#v41.hc_grad_rel_softcap_0930.
    """
    return (hc if ".hc.hc_" in name else GRAD_RTOL)

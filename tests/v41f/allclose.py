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

#!/usr/bin/env python3
"""CPU toy for ProductKeyMemory: d128, 64x64 = 4096 values, top_k 8. For tilerl's item 3
(torch.compile graph-break, FP8 exclusion at the module boundary) and item 1 (the lookup
kernel), runnable with no GPU, no fla, and no checkpoint.

WHAT IT IS FOR. tilerl owns the kernel and the sparse DDP gradient path and needs the module's
exact call shape without standing up a 200M model on a card. This builds the real
model.ProductKeyMemory -- not a reimplementation -- at 1/256 of M1's value count, so the shapes
are the arm's shapes and only the constants are small. A reimplemented toy would let the two
copies drift, and the thing being handed over is precisely the boundary between them.

WHY 64x64 AND NOT SOMETHING ROUNDER. side must satisfy top_k <= side (the module raises
otherwise: each half keeps top_k of its own sqrt(V) keys), and 4096 values at d128 is 512K
parameters, which builds in well under a second on a laptop.

    python3 probes/mem_toy.py            # shapes, the COO grad, the compile boundary
    python3 probes/mem_toy.py --compile  # also try torch.compile and report where it breaks
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ProductKeyMemory  # noqa: E402

D, SIDE, TOP_K = 128, 64, 8
B, T = 2, 16


def build(sparse=True):
    torch.manual_seed(0)
    return ProductKeyMemory(SIDE * SIDE, D, top_k=TOP_K, sparse=sparse)


def report_shapes():
    m = build()
    x = torch.randn(B, T, D)
    y = m(x)
    print(f"in  {tuple(x.shape)} -> out {tuple(y.shape)}   (identical shape: it is a residual)")
    print(f"keys      {tuple(m.keys.shape)}       2 sub-tables x side x key_dim (key_dim = d//2)")
    print(f"values    {tuple(m.values.weight.shape)}   sparse={m.values.sparse}")
    print(f"query     {tuple(m.query.weight.shape)}     one head, both halves at once")
    print(f"candidates per token: top_k*top_k = {TOP_K * TOP_K}, then top_k = {TOP_K} rows gathered")
    return m, x, y


def report_sparse_grad():
    """THE PRECONDITION FOR THE DDP PATH: the value table's grad is COO, holding only touched
    rows. Row count, not element count, is what an index exchange sends instead of the table."""
    m = build(sparse=True)
    m(torch.randn(B, T, D)).sum().backward()
    g = m.values.weight.grad
    assert g.is_sparse, "sparse=True must give a COO grad; a dense one defeats the whole path"
    rows = torch.unique(g._indices()[0]).numel()
    print(f"\nvalue grad: is_sparse={g.is_sparse} nnz={g._nnz()} distinct rows={rows}/{SIDE*SIDE} "
          f"({100*rows/(SIDE*SIDE):.1f}% of the table)")
    print(f"  dense equivalent would be {SIDE*SIDE*D} elements; the COO holds {g._nnz()*D}")
    # AdamW REFUSES a sparse grad, which is why the charter routes this group elsewhere. Shown
    # rather than asserted in prose: the exception is the reason for the Adagrad group.
    p = nn.Parameter(m.values.weight.detach().clone())
    p.grad = g.clone()
    try:
        torch.optim.AdamW([p], lr=1e-3).step()
        print("  AdamW accepted it -- UNEXPECTED, the optimizer choice needs re-deriving")
    except RuntimeError as e:
        print(f"  AdamW raises: {str(e).splitlines()[0][:90]}")
    torch.optim.Adagrad([p], lr=1e-3).step()
    print("  Adagrad steps on it (dense state, one moment)")
    m2 = build(sparse=False)
    m2(torch.randn(B, T, D)).sum().backward()
    print(f"  --no-mem_sparse for comparison: is_sparse={m2.values.weight.grad.is_sparse}")


def report_fp8_boundary():
    """FP8 EXCLUSION IS BY FQN, AND THE LEAF NAME IS NOT ENOUGH -- the thing to check with
    tilerl. train._fp8_ok is called as _fp8_ok(m, fqn.rsplit(".", 1)[-1]) (train.py:543), so
    inside the memory it sees `query`, `gate`, `out` -- names that carry no hint of belonging to
    the memory, and every one of them is 128x128 here, so `all(d % 16 == 0)` passes and torchao
    would convert them. Excluding the memory therefore needs the FULL fqn at that call site, not
    a name added to _fp8_ok's tuple."""
    m = build()
    print("\nlinear submodules as the fp8 filter sees them:")
    for fqn, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            leaf = fqn.rsplit(".", 1)[-1]
            ok16 = all(d % 16 == 0 for d in mod.weight.shape)
            print(f"  fqn={'mem.' + fqn:<12} leaf={leaf:<8} shape={tuple(mod.weight.shape)} "
                  f"16-aligned={ok16}  -> converted unless the filter sees the full fqn")


def report_compile():
    """topk + gather + index_put is the part to watch. Reported, not asserted: what breaks is a
    property of the installed torch, and hard-coding an expectation would go stale silently.

    THE TWO HALVES ARE REPORTED SEPARATELY because they fail for unrelated reasons and only the
    first is about this module. dynamo.explain says whether the PYTHON traces into one graph;
    running the compiled callable additionally exercises the BACKEND, and on a laptop that is
    Inductor's C++ codegen against the local toolchain. Measured 2026-09-05: 1 graph, 0 breaks,
    then `InductorError: CppCompileError` -- reporting that as a graph break would have sent
    tilerl after a defect in the memory layer that the trace says is not there.
    """
    m = build()
    x = torch.randn(B, T, D)
    try:
        import torch._dynamo as dynamo
        dynamo.reset()
        expl = dynamo.explain(m)(x)
        print(f"\ndynamo trace: {expl.graph_count} graph(s), {expl.graph_break_count} break(s)"
              f"   <- this is the part that is about the module")
        for r in getattr(expl, "break_reasons", [])[:6]:
            print(f"  break: {str(getattr(r, 'reason', r))[:140]}")
    except Exception as e:  # noqa: BLE001 -- reporting the failure IS the output here
        print(f"\ndynamo unavailable on this build: {type(e).__name__}: "
              f"{str(e).splitlines()[0][:160]}")
        return
    try:
        c = torch.compile(m)
        a, b = m(x), c(x)
        print(f"backend: compiled vs eager max|diff| = {(a - b).abs().max().item():.3e}")
    except Exception as e:  # noqa: BLE001
        print(f"backend: {type(e).__name__} -- the BACKEND, not a graph break; on a laptop this "
              f"is Inductor's C++ codegen against the local toolchain and says nothing about "
              f"this module. Re-run on the pod for the real answer.\n"
              f"  {str(e).splitlines()[0][:160]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compile", action="store_true", help="also probe torch.compile graph breaks")
    a = ap.parse_args()
    print(f"torch {torch.__version__}  ProductKeyMemory({SIDE*SIDE}, {D}, top_k={TOP_K})")
    report_shapes()
    report_sparse_grad()
    report_fp8_boundary()
    if a.compile:
        report_compile()


if __name__ == "__main__":
    main()

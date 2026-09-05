#!/usr/bin/env python3
"""The pre-flag memory forward, pinned, so "the defaults reproduce today's arm" is checkable.

WHY THIS EXISTS, AND WHY NOW. b0 is adding --mem_sel_lr and --mem_query_norm {none,l2,bn}. My
acceptance criterion as reviewer (4c, 2026-09-05) is that the DEFAULTS reproduce today's arm
bit-for-bit. That is a claim about the tree as it stands BEFORE the flags land, and after they
land there is nothing left to compare against -- "today's arm" is then whatever the new default
branch computes. So the reference has to be taken now, on origin/main f6611742, or the criterion
is unfalsifiable by the time it is testable.

WHAT IS PINNED, AND WHY NOT A GOLDEN VECTOR. Not the output values: this runs on CPU here and on
CUDA on the pod, and a bf16 autocast matmul does not agree bit-for-bit across those. What is
pinned is the SELECTION and the WEIGHTS -- which rows the lookup reaches (`flat`) and the softmax
over the combined scores -- at fp32 on a seeded module, plus the read those two produce. Those are
the three things a query-norm or a selection-lr change moves, and they are exactly what the two
new flags touch. A golden file of output floats would be pinned to this machine; a recomputation
of the same arithmetic from the module's own parameters is pinned to the ALGORITHM, which is the
subject of the review.

THE ORACLE IS AN INDEPENDENT RE-IMPLEMENTATION, not a call into the module under test. Calling
ProductKeyMemory.forward and comparing it with itself would pass under every change to it, which
is the failure mode this file is supposed to catch. The oracle below is written from the product-
key definition (two halves score sqrt(V) keys each, the Cartesian sums pick top_k) and it does NOT
share code with model.py, so a change to either side shows up as a disagreement.

HOW TO USE IT AFTER THE FLAGS LAND: run it unchanged. If the defaults are what b0 says they are,
it passes untouched. If it fails, the defaults moved the arm, and the failure names which of the
three (selection / weights / read) moved. A green run is the review's evidence; my reading of the
diff is not.

    python3 scripts/test_mem_defaults_frozen.py
"""
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def oracle(mem, x, l2=False):
    """Product-key lookup, written from the definition rather than from model.py.

    Returns (rows, weights, read) at fp32. The dtype is deliberate: the module computes the
    softmax at fp32 already, and doing the scores at fp32 here keeps the comparison about the
    ALGORITHM rather than about autocast, which differs between this laptop and the pod.

    `l2` computes what --mem_query_norm=l2 would select. Check 4 needs BOTH answers: "the module
    matches the plain oracle" alone cannot tell a working default from a build where the two
    happen to coincide."""
    B, T, d = x.shape
    h = mem.n_mem(x)
    q = mem.query(h).view(B * T, 2, mem.key_dim).float()
    if l2:
        q = F.normalize(q, dim=-1)
    k = mem.keys.float()
    s0, s1 = q[:, 0] @ k[0].T, q[:, 1] @ k[1].T
    v0, i0 = s0.topk(mem.top_k, dim=-1)
    v1, i1 = s1.topk(mem.top_k, dim=-1)
    # The Cartesian combine: the full key is the concatenation of the two halves, so the score of
    # a pair is the sum of the two half-scores.
    cand = (v0[:, :, None] + v1[:, None, :]).reshape(B * T, mem.top_k * mem.top_k)
    idx = (i0[:, :, None] * mem.side + i1[:, None, :]).reshape(B * T, mem.top_k * mem.top_k)
    w, sel = cand.topk(mem.top_k, dim=-1)
    rows = idx.gather(1, sel)
    w = torch.softmax(w, dim=-1)
    vals = mem.values.weight.detach()[rows.reshape(-1)].float().view(B * T, mem.top_k, d)
    read = torch.einsum("nkd,nk->nd", vals, w).view(B, T, d)
    return rows, w, read


def build(seed=7, d=64, n_values=256, top_k=4):
    """A seeded ProductKeyMemory at fp32. Small enough to be a laptop self-check, and every
    dimension is one the real arm also has (a square value count, top_k <= sqrt(V))."""
    from model import ProductKeyMemory

    torch.manual_seed(seed)
    return ProductKeyMemory(n_values, d, top_k=top_k)


def main():
    bad = 0
    mem = build()
    mem.eval()  # the bookkeeping branch runs either way; eval keeps the check about the read
    torch.manual_seed(11)
    x = torch.randn(2, 5, 64)

    rows, w, read = oracle(mem, x)

    # 1. THE MODULE'S OWN OUTPUT, against the oracle's read pushed through the same gate. This is
    # the end-to-end statement: the value a block adds to its residual is what the product-key
    # definition says it should be.
    with torch.no_grad():
        got = mem(x)
        h = mem.n_mem(x)
        want = mem.out(F.silu(mem.gate(h)) * read.to(x.dtype))
    ok = torch.allclose(got, want, atol=1e-5)
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the module's read matches an independent product-key "
          f"oracle (max|d| = {(got - want).abs().max():.2e})")

    # 2. THE SELECTION ITSELF, recorded as a number a later run can compare against. `touched`
    # is the module's own record of which rows the lookup reached, and the oracle predicts the
    # same set -- so this asserts the ROWS, not just the arithmetic over them. --mem_query_norm
    # changes the query, which changes exactly this.
    #
    # RUN IN EVAL, and the mode is load-bearing rather than incidental: the bookkeeping block is
    # guarded by `not self.training or torch.is_grad_enabled()`, so under train() inside
    # no_grad() BOTH terms are false and `touched` stays empty -- 0 rows, which reads exactly
    # like a lookup that reached nothing. eval() makes the first term true. (Measured here: the
    # train()+no_grad() form reported 0 touched rows against 39 selected.)
    mem.touched.zero_()
    with torch.no_grad():
        mem(x)
    reached = set(torch.nonzero(mem.touched).flatten().tolist())
    predicted = set(rows.reshape(-1).tolist())
    ok = reached == predicted
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the rows the module touches are the rows the oracle "
          f"selects ({len(reached)} rows, {len(reached ^ predicted)} differing)")

    # 3. THE WEIGHTS ARE A SOFTMAX OVER THE SELECTED PAIRS -- normalised, and NOT uniform. The
    # second half is the one that matters: a query-norm bug that collapses every score to the
    # same value still produces a valid normalised distribution, and a test that only checked
    # sum==1 would pass through it. Measured on this seed, the spread is real.
    sums = w.sum(-1)
    spread = (w.max(-1).values - w.min(-1).values).min().detach()
    ok = torch.allclose(sums, torch.ones_like(sums), atol=1e-6) and float(spread) > 1e-3
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the read weights are a softmax and are not uniform "
          f"(min within-token spread {float(spread):.4f})")

    # 4. THE MODULE IS NOT L2-NORMALISING TODAY, asserted against an oracle that DOES. The
    # obvious form of this check -- measure ‖q‖ on `mem.query(...)` and assert it is not 1 --
    # is blind to the mutation it names: normalisation applied INSIDE forward leaves the raw
    # projection untouched, so that check stayed green under a forward that l2-normalised
    # (measured 2026-09-05, world A). What discriminates is running the oracle BOTH ways and
    # asserting the module agrees with the unnormalised one and differs from the normalised one.
    # The second half is the real content: without it a build where l2 happened to change
    # nothing would read as a pass.
    rows_l2, _, _ = oracle(mem, x, l2=True)
    plain_rows = set(rows.reshape(-1).tolist())
    l2_rows = set(rows_l2.reshape(-1).tolist())
    # `reached`, not `predicted`: comparing the plain oracle against itself would be a tautology
    # that never looks at the module (caught here 2026-09-05 -- the first form did exactly that
    # and stayed green under world A). The second clause is what makes the first one mean
    # something: on this seed l2 really does select different rows, so agreeing with plain is a
    # discriminating statement rather than a coincidence.
    ok = reached == plain_rows and l2_rows != plain_rows
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the module selects the UNNORMALISED query's rows, and "
          f"l2 would select {len(l2_rows ^ plain_rows)} different ones; "
          f"--mem_query_norm=none must keep it so")

    # 5. AND NO LEARNED TEMPERATURE: the scores enter the softmax with coefficient 1. A scale
    # parameter added for the l2 arm must not exist, or must be exactly 1.0, in the default. Named
    # explicitly because card 3's cell is "l2 + learned temperature" and the flag's default has to
    # leave that machinery inert rather than merely initialised near one.
    scales = [n for n, _ in mem.named_parameters()
              if any(t in n for t in ("temp", "scale", "logit_scale"))]
    extra = [n for n, _ in mem.named_parameters()
             if n not in {"query.weight", "keys", "values.weight", "gate.weight", "out.weight",
                          "n_mem.g"}]
    ok = not scales and not extra
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the default module has no temperature/scale parameter "
          f"and no parameter beyond the six{'' if ok else f': {scales or extra}'}")

    # 6. THE OPTIMIZER GROUPING, which --mem_sel_lr moves. Today keys AND values sit in ONE group
    # at mem_lr; a selection lr splits the keys out. The default must leave one group holding
    # both, so this records the pre-flag shape: a single mem group, and the keys inside it.
    try:
        import train
        n_mem_groups, keys_in_mem = _group_shape(train)
        ok = n_mem_groups == 1 and keys_in_mem
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BUG '} the keys and the table share ONE optimizer group "
              f"today ({n_mem_groups} mem group(s), keys inside = {keys_in_mem})")
    except Exception as e:  # noqa: BLE001 -- an import failure here is a real finding, not a skip
        bad += 1
        print(f"  BUG  could not read the optimizer grouping: {type(e).__name__}: {e}")

    n = 6
    print(f"test_mem_defaults_frozen: {n - bad}/{n} pass")
    return 1 if bad else 0


def _group_shape(train):
    """How many optimizer groups hold memory parameters, and are the keys among them.

    Built on a REAL model rather than the bare module, because the grouping is decided by
    build_optimizers over named_parameters() of the whole net -- _is_mem_fqn matches on the
    A SUBCLASS, NOT setattr ON train.Cfg. Cfg is a class whose fields are class attributes, and
    copy.deepcopy returns a class unchanged -- so assigning to it would edit the real Cfg for
    every later import in this process."""
    cfg = type("CfgFrozenCheck", (train.Cfg,), dict(
        d=64, heads=8, layers=4, ffn_hidden=128, vocab=100, seq=16, attn_every=4,
        mem_values=256, mem_top_k=4, mem_layers="1", mem_sparse=False,
        attn_res=False, grad_ckpt=False, head_mixed=0,
    ))
    m = train.HybridLM(cfg)
    opts = train.build_optimizers(m, cfg)
    mem_params = {id(p) for n, p in m.named_parameters() if train._is_mem_fqn(n)}
    key_id = id(m.memory.keys)
    holding = [o for o in opts
               if any(id(p) in mem_params for g in o.param_groups for p in g["params"])]
    keys_in = any(id(p) == key_id for o in holding for g in o.param_groups for p in g["params"])
    return len(holding), keys_in


if __name__ == "__main__":
    # --selftest is the hook's calling convention; an unknown flag must fail rather than run the
    # checks and report a pass for a call nobody meant to make.
    if len(sys.argv) > 1 and sys.argv[1:] != ["--selftest"]:
        sys.exit(f"usage: {os.path.basename(__file__)} [--selftest]  (got {sys.argv[1:]})")
    sys.exit(main())

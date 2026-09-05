#!/usr/bin/env python3
"""The pre-MoE FFN and optimizer grouping, pinned, so "--moe defaults are bit-for-bit off" is
checkable after the flags land.

WHY THIS EXISTS AND WHY IT IS COMMITTED ALONE, BEFORE THE MODULE. 4c's acceptance criterion for
the MoE arm (2026-09-05) is that the defaults reproduce today's dense arm bit-for-bit. That is a
claim about the tree as it stands BEFORE `--moe_experts` exists, and once the module lands there
is nothing left to compare against -- "today's arm" becomes whatever the new default branch
computes. Same reasoning tilerl wrote into test_mem_defaults_frozen.py:5-11 for the memory flags,
and the same reason this file is its own commit: a reference taken in the same commit as the
change it is meant to detect is not a reference.

WHAT IS PINNED. Not output floats -- this runs on CPU here and CUDA on the pod, and a bf16
autocast matmul does not agree bit-for-bit across those. Pinned instead:
  1. The FFN's FUNCTION, recomputed from the K3 SiTU-GLU definition by an independent oracle.
  2. The FFN's parameter set and shapes -- exactly w13 and w2, no router, no expert stack.
  3. The optimizer grouping: the FFN matrices are in MUON and no separate router group exists.
Those are the three things the MoE flags move.

THE ACTIVATION IS THE POINT OF CHECK 1, and it is not a formality. model.SwiGLU is NOT a plain
SwiGLU: it is K3 SiTU-GLU, `beta1*tanh(a/beta1)*sigmoid(b)` then `beta2*tanh(w2(gate)/beta2)`
with beta1=4.0, beta2=25.0 (model.py:339-346). An expert that computes the textbook
`a*sigmoid(b)` instead differs from the control in ACTIVATION as well as in sparsity, and
readout 1's delta would then be unattributable to the sparsity it is registered to measure.
That defect was found in scripts/moe_dispatch_bench.py before it reached the module (b0
2026-09-05, both dispatch paths ran plain SwiGLU while the dense baseline ran SiTU-GLU, so the
timed 0.66x was a ratio between two different functions). This check is what makes the same
mistake in the module impossible to land green.

THE ORACLE IS AN INDEPENDENT RE-IMPLEMENTATION, not a call into SwiGLU. Comparing the module
against itself passes under every change to it, which is the failure this file exists to catch.

HOW TO USE IT AFTER THE FLAGS LAND: run it unchanged. If the defaults are off, it passes
untouched. If it fails, the defaults moved the dense arm, and the failure names which of the
three moved.

    python3 scripts/test_moe_defaults_frozen.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

D, FFN = 64, 128


def oracle(ffn, x):
    """K3 SiTU-GLU written from the definition rather than from model.py.

    Uses the module's WEIGHTS (there is no other source of truth for them) but none of its code,
    so a change to the activation shows up as a disagreement. fp32 throughout: the comparison is
    about the algorithm, not about autocast.
    """
    w13 = ffn.w13.weight.detach().float()
    w2 = ffn.w2.weight.detach().float()
    b1, b2 = float(ffn.beta1), float(ffn.beta2)
    h = x.float() @ w13.T
    a, b = h.chunk(2, dim=-1)
    gate = b1 * torch.tanh(a / b1) * torch.sigmoid(b)
    return b2 * torch.tanh((gate @ w2.T) / b2)


def plain_swiglu_oracle(ffn, x):
    """The TEXTBOOK SwiGLU -- `a * sigmoid(b)`, no tanh, no betas.

    Check 1's second half needs this. "The module matches the SiTU-GLU oracle" alone cannot
    distinguish a correct module from one where the two happen to coincide; asserting the module
    also DIFFERS from the plain form is what makes the first half discriminating. If these two
    ever agree on this seed, check 1 has stopped being able to see the defect it exists for.
    """
    w13 = ffn.w13.weight.detach().float()
    w2 = ffn.w2.weight.detach().float()
    h = x.float() @ w13.T
    a, b = h.chunk(2, dim=-1)
    return (a * torch.sigmoid(b)) @ w2.T


def main():
    bad, n = 0, 4
    from model import SwiGLU

    class _C:
        d, ffn_hidden = D, FFN

    torch.manual_seed(7)
    ffn = SwiGLU(_C())
    ffn.eval()
    torch.manual_seed(11)
    x = torch.randn(2, 5, D)

    # 1. THE FUNCTION, against an independent SiTU-GLU oracle AND against the plain SwiGLU it is
    # not. The second clause is the content: it fails if an expert implementation drops the two
    # tanh bounds, which is the specific error already caught once in the dispatch bench.
    with torch.no_grad():
        got = ffn(x)
    want = oracle(ffn, x)
    plain = plain_swiglu_oracle(ffn, x)
    d_situ = float((got.float() - want).abs().max())
    d_plain = float((got.float() - plain).abs().max())
    ok = d_situ < 1e-5 and d_plain > 1e-3
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the FFN is K3 SiTU-GLU, not plain SwiGLU "
          f"(max|d| to SiTU-GLU {d_situ:.2e}, to plain SwiGLU {d_plain:.2e} -- the second must "
          f"be LARGE or this check cannot see a dropped activation)")

    # 2. THE PARAMETER SET. Two matrices, named and shaped. An expert stack (E, N, K), a router,
    # or a per-expert bias vector appearing here means the default constructed MoE machinery.
    names = {k: tuple(v.shape) for k, v in ffn.named_parameters()}
    want_names = {"w13.weight": (2 * FFN, D), "w2.weight": (D, FFN)}
    ok = names == want_names
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the default FFN has exactly w13 and w2"
          f"{'' if ok else f' -- got {names}'}")

    # 3. NO BETA IS A PARAMETER. They are python floats (model.py:339-340), so they carry no
    # gradient and appear in no optimizer group. If an MoE refactor turned them into buffers or
    # parameters the state dict would change shape and every existing checkpoint would need a
    # load-path branch -- worth pinning before, not discovering after.
    extras = [k for k, _ in ffn.named_parameters() if k not in want_names]
    bufs = list(ffn.named_buffers())
    ok = not extras and not bufs
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the betas are plain scalars: no extra parameter, no "
          f"buffer{'' if ok else f' -- params {extras}, buffers {[b[0] for b in bufs]}'}")

    # 4. THE OPTIMIZER GROUPING. Today every FFN matrix is 2D and lands in MUON by the
    # `p.ndim == 2` branch (train.py:1069), and NO group holds a router. --moe_router_lr must add
    # an AdamW group for the router while leaving the experts in Muon exactly as the dense FFN
    # they replace (4c ruling (f)), so this records the pre-flag shape: FFN in Muon, no router
    # anywhere. Built on a REAL model because the grouping is decided by build_optimizers over
    # named_parameters() of the whole net, not by the module in isolation.
    try:
        import train
        in_muon, n_router_groups, router_names = _group_shape(train)
        ok = in_muon and n_router_groups == 0 and not router_names
        bad += 0 if ok else 1
        # EVERY CLAUSE IN THE MESSAGE. The first version printed only in_muon and the group count,
        # so mutant M4 (a default nn.Linear router in Block) failed the check while the message
        # read `ffn_in_muon=True, router groups=0` -- both halves looking correct while the third
        # clause was what fired. A failure whose message does not name the clause that fired sends
        # the reader to the wrong place.
        print(f"  {'ok  ' if ok else 'BUG '} every FFN matrix is in Muon today and no router "
              f"exists (ffn_in_muon={in_muon}, router optimizer groups={n_router_groups}, "
              f"router/expert params={router_names or 'none'})")
    except Exception as e:  # noqa: BLE001 -- an import failure here is a finding, not a skip
        bad += 1
        print(f"  BUG  could not read the optimizer grouping: {type(e).__name__}: {e}")

    print(f"test_moe_defaults_frozen: {n - bad}/{n} pass")
    return 1 if bad else 0


def _group_shape(train):
    """Are the FFN matrices in Muon, and does any group hold a router.

    A SUBCLASS, NOT setattr ON train.Cfg: Cfg's fields are class attributes and assigning to it
    would edit the real Cfg for every later import in this process (the trap
    test_mem_defaults_frozen.py:_group_shape documents).

    CONSTRUCTION ONLY, no forward: chunk_kda and l2norm are Triton with no CPU fallback, so this
    model cannot forward here. The grouping is decided at construction, so that costs nothing.
    """
    cfg = type("CfgMoeFrozenCheck", (train.Cfg,), dict(
        d=64, heads=8, layers=4, ffn_hidden=FFN, vocab=100, seq=16, attn_every=4,
        mem_values=0, attn_res=False, grad_ckpt=False, head_mixed=0,
    ))
    m = train.HybridLM(cfg)
    opts = train.build_optimizers(m, cfg)
    ffn_ids = {id(p) for n, p in m.named_parameters() if ".ffn." in n}
    assert ffn_ids, "no .ffn. parameters found -- the fqn shape changed and this check is blind"
    muon = [o for o in opts if type(o).__name__ == "Muon"]
    muon_ids = {id(p) for o in muon for g in o.param_groups for p in g["params"]}
    all_ids = {id(p) for o in opts for g in o.param_groups for p in g["params"]}
    # NO COUNT-BASED FALLBACK. The first version of this check fell back to "Muon holds at least
    # as many params as the FFN has" when the ids did not line up, and that fallback SURVIVED the
    # mutation it exists to catch: routing every .ffn. matrix out of Muon left the check green,
    # because Muon still holds 21 params against the FFN's 8 and the count says nothing about
    # WHICH. Measured 2026-09-05 -- mutant M3 (`(scalar if ".ffn." in n else muon).append(q)`)
    # passed 4/4. So identity is required, and if it is unavailable this check FAILS loudly
    # rather than answering a weaker question: under --fp32_master the optimizer holds master
    # copies, not the model's tensors, and a check that cannot see the FFN must say so.
    if not ffn_ids <= all_ids:
        raise AssertionError(
            f"the optimizers do not hold the model's own FFN tensors "
            f"({len(ffn_ids & all_ids)}/{len(ffn_ids)} by identity) -- probably an fp32-master "
            f"path. This check cannot answer the grouping question that way; fix the fixture "
            f"rather than weakening the assertion")
    in_muon = ffn_ids <= muon_ids
    router_names = [n for n, _ in m.named_parameters()
                    if "router" in n or "moe" in n.lower() or "expert" in n]
    n_router_groups = sum(
        1 for o in opts for g in o.param_groups
        if g.get("name", "") in ("router", "moe_router")
    )
    return in_muon, n_router_groups, router_names


if __name__ == "__main__":
    # --selftest is the hook's calling convention; an unknown flag must fail rather than run the
    # checks and report a pass for a call nobody meant to make.
    if len(sys.argv) > 1 and sys.argv[1:] != ["--selftest"]:
        sys.exit(f"usage: {os.path.basename(__file__)} [--selftest]  (got {sys.argv[1:]})")
    sys.exit(main())

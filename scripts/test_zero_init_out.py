#!/usr/bin/env python3
# restartable: builds two tiny CPU models and runs four steps. Seconds; an interrupt costs
# nothing worth sharding.
"""A/B (3) `--zero_init_out`: the arm must differ from the baseline, and only where intended.

    python3 scripts/test_zero_init_out.py        # exit 1 on any difference from the contract

WHY THIS EXISTS. Two ways this arm can cost a full run and report nothing:

  1. IT ZEROES NOTHING. `o` and `w2` are matched by NAME. Rename either one and the arm
     becomes a bit-exact copy of the baseline that still logs as the arm -- the most
     expensive failure available here, since it spends four cards to learn nothing. model.py
     asserts the count, and this test asserts the assert fires (a renamed module raises).

  2. IT LOOKS BROKEN AND IS NOT. With every output projection zeroed, `w13` and `qkv` get
     EXACTLY zero gradient on step 0 -- the sublayer output is 0 * x, so nothing upstream of
     the zeroed matrix is reachable. That reads exactly like a dead branch, and the honest
     reaction to seeing it is to kill the run. Measured here: it recovers at step 1 (w13
     1.24e-02, qkv 7.83e-03) because w2 itself has a gradient at step 0 and moves off zero.
     So the contract is "zero at step 0, nonzero at step 1", and this test pins BOTH halves.
     Without the second half, a real dead branch would be indistinguishable from this.
"""
import sys

import torch
import torch.nn as nn

sys.path.insert(0, __file__.rsplit("/", 2)[0])

import model as M  # noqa: E402
import train  # noqa: E402


def build(zero):
    """A 4-layer CPU model, small but containing both block kinds and a real FFN."""
    # UNCONDITIONAL stub, not `if chunk_kda is None`. This test runs on CPU, and on the pod
    # fla IS importable, so the conditional left the real Triton kernel in place and it died
    # with "Pointer argument cannot be accessed from Triton (cpu tensor?)". The kernel is not
    # what this test covers -- zero-init is -- so the stand-in is always installed. Shape and
    # dtype preserving, the same stand-in test_arch_compat.py:33 uses.
    M.chunk_kda = lambda q, k, v, **kw: (q * 0 + v, None)  # noqa: E731
    # flash_attn is a SECOND pod-only path. It IS importable there, so model.HAS_FA is True and
    # the MLA block calls flash_attn_func, which asserts fp16/bf16/fp8 and rejects the fp32 this
    # test builds ("inputs must be float16, bfloat16, ..."). The gradient contract under test is
    # dtype-independent, so HAS_FA is forced off and the SDPA fallback runs -- ~20x slower per
    # step and irrelevant at this size. Forced, not conditional: the conditional stub above is
    # exactly the bug this line is fixing, green on a dev box and red on the pod.
    M.HAS_FA = False
    cfg = train.Cfg
    cfg.d, cfg.layers, cfg.heads, cfg.ffn_hidden = 128, 4, 4, 256
    cfg.vocab, cfg.vocab_real = 512, 500
    cfg.zero_init_out = zero
    torch.manual_seed(0)
    return M.HybridLM(cfg).train(), cfg


def out_projections(m):
    return [(n, mod) for n, mod in m.named_modules()
            if isinstance(mod, nn.Linear) and (n.endswith(".o") or n.endswith(".w2"))]


def grads(m, x, steps, lr=1e-3):
    """Per-step grad absmax for the three tensors that tell the story."""
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    names = ("blocks.0.ffn.w13.weight", "blocks.0.mixer.qkv.weight", "blocks.0.ffn.w2.weight")
    hist = []
    for _ in range(steps):
        out = m(x)
        lg = out[0] if isinstance(out, tuple) else out
        loss = -lg.float().log_softmax(-1)[:, :, 0].mean()
        opt.zero_grad()
        loss.backward()
        d = dict(m.named_parameters())
        hist.append({n.split(".")[-2]: float(d[n].grad.abs().max().detach()) for n in names})
        opt.step()
    return hist


def main():
    fails = []
    x = torch.randint(0, 500, (2, 16))

    base, cfg = build(False)
    arm, _ = build(True)

    # 1. the baseline must zero NOTHING, or the arm is not an arm.
    zb = [n for n, mod in out_projections(base) if float(mod.weight.abs().max().detach()) == 0]
    if zb:
        fails.append(f"baseline already has zeroed output projections: {zb}")

    # 2. the arm must zero every one of them, and the count must be 2 per layer.
    proj = out_projections(arm)
    za = [n for n, mod in proj if float(mod.weight.abs().max().detach()) == 0]
    if len(za) != len(proj) or len(za) != 2 * cfg.layers:
        fails.append(f"arm zeroed {len(za)} of {len(proj)} projections, expected {2 * cfg.layers}")

    # 3. nothing ELSE may be zeroed. A blanket zero-init would also pass check 2.
    others = [n for n, p in arm.named_parameters()
              if p.ndim == 2 and not (n.endswith(".o.weight") or n.endswith(".w2.weight"))
              and float(p.abs().max().detach()) == 0]
    # dyn[1] is zero-init by design (model.py) and the vocab padding rows are zeroed, but
    # those are ROWS of a nonzero tensor, so a whole-tensor check does not see them.
    others = [n for n in others if ".dyn." not in n]
    if others:
        fails.append(f"the arm zeroed tensors it should not have: {others}")

    # 4. THE STEP-0 / STEP-1 CONTRACT. Zero upstream gradient at step 0 is expected; still
    #    zero at step 1 would be a real dead branch and must be distinguishable.
    h = grads(arm, x, 2)
    if h[0]["w13"] != 0.0 or h[0]["qkv"] != 0.0:
        fails.append(f"step 0 upstream grads are not zero: {h[0]} -- the arm is not zeroing "
                     f"the output projections, or the graph changed")
    if h[1]["w13"] <= 0.0 or h[1]["qkv"] <= 0.0:
        fails.append(f"step 1 upstream grads are STILL zero: {h[1]} -- this IS a dead branch, "
                     f"not the expected one-step delay; do not spend cards on this arm")
    if h[0]["w2"] <= 0.0:
        fails.append(f"w2 has no gradient at step 0 ({h[0]['w2']}), so nothing can ever move "
                     f"off zero and every upstream matrix is dead forever")

    # 5. and the baseline has NO such hole, which is what makes 4 a property of the arm.
    hb = grads(base, x, 1)
    if hb[0]["w13"] <= 0.0 or hb[0]["qkv"] <= 0.0:
        fails.append(f"baseline also has zero upstream grads at step 0 ({hb[0]}), so check 4 "
                     f"is not testing the arm")

    for f in fails:
        print(f"  FAIL {f}")
    if fails:
        print(f"\n{len(fails)} failure(s)")
        return 1
    print(f"zero_init_out OK: {len(za)} projections zeroed (2 x {cfg.layers} layers), baseline "
          f"zeroes none, nothing else zeroed; upstream grad 0 at step 0 -> "
          f"w13 {h[1]['w13']:.2e} / qkv {h[1]['qkv']:.2e} at step 1 (recovers, not dead)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

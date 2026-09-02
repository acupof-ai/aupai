#!/usr/bin/env python3
# restartable: two tiny optimizer steps on CPU. Seconds; nothing to shard.
"""A/B (2a) `--muon_shape_lr`: the factor is applied, is per SHAPE, and the baseline is untouched.

    python3 scripts/test_muon_shape_lr.py       # exit 1 on any difference from the contract

WHY THIS EXISTS. The factor is max(1, out/in)^0.5 on PyTorch's [out, in] weight, and there are
two silent ways to get it wrong, both of which still train and still report as the arm:

  1. THE RATIO INVERTED. in/out instead of out/in shrinks exactly the matrices the record says
     to grow, and the clamp then hides it: every affected tensor gets 1.0, so the arm becomes a
     bit-exact baseline. This test pins the DIRECTION by asserting the update on a tall matrix
     grows and on a wide one does not move at all.

  2. THE FACTOR SHARED ACROSS SHAPES. Muon batches params by shape and multiplies one scalar
     per batch, so a factor computed once per GROUP (rather than per shape) would give every
     tensor the first shape's number. Asserted by requiring two different shapes in one group
     to receive two different multipliers.

Measured factors on the real 200M config (d=1024, layers=12, heads=8, ffn_hidden=3072), for the
record and because the affected set is the load-bearing part of the hypothesis:

    kv_up   [2048, 256]    2.8284      qg      [2048, 1024]   1.4142
    w13     [6144, 1024]   2.4495      gb      [1040, 1024]   1.0078
    qkv     [3072, 1024]   1.7321      o/w2/kv_down           1.0000

36 of 63 Muon tensors move; the other 27 are wide or square and get exactly 1.0.
"""
import sys

import torch

# Muon.step compiles muon_update, and a dev box may have no working C compiler for inductor.
# The factor under test is a scalar multiply on lr, identical eager or compiled, so running
# eager here tests the arithmetic and not the backend. Set before importing train.
torch._dynamo.config.suppress_errors = True
torch.compiler.reset()

sys.path.insert(0, __file__.rsplit("/", 2)[0])

import train  # noqa: E402

# Eager: see the note above. torch.compile(fn) still returns a callable that falls back.
torch._dynamo.config.disable = True


def make(shape):
    """A parameter and grad determined ONLY by the shape, never by call order.

    The first version of this test seeded once and then created params in sequence, so a
    parameter's weights and grad depended on how many params were made before it. Check 2 then
    compared a param alone against the same param second in a group -- two different random
    draws -- and reported "the factor is being shared across shapes" for a difference the
    factor had nothing to do with. A test whose inputs depend on call order cannot compare two
    call orders.
    """
    g = torch.Generator().manual_seed(abs(hash(shape)) % (2 ** 31))
    p = torch.nn.Parameter(torch.randn(*shape, generator=g) * 0.02)
    p.grad = torch.randn(*shape, generator=g)
    return p


def one_step(shape, shape_lr, lr=0.1):
    """Return |delta weight| after a single Muon step on one matrix of this shape."""
    train.Cfg.muon_shape_lr = shape_lr
    p = make(shape)
    opt = train.Muon([p], lr=lr, momentum=0.95, ns_steps=5, weight_decay=0.0)
    before = p.detach().clone()
    opt.step()
    return float((p.detach() - before).abs().mean())


def main():
    fails = []
    TALL, WIDE, SQUARE = (256, 64), (64, 256), (128, 128)

    # 1. DIRECTION. A tall matrix must move MORE with the flag on; a wide one must not move at
    #    all differently, because the clamp pins it to 1.0.
    t_off, t_on = one_step(TALL, False), one_step(TALL, True)
    w_off, w_on = one_step(WIDE, False), one_step(WIDE, True)
    s_off, s_on = one_step(SQUARE, False), one_step(SQUARE, True)
    want = (256 / 64) ** 0.5  # 2.0
    if t_off <= 0:
        fails.append(f"the baseline does not move a tall matrix at all ({t_off}); the test "
                     f"cannot see a factor applied to zero")
    elif abs(t_on / t_off - want) > 0.02:
        fails.append(f"tall {TALL}: |delta| ratio on/off is {t_on / t_off:.4f}, expected "
                     f"{want:.4f} = max(1, out/in)^0.5. If it is {1 / want:.4f} the ratio is "
                     f"INVERTED (in/out), which shrinks the matrices the record says to grow")
    if w_off > 0 and abs(w_on / w_off - 1.0) > 1e-6:
        fails.append(f"wide {WIDE} changed by {w_on / w_off:.6f}x; the clamp must pin it to 1.0")
    if s_off > 0 and abs(s_on / s_off - 1.0) > 1e-6:
        fails.append(f"square {SQUARE} changed by {s_on / s_off:.6f}x; out/in is 1 so the "
                     f"factor must be exactly 1.0")

    # 2. PER SHAPE, NOT PER GROUP. Two shapes in ONE optimizer must get DIFFERENT multipliers.
    #    A factor hoisted to the group would give both the same one, and that still trains.
    train.Cfg.muon_shape_lr = True
    shapes = (TALL, (512, 64))                # factors 2.0 and sqrt(8) = 2.8284
    ps = [make(s) for s in shapes]            # shape-determined, so identical to the solo runs
    opt = train.Muon(ps, lr=0.1, momentum=0.95, ns_steps=5, weight_decay=0.0)
    befores = [p.detach().clone() for p in ps]
    opt.step()
    moved = [float((p.detach() - b).abs().mean()) for p, b in zip(ps, befores)]
    if moved[0] <= 0 or moved[1] <= 0:
        fails.append(f"a parameter did not move in the two-shape group: {moved}")
    else:
        # The two shapes have different factors, so their |delta| ratio must NOT be the ratio
        # a shared factor would give. Compare against the single-param runs.
        for i, s in enumerate(shapes):
            solo = one_step(s, True)
            if abs(moved[i] / solo - 1.0) > 1e-5:
                fails.append(f"{s} in a two-shape group moved {moved[i]:.6e} but alone moved "
                             f"{solo:.6e}; the factor is being shared across shapes")

    # 3. THE FLAG MUST BE A REAL FIELD, not a getattr default. Muon reads it in __init__ and
    #    asserts hasattr, so a rename fails loudly instead of silently running the baseline.
    saved = train.Cfg.muon_shape_lr
    try:
        del train.Cfg.muon_shape_lr
    except AttributeError:
        fails.append("Cfg.muon_shape_lr is not a class attribute, so the hasattr guard in "
                     "Muon.__init__ cannot be what protects the arm")
    else:
        try:
            train.Muon([torch.nn.Parameter(torch.zeros(4, 4))], lr=0.1)
            fails.append("Muon accepted a missing Cfg.muon_shape_lr: the arm can silently run "
                         "the baseline")
        except AssertionError:
            pass
        train.Cfg.muon_shape_lr = saved

    for f in fails:
        print(f"  FAIL {f}")
    if fails:
        print(f"\n{len(fails)} failure(s)")
        return 1
    print(f"muon_shape_lr OK: tall {TALL} moves {t_on / t_off:.4f}x (want {want:.4f}), wide and "
          f"square pinned at 1.0, per-shape factors survive batching, missing field asserts")
    return 0


if __name__ == "__main__":
    sys.exit(main())

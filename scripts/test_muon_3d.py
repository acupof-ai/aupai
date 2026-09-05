#!/usr/bin/env python3
"""Muon takes one step on a 3-D parameter, and takes the SAME step it always did on a 2-D one.

WHY THIS FILE EXISTS. E1's fourth launch passed the forward and died at the first optimizer step
(runs/b0_moe_e1.log:231-235, 2026-09-05T08:14Z): dynamo's fake run of `torch.baddbmm` inside
muon_update got three tensors of shape (12, 24, 1024, 1024) and raised

    RuntimeError: expand: the requested shape has too few dimensions!

`baddbmm` is strictly 3-D. Muon.step() buckets parameters by shape and stacks each bucket, so a
2-D weight arrives in muon_update as (n, out, in) and the MoE experts -- 3-D Parameters of
(24, 1536, 1024) and (24, 1024, 768), one per layer -- arrive as (12, 24, out, in). Every dense
parameter in this model is 2-D, so before the MoE arm the stack was always 3-D and the limit was
never reached.

WHAT THE MODULE TESTS COULD NOT SEE, and it is the reason this is a separate file rather than a
tenth check: scripts/test_moe_module.py asserts which OPTIMIZER GROUP each parameter lands in
(ruling (f)) and never calls .step(). Group membership and a working update are different claims,
and the first was green through both of E1's deaths. The gap was one optimizer step on the shape
the arm actually introduces.

    python3 scripts/test_muon_3d.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# EAGER, DELIBERATELY. muon_update is torch.compile'd and inductor needs a C++ toolchain that a
# laptop may not have; the defect is in the OPERATION's rank, not in the compiler, and it
# reproduces identically in eager. The pod runs the compiled path and that is what the launch
# exercises. force_eager here is what makes this file runnable where the fix gets written.
torch.compiler.set_stance("force_eager")


def _step(Muon, shape, layers=3, seed=0):
    """One Muon step over `layers` parameters of `shape`. Returns the parameters after the step."""
    torch.manual_seed(seed)
    params = [torch.nn.Parameter(torch.randn(*shape)) for _ in range(layers)]
    for p in params:
        p.grad = torch.randn_like(p)
    Muon(params, lr=0.01, momentum=0.95, weight_decay=0.0).step()
    return params


def _muon_without_the_flatten():
    """train.Muon with the (-1, M, N) view removed -- the code as it was when E1 died.

    Mutated as a STRING and exec'd. Nothing is written to disk: a mutation that touches the working
    tree is a mutation that can be committed, and that has already cost this repo a run
    (cf. cf3dbaea). Raises if the lines it removes are not found, so this cannot silently become a
    copy of the fixed code and pass.
    """
    with open(os.path.join(ROOT, "train.py")) as f:
        src = f.read()
    pre = src.replace(
        "                _lead = X.shape[:-2]\n"
        "                X = X.reshape(-1, X.shape[-2], X.shape[-1])\n"
        "                X = X / (X.norm",
        "                X = X / (X.norm", 1)
    pre = pre.replace(
        "                X = X.reshape(*_lead, X.shape[-2], X.shape[-1])\n"
        "                mask = (grads * weights) >= 0",
        "                mask = (grads * weights) >= 0", 1)
    if pre == src or "_lead" in pre:
        raise AssertionError("could not remove the flatten -- this test has no subject any more")
    ns = {"__file__": os.path.join(ROOT, "train.py"), "__name__": "train_without_flatten"}
    exec(compile(pre, "train_without_flatten.py", "exec"), ns)  # noqa: S102
    return ns["Muon"]


def main():
    import train
    bad, n = 0, 5

    # 1. A 3-D PARAMETER STEPS, at the shape class E1 introduces: one Parameter per layer holding
    # E expert matrices, bucketed across layers. Every parameter must MOVE -- a step that silently
    # did nothing would satisfy "no exception" and leave the experts frozen, which is the failure
    # that looks like a working run.
    try:
        after = _step(train.Muon, (4, 24, 16))
        moved = [float(p.abs().sum()) > 0 for p in after]
        ok = all(moved)
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BUG '} Muon steps on 3-D params stacked across layers "
              f"((3, 4, 24, 16) after the stack), and every parameter moved ({sum(moved)}/3)")
    except Exception as e:  # noqa: BLE001 -- a raise here is the defect this file exists for
        bad += 1
        print(f"  BUG  Muon raised on a 3-D parameter: {type(e).__name__}: {str(e)[:140]}")

    # 2. THE 2-D PATH IS BIT-IDENTICAL. The flatten is a no-op for a 3-D stack -- reshape(-1, M, N)
    # of an already-(n, M, N) tensor returns the same view -- but "should be" is not a measurement,
    # and every dense matrix in every arm and every past checkpoint goes through this function. An
    # update that moved by 1e-8 would silently decouple this arm from the control it is scored
    # against.
    try:
        pre_fix = _muon_without_the_flatten()
        a = _step(pre_fix, (24, 16))
        b = _step(train.Muon, (24, 16))
        same = all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
        bad += 0 if same else 1
        deltas = [float((x - y).abs().max()) for x, y in zip(a, b, strict=True)]
        print(f"  {'ok  ' if same else 'BUG '} the 2-D dense path is bit-identical with and "
              f"without the flatten (max|d| {max(deltas):.1e})")
    except Exception as e:  # noqa: BLE001
        bad += 1
        print(f"  BUG  could not compare the 2-D path: {type(e).__name__}: {e}")

    # 3. AND THE PRE-FIX CODE MUST RAISE. Without this, checks 1 and 2 pass on any build whose
    # baddbmm happens to accept 4-D, so they would stop discriminating the day the operator
    # changed -- and this file would keep printing ok while covering nothing.
    try:
        pre_fix = _muon_without_the_flatten()
        try:
            _step(pre_fix, (4, 24, 16))
            bad += 1
            print("  BUG  removing the flatten changed nothing -- E1's step-0 death does not "
                  "reproduce, so this file is not testing the fix")
        except RuntimeError as e:
            msg = str(e)
            hit = "expand" in msg and ("too few dimensions" in msg or "number of sizes" in msg)
            bad += 0 if hit else 1
            print(f"  {'ok  ' if hit else 'BUG '} without the flatten, a 3-D param raises the "
                  f"launch's own error: {msg[:96]}")
    except Exception as e:  # noqa: BLE001
        bad += 1
        print(f"  BUG  could not build the pre-fix Muon: {type(e).__name__}: {e}")

    # 4. THE BATCHED UPDATE EQUALS THE SLICED ONE (e1's review request, 2026-09-05). Checks 1-3
    # cannot distinguish a correct batched update from a SCRAMBLED one: a flatten that reordered
    # the stack would still let every parameter move, still be bit-identical on 2-D input, and
    # still raise without the fix. This is the assertion that fails if someone later "simplifies"
    # the reshape pair into something that permutes rather than views.
    #
    # Same grads, same weights, same seed: one 3-D parameter of (E, M, N) updated inside a stack
    # must land exactly where it lands when updated alone.
    try:
        E, M, N, layers = 4, 24, 16, 3
        torch.manual_seed(11)
        stacked = [torch.nn.Parameter(torch.randn(E, M, N)) for _ in range(layers)]
        grads = [torch.randn_like(p) for p in stacked]
        sliced = [torch.nn.Parameter(p.detach().clone()) for p in stacked]
        for p, gr in zip(stacked, grads, strict=True):
            p.grad = gr.clone()
        train.Muon(stacked, lr=0.01, momentum=0.95, weight_decay=0.0).step()
        # Each parameter alone: its own optimizer, so its bucket holds one tensor and the stack
        # is (1, E, M, N) instead of (layers, E, M, N).
        for p, gr in zip(sliced, grads, strict=True):
            p.grad = gr.clone()
            train.Muon([p], lr=0.01, momentum=0.95, weight_decay=0.0).step()
        deltas = [float((a - b).abs().max()) for a, b in zip(stacked, sliced, strict=True)]
        same = all(torch.equal(a, b) for a, b in zip(stacked, sliced, strict=True))
        bad += 0 if same else 1
        print(f"  {'ok  ' if same else 'BUG '} the batched update equals the per-parameter one "
              f"(bitwise {same}, max|d| {max(deltas):.1e}) -- a flatten that reordered the stack "
              f"would still let every param move, so this is what catches one")
    except Exception as e:  # noqa: BLE001
        bad += 1
        print(f"  BUG  could not compare batched against sliced: {type(e).__name__}: {e}")

    # 5. ONE MATRIX'S SCALE DOES NOT LEAK INTO ANOTHER'S (e1's review, 4c's third item). The
    # normalisation is per-matrix over dim=(-2, -1); flattening the leading dims must not make it
    # per-stack. Scale ONE matrix by 1000x: if the flatten mixed matrices, that row would dominate
    # every other entry's update. Muon's update is orthogonalised, so a correct implementation
    # gives every matrix an update of the same order regardless of its input scale -- which is the
    # property being asserted, not merely "nothing crashed".
    try:
        torch.manual_seed(12)
        p = torch.nn.Parameter(torch.randn(2, 3, 16, 8))
        with torch.no_grad():
            p[0, 0] *= 1000.0
        p.grad = torch.randn_like(p)
        with torch.no_grad():
            p.grad[0, 0] *= 1000.0
        before = p.detach().clone()
        train.Muon([p], lr=1.0, momentum=0.0, weight_decay=0.0).step()
        per = (p - before).abs().amax(dim=(-2, -1))          # (2, 3) max update per matrix
        lo, hi = float(per.min()), float(per.max())
        # An order of magnitude is the bar: orthogonalised updates land within a small factor of
        # each other, while a leaked 1000x scale would separate them by ~1000.
        ok = hi / max(lo, 1e-12) < 10.0
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BUG '} the per-matrix norm survives the flatten: one matrix "
              f"scaled 1000x, per-matrix updates span {lo:.4f}-{hi:.4f} (ratio {hi / max(lo, 1e-12):.2f}, "
              f"a leak would show ~1000)")
    except Exception as e:  # noqa: BLE001
        bad += 1
        print(f"  BUG  could not check per-matrix scaling: {type(e).__name__}: {e}")

    print(f"test_muon_3d: {n - bad}/{n} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1:] != ["--selftest"]:
        sys.exit(f"usage: {os.path.basename(__file__)} [--selftest]  (got {sys.argv[1:]})")
    sys.exit(main())

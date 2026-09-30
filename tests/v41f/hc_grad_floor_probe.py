"""Measure the healthy relative-gradient floor between two numerically-equivalent v41f
attention implementations, and the signal a real break produces against it.

The floor is host-class dependent and is what sets allclose.HC_GRAD_RTOL; the numbers this
prints are recorded in facts/v41.json#v41.hc_grad_rel_floor_0930.

Run:  CUDA_VISIBLE_DEVICES= python3 tests/v41f/hc_grad_floor_probe.py
      CUDA_VISIBLE_DEVICES= python3 tests/v41f/hc_grad_floor_probe.py --selftest
"""
import argparse
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_p1_docpack import _batch, _cfg, _grads, _model  # noqa: E402
from v41f import docpack  # noqa: E402


def worst_rel(mutate=None):
    """Worst relative gradient difference between the ref and chunked implementations.

    mutate, when given, is applied to the chunked model only, so it stands in for a real
    disagreement rather than float accumulation order.
    """
    ids, cu = _batch()
    ref = _model(_cfg())
    chk = _model(_cfg(attn_impl="chunked"))
    chk.load_state_dict(ref.state_dict())
    if mutate is not None:
        mutate(chk)
    old = docpack.ATTN_CHUNK
    docpack.ATTN_CHUNK = 5
    try:
        _, ga = _grads(ref, ids, cu)
        _, gb = _grads(chk, ids, cu)
    finally:
        docpack.ATTN_CHUNK = old
    best_rel, best_name, best_abs = 0.0, "", 0.0
    for n in ga:
        dn = (ga[n] - gb[n]).norm()
        rel = (dn / ga[n].norm().clamp_min(1e-12)).item()
        if rel > best_rel:
            best_rel, best_name, best_abs = rel, n, dn.item()
    return best_rel, best_name, best_abs


def _add_base(model):
    for n, p in model.named_parameters():
        if n.endswith("layers.3.hc.hc_attn_base"):
            with torch.no_grad():
                p.add_(1e-3)


def _scale_fn(model):
    for n, p in model.named_parameters():
        if "layers.3" in n and n.endswith("hc_attn_fn"):
            with torch.no_grad():
                p.mul_(1.0 + 1e-4)


MUTANTS = (("hc_attn_base += 1e-3", _add_base), ("hc_attn_fn *= 1+1e-4", _scale_fn))


def selftest():
    """Known answer: the healthy floor must sit below the hc threshold, and the coarse
    mutant must sit above it. A run where the mutant does not clear the threshold means the
    comparison has lost its resolution on this host, which is the failure this probe exists
    to make visible."""
    from allclose import HC_GRAD_RTOL

    healthy, hname, _ = worst_rel()
    assert healthy < HC_GRAD_RTOL, f"healthy floor {healthy:.3e} ({hname}) already exceeds {HC_GRAD_RTOL:.0e}"
    coarse, cname, _ = worst_rel(_add_base)
    assert coarse > HC_GRAD_RTOL, f"coarse mutant {coarse:.3e} ({cname}) does not clear {HC_GRAD_RTOL:.0e}"
    assert coarse > healthy * 3, f"coarse mutant {coarse:.3e} is not 3x the floor {healthy:.3e}"
    print(f"hc_grad_floor_probe selftest ok: healthy {healthy:.3e} < {HC_GRAD_RTOL:.0e} < mutant {coarse:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    print(f"{platform.machine()} {platform.system()} torch={torch.__version__} threads={torch.get_num_threads()}")
    rel, name, dabs = worst_rel()
    print(f"  healthy              rel {rel:.3e}  abs {dabs:.3e}  {name}")
    for tag, fn in MUTANTS:
        rel, name, dabs = worst_rel(fn)
        print(f"  {tag:22s} rel {rel:.3e}  abs {dabs:.3e}  {name}")


if __name__ == "__main__":
    main()

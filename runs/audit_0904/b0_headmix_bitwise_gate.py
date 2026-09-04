#!/usr/bin/env python3
"""Bitwise gate for the `inner=` edit to DeltaRecurrence and GatedMLA (b0, head-hybrid arm B).

WHY THIS EXISTS. Arm B needs the two mixers to project from the residual width d to an
explicit INNER width and back, instead of reading their width off `x.shape`
(DeltaRecurrence.forward:107, GatedMLA.forward:198, and the output reshapes at :175 and
:248). Those two classes are on the path of every checkpoint in this repo -- the Stage D/E
arms, both N2 legs, the 15b and 30b runs. The risk is not that arm B fails; it is that arm A
and every earlier arm change bitwise while the tests still pass, which is exactly the shape
of eff.eval_path_cu_artifact_ce: a number that moved because the instrument moved.

WHAT IT PROVES. `inner=None` must mean "cfg.d", and that path must be the code it is today
to the last bit. Run before the edit to record logits, after the edit to compare:

    python3 runs/audit_0904/b0_headmix_bitwise_gate.py --record   # pre-change
    <edit model.py>
    python3 runs/audit_0904/b0_headmix_bitwise_gate.py --check    # post-change

--check FAILS on any nonzero difference and names the first differing module rather than
proposing a fix (6e's instruction 2026-09-04). Two shapes, because both are live: d1024 L12
h8 (the ladder shape arm A/B use) and d768 h6 (Stage D/E).

FIXED INPUT, FIXED SEED, EAGER. torch.compile is off and dtype is fp32: the question is
whether the ARITHMETIC changed, and a compiled bf16 path has its own nondeterminism that
would mask or fake a difference. A real checkpoint's weights, not random init -- random
weights would agree bitwise under any edit that preserved shapes, which is the failure this
gate is aimed at.
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

OUT = os.path.join(ROOT, "runs", "b0_headmix_bitwise.json")
# ONE BASELINE PER SHAPE, keyed by the shape the checkpoint actually holds. A single-slot
# baseline file is what turns two shapes into one: recording d768 after d1024 would silently
# replace it, and --check would then compare the wrong pair and report IDENTICAL. Same defect
# as launch_tests keying on the test path alone (audit MT-4), one file over.


def _logits(ckpt_path, seed=904):
    """Logits for one checkpoint on fixed input, eager, ON A GPU, in the training dtype.

    CUDA IS NOT OPTIONAL AND CPU IS NOT A FALLBACK. DeltaRecurrence routes through
    fla.ops.kda.chunk_kda, a Triton kernel: on CPU tensors it raises
    `ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)`
    (measured 2026-09-04). A gate that silently fell back to a CPU path would be comparing
    a code path the training run never takes.

    NOR IS fp32. Forcing the model to float() and running it dies with
    `torch.AcceleratorError: CUDA error: misaligned address` (measured 2026-09-04, card 5):
    the KDA kernel is built for the bf16 layout the training path feeds it, and fp32 is not a
    "more precise" version of that path, it is a different one that does not run. So this uses
    bf16 autocast exactly as training does. Eager, never compiled: inductor is free to reorder
    arithmetic between two builds of the same source, which would show up here as a difference
    the edit did not cause.
    """
    import model as M

    if not torch.cuda.is_available():
        raise SystemExit("REFUSING: no CUDA device. The KDA mixer is a Triton kernel and "
                         "cannot run on CPU, so a CPU run would prove nothing about the "
                         "training path. Run this on a GRANTED card: "
                         "CUDA_VISIBLE_DEVICES=<card> python3 " + os.path.relpath(__file__, ROOT))
    dev = "cuda"
    torch.use_deterministic_algorithms(False)  # cuDNN/Triton paths here are already fixed-seed
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    if isinstance(cfg, dict):
        class C:
            pass
        c = C()
        for k, v in cfg.items():
            setattr(c, k, v)
        cfg = c
    torch.manual_seed(seed)
    m = M.HybridLM(cfg)
    sd = M.remap_legacy_state_dict(ck["model"])
    missing, unexpected = m.load_state_dict(sd, strict=False)
    m.eval().to(dev)
    torch.manual_seed(seed)
    x = torch.randint(0, int(getattr(cfg, "vocab_real", cfg.vocab)), (2, 64), device=dev)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = m(x)
    if isinstance(y, tuple):
        y = y[0]
    # HASH THE REAL VOCAB ONLY. model.py:557 sets columns [vocab_real:vocab] to
    # torch.finfo(dtype).min so alignment padding stays neutral in the softmax -- deliberate,
    # not corruption. But -3.4e38 in 11 of 32784 columns dominates any sum or mean over the
    # whole tensor: the first baseline printed sum -4.79e+41 and mean -inf, which says nothing
    # about the 32773 columns the model actually predicts. A digest over the padding is also
    # blind to a change inside it, in exchange for a number no reader can sanity-check.
    real = int(getattr(cfg, "vocab_real", cfg.vocab))
    y = y[..., :real].float().cpu()
    return y, dict(missing=len(missing), unexpected=len(unexpected), vocab_real=real,
                   d=cfg.d, heads=cfg.heads, layers=cfg.layers)


def _fp(t):
    """A fingerprint that a rounding change cannot hide: exact bytes, not a rounded print."""
    import hashlib
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def _key(meta):
    return f"d{meta['d']}L{meta['layers']}h{meta['heads']}"


def record(ckpt):
    y, meta = _logits(ckpt)
    k = _key(meta)
    all_ = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    all_[k] = {"ckpt": os.path.basename(ckpt), "sha256": _fp(y), "shape": list(y.shape),
               "sum": float(y.double().sum()), "meta": meta}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(all_, f, indent=1, sort_keys=True)
    r = all_[k]
    print(f"recorded {k} from {r['ckpt']}")
    print(f"  logits {r['shape']}  sha256 {r['sha256'][:16]}  sum {r['sum']!r}")
    print(f"  missing {meta['missing']} unexpected {meta['unexpected']} keys")
    print(f"-> {OUT} (now holds: {', '.join(sorted(all_))})")
    return 0


def check(ckpt):
    y, meta = _logits(ckpt)
    k = _key(meta)
    all_ = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    if k not in all_:
        print(f"REFUSING: no baseline for {k} in {OUT} (have: {', '.join(sorted(all_)) or 'none'}). "
              f"Run --record on this shape BEFORE editing model.py; a baseline taken after the "
              f"edit compares the new code with itself.")
        return 2
    want = all_[k]
    got = _fp(y)
    same = got == want["sha256"]
    print(f"{k}: baseline {want['sha256'][:16]}  now {got[:16]}  ->  "
          f"{'IDENTICAL' if same else 'DIFFERS'}")
    if same:
        print(f"  logits {list(y.shape)}: the inner=None path is bitwise the code it replaced")
        return 0
    print(f"  sum was {want['sum']!r}, now {float(y.double().sum())!r}")
    print("  THE EDIT CHANGED THE FULL-WIDTH PATH. Not proposing a fix (6e 2026-09-04).")
    print("  The width now comes from the module rather than from x.shape, so the candidates")
    print("  are DeltaRecurrence.forward and GatedMLA.forward. To name the first differing")
    print("  module, re-record on the pre-edit tree with per-module hooks; this gate reports")
    print("  the model output only.")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ckpt", default="ckpt_b0_sd_unlooped.pt")
    a = ap.parse_args()
    if a.record == a.check:
        print("exactly one of --record / --check")
        raise SystemExit(2)
    raise SystemExit(record(a.ckpt) if a.record else check(a.ckpt))

#!/usr/bin/env python3
"""Equivalence gate for the `inner=` edit to DeltaRecurrence and GatedMLA (b0, head-hybrid arm B).

WHY THIS EXISTS. Arm B needs the two mixers to project from the residual width d to an
explicit INNER width and back, instead of reading their width off `x.shape`
(DeltaRecurrence.forward:107, GatedMLA.forward:198, and the output reshapes at :175 and
:248). Those two classes are on the path of every checkpoint in this repo -- the Stage D/E
arms, both N2 legs, the 15b and 30b runs. The risk is not that arm B fails; it is that arm A
and every earlier arm change numerically while the tests still pass, which is exactly the shape
of eff.eval_path_cu_artifact_ce: a number that moved because the instrument moved.

WHAT IT PROVES. `inner=None` must mean "cfg.d", and that path must produce the same logits as
the code it is today AT BF16 RESOLUTION (see the bound below). Run before the edit to record
logits, after the edit to compare:

    CUDA_VISIBLE_DEVICES=<granted card> python3 runs/audit_0904/b0_headmix_bitwise_gate.py \
        --record --ckpt <ckpt>          # pre-change, once per shape
    <edit model.py>
    ... --check --ckpt <ckpt>           # post-change
    ... --check --ckpt <ckpt> --perturb kda    # must FAIL
    ... --check --ckpt <ckpt> --perturb mla    # must FAIL

--check FAILS on any nonzero difference and names the two candidate modules rather than
proposing a fix (6e's instruction 2026-09-04). One shape per invocation, keyed by the shape
the checkpoint holds; run it once per live width.

AND THE PASS IS WORTH NOTHING WITHOUT THE TWO FAILS. This gate was adjusted three times
before it first passed -- CPU to CUDA, fp32 to bf16, and the summary statistic -- which is the
shape of an instrument tuned until it agrees. `--perturb` is the answer: it steps one weight ROW
by one BF16 ulp and --check must report DIFFERS. Committed beside the PASS.

WHAT THE PASS BOUNDS, stated because "bitwise" would overclaim it: identical AT BF16
RESOLUTION, the precision the training path runs at, TO A PERTURBATION OF ONE WEIGHT ROW. Both
limits were measured, not assumed. A change smaller than one bf16 ulp in a weight is invisible
by construction (the cast erases it -- the first fixture's defect). And a one-ulp change to a
SINGLE ELEMENT of a d=768 input row is also invisible: it is ~1/30th of the bf16 ulp of the
accumulator it lands in, and measured exactly 0.0. See the perturb block for both numbers.

REQUIRES A GPU AND bf16, both learned by running it: see _logits for the two errors.
A real checkpoint's weights, not random init -- random weights agree under any edit that
preserves shapes, which is the failure this gate is aimed at.
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


def _logits(ckpt_path, seed=904, perturb=None):
    """Logits for one checkpoint on fixed input, eager, ON A GPU, in the training dtype.

    `perturb` is the broken-world lever (6e, 2026-09-04): "kda" steps block 0's KDA qkv by one
    BF16 ulp, "mla" steps the LAST MLA layer's kv_down. A gate that reports IDENTICAL under a
    changed weight is not measuring what it claims, and this gate was adjusted three times
    before it first passed (CUDA, bf16, the summary statistic) -- the history that makes an
    unperturbed PASS worth nothing on its own. The first fixture DID report IDENTICAL; see the
    perturb block for why (fp32 ulp erased by the bf16 cast).

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
    if perturb:
        # PERTURB AT THE RESOLUTION OF THE PATH BEING CERTIFIED. The first version of this
        # nudged an fp32 weight by one fp32 ulp and BOTH fixtures reported IDENTICAL: weights
        # are fp32, the forward runs bf16 autocast, and one fp32 ulp at 0.017333984375 is
        # 0.01733398623764515 while bf16's neighbouring value is 0.0174560546875 -- ~700x
        # further away. Autocast rounded the perturbation back to the same bf16 value before
        # the kernel saw it, so the gate was blind to every change below one bf16 step and its
        # PASS certified nothing (measured 2026-09-04, card 5).
        #
        # So: cast to bf16, step one bf16 ulp, cast back. Now the fixture moves the value the
        # kernel actually receives.
        #
        # TWO SITES, and the second is chosen for propagation depth (6e): block 0's KDA qkv is
        # the deepest path (its change traverses every later block), and the LAST MLA layer's
        # kv_down is the shallowest -- only the final norm and head follow it. At L12 with
        # attn_every 4 the MLA blocks are i%4==3, i.e. 3, 7 and 11, so the last MLA is block
        # 11, NOT a KDA block. If a change there vanishes, everything upstream is untestable.
        # AND THE UNIT IS A ROW, NOT AN ELEMENT -- the second thing the fixture got wrong, for
        # the same reason as the first. Stepping ONE element of blocks.0.mixer.qkv.weight by one
        # bf16 ulp moved the logits by exactly 0.0 (measured 2026-09-04, card 5), while x1.5 and
        # x2.0 on that same element moved them by 1.63e-01. That is not a saturation ceiling:
        # at the argmax the logit is 8.905 and NO logit is within 0.01 of SOFTCAP=15. It is the
        # path's real sensitivity. Weight row 0 supplies 1 of d=768 terms in a dot product; one
        # bf16 ulp there is dw=1.2207e-04, and the pre-activation it lands in is stored at bf16,
        # whose ulp near 1.0 is ~4e-3. The change is ~1/30th of the accumulator's own resolution,
        # so the store rounds it away. A whole row -- 128 elements, one ulp each -- moves it
        # (1.485e-01), because a row is the unit that feeds one output channel.
        # MLA's kv_down fired as a single element only because its rows are narrower; relying on
        # that would leave the KDA half of a head-hybrid gate untested, which is the half arm B
        # changes. So both sites step a row.
        want, pick_last = {"kda": ("mixer.qkv.weight", False),
                           "mla": ("mixer.kv_down.weight", True)}[perturb]
        hit = [(n, p) for n, p in m.named_parameters() if n.endswith(want)]
        if not hit:
            raise SystemExit(f"REFUSING: --perturb {perturb} matched no tensor ending "
                             f"{want}; the fixture would not have fired.")
        n, p = hit[-1] if pick_last else hit[0]
        with torch.no_grad():
            row = p[0].view(-1)
            before = row.clone()
            b16 = row.to(torch.bfloat16)
            stepped = torch.nextafter(b16, torch.full_like(b16, float("inf")))
            row.copy_(stepped.to(row.dtype))
            moved = int((row != before).sum())
        if moved == 0:
            raise SystemExit(f"REFUSING: the bf16 step moved 0 of {row.numel()} elements in "
                             f"{n}[0]; a fixture that changes nothing cannot show the gate "
                             f"discriminates.")
        print(f"  perturbed {n}[0,:] ({row.numel()} elements, {moved} moved by one BF16 ulp; "
              f"{'last' if pick_last else 'first'} of {len(hit)} such tensors)")
    torch.manual_seed(seed)
    x = torch.randint(0, int(getattr(cfg, "vocab_real", cfg.vocab)), (2, 64), device=dev)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = m(x)
    if isinstance(y, tuple):
        y = y[0]
    # THE PADDING COLUMNS ARE HASHED TOO, and the reason matters (6e, 2026-09-04).
    # model.py:555-557 overwrites [vocab_real:vocab] with torch.finfo(out.dtype).min AFTER the
    # softcap -- a deterministic constant assignment, not uninitialized memory. So they are a
    # legitimate part of the comparison: if the edit ever stopped writing them, or wrote a
    # different constant, that is a real change and this gate must catch it. Excluding them
    # because they dominated a printed SUM would be discarding coverage to fix a display.
    # What was actually wrong was the summary statistic, so the digest covers the whole tensor
    # and the printed sum reports the real-vocab slice separately, where a reader can judge it.
    real = int(getattr(cfg, "vocab_real", cfg.vocab))
    y = y.float().cpu()
    return y, dict(missing=len(missing), unexpected=len(unexpected), vocab_real=real,
                   d=cfg.d, heads=cfg.heads, layers=cfg.layers)


def _fp(t):
    """A fingerprint that a rounding change cannot hide: exact bytes, not a rounded print."""
    import hashlib
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def _sum_real(t, meta):
    """Sum over the REAL vocab only. The digest covers the whole tensor including the padding
    constant; this number exists so a reader can judge the magnitude, which -3.4e38 in the
    padding columns makes impossible over the full width."""
    return float(t[..., : meta["vocab_real"]].double().sum())


def _key(meta):
    return f"d{meta['d']}L{meta['layers']}h{meta['heads']}"


def record(ckpt):
    y, meta = _logits(ckpt)
    k = _key(meta)
    all_ = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    all_[k] = {"ckpt": os.path.basename(ckpt), "sha256": _fp(y), "shape": list(y.shape),
               "sum_real": _sum_real(y, meta), "meta": meta}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(all_, f, indent=1, sort_keys=True)
    r = all_[k]
    print(f"recorded {k} from {r['ckpt']}")
    print(f"  logits {r['shape']}  sha256 {r['sha256'][:16]}  "
          f"sum over [:{meta['vocab_real']}] {r['sum_real']!r}")
    print(f"  missing {meta['missing']} unexpected {meta['unexpected']} keys")
    print(f"-> {OUT} (now holds: {', '.join(sorted(all_))})")
    return 0


def check(ckpt, perturb=None):
    y, meta = _logits(ckpt, perturb=perturb)
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
        print(f"  logits {list(y.shape)}: the inner=None path matches the code it replaced "
              f"at bf16 resolution, the precision the training path runs at")
        return 0
    print(f"  sum over [:{meta['vocab_real']}] was {want['sum_real']!r}, "
          f"now {_sum_real(y, meta)!r}")
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
    ap.add_argument("--perturb", choices=["kda", "mla"],
                    help="broken-world: nudge one weight by 1 ulp; --check must FAIL")
    a = ap.parse_args()
    if a.record == a.check:
        print("exactly one of --record / --check")
        raise SystemExit(2)
    if a.record and a.perturb:
        print('REFUSING: --perturb with --record would bake the perturbation into the baseline')
        raise SystemExit(2)
    raise SystemExit(record(a.ckpt) if a.record else check(a.ckpt, a.perturb))

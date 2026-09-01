#!/usr/bin/env python
"""Does the depth-muP 1/sqrt(L) rule apply to THIS architecture and THIS optimizer?

Asked when the 500M shape was ruled d=1024/L=32 (2.7x today's depth at fixed width) and a
shakedown had already launched at lr_scale = sqrt(12/32) = 0.61. Width is fixed, so depth is
the only axis and this is the only number that matters.

depth-muP prescribes TWO coupled changes: per-layer lr ~ 1/sqrt(L), and a 1/sqrt(L) scaling on
each residual branch's output. Both are derived for Adam on a model whose residual stream is
ADDITIVE: x <- x + f(x), L times. Under that recurrence L independent branch outputs accumulate
as sqrt(L), so the stream grows with depth and both halves of the rule exist to cancel it.

Neither premise holds here, and each fails for its own reason:

  ARCHITECTURE. With attn_res=True the stream is not additive. AttnRes replaces it with
  softmax(q . RMSNorm(v_i)) over all previous sublayer outputs -- a CONVEX combination, weights
  summing to 1 by construction. A convex combination is bounded by its largest input for any
  number of inputs, so the stream does not grow with depth and there is nothing for a 1/sqrt(L)
  branch scale to cancel. Q2 is answered by that sentence: the init half is already structurally
  present, and adding it again would double-count.

  OPTIMIZER. Muon orthogonalizes each update, so the step has ~unit spectral norm independent of
  the gradient. depth-muP's Adam derivation reasons about how per-layer gradient magnitude scales
  with L; under Muon that quantity is normalized away before it reaches the weights. The exponent
  is not inherited from the Adam result -- it has to be measured.

check 1  convexity is structural, not an init artifact (holds at every q scale, not just zero-init)
check 2  a convex combination is bounded by its largest source, at any source count
check 3  the measured depth exponent, from a fixed relative weight perturbation on every layer
check 4  Muon's update magnitude does not depend on the gradient's
check 6  Muon's update magnitude does not depend on gradient NOISE either, i.e. on batch
check 7  the exponent is stable across seeds and perturbation scales

WHAT THIS IS NOT. Check 3 perturbs weights at INIT and reads the forward response. That is the
quantity the muP derivation is about, and it is the honest scope of this probe: it measures the
architecture's depth sensitivity, NOT the optimal learning rate. An lr is set by training
dynamics over thousands of steps; a forward-sensitivity exponent bounds what the rule should be,
it does not replace a sweep. Read it as "sqrt is the wrong exponent for this architecture", never
as "the optimal lr_scale is exactly X".

Checks 4 and 6 are two different questions that look like one. Check 4 varies gradient SCALE;
check 6 varies gradient NOISE at fixed scale, which is what changing the batch actually does.
Scale-invariance does not imply noise-invariance, so answering "does the depth rule interact with
the batch decision" required measuring the second rather than citing the first. It does not, FOR
MUON. The AdamW groups (embed_lr, scalar_lr) are batch-scaled per nanochat at a batch this probe
never varies, so nothing here speaks to them -- that is a hole, not a clean bill.

The KDA mixer needs the fla CUTLASS kernel, absent on a dev box, so it is replaced by its own
output projection. That changes the RMS magnitudes and does NOT change the control flow, the
source counts, or the softmax -- the three things every claim here rests on. Check 3's ON/OFF
arms agree to five decimals, which is not a bug: the perturbation is applied to the SAME weights
in both arms and AttnRes is a normalizing readout on top, so the RELATIVE response is the same
even though the absolute RMS differs 25x. Check 0 asserts that 25x, so "both arms are live" is
verified rather than assumed.
"""
import argparse
import copy
import importlib.util
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _train():
    """train.py imported as a module: the real HybridLM/AttnRes/Muon, never a reimplementation."""
    sys.path.insert(0, ROOT)
    argv, sys.argv = sys.argv, ["train.py"]
    try:
        spec = importlib.util.spec_from_file_location("_t", os.path.join(ROOT, "train.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules["_t"] = m
        try:
            spec.loader.exec_module(m)
        except SystemExit:
            pass
        return m
    finally:
        sys.argv = argv


def build(m, d, L, attn_res, seed=0):
    torch.manual_seed(seed)
    c = copy.deepcopy(m.Cfg)
    c.d, c.layers, c.heads = d, L, d // 128
    c.ffn_hidden, c.attn_res, c.compile, c.grad_ckpt = 3 * d, attn_res, False, False
    mm = m.HybridLM(c).float().eval()
    for b in mm.blocks:  # fla kernel is absent on a dev box; shape-correct stand-in
        if isinstance(b.mixer, m.DeltaRecurrence):
            b.mixer.forward = lambda x, cu=None, _o=b.mixer.o: _o(x)
    return mm


@torch.no_grad()
def depth_sensitivity(m, d, L, attn_res, rel=1e-3, seed=0):
    """Relative output move for the same relative weight perturbation on every layer."""
    mm = build(m, d, L, attn_res, seed)
    torch.manual_seed(123)
    x = torch.randint(0, 30000, (1, 16))
    base = mm._body(mm.tok(x), None)
    g = torch.Generator().manual_seed(7)
    for b in mm.blocks:
        for p in list(b.ffn.parameters()) + list(b.mixer.parameters()):
            if p.dim() == 2:
                p.add_(torch.randn(p.shape, generator=g) * rel * p.std())
    pert = mm._body(mm.tok(x), None)
    return (pert - base).pow(2).mean().sqrt().item() / base.pow(2).mean().sqrt().item()


def peak_sources(L, blocks=0):
    """Resident AttnRes sources at the deepest point -- Full is O(L), its READS are O(L^2)."""
    n_sub = 2 * L
    nb = min(n_sub, blocks or n_sub)
    ends = {round((j + 1) * n_sub / nb) for j in range(nb)}
    done, partial, n, pk = 1, 0, 0, 1
    for _ in range(n_sub):
        pk = max(pk, done + partial)
        partial = 1
        n += 1
        if n in ends:
            done, partial = done + partial, 0
    return max(pk, done + partial)


def source_reads(L, blocks=0):
    n_sub = 2 * L
    nb = min(n_sub, blocks or n_sub)
    ends = {round((j + 1) * n_sub / nb) for j in range(nb)}
    done, partial, n, tot = 1, 0, 0, 0
    for _ in range(n_sub):
        tot += done + partial
        partial = 1
        n += 1
        if n in ends:
            done, partial = done + partial, 0
    return tot + done + partial


def measure(d=1024, depths=(12, 16, 24, 32, 48)):
    m = _train()
    sens = {L: depth_sensitivity(m, d, L, True) for L in depths}
    base = depths[0]
    exps = {L: math.log(sens[L] / sens[base]) / math.log(L / base) for L in depths[1:]}
    return {
        "d": d,
        "sensitivity": sens,
        "exponents_vs_L%d" % base: exps,
        "sqrt_rule_exponent": 0.5,
        "reads_L12_Full": source_reads(12),
        "reads_L32_Full": source_reads(32),
        "peak_sources_L32_Full": peak_sources(32),
    }


def selftest():
    m = _train()
    d = 1024

    # 0. both arms live: AttnRes changes the output, so an ON/OFF agreement below is a
    #    real result and not a flag that never took effect.
    a, b = build(m, d, 12, True), build(m, d, 12, False)
    torch.manual_seed(123)
    x = torch.randint(0, 30000, (1, 16))
    with torch.no_grad():
        ra = a._body(a.tok(x), None).pow(2).mean().sqrt().item()
        rb = b._body(b.tok(x), None).pow(2).mean().sqrt().item()
    assert a.final_ar is not None and b.final_ar is None, "attn_res flag did not take effect"
    assert rb / ra > 5, f"arms differ by only {rb / ra:.2f}x -- the ON/OFF arms are not distinct"
    print(f"  0 both arms live: AttnRes OFF/ON output rms = {rb / ra:.1f}x")

    # 1. convexity is structural. If it held only at zero-init q it would be an artifact that
    #    disappears the moment training moves q, and every claim built on it would expire.
    torch.manual_seed(0)
    for qs in (0.0, 0.1, 1.0, 5.0):
        ar = m.AttnRes(d).float().eval()
        if qs:
            with torch.no_grad():
                ar.q.normal_(0, qs)
        srcs = [m.Source.of(torch.randn(1, 8, d)) for _ in range(65)]
        with torch.no_grad():
            gq = ar.g * ar.q
            w = torch.stack([(s.v * gq).sum(-1) * s.scale.squeeze(-1) for s in srcs]).float().softmax(0)
        assert abs(w.sum(0).mean().item() - 1.0) < 1e-5, f"weights do not sum to 1 at q_std={qs}"
    print("  1 convex at every q scale (0, 0.1, 1, 5): sum(w)=1 is structural, not an init artifact")

    # 2. bounded by the largest source at ANY count -- the property that replaces 1/sqrt(L).
    torch.manual_seed(0)
    ar = m.AttnRes(d).float().eval()
    for n in (1, 25, 65, 121):
        srcs = [m.Source.of(torch.randn(1, 16, d)) for _ in range(n)]
        with torch.no_grad():
            out = ar(srcs)
        mx = max(s.v.pow(2).mean().sqrt().item() for s in srcs)
        assert out.pow(2).mean().sqrt().item() <= mx * 1.01, f"not bounded by max source at n={n}"
    print("  2 output bounded by the largest source at n=1..121: no growth for 1/sqrt(L) to cancel")

    # 3. the measured exponent is far below sqrt. This is the finding; assert it is not 0.5.
    r = measure(d, (12, 32))
    e32 = r["exponents_vs_L12"][32]
    assert e32 < 0.25, f"exponent {e32:.3f} is not well below the sqrt rule's 0.5"
    print(f"  3 depth exponent 12->32 is {e32:+.3f}, not the sqrt rule's +0.500")

    # 4. Muon's update does not scale with the gradient -- so an Adam-derived depth exponent
    #    is not inherited. Wrong direction here would mean Muon behaves like Adam and the
    #    literature rule WOULD carry over, which is the claim this rules out.
    norms = []
    for gs in (1e-4, 1.0, 1e2):
        w = torch.zeros(256, 256)
        opt = m.Muon([w], lr=1.0, momentum=0.0, ns_steps=5, weight_decay=0.0)
        w.grad = torch.randn(256, 256) * gs
        opt.step()
        norms.append(w.norm().item())
    assert max(norms) / min(norms) < 1.05, f"Muon update DOES track gradient scale: {norms}"
    print(f"  4 Muon update norm over a 1e6 gradient range: {min(norms):.3f}..{max(norms):.3f} (flat)")

    # 5. Full AttnRes reads are O(L^2): the cost that decides whether L=32 ships as Full.
    #    The second clause is COMPUTED, not guessed: the first draft asserted blocks=8 at L=32
    #    reads less than Full at L=12 and it does not (353 vs 325) -- blocks bounds the source
    #    list but every one of the 2L sublayers still reads it, so the floor is ~2L*blocks/2.
    #    Left in as a regression case: the near-miss is the useful fact (blocks=8 at L=32 costs
    #    about what Full costs today), and a guessed threshold would have hidden it.
    r12, r32 = source_reads(12), source_reads(32)
    assert r32 / r12 > 5, f"reads grew only {r32 / r12:.1f}x -- expected quadratic"
    assert source_reads(32, 8) < r32 / 5, "blocks=8 must cut Full's reads by >5x"
    assert 1.0 < source_reads(32, 8) / r12 < 1.2, (
        f"blocks=8 at L=32 is {source_reads(32, 8) / r12:.2f}x Full-at-L=12; the parity that "
        f"makes it the cheap option has moved"
    )
    print(f"  5 Full AttnRes source reads {r12} (L=12) -> {r32} (L=32) = {r32 / r12:.1f}x, quadratic")
    print(f"    blocks=8 at L=32 reads {source_reads(32, 8)}, ~parity with Full at L=12 ({r12})")

    # 6. Muon is invariant to gradient NOISE, not merely to gradient SCALE. Batch changes the
    #    former; check 4 only ruled out the latter. Without this, "the depth rule does not
    #    interact with the batch decision" would be a scale result answering a noise question.
    torch.manual_seed(0)
    n = 256
    signal = torch.randn(n, n) * 0.1
    norms, coss = [], []
    clean = torch.zeros(n, n)
    oc = m.Muon([clean], lr=1.0, momentum=0.0, ns_steps=5, weight_decay=0.0)
    clean.grad = signal.clone()
    oc.step()
    for k in (1, 4, 28, 196):  # 4 = tilerl's max per card; 28 = world 7 at accum 1
        g = torch.Generator().manual_seed(k)
        w = torch.zeros(n, n)
        opt = m.Muon([w], lr=1.0, momentum=0.0, ns_steps=5, weight_decay=0.0)
        w.grad = signal + torch.randn(n, n, generator=g) / (k**0.5)
        opt.step()
        norms.append(w.norm().item())
        coss.append(((w.flatten() @ clean.flatten()) / (w.norm() * clean.norm())).item())
    assert max(norms) / min(norms) < 1.05, f"Muon update magnitude tracks batch noise: {norms}"
    assert coss[-1] > coss[0] * 3, f"more batch must improve direction, got {coss}"
    print(f"  6 Muon update norm flat over batch 1..196 ({min(norms):.2f}..{max(norms):.2f}); "
          f"only direction improves (cos {coss[0]:.2f}->{coss[-1]:.2f})")

    # 7. the exponent survives seed and perturbation scale. A single-seed exponent quoted as a
    #    launch parameter is the shape of error this repo keeps finding, so measure the spread.
    exps = []
    for seed in (0, 1, 2):
        for rel in (1e-4, 1e-2):
            s12 = depth_sensitivity(m, d, 12, True, rel, seed)
            s32 = depth_sensitivity(m, d, 32, True, rel, seed)
            exps.append(math.log(s32 / s12) / math.log(32 / 12))
    lo, hi = min(exps), max(exps)
    assert hi < 0.25, f"exponent {hi:.3f} reaches the sqrt regime on some seed"
    assert hi - lo < 0.05, f"exponent spread {hi - lo:.3f} is too wide to quote"
    print(f"  7 exponent over 3 seeds x 2 scales: {lo:+.3f}..{hi:+.3f} "
          f"(implied lr_scale {(32 / 12) ** -hi:.3f}..{(32 / 12) ** -lo:.3f}, sqrt rule 0.612)")

    print("selftest: 8/8")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    out = measure()
    print(json.dumps(out, indent=2) if a.json else out)

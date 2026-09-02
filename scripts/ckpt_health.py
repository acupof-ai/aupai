#!/usr/bin/env python3
"""b0-10: offline checkpoint health read. CPU only, no GPU, no training interference.

Reads two checkpoints of the same run and reports, per layer:
  - weight-norm ratio between the two steps (is a layer frozen, or running away)
  - KDA decay parameters (A_log -> the per-head decay, dt_bias)
  - embedding norm
  - the attention / KDA branch output-scale ratio

Layers whose ratio is more than N sigma from the median across layers are named. The
threshold is on the DISTRIBUTION ACROSS LAYERS, not an absolute bound: a run where every
layer's norm grows 3% is healthy, a run where one layer grows 3% and the rest 0.1% is not,
and only the second is visible in the spread.

WHY MEDIAN AND MAD, not mean and std: the outlier we are looking for is exactly what would
drag a mean, so a mean-based z-score partly hides the thing it is meant to surface. MAD is
scaled by 1.4826 so that for Gaussian data it estimates the same sigma.

    python3 scripts/ckpt_health.py A.pt B.pt            # A = earlier, B = later
    python3 scripts/ckpt_health.py A.pt B.pt --sigma 3   # default 3
    python3 scripts/ckpt_health.py --selftest
"""
import argparse
import re
import sys

SCALE = 1.4826  # MAD -> sigma for Gaussian data


def robust_z(values):
    """(median, sigma_from_mad, [z...]) -- sigma 0.0 when the spread is degenerate."""
    xs = sorted(values)
    n = len(xs)
    med = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    devs = sorted(abs(x - med) for x in values)
    mad = devs[n // 2] if n % 2 else (devs[n // 2 - 1] + devs[n // 2]) / 2
    sigma = SCALE * mad
    if sigma == 0:
        # every layer identical, or so tight MAD underflows. A z of 0 is the honest answer:
        # "no spread to measure", not "no outlier" -- the caller prints sigma so a reader
        # can tell the two apart.
        return med, 0.0, [0.0] * n
    return med, sigma, [(x - med) / sigma for x in values]


def layer_of(key):
    m = re.match(r"blocks\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def load(path):
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    step = sd.get("step")
    return sd.get("model", sd), step


def norms(model):
    """key -> L2 norm, float32 regardless of stored dtype (bf16 norm of a big tensor
    accumulates visible error; the ratio we report is between two such numbers)."""
    import torch

    out = {}
    for k, v in model.items():
        if not hasattr(v, "shape") or v.numel() == 0:
            continue
        out[k] = float(torch.linalg.vector_norm(v.to(torch.float32)).item())
    return out


def report(a_path, b_path, sigma_thresh):
    import torch

    ma, step_a = load(a_path)
    mb, step_b = load(b_path)
    shared = [k for k in ma if k in mb]
    missing = sorted(set(ma) ^ set(mb))
    print(f"# checkpoint health: step {step_a} -> step {step_b}")
    print(f"#   {len(shared)} shared tensors" + (f", {len(missing)} only in one" if missing else ""))
    if missing:
        for k in missing[:8]:
            print(f"#   ONLY IN ONE: {k}")

    na, nb = norms(ma), norms(mb)

    # --- per-layer weight-norm ratio, one number per block ---
    per_layer = {}
    for k in shared:
        li = layer_of(k)
        if li is None or k not in na or na[k] == 0:
            continue
        per_layer.setdefault(li, []).append(nb[k] / na[k])
    layers = sorted(per_layer)
    ratios = [sum(v) / len(v) for _, v in sorted(per_layer.items())]
    med, sig, zs = robust_z(ratios)
    print(f"\n## per-layer weight-norm ratio (step{step_b}/step{step_a})")
    print(f"median {med:.6f}  sigma(MAD) {sig:.2e}  layers {len(layers)}")
    flagged = [(li, r, z) for li, r, z in zip(layers, ratios, zs) if abs(z) > sigma_thresh]
    for li, r, z in flagged:
        print(f"  FLAG layer {li:2d}: ratio {r:.6f}  z {z:+.2f}")
    if not flagged:
        print(f"  no layer beyond {sigma_thresh} sigma")

    # --- KDA decay: A_log is per-head, exp(A_log) is the decay rate ---
    print("\n## KDA decay (A_log per head)")
    for label, m in (("step%s" % step_a, ma), ("step%s" % step_b, mb)):
        vals = []
        for k, v in m.items():
            if k.endswith("mixer.A_log"):
                vals += v.to(torch.float32).flatten().tolist()
        if not vals:
            print(f"  {label}: no A_log found")
            continue
        vals.sort()
        decay = [pow(2.718281828459045, x) for x in vals]
        print(f"  {label}: n={len(vals)}  A_log min {vals[0]:+.4f} med {vals[len(vals)//2]:+.4f} "
              f"max {vals[-1]:+.4f} | exp() min {min(decay):.4f} max {max(decay):.4f}")

    # --- embedding norm ---
    print("\n## embedding")
    for k in ("tok.weight", "head.weight"):
        if k in na:
            r = nb[k] / na[k] if na[k] else float("nan")
            print(f"  {k}: {na[k]:.2f} -> {nb[k]:.2f}  ratio {r:.6f}")

    # --- branch output scale: KDA mixer.o vs the FFN's second matrix, per layer ---
    print("\n## branch output-scale ratio (KDA mixer.o / ffn output), later step")
    pairs = []
    for li in layers:
        mo = nb.get(f"blocks.{li}.mixer.o.weight")
        fo = next((nb[k] for k in (f"blocks.{li}.ffn.w2.weight", f"blocks.{li}.ffn.o.weight",
                                   f"blocks.{li}.ffn.down.weight") if k in nb), None)
        if mo and fo:
            pairs.append((li, mo / fo))
    if pairs:
        med2, sig2, zs2 = robust_z([p[1] for p in pairs])
        print(f"median {med2:.6f}  sigma(MAD) {sig2:.2e}")
        for (li, r), z in zip(pairs, zs2):
            if abs(z) > sigma_thresh:
                print(f"  FLAG layer {li:2d}: mixer.o/ffn {r:.6f}  z {z:+.2f}")
    else:
        print("  no (mixer.o, ffn-out) pair matched -- key names changed, update this block")
    return 0


def selftest():
    # robust_z on a planted outlier: 11 values, ten tight and one far. The point of the
    # check is that the outlier does NOT inflate the sigma used to judge it, which is what
    # a mean/std version does and why this uses median/MAD.
    vals = [1.0, 1.001, 0.999, 1.0, 1.002, 0.998, 1.001, 0.999, 1.0, 1.001, 1.5]
    med, sig, zs = robust_z(vals)
    assert abs(med - 1.0) < 0.002, med
    assert zs[-1] > 100, f"planted outlier should be far beyond 3 sigma, got z={zs[-1]:.1f}"
    assert all(abs(z) < 3 for z in zs[:-1]), "the tight values must not be flagged"

    # a mean/std version on the SAME data, to show what it would have reported
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    z_mean = (vals[-1] - mean) / (var ** 0.5)
    # A single outlier's z under population std CANNOT exceed sqrt(n-1) -- 3.162 at n=11 --
    # no matter how far out it sits, because the same point inflates the std it is divided by.
    # (My first version of this comment said the bound was (n-1)/sqrt(n) = 3.02; that is the
    # SAMPLE-std, ddof=1, ceiling. The code here divides by population std, so 3.162 is the
    # right number and the measured 3.16 is sitting on it, not near it.)
    #
    # So the mean/std failure mode is not "hides the outlier" -- at n=11 it clears a 3-sigma
    # bar -- but "reports the same number for a 1.5 and for a 15.0", and saturates below the
    # bar entirely once n < 10. MAD has no such ceiling: it scales with the distance.
    n = len(vals)
    ceiling = (n - 1) ** 0.5
    assert z_mean < ceiling + 1e-9, f"{z_mean} must not exceed sqrt(n-1)={ceiling}"
    assert ceiling - z_mean < 0.01, f"one far outlier should sit AT the ceiling, gap {ceiling - z_mean}"
    far = vals[:-1] + [15.0]
    m2 = sum(far) / len(far)
    z_far = (far[-1] - m2) / ((sum((x - m2) ** 2 for x in far) / len(far)) ** 0.5)
    assert z_far < ceiling + 1e-9, f"10x further out and still capped: {z_far:.4f} vs {ceiling:.4f}"
    _, _, zs_far = robust_z(far)
    # MAD z is LINEAR in the distance: the median and MAD are set by the ten tight values and
    # do not move when the outlier moves, so z tracks it exactly. 28x further out -> 28x the z.
    # (My first assertion here demanded > 100x, a number with no derivation behind it; the
    # measured ratio is 28.0 against a distance ratio of 28.0, which is the property worth
    # asserting -- proportionality, not a magnitude I picked.)
    dist_ratio = (15.0 - med) / (1.5 - med)
    z_ratio = zs_far[-1] / zs[-1]
    assert abs(z_ratio / dist_ratio - 1) < 0.01, f"z ratio {z_ratio:.2f} vs distance {dist_ratio:.2f}"

    # degenerate spread must return sigma 0 and z 0, not divide by zero
    med0, sig0, zs0 = robust_z([2.0] * 7)
    assert sig0 == 0.0 and set(zs0) == {0.0}

    assert layer_of("blocks.11.mixer.o.weight") == 11
    assert layer_of("tok.weight") is None
    print(f"selftest OK: outlier at 1.5 -> z {zs[-1]:.0f} (MAD) vs {z_mean:.3f} (mean/std); "
          f"at 15.0 -> {zs_far[-1]:.0f} vs {z_far:.3f}. One outlier's mean/std z is capped at "
          f"sqrt(n-1)={ceiling:.3f}, so it reports the same severity for a 10x worse layer; "
          f"MAD is linear in it ({z_ratio:.1f}x z for {dist_ratio:.1f}x distance)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("earlier", nargs="?")
    ap.add_argument("later", nargs="?")
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.earlier and a.later):
        ap.error("need two checkpoint paths, or --selftest")
    return report(a.earlier, a.later, a.sigma)


if __name__ == "__main__":
    sys.exit(main())

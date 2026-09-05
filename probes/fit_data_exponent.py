#!/usr/bin/env python3
"""Fit L(D) = a*D^-b + c to a run's val trace, and say whether b is IDENTIFIED.

The point of this script is the second half. A least-squares fit always returns a b; whether
that b is a measurement depends on how much of (b, c) space fits the data within its own
noise. On the b0 arms' 7-point traces the answer is that b is not identified at all -- any
b from 0.11 to 1.00 fits within the traces' noise -- so the script reports the admissible
INTERVAL and refuses to present a point estimate as a measurement.

Why this matters for a design decision: sample efficiency is a horizontal shift r on this
curve, and the vertical gap a shift produces is R*(1 - r^-b) with R = a*D^-b the reducible
loss still on the table. Every statement of the form "a 2x data advantage shows as X nat"
is proportional to b. Reading a fitted b as measured, when the data cannot constrain it,
turns an unfalsifiable number into an eval-set sizing decision.

Usage:
  fit_data_exponent.py --trace 500:2.917,1000:2.589,... --tok_per_step 262144
  fit_data_exponent.py --log runs/b0_headmix_armA.log --tok_per_step 262144
  fit_data_exponent.py --selftest
"""
import argparse
import math
import os
import re
import sys


def parse_log(path, total_steps=None):
    """Val points from a train.py log: `step <s>/<total> val <loss>`.

    The epoch-end line (`ep 1/1 train ... val ...`) is DELIBERATELY EXCLUDED: it is computed
    with Cfg.val_batches_full (100 batches) while the periodic lines use Cfg.val_batches
    (20), so it is a different estimator on a different sample. Mixing the two puts a point
    with a different noise level and a different bias at the end of the range, exactly where
    the fit is most sensitive to it.
    """
    pts = {}
    rx = re.compile(r"^step (\d+)/(\d+) val ([\d.]+)\s*$")
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = rx.match(line.strip())
            if m:
                s, tot, v = int(m.group(1)), int(m.group(2)), float(m.group(3))
                if total_steps is None or tot == total_steps:
                    pts[s] = v
    return pts


def parse_trace(spec):
    out = {}
    for part in spec.split(","):
        s, v = part.split(":")
        out[int(s)] = float(v)
    return out


def fit_a(pts, tok, c, b):
    """SSE and the closed-form a at fixed (c, b).

    The fit is in LOSS space, not log space. `a` enters L = a*D^-b + c linearly once c and b
    are fixed, so a is a one-line least-squares solution and no optimiser is needed. Fitting
    log(L - c) instead reweights the points -- the small-residual end dominates -- and on
    these traces that reweighting alone drives the best c to 0, which is how a pure power law
    wins a comparison it should not.
    """
    xs = [(s * tok) ** -b for s in sorted(pts)]
    ys = [pts[s] - c for s in sorted(pts)]
    den = sum(x * x for x in xs)
    if den <= 0:
        return float("inf"), 0.0
    a = sum(x * y for x, y in zip(xs, ys, strict=True)) / den
    return sum((a * x - y) ** 2 for x, y in zip(xs, ys, strict=True)), a


def noise_estimate(pts):
    """Per-point noise from the trace's own roughness, via second differences.

    A loss curve is smooth and convex in log D, so its second difference is small and its
    SCATTER is dominated by measurement error rather than by curvature. For independent
    errors of SD s, Var(second difference) = 6 s^2, hence the sqrt(6). This is an estimate,
    and an UPPER bound on the noise: real curvature inflates it. That direction is the safe
    one here -- it widens the admissible interval rather than narrowing it.
    """
    ks = sorted(pts)
    v = [pts[k] for k in ks]
    if len(v) < 3:
        return float("nan")
    d2 = [v[i + 1] - 2 * v[i] + v[i - 1] for i in range(1, len(v) - 1)]
    return math.sqrt(sum(x * x for x in d2) / len(d2)) / math.sqrt(6)


def admissible(pts, tok, noise, b_lo=0.005, b_hi=1.0, b_steps=400, c_steps=460):
    """Every (b, c) whose RMSE is within `noise`, plus the best fit.

    Returns (b_min, b_max, n_pairs, best) where best is (rmse, c, b, a).
    """
    lo_loss = min(pts.values())
    best = None
    bs = []
    n = len(pts)
    for i in range(b_steps + 1):
        b = b_lo + (b_hi - b_lo) * i / b_steps
        hit = False
        for j in range(c_steps + 1):
            c = lo_loss * j / c_steps
            if c >= lo_loss:
                break
            sse, a = fit_a(pts, tok, c, b)
            rmse = math.sqrt(sse / n)
            if best is None or rmse < best[0]:
                best = (rmse, c, b, a)
            if rmse <= noise:
                hit = True
        if hit:
            bs.append(b)
    if not bs:
        return None, None, 0, best
    return min(bs), max(bs), len(bs), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="a train.py run log to read `step N/T val X` lines from")
    ap.add_argument("--trace", help="inline points, step:loss,step:loss,...")
    ap.add_argument("--tok_per_step", type=int, required=False, default=None,
                    help="batch x accum x seq x world; the x-axis is tokens, not steps")
    ap.add_argument("--noise", type=float, default=None,
                    help="per-point noise in nat; default is estimated from the trace")
    ap.add_argument("--name", default="trace")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if not (a.log or a.trace):
        sys.exit("--log or --trace is required")
    if not a.tok_per_step:
        sys.exit("--tok_per_step is required: b is an exponent on TOKENS, and fitting against "
                 "step index silently rescales it by ln(tok_per_step)")

    pts = parse_trace(a.trace) if a.trace else parse_log(a.log)
    if len(pts) < 3:
        sys.exit(f"only {len(pts)} val point(s); a 3-parameter curve needs more")
    tok = a.tok_per_step
    noise = a.noise if a.noise is not None else noise_estimate(pts)
    ds = sorted(s * tok for s in pts)
    print(f"{a.name}: {len(pts)} points, {ds[0] / 1e9:.3f}B -> {ds[-1] / 1e9:.3f}B tokens "
          f"({ds[-1] / ds[0]:.1f}x range)")
    print(f"  per-point noise: {noise:.4f} nat "
          f"({'given' if a.noise is not None else 'estimated from second differences'})")
    b_lo, b_hi, npair, best = admissible(pts, tok, noise)
    rmse, c, b, aa = best
    print(f"  best fit: b={b:.4f} c={c:.4f} a={aa:.4g} RMSE={rmse:.5f} nat")
    if b_lo is None:
        print("  NO (b,c) fits within the noise -- the model is rejected on this range")
        return 0
    print(f"  admissible b within noise: [{b_lo:.3f}, {b_hi:.3f}] over {npair} b-values")
    width = b_hi / b_lo if b_lo > 0 else float("inf")
    if width > 2.0:
        print(f"  VERDICT: b is NOT IDENTIFIED. The interval spans {width:.1f}x, so the best-fit "
              f"b={b:.4f} is not a measurement and must not be cited as one. Any design number "
              f"proportional to b inherits this whole range.")
    else:
        print(f"  VERDICT: b is identified to within {width:.2f}x on this range.")
    return 0


def _selftest():
    TOK = 262144
    steps = [500, 1000, 1500, 2000, 2500, 3000, 3500]

    # 1. NOISELESS RECOVERY. A correct fitter must return the truth when there is no noise;
    #    without this, an unidentifiability verdict below could just be a broken fitter.
    #
    #    TOLERANCES ARE THE GRID'S, NOT THE ONES I WANTED. admissible() searches b on a
    #    0.0025 grid and c on a min(loss)/460 ~= 0.005 grid, so the best GRID POINT is offset
    #    from the truth by up to a few cells: at b_true=0.13 the truth has RMSE 2e-16 while
    #    the grid's best is 3.5e-5 at c off by 0.021. That is discretization, not a fitting
    #    error, and asserting a tighter number would only be asserting a finer grid. The
    #    valley is genuinely flat -- which is the same geometry that makes b unidentifiable
    #    at real noise, so it belongs in the record rather than being tuned away.
    for b_true, c_true in ((0.13, 1.0), (0.30, 1.6), (0.50, 1.9)):
        a_true = (2.4 - c_true) / ((2000 * TOK) ** -b_true)
        pts = {s: a_true * (s * TOK) ** -b_true + c_true for s in steps}
        lo, hi, _, best = admissible(pts, TOK, 1e-6)
        assert abs(best[2] - b_true) < 0.01, f"noiseless fit missed b: {best[2]} vs {b_true}"
        assert abs(best[1] - c_true) < 0.03, f"noiseless fit missed c: {best[1]} vs {c_true}"
        # And the residual at the grid's best must be near-zero: a fitter that is merely
        # CLOSE on (b, c) but fits badly would pass the two assertions above.
        assert best[0] < 1e-3, f"noiseless RMSE too large: {best[0]}"
        # The truth itself must be admissible at any sane noise floor. This is the assertion
        # that a wrong objective (log-space) fails while still landing near b_true.
        assert math.sqrt(fit_a(pts, TOK, c_true, b_true)[0] / len(pts)) < 1e-9

    # 2. AND THE SAME DESIGN LOSES b AT REALISTIC NOISE. This is the finding, checked as a
    #    property of the design rather than asserted: 7 points over 7x at 0.039 nat cannot
    #    constrain b, whatever the truth is.
    offs = [+1, -1, +1, -1, +1, -1, +1]  # worst case, alternating: deterministic, no RNG
    for b_true, c_true in ((0.13, 1.0), (0.30, 1.6)):
        a_true = (2.4 - c_true) / ((2000 * TOK) ** -b_true)
        pts = {s: a_true * (s * TOK) ** -b_true + c_true + o * 0.039
               for s, o in zip(steps, offs, strict=True)}
        lo, hi, _, _ = admissible(pts, TOK, 0.039)
        assert hi / lo > 3.0, f"expected b unidentified at 0.039 nat, got [{lo},{hi}]"

    # 3. A LONGER LEVER IDENTIFIES IT. If the verdict were an artifact of the method rather
    #    than of the data, widening the token range would not help. 100x with the same noise
    #    must narrow the interval, or the script is measuring itself.
    wide = [500 * 2 ** i for i in range(8)]  # 500 -> 64000, 128x
    b_true, c_true = 0.30, 1.6
    a_true = (2.4 - c_true) / ((2000 * TOK) ** -b_true)
    pts = {s: a_true * (s * TOK) ** -b_true + c_true + (0.039 if i % 2 else -0.039)
           for i, s in enumerate(wide)}
    lo_w, hi_w, _, _ = admissible(pts, TOK, 0.039)
    narrow = [500, 1000, 1500, 2000, 2500, 3000, 3500]
    pts_n = {s: a_true * (s * TOK) ** -b_true + c_true + (0.039 if i % 2 else -0.039)
             for i, s in enumerate(narrow)}
    lo_n, hi_n, _, _ = admissible(pts_n, TOK, 0.039)
    assert (hi_w / lo_w) < (hi_n / lo_n), (
        f"a 128x range did not beat a 7x range ({hi_w / lo_w:.1f} vs {hi_n / lo_n:.1f}); "
        f"the verdict would then be an artifact of the method, not the data"
    )

    # 4. THE LOSS-SPACE OBJECTIVE IS WHAT MAKES c IDENTIFIABLE AT ALL. Fitting log(L - c)
    #    reweights the points and drives c to 0 on a curve that has a real floor. Checked by
    #    building a curve with c=1.5 and confirming loss-space recovers it.
    a_true = (2.4 - 1.5) / ((2000 * TOK) ** -0.3)
    pts = {s: a_true * (s * TOK) ** -0.3 + 1.5 for s in steps}
    _, _, _, best = admissible(pts, TOK, 1e-6)
    assert abs(best[1] - 1.5) < 0.02, f"loss-space fit lost the floor: c={best[1]}"

    # 5. THE EPOCH-END LINE IS EXCLUDED BY parse_log. It uses val_batches_full (100) where
    #    the periodic lines use val_batches (20) -- a different estimator, and it sits at the
    #    end of the range where the fit is most sensitive.
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write("step 500/3815 val 2.917\n")
        fh.write("noise line\n")
        fh.write("step 1000/3815 val 2.589\n")
        fh.write("ep 1/1 train 1.768 val 2.117 6968s\n")
        p = fh.name
    try:
        got = parse_log(p)
        assert got == {500: 2.917, 1000: 2.589}, got
        assert 3815 not in got, "the epoch-end val leaked into the trace"
    finally:
        os.unlink(p)

    # 6. NOISE FROM SECOND DIFFERENCES: exact on a linear ramp (second difference 0 -> noise
    #    0), and it must not read curvature alone as huge noise.
    assert noise_estimate({1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}) == 0.0
    n_est = noise_estimate({500: 2.917, 1000: 2.589, 1500: 2.464, 2000: 2.399,
                            2500: 2.343, 3000: 2.284, 3500: 2.241})
    assert 0.03 < n_est < 0.05, f"armA noise estimate moved: {n_est}"

    # 7. THE X-AXIS IS TOKENS, NOT STEPS, and `a` absorbs the difference so nothing goes wrong
    #    visibly. Every world above builds its curve and fits it in the same units, so all of
    #    them stay green if fit_a silently drops `tok` -- verified: that mutant was the one
    #    survivor of a five-mutant run. b is invariant to the rescaling but `a` is not, by
    #    exactly tok^-b, so the units are pinned through a: a fit against step index returns
    #    a_steps = a_tokens * tok^-b.
    #
    #    Why it matters rather than being a units nicety: two runs at different world sizes
    #    have different tokens-per-step, so a step-indexed `a` is not comparable across them,
    #    and the reducible term R = a*D^-b that every design number is proportional to would
    #    be read off the wrong axis.
    b_t, c_t = 0.30, 1.6
    a_tok = (2.4 - c_t) / ((2000 * TOK) ** -b_t)
    pts = {s: a_tok * (s * TOK) ** -b_t + c_t for s in steps}
    sse_tok, a_fit = fit_a(pts, TOK, c_t, b_t)
    assert sse_tok < 1e-18, sse_tok
    assert abs(a_fit - a_tok) / a_tok < 1e-9, (a_fit, a_tok)
    # The same points fitted against the step index recover a DIFFERENT a, by tok^-b. If
    # fit_a ignored `tok`, these two would be equal and this assertion fires.
    _, a_steps = fit_a(pts, 1, c_t, b_t)
    assert abs(a_steps - a_tok * TOK ** -b_t) / a_steps < 1e-9, (a_steps, a_tok)
    assert a_steps < a_tok / 10, (
        f"a is unchanged by the token scaling ({a_steps} vs {a_tok}): fit_a is fitting against "
        f"the step index, so `a` and the reducible term R are on the wrong axis"
    )

    print("fit_data_exponent selftest OK: the fitter recovers b to within 0.01 and c to within "
          "0.03 (the grid's resolution, with the truth itself fitting to 1e-9) at zero noise on "
          "three truths, the same 7-point/7x design loses b at 0.039 nat whatever the truth "
          "(interval > 3x), a 128x range narrows it again so the verdict is the data's and not "
          "the method's, the loss-space objective recovers a c=1.5 floor that a log-space fit "
          "drives to 0, parse_log excludes the epoch-end val line (a different estimator), the "
          "second-difference noise reads 0 on a ramp and 0.039 on armA, and the x-axis is "
          "pinned to TOKENS through `a` (a step-indexed fit returns a * tok^-b, which b alone "
          "cannot detect)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

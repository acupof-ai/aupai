#!/usr/bin/env python3
"""Fit L(D) = E + B/D^beta over the six budget points, per the frozen protocol
docs/lessons/scaling_fit_protocol.md. Huber delta=1e-3, L-BFGS-B, 252-start
grid (Chinchilla recipe, same loss scale). Diagnostics: residual sign patterns
(U-shape, monotone drift), LOO beta stability, beta profile CI, verdict.

Usage:
    python scripts/fit_scaling.py --points points.json
    python scripts/fit_scaling.py --selftest

points.json: {"0.2b": {"D": 2e8, "loss": 2.01}, ...}
"""

import argparse
import json
import sys

import numpy as np
from scipy.optimize import minimize

DELTA = 1e-3  # Chinchilla: same loss scale (theirs 1.69-3, ours 1.66-1.99)
B_GRID = [100, 300, 1000, 3000, 10000, 30000]  # D in raw tokens: B~1e3-1e4 for a ~0.3-nat span
BETA_GRID = [0.25 * i for i in range(1, 7)]  # 0.25..1.5
E_GRID = [1.5 + 0.1 * i for i in range(6)]   # 1.5..2.0


def huber(resid, delta=DELTA):
    a = np.abs(resid)
    return np.where(a <= delta, 0.5 * resid**2, delta * (a - 0.5 * delta)).sum()


def model(D, E, B, beta):
    return E + B / D**beta


def fit(D, L, grid_start=True):
    """Best of the 252-start grid + L-BFGS-B polish. Returns (params, obj)."""
    D = np.asarray(D, dtype=float)
    L = np.asarray(L, dtype=float)
    best = None
    starts = [(E, B, beta) for E in E_GRID for B in B_GRID for beta in BETA_GRID] if grid_start else [(1.8, 50.0, 0.5)]
    for E0, B0, beta0 in starts:
        r = minimize(lambda p: huber(L - model(D, *p)), [E0, B0, beta0],
                     method="L-BFGS-B", bounds=[(None, None), (0, None), (0.01, 3.0)])
        if best is None or r.fun < best[1]:
            best = (r.x, float(r.fun))
    return best


def beta_profile_ci(D, L, best_obj, best_params):
    """95% profile CI: beta values with min_obj(beta) <= best_obj + 3.84."""
    D = np.asarray(D, dtype=float)
    L = np.asarray(L, dtype=float)

    def min_at_beta(beta):
        r = minimize(lambda p: huber(L - model(D, p[0], p[1], beta)),
                     [best_params[0], best_params[1]], method="L-BFGS-B",
                     bounds=[(None, None), (0, None)])
        return r.fun

    betas = np.linspace(0.05, 1.8, 120)
    objs = [min_at_beta(b) for b in betas]
    inside = [b for b, o in zip(betas, objs) if o <= best_obj + 3.84]
    return (min(inside), max(inside)) if inside else (None, None)


def diagnose(D, L, params):
    """Protocol §3/§4: RMS/max, sign patterns, LOO beta stability, verdict."""
    D = np.asarray(D, dtype=float)
    L = np.asarray(L, dtype=float)
    order = np.argsort(D)
    resid = L - model(D, *params)
    rms = float(np.sqrt(np.mean(resid**2)))
    mx = float(np.max(np.abs(resid)))
    signs = np.sign(resid[order])
    pat = "".join("+" if s > 0 else "-" for s in signs)

    # §4.1 U/inverted-U: exact 6-point pattern ++--++ / --++--, endpoints > delta.
    # p = 2/64 = 3.1% under random signs.
    u = np.array([1, 1, -1, -1, 1, 1])
    u_shape = bool((np.all(signs == u) or np.all(signs == -u))
                   and abs(resid[order][0]) > DELTA and abs(resid[order][-1]) > DELTA)
    # §4.2 monotone drift: run of >=4 same sign, all > 0.01 in magnitude
    big = np.abs(resid[order]) > 0.01
    drift = False
    for i in range(len(signs) - 3):
        if np.all(signs[i:i+4] == signs[i]) and np.all(big[order][i:i+4]):
            drift = True
    # §4.3 LOO beta stability -- imprecision flag, NOT a form falsifier: the form
    # can hold (small RMS) while beta is poorly identified. Reported separately.
    loo = []
    for k in range(len(D)):
        m = np.arange(len(D)) != k
        p_loo, _ = fit(D[m], L[m])
        loo.append(abs(params[2] - p_loo[2]))
    beta_unstable = bool(max(loo) > 0.15)
    # §4.4 endpoints same sign, both > delta
    endpoints_fail = signs[0] == signs[-1] and abs(resid[order][0]) > DELTA and abs(resid[order][-1]) > DELTA

    # form falsifiers only (protocol §4.1/4.2/4.4); beta instability is §4.3
    falsified = [n for n, hit in [("u_shape", u_shape), ("monotone_drift", drift),
                                  ("endpoints_fail", endpoints_fail)] if hit]
    verdict = "FAIL" if rms > 0.10 else ("WEAK" if rms > 0.05 else "ACCEPT")
    if falsified:
        verdict = "FAIL"  # §4: form falsified regardless of RMS
    return {
        "residuals": [float(x) for x in resid[order]],
        "sign_pattern": pat,
        "rms": rms, "max_abs": mx,
        "loo_beta_max_delta": float(max(loo)),
        "beta_unstable": beta_unstable,
        "falsified": falsified,
        "verdict": verdict,
    }


def _selftest():
    E, B, beta = 1.8, 6000.0, 0.5  # span ~0.32 nat over the ladder, like the real 1.66-1.99
    D = np.array([0.2e9, 0.3e9, 0.4e9, 0.8e9, 1.6e9, 3.24e9])
    # 1. noise-free: exact recovery (plumbing)
    L0 = model(D, E, B, beta)
    p0, _ = fit(D, L0)
    assert abs(p0[0] - E) < 0.02 and abs(p0[2] - beta) < 0.05, p0
    # 2. noisy: verdict must be ACCEPT, beta in the right ballpark (6 points, L1-ish)
    rng = np.random.RandomState(0)
    L = L0 + rng.normal(0, 0.01, len(D))
    params, _ = fit(D, L)
    diag = diagnose(D, L, params)
    assert diag["verdict"] == "ACCEPT", diag
    assert abs(params[2] - beta) < 0.25, params
    # 3. pattern detection on controlled residuals (a 3-param fit absorbs part of
    # a U-shape; the falsifier must fire on what the fit cannot absorb)
    p_star = np.array([E, B, beta])
    base = model(D, *p_star)
    diag_u = diagnose(D, base + np.array([0.05, 0.05, -0.05, -0.05, 0.05, 0.05]), p_star)
    assert "u_shape" in diag_u["falsified"], diag_u
    diag_d = diagnose(D, base + np.array([0.05, 0.05, 0.05, 0.05, -0.05, -0.05]), p_star)
    assert "monotone_drift" in diag_d["falsified"], diag_d
    diag_e = diagnose(D, base + np.array([0.05, -0.02, -0.02, -0.02, -0.02, 0.05]), p_star)
    assert "endpoints_fail" in diag_e["falsified"], diag_e
    print(f"fit_scaling self-test OK (noise-free: E={p0[0]:.3f} B={p0[1]:.0f} beta={p0[2]:.3f}; "
          f"noisy ACCEPT beta={params[2]:.2f}; U/drift/endpoint patterns detected)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", help="json: {name: {D, loss}}")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.points:
        ap.error("--points required (or --selftest)")
    pts = json.load(open(a.points))
    names = sorted(pts, key=lambda n: pts[n]["D"])
    D = [pts[n]["D"] for n in names]
    L = [pts[n]["loss"] for n in names]
    params, obj = fit(D, L)
    lo, hi = beta_profile_ci(D, L, obj, params)
    result = {
        "points": names,
        "E": float(params[0]), "B": float(params[1]), "beta": float(params[2]),
        "beta_ci95": [lo, hi],
        "huber_objective": obj,
        **diagnose(D, L, params),
    }
    if a.out:
        json.dump(result, open(a.out, "w"), indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

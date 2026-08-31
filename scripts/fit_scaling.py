#!/usr/bin/env python3
"""Six-point scaling fit per docs/lessons/scaling_fit_protocol.md v1.10 (frozen).

L(D) = E + B/D^beta, Huber delta=1e-3, L-BFGS-B, 252-start grid (protocol S2).
Prints: parameters + profile-likelihood beta CI, RMS/max resid, verdict per S9
(ACCEPT 0.0588 / WEAK 0.1176, de-label 0.0366), S4 falsifier forms, local slope
at the largest point per e-fold (for tilerl's tolerance).

Usage: python scripts/fit_scaling.py
"""
import numpy as np
from scipy.optimize import minimize

# (D_scheduled in tokens, agg val NLL). p324 D from the train log line
# (3.1955B scheduled vs 3.2449B requested, gap 1.523% < 10% -> absorbed, S11);
# the five small points have no cap (protocol S1: verified).
D = np.array([0.2e9, 0.3e9, 0.4e9, 0.8e9, 1.6e9, 3.1955e9])
L = np.array([3.691, 3.322, 3.159, 2.792, 2.570, 2.415])
D_REQUESTED = np.array([0.2e9, 0.3e9, 0.4e9, 0.8e9, 1.6e9, 3.2449e9])

BETA_GRID = [0.25 * i for i in range(7)]            # 0 .. 1.5
E_GRID = [1.5 + 0.1 * i for i in range(6)]          # 1.5 .. 2.0
B_GRID = [100, 300, 1000, 3000, 10000, 30000]
HUBER_D = 1e-3
SIGMA_HAT = 0.0516  # ds.seed_variance_0p2b, df=3


def huber(r, d=HUBER_D):
    a = np.abs(r)
    return np.where(a <= d, 0.5 * r**2, d * (a - 0.5 * d)).sum()


def fit(D, L, beta_fixed=None):
    best = None
    for b0 in BETA_GRID if beta_fixed is None else [beta_fixed]:
        for e0 in E_GRID:
            for B0 in B_GRID:
                if beta_fixed is None:
                    x0 = [e0, B0, b0]
                    bounds = [(None, None), (1e-6, None), (1e-6, 5.0)]

                    def obj(p):
                        return huber(L - (p[0] + p[1] / D**p[2]))
                else:
                    x0 = [e0, B0]
                    bounds = [(None, None), (1e-6, None)]

                    def obj(p):
                        return huber(L - (p[0] + p[1] / D**beta_fixed))
                r = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
                if best is None or r.fun < best.fun:
                    best = r
    return best


def resid(D, L, p):
    return L - (p[0] + p[1] / D**p[2])


res = fit(D, L)
E, B, beta = res.x
r = resid(D, L, res.x)
rms = float(np.sqrt((r**2).mean()))
maxr = float(np.abs(r).max())
print(f"E={E:.4f}  B={B:.1f}  beta={beta:.4f}  (Huber obj={res.fun:.6f})")
print(f"RMS={rms:.4f}  max|resid|={maxr:.4f}")
print("residuals by point:", " ".join(f"{v:+.4f}" for v in r))
print("sign pattern:", "".join("+" if v > 0 else "-" for v in r))

# S9 verdict
if rms <= 1.14 * SIGMA_HAT:
    verdict = "ACCEPT" + (" (de-labeled: RMS < 0.71 sigma-hat)" if rms < 0.71 * SIGMA_HAT else " (noise-loose)")
elif rms <= 2.28 * SIGMA_HAT:
    verdict = "WEAK"
else:
    verdict = "FAIL"
print(f"verdict: {verdict}  (ACCEPT<=0.0588, WEAK<=0.1176)")

# S4 falsifier forms
signs = "".join("+" if v > 0 else "-" for v in r)
u_shape = signs in ("++--++", "--++--") and abs(r[0]) > 0.05 and abs(r[-1]) > 0.05
print(f"S4.1 U/inverted-U: {u_shape}")
runs = []
cur = signs[0]
n = 1
for s in signs[1:]:
    if s == cur:
        n += 1
    else:
        runs.append((cur, n))
        cur, n = s, 1
runs.append((cur, n))
drift = False
i = 0
for c, n2 in runs:
    if n2 >= 4 and all(abs(r[j]) > 0.01 for j in range(i, i + n2)):
        drift = True
    i += n2
print(f"S4.2 monotone drift (>=4 same sign, all |r|>0.01): {drift}  runs={runs}")
loo = []
for k in range(6):
    m = np.arange(6) != k
    rk = fit(D[m], L[m])
    loo.append(rk.x[2])
    slope_k = -rk.x[2] * (L[-1] - rk.x[0])
    print(f"  LOO drop p{k+1} (D={D[k]/1e9:.3f}b): beta={rk.x[2]:.4f} E={rk.x[0]:.4f} slope@p324={slope_k:.4f}  (|dbeta|={abs(beta - rk.x[2]):.4f})")
print(f"S4.3 LOO beta unstable (>0.15): {any(abs(beta - b) > 0.15 for b in loo)}")
endpt = abs(r[0]) > 0.05 and abs(r[-1]) > 0.05 and np.sign(r[0]) == np.sign(r[-1])
print(f"S4.4 endpoints same sign, both |r|>0.05: {endpt}")

# profile-likelihood CI for beta (Delta = 3.84, chi2_1 95%)
target = res.fun + 3.84
lo_beta, hi_beta = None, None
grid = np.linspace(0.01, 5.0, 999)
prof = []
for bf in grid:
    rf = fit(D, L, beta_fixed=bf)
    prof.append(rf.fun)
prof = np.array(prof)
below = np.where(prof <= target)[0]
if len(below):
    lo_beta, hi_beta = grid[below[0]], grid[below[-1]]
if lo_beta == grid[0] and hi_beta == grid[-1]:
    print("profile 95% CI beta: VACUOUS (spans full grid) -- Huber objective is L1 at this")
    print("  noise scale (all |r|>1e-3), obj~1e-4, so Delta=3.84 dwarfs the surface; the")
    print("  L2 chi2 construct does not apply (protocol S9 Huber warning, same phenomenon).")
    print("  beta identification rests on LOO stability above.")
else:
    print(f"profile 95% CI beta: [{lo_beta}, {hi_beta}]  (informative iff inside (0, 1.5))")

# local slope at the largest point, per e-fold (tilerl's tolerance input)
slope_ef = -beta * (L[-1] - E)
print(f"local slope at 3.1955b: {slope_ef:.4f} nat/e-fold  (= -beta*(L-E), L-E={L[-1]-E:.4f})")

# D gaps per point (S1 record)
for k in range(6):
    gap = (D_REQUESTED[k] - D[k]) / D_REQUESTED[k] * 100
    print(f"D[{k}] requested={D_REQUESTED[k]/1e9:.4f}b scheduled={D[k]/1e9:.4f}b gap={gap:.3f}%")

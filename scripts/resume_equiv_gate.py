#!/usr/bin/env python3
# restartable: read-only judge over a dump directory; --selftest is numpy-only, <1s, writes
# only tempfiles it removes. An interrupted calibration run leaves one JSON the caller named.
"""Resume-equivalence gate: judge a save/load resume at the granularity that is reproducible.

Why this exists (D17 / D61 / K6, 2026-09-22)
--------------------------------------------
The old gate (tests/v41f/test_p1_train_ckpt.py::gate_resume_equivalent_to_uninterrupted)
ran control and save/load-restart as TWO SEPARATE PROCESSES and required the fp32 master to
be BIT-IDENTICAL. That is the wrong contract at the wrong granularity:

* The fp32 master accumulates sub-bf16 information. Independent runner processes run
  non-deterministic bf16 forward/backward kernels (atomic reductions); the resulting bf16
  gradients differ process-to-process, are cast into the fp32 master, and AdamW compounds
  the difference over steps. Measured (red run 35677058625): 96.6% of fp32 master elements
  differ and 51.5% even after a bf16 cast, yet the TENSORS agree -- relative-L2 ~1.6%,
  cosine ~0.99987. Per-coordinate equality is not reproducible across runner processes;
  whole-tensor (aggregate / geometric) equivalence is.

So this gate splits the verdict into TWO layers that answer two different questions:

1. DETERMINISTIC STATE (bit-exact, hard fail). The things a checkpoint genuinely promises
   to reproduce and that do NOT depend on kernel nondeterminism:
     - the saved/restored optimizer triple per leaf (exp_avg, exp_avg_sq, step) of the
       restart trajectory is identical BEFORE vs AFTER the save/load (optK == loadK);
     - no populated leaf is silently dropped from the saved key set;
     - the saved/restored model has no missing/unexpected keys.
   These use torch.equal semantics: any mismatch is a real save/load bug.

2. TRAINING TRAJECTORY (whole-tensor, calibrated tolerance). The fp32 master AND the bf16
   run weight of control vs restart are compared as GEOMETRIC objects:
     - relative L2  ||a - b|| / ||a||        <= bound_l2
     - cosine       a.b / (|a||b|)           >= bound_cos
     - a near-zero-aware ABSOLUTE floor on    ||a-b|| / sqrt(N)  <= bound_atol
       (tiny coordinates random-walk and make per-element relative error meaningless;
        an RMSE floor is scale-relative to the reference tensor, not to one element).
   Elementwise equality is deliberately NOT asserted at either precision.

The bounds are NOT hardcoded and NOT guessed from one red run. They are read from a
calibration JSON produced from paired healthy control/restart runs on the RUNNER (the
noisy environment the gate actually executes in), taken at a margin over the observed green
noise. Without a calibration file layer 2 is reported UNCALIBRATED and passes open ONLY when
explicitly allowed -- the gate never invents a tolerance. Calibrate after the CED forward
lands (the flat-forward bounds would be stale).

Dump layout consumed (the #624 / diag_resume_bimodal artifact shape):
    <dump>/
      fp32_master.want.bin / .got.bin / .want.json / .got.json   (np.float32 raw tensors)
      <run>/leaves.json                                           (list of leaf names)
      restart/optK.l{j}.{exp_avg,exp_avg_sq,step}.pt
      restart/loadK.l{j}.{exp_avg,exp_avg_sq,step}.pt
      restart/ckpt_identity.json
The .pt files are torch checkpoints; torch is imported lazily only for layer 1 so the
calibration/selftest path (numpy synthetic tensors) needs no torch and no GPU.

Exit codes: 0 pass (or trajectory layer uncalibrated-with-permission); 1 deterministic
fail; 2 trajectory-equivalence fail; 3 unreadable/malformed input; 4 uncalibrated and not
permitted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

# Exit codes, distinct on purpose (the caller reads $?, not stdout).
RC_PASS = 0
RC_DETERMINISTIC_FAIL = 1
RC_TRAJECTORY_FAIL = 2
RC_UNREADABLE = 3
RC_UNCALIBRATED = 4


# --------------------------------------------------------------------------- whole-tensor

def _f32(path: str):
    """Read a raw little-endian float32 tensor (the #624 .bin format)."""
    import numpy as np

    with open(path, "rb") as fh:
        buf = fh.read()
    if len(buf) % 4:
        raise ValueError(f"{path}: {len(buf)} bytes is not a whole float32 count")
    return np.frombuffer(buf, dtype="<f4").astype(np.float64)


def rel_l2(a, b) -> float:
    """||a-b|| / ||a|| -- aggregate energy difference, scale-normalized."""
    import numpy as np

    na = float(np.linalg.norm(a))
    if na == 0.0:
        return float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / na)


def cosine(a, b) -> float:
    """Direction agreement in [-1, 1]."""
    import numpy as np

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(a, b) / (na * nb))


def rmse(a, b) -> float:
    """RMS difference; the near-zero-aware absolute floor is compared against reference RMS."""
    import numpy as np

    return float(math.sqrt(float(np.mean((a - b) ** 2))))


def ref_rms(a) -> float:
    import numpy as np

    return float(math.sqrt(float(np.mean(a ** 2))))


def trajectory_metrics(want, got) -> dict:
    """All aggregate measures for one weight tensor pair."""
    return {
        "rel_l2": rel_l2(want, got),
        "cosine": cosine(want, got),
        "rmse": rmse(want, got),
        "ref_rms": ref_rms(want),
        "n": int(want.shape[0]),
    }


def trajectory_pass(m: dict, cal: dict) -> tuple[bool, list[str]]:
    """Apply calibrated bounds. cal holds per-or-numeric bounds: bound_l2, bound_cos,
    bound_atol_frac (RMSE/ref_rms ceiling). Returns (pass, reasons)."""
    reasons = []
    b_l2 = float(cal["bound_l2"])
    b_cos = float(cal["bound_cos"])
    b_atol = float(cal["bound_atol_frac"])  # RMSE as a fraction of reference RMS
    if not (m["rel_l2"] <= b_l2):
        reasons.append(f"rel_l2 {m['rel_l2']:.4e} > {b_l2:.4e}")
    if not (m["cosine"] >= b_cos):
        reasons.append(f"cosine {m['cosine']:.8f} < {b_cos:.8f}")
    rel_rmse = m["rmse"] / m["ref_rms"] if m["ref_rms"] else 0.0
    if not (rel_rmse <= b_atol):
        reasons.append(f"rmse/ref_rms {rel_rmse:.4e} > {b_atol:.4e}")
    return (not reasons), reasons


# ------------------------------------------------------------------------- deterministic pt

def _load_pt(path: str):
    import torch  # lazy: keeps selftest/calibration torch-free

    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj


def _pt_scalar_equal(x, y) -> bool:
    import torch

    if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
        return bool(torch.equal(x, y))
    return x == y


def check_optimizer_pair(opt_dir: str, load_dir: str, leaves: list[str], rows: list[str]) -> int:
    """Layer 1: restart's optK (pre-save) vs loadK (post-restore) bit-exact for every leaf x
    {exp_avg, exp_avg_sq, step}. Returns RC_PASS or RC_DETERMINISTIC_FAIL."""
    fails = 0
    checked = 0
    for j in range(len(leaves)):
        for part in ("exp_avg", "exp_avg_sq", "step"):
            op = os.path.join(opt_dir, f"optK.l{j}.{part}.pt")
            lp = os.path.join(load_dir, f"loadK.l{j}.{part}.pt")
            if not (os.path.exists(op) and os.path.exists(lp)):
                rows.append(f"  MISSING l{j}.{part}: optK={os.path.exists(op)} loadK={os.path.exists(lp)}")
                fails += 1
                continue
            if not _pt_scalar_equal(_load_pt(op), _load_pt(lp)):
                rows.append(f"  BIT-DIFF l{j}.{part} ({leaves[j]}): optimizer state changed by save/load")
                fails += 1
            checked += 1
    if fails:
        rows.insert(0, f"DETERMINISTIC FAIL: {fails} optimizer leaf-state(s) not bit-identical across save/load")
        return RC_DETERMINISTIC_FAIL
    rows.append(f"  deterministic optimizer state bit-exact: {checked} leaf-state(s) across {len(leaves)} leaves")
    return RC_PASS


def check_identity(identity_path: str, rows: list[str]) -> int:
    """Layer 1b: ckpt_identity must show no dropped populated leaf and no missing/unexpected
    model keys. The artifact is written by diag_resume_bimodal._ckpt_identity."""
    try:
        with open(identity_path) as fh:
            ident = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        rows.append(f"  unreadable ckpt_identity: {exc}")
        return RC_UNREADABLE
    dropped = ident.get("populated_but_dropped") or []
    missing = ident.get("model_missing_keys") or []
    unexpected = ident.get("model_unexpected_keys") or []
    bad = len(dropped) + len(missing) + len(unexpected)
    if bad:
        rows.append(
            f"DETERMINISTIC FAIL: dropped={dropped[:5]} missing={missing[:5]} unexpected={unexpected[:5]}"
        )
        return RC_DETERMINISTIC_FAIL
    rows.append("  deterministic key sets intact: no populated leaf dropped, no missing/unexpected keys")
    return RC_PASS


# ----------------------------------------------------------------------------------- evaluate

def evaluate_dump(dump: str, calibration: str | None, allow_uncalibrated: bool, rows: list[str]) -> int:
    """Evaluate an on-disk artifact. Layer 1 (deterministic) always runs; layer 2 (trajectory)
    runs only with a calibration file unless allow_uncalibrated."""
    restart = os.path.join(dump, "restart")
    leaves_path = os.path.join(restart, "leaves.json")
    try:
        with open(leaves_path) as fh:
            leaves = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        rows.append(f"NO-GO: cannot read {leaves_path}: {exc}")
        return RC_UNREADABLE

    # Layer 1 -- bit-exact save/load.
    rc = check_optimizer_pair(restart, restart, leaves, rows)  # optK/loadK both under restart/
    if rc != RC_PASS:
        return rc
    idp = os.path.join(restart, "ckpt_identity.json")
    if os.path.exists(idp):
        rc = check_identity(idp, rows)
        if rc != RC_PASS:
            return rc

    # Layer 2 -- calibrated whole-tensor trajectory equivalence.
    tensors = []
    for name, stem in (("fp32_master", "fp32_master"), ("bf16_run_weight", "bf16_run_weight")):
        w = os.path.join(dump, f"{stem}.want.bin")
        g = os.path.join(dump, f"{stem}.got.bin")
        if os.path.exists(w) and os.path.exists(g):
            tensors.append((name, w, g))
    if not tensors:
        rows.append("NO-GO: no want/got trajectory tensors (.bin) found in dump")
        return RC_UNREADABLE

    if calibration is None:
        rows.append(
            "  TRAJECTORY layer UNCALIBRATED: no --calibration; deterministic state passed but "
            "whole-tensor equivalence not judged (run calibration on paired healthy runner runs)"
        )
        return RC_PASS if allow_uncalibrated else RC_UNCALIBRATED

    try:
        with open(calibration) as fh:
            cal = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        rows.append(f"NO-GO: cannot read calibration {calibration}: {exc}")
        return RC_UNREADABLE

    worst = RC_PASS
    for name, wp, gp in tensors:
        want, got = _f32(wp), _f32(gp)
        if want.shape != got.shape:
            rows.append(f"  {name}: shape mismatch {want.shape} vs {got.shape}")
            worst = RC_TRAJECTORY_FAIL
            continue
        m = trajectory_metrics(want, got)
        bounds_table = cal.get("bounds", cal)  # accept the write_calibration schema or a flat one
        c = bounds_table.get(name) or bounds_table.get("default")
        if c is None:
            rows.append(f"  {name}: no bounds in calibration (need key '{name}' or 'default')")
            worst = RC_UNCALIBRATED
            continue
        ok, reasons = trajectory_pass(m, c)
        stat = (
            f"  {name}: rel_l2={m['rel_l2']:.4e} cosine={m['cosine']:.8f} "
            f"rmse/ref_rms={m['rmse'] / max(m['ref_rms'], 1e-300):.4e} n={m['n']}"
        )
        rows.append(stat + ("  PASS" if ok else "  FAIL: " + "; ".join(reasons)))
        if not ok:
            worst = RC_TRAJECTORY_FAIL
    if worst == RC_PASS:
        rows.append("TRAJECTORY: all tensors within calibrated whole-tensor bounds")
    return worst


# -------------------------------------------------------------------------------- calibration

def write_calibration(samples: list[dict], margin_l2: float, margin_cos: float, margin_atol: float,
                      provenance: dict, out: str) -> dict:
    """Build a calibration JSON from collected healthy-run metrics. For each tensor kind the
    bound is the observed worst green metric widened by the margin:
      bound_l2 = max(rel_l2) * margin_l2 ; bound_atol_frac likewise
      bound_cos = 1 - (1 - min(cos)) * margin_cos   (cosine is "closer to 1 is better")
    `samples` = list of trajectory_metrics()-like dicts keyed by tensor name, e.g.
    [{"fp32_master": {...}, "bf16_run_weight": {...}}, ...]. Empty => ValueError (no fake bounds).
    """
    if not samples:
        raise ValueError("cannot calibrate from zero samples; bounds would be invented")
    kinds = sorted({k for s in samples for k in s})
    bounds = {}
    for k in kinds:
        ms = [s[k] for s in samples if k in s]
        max_l2 = max(m["rel_l2"] for m in ms)
        min_cos = min(m["cosine"] for m in ms)
        max_rmse_frac = max(m["rmse"] / max(m["ref_rms"], 1e-300) for m in ms)
        bounds[k] = {
            "bound_l2": max_l2 * margin_l2,
            "bound_cos": 1.0 - (1.0 - min_cos) * margin_cos,
            "bound_atol_frac": max_rmse_frac * margin_atol,
            "observed_n": len(ms),
            "observed_worst": {"rel_l2": max_l2, "cosine": min_cos, "rmse_over_ref_rms": max_rmse_frac},
        }
    cal = {
        "schema": "resume-equivalence-calibration/v1",
        "provenance": provenance,
        "margin": {"l2": margin_l2, "cos": margin_cos, "atol": margin_atol},
        "bounds": bounds,
        "note": (
            "Whole-tensor bounds over healthy paired control/restart runner runs. Per-coordinate "
            "equality is not reproducible across runner processes; these bound aggregate/geometric "
            "divergence. Re-calibrate after a forward-path change (e.g. CED)."
        ),
    }
    with open(out, "w") as fh:
        json.dump(cal, fh, indent=2, sort_keys=True)
    return cal


# ------------------------------------------------------------------------------------ selftest

def _np():
    import numpy as np

    return np


def _selftest() -> None:
    np = _np()
    rng = np.random.default_rng(7)

    # Pure geometry checks on known vectors.
    a = rng.standard_normal(100000)
    assert rel_l2(a, a) == 0.0
    assert cosine(a, a) > 0.99999999
    # orthogonal => cosine 0; scaled copy => rel_l2 exact ratio, cosine 1.
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    assert abs(cosine(x, y)) < 1e-12
    assert abs(cosine(x, 2.0 * x) - 1.0) < 1e-12
    assert abs(rel_l2(x, 2.0 * x) - 1.0) < 1e-12

    # Tight bounds distinguish a small aggregate perturbation (pass) from a large one (fail).
    base = rng.standard_normal(200000)
    small = base + rng.standard_normal(200000) * (0.001 * np.abs(base) + 1e-6)
    m_small = trajectory_metrics(base, small)
    assert m_small["rel_l2"] < 0.01 and m_small["cosine"] > 0.9999, m_small
    # a localized 10% block perturbation must be caught (mutation must not pass a vacuous bound)
    bad = base.copy()
    bad[:20000] *= 1.1
    m_bad = trajectory_metrics(base, bad)
    assert m_bad["rel_l2"] > m_small["rel_l2"] * 5, (m_small, m_bad)
    bounds = {
        "bound_l2": m_small["rel_l2"] * 3,
        "bound_cos": 1.0 - (1.0 - m_small["cosine"]) * 3,
        "bound_atol_frac": (m_small["rmse"] / m_small["ref_rms"]) * 3,
    }
    assert trajectory_pass(m_small, bounds)[0] is True
    assert trajectory_pass(m_bad, bounds)[0] is False, "a localized block regression must FAIL"

    # Near-zero tail: elementwise relative explodes but aggregate stays bounded -- proves why the
    # gate uses rel_l2/cosine/RMS, not per-element rtol. A 1e-9 absolute nudge on 1e-6 values is
    # a 0.1% per-element shift but negligible aggregate energy/cosine.
    z = np.full(1000, 1e-6)
    z2 = z + 1e-9
    assert rel_l2(z, z2) < 2e-3 and cosine(z, z2) > 0.9999

    # Calibration: zero samples must be refused (never invent bounds).
    try:
        write_calibration([], 3, 3, 3, {}, os.devnull)
        raise AssertionError("empty calibration should refuse")
    except ValueError:
        pass
    # calibration widens observed green noise and stays tighter than a real regression.
    samples = [{"t": m_small}]
    import tempfile

    fd, tmpname = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w"):
        pass
    cal = write_calibration(samples, 3.0, 3.0, 3.0, {"source": "selftest", "runs": 1}, tmpname)
    b = cal["bounds"]["t"]
    assert trajectory_pass(m_small, b)[0] and not trajectory_pass(m_bad, b)[0]
    assert cal["bounds"]["t"]["observed_n"] == 1
    os.unlink(tmpname)

    # Deterministic layer semantics without torch: emulate the predicate pairs as
    # equal/not-equal so the rc mapping is exercised. (The real .pt comparison goes through
    # torch.equal in check_optimizer_pair; here we assert the rc constants the rows encode.)
    assert RC_PASS != RC_DETERMINISTIC_FAIL != RC_TRAJECTORY_FAIL != RC_UNCALIBRATED

    print(
        "resume_equiv_gate selftest ok: geometry (rel_l2/cosine/RMS) distinguishes small aggregate "
        "noise from a localized block regression; near-zero tail handled aggregate-wise; empty "
        "calibration refused; rc codes distinct"
    )


# --------------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dump", nargs="?", help="artifact dir (control/restart + want/got tensors)")
    ap.add_argument("--calibration", help="calibration JSON from healthy paired runner runs")
    ap.add_argument("--allow-uncalibrated", action="store_true",
                    help="exit 0 (with a loud UNCALIBRATED row) when no calibration is supplied")
    ap.add_argument("--selftest", action="store_true", help="offline numpy checks; no torch/GPU")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return RC_PASS
    if not args.dump:
        ap.error("need a dump dir, or --selftest")
    rows: list[str] = []
    rc = evaluate_dump(args.dump, args.calibration, args.allow_uncalibrated, rows)
    if rc == RC_PASS:
        print("GO: deterministic state bit-exact; trajectory within calibrated bounds (or uncalibrated-permitted)")
    elif rc == RC_DETERMINISTIC_FAIL:
        print("NO-GO: deterministic save/load state differs (real checkpoint bug)")
    elif rc == RC_TRAJECTORY_FAIL:
        print("NO-GO: training trajectory outside calibrated whole-tensor bounds")
    elif rc == RC_UNCALIBRATED:
        print("NO-GO: trajectory layer uncalibrated and --allow-uncalibrated not given")
    else:
        print("NO-GO: unreadable/malformed dump")
    for r in rows:
        print(r)
    return rc


if __name__ == "__main__":
    sys.exit(main())

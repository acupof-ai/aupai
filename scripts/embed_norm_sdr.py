#!/usr/bin/env python3
# restartable: reads checkpoints on CPU, one pair at a time; minutes, nothing to shard.
"""Embedding-norm growth decomposed into per-step drift and effective same-direction rate.

    python3 scripts/embed_norm_sdr.py --ckpt ckpt_p324.pt --steps 1000,2000,3000
    python3 scripts/embed_norm_sdr.py --scan            # every multi-step ckpt family
    python3 scripts/embed_norm_sdr.py --selftest

WHAT THIS MEASURES. Adam's update is scale-free (magnitude ~ lr), so if every step pushed the
embedding norm the same way, one interval of n steps would multiply it by about
`1 + lr*|upd|/|w| * n`. It does not: the measured growth is far smaller, and the ratio

    same_direction_rate = (measured - 1) / (predicted - 1)

is the fraction of update magnitude that lands in the norm-increasing direction. At L12 step
832->1192 it is 7.1%: the embedding is a random walk with a small net drift, not a parameter
being pushed one way.

WHY IT IS A SCRIPT AND NOT A ONE-OFF. structure_experiments.md decomposed the L12/L32 growth gap
(2.43x) into an lr/weight-scale term (1.42x) and a "L12's updates are more consistent across
steps" term (1.71x), and pre-registered a THIRD point at p300m/L18 predicting the rate lands
between 4.6% and 7.9% and falls monotonically with depth. Both points in that decomposition were
read at different STEPS -- L12 at ~1000, L32 at ~2750 -- and nothing had measured how much this
quantity moves with step at FIXED depth.

It moves far more than 1.71x. Measured (--scan, 2026-09-03), equal-length intervals only:

    p324         L12 lr=0.1   step  1000-> 2000  n=1000   4.10%   same run, same depth, same
    p324         L12 lr=0.1   step  2000-> 3000  n=1000   1.01%   lr, same n -> 4.1x from
                                                                  step position alone
    p500m        L32 lr=0.085 step  2000-> 2500  n=500    3.23%
    p500m        L32 lr=0.085 step  2500-> 3000  n=500    3.76%
    pretrain_30b L12 lr=0.1   step 25500->26000  n=500    0.32%
    pretrain_30b L12 lr=0.1   step 26000->26500  n=500    0.32%

Two things follow, and the second is the one that matters:

1. Step position moves the rate 4.1x within one run at fixed depth, lr and n -- larger than the
   1.71x the decomposition attributes to depth.
2. At the CLOSEST matched comparison available (n=500, step ~2750), the ORDERING INVERTS:
   L32 reads 3.76% against L12's 0.32% at step 26000. The decomposition rests on L12 > L32
   (7.9% vs 4.6%); every equal-n pair in this repo has L32 >= L12.

So the L18 reading as pre-registered cannot settle the mechanism: whatever it returns is
consistent with any depth story, because step position is uncontrolled and dominates.

TWO GUARDS, and both refusals are the deliverable rather than the number. compare() refuses an
unmatched STEP position (the comparison that produced the 1.71x), and it refuses unequal
INTERVAL LENGTHS -- `predicted` is linear in n while real growth compounds, so under identical
per-step behaviour the recovered rate moves ~2000x from n=10 to n=1000. The original scan mixed
n=25 with n=6500 freely; those numbers were never comparable, mine included.
"""

import argparse
import glob
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("FLA_FLASH_KDA", "0")

# How close two intervals' step positions must be to count as matched. 0.25 is deliberately
# loose: the measured step dependence is ~4x per 1000 steps at step 2000, so anything tighter
# would refuse every pair the repo actually has, and anything looser stops meaning "matched".
STEP_MATCH_TOL = 0.25


def read_embed_state(path):
    """lr, wd, mean |Adam update| and the embedding norm from one checkpoint.

    The embedding group is optimizers[1] (AdamW) and holds exactly ONE parameter because the
    embedding is tied -- asserted, not assumed: if a future model unties them this returns None
    rather than silently averaging two different tensors' statistics into one number.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "opt" not in ck:
        return None
    opt = ck["opt"][1]
    st = opt["state"]
    if len(st) != 1:
        return None
    s = next(iter(st.values()))
    g = opt["param_groups"][0]
    t = s["step"].item() if torch.is_tensor(s["step"]) else s["step"]
    b1, b2 = g["betas"]
    m = s["exp_avg"].float() / (1 - b1**t)
    v = s["exp_avg_sq"].float() / (1 - b2**t)
    upd = m / (v.sqrt() + g.get("eps", 1e-8))
    # `a or b` on a TENSOR raises: truthiness of a multi-element tensor is ambiguous. So the
    # fallback is an explicit None check, not `or`. This crashed on the real checkpoints while
    # --selftest stayed green, because the selftest hands `interval()` dicts and never calls this
    # function at all -- a shape worth remembering: the guarded code and the guard tested
    # different halves of the file.
    w = ck["model"].get("tok.weight")
    if w is None:
        w = next((val for k, val in ck["model"].items() if k.endswith("tok.weight")), None)
    if w is None:
        return None
    w = w.float()
    cfg = ck.get("cfg") or {}
    return {
        "path": os.path.basename(path),
        "step": ck.get("step"),
        "layers": cfg.get("layers"),
        "lr": g["lr"],
        "wd": g["weight_decay"],
        "upd_mean": upd.abs().mean().item(),
        "w_norm": w.norm().item(),
        "w_mean_abs": w.abs().mean().item(),
    }


def interval(a, b):
    """The decomposition over one interval. `a` is the earlier checkpoint."""
    n = b["step"] - a["step"]
    if n <= 0:
        raise ValueError(f"interval is not forward: step {a['step']} -> {b['step']}")
    per_step = a["lr"] * a["upd_mean"] / a["w_mean_abs"]
    predicted = 1 + per_step * n
    measured = b["w_norm"] / a["w_norm"]
    return {
        "step_from": a["step"], "step_to": b["step"], "n": n,
        "layers": a["layers"], "lr": a["lr"],
        "per_step_drift": per_step,
        "predicted_if_all_aligned": predicted,
        "measured_growth": measured,
        "same_direction_rate": (measured - 1) / (predicted - 1) if predicted > 1 else float("nan"),
    }


def compare(x, y, allow_unmatched=False):
    """Two intervals' rates side by side -- or a REFUSAL, on unequal length or unmatched step.

    Both guards exist because the comparisons they refuse are the ones that produced a mechanism
    claim. Unequal INTERVAL LENGTH comes first because it is the harder error to see: the rate is
    a within-length statistic (linear `predicted` against compounding growth moves it ~2000x from
    n=10 to n=1000), so the original scan's mix of n=25 and n=6500 was never comparable. Attributing a rate difference to depth requires everything else held:
    step position above all, since the rate falls ~4x per 1000 steps at step 2000 and 140x
    across the L12 family. lr is reported rather than gated -- it is already an explicit term in
    the decomposition, so a reader who sees both lrs can price it; step was not a term at all,
    which is exactly how it went unnoticed.
    """
    mid_x = (x["step_from"] + x["step_to"]) / 2
    mid_y = (y["step_from"] + y["step_to"]) / 2
    rel = abs(mid_x - mid_y) / max(mid_x, mid_y)
    n_x, n_y = x["step_to"] - x["step_from"], y["step_to"] - y["step_from"]
    out = {
        "x": {k: x[k] for k in ("layers", "lr", "step_from", "step_to", "same_direction_rate")},
        "y": {k: y[k] for k in ("layers", "lr", "step_from", "step_to", "same_direction_rate")},
        "step_midpoint_rel_gap": rel,
        "interval_lengths": [n_x, n_y],
        "lr_ratio": x["lr"] / y["lr"] if y["lr"] else float("nan"),
    }
    if n_x != n_y and not allow_unmatched:
        out["REFUSED"] = (
            f"interval lengths differ ({n_x} vs {n_y} steps). The rate is NOT comparable across "
            f"lengths: `predicted` is linear in n while real growth compounds, so under identical "
            f"per-step behaviour the recovered rate moves ~2000x from n=10 to n=1000 (asserted in "
            f"--selftest). Read both points over the same number of steps.")
        return out
    if rel > STEP_MATCH_TOL and not allow_unmatched:
        out["REFUSED"] = (
            f"step positions differ by {rel:.0%} (midpoints {mid_x:.0f} vs {mid_y:.0f}), above "
            f"the {STEP_MATCH_TOL:.0%} tolerance. The same-direction rate falls ~4x per 1000 "
            f"steps at step 2000 and spans 140x across one depth, so a difference read across "
            f"unmatched steps says nothing about depth. This is the comparison that produced "
            f"the 1.71x 'updates are more consistent at L12' term. Read both depths at the SAME "
            f"step, or pass allow_unmatched and state in the writeup that step is uncontrolled.")
        return out
    out["rate_ratio"] = (x["same_direction_rate"] / y["same_direction_rate"]
                         if y["same_direction_rate"] else float("nan"))
    return out


def _fake(step, layers, lr, upd, w_norm, w_mean_abs):
    return {"path": "fake", "step": step, "layers": layers, "lr": lr, "wd": 0.001,
            "upd_mean": upd, "w_norm": w_norm, "w_mean_abs": w_mean_abs}


def _selftest():
    fails = []

    # 1. KNOWN ANSWER, 100%: every step aligned means measured growth EQUALS the prediction.
    a = _fake(0, 12, 0.1, 1.0, 100.0, 1.0)          # per-step drift = 0.1*1.0/1.0 = 0.1
    b = _fake(10, 12, 0.1, 1.0, 100.0 * 2.0, 1.0)   # predicted 1 + 0.1*10 = 2.0x
    r = interval(a, b)
    if abs(r["predicted_if_all_aligned"] - 2.0) > 1e-9 or abs(r["same_direction_rate"] - 1.0) > 1e-9:
        fails.append(f"fully-aligned interval read {r['same_direction_rate']:.4f}, expected 1.0")

    # 2. KNOWN ANSWER, 0%: the norm did not move at all, so no update was norm-increasing.
    r0 = interval(a, _fake(10, 12, 0.1, 1.0, 100.0, 1.0))
    if abs(r0["same_direction_rate"]) > 1e-12:
        fails.append(f"a flat norm read {r0['same_direction_rate']:.4g}, expected 0")

    # 3. THE TWO REFUSALS, each isolated so that each is the ONLY thing standing between the
    #    input and a wrong answer. A single fixture violating both passes with either guard
    #    disabled -- measured: the first version of this check did exactly that, and the selftest
    #    stayed green with each `if` turned off in turn.
    early = {"layers": 12, "lr": 0.1, "step_from": 1000, "step_to": 2000,
             "same_direction_rate": 0.0410}
    # 3a. UNEQUAL LENGTH ONLY: same depth, same lr, overlapping step position (midpoints 1500 vs
    #     1500, so the step guard has nothing to say). Only the length differs.
    len_only = {"layers": 12, "lr": 0.1, "step_from": 1250, "step_to": 1750,
                "same_direction_rate": 0.0300}
    c_len = compare(early, len_only)
    if "REFUSED" not in c_len or "interval lengths differ" not in c_len["REFUSED"]:
        fails.append(f"compare() accepted n=1000 against n=500 at the same midpoint; the rate is "
                     f"a within-length statistic and moves ~2000x across n. Got: {c_len}")
    if "rate_ratio" in c_len:
        fails.append("a length-REFUSED comparison still returned rate_ratio, so a caller reading "
                     "the number would never see the refusal")
    # 3b. UNMATCHED STEP ONLY: EQUAL length (n=1000 both), so 3a's guard passes and only the step
    #     guard can refuse. This is the shape of the comparison behind the 1.71x term -- L12 read
    #     at ~1000 against L32 read at ~2750.
    step_only = {"layers": 32, "lr": 0.085, "step_from": 2500, "step_to": 3500,
                 "same_direction_rate": 0.0376}
    c_step = compare(early, step_only)
    if "REFUSED" not in c_step or "step positions differ" not in c_step["REFUSED"]:
        fails.append(f"compare() accepted a cross-depth reading at equal length whose step "
                     f"midpoints are 1500 against 3000; that is the unmatched comparison behind "
                     f"the 1.71x term. Got: {c_step}")
    if "rate_ratio" in c_step:
        fails.append("a step-REFUSED comparison still returned rate_ratio")
    late = step_only
    # ...and it must go through when the steps DO match, or the guard is just an always-refuse.
    # NOTE the fixture: comparing L12's 1000-2000 against L32's 2000-3000 puts the midpoints at
    # 1500 and 2500, a 40% gap -- REFUSED, correctly. My first version of this assertion used
    # exactly that pair and read the refusal as "the guard refuses everything". A matched pair
    # means the same interval, which is what reading both depths at the same step produces.
    matched = dict(late, step_from=1000, step_to=2000, layers=32)
    c2 = compare(early, matched)   # equal n=1000, identical midpoint -> must pass both guards
    if "REFUSED" in c2:
        fails.append(f"compare() refused a matched-step pair (midpoints 1500 vs 2500, "
                     f"{c2['step_midpoint_rel_gap']:.0%}): the guard refuses everything")
    elif "rate_ratio" not in c2:
        fails.append("a matched comparison returned no rate_ratio")
    # The explicit override must work, since some writeups legitimately compare unmatched points
    # and say so.
    if "REFUSED" in compare(early, late, allow_unmatched=True):
        fails.append("allow_unmatched did not lift the refusal")

    # 4. A BACKWARD INTERVAL IS AN ERROR, not a negative rate. Silently returning one would
    #    invert the sign of every conclusion drawn from it.
    try:
        interval(b, a)
        fails.append("interval() accepted a backward pair and returned a number")
    except ValueError:
        pass

    # 5. THE RATE IS ONLY COMPARABLE ACROSS EQUAL-LENGTH INTERVALS, and this asserts the limit
    #    rather than a property the metric does not have. `predicted` is LINEAR in n while real
    #    growth COMPOUNDS, so the ratio between them is n-dependent by construction.
    #
    #    An earlier version of this check "proved" n-invariance with a fixture whose growth was
    #    g = 1 + 0.01*n -- linear, i.e. the same functional form as `predicted`, which makes the
    #    ratio invariant by construction and tests nothing. Under the compounding shape the real
    #    runs have (g = 1.01**n) the recovered rate moves 0.1046 -> 0.1705 -> 209.6 across
    #    n = 10, 100, 1000: a factor of 2000, dwarfing every effect anyone wants to read.
    #
    #    So the metric is a within-interval-length statistic. compare()'s step guard is necessary
    #    but NOT sufficient; equal n is the second requirement, asserted here.
    lin = [interval(_fake(0, 12, 0.1, 1.0, 100.0, 1.0),
                    _fake(n, 12, 0.1, 1.0, 100.0 * (1 + 0.01 * n), 1.0))["same_direction_rate"]
           for n in (10, 100, 1000)]
    if max(lin) - min(lin) > 1e-9:
        fails.append(f"the rate moved {min(lin):.4f}..{max(lin):.4f} across n under LINEAR "
                     f"growth, where it is invariant by construction -- the arithmetic broke")
    comp = [interval(_fake(0, 12, 0.1, 1.0, 100.0, 1.0),
                     _fake(n, 12, 0.1, 1.0, 100.0 * 1.01**n, 1.0))["same_direction_rate"]
            for n in (10, 100, 1000)]
    if max(comp) / min(comp) < 100:
        fails.append(f"under COMPOUNDING growth the rate spans only {max(comp) / min(comp):.1f}x "
                     f"across n=10..1000, but the linear-vs-compounding mismatch makes it "
                     f"~2000x. If the definition changed to be n-invariant, that is an "
                     f"improvement -- update this check and the docstring, which currently warns "
                     f"readers that unequal-length intervals are not comparable.")

    # 6. read_embed_state() ON A REAL CHECKPOINT STRUCTURE. Checks 1-5 hand `interval()` plain
    #    dicts, so they never execute the function that opens a checkpoint -- and a `w = a or b`
    #    there raised "Boolean value of Tensor with more than one value is ambiguous" on every
    #    real ckpt while this selftest stayed green. The guard and the guarded code were in
    #    different halves of the file. So build a checkpoint in memory and read it.
    import tempfile
    v, d = 32, 8
    w = torch.randn(v, d)
    ckpt = {
        "step": 100,
        "cfg": {"layers": 12, "dim": d},
        "model": {"blocks.0.x": torch.randn(2, 2), "tok.weight": w},
        "opt": [
            {"state": {}, "param_groups": [{"lr": 0.02}]},                      # muon
            {"state": {0: {"step": torch.tensor(100.0), "exp_avg": torch.randn(v, d),
                           "exp_avg_sq": torch.rand(v, d) + 0.1}},
             "param_groups": [{"lr": 0.1, "weight_decay": 0.001, "betas": (0.9, 0.95)}]},
        ],
    }
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "ckpt_fixture.pt.step100")
        torch.save(ckpt, fp)
        r = read_embed_state(fp)
        if r is None:
            fails.append("read_embed_state returned None on a well-formed checkpoint")
        else:
            if abs(r["w_norm"] - w.norm().item()) > 1e-5:
                fails.append(f"w_norm {r['w_norm']} != {w.norm().item()} -- it read the wrong "
                             f"tensor, and `model` deliberately holds a non-tok key first")
            if r["step"] != 100 or r["layers"] != 12 or r["lr"] != 0.1:
                fails.append(f"read the wrong group or metadata: {r}")
        # A TIED-EMBEDDING VIOLATION must return None, not average two tensors' statistics into
        # one number: the whole reading assumes the embed group holds exactly one parameter.
        ckpt["opt"][1]["state"][1] = dict(ckpt["opt"][1]["state"][0])
        fp2 = os.path.join(td, "ckpt_untied.pt.step100")
        torch.save(ckpt, fp2)
        if read_embed_state(fp2) is not None:
            fails.append("read_embed_state accepted an embed group with TWO parameters; the "
                         "reading assumes one (tied) and would silently average two tensors")

    if fails:
        for f in fails:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("embed_norm_sdr selftest OK: known answers at 100% (measured growth equals the "
          "all-aligned prediction) and 0% (a flat norm); compare() REFUSES the unmatched-step "
          "cross-depth reading that produced the 1.71x term, still passes a matched pair, and "
          "honours allow_unmatched; a backward interval raises instead of returning a negative "
          "rate; and the rate is NOT comparable across interval lengths (linear `predicted` "
          "against compounding real growth moves it ~2000x from n=10 to n=1000), which compare() "
          "refuses -- an earlier version of this check asserted the opposite using a linear "
          "fixture whose form matched `predicted`, so invariance held by construction.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="checkpoint family prefix, e.g. ckpt_p324.pt")
    ap.add_argument("--steps", help="comma-separated steps to read, e.g. 1000,2000,3000")
    ap.add_argument("--scan", action="store_true", help="every multi-step family found")
    ap.add_argument("--out", help="write results json here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()

    families = {}
    if a.scan:
        for p in sorted(glob.glob(os.path.join(ROOT, "ckpt_*.step*"))):
            families.setdefault(p.rsplit(".step", 1)[0], []).append(p)
    elif a.ckpt and a.steps:
        base = a.ckpt if os.path.isabs(a.ckpt) else os.path.join(ROOT, a.ckpt)
        families[base] = [f"{base}.step{s}" for s in a.steps.split(",")]
    else:
        ap.error("--scan, or --ckpt with --steps (or --selftest)")

    results = []
    for base, paths in sorted(families.items()):
        paths = sorted(paths, key=lambda p: int(p.rsplit(".step", 1)[1]))
        rs = [r for r in (read_embed_state(p) for p in paths if os.path.exists(p)) if r]
        if len(rs) < 2:
            continue
        print(f"\n{os.path.basename(base)}  L={rs[0]['layers']}  lr={rs[0]['lr']:.4g}")
        for x, y in zip(rs, rs[1:]):
            iv = interval(x, y)
            iv["ckpt"] = os.path.basename(base)
            results.append(iv)
            print(f"  step {iv['step_from']:>6} -> {iv['step_to']:<6} n={iv['n']:<5} "
                  f"drift {iv['per_step_drift']:.6f}  predicted {iv['predicted_if_all_aligned']:.3f}x"
                  f"  measured {iv['measured_growth']:.4f}x  "
                  f"same-dir {iv['same_direction_rate'] * 100:.2f}%")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

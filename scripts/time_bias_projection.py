#!/usr/bin/env python3
"""Cost of the zero-mean projection in MoEFFN.update_bias, measured directly on one card.

WHY THIS EXISTS. The resumed 1.5b-a0.2b-e48_8b run failed its pre-registered speed gate by
+2.9% s/step (median 2.8101 against a bar of 2.81), and the projection added at model.py:1034
was one candidate cause. A six-card projection-on/off A/B was struck (prereg moe_0905
amendment 22, reinstated, then struck again): the off arm cannot be produced without either a
Cfg flag or a commit that removes a verified fix from main, both arms would resume from a
checkpoint whose bias vector the projection has already held at zero mean -- so the arm could
never be labelled "pre-fix" -- and at nine intervals the design fires one run in three.

This measures the thing that A/B was supposed to measure, with none of its confounds: no host
contention, no checkpoint state, no second training run. The question is narrow and answerable
in ten minutes on one card -- how many milliseconds per optimizer step does the projection add
across 12 MoE layers -- and that is all this script answers.

MODEL.PY IS NOT EDITED AND NOT IMPORTED FOR ITS METHOD. The two variants are built here from
the same four statements as model.py:1015-1034, verified line-for-line against the source at
runtime (see _assert_replica_matches_source). Editing update_bias to add a flag, or
monkey-patching it, would put the measurement on a code path nobody ships.

WHAT THE GUARD DOES NOT CHECK: it is a substring test, so it would still pass if someone
wrapped the projection in `if self.project:` -- the statement would be present and still last.
That is deliberate. The guard's job is drift in WHAT IS TIMED; a wrapped projection is still
the statement being timed, and what would then be false is this script's stated reason for
existing (that no flag exists), which is a claim in prose and not something a substring check
should police. Verified at the time of writing, by two sessions independently: model.py:1034 is
the last statement of update_bias, there is no conditional between the def and it, and
`grep -rn 'zero_mean\|no_projection\|bias_project\|proj_off\|expert_bias_mean' model.py
train.py` returns nothing.

THE TRAP THIS SCRIPT IS BUILT AROUND: a synthetic replica can drift from the method it claims
to time, and then the number describes the replica. So the replica's source is compared
against the real method's source at runtime and the script REFUSES if the four statements are
not present in update_bias. That check fails loudly rather than measuring the wrong thing --
the failure mode this session hit five times tonight was a number whose instrument was never
checked.
"""

import argparse
import inspect
import os
import statistics
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# The four statements of model.py:1015-1034, as shipped. Kept as source text so the runtime
# check below can compare them against the real method rather than trusting this comment.
_SHIPPED = [
    "mean = counts.float().mean()",
    "err = counts.float() - mean",
    "self.expert_bias -= self.gamma * torch.sign(err).to(self.expert_bias.dtype)",
    "self.expert_bias -= self.expert_bias.mean()",
]


def _assert_replica_matches_source():
    """REFUSE unless every statement this script times appears in the real update_bias.

    Without this the script measures a replica that may have drifted from the method, and the
    result would describe code nobody runs. Checked on the SOURCE TEXT of the live method, so
    a future edit to model.py breaks this rather than silently changing what is timed.
    """
    from model import MoEFFN

    src = inspect.getsource(MoEFFN.update_bias)
    missing = [s for s in _SHIPPED if s not in src]
    if missing:
        raise SystemExit(
            "REFUSING to time a replica that does not match model.py.\n"
            "  MoEFFN.update_bias no longer contains:\n    "
            + "\n    ".join(missing)
            + "\n  Update _SHIPPED to the current statements and re-verify what is being timed."
        )
    # The projection must be the LAST statement, because "with" vs "without" is defined as
    # dropping exactly it. If it moved, dropping the last line stops meaning what this claims.
    body = [ln.strip() for ln in src.split("\n") if ln.strip() and not ln.strip().startswith("#")]
    if body[-1] != _SHIPPED[-1]:
        raise SystemExit(
            f"REFUSING: the last statement of update_bias is {body[-1]!r}, not the projection "
            f"{_SHIPPED[-1]!r}. 'Without the projection' is defined as dropping the last "
            f"statement; that definition no longer holds."
        )
    return src


def _with_projection(bias, counts, gamma):
    mean = counts.float().mean()
    err = counts.float() - mean
    bias -= gamma * torch.sign(err).to(bias.dtype)
    bias -= bias.mean()


def _without_projection(bias, counts, gamma):
    mean = counts.float().mean()
    err = counts.float() - mean
    bias -= gamma * torch.sign(err).to(bias.dtype)


def _time(fn, bias, counts, gamma, iters, layers):
    """Median per-STEP wall time over `iters` steps of `layers` layers each.

    CUDA is asynchronous: without the synchronize, this times kernel LAUNCH and reads ~0.
    Warmup is separate and discarded -- the first call compiles and allocates, and this
    session already made the mistake of reporting a first-pass cost as a steady-state one.
    """
    for _ in range(20):                      # warmup, discarded
        for _ in range(layers):
            fn(bias, counts, gamma)
    torch.cuda.synchronize()

    per_step = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for _ in range(layers):
            fn(bias, counts, gamma)
        torch.cuda.synchronize()
        per_step.append(time.perf_counter() - t0)
    return per_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="take the bias tensor and expert count from here")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--layers", type=int, default=12, help="MoE layers per optimizer step")
    ap.add_argument("--experts", type=int, default=48)
    ap.add_argument("--gamma", type=float, default=0.001)
    ap.add_argument("--step-seconds", type=float, default=2.8121,
                    help="the run's measured s/step, for the ratio")
    ap.add_argument("--selftest", action="store_true",
                    help="known answers, no GPU: the replica guard fires, and the verdict "
                         "statistic does not move with the sample count")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()

    # THE REPLICA GUARD RUNS FIRST, BEFORE THE GPU CHECK. It only reads source text, so
    # gating it behind CUDA meant the guard could only be exercised on a card -- which made
    # its own mutation test cost card time, and an untested guard was the thing this script
    # exists to avoid. Reordered so `--selftest`-style verification runs anywhere.
    src = _assert_replica_matches_source()

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU: the projection runs on the training device")

    # CLAIM THE CARD BEFORE TOUCHING IT (gpu_entry_points_claim). Without this the card reads
    # ORPHAN in runs/claims/ and a second job may take it mid-timing -- which for THIS script
    # would silently produce the co-tenancy confound it exists to avoid, so the claim is part
    # of the measurement, not paperwork around it.
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from loader import claim_my_cards

    claimed = claim_my_cards("time_bias_projection",
                             note="micro-timing the expert_bias zero-mean projection, <=10 min")
    print(f"claimed card(s): {claimed}")

    print(f"replica verified against MoEFFN.update_bias ({len(src.splitlines())} source lines)")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')} "
          f"device {torch.cuda.get_device_name(0)}")

    n_experts = a.experts
    bias0 = None
    if a.ckpt:
        # THE REAL BIAS VALUES, not zeros: sign() and mean() cost the same on any input, but a
        # tensor of the wrong SHAPE or dtype would not. Read shape and dtype from the artifact.
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck)
        keys = [k for k in sd if k.endswith("expert_bias")]
        if not keys:
            raise SystemExit(f"no expert_bias tensor in {a.ckpt} -- {len(sd)} keys")
        bias0 = sd[keys[0]]
        n_experts = bias0.numel()
        print(f"bias from {os.path.basename(a.ckpt)}: {len(keys)} layers, "
              f"{n_experts} experts, dtype {bias0.dtype}, absmax {bias0.abs().max().item():.6f}")

    dev = torch.device("cuda")
    bias_ref = (bias0.clone() if bias0 is not None
                else torch.zeros(n_experts, dtype=torch.float32)).to(dev)
    if bias_ref.dtype != torch.float32:
        print(f"  NOTE: checkpoint bias is {bias_ref.dtype}; the shipped buffer is fp32 "
              f"(model.py:958 holds it through the cast), so timing at fp32")
        bias_ref = bias_ref.float()
    # Integer counts, as update_bias receives them from step_tokens_per_expert. HELD CONSTANT
    # across all 12 layers where the real method gets fresh per-layer counts: sign() and mean()
    # cost the same on any input of this shape, so this changes no timing, but a reader should
    # not have to discover it.
    counts = torch.randint(0, 4096, (n_experts,), device=dev, dtype=torch.long)

    rows, raw = [], []
    for label, fn in (("with projection", _with_projection),
                      ("without projection", _without_projection)):
        # FRESH TENSOR PER ARM. _time mutates bias in place 12,000 times, and the "without"
        # variant is the unbounded integrator this fix exists to stop -- left to run on the
        # first arm's output it would start from a drifted vector and, worse, the two arms
        # would be timed on different values. sign() and mean() cost the same at any
        # magnitude, so this changes no number here; it removes a difference between the arms
        # that a reader would be right to ask about.
        bias = bias_ref.clone()
        s = _time(fn, bias, counts, a.gamma, a.iters, a.layers)
        med = statistics.median(s)
        rows.append((label, med, statistics.mean(s), min(s), max(s)))
        raw.append(s)
        print(f"{label:20s} median {med * 1e3:7.4f} ms/step   mean {statistics.mean(s) * 1e3:7.4f}"
              f"   min {min(s) * 1e3:7.4f}   max {max(s) * 1e3:7.4f}   (n={len(s)})")

    delta = rows[0][1] - rows[1][1]
    print()
    print(f"PROJECTION COST: {delta * 1e3:.4f} ms per optimizer step across {a.layers} layers")
    print(f"  against the run's measured {a.step_seconds:.4f} s/step: "
          f"{delta / a.step_seconds * 100:.4f}%")
    print(f"  the unexplained speed gate gap was 0.0783 s/step = 2.9%, i.e. "
          f"{0.0783 / delta:.0f}x this cost" if delta > 0 else
          "  delta is zero or negative: the projection's cost is below this timer's resolution")
    # THE FLOOR MUST NOT BE AN EXTREME. The first version of this line took
    # median(|min - med|, |max - med|) -- the half-range of 1000 samples, which GROWS with
    # --iters exactly as max-min grew with resample count in the bootstrap that produced
    # amendments 19-21. de measured it: 0.048 ms at iters=100 rising to 2.07 ms at iters>=1000
    # on one fixed distribution, so a real 0.08 ms delta read ABOVE at n=100 and BELOW at every
    # larger n -- the verdict was a statement about --iters, not about the delta. pstdev and the
    # 95th percentile of |x - median| both converge instead.
    with_s, without_s = raw[0], raw[1]
    sd_w, sd_wo = statistics.pstdev(with_s), statistics.pstdev(without_s)
    dev = sorted(abs(x - rows[0][1]) for x in with_s)
    p95 = dev[int(0.95 * len(dev))]
    # The delta's own uncertainty, not just the arms': independent arms, so add in quadrature.
    sd_delta = (sd_w ** 2 + sd_wo ** 2) ** 0.5
    # pstdev is REPORTED, NOT the verdict: measured on a spiked distribution it moves
    # 0.0202 -> 2.2653 ms between n=100 and n=1000 (a 1% spike population barely appears
    # at n=100 and then dominates the second moment), while p95 of |x - median| holds
    # 0.0422 -> 0.0410 across three orders. So the ABOVE/BELOW line keys on p95.
    print(f"  per-step sd: with {sd_w * 1e3:.4f} ms, without {sd_wo * 1e3:.4f} ms  (n={len(with_s)} each)")
    print(f"  delta {delta * 1e3:.4f} +/- {sd_delta * 1e3:.4f} ms (1 sd of a single step)")
    print(f"  p95 of |x - median| on the 'with' arm: {p95 * 1e3:.4f} ms -- the delta is "
          f"{'ABOVE' if abs(delta) > p95 else 'BELOW'} it")
    # THE ESTIMATOR'S UNCERTAINTY, NOT A SINGLE STEP'S. sd_delta above is single-step
    # dispersion, inflated by rare scheduling spikes -- de measured it 25x the standard error
    # of the median on this shape (0.5629 ms against 0.0223 ms) -- and it cannot shrink with
    # more steps, so a conditional keyed on it would fire across the whole range this script
    # expects (120/120 at 0.10 ms) and read as a hedge on a delta that n=1000 resolves. The
    # median's own standard error does shrink, so it is printed instead and no verdict is
    # drawn from either: p95 above is the verdict.
    se_med = 1.253 * sd_delta / (len(with_s) ** 0.5)
    print(f"  se of the median delta: {se_med * 1e3:.4f} ms  (1.253 x sd_delta / sqrt(n))")


def _selftest():
    """Known answers, no GPU. Two properties, each with its failing case.

    1. THE REPLICA GUARD FIRES. Not "the guard passes on main" -- that is what an untested
       guard looks like. Two mutated copies of the real update_bias source are fed to the same
       predicate the guard uses: one with the projection deleted (the substring branch), one
       with the projection present but followed by a clamp so it is no longer last (the
       body[-1] branch, which is the one that never executes in normal operation).
    2. THE VERDICT STATISTIC DOES NOT MOVE WITH THE SAMPLE COUNT. p95 of |x - median| must be
       stable across three orders of magnitude on a spiked distribution, where the half-range
       (the first version of this script's floor) and pstdev both are not. This is the property
       whose absence made the earlier verdict a statement about --iters.
    """
    import random

    fails = []

    src = inspect.getsource(_real_update_bias_source())
    def _verdict(text):
        missing = [x for x in _SHIPPED if x not in text]
        if missing:
            return "substring"
        body = [ln.strip() for ln in text.split("\n")
                if ln.strip() and not ln.strip().startswith("#")]
        return "last" if body[-1] != _SHIPPED[-1] else "pass"

    print(f"A unmodified source                -> {_verdict(src)}")
    if _verdict(src) != "pass":
        fails.append(f"A: the guard refuses the SHIPPED method ({_verdict(src)}) -- it cannot run at all")

    b = src.replace("        self.expert_bias -= self.expert_bias.mean()\n", "", 1)
    print(f"B projection deleted               -> {_verdict(b)}")
    if _verdict(b) != "substring":
        fails.append(f"B: deleting the projection gave {_verdict(b)!r}, want 'substring'")

    c = src.replace("        self.expert_bias -= self.expert_bias.mean()\n",
                    "        self.expert_bias -= self.expert_bias.mean()\n"
                    "        self.expert_bias.clamp_(-1.0, 1.0)\n", 1)
    print(f"C projection present but not last  -> {_verdict(c)}")
    if _verdict(c) != "last":
        fails.append(f"C: a projection followed by a clamp gave {_verdict(c)!r}, want 'last' -- "
                     f"the body[-1] branch is the only thing that catches a MOVE")

    # 2. the verdict statistic's stability. One fixed distribution, three sample counts.
    random.seed(3)
    def synth(n):
        return [0.0012 + random.gauss(0, 2e-5) + (0.02 if random.random() < 0.01 else 0)
                for _ in range(n)]
    rows = []
    for n in (100, 10000, 100000):
        v = synth(n)
        med = statistics.median(v)
        dev = sorted(abs(x - med) for x in v)
        rows.append((n, statistics.median([abs(min(v) - med), abs(max(v) - med)]),
                     statistics.pstdev(v), dev[int(0.95 * len(dev))]))
    print("   n      half-range      pstdev         p95|x-med|")
    for n, hr, sd, p95 in rows:
        print(f"{n:8d}  {hr * 1e3:11.4f}  {sd * 1e3:11.4f}  {p95 * 1e3:12.4f}")
    p95s = [r[3] for r in rows]
    if max(p95s) / min(p95s) > 1.5:
        fails.append(f"p95 moved {max(p95s) / min(p95s):.1f}x across n -- it is not a stable verdict")
    hrs = [r[1] for r in rows]
    if max(hrs) / min(hrs) < 2.0:
        fails.append("the half-range did NOT grow with n on this fixture, so the fixture no "
                     "longer reproduces the defect this check exists for")

    print()
    if fails:
        print("FAIL")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"PASS: {3 + 2} checks (guard fires on B and C, passes A; p95 stable, half-range grows)")
    return 0


def _real_update_bias_source():
    from model import MoEFFN
    return MoEFFN.update_bias


if __name__ == "__main__":
    raise SystemExit(main())

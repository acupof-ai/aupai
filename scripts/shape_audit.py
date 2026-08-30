"""Startup shape audit: catch GEMM-hostile dimensions before a run burns GPU hours.

The 2026-08-30 finding: vocab 32773 left the logits' leading dimension 2-byte aligned, so cuBLAS
fell back to an SM75 align-1 kernel on a Hopper card and the LM head ran at 41% of bf16 peak.
Padding to 32776 was worth +14-16% end to end. It went unnoticed because nothing checked shapes.

This is the cheap half of the fix: a static check costing milliseconds at startup, no profiler
and no GPU, that would have caught it before the first step. Sampling achieved TFLOPS during a
run tells you a kernel is slow; this tells you why, before you run it.

Thresholds: 8 is the alignment cuBLAS needs to pick its fast bf16 kernels; 16 is what _fp8_ok
requires for a weight to be eligible for fp8 at all.

Run: python scripts/shape_audit.py [--json out.json]
     python scripts/shape_audit.py --selftest
Exit code 1 when any FAIL is present, so it can gate a run.
"""

import argparse
import json
import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train  # noqa: E402

FAST_ALIGN = 8  # below this, cuBLAS drops to an align-1 kernel
FP8_ALIGN = 16  # _fp8_ok's requirement (train.py)


def check(where, dim, value):
    """One dimension, one verdict. Both tiers FAIL: 8-misalignment costs throughput now,
    16-misalignment silently drops fp8 (_fp8_ok rejects the weight) -- this repo trains in
    fp8 by default, so a run that silently stays bf16 is a wrong run, not a warning."""
    if value % FAST_ALIGN:
        return [
            {
                "level": "FAIL",
                "where": where,
                "dim": dim,
                "value": value,
                "why": f"{value} % {FAST_ALIGN} = {value % FAST_ALIGN}; cuBLAS drops to an align-1 "
                f"kernel -- measured at 41% of bf16 peak for the LM head",
                "fix": f"pad to {(value // FAST_ALIGN + 1) * FAST_ALIGN}",
            }
        ]
    if value % FP8_ALIGN:
        return [
            {
                "level": "FAIL",
                "where": where,
                "dim": dim,
                "value": value,
                "why": f"{value} % {FP8_ALIGN} = {value % FP8_ALIGN}; _fp8_ok rejects this weight, "
                f"so the run silently stays bf16 instead of the fp8 it was launched with",
                "fix": f"pad to {(value // FP8_ALIGN + 1) * FP8_ALIGN}",
            }
        ]
    return []


def config_findings(cfg):
    """The half that needs no model: dimensions fixed by Cfg alone. `heads` is a count rather than
    a GEMM dimension -- DeltaRecurrence already pads beta's rows via beta_pad -- so it is
    deliberately not checked; a check that cries wolf gets ignored."""
    out = []
    for name in ("vocab", "d", "ffn_hidden"):
        v = getattr(cfg, name, None)
        if isinstance(v, int):
            out += check("Cfg", name, v)
    if getattr(cfg, "heads", 0):
        out += check("Cfg", "d // heads (head_dim)", cfg.d // cfg.heads)
    return out


def audit(cfg):
    findings = config_findings(cfg)
    model = train.HybridLM(cfg)
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            findings += check(f"Linear {name}", "out_features", mod.out_features)
            findings += check(f"Linear {name}", "in_features", mod.in_features)
    return findings, model


def selftest():
    """The regression this file exists for: vocab 32773 must come back as a FAIL."""

    class C:
        vocab, d, ffn_hidden, heads = 32773, 1024, 3072, 8

    bad = [f for f in config_findings(C) if f["level"] == "FAIL" and f["dim"] == "vocab"]
    assert bad, "shape_audit no longer catches the vocab 32773 regression"
    return bad[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print("selftest OK:", selftest()["why"])
        return 0

    findings, model = audit(train.Cfg)
    fails = [f for f in findings if f["level"] == "FAIL"]
    warns = [f for f in findings if f["level"] == "WARN"]
    for f in fails + warns:
        print(
            f"[{f['level']}] {f['where']}.{f['dim']} = {f['value']}\n"
            f"        {f['why']}\n        fix: {f['fix']}"
        )
    n_lin = sum(isinstance(m, nn.Linear) for m in model.modules())
    print(
        f"shape_audit: {len(fails)} FAIL, {len(warns)} WARN over {n_lin} Linear layers "
        f"(align {FAST_ALIGN} for fast bf16, {FP8_ALIGN} for fp8 eligibility)"
    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "findings": findings,
                    "linear_layers": n_lin,
                    "fast_align": FAST_ALIGN,
                    "fp8_align": FP8_ALIGN,
                },
                f,
                indent=2,
            )
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

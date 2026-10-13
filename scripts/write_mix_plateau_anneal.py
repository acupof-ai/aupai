#!/usr/bin/env python3
# restartable: CPU mmap reads of pool sizes plus one small JSON written at the end; an interrupt
# leaves no partial mix file and a rerun recomputes the same weights.
"""Write the plateau-break ANNEAL mix: a short high-quality segment that decays LR to zero.

WHY. Stage-3 trained 54051 -> 62269 (about 6.4B tokens) on the stage-3 mix and HumanEval
stayed flat (paired test: no detectable difference). 83 of 129 common tasks failed at every
checkpoint, so more of the same distribution is not the lever. The hypothesis this segment
tests is the standard one (SmolLM2 / Qwen3-style decay on curated data): the last tokens a
model sees under a decaying LR shape it disproportionately, and teaching-grade rewritten code
and math move the hard-task tail more than another web-scale pass.

WHAT IT IS NOT. Not a new 100B recipe and not a guaranteed gain. It is one measurable arm:
resume from a stage-3 checkpoint, anneal on this mix, then run eval/paired_compare.py against
the resume point. If the paired interval contains zero, the hypothesis is rejected.

SHAPE. A resume mix: written against the checkpoint's row cursor, carries _derived_against
(train._assert_mix_derived_against refuses a different cursor). anneal_frac 0 in the file --
the WHOLE segment is the anneal; the LR decay comes from the schedule flags printed below,
not from build_mix's two phases.

DOMAINS AND WHY.
  swallowcode_scor_dc  tokyotech-llm/swallow-code exp11 SCOR, LLM-rewritten teaching-grade
                       Python, 17.82B measured, 4-gate decontaminated (0.024% dropped).
  swallow_math_dc      tokyotech-llm/swallow-math, finemath rewritten, 3.61B measured.
  cosmo_en_dc          HuggingFaceTB/cosmopedia textbook/tutorial text, 6.67B measured.
  code_ultra_l2_dc     replay of the stage-3 code distribution, so the decay does not
                       overwrite what the base already does well (forgetting control).
  en_fwe4_dc           replay of general English for the same reason.
Every share is a CATEGORY share; a domain whose measured pool cannot cover its draw is
REFUSED, never capped silently and never repeated.

LICENSES ARE A GATE, NOT A COMMENT. The stage-3 mix description says only license-allowed
domains are active. This writer refuses to emit a domain whose license is not confirmed in
--licenses (a JSON {domain: {"license": str, "confirmed_by": str}}). swallow-code/swallow-math
are derived from Llama-3.3-70B rewrites; their terms must be read and recorded by a person
before this mix can be written. That is deliberate: a script cannot decide a license.

    # on node0, after a person records licenses (no GPU, CPU mmap only):
    python3 scripts/write_mix_plateau_anneal.py \
        --resume /work/aupai/ckpt_s3a_seg1.pt.interrupt.step62269 \
        --tokens 6e9 --licenses data/licenses_plateau_anneal.json \
        --out data/mix_plateau_anneal.json
    python3 scripts/write_mix_plateau_anneal.py --check data/mix_plateau_anneal.json
    python3 scripts/write_mix_plateau_anneal.py --selftest
"""
import argparse
import json
import os
import sys

SEQ = 4096
ROOT = os.environ.get("AUPAI_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Category -> (domain, share). Shares sum to 1. Rationale per row in the module docstring.
RECIPE = {
    "swallowcode_scor_dc": 0.40,
    "swallow_math_dc": 0.20,
    "cosmo_en_dc": 0.15,
    "code_ultra_l2_dc": 0.15,
    "en_fwe4_dc": 0.10,
}
HQ_DOMAINS = ("swallowcode_scor_dc", "swallow_math_dc", "cosmo_en_dc")


def weight_for_rows(rows, budget):
    """Shortest decimal w with int(budget*w) == rows, so build_mix draws exactly `rows`."""
    for places in range(5, 14):
        w = round(rows / budget, places)
        if int(budget * w) == rows:
            return w
    raise AssertionError(f"no weight up to 13dp draws {rows} of {budget:g}")


def read_cursor(ckpt_path):
    """(row_cursor, srcfp, seed, step, total_steps) from a checkpoint, CPU mmap."""
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    return (dict(ck.get("row_cursor") or {}), dict(ck.get("row_cursor_srcfp") or {}),
            ck.get("row_cursor_seed"), int(ck.get("step", 0)), ck.get("total_steps"),
            (ck.get("cfg") or {}).get("v42_engram"))


def measure_pools(domains):
    """Trainable pool rows per domain via train's own cache path and val rule (mmap only)."""
    import torch
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    import cache_guard  # noqa: E402
    import train  # noqa: E402
    pools, details = {}, {}
    for name in domains:
        # Refuses a tens-of-GB read beside a live training claim on the same cache.
        cache_guard.assert_not_co_resident([name])
        cache = train._domain_cache_path(name)
        data = torch.load(cache, map_location="cpu", weights_only=True, mmap=True)
        n_rows = len(data) // (SEQ + 1)
        n_val = train.val_split_n(name, n_rows, {"domains": {name: {}}})
        with open(cache + ".srcfp") as fh:
            srcfp = fh.read().strip()
        pools[name] = n_rows - n_val
        details[name] = {"cache_seq_rows": n_rows, "val_rows_held_out": n_val, "srcfp": srcfp}
    return pools, details


def check_licenses(licenses, domains):
    missing = [d for d in domains
               if not (licenses.get(d) or {}).get("license") or not licenses[d].get("confirmed_by")]
    if missing:
        raise SystemExit(
            "refusing to write the mix: license not confirmed for " + ", ".join(missing)
            + ". Record {domain: {license, confirmed_by}} in --licenses after a person reads the "
              "source terms. A script cannot decide a license.")


def build(tokens, pools, details, cursor, srcfp, seed, step, licenses):
    total_rows = int(tokens // SEQ)
    domains = {}
    used_total = 0
    for name, share in RECIPE.items():
        used = int(cursor.get(name, 0))
        want = int(total_rows * share)
        avail = pools[name] - used
        if want > avail:
            raise SystemExit(f"{name}: wants {want:,} rows, only {avail:,} unread "
                             f"(pool {pools[name]:,}, cursor {used:,}). Lower --tokens or its share.")
        # build_mix: want = int(rows * 1.0 * weight) with rows = (total - spent) and
        # spent = sum(cursor over THIS mix's domains). Weights are derived against that rows.
        domains[name] = {"_want": want, "_used": used}
        used_total += used
    budget_rows = total_rows  # rows build_mix plans for this segment after subtracting spent
    out = {}
    for name, d in domains.items():
        w = weight_for_rows(d["_want"], budget_rows)
        # epochs 1: build_mix caps a draw at int(pool * epochs) - cursor. The unread check above
        # already guarantees want <= pool - cursor, so 1 never caps and never repeats.
        out[name] = {
            "weight": w, "anneal": w, "epochs": 1,
            "fingerprint": details[name]["srcfp"],
            "pool_rows_measured": pools[name], "cursor_rows": d["_used"], "segment_rows": d["_want"],
            "epochs_used": round((d["_used"] + d["_want"]) / pools[name], 4),
            "license": licenses.get(name, {}).get("license"),
        }
    return {
        "_comment": ("Plateau-break anneal segment. Whole segment decays LR to zero; anneal_frac 0 "
                     "because the decay is in the schedule flags, not in build_mix phases. "
                     "Regenerate with scripts/write_mix_plateau_anneal.py; never hand-edit."),
        "total_tokens": (budget_rows + used_total) * SEQ,
        "segment_rows": budget_rows,
        "anneal_frac": 0.0,
        "seq": SEQ,
        "resume_step": step,
        "domains": out,
        "_derived_against": {"row_cursor": {k: int(cursor[k]) for k in out if k in cursor},
                             "row_cursor_srcfp": {k: srcfp[k] for k in out if k in srcfp},
                             "row_cursor_seed": seed},
    }


def validate(m):
    """Every domain draws exactly its recorded rows and stays inside its unread pool."""
    rows = m["segment_rows"]
    for name, d in m["domains"].items():
        assert int(rows * d["weight"]) == d["segment_rows"], name
        assert d["epochs"] == 1, f"{name}: this segment never repeats data"
        assert d["cursor_rows"] + d["segment_rows"] <= d["pool_rows_measured"], f"{name} exceeds unread pool"
        assert d.get("license"), f"{name} has no recorded license"
    assert abs(sum(RECIPE.values()) - 1.0) < 1e-9
    return m


def launch_flags(m, ckpt, rows_per_step):
    """The schedule that makes this an anneal: re-join at resume step, short warmup, decay to 0."""
    steps = m["segment_rows"] // rows_per_step
    total = m["resume_step"] + steps
    return [
        "--resume", ckpt, "--mix", "data/mix_plateau_anneal.json",
        "--lr_origin_step", str(m["resume_step"]), "--lr_peak_mult", "0.10",
        "--warmup", "100", "--warmdown", "1.0",  # cosine over the whole segment to zero
        "--anneal_frac", "0.0", "--max_steps", str(total),
        "--v42_engram", "1,5,9,13,17,21",
    ], steps


def _selftest():
    pools = {"swallowcode_scor_dc": 4_349_725, "swallow_math_dc": 881_822, "cosmo_en_dc": 1_628_713,
             "code_ultra_l2_dc": 8_821_264, "en_fwe4_dc": 6_150_000}
    details = {k: {"srcfp": "f" * 16} for k in pools}
    lic = {k: {"license": "test", "confirmed_by": "selftest"} for k in pools}
    cursor = {"code_ultra_l2_dc": 3_000_000, "en_fwe4_dc": 500_000}
    m = build(6e9, pools, details, cursor, {k: "f" * 16 for k in cursor}, 42, 62269, lic)
    validate(m)
    assert m["segment_rows"] == int(6e9 // SEQ)
    assert m["domains"]["swallowcode_scor_dc"]["segment_rows"] == int(m["segment_rows"] * 0.40)
    # Over-ask refuses instead of capping.
    try:
        build(9e10, pools, details, cursor, {}, 42, 62269, lic)
        raise AssertionError("over-ask must refuse")
    except SystemExit:
        pass
    # Missing license refuses.
    try:
        check_licenses({}, HQ_DOMAINS)
        raise AssertionError("missing license must refuse")
    except SystemExit:
        pass
    flags, steps = launch_flags(m, "ckpt", 192)
    assert "--v42_engram" in flags and steps > 0
    print(f"write_mix_plateau_anneal selftest OK ({steps} steps at 192 rows/step)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--resume", help="checkpoint whose row cursor the mix is derived against")
    ap.add_argument("--tokens", type=float, default=6e9, help="segment tokens (default 6e9)")
    ap.add_argument("--licenses", help="JSON {domain: {license, confirmed_by}}")
    ap.add_argument("--rows_per_step", type=int, default=96, help="world8 x batch2 x accum6 = 96")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "mix_plateau_anneal.json"))
    ap.add_argument("--check")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return 0
    if args.check:
        m = validate(json.load(open(args.check, encoding="utf-8")))
        print(f"OK {args.check}: {m['segment_rows']:,} rows, {len(m['domains'])} domains, no repeat")
        return 0
    if not (args.resume and args.licenses):
        ap.error("--resume and --licenses are required to write a mix")
    licenses = json.load(open(args.licenses, encoding="utf-8"))
    check_licenses(licenses, list(RECIPE))
    cursor, srcfp, seed, step, _total, engram = read_cursor(args.resume)
    if engram != "1,5,9,13,17,21":
        raise SystemExit(f"checkpoint Engram layers are {engram!r}; this segment keeps 1,5,9,13,17,21")
    pools, details = measure_pools(list(RECIPE))
    m = validate(build(args.tokens, pools, details, cursor, srcfp, seed, step, licenses))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=1)
        fh.write("\n")
    flags, steps = launch_flags(m, args.resume, args.rows_per_step)
    print(f"wrote {args.out}: {m['segment_rows']:,} rows = {steps} steps at {args.rows_per_step} rows/step")
    print("launch (NOT run by this script): bash run_ddp.sh " + " ".join(flags))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

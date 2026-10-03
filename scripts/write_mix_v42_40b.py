#!/usr/bin/env python3
"""Write the V4.2 single-stage 40B mix: a main phase (85%) and an anneal phase (15%).

    python3 scripts/write_mix_v42_40b.py              # writes data/mix_v42_40b.json
    python3 scripts/write_mix_v42_40b.py --check data/mix_v42_40b.json

Derived artifact: per-domain trainable pools are measured by mmapping the token caches at
write time; never hand-edit. From scratch (no resume cursor), so every domain starts at
packed row 0 and epochs are drawn / trainable-pool.

Geometry (v42_gate_1001r ruling): world 16 x batch 4 x accum 3 = 192 rows/step = 786,432
tokens/step. 40B budget -> 50,862 steps = 9,765,504 rows. anneal_frac 0.15. build_mix
(train.py) draws each domain int(total_rows * frac * weight) per phase, independently, and
does NOT largest-remainder and does NOT require phase draws to sum to total_rows*frac. So
here: the phase budgets are the same floats (TOTAL_ROWS*.85 / *.15), the weight is chosen so
that int(budget*weight) equals the recorded draw exactly, and recorded rows are what train
will draw -- the sum lands a handful of rows short of the budget because of per-domain floor.

Composition (controller 2026-10-03):
  main   : code .78  math .10  en .05  zh .07
  anneal : code .55  math .25  zh .12  en .05  cot .03

Domain mapping / supply-driven choices:
  code   = the three Ultra/starcoder pools, split in trainable-pool proportion (each lands
           at ~0.42 epochs, no repeat).
  math   = math_owm_stage2_g4_dc + math_cot2_dc. The g4 owm domain is the v41 owm pool
           re-gated against GSM8K/MATH-500 as well (controller ruling v42-40b-decisions,
           2026-10-04); the v41 gate keeps reading the old frozen domain, so g4 is a separate
           name. cot2's whole trainable pool (288,190 rows) is the most a no-repeat run can
           draw; it sits in the anneal and owm fills the rest of the 25% math share (owm
           backfill accepted, no pure-CoT requirement).
  en/zh single-source (en_c4_stage2_dc / zh_web_dc); cot is cot_g4_dc (cot_dc four-way re-gated).

Every domain is 13-gram decontaminated for the gate benchmarks (suffix _dc) and every cache
carries the vocab/srcfp/seed triple; `fingerprint` records each cache's srcfp for the reader.
"""
import argparse
import json
import os
import sys

SEQ = 4096
ROWS_PER_STEP = 16 * 4 * 3  # 192 (world16 x batch4 x accum3) = 786,432 tokens
TOTAL_STEPS = 50862
TOTAL_ROWS = TOTAL_STEPS * ROWS_PER_STEP
MAIN_FRAC, ANN_FRAC = 0.85, 0.15
MAIN_BUDGET = TOTAL_ROWS * MAIN_FRAC   # float, as build_mix uses it
ANN_BUDGET = TOTAL_ROWS * ANN_FRAC

ROOT = os.environ.get("AUPAI_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Default lives next to the other mixes; override for running the writer from /tmp on the pod
# (the mix is pulled back, no pod_push while a gate runs).
OUT = os.environ.get("AUPAI_V42_MIX_OUT") or os.path.join(ROOT, "data", "mix_v42_40b.json")

CODE_DOMAINS = (
    "code_ultra_l2_dc",
    "code_ultra_l3_noexec_dc",
    "code_py_starcoder_dc",
)
MATH_DOMAINS = ("math_owm_stage2_g4_dc", "math_cot2_dc")
OWM_DOMAIN = MATH_DOMAINS[0]
EN_DOMAIN = "en_c4_stage2_dc"
ZH_DOMAIN = "zh_web_dc"
COT_DOMAIN = "cot_g4_dc"
DOMAINS = (*CODE_DOMAINS, *MATH_DOMAINS, EN_DOMAIN, ZH_DOMAIN, COT_DOMAIN)

# Category shares within each phase.
MAIN_CAT = {"code": 0.78, "math": 0.10, "en": 0.05, "zh": 0.07}
ANNEAL_CAT = {"code": 0.55, "math": 0.25, "zh": 0.12, "en": 0.05, "cot": 0.03}


def weight_for_rows(rows, budget):
    """Shortest-decimal w with int(budget*w) == rows, so build_mix draws exactly `rows`."""
    for places in range(5, 13):
        w = round(rows / budget, places)
        if int(budget * w) == rows:
            return w
    raise AssertionError(f"no weight up to 12dp draws {rows} of {budget:g}")


def measure_pools():
    """Trainable pool rows per domain, via train's own accessor and val rule.

    Mirrors scripts/write_mix_stage2_v41.py: mmap the cache (no tokenize), subtract the
    val holdout train.val_split_n computes. cache_guard refuses a tens-of-GB read beside a
    live claim; the 40B gate run is the intended job, the writer is not run concurrently.
    """
    import torch
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    import cache_guard  # noqa: E402

    import train  # noqa: E402

    pools, details = {}, {}
    for name in DOMAINS:
        cache_guard.assert_not_co_resident([name])
        cache = train._domain_cache_path(name)
        data = torch.load(cache, map_location="cpu", weights_only=True, mmap=True)
        n_rows = len(data) // (SEQ + 1)
        n_val = train.val_split_n(name, n_rows, {"domains": {name: {}}})
        pools[name] = n_rows - n_val
        with open(cache + ".srcfp") as fh:  # sidecar: tokens_x.pt.srcfp
            srcfp = fh.read().strip()
        details[name] = {
            "cache_seq_rows": n_rows,
            "val_rows_held_out": n_val,
            "pool_rows_measured": n_rows - n_val,
            "srcfp": srcfp,
        }
    return pools, details


# Domains carried verbatim from the prior (pre-g4) mix under the same name; the two g4
# refiltered domains are NOT here and must come from a measured override (refilter drops 132
# docs, so their packed pools must be read from the rebuilt caches, not carried over).
REUSED_FROM_PRIOR = (
    "code_ultra_l2_dc",
    "code_ultra_l3_noexec_dc",
    "code_py_starcoder_dc",
    "math_cot2_dc",
    "en_c4_stage2_dc",
    "zh_web_dc",
)
# Prior mix name -> g4 refiltered name, for a clear error if the override is missing.
G4_DOMAINS = {
    "math_owm_stage2_g4_dc": "math_owm_stage2_dc",
    "cot_g4_dc": "cot_dc",
}


def pools_from_mix(path, override_path):
    """Reuse the six unchanged domains' pools/triple-stamps recorded by an earlier write
    (frozen domains), and take the two g4-refiltered domains from a measured override JSON.

    override JSON: {<g4 domain>: {"pool_rows_measured": int, "cache_seq_rows": int,
    "val_rows_held_out": int, "srcfp": str}}. The g4 pools are rebuilt caches; never infer
    them from the pre-refilter numbers.
    """
    with open(path, encoding="utf-8") as fh:
        old = json.load(fh)["domains"]
    if override_path:
        with open(override_path, encoding="utf-8") as fh:
            override = json.load(fh)
    else:
        override = {}
    pools, details = {}, {}
    for name in REUSED_FROM_PRIOR:
        d = old[name]
        pools[name] = d["pool_rows_measured"]
        details[name] = {
            "cache_seq_rows": d["cache_seq_rows"],
            "val_rows_held_out": d["val_rows_held_out"],
            "pool_rows_measured": d["pool_rows_measured"],
            "srcfp": d["fingerprint"],
        }
    for name in G4_DOMAINS:
        if name not in override:
            raise SystemExit(
                f"missing measured pool override for refiltered domain {name}; rebuild its "
                "cache and supply --g4-pools with pool_rows_measured/cache_seq_rows/"
                "val_rows_held_out/srcfp")
        o = override[name]
        pools[name] = int(o["pool_rows_measured"])
        details[name] = {
            "cache_seq_rows": int(o["cache_seq_rows"]),
            "val_rows_held_out": int(o["val_rows_held_out"]),
            "pool_rows_measured": int(o["pool_rows_measured"]),
            "srcfp": o["srcfp"],
        }
    return pools, details


def build(pools, details):
    # Code sub-share within the code category = trainable-pool proportion.
    code_pool = sum(pools[c] for c in CODE_DOMAINS)
    code_frac = {c: pools[c] / code_pool for c in CODE_DOMAINS}

    # Anneal math: cot2's entire trainable pool goes in (no repeat); owm fills the rest of the
    # 25% math share. Draw owm as (floor(budget*.25) - cot2) so math lands on the category draw.
    cot2_rows = pools["math_cot2_dc"]
    math_ann_draw = int(ANN_BUDGET * ANNEAL_CAT["math"])
    owm_ann_rows = math_ann_draw - cot2_rows
    if owm_ann_rows < 0:
        raise SystemExit(
            f"math_cot2 pool {cot2_rows:,} exceeds the anneal math budget {math_ann_draw:,} "
            "rows; the math-supply audit must resolve this")

    # Desired within-phase share per domain (floats; code split by pool proportion).
    share = {d: {"m": 0.0, "a": 0.0} for d in DOMAINS}
    for c in CODE_DOMAINS:
        share[c]["m"] = MAIN_CAT["code"] * code_frac[c]
        share[c]["a"] = ANNEAL_CAT["code"] * code_frac[c]
    share[OWM_DOMAIN]["m"] = MAIN_CAT["math"]
    share[EN_DOMAIN]["m"] = MAIN_CAT["en"]
    share[ZH_DOMAIN]["m"] = MAIN_CAT["zh"]
    share[EN_DOMAIN]["a"] = ANNEAL_CAT["en"]
    share[ZH_DOMAIN]["a"] = ANNEAL_CAT["zh"]
    share[COT_DOMAIN]["a"] = ANNEAL_CAT["cot"]

    # Recorded draw = build_mix's own per-domain floor; cot2/owm anneal are supply-pinned.
    mrows, arows = {}, {}
    for d in DOMAINS:
        mrows[d] = int(MAIN_BUDGET * share[d]["m"])
        arows[d] = int(ANN_BUDGET * share[d]["a"])
    arows["math_cot2_dc"] = cot2_rows
    arows[OWM_DOMAIN] = owm_ann_rows

    domains = {}
    for name in DOMAINS:
        mr, ar = mrows[name], arows[name]
        total = mr + ar
        pool = pools[name]
        epochs = 1
        assert total <= pool, (
            f"{name}: draws {total:,} rows but trainable pool is {pool:,} (epochs {epochs}); "
            "lower its share, add supply, or (user decision) repeat -- this writer never repeats")
        wm = weight_for_rows(mr, MAIN_BUDGET) if mr else 0.0
        wa = weight_for_rows(ar, ANN_BUDGET) if ar else 0.0
        assert int(MAIN_BUDGET * wm) == mr and int(ANN_BUDGET * wa) == ar
        domains[name] = {
            "weight": wm,
            "anneal": wa,
            "epochs": epochs,
            "fingerprint": details[name]["srcfp"],
            "pool_rows_measured": pool,
            "cache_seq_rows": details[name]["cache_seq_rows"],
            "val_rows_held_out": details[name]["val_rows_held_out"],
            "main_rows": mr,
            "anneal_rows": ar,
            "total_rows": total,
            "epochs_used": round(total / pool, 4),
            "epochs_pool_source": (
                "mmap of the domain token cache tokens_"
                f"{name}.pt at write time (packed "
                f"{details[name]['cache_seq_rows']:,} seq-rows minus "
                f"{details[name]['val_rows_held_out']:,} val)"),
        }

    main_sum = sum(mrows.values())
    ann_sum = sum(arows.values())
    return {
        "_comment": (
            "V4.2 single-stage 40B mix (from scratch; world16 x batch4 x accum3 = 786,432 "
            f"tokens/step). {TOTAL_STEPS} steps = {TOTAL_ROWS:,} rows. anneal_frac 0.15: main "
            "code78/math10/en5/zh7; anneal code55/math25/zh12/en5/cot3. Every domain is _dc "
            "(13-word-token decontaminated) and every draw is within its trainable pool "
            "(max epochs 1.0, no repeat). build_mix floors per domain, so phase draws sum to "
            f"{main_sum:,}/{ann_sum:,} (a few rows under the {MAIN_BUDGET:,.0f}/"
            f"{ANN_BUDGET:,.0f} float budgets). math_cot2_dc's whole pool fills the anneal math "
            "share; the math-supply audit (math) confirms its usable size. Regenerate with "
            "scripts/write_mix_v42_40b.py; never hand-edit."),
        "total_tokens": TOTAL_ROWS * SEQ,
        "total_rows": TOTAL_ROWS,
        "seq": SEQ,
        "anneal_frac": 0.15,
        "rows_per_step": ROWS_PER_STEP,
        "total_steps": TOTAL_STEPS,
        "main_rows_drawn": main_sum,
        "anneal_rows_drawn": ann_sum,
        "domains": domains,
    }


def validate(path):
    with open(path, encoding="utf-8") as fh:
        m = json.load(fh)
    assert m["anneal_frac"] == 0.15
    assert m["total_rows"] == TOTAL_ROWS
    mtot = atot = 0
    for name, d in m["domains"].items():
        # build_mix reproduces exactly these draws.
        assert int(TOTAL_ROWS * MAIN_FRAC * d["weight"]) == d["main_rows"], name
        assert int(TOTAL_ROWS * ANN_FRAC * d["anneal"]) == d["anneal_rows"], name
        assert d["main_rows"] + d["anneal_rows"] == d["total_rows"], name
        assert d["total_rows"] <= d["pool_rows_measured"] * d["epochs"], name
        mtot += d["main_rows"]
        atot += d["anneal_rows"]
    assert mtot == m["main_rows_drawn"] and atot == m["anneal_rows_drawn"]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check")
    ap.add_argument("--pools-from",
                    help="reuse the six unchanged domains' measured pools/srcfp from an "
                         "earlier mix JSON (frozen domains; no pod mmap). Default mmaps all caches.")
    ap.add_argument("--g4-pools",
                    help="with --pools-from, JSON of measured pools for the two g4 "
                         "refiltered domains (required).")
    args = ap.parse_args()
    if args.check:
        m = validate(args.check)
        print(f"OK {args.check}: main {m['main_rows_drawn']:,} + anneal "
              f"{m['anneal_rows_drawn']:,} rows, every draw reproduced by build_mix and within pool")
        return 0
    if args.pools_from:
        pools, details = pools_from_mix(args.pools_from, args.g4_pools)
    else:
        pools, details = measure_pools()
    mix = build(pools, details)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(mix, fh, indent=1)
        fh.write("\n")
    print(f"wrote {OUT}: {TOTAL_STEPS} steps ({TOTAL_ROWS*SEQ/1e9:.4f}B), main "
          f"{mix['main_rows_drawn']:,} + anneal {mix['anneal_rows_drawn']:,}")
    for n, d in mix["domains"].items():
        print(f"  {n:26} main {d['main_rows']:>8,} ann {d['anneal_rows']:>7,} "
              f"tot {d['total_rows']:>8,} / pool {d['pool_rows_measured']:>9,} ep {d['epochs_used']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

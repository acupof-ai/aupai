#!/usr/bin/env python3
"""Write the V4.1 CED stage-2 continuation mix, derived from the 30B final checkpoint.

    python3 scripts/write_mix_stage2_v41.py --ckpt ckpt_v41_ced_0926.pt --mode full
    python3 scripts/write_mix_stage2_v41.py --ckpt ckpt_v41_ced_0926.pt --mode smoke
    python3 scripts/write_mix_stage2_v41.py --check data/mix_v41_stage2.json

Runs ON THE POD: it reads the checkpoint cursor and mmaps every token cache to measure
each trainable pool at runtime -- no supply constant is transcribed. The stage-2 mix is a
derived artifact; regenerate it against the resume checkpoint, never hand-edit.

Geometry: world 8 x batch 4 x accum 6 = 192 packed rows/step. The full segment is 12,715
steps = 2,441,280 rows = 9.999B tokens on top of the 30B cursor. The smoke segment is 50
steps = 9,600 rows over the six domains that already exist (zh_web_dc and
math_cot2_dc are still being built and are absent until --mode full).

LR is not in the mix: the launcher passes --lr_origin_step <ckpt step> --lr_peak_mult
0.30 --warmup 500 --warmdown 1.0 --anneal_frac 0.0, which rebases the schedule at the
join (absolute 500-step re-warmup to 30% of peak, cosine to zero).
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))
import cache_guard  # noqa: E402
import train  # noqa: E402

SEQ = 4096
ROWS_PER_STEP = 8 * 4 * 6  # 192

# Domains carried over from stage 1 (names identical, so the checkpoint cursor seeds them).
CARRYOVER = {
    "code_ultra_l2_dc", "code_ultra_l3_noexec_dc", "code_py_starcoder_dc",
    "math_owm_stage2_dc", "en_c4_stage2_dc",
}
# New stage-2 domains with no checkpoint cursor; valid only in full mode.
FULL_ONLY = {"zh_web_dc", "math_cot2_dc"}

# Full-mode target composition, in segment rows. code 60 / zh 15 / math 20 / en 5.
FULL_STEPS = 12_715
FULL_SHARES = [
    # (domain, share, epochs)  code split 2.5/2.5/1.0B, math 1.2B owm + 0.8B cot2
    ("code_ultra_l2_dc", 0.25, 1),
    ("code_ultra_l3_noexec_dc", 0.25, 1),
    ("code_py_starcoder_dc", 0.10, 1),
    ("zh_web_dc", 0.15, 1),
    ("math_owm_stage2_dc", 0.12, 1),
    ("math_cot2_dc", 0.08, 1),
    ("en_c4_stage2_dc", 0.05, 1),
]

# Smoke mode: 50 steps over the six existing domains. zh_web_dc is absent, so its 15% is
# spread over the survivors; proportions are not the experiment here, cap coverage is.
# cot_dc carries epochs=2: its stage-1 cursor is already past one trainable pool, which
# exercises the modulo/cap path with a real repeat.
SMOKE_STEPS = 50
SMOKE_SHARES = [
    ("code_ultra_l2_dc", 0.35, 1),
    ("code_ultra_l3_noexec_dc", 0.20, 1),
    ("code_py_starcoder_dc", 0.10, 1),
    ("math_owm_stage2_dc", 0.20, 1),
    ("en_c4_stage2_dc", 0.05, 1),
    ("cot_dc", 0.10, 2),
]

OUT = {
    "full": "data/mix_v41_stage2.json",
    "smoke": "data/mix_v41_stage2_smoke.json",
}


def read_cursor(ckpt_path):
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {
        "step": int(ck["step"]),
        "row_cursor": {k: int(v) for k, v in ck["row_cursor"].items()},
        "row_cursor_srcfp": dict(ck["row_cursor_srcfp"]),
        "row_cursor_seed": ck["row_cursor_seed"],
    }


def measure_pool(name):
    """Trainable pool rows = packed seq-rows in the cache minus the val holdout.

    Mirrors train._domain_seqs + train.val_split_n exactly: the cache is flat tokens
    reshaped to [-1, seq+1], then n_val = min(max(1, int(n*0.05)), 5000) come off the
    front. Measured at runtime through train's own accessors, never transcribed.
    """
    import torch
    cache_guard.assert_not_co_resident([name])
    cache = train._domain_cache_path(name)
    data = torch.load(cache, map_location="cpu", weights_only=True, mmap=True)
    n_rows = len(data) // (SEQ + 1)
    n_val = train.val_split_n(name, n_rows, {"domains": {name: {}}})
    return n_rows, n_val, n_rows - n_val


def weight_for_rows(rows, total):
    for places in range(5, 13):
        w = round(rows / total, places)
        if int(total * w) == rows:
            return w
    raise AssertionError(f"no weight up to 12dp draws {rows} of {total}")


def largest_remainder(total, shares):
    raw = [(n, total * s, ep) for n, s, ep in shares]
    base = {n: int(v) for n, v, _ in raw}
    leftover = total - sum(base.values())
    order = [n for n, v, _ in sorted(raw, key=lambda t: t[1] - int(t[1]), reverse=True)]
    for n in order[:leftover]:
        base[n] += 1
    return base


def build(cur, mode):
    shares = FULL_SHARES if mode == "full" else SMOKE_SHARES
    seg_steps = FULL_STEPS if mode == "full" else SMOKE_STEPS
    R = seg_steps * ROWS_PER_STEP
    rows = largest_remainder(R, shares)
    assert abs(sum(s for _, s, _ in shares) - 1.0) < 1e-9, "shares must sum to 1"
    assert sum(rows.values()) == R
    cursor = cur["row_cursor"]
    domains = {}
    for name, _share, epochs in shares:
        is_new = name in FULL_ONLY
        used = 0 if is_new else cursor[name]
        if not is_new:
            assert name in cur["row_cursor_srcfp"], f"{name}: carryover domain missing cursor srcfp"
        n_rows, n_val, pool = measure_pool(name)
        # epochs=1 assumes unread rows cover the draw for carryover domains; the assert below
        # proves it. cot_dc in smoke mode has a cursor past one pool, so it passes epochs=2.
        cap = pool * epochs - used
        want = rows[name]
        assert want <= cap, (
            f"{name}: wants {want} rows but epoch cap leaves {cap} (pool {pool} x {epochs} "
            f"- cursor {used}); lower its share or raise epochs")
        w = weight_for_rows(want, R)
        e = {
            "weight": w,
            "anneal": w,  # anneal_frac 0; kept equal so a nonzero anneal_frac fails honestly
            "epochs": epochs,
            "fingerprint": cur["row_cursor_srcfp"].get(name),
            "pool_rows_measured": pool,
            "cache_seq_rows": n_rows,
            "val_rows_held_out": n_val,
            "cursor_used_rows": used,
            "unread_rows": pool - used if not is_new else pool,
            "segment_rows": want,
            "epochs_pool_source": (
                f"mmap of {train._domain_cache_path(name)} at write time "
                f"(packed {n_rows:,} seq-rows minus {n_val:,} val)"),
        }
        if is_new:
            e.pop("fingerprint")
            e["stage2_only_domain"] = True
        domains[name] = e
    da_rows = {n: cursor[n] for n, _, _ in shares if n in cursor}
    da_fp = {n: cur["row_cursor_srcfp"][n] for n in da_rows}
    cursor_total = sum(da_rows.values())  # mix-named domains only; build_mix subtracts the same sum
    total_rows = cursor_total + R
    return {
        "_comment": (
            f"V4.1 CED stage-2 {mode} mix, derived against step {cur['step']} of the 30B run "
            f"({cur.get('ckpt_name', 'ckpt_v41_ced_0926.pt')}). Segment {seg_steps} steps = "
            f"{R:,} plan rows ({ROWS_PER_STEP}/step) = {R * SEQ / 1e9:.3f}B tokens; "
            f"total_tokens is cursor {cursor_total:,} rows + segment = {total_rows:,} rows. "
            "Single phase (anneal_frac 0). epochs are TOTALS against the trainable pool and "
            "every pool was measured from its cache at write time. Launch with "
            f"--lr_origin_step {cur['step']} --lr_peak_mult 0.30 --warmup 500 --warmdown 1.0 "
            "--anneal_frac 0.0. Regenerate with scripts/write_mix_stage2_v41.py; never edit."),
        "total_tokens": total_rows * SEQ,
        "total_rows": total_rows,
        "seq": SEQ,
        "anneal_frac": 0.0,
        "join_step": cur["step"],
        "segment_steps": seg_steps,
        "segment_plan_rows": R,
        "domains": domains,
        "_derived_against": {
            "row_cursor": da_rows,
            "row_cursor_srcfp": da_fp,
            "row_cursor_seed": cur["row_cursor_seed"],
            "_note": (f"30B resume cursor at step {cur['step']}. Stage-2-only domains "
                      f"{sorted(FULL_ONLY)} are deliberately absent: they start at packed "
                      "row 0 and the checkpoint carries no cursor for them."),
        },
    }


def validate(path):
    with open(path, encoding="utf-8") as fh:
        m = json.load(fh)
    assert m["anneal_frac"] == 0.0
    R = int(m["segment_plan_rows"])
    assert int(m["segment_steps"]) * ROWS_PER_STEP == R
    da = m["_derived_against"]
    total = 0
    for name, d in m["domains"].items():
        want = d["segment_rows"]
        total += want
        assert int(R * d["weight"]) == want, (name, d["weight"], want)
        assert d["anneal"] == d["weight"]
        assert int(d["pool_rows_measured"] * d["epochs"]) - d["cursor_used_rows"] >= want, name
        if name in da["row_cursor"]:
            assert d["cursor_used_rows"] == da["row_cursor"][name], name
            assert d["fingerprint"] == da["row_cursor_srcfp"][name], name
        else:
            assert d.get("stage2_only_domain"), f"{name}: no cursor and not marked stage-2-only"
    assert total == R, (total, R)
    assert m["total_rows"] == sum(da["row_cursor"].values()) + R
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--mode", choices=["full", "smoke"])
    ap.add_argument("--check")
    args = ap.parse_args()
    if args.check:
        R = validate(args.check)
        print(f"OK {args.check}: segment rows sum to {R:,}, caps measured, cursor triple present")
        return 0
    if not args.ckpt or not args.mode:
        ap.error("--ckpt and --mode are required (or --check <mix>)")
    cur = read_cursor(args.ckpt)
    cur["ckpt_name"] = os.path.basename(args.ckpt)
    mix = build(cur, args.mode)
    out = OUT[args.mode]
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(mix, fh, indent=1)
        fh.write("\n")
    print(f"wrote {out}: join step {mix['join_step']}, segment {mix['segment_steps']} steps "
          f"({mix['segment_plan_rows']:,} rows = {mix['segment_plan_rows']*SEQ/1e9:.3f}B)")
    for n, d in mix["domains"].items():
        print(f"  {n:28} want {d['segment_rows']:>7,} pool {d['pool_rows_measured']:>9,} "
              f"cursor {d['cursor_used_rows']:>9,} ep {d['epochs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

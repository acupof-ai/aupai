#!/usr/bin/env python3
"""Write the two v42 textbook-continuation stage-2 mixes (T/C), derived from an r3 checkpoint.

    python3 scripts/write_mix_v42_stage2.py --ckpt ckpt_v41_r3_0914.pt.step23000   # write T and C
    python3 scripts/write_mix_v42_stage2.py --check                              # validate committed JSON (CI)

The mixes are derived artifacts: regenerate against the resume checkpoint, never hand-edit.
Run this ON THE POD (it reads the ckpt cursor); the token caches are not needed because the
pool rows are measured constants (the caches are frozen and pool rows do not change between
the step23000 dry-run and the r3-final regeneration).

Why two files:
  data/mix_textbook_cont.json  T = textbook_claude_v41_dc at weight TB_W + the six r3 domains
                                on the residual. Textbook draws EXACTLY 4 trainable-pool epochs.
  data/mix_cont_ctrl.json      C = the same six domains at their r3 anneal proportions, sum 1.

Segment geometry (fb ruling 2026-09-14, amendment_3):
  SEG=137 steps, world 8 x batch 4 x accum 6 = 192 rows/step -> R = 26,304 plan rows, single
  phase (anneal_frac 0; it is one cosine warmdown tail). Textbook trainable pool is 1,972 rows
  (2,075 packed seq-rows MINUS 103 held out by the default 5% val split; the 5,000 cap does not
  bind on a domain this small), so 4 epochs = 7,888 rows -- NOT the cache's 2,075-row total.
  build_mix's epoch cap runs on the trainable pool AFTER val, which is the only row count it
  caps against. At weight 0.2999 int(R*w) == 7,888 exactly; the six domains share the 18,416
  residual. The six T weights and all C weights are chosen so int(R*w) sums to EXACTLY R by
  largest-remainder, which keeps total segment steps at 137 (a 1-row floor shortfall would make
  train.py floor total_steps to 136 and move the warmdown ratio).

Plain *_dc binding: the six domains point at their FULL plain caches and corpus fingerprints
(the srcfp the r3 cursor carries), never the .excl holdout cache. No domain carries
cache_exclude: that field belongs to stage 1's SFT-holdout contract, and the stage-2 launch
binds the plain pools the cursor was written against.
"""
import argparse
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T_OUT = os.path.join(ROOT, "data", "mix_textbook_cont.json")
C_OUT = os.path.join(ROOT, "data", "mix_cont_ctrl.json")

SEQ = 4096
WORLD, BATCH, ACCUM = 8, 4, 6
ROWS_PER_STEP = WORLD * BATCH * ACCUM  # 192
SEG = 137
TB_NAME = "textbook_claude_v41_dc"
TB_W = 0.2999  # int(R*TB_W) == 4 * TB pool rows (7,888); fb ruling, amendment_3

# The six r3 continuation domains, in r3 mix order. ANN is the r3 anneal proportion set
# (.45/.25/.18/.05/.02/.05, sums to 1): the C arm draws at exactly these, and the T arm's six
# weights are these proportions over the residual left after the textbook share.
SIX = [
    ("code_ultra_l3_stub_dc", 0.45),
    ("code_keep_p1_dc", 0.25),
    ("code_ultra_l2_dc", 0.18),
    ("math_owm_stage2_dc", 0.05),
    ("en_c4_stage2_dc", 0.02),
    ("cot_dc", 0.05),
]

# Trainable pool ROWS (packed seq-rows minus the domain's val split), measured on the pod
# 2026-09-14 through train._domain_seqs + train.val_split_n against the live plain caches.
# These are the exact len(pool) build_mix caps with. The 5% cap binds for the six (n_val 5000)
# and the per-domain 5% binds for the textbook (2,075 seq-rows -> n_val 103).
POOL_ROWS = {
    "code_ultra_l3_stub_dc": 1_213_934,  # 1,218,934 seq - 5,000
    "code_keep_p1_dc": 636_195,          # 641,195 - 5,000
    "code_ultra_l2_dc": 3_739_417,       # 3,744,417 - 5,000
    "math_owm_stage2_dc": 1_424_819,     # 1,429,819 - 5,000
    "en_c4_stage2_dc": 479_552,          # 484,552 - 5,000
    "cot_dc": 92_750,                    # 97,631 seq; 5% = 4,881 < 5,000 cap
    TB_NAME: 1_972,                      # 2,075 seq - 103 val (5%, cap does not bind)
}

# Plain *_dc corpus fingerprints (the .srcfp each cache carries), read on the pod 2026-09-14.
# The six are identical to the r3 step23000 cursor's row_cursor_srcfp, which is why the cursor
# seeds them. The textbook fp is the rebuilt 0e pool. No cache_exclude anywhere.
FINGERPRINT = {
    "code_ultra_l3_stub_dc": "12ec3cd220b8fe13",
    "code_keep_p1_dc": "d7b4f3a09c5707b0",
    "code_ultra_l2_dc": "adf2ff20698dae8a",
    "math_owm_stage2_dc": "4b1469bfe4667706",
    "en_c4_stage2_dc": "c59c2e4227d875ec",
    "cot_dc": "0d9f495913438657",
    TB_NAME: "e473e748ae3ea4e2",
}

POOL_SOURCE = ("trainable pool rows measured on the pod 2026-09-14 via train._domain_seqs + "
               "train.val_split_n over each live plain domain token cache via "
               "train._domain_cache_path (packed "
               "seq-rows minus the domain val split); vocab stamp f1f860970d15d623.")


def read_cursor(ckpt):
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    return {
        "step": int(ck.get("step", 0)),
        "row_cursor": {k: int(v) for k, v in (ck.get("row_cursor") or {}).items()},
        "row_cursor_srcfp": dict(ck.get("row_cursor_srcfp") or {}),
        "row_cursor_seed": ck.get("row_cursor_seed"),
    }


def weight_for_rows(rows, R):
    """Shortest decimal weight w with int(R*w) == rows exactly (build_mix floors per phase)."""
    for places in range(5, 13):
        w = round(rows / R, places)
        if int(R * w) == rows:
            return w, places
    raise AssertionError(f"no weight up to 12dp draws {rows} rows of {R}")


def largest_remainder(total, props):
    """Integer rows summing exactly to total, apportioned by props via largest remainder."""
    raw = {k: total * p for k, p in props}
    base = {k: int(v) for k, v in raw.items()}
    leftover = total - sum(base.values())
    order = sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)
    out = dict(base)
    for k in order[:leftover]:
        out[k] += 1
    assert sum(out.values()) == total
    return out


def domain_entry(name, rows, used, epochs, w, places, role, cursor_fp):
    cap = int(POOL_ROWS[name] * epochs) - used
    assert rows <= cap, (f"{name}: wants {rows} but epoch cap leaves {cap} "
                         f"(pool {POOL_ROWS[name]} x {epochs} - used {used}); build_mix would cap")
    e = {
        "weight": w,
        "anneal": w,  # single phase: anneal share equals main share (defensive; anneal_frac is 0)
        "epochs": epochs,
        "fingerprint": FINGERPRINT[name],
        "role": role,
        "pool_rows": POOL_ROWS[name],
        "cursor_used_rows": used,
        "segment_rows": rows,
        "cumulative_epochs_on_pool": round((used + rows) / POOL_ROWS[name], 4),
        "weight_decimals": places,
        "rows_from_weight_at_runtime": int(26_304 * w),
        "epochs_pool_source": POOL_SOURCE,
    }
    if cursor_fp is not None:
        e["cursor_srcfp"] = cursor_fp
    return e


def build_mixes(cur):
    R = SEG * ROWS_PER_STEP
    assert R == 26_304
    spent = cur["row_cursor"]
    # The textbook must be absent from the cursor: a stage-2 domain that legitimately starts at
    # packed row 0. The six must all be present and fingerprint-match the cache constants.
    assert TB_NAME not in spent, f"{TB_NAME} unexpectedly carries an r3 cursor; must start row 0"
    for name, _ in SIX:
        assert name in spent, f"{name} missing from the r3 cursor"
        assert cur["row_cursor_srcfp"].get(name) == FINGERPRINT[name], (
            f"{name}: cursor srcfp {cur['row_cursor_srcfp'].get(name)} != plain cache "
            f"{FINGERPRINT[name]}")

    tb_rows = int(R * TB_W)
    assert tb_rows == 4 * POOL_ROWS[TB_NAME] == 7_888, tb_rows

    props = [(n, p) for n, p in SIX]
    t_six_rows = largest_remainder(R - tb_rows, props)
    c_six_rows = largest_remainder(R, props)

    def make(six_rows, arm):
        domains = {}
        if arm == "T":
            assert int(R * TB_W) == tb_rows, (TB_W, int(R * TB_W), tb_rows)
            w, p = TB_W, 4  # fb pinned 0.2999; 0.29988 also ints to 7,888 but 0.2999 is the ruling
            domains[TB_NAME] = domain_entry(
                TB_NAME, tb_rows, 0, 4, w, p,
                "TREATMENT: rebuilt Claude Python textbook (0e 1938-row keep pool); exactly 4 "
                "trainable-pool epochs (7,888 rows), starts row 0 (no r3 cursor)", None)
        check_sum = tb_rows if arm == "T" else 0
        for name, _prop in props:
            rows = six_rows[name]
            used = spent[name]
            epochs = math.ceil((used + rows) / POOL_ROWS[name])
            w, p = weight_for_rows(rows, R)
            role = ("continuation domain shared with the C arm; six-domain T share is the r3 "
                    "anneal proportion over the residual" if arm == "T"
                    else "CONTROL: r3 anneal proportion over the full segment")
            domains[name] = domain_entry(
                name, rows, used, epochs, w, p, role, cur["row_cursor_srcfp"][name])
            check_sum += int(R * w)
        assert check_sum == R, (arm, check_sum, R)
        total_rows = sum(spent[n] for n, _ in props) + R
        join_step = cur["step"]
        warmdown = round(SEG / (join_step + SEG), 6)
        total_after = join_step + SEG
        return {
            "_comment": (
                f"v42 stage-2 {arm} arm, derived against {cur.get('ckpt_name', 'the r3 resume ckpt')} "
                f"(step {join_step}). Single-phase {SEG}-step segment = {R} plan rows "
                f"({WORLD}x{BATCH}x{ACCUM}/step), anneal_frac 0; total_steps is "
                f"{join_step}+{SEG}={total_after} and warmdown {SEG}/{total_after}={warmdown} "
                f"(set --warmdown on the launch line). Plain *_dc caches, NO cache_exclude. "
                f"Regenerate with scripts/write_mix_v42_stage2.py --ckpt <final r3 ckpt> before "
                f"launch; the dry-run asserts this warmdown against the ckpt it is run with."),
            "total_tokens": total_rows * SEQ,
            "total_rows": total_rows,
            "seq": SEQ,
            "anneal_frac": 0.0,
            "segment_steps": SEG,
            "segment_plan_rows": R,
            "join_step": join_step,
            "warmdown": warmdown,
            "domains": domains,
            "_derived_against": {
                "row_cursor": {n: spent[n] for n, _ in props},
                "row_cursor_srcfp": {n: cur["row_cursor_srcfp"][n] for n, _ in props},
                "row_cursor_seed": cur["row_cursor_seed"],
                "_note": (f"r3 resume cursor at step {cur['step']} (six domains). "
                          f"{TB_NAME} is deliberately absent: a new stage-2 domain starting at "
                          "row 0. epochs are TOTALS = cursor rows + this segment."),
            },
        }

    return make(t_six_rows, "T"), make(c_six_rows, "C")


def validate(path):
    """Structural validation of a committed mix, with no checkpoint (CI)."""
    m = json.load(open(path, encoding="utf-8"))
    R = SEG * ROWS_PER_STEP
    assert m["anneal_frac"] == 0.0, path
    assert m["segment_plan_rows"] == R
    # warmdown must be present and equal SEG/(join_step+SEG) for the join_step the mix records.
    join = int(m["join_step"])
    assert abs(float(m["warmdown"]) - round(SEG / (join + SEG), 6)) < 5e-7, (
        path, m.get("warmdown"), join)
    doms = m["domains"]
    assert "cache_exclude" not in json.dumps(doms), f"{path}: cache_exclude leaked into a domain"
    total = 0
    da = m["_derived_against"]
    for name, d in doms.items():
        rows = d["segment_rows"]
        assert int(R * d["weight"]) == rows, (name, d["weight"], rows)
        assert d["anneal"] == d["weight"]
        total += rows
        used = d["cursor_used_rows"]
        assert int(d["pool_rows"] * d["epochs"]) - used >= rows, name
        if name != TB_NAME:
            assert used == da["row_cursor"][name], name
            assert d["fingerprint"] == da["row_cursor_srcfp"][name], name
    assert total == R, (path, total, R)
    if TB_NAME in doms:
        tb = doms[TB_NAME]
        assert tb["cursor_used_rows"] == 0
        assert tb["segment_rows"] == 4 * tb["pool_rows"] == 7_888
        assert tb["epochs"] == 4
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="r3 checkpoint to derive the cursor from (write mode)")
    ap.add_argument("--check", action="store_true", help="validate committed T and C JSON, no ckpt")
    args = ap.parse_args()
    if args.check:
        for p in (T_OUT, C_OUT):
            n = validate(p)
            print(f"OK {os.path.basename(p)}: segment rows sum to {n} == {SEG*ROWS_PER_STEP}, "
                  f"epochs cover cursor+segment, no cache_exclude")
        return 0
    if not args.ckpt:
        ap.error("--ckpt is required to write the mixes (use --check to validate committed ones)")
    cur = read_cursor(args.ckpt)
    cur["ckpt_name"] = os.path.basename(args.ckpt)
    t, c = build_mixes(cur)
    for path, obj in ((T_OUT, t), (C_OUT, c)):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1)
            fh.write("\n")
        print(f"wrote {path}: {len(obj['domains'])} domains, "
              f"total_tokens={obj['total_tokens']:,} ({obj['total_rows']} rows budget), "
              f"segment {SEG} steps")
    print("next: python scripts/dryrun_v42_textbook_ab.py --t %s --c %s --ckpt %s"
          % (os.path.basename(T_OUT), os.path.basename(C_OUT), args.ckpt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

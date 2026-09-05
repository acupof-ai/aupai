#!/usr/bin/env python3
"""Generate data/mix_e1_<arm>.json for experiment 1 from MEASURED pool rows.

    python3 scripts/e1_build_arm_mixes.py --write
    python3 scripts/e1_build_arm_mixes.py            # print, change nothing

WHY THIS EXISTS RATHER THAN FIVE HAND-EDITED FILES. The injection weights are
`packed_rows / budget_rows`, and packed_rows is a property of the tokenized stream that
nobody can type correctly: it depends on the tokenizer, on seq+1 packing, and on the <eos>
train.encode appends per document. Every previous value was hand-derived and every one was
wrong at least once:

  * The first set used the prereg's 104 tokens/doc mean for BOTH pools, but P's mean is
    84.43, so p_format asked 25 rows of a 20-row pool.
  * The second set divided the token count by seq (4096) where a pool packs at seq+1, so
    every arm asked one row more than it held.
  * The third set -- the one this script replaces -- counted sum(len(ids)) and MISSED THE
    <eos> train.encode appends per document (train.py:1707: np.append(p, eos)). At n8 that
    is 8,000 tokens, so the pool holds 204 rows and the weight asked 202. Measured against
    the caches the pod built: 204/1639/6557 against 202/1623/6495.

That last one is the reason this file exists at all. The error was INVISIBLE to the plan
check, because check_arm derived `want` and `pool_rows` from the SAME undercount -- so
`want == pool` passed on a shared error, five arms green, twice reviewed. The count now has
exactly one source: this script tokenizes the shard the way train.encode does and writes the
number it gets. The check then recomputes it independently and compares against the CACHE
the pod built, which is the only reading neither of them produced.

p_format and n1 agree either way (25.4 rows; the floor absorbs 1,000 tokens), which is why
the two smallest arms could not reveal the defect -- a reminder that an arm too small to
discriminate is not evidence.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

BASE = os.path.join(ROOT, "data", "mix_200m_8b.json")
ARMS = (("control_arm", 0), ("n1", 1), ("n8", 8), ("n64", 64), ("n256", 256))
#: Rows ckpt_b0_headmix_armA.pt's row_cursor already consumed; confirmed live on the pod
#: 2026-09-05 (as_of_step 3815, seed 42, nine domains). build_mix subtracts this before
#: computing want, so the weights are shares of the REMAINDER, not of total_tokens.
CURSOR_ROWS = 244160
ARM_ROWS = 32000  # the arm's own budget, on top of the cursor


def packed_rows(domain, seq, tok):
    """Rows the token cache will hold for `domain`, counted the way train.encode builds it.

    THE <eos> IS THE POINT. train.encode emits one <eos>-separated stream --
    np.append(p, np.int32(eos)) per document -- so the stream is sum(len(ids)) + n_docs and
    the cache reshapes it at seq+1 (train.py:1957, `n = len(data) // (seq+1)`). Dropping the
    +n_docs is the defect this script was written to end.
    """
    p = os.path.join(ROOT, "data", "corpus", domain, f"{domain}_000.jsonl")
    toks, ndocs = 0, 0
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "_header" in r:
                raise SystemExit(
                    f"REFUSING: {os.path.relpath(p, ROOT)} carries an in-band _header line; "
                    f"train._jsonl_content reads [\"content\"] on every line and would raise. "
                    f"Rebuild with datagen/build_s_inject.py.")
            toks += len(tok.encode(r["content"]).ids)
            ndocs += 1
    return (toks + ndocs) // (seq + 1), toks + ndocs, ndocs


def build(arm, n, base, tok, budget_rows):
    inj = f"s_inject_n{n}"
    doms = {}
    rows = {}
    for d in ([] if n == 0 else [inj]) + ["p_format"]:
        pr, stream, ndocs = packed_rows(d, base["seq"], tok)
        rows[d] = (pr, stream, ndocs)
    w_s = rows[inj][0] / budget_rows if n else 0.0
    w_p = rows["p_format"][0] / budget_rows
    # THE NATURAL DOMAINS KEEP THE CONTROL'S RATIOS EXACTLY. Scaling by
    # (1 - w_S - w_P)/sum(base weights) rather than by (1 - w_S - w_P) divides out the base's
    # own sum (1.00000005851), so every ratio between two natural domains is the base's to
    # 1e-16 while this arm's weights sum to 1.
    base_sum = sum(v["weight"] for v in base["domains"].values())
    scale = (1.0 - w_s - w_p) / base_sum
    for name, v in base["domains"].items():
        d = dict(v)
        d["weight"] = v["weight"] * scale
        if "anneal" in d:
            d["anneal"] = d["weight"]
        d["role"] = (
            (v.get("role", "").split(" Weight is ")[0].rstrip()
             + f" Weight is mix_200m_8b's x {scale!r} = (1 - w_S - w_P)/{base_sum!r}, so every "
               f"ratio between two natural domains is exactly the control's while this arm's "
               f"total is 1.").strip())
        doms[name] = d
    for d, (pr, stream, ndocs) in rows.items():
        w = w_s if d.startswith("s_inject") else w_p
        fam = "S injection" if d.startswith("s_inject") else "P format control"
        doms[d] = {
            "weight": w,
            "anneal": w,
            "epochs": 1,
            "val_frac": 0,
            "weight_decimals": 12,
            "role": (
                f"{fam}: {ndocs} document lines, {stream} tokens in the <eos>-separated stream "
                f"({ndocs} of them the per-document <eos> train.encode appends), packing to "
                f"{pr} rows at seq+1={base['seq'] + 1}. Weight = {pr}/{budget_rows} budget rows, "
                f"COMPUTED by scripts/e1_build_arm_mixes.py rather than typed: the three previous "
                f"hand-derived values were each wrong (P's mean read as S's 104, a division by "
                f"seq instead of seq+1, and a missing <eos> per document). val_frac 0 because "
                f"this domain's row count IS the measurement."),
        }
    out = {k: v for k, v in base.items() if k != "domains"}
    out["total_tokens"] = (CURSOR_ROWS + ARM_ROWS) * base["seq"]
    out["total_rows"] = CURSOR_ROWS + ARM_ROWS
    out["seq"] = base["seq"]
    out["_comment"] = (
        f"Experiment 1 arm {arm} (runs/prereg.jsonl#conversion_rate_0905). GENERATED by "
        f"scripts/e1_build_arm_mixes.py -- do not hand-edit a weight; rerun it. Resumes "
        f"ckpt_b0_headmix_armA.pt, whose row_cursor sums to {CURSOR_ROWS} rows, so total_tokens "
        f"carries cursor + this arm's {ARM_ROWS} rows and build_mix's weights are shares of the "
        f"{budget_rows}-row remainder.")
    out["domains"] = doms
    return out, w_s, w_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write the five files (default: print)")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
    with open(BASE, encoding="utf-8") as fh:
        base = json.load(fh)
    budget_rows = ARM_ROWS  # (cursor + arm) - cursor; stated as the subtraction build_mix does
    for arm, n in ARMS:
        mix, w_s, w_p = build(arm, n, base, tok, budget_rows)
        s = sum(v["weight"] for v in mix["domains"].values())
        p = os.path.join(ROOT, "data", f"mix_e1_{arm}.json")
        print(f"{arm}: w_S={w_s!r} w_P={w_p!r} sum={s!r}")
        if a.write:
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(mix, fh, ensure_ascii=False, indent=1)
            print(f"  wrote {os.path.relpath(p, ROOT)}")
    if not a.write:
        print("(nothing written; pass --write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Generate data/mix_30b_stage2.json. The file is derived, so it is regenerated, never hand-edited.

    python3 scripts/write_mix_stage2.py            # write
    python3 scripts/write_mix_stage2.py --check    # assert the committed file matches (CI / pre-merge)

Every number below is an input with a source. The assertions are the point: a hand-merge of the
JSON, or a weight typed to too few decimals, fails here instead of at launch.

Why a writer and not an edited file: this file was lost once inside another session's merge
resolution and its stage-2 line in launch_30b.sh was silently reverted twice. A derived artifact
that only exists as committed JSON cannot be checked against its inputs; this one can.
"""

import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "mix_30b_stage2.json")
ROWS = 3_662_109  # int(15e9 / 4096): stage 2 is 15B on top of stage 1's 14.9384B
SEQ = 4096

# name: (A-prime stage-2 rows, stage-1 rows drawn, stage-1 trainable pool rows, role)
# Rows are b0's A-prime table (b0 6375636). Stage-1 rows are from the RUN'S OWN LOG, not the
# table -- stage 1 under-drew cot when its epoch cap fired. Pools are measured directly from the
# token caches: rows of seq+1, minus n_val = min(int(rows*0.05), 5000).
# LANDED: domain -> (stamp tokens, fingerprint). A domain leaves _blocked only when it is stamped
# AND its supply covers its want. Epochs stay measured on the STAGE-1 pool until a token cache for
# the stage-2 corpus exists: the real pool is rows-of-seq+1 minus n_val, which only the cache gives.
LANDED = {
    "en_c4": (2_403_694_865, "05e0fc6f14704056"),
}

SPEC = {
    "code_rp1t": (1_074_090, 1_362_304, 1_842_469, "code raw (RedPajama-1T github); 30B cumulative third"),
    "math_owm": (1_173_995, 671_374, 979_824, "math/reasoning raw (OpenWebMath + finemath)"),
    "cot": (295_512, 295_512, 98_504, "chain-of-thought (NuminaMath-CoT); 6 pool-epochs, N=5.0B between exposures"),
    "en_c4": (571_451, 768_566, 1_168_802, "English general (C4)"),
    "zh_web": (401_178, 402_832, 5_192_316, "Chinese web (CCI3/wanjuan)"),
    "textbook_30b": (121_581, 122_058, 388_021, "textbook/instructional"),
    "wiki_chat": (24_302, 24_426, 65_831, "wiki + chat"),
}
# fb's 5-decimal weights, kept for audit. They do NOT hit the row targets -- see _weight_for_rows.
FB_5DP = {
    "code_rp1t": 0.29330, "math_owm": 0.32058, "cot": 0.08069, "en_c4": 0.15604,
    "zh_web": 0.10955, "textbook_30b": 0.03320, "wiki_chat": 0.00664,
}


def _weight_for_rows(rows):
    """Shortest decimal weight w where int(ROWS*w) == rows exactly.

    build_mix computes want = int(total_rows * weight), so the weight is the only executable
    denomination: tokens and rows are views of it. At 5 decimals cot draws 295,495 rather than
    295,512, and modulo indexing leaves 17 rows one exposure short -- which falsified the
    "cot is the uniform 6x case" the non-uniformity analysis was anchored on.
    """
    for places in range(5, 13):
        w = round(rows / ROWS, places)
        if int(ROWS * w) == rows:
            return w, places
    raise AssertionError(f"no weight up to 12dp yields {rows} rows")


def build():
    landed, blocked, total = {}, {}, 0
    for name, (s2, s1, pool, role) in SPEC.items():
        w, places = _weight_for_rows(s2)
        runtime = int(ROWS * w)
        assert runtime == s2, f"{name}: weight {w} draws {runtime}, want {s2}"
        epochs = math.ceil(runtime / pool)
        total += runtime
        entry = {
            "weight": w,
            "epochs": epochs,
            "anneal": w,
            "role": role,
            "weight_fb_5dp": FB_5DP[name],
            "weight_decimals": places,
            "rows_from_weight_at_runtime": runtime,
            "rows_table": s2,
            "stage1_rows": s1,
            "stage1_pool_rows": pool,
            "cumulative_rows": s1 + runtime,
            "cumulative_epochs_on_stage1_pool": round((s1 + runtime) / pool, 4),
            "epoch_cap_note": (
                f"epochs {epochs} = ceil({runtime}/{pool}) on the STAGE-1 pool. build_mix caps at "
                f"int(pool*epochs) in ROWS, so the cap is the integer ceiling and never the ratio "
                f"{runtime / pool:.3f}: a fractional cap re-creates stage 1's 61,593,088-token under-draw."
            ),
        }
        if name in LANDED:
            supply, fp = LANDED[name]
            want_tok = runtime * SEQ
            assert supply >= want_tok, (
                f"{name}: stamp supply {supply} < want {want_tok} -- do not land a domain that "
                "cannot cover its draw"
            )
            entry["supply_tokens"] = supply
            entry["fingerprint"] = fp
            entry["supply_margin_pct"] = round((supply / want_tok - 1) * 100, 2)
            entry["status"] = (
                f"LANDED: data/corpus/{name}_stage2 stamped, fp {fp}, supply {supply} tokens covers the "
                f"want of {want_tok} by {entry['supply_margin_pct']}%. epochs {epochs} is still measured "
                "on the STAGE-1 pool; re-derive it against the stage-2 pool once a token cache exists."
            )
            landed[f"{name}_stage2"] = entry
        else:
            entry["status"] = (
                f"BLOCKED: data/corpus/{name}_stage2 is not stamped (no build_corpus_stats.json). "
                "Move to domains only when it is stamped AND its supply covers its want."
            )
            blocked[f"{name}_stage2"] = entry
    assert total == ROWS, f"runtime rows {total} != {ROWS}"
    wsum = sum(b["weight"] for b in list(landed.values()) + list(blocked.values()))
    assert abs(wsum - 1.0) < 1e-3, f"weights sum {wsum}, outside the contract's 1e-3"
    assert len(landed) + len(blocked) == len(SPEC), "a domain went missing"
    return {
        "_comment": [
            "Stage 2 of the staged 15B->30B pretrain (t22), Case A-prime. fb ruling 2026-08-31.",
            "GENERATED by scripts/write_mix_stage2.py -- regenerate, never hand-edit. --check asserts",
            "the committed file still matches its inputs.",
            "",
            "THREE DENOMINATIONS, and only the weight is executable: tokens, rows, weights. build_mix",
            "computes want = int(total_rows*weight), so stating rows without the weight that produces",
            "them is the same gap as stating tokens without rows. fb's 5-decimal weights miss their row",
            "targets by -17..+14 rows (total -3); each weight here carries the decimals needed to hit",
            "its target exactly, 6dp for math_owm through 9dp for textbook_30b.",
            "",
            "EPOCHS ARE MEASURED ON THE STAGE-1 POOL, including for domains already in `domains`.",
            "Pool = rows of seq+1 in the token cache minus n_val = min(int(rows*0.05), 5000), so the",
            "stage-2 pool is unknowable until that corpus has a token cache -- a stamp gives tokens, not",
            "packed rows. A landed domain therefore carries a stage-1-pool epochs value that MUST be",
            "re-derived as ceil(rows/pool) against its own pool once the cache exists, and before stage 2",
            "launches. Landing on the stamp is a supply check, not an epoch check.",
            "",
            "Stage 1 drew 3,647,072 rows = 14.9384B, not the commissioned 15.000B, because cot's epoch",
            "cap fired on ROWS while its demand was written in TOKENS. So cot has seen 5.71 raw-supply",
            "epochs after this file's 3 more, not 6: 6.00 is on the pool, 5.71 on raw supply. cot epochs",
            "stay a live lever if a math or reasoning metric reads flat at the readout (fb, 44).",
            "",
            "KNOWN DEVIATION, pre-registered: used[] restarts at 0 on resume (train.py:1591), so stage 2",
            "re-reads rows stage 1 consumed while fresh rows sit unread -- zh_web re-reads 8% with 92%",
            "never seen, code_rp1t 58% with 26% never seen. EVERY epochs figure here is a mean over a",
            "non-uniform pass, cot included: its 17 rounding rows land at 5x against 98,487 at 6x under",
            "fb's 5dp weight, and at the exact weight the pass is uniform only if used[] is seeded.",
            "de-7 persists used[] and seeds it on resume; that check must compare against the RUN'S LOG",
            "(3,647,072 total and the per-domain startup lines), never against this table -- cot's two",
            "figures agree only because its cap happened to fire at exactly 3x pool.",
            "",
            "Cooldown constraint (arXiv 2408.10914, fb): the 10% warmdown holds >=20% code; A-prime",
            "carries 29.33%. Code stays at the 30B cumulative third: the paper's 25% optimum is for",
            "non-code and world knowledge, which is not this run's objective.",
        ],
        "total_tokens": ROWS * SEQ,
        "total_rows": ROWS,
        "seq": SEQ,
        "anneal_frac": 0.0,
        "warmdown": 0.10,
        "_runtime_rows_total": total,
        "_runtime_rows_note": (
            f"{total} rows = {total * SEQ / 1e9:.4f}B, exactly total_rows, because every weight carries "
            "enough decimals to hit its row target. No per-domain flooring loss remains."
        ),
        "domains": landed,
        "_blocked": blocked,
    }


if __name__ == "__main__":
    mix = build()
    if "--check" in sys.argv:
        with open(OUT, encoding="utf-8") as f:
            on_disk = json.load(f)
        if on_disk != mix:
            print(f"FAIL: {OUT} differs from what the writer generates -- regenerate it", file=sys.stderr)
            sys.exit(1)
        print(f"OK: {OUT} matches its inputs ({len(mix['_blocked'])} blocked, {mix['_runtime_rows_total']:,} rows)")
    else:
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(mix, f, indent=1, ensure_ascii=False)
        print(f"wrote {OUT}")
        for n, b in mix["_blocked"].items():
            print(f"  {n:20} w {b['weight']:<12} rows {b['rows_from_weight_at_runtime']:>9,} "
                  f"ep {b['epochs']} cum {b['cumulative_epochs_on_stage1_pool']}")

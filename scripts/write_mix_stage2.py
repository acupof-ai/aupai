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
    "cot": (424_056_227, "388496b76ed9bf88"),
    "code_rp1t": (7_569_415_401, "d8b9b18ba080f487"),
    "zh_web": (21_293_403_945, "a0d44fc44a289d60"),
    "textbook_30b": (1_610_210_330, "3f237c5191cb8571"),
    "wiki_chat": (283_903_257, "b864d32f9452a7c8"),
    # fb's message quoted fp a67cde07d3b3 (12 chars); the stamp itself reads a67cde07d3b3f63d and
    # _corpus_fp returns 16, so the message was truncated. Stamp value used.
    # re-stamped after the U+2028 strip settled: the first stamp was taken mid-rewrite,
    # with shards newer than the cache beside it (3b, fb).
    "math_owm": (6_513_304_690, "1e687e4b5ce37598"),
}
# Domains that keep their STAGE-1 name in the stage-2 mix, so the mix binds to the stage-1
# corpus directory rather than a *_stage2 copy. fb ruling 2026-08-31 for cot: the name binds a
# mix to a corpus and cot's corpus IS the stage-1 one, so a copy under a new name would falsify
# that binding rather than honour a convention. Mechanically required too -- _assert_mix_domains
# raises on a domain with no data/corpus/<name>/*.jsonl, _domain_cache_path derives
# tokens_<name>.pt from the name, and a symlink is the one case _assert_mix_domains refuses
# outright (a symlinked domain is a different corpus wearing another domain's name).
SAME_CORPUS_AS_STAGE1 = {"cot", "code_rp1t", "zh_web", "textbook_30b", "wiki_chat"}

# The stage-1 row cursor from ckpt_pretrain_15b_s1.pt.step16000, which seeds used[] in
# build_mix. THE CAP MUST COVER used + want, NOT want alone: build_mix computes
# cap = int(pool * epochs) - used[name] (train.py:1802) and then draws
# arange(used, used+want) (:1810). Deriving epochs from want/pool alone -- which is what the
# first version of this writer did -- gave cot 3 against a need of 6, so its cap left ~5K rows
# of a 295,512-row draw and stage 2 would have trained on essentially NO cot. It killed the
# first stage-2 launch at the JOIN line: total_steps 28,505 instead of 32,348.
#
# The cursor is keyed by the MIX's domain name (train.py:1774 looks up row_cursor[name]), so a
# renamed domain does not match and seeds used = 0. en_c4_stage2 and math_owm_stage2 are new
# dirs with new names, so they legitimately start at row 0 and their epochs 1 already suffices.
STAGE1_CURSOR = {
    "code_rp1t": 1_338_744,
    "cot": 290_401,
    "zh_web": 395_865,
    "textbook_30b": 119_947,
    "wiki_chat": 24_003,
    # en_c4 755,274 and math_owm 659,763 are in the checkpoint but keyed to the OLD names;
    # the stage-2 mix calls them en_c4_stage2 / math_owm_stage2, so they do not seed.
}
# POOLS: domain -> stage-2 trainable pool rows, MEASURED from its token cache, not from the stamp.
# A stamp gives tokens; a pool is rows of seq+1 minus n_val = min(int(rows*0.05), 5000), which only
# the cache yields. A domain here gets its epochs re-derived against its OWN pool; without an entry
# the epochs value is the stage-1-pool figure and is provisional.
STAGE2_POOLS = {
    "en_c4": 581_073,  # cache 2,401,144,188 tok = 586,073 rows of 4097, minus 5,000 val
    # cot reuses the stage-1 cache, so its stage-2 pool IS the stage-1 pool: 424,056,227 tok =
    # 103,504 rows of 4097; int(103,504*0.05) = 5,175 exceeds the 5,000 cap so n_val = 5,000.
    # Start from the CACHE row count, never from the pool -- computing 5% of 98,504 to derive
    # 98,504 is circular, and 3b and I both did it and got the right answer only by cancellation.
    "cot": 98_504,
    # The other reused dirs, pools measured from their own caches (identical to the stage-1
    # figures, as expected for the same corpus -- verified, not assumed).
    "code_rp1t": 1_842_469,   # 1,847,469 rows - 5,000
    "zh_web": 5_192_316,      # 5,197,316 rows - 5,000
    "textbook_30b": 388_021,  # 393,021 rows - 5,000
    # wiki_chat is the one domain where the 5% side of min() binds: 69,295 rows, 5% = 3,464
    # < the 5,000 cap, so n_val = 3,464. This is why the arithmetic must start from the cache
    # row count -- for a sub-100,000-row domain the cap does NOT bind.
    "wiki_chat": 65_831,      # 69,295 rows - 3,464
    # math_owm_stage2 is a NEW corpus, 6.53B against stage 1's 4.03B, so its pool is 62% larger
    # and its cap drops from the provisional 2 to 1: cache 6,528,628,253 tok = 1,593,514 rows of
    # 4097, minus 5,000 val. want/pool 0.7391, so it draws 0.74 epochs and repeats NOTHING --
    # the A-prime table's "math_owm repeats at 1.88 cumulative" was an artifact of measuring
    # against the stage-1 pool. Cumulative on its own pool is 1.16, not 1.88.
    "math_owm": 1_588_494,  # 1,593,494 rows of 4097 - 5,000 val; was 1,588,514 pre-restrip (-20)
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
        key = name if name in SAME_CORPUS_AS_STAGE1 else f"{name}_stage2"
        s2_pool = STAGE2_POOLS.get(name)
        cap_pool = s2_pool if s2_pool else pool
        key_for_cursor = name if name in SAME_CORPUS_AS_STAGE1 else f"{name}_stage2"
        used = STAGE1_CURSOR.get(key_for_cursor, 0)
        epochs = math.ceil((used + runtime) / cap_pool)
        assert cap_pool * epochs >= used + runtime, (
            f"{name}: pool {cap_pool} x epochs {epochs} = {cap_pool * epochs} < used {used} + "
            f"want {runtime} = {used + runtime} -- build_mix would clamp the draw and silently "
            "under-train this domain"
        )
        # A provisional cap on a *_stage2 domain is measured against the STAGE-1 corpus, which is a
        # different body of text. math_owm_stage2 is the live case: its stamp is 6.513B against
        # stage 1's 4.035B, so its real pool is ~62% larger and its cap is probably 1, not the 2 the
        # stage-1 pool implies. A too-high cap never truncates -- int(pool*epochs) only grows -- but
        # it does misstate the recipe, since epochs 2 reads as "this domain repeats" when it may not.
        provisional_new_corpus = s2_pool is None and name not in SAME_CORPUS_AS_STAGE1
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
            # Cumulative epochs only MEAN anything when both stages read the same corpus. For a
            # reused dir that is s1+s2 over the shared pool. For a NEW corpus the two draws come
            # from different bodies of text, so a single cumulative figure is a category error:
            # report each stage against its own pool instead.
            **({"cumulative_epochs_on_shared_pool": round((s1 + runtime) / pool, 4)}
               if name in SAME_CORPUS_AS_STAGE1 else
               {"stage1_epochs_on_stage1_pool": round(s1 / pool, 4),
                "stage2_epochs_on_stage2_pool": round(runtime / cap_pool, 4) if s2_pool else None,
                "cumulative_epochs_not_defined": (
                    "stage 1 and stage 2 read DIFFERENT corpora for this domain, so there is no "
                    "shared pool to count epochs against. The A-prime table's cumulative figure for "
                    "this domain was computed against the stage-1 pool and is an artifact.")}),
            "pool_rows": cap_pool,  # alias b0's readout draws_equal() reads
            "cursor_used_rows": used,
            "cap_covers": used + runtime,
            "epochs_pool_source": "stage-2 cache (measured)" if s2_pool else "stage-1 pool (PROVISIONAL, different corpus)",
            "stage2_pool_rows": s2_pool,
            "epoch_cap_note": (
                f"epochs {epochs} = ceil({runtime}/{cap_pool}) on the "
                f"{'STAGE-2 pool measured from its token cache' if s2_pool else 'STAGE-1 pool, PROVISIONAL until its cache exists'}. "
                f"build_mix caps at int(pool*epochs) in ROWS, so the cap is the integer ceiling and never "
                f"the ratio {runtime / cap_pool:.4f}: a fractional cap re-creates stage 1's "
                f"61,593,088-token under-draw. Headroom {(cap_pool - runtime) / runtime * 100:+.2f}%."
            ),
        }
        if name in LANDED:
            supply, fp = LANDED[name]
            want_tok = runtime * SEQ
            # supply is ONE epoch. A domain drawing >1 epoch covers its want by repeating, so the
            # test is supply*epochs, not supply. cot draws 3 epochs of a 0.424B corpus for 1.210B
            # tokens and a bare supply >= want would have refused a correct recipe.
            assert supply * epochs >= want_tok, (
                f"{name}: supply {supply} x {epochs} epochs = {supply * epochs} < want {want_tok} "
                "-- do not land a domain that cannot cover its draw"
            )
            entry["supply_tokens"] = supply
            entry["fingerprint"] = fp
            if provisional_new_corpus:
                est_rows = supply // (SEQ + 1)
                est_pool = est_rows - min(max(1, int(est_rows * 0.05)), 5000)
                entry["cap_provisional_warning"] = (
                    f"epochs {epochs} is derived from the STAGE-1 pool ({cap_pool} rows) because this "
                    f"corpus has no token cache yet, but it is a DIFFERENT corpus: the stamp's "
                    f"{supply} tokens imply ~{est_pool} pool rows and a cap of "
                    f"{math.ceil(runtime / est_pool)}. Safe (a high cap cannot truncate) but it "
                    f"overstates repetition -- re-derive from the cache before launch."
                )
                entry["cap_from_stamp_estimate"] = math.ceil(runtime / est_pool)
                entry["pool_rows_from_stamp_estimate"] = est_pool
            entry["supply_epochs"] = epochs
            entry["supply_x_epochs_tokens"] = supply * epochs
            entry["supply_margin_pct"] = round((supply * epochs / want_tok - 1) * 100, 2)
            entry["status"] = (
                f"LANDED: data/corpus/{key} stamped, fp {fp}, supply {supply} tokens covers the "
                f"want of {want_tok} at {epochs} epoch(s) by {entry['supply_margin_pct']}%. epochs {epochs} is still measured "
                "on the STAGE-1 pool; re-derive it against the stage-2 pool once a token cache exists."
            )
            landed[key] = entry
        else:
            entry["status"] = (
                f"BLOCKED: data/corpus/{key} is not stamped (no build_corpus_stats.json). "
                "Move to domains only when it is stamped AND its supply covers its want."
            )
            blocked[key] = entry
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
            "CORPUS REUSE, five of seven (fb ruling 2026-08-31, 44 confirmed). cot, code_rp1t, zh_web,",
            "textbook_30b and wiki_chat keep their BARE stage-1 names and reuse those directories and",
            "caches; only en_c4_stage2 and math_owm_stage2 are new builds. The name binds a mix to a",
            "corpus, so a copy under a *_stage2 name would falsify that binding rather than honour a",
            "convention -- and mechanically, _assert_mix_domains raises on a domain with no",
            "data/corpus/<name>/*.jsonl, _domain_cache_path derives tokens_<name>.pt from the name, and a",
            "symlink is the one case _assert_mix_domains refuses outright.",
            "",
            "CROSS-STAGE JUDGEABILITY (44, prereg 7.1/7.2). code_rp1t is unjudgeable by 7.1 alone, a",
            "weight-only change caught by the ratio guard. en_c4 and math_owm carry 7.1 plus 7.2's",
            "rebuilt-corpus verification. The four weight-stable reused domains -- cot, zh_web,",
            "textbook_30b, wiki_chat -- keep identical .srcfp and ARE judgeable cross-stage.",
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

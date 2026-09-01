#!/usr/bin/env python3
# restartable: pure function of its constants plus --code-tokens; writes one small JSON in
# one pass and runs in well under a second, so an interrupt costs a rerun and nothing else.
"""Generate data/mix_500m.json for the 493.6M run. Derived: regenerate, never hand-edit.

    python3 scripts/write_mix_500m.py --code-tokens 3.8e9      # write, code supply measured
    python3 scripts/write_mix_500m.py --check                  # assert the committed file matches
    python3 scripts/write_mix_500m.py --selftest

WHY THIS FILE EXISTS SEPARATELY FROM write_mix_stage2.py. That writer encodes a 200M objective
whose composition was set by data SUPPLY -- code got 29.33% because 7.57B tokens of RedPajama
github existed, not because the target asked for it. The user named that as the root cause. This
writer inverts the dependency: weights come from the capability target, and supply appears only
as a feasibility constraint that can REFUSE a weight, never as the thing that sets it.

THE CODE WEIGHT IS A FUNCTION, NOT A CONSTANT. code_rp1t turned out to be unlabelled
multi-language soup: of 3,747,157 rows only 209,668 parse as Python 3, so a 7.57B stamp is 0.42B
of Python. 3b is fetching starcoderdata's Python split and the surviving count is not known yet.
Passing --code-tokens re-derives the mix against whatever actually lands, so nobody has to revise
a number by hand. Below CODE_FLOOR the writer REFUSES rather than quietly shipping a code-light
run under a code-first objective.

BUDGET IS DECIDED HERE, NOT INHERITED. 30B was set for a 200M model. See the rationale doc; the
short version is that the binding constraint is not compute, it is that only ~15B tokens of
non-zh_web material exist at one epoch, and zh_web is capped at 3% by the objective so it cannot
backfill. TOTAL is 20B and the reasoning is in docs/lessons/mix_500m_rationale.md.

EPOCHS ARE CUMULATIVE AND COMPUTED, NEVER TYPED. cap = ceil((used + want) / pool). The used[]
term is the trap that killed a stage-2 launch once: build_mix computes cap = int(pool*epochs) -
used[name] and then draws arange(used, used+want), so an epochs derived from want/pool alone
leaves the draw short. Every domain here starts at used=0 because this is a fresh run with new
names, and that is asserted rather than assumed.
"""

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "mix_500m.json")

SEQ = 4096
TOTAL_TOKENS = 20_000_000_000        # decided; see docs/lessons/mix_500m_rationale.md
ROWS = TOTAL_TOKENS // SEQ           # 4,882,812
CODE_FLOOR = 2.5e9                   # below this the code-first objective is not fundable

# The objective, as weights. Each carries the reason it is that number rather than another.
# ORDER MATTERS: code first, because the target is a code model and every other weight is set
# against what code leaves. This is the inversion -- supply does not appear in this dict.
OBJECTIVE = {
    "code_py_starcoder": (0.34, "code, Python only (ast.parse is both language ID and syntax "
                                "filter). Phi-1: one language, 6B curated Python, HumanEval 50.6 "
                                "at 1.3B. The largest single weight because the objective is code."),
    "math_owm_stage2":   (0.26, "math/reasoning. Second-largest: the 16B readout put math and "
                                "reasoning on the steepest part of their curves, and code and math "
                                "share the symbolic-structure transfer this size can still absorb."),
    "en_c4_stage2":      (0.16, "English general. Carries the natural-language competence code "
                                "docstrings and problem statements are written in; below ~15% the "
                                "prose side of a code model degrades before the code side does."),
    "cot":               (0.08, "chain-of-thought. Supply is 0.424B, the smallest of the seven, so "
                                "this weight is an epoch decision as much as a composition one: at "
                                "9% it draws 4.24 epochs and crosses the 4-epoch line, at 8% it "
                                "draws 3.77 and stays under. Set to 8% for that reason -- past 4 "
                                "epochs the weight buys re-reads rather than information, and the "
                                "16B readout already measured cot at 0.1180 nat/B own-token, the "
                                "second-lowest of seven. Paying for re-reads of the flattest domain "
                                "is the worst available use of the budget."),
    "textbook_30b":      (0.10, "textbook/instructional. The Phi-1 'textbook quality' arm; the "
                                "16B readout measured it at 0.5024 nat/B own-token, second-steepest "
                                "of the seven and 30x cot's rate. Took cot's released point for "
                                "exactly that reason: same cost, 4.3x the measured rate."),
    "chatml":            (0.03, "ChatML-rendered chat. The corpus contains effectively none "
                                "(<0.075% per domain) and the 200M round showed SFT installs only "
                                "the canon it is fed -- relying on SFT to teach a prefix the base "
                                "has never seen is a bet this project already lost once."),
    "zh_web":            (0.03, "Chinese web, capped by the objective. Was 10.955%; the target is "
                                "an English-language code model and zh_web's 0.1657 nat/B was the "
                                "third-lowest of seven. 21.3B of supply is now irrelevant -- this "
                                "is the clearest single case of composition following the target "
                                "rather than the warehouse."),
}

# One-epoch supply, tokens. Measured stamps for the landed domains; code is a parameter.
# chatml is a RE-RENDER of wiki_chat's chat rows, not new data, so its supply is bounded by
# the source it is rendered from -- recorded as such rather than as an independent corpus.
SUPPLY = {
    "math_owm_stage2": 6_513_304_690,   # stamped, fp 1e687e4b5ce37598
    "en_c4_stage2":    2_403_694_865,   # stamped, fp 05e0fc6f14704056
    "cot":               424_056_227,   # stamped, fp 388496b76ed9bf88
    "textbook_30b":    1_610_210_330,   # stamped, fp 3f237c5191cb8571
    "zh_web":         21_293_403_945,   # stamped, fp a0d44fc44a289d60
    "chatml":            283_903_257,   # bounded by wiki_chat's stamp (b864d32f9452a7c8)
}

# Directories named by any data/mix_scale_*.json. Writing new corpus into one of these
# falsifies the ladder's fingerprint; an A/B was correctly stopped at startup over exactly
# this two days ago. The writer refuses a domain whose name collides.
LADDER_DIRS = {"chat", "code", "en", "math", "textbook", "web_hq", "wiki"}

# Muennighoff's repeat-decay constant: R_D* = 15.4, and <=4 epochs costs ~nothing
# (ds.muennighoff_four_epoch: 8.7B at 4 epochs finishes +0.5% val vs single-epoch).
EPOCH_SOFT_CEILING = 4


def _weight_for_rows(rows, total_rows):
    """Shortest decimal weight w with int(total_rows*w) == rows exactly.

    build_mix computes want = int(total_rows * weight), so the weight is the only executable
    denomination -- tokens and rows are views of it. Stating rows without the weight that
    produces them is how stage 1 under-drew cot by 61,593,088 tokens.
    """
    for places in range(5, 13):
        w = round(rows / total_rows, places)
        if int(total_rows * w) == rows:
            return w, places
    raise AssertionError(f"no weight up to 12dp yields {rows} rows")


def build(code_tokens):
    if code_tokens < CODE_FLOOR:
        raise SystemExit(
            f"REFUSING: code supply {code_tokens / 1e9:.2f}B is below the {CODE_FLOOR / 1e9:.1f}B "
            f"floor. At weight {OBJECTIVE['code_py_starcoder'][0]:.0%} of {TOTAL_TOKENS / 1e9:.0f}B "
            f"the code domain wants {OBJECTIVE['code_py_starcoder'][0] * TOTAL_TOKENS / 1e9:.2f}B, "
            f"which is {OBJECTIVE['code_py_starcoder'][0] * TOTAL_TOKENS / code_tokens:.1f} epochs. "
            "A code-first objective funded by re-reading the same code four times over is a "
            "different experiment; lower TOTAL or raise supply, do not lower the code weight."
        )
    supply = dict(SUPPLY, code_py_starcoder=code_tokens)
    assert abs(sum(w for w, _ in OBJECTIVE.values()) - 1.0) < 1e-9, "weights must sum to 1"

    domains, warnings = {}, []
    total_rows_used = 0
    for name, (w_target, why) in OBJECTIVE.items():
        assert name not in LADDER_DIRS, (
            f"{name} collides with a directory named by data/mix_scale_*.json; a new corpus there "
            "falsifies the ladder's fingerprint"
        )
        rows = int(ROWS * w_target)
        w, places = _weight_for_rows(rows, ROWS)
        runtime = int(ROWS * w)
        assert runtime == rows, f"{name}: weight {w} draws {runtime}, want {rows}"

        want_tok = runtime * SEQ
        pool_tok = supply[name]
        # Pool in ROWS is what build_mix caps against, and it is rows-of-(seq+1) minus n_val,
        # which only a token cache yields. No cache exists for these domains yet, so the pool is
        # ESTIMATED from the stamp and every epochs value here is provisional by construction.
        # Recorded as such: a provisional cap that reads as measured is the defect this repo
        # keeps finding, and the re-derivation is a launch precondition, not a nicety.
        pool_rows_est = pool_tok // (SEQ + 1)
        n_val = min(int(pool_rows_est * 0.05), 5000)
        pool_rows_est -= n_val
        used = 0  # fresh run, new names; asserted rather than assumed
        epochs = math.ceil((used + runtime) / pool_rows_est)
        assert pool_rows_est * epochs >= used + runtime, (
            f"{name}: pool {pool_rows_est} x epochs {epochs} < used {used} + want {runtime}; "
            "build_mix would clamp the draw and silently under-train this domain"
        )
        if epochs > EPOCH_SOFT_CEILING:
            warnings.append(
                f"{name}: {runtime * SEQ / pool_tok:.2f} epochs exceeds the {EPOCH_SOFT_CEILING}-epoch "
                f"line (ds.muennighoff_four_epoch). Past 4 epochs repeated tokens stop paying and "
                f"this weight is buying re-reads, not information."
            )
        total_rows_used += runtime
        domains[name] = {
            "weight": w,
            "epochs": epochs,
            "anneal": w,
            "role": why,
            "weight_decimals": places,
            "rows_from_weight_at_runtime": runtime,
            "want_tokens": want_tok,
            "supply_tokens_one_epoch": pool_tok,
            "epochs_fractional": round(want_tok / pool_tok, 4),
            "pool_rows_estimated": pool_rows_est,
            "cursor_used_rows": used,
            "cap_covers": used + runtime,
            "epochs_pool_source": "ESTIMATED from the stamp; no token cache exists yet",
            "epoch_cap_note": (
                f"epochs {epochs} = ceil(({used}+{runtime})/{pool_rows_est}). PROVISIONAL: the pool "
                f"is estimated as stamp_tokens//(seq+1) minus n_val, not measured from a token "
                f"cache. Re-derive against the real cache before launch -- a stamp gives tokens, "
                f"a pool is packed rows."
            ),
        }

    assert total_rows_used <= ROWS, f"{total_rows_used} rows drawn exceeds the {ROWS} budget"
    return {
        "_comment": [
            "The 500M run's mix. GENERATED by scripts/write_mix_500m.py -- regenerate, never",
            "hand-edit. --check asserts the committed file still matches its inputs.",
            "",
            "COMPOSITION FOLLOWS THE OBJECTIVE, NOT THE SUPPLY. This is the whole point of the",
            "file and the thing the user asked for by name. mix_30b_stage2.json gave code 29.33%",
            "because 7.57B tokens of RedPajama github existed; that stamp turned out to be 0.42B",
            "of Python. Here the weights come from the capability target and supply appears only",
            "as a constraint that can REFUSE a weight -- see CODE_FLOOR.",
            "",
            "zh_web 10.955% -> 3.00% is the clearest single instance: 21.3B tokens of supply is",
            "now irrelevant because the target is an English-language code model.",
            "",
            "EVERY EPOCHS VALUE IS PROVISIONAL. Pools are estimated from stamps because no token",
            "cache exists for these domains yet. Re-derive each against its own cache before",
            "launch; a stamp gives tokens, a pool is packed rows of seq+1 minus n_val.",
            "",
            "TOTAL is 20B, decided rather than inherited. docs/lessons/mix_500m_rationale.md has",
            "the derivation; the binding constraint is supply at one epoch, not compute.",
        ],
        "total_tokens": ROWS * SEQ,
        "total_rows": ROWS,
        "seq": SEQ,
        "anneal_frac": 0.0,
        "warmdown": 0.1,
        "_budget_rationale": "docs/lessons/mix_500m_rationale.md",
        "_code_supply_tokens_at_generation": code_tokens,
        "_warnings": warnings,
        "domains": domains,
    }


def selftest():
    # 1. the code weight is a function of supply, not a constant
    a = build(3.0e9)["domains"]["code_py_starcoder"]
    b = build(6.0e9)["domains"]["code_py_starcoder"]
    assert a["weight"] == b["weight"], "the WEIGHT is the objective and must not move with supply"
    assert a["epochs"] > b["epochs"], (
        f"epochs must fall as supply rises: {a['epochs']} at 3B vs {b['epochs']} at 6B"
    )
    print(f"  1 supply moves epochs ({a['epochs']} at 3B -> {b['epochs']} at 6B), never the weight")

    # 2. the floor refuses rather than silently shipping a code-light code-first run
    try:
        build(1.0e9)
        raise AssertionError("build accepted a code supply below the floor")
    except SystemExit as e:
        assert "REFUSING" in str(e) and "different experiment" in str(e), str(e)
    print("  2 below CODE_FLOOR the writer refuses, and the message says why lowering the weight "
          "is the wrong fix")

    # 3. every weight hits its row target exactly. At 5dp the stage-1 cot draw was 17 rows short
    #    and that falsified the uniformity analysis the whole epoch account rested on.
    m = build(3.8e9)
    for name, d in m["domains"].items():
        assert int(ROWS * d["weight"]) == d["rows_from_weight_at_runtime"], name
    print(f"  3 all {len(m['domains'])} weights hit their row targets exactly (5-12 dp)")

    # 4. caps cover used+want, the shape that killed a stage-2 launch
    for name, d in m["domains"].items():
        assert d["pool_rows_estimated"] * d["epochs"] >= d["cap_covers"], name
    print("  4 every cap covers used+want, not want alone")

    # 5. a domain colliding with a ladder directory is refused
    saved = OBJECTIVE.pop("zh_web")
    OBJECTIVE["web_hq"] = saved
    try:
        build(3.8e9)
        raise AssertionError("build accepted a ladder-fingerprinted directory name")
    except AssertionError as e:
        assert "falsifies the ladder" in str(e), str(e)
    finally:
        OBJECTIVE.pop("web_hq")
        OBJECTIVE["zh_web"] = saved
    print("  5 a domain named after a mix_scale_* directory is refused")

    # 6. the 4-epoch line is enforced at the shipped configuration, not merely reported. cot was
    #    the live case: 0.424B of supply at 9% of 20B is 4.24 epochs, so the weight was cut to 8%
    #    (3.77) and the released point went to textbook, which the 16B readout measured at 4.3x
    #    cot's nat/B. Both directions are asserted -- that the shipped config is clean, AND that
    #    the warning still fires on the configuration that produced the cut. A ceiling that has
    #    never fired is indistinguishable from one that cannot.
    assert not build(3.8e9)["_warnings"], (
        f"shipped config crosses the 4-epoch line: {build(3.8e9)['_warnings']}"
    )
    saved = OBJECTIVE["cot"]
    OBJECTIVE["cot"] = (0.09, saved[1])
    OBJECTIVE["textbook_30b"] = (0.09, OBJECTIVE["textbook_30b"][1])
    try:
        warns = build(3.8e9)["_warnings"]
        assert any("cot" in w for w in warns), f"cot at 9% must warn, got {warns}"
    finally:
        OBJECTIVE["cot"] = saved
        OBJECTIVE["textbook_30b"] = (0.10, OBJECTIVE["textbook_30b"][1])
    print("  6 shipped config is under the 4-epoch line, and the ceiling still fires on the "
          "9% cot config that produced the cut")

    print("selftest: 6/6")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--code-tokens", type=float, default=3.8e9,
                    help="parse-verified Python tokens that actually landed; the code weight is "
                         "a function of this, so pass the measured number")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    m = build(a.code_tokens)
    text = json.dumps(m, indent=1, ensure_ascii=False) + "\n"
    if a.check:
        if not os.path.exists(OUT):
            print(f"{OUT} does not exist", file=sys.stderr)
            return 1
        if open(OUT, encoding="utf-8").read() != text:
            print(f"REFUSING: {OUT} does not match its inputs -- regenerate", file=sys.stderr)
            return 1
        print(f"{OUT} matches its inputs")
        return 0
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {OUT}: {len(m['domains'])} domains, {m['total_tokens'] / 1e9:.1f}B tokens")
    for w in m["_warnings"]:
        print(f"  WARNING {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

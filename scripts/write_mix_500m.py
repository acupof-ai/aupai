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
of Python. Passing --code-tokens re-derives the mix against whatever actually lands, so nobody
has to revise a number by hand. Below CODE_FLOOR the writer REFUSES rather than quietly shipping
a code-light run under a code-first objective.

The default 8.85e9 is a PROJECTION, not a measurement: 3b measured 151.5M and 147.0M
parse-verified tokens on shards 0 and 1 and extrapolated x59. That is a 3.4% sample. I read a
0.08%-sampled near-dup rate as a corpus fact earlier today and it was the sampling rate, so
this default is labelled and must be replaced with the landed total when 3b reports it.

BUDGET IS DECIDED HERE, NOT INHERITED, AND ITS FIRST ARGUMENT WAS WRONG. 30B was set for a
200M model. The original case for 20B was that only ~15B of non-zh_web material existed at one
epoch -- that died when starcoder measured 8.85B instead of the assumed 3.80B, putting non-zh
supply at 20.51B. 20B survived the re-derivation on weaker grounds: it is 2.03x
compute-optimal where 30B is 3.04x, for 0.0575 nat, on a run that changed shape, optimizer
scaling and corpus at once. Note what does NOT justify it: '20B is the largest budget where
no domain crosses 4 epochs' is circular, because the only domain that crosses is cot and
cot's weight was itself set by that ceiling. docs/lessons/mix_500m_rationale.md 2.2 has the
full re-derivation and says plainly that the argument is weaker than it was.

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
#
# CODE IS ONE OBJECTIVE SPLIT ACROSS TWO CORPORA. The target says "34% Python", not "34%
# starcoder"; the split between the two sources is a supply fact and belongs in the split
# function, not here. Keeping them separate domains rather than one is deliberate: their
# provenance and filtering differ (starcoder is already deduplicated and quality-filtered,
# rp1t_python is our own ast.parse pass over raw github), so merging them would merge two
# fingerprints into one and lose the ability to attribute a per-role reading to either.
CODE_TOTAL = 0.34
CODE_WHY = ("code, Python only (ast.parse is both language ID and syntax filter). Phi-1: one "
            "language, 6B curated Python, HumanEval 50.6 at 1.3B. The largest single objective "
            "because the target is code.")
OBJECTIVE = {
    "math_owm_stage2":   (0.26, "math/reasoning. Second-largest: the 16B readout put math and "
                                "reasoning on the steepest part of their curves, and code and math "
                                "share the symbolic-structure transfer this size can still absorb."),
    "en_c4_stage2":      (0.16, "English general. Carries the natural-language competence code "
                                "docstrings and problem statements are written in; below ~15% the "
                                "prose side of a code model degrades before the code side does."),
    "cot":               (0.08, "chain-of-thought. 8% is a CEILING, not a target: cot's supply is "
                                "0.424B, the smallest of the eight, so at 9% of 20B it draws 4.24 "
                                "epochs and crosses Muennighoff's 4-epoch line while 8% draws 3.77. "
                                "Held at the largest value that stays under. WHAT THIS WEIGHT IS "
                                "NOT: it is not derived from cot's value to the objective, because "
                                "nothing measures that. The 16B readout's 0.1180 nat/B is a "
                                "within-role rate and docs/standards/training_loop.md section 6 -- "
                                "which I wrote -- forbids using nat/B for cross-role allocation, "
                                "since transfer inflates every such rate by an unmeasured and "
                                "role-dependent amount. An earlier draft of this file justified "
                                "moving a point from cot to textbook as 'same cost, 4.3x the "
                                "measured rate'. That is exactly the forbidden comparison and it is "
                                "withdrawn."),
    "textbook_30b":      (0.10, "textbook/instructional. The Phi-1 'textbook quality' arm: curated "
                                "instructional prose is the ingredient that paper credits, and this "
                                "is the closest domain we have to it. 10% rather than 9% because "
                                "cot's ceiling released a point and textbook is the domain the "
                                "OBJECTIVE ranks next -- not because its measured nat/B is higher, "
                                "which would be the cross-role comparison section 6 forbids."),
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

# Parse-verified Python recovered from the code_rp1t stamp: 209,668 of 3,747,157 rows survive
# ast.parse, for 420,646,182 tokens. We already paid a full-corpus pass to establish this, and
# it is the same kind of data as the starcoder fetch -- dropping it would forfeit 10% of the
# code domain's one-epoch supply for nothing (fb, 2026-09-01).
RP1T_PYTHON_TOKENS = 420_646_182


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


def _rows_for_weights(weights, total_rows):
    """Allocate total_rows across weights by LARGEST REMAINDER, so the parts sum to the whole.

    int(total_rows*w) per domain and hope is what the first version did, and the sum came to
    4,882,809 of a 4,882,812 budget -- three rows, 12,288 tokens, quietly not trained on. That
    is the same shape as stage 1's 61,593,088-token under-draw: an executable quantity missing
    its stated target because every part floored independently and nothing checked the sum.
    Largest-remainder gives the leftover rows to the domains that lost the most to flooring.
    """
    floors = {n: int(total_rows * w) for n, w in weights.items()}
    short = total_rows - sum(floors.values())
    order = sorted(weights, key=lambda n: total_rows * weights[n] - floors[n], reverse=True)
    for n in order[:short]:
        floors[n] += 1
    assert sum(floors.values()) == total_rows, floors
    return floors


def _code_split(starcoder_tokens, code_rows):
    """Split code_rows between the two Python corpora IN PROPORTION TO SUPPLY.

    Proportional is the only split that gives both corpora the same epoch count, which is what
    we want: neither source is known to be better per-token, so there is no reason to re-read
    one more than the other. Any other ratio is an unmeasured quality claim.

    Splitting in ROWS with the second corpus taking the remainder, rather than splitting a
    weight and letting each half floor, for the reason in _rows_for_weights.
    """
    total = starcoder_tokens + RP1T_PYTHON_TOKENS
    sc_rows = round(code_rows * starcoder_tokens / total)
    return {
        "code_py_starcoder": (sc_rows, starcoder_tokens),
        "code_py_rp1t": (code_rows - sc_rows, RP1T_PYTHON_TOKENS),
    }


def build(code_tokens):
    """code_tokens is the STARCODER supply; rp1t's parse-verified Python is added to it."""
    code_supply = code_tokens + RP1T_PYTHON_TOKENS
    if code_supply < CODE_FLOOR:
        raise SystemExit(
            f"REFUSING: code supply {code_supply / 1e9:.2f}B (starcoder {code_tokens / 1e9:.2f}B "
            f"+ rp1t python {RP1T_PYTHON_TOKENS / 1e9:.2f}B) is below the {CODE_FLOOR / 1e9:.1f}B "
            f"floor. At {CODE_TOTAL:.0%} of {TOTAL_TOKENS / 1e9:.0f}B the code objective wants "
            f"{CODE_TOTAL * TOTAL_TOKENS / 1e9:.2f}B, which is "
            f"{CODE_TOTAL * TOTAL_TOKENS / code_supply:.1f} epochs. A code-first objective funded "
            "by re-reading the same code four times over is a different experiment; lower TOTAL "
            "or raise supply, do not lower the code weight."
        )
    # Rows come from a single largest-remainder allocation over ALL domains, code counted as one
    # objective, so the parts sum to the budget exactly. Then the code objective's rows are split
    # between its two corpora. Two levels, each summing exactly, rather than seven independent
    # floors.
    # Derived here, not cached at module level: the ladder-dir selftest mutates OBJECTIVE,
    # and a module-level snapshot would not follow it -- the refusal then dies on a
    # KeyError instead of its own assertion, which is a guard failing for the wrong reason.
    alloc = dict({n: w for n, (w, _) in OBJECTIVE.items()}, code=CODE_TOTAL)
    rows_by_name = _rows_for_weights(alloc, ROWS)
    code = _code_split(code_tokens, rows_by_name.pop("code"))
    spec = {n: (rows_by_name[n], why) for n, (_, why) in OBJECTIVE.items()}
    supply = dict(SUPPLY)
    code_epochs = CODE_TOTAL * TOTAL_TOKENS / code_supply
    for name, (rows, sup) in code.items():
        spec[name] = (rows, CODE_WHY + f" Split across two corpora in proportion to supply, so "
                                       f"both draw the same {code_epochs:.2f} epochs; the split "
                                       f"is a supply fact, the {CODE_TOTAL:.0%} is the objective.")
        supply[name] = sup
    assert sum(r for r, _ in spec.values()) == ROWS, (
        f"rows sum to {sum(r for r, _ in spec.values())}, not the {ROWS} budget"
    )

    domains, warnings = {}, []
    total_rows_used = 0
    for name, (rows, why) in spec.items():
        assert name not in LADDER_DIRS, (
            f"{name} collides with a directory named by data/mix_scale_*.json; a new corpus there "
            "falsifies the ladder's fingerprint"
        )
        rows = int(rows)
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
    # 1. the OBJECTIVE is fixed and only its internal split tracks supply. The first version of
    #    this check asserted the per-domain code weight never moves, which was right when code
    #    was one domain and became wrong the moment it became two -- the invariant is the SUM.
    #    Asserted in ROWS, not weights: rows are what build_mix draws, and each weight carries
    #    only enough decimals to hit its own row target, so the weights sum to 0.34 +- 2e-8 by
    #    construction. Testing the encoding instead of the quantity would fail on rounding that
    #    cannot move a single row.
    code_rows_target = _rows_for_weights(dict({n: w for n, (w, _) in OBJECTIVE.items()}, code=CODE_TOTAL), ROWS)["code"]
    for cs in (3.0e9, 6.0e9):
        m = build(cs)["domains"]
        got = (m["code_py_starcoder"]["rows_from_weight_at_runtime"]
               + m["code_py_rp1t"]["rows_from_weight_at_runtime"])
        assert got == code_rows_target, (
            f"code objective drifted to {got} rows against {code_rows_target} at supply {cs}"
        )
    a3, b3 = build(3.0e9)["domains"], build(6.0e9)["domains"]
    assert a3["code_py_starcoder"]["weight"] < b3["code_py_starcoder"]["weight"], (
        "starcoder's share of the code objective must rise with its supply"
    )
    assert a3["code_py_starcoder"]["epochs"] > b3["code_py_starcoder"]["epochs"], (
        f"epochs must fall as supply rises: {a3['code_py_starcoder']['epochs']} at 3B vs "
        f"{b3['code_py_starcoder']['epochs']} at 6B"
    )
    # both corpora draw the SAME epochs -- that is what proportional-to-supply buys, and it is
    # the property that makes the split a supply fact rather than a quality claim.
    for cs in (3.0e9, 3.8e9, 6.0e9):
        m = build(cs)["domains"]
        e1, e2 = m["code_py_starcoder"]["epochs_fractional"], m["code_py_rp1t"]["epochs_fractional"]
        assert abs(e1 - e2) < 0.01, f"code corpora draw different epochs at {cs}: {e1} vs {e2}"
    print(f"  1 code objective pinned at {code_rows_target} rows across supplies; only the split "
          f"moves, and both corpora draw equal epochs")

    # 1b. the whole mix sums to the budget. Seven independent floors lost 3 rows (12,288 tokens)
    #     before largest-remainder allocation replaced them, and nothing was checking the sum.
    for cs in (3.0e9, 3.8e9, 6.0e9):
        m = build(cs)
        got = sum(d["rows_from_weight_at_runtime"] for d in m["domains"].values())
        assert got == ROWS, f"rows sum to {got}, not the {ROWS} budget, at supply {cs}"
    print(f"  1b all domains sum to exactly {ROWS} rows (largest-remainder, no floor leakage)")

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
    ap.add_argument("--code-tokens", type=float, default=8.85e9,
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

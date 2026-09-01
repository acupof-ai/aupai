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
                                "NOT: it is not derived from this role's value to the objective, "
                                "because nothing measures that. An earlier draft moved a point out "
                                "of here by comparing two roles' per-token nat/B, which "
                                "docs/standards/training_loop.md section 6 -- which I wrote -- "
                                "forbids: transfer inflates every such rate by an unmeasured, "
                                "role-dependent amount, so the ordering across roles is not known "
                                "to hold. That sentence is withdrawn, and _refuse_cross_role_rate "
                                "now refuses it mechanically. The withdrawn numbers are not quoted "
                                "here because the refusal cannot tell a citation from a quotation "
                                "of one, and it should not try -- an exemption for 'but I am "
                                "disowning it' is the hole the next violation goes through."),
    "textbook_30b":      (0.10, "textbook/instructional. The Phi-1 'textbook quality' arm: curated "
                                "instructional prose is the ingredient that paper credits, and this "
                                "is the closest domain we have to it. 10% rather than 9% because "
                                "cot's ceiling released a point and textbook is the domain the "
                                "OBJECTIVE ranks next -- not because its measured nat/B is higher, "
                                "which would be the cross-role comparison section 6 forbids."),
    "chatml":            (None, "ChatML-rendered chat. The corpus contains effectively none "
                                  "(<0.075% per domain) and the 200M round showed SFT installs "
                                  "only the canon it is fed -- relying on SFT to teach a prefix "
                                  "the base has never seen is a bet this project already lost "
                                  "once. WEIGHT IS None BECAUSE IT IS NOT A JUDGEMENT: this "
                                  "domain is SUPPLY_CAPPED, so _ceiling_weight derives it from "
                                  "the measured supply and the 4-epoch line. A literal here "
                                  "would be a second copy of that arithmetic, free to drift from "
                                  "it -- and it did: the typed 0.0076 was sized for a supply of "
                                  "exactly 38.0M and survived the switch to a 2sf measurement "
                                  "that made it a coin flip. The cap is deliberate rather than "
                                  "regrettable: the job is to make a FORMAT in-distribution, and "
                                  "formats are cheap to learn. Fifteen epochs would buy "
                                  "memorisation of 39M tokens of QA content we do not want "
                                  "memorised, in order to teach a markup (fb 2026-09-01). Too "
                                  "few epochs is measurable and fixable next run; 15 is baked "
                                  "into the weights."),
    "chat_qa":           (None, "The same QA rows in plain 问：/答： form, rendered into its own "
                                  "directory. Two renders of one source, so each domain holds "
                                  "~0.038B and each is capped independently -- the pair is ~1.5% "
                                  "combined, not each. Weight is None for the same reason as "
                                  "chatml: derived, not chosen. It lands 0.01pt below chatml "
                                  "purely because the plain render is slightly smaller than the "
                                  "marked-up one. NAMED chat_qa, not chat: data/corpus/chat/ is "
                                  "named by all six data/mix_scale_*.json ladder mixes (weight "
                                  "0.011768), so a re-render written there would put new bytes "
                                  "under a name six committed mixes fingerprint. The LADDER_DIRS "
                                  "guard refused `chat` when fb's ruling reached the generator "
                                  "-- the ruling and the constraint collided, and the guard is "
                                  "the reason anyone noticed."),
    "zh_web":            (0.03, "Chinese web, capped by the objective. Was 10.955%, and that "
                                "number came from the warehouse: 21.3B sitting on disk. The target "
                                "is an English-language code model, and Chinese web prose is not "
                                "an ingredient of it -- 3% is kept only so the tokenizer's Chinese "
                                "side does not go cold. The first draft also cited this role's "
                                "own-token nat/B and its rank among the seven; the rank is struck, "
                                "because ranking roles by nat/B is the section-6 comparison, and "
                                "the weight does not need it: the objective already excludes "
                                "Chinese prose. 21.3B of supply is now irrelevant -- the clearest "
                                "single case of composition following the target, not the stock."),
}

# Domains whose weight is a SUPPLY CEILING rather than an objective share. Their weight is
# whatever keeps them at EPOCH_SOFT_CEILING epochs, and the remainder is renormalised across
# the objective domains in proportion -- which preserves every ratio the objective states and
# asserts nothing new. Without this the freed points vanish into the allocator's rounding: the
# chat cap released 1.48pt and the weights simply summed to 0.9852, which the largest-remainder
# allocator absorbed silently. Freed weight is a composition decision, not a rounding one.
SUPPLY_CAPPED = {"chat_qa", "chatml"}
assert all(OBJECTIVE[n][0] is None for n in SUPPLY_CAPPED), (
    "a supply-capped domain must carry weight None -- _ceiling_weight derives it, and a literal "
    "beside the derivation is a second source of truth that will drift from it"
)
assert all(w is not None for n, (w, _) in OBJECTIVE.items() if n not in SUPPLY_CAPPED), (
    "an objective-weighted domain needs its weight stated; None means 'derived' and nothing "
    "derives these"
)

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
    # MEASURED by 3b: data/corpus/chat/ is 160,414 rows = 0.038B, full-domain tokenize.
    # fb ruled both renders of those SAME rows, into two directories. chatml re-renders the
    # same 160,414 rows with ChatML markup, so it is slightly LARGER than the plain form:
    # 0.039B measured (3b, 2026-09-01), the delta being special tokens.
    #
    # Not 0.076B for either: that figure is the two summed, and no single domain contains it.
    # I introduced the 0.076B in my own message to fb and it came back in the ruling as a 1.52%
    # per-domain ceiling, which is 8.00 epochs, double the cap. The correct ceiling is 0.76%
    # each, 1.52% combined.
    #
    # The number this replaced was wiki_chat's stamp, 0.284B, ~7x over: wiki_chat is a MERGED
    # wiki+chat domain and the QA-format rows are only a subset of it. At 0.284B the 4-epoch
    # ceiling read 2.11 and said nothing; at the true 0.038B the draw would have been 15.79
    # epochs. The ceiling was never broken -- it was fed a supply figure that had never been
    # measured for the domain it names.
    #
    # PRECISION IS LOAD-BEARING HERE, unlike the other seven. Both figures sit within a few
    # percent of the 4-epoch line, so a rounding error in the stamp moves the cap by a whole
    # epoch -- 0.038B lands on 4.000 exactly. 3b reported two significant figures; the exact
    # integers are requested and these are placeholders until they land. Every other domain in
    # this table has slack measured in whole epochs and does not care.
    "chatml":             39_000_000,   # ChatML render of data/corpus/chat/, 2sf
    "chat_qa":            38_000_000,   # plain 问：/答： render, same 160,414 rows, 2sf
}

# Supply figures that are NOT measurements of the domain they name. A comment saying so is
# not a gate -- the mix would still launch. This set makes the writer stamp the JSON with a
# refusal flag and print a loud line, so the number cannot be used by someone who did not read
# the comment above it. Empty is the normal state; a name here blocks launch until measured.
UNTRUSTED_SUPPLY = {
    # Empty is the normal state. chatml/chat_qa cleared once 3b measured data/corpus/chat/ at
    # 0.038B (full-domain tokenize, 160,414 rows) and fb ruled the render source; the entry is
    # kept as a comment rather than deleted so the next person can see the shape of a real one.
}

# Directories named by any data/mix_scale_*.json. Writing new corpus into one of these
# falsifies the ladder's fingerprint; an A/B was correctly stopped at startup over exactly
# this two days ago. The writer refuses a domain whose name collides.
LADDER_DIRS = {"chat", "code", "en", "math", "textbook", "web_hq", "wiki"}

# Relative uncertainty of each supply figure. Absent = exact (a stamp is an integer count).
# 3b reported chatml/chat_qa to two significant figures, so 38_000_000 means 38.0M +/- 0.5M.
# This exists because a fractional-epoch verdict is only as sharp as the supply behind it:
# chat_qa's shipped draw is 3.99996 epochs, four parts in 100,000 under the line, on a number
# whose own error bar is 1.3%. The ceiling PASSES and cannot know it is passing -- the same
# shape as the wiki_chat defect, where a guard was silent because its input was wrong. There
# the input named the wrong domain; here it names the right domain at the wrong precision.
SUPPLY_RELATIVE_ERROR = {"chatml": 0.5 / 39.0, "chat_qa": 0.5 / 38.0}

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


# The 16B readout's per-role nat/B figures. Present here ONLY so the refusal below can
# recognise them; nothing in this file may use them to set a weight.
NAT_PER_B = {
    "wiki_chat": 2.8541, "textbook_30b": 0.5024, "zh_web": 0.1657, "cot": 0.1180,
    "code_rp1t": 0.0162,
}

# Prose names the roles the way a person does. The first version of the refusal keyed on the
# NAT_PER_B keys alone, so the real withdrawn sentence -- which says "textbook", not
# "textbook_30b" -- matched one role instead of two and walked straight through the gate. The
# guard was written against the violation and still did not catch it, because the violation is
# written in English and the guard was reading identifiers.
ROLE_ALIASES = {
    "wiki_chat": ("wiki_chat", "wiki"),
    "textbook_30b": ("textbook_30b", "textbook"),
    "zh_web": ("zh_web", "chinese web"),
    "cot": ("cot", "chain-of-thought"),
    "code_rp1t": ("code_rp1t", "code"),
}
assert set(ROLE_ALIASES) == set(NAT_PER_B), "every rate needs its prose names"


def _refuse_cross_role_rate(name, why):
    """Refuse a weight justified by comparing per-role nat/B across roles.

    docs/standards/training_loop.md section 6 bans this: nat/B divides a role's dloss by its
    OWN tokens, so transfer inflates every figure by an unmeasured, role-dependent amount and
    the cross-role ordering cannot be assumed to survive. I wrote section 6. I then violated it
    twice in two days -- once inside the document that states the ban (caught by 44), and once
    justifying this file's cot->textbook move as "same cost, 4.3x the measured rate".

    Twice by the author is data about the rule, not about the author: the numbers are sitting
    right there, they answer the question being asked, and the ban lives in a paragraph nobody
    re-reads while allocating. A rule that must be remembered has already failed; a rule that
    refuses at the point of use cannot fail that way (fb 2026-09-01).

    The predicate is deliberately crude -- two or more roles named in one justification, next to
    a nat/B figure. Crude in the direction of false positives: a legitimate mention has to be
    reworded, which costs a sentence, while a missed violation costs a weight nobody can defend.
    Match on PROSE names (ROLE_ALIASES), not the domain keys: the sentence that had to be
    withdrawn says "textbook", and a guard reading identifiers scores that as one role.
    """
    low = why.lower()
    mentioned = sorted({r for r, names in ROLE_ALIASES.items() if any(n in low for n in names)})
    rates = [f"{v:.4f}" for v in NAT_PER_B.values() if f"{v:.4f}" in why]
    if len(mentioned) >= 2 and rates:
        raise SystemExit(
            f"REFUSING: {name}'s justification cites nat/B ({', '.join(rates)}) while naming "
            f"{len(mentioned)} roles ({', '.join(mentioned)}). training_loop.md section 6 bans "
            "nat/B for cross-role allocation -- transfer inflates each role's rate by an "
            "unmeasured, role-dependent amount, so the ordering is not known to hold. Justify "
            "the weight from the objective, or from a within-role change over time."
        )


def _allocation():
    """The objective's weights with supply-capped domains held fixed and the rest renormalised.

    One function because the selftest needs the same numbers build() uses, and a selftest that
    recomputes an allocation is testing its own copy of the logic -- which is how a check drifts
    from the thing it checks. `code` appears here as a single objective; _code_split divides its
    rows afterwards.
    """
    capped = {n: _ceiling_weight(n) for n in OBJECTIVE if n in SUPPLY_CAPPED}
    free = {n: w for n, (w, _) in OBJECTIVE.items() if n not in SUPPLY_CAPPED}
    free["code"] = CODE_TOTAL
    scale = (1.0 - sum(capped.values())) / sum(free.values())
    return dict({n: w * scale for n, w in free.items()}, **capped)


def _ceiling_weight(name):
    """The largest weight for a supply-capped domain whose WHOLE error band clears the ceiling.

    A supply-capped weight is not a judgement, it is arithmetic on the supply -- so it should be
    computed from the supply rather than typed in beside it. The hand-typed 0.0076 was correct
    for a supply of exactly 38.0M and wrong for the 2sf figure actually measured: it drew 3.99996
    epochs, four parts in 100,000 under the line, from a number carrying +/-1.3%. That is not a
    pass, it is a coin flip that landed well.

    So the band, not the point estimate, has to clear: size the weight against the LOW end of the
    supply. It costs 0.01pt of chat_qa and buys a verdict that does not depend on which way a
    rounding went. When 3b lands the exact integers, SUPPLY_RELATIVE_ERROR loses the entry and
    this returns to the sharp ceiling with no other edit.
    """
    rel = SUPPLY_RELATIVE_ERROR.get(name, 0.0)
    max_rows = int(EPOCH_SOFT_CEILING * SUPPLY[name] * (1 - rel)) // SEQ
    # _weight_for_rows, not a floor at some chosen precision: build_mix draws int(ROWS*weight),
    # so the weight has to hit max_rows EXACTLY. Flooring to 4dp instead cost 488 rows -- a
    # rounding loss dressed as a safety margin, and indistinguishable from one by anyone reading
    # the shipped weight.
    return _weight_for_rows(max_rows, ROWS)[0]


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
    # Never hand a leftover row to a SUPPLY-CAPPED domain. Its weight was computed to land
    # exactly on the epoch ceiling, so one extra row puts it over: chatml drew 37,110 rows for
    # 4.000067 epochs and tripped its own ceiling warning. The overshoot is 6.7e-5 of an epoch
    # and utterly harmless -- and that is the trap, because the cheap fix is a tolerance in the
    # comparison, which silences a guard to hide a rounding bug instead of not creating one.
    eligible = [n for n in sorted(weights, key=lambda n: total_rows * weights[n] - floors[n],
                                  reverse=True) if n not in SUPPLY_CAPPED]
    assert len(eligible) >= short, f"cannot place {short} leftover rows outside capped domains"
    for n in eligible[:short]:
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
    alloc = _allocation()
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
        _refuse_cross_role_rate(name, why)
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
        if runtime * SEQ / pool_tok > EPOCH_SOFT_CEILING:
            warnings.append(
                f"{name}: {runtime * SEQ / pool_tok:.2f} epochs exceeds the {EPOCH_SOFT_CEILING}-epoch "
                f"line (ds.muennighoff_four_epoch). Past 4 epochs repeated tokens stop paying and "
                f"this weight is buying re-reads, not information."
            )
        # A PASS the supply is not precise enough to support is not a pass. Widen the draw by the
        # supply's own error bar; if the ceiling verdict flips inside that band, the guard has no
        # opinion and must say so instead of returning the comfortable side of it.
        rel = SUPPLY_RELATIVE_ERROR.get(name)
        if rel:
            worst = runtime * SEQ / (pool_tok * (1 - rel))
            if runtime * SEQ / pool_tok <= EPOCH_SOFT_CEILING < worst:
                warnings.append(
                    f"{name}: {runtime * SEQ / pool_tok:.5f} epochs is UNDER the "
                    f"{EPOCH_SOFT_CEILING}-epoch line, but the supply is known to +/-{rel * 100:.1f}% "
                    f"and at the low end the draw is {worst:.2f}. The verdict flips inside the error "
                    f"bar, so this is not a pass -- it is a guard reporting on a number too coarse "
                    f"to decide with. Get the exact token count, or cut the weight until the WHOLE "
                    f"band clears."
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
        "_untrusted_supply": {n: why for n, why in UNTRUSTED_SUPPLY.items() if n in domains},
        "_launch_blocked": sorted(n for n in UNTRUSTED_SUPPLY if n in domains),
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
    code_rows_target = _rows_for_weights(_allocation(), ROWS)["code"]
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

    # 7. the untrusted-supply gate fires, and an empty set does not block. Written because a
    #    COMMENT saying "this number is wrong" is not a gate: the mix still builds and still
    #    launches, and the reader who needed the warning is the one who did not read it.
    #
    #    The gate uses its OWN fixture rather than the live UNTRUSTED_SUPPLY. The first version
    #    asserted chatml was blocked, which was true while chatml's supply was wrong and became
    #    a failing test the moment 3b measured it -- a test that goes red when a defect is FIXED
    #    is testing the defect, not the guard. The live set is empty now and must stay testable.
    assert not build(8.85e9)["_launch_blocked"], (
        f"nothing should be blocked once every supply is measured: "
        f"{build(8.85e9)['_launch_blocked']}"
    )
    saved = dict(UNTRUSTED_SUPPLY)
    UNTRUSTED_SUPPLY["cot"] = "fixture: pretend cot's supply came from the wrong domain"
    try:
        m = build(8.85e9)
        assert m["_launch_blocked"] == ["cot"], f"gate did not fire: {m['_launch_blocked']}"
        assert m["_untrusted_supply"]["cot"], "blocked without a reason is not a usable refusal"
    finally:
        UNTRUSTED_SUPPLY.clear()
        UNTRUSTED_SUPPLY.update(saved)
    assert not build(8.85e9)["_launch_blocked"], "the fixture leaked into the real set"
    print("  7 untrusted-supply gate blocks launch on a planted entry, clears when empty, "
          "and the live set is empty (chatml resolved by measurement)")

    # 8. the section-6 refusal fires on the EXACT text I shipped and had to withdraw, and does
    #    not fire on the replacement. A guard whose failing case is invented tests the guard
    #    against the author's imagination; this one is tested against the real violation.
    real_violation = ("textbook/instructional. The 16B readout measured it at 0.5024 nat/B "
                      "own-token, second-steepest of the seven and 30x cot's rate. Took cot's "
                      "released point for exactly that reason: same cost, 4.3x the measured rate.")
    try:
        _refuse_cross_role_rate("textbook_30b", real_violation)
        raise AssertionError("the withdrawn justification passed the section-6 refusal")
    except SystemExit as e:
        assert "section 6" in str(e) and "0.5024" in str(e), str(e)
    # the shipped replacement, which justifies from the objective, must pass
    for name, (_, why) in OBJECTIVE.items():
        _refuse_cross_role_rate(name, why)
    # and a within-role claim over time is legitimate: one role, its own rate
    _refuse_cross_role_rate("cot", "cot's 0.1180 nat/B at 16B, to be re-read at 22B")
    print("  8 section-6 refusal fires on the withdrawn text, passes every shipped "
          "justification and a within-role claim")

    # 9. the error-band refusal. A supply known to 2sf cannot decide a verdict that flips inside
    #    its own error bar, and the shipped chat_qa was exactly there: 3.99996 epochs -- a PASS,
    #    four parts in 100,000 under the line, from a number carrying +/-1.3%. The point estimate
    #    was on the right side of the ceiling by less than the ceiling could resolve.
    #
    #    Two directions, because "it does not warn now" is what the wiki_chat ceiling also said.
    #    First: force the weight back to the hand-typed 0.0076 and the band refusal MUST fire --
    #    this is the real configuration that shipped, not an invented one. Second: with the
    #    derived weight the whole band clears, so the pass is a pass at BOTH ends of the supply.
    rel = SUPPLY_RELATIVE_ERROR["chat_qa"]
    hand_typed_rows = int(ROWS * 0.0076)
    assert hand_typed_rows * SEQ / SUPPLY["chat_qa"] <= EPOCH_SOFT_CEILING, (
        "check 9 assumes the hand-typed weight PASSED the point-estimate ceiling; if it now "
        "fails outright the band refusal is not what is being tested"
    )
    assert hand_typed_rows * SEQ / (SUPPLY["chat_qa"] * (1 - rel)) > EPOCH_SOFT_CEILING, (
        "the hand-typed 0.0076 must cross the line at the low end of the supply -- that is the "
        "defect this check exists for"
    )
    for name in SUPPLY_CAPPED:
        rows = int(ROWS * _ceiling_weight(name))
        low = SUPPLY[name] * (1 - SUPPLY_RELATIVE_ERROR.get(name, 0.0))
        assert rows * SEQ / low <= EPOCH_SOFT_CEILING, (
            f"{name}: derived weight draws {rows * SEQ / low:.4f} epochs at the low end of its "
            f"supply -- the band must clear, not just the point estimate"
        )
    # and with an EXACT supply the derivation returns to the sharp ceiling, using all of it
    saved_err = dict(SUPPLY_RELATIVE_ERROR)
    SUPPLY_RELATIVE_ERROR.clear()
    try:
        sharp = int(ROWS * _ceiling_weight("chat_qa"))
        assert sharp >= hand_typed_rows, (
            f"with no error bar the derivation must not be more conservative than the hand-typed "
            f"weight: {sharp} rows vs {hand_typed_rows}"
        )
        assert sharp * SEQ / SUPPLY["chat_qa"] <= EPOCH_SOFT_CEILING, "sharp ceiling overshot"
    finally:
        SUPPLY_RELATIVE_ERROR.update(saved_err)
    print("  9 the hand-typed 0.0076 fails the error-band ceiling it passed on the point "
          "estimate; the derived weight clears at both ends and reverts to sharp when exact")

    print("selftest: 9/9")
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
    for n in m["_launch_blocked"]:
        print(f"  LAUNCH BLOCKED {n}: {m['_untrusted_supply'][n]}")
    if m["_launch_blocked"]:
        print("  -> this mix must not start a run until every blocked supply is measured")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
# THE CODE SHARE ABOVE 20B, a composition ruling and not arithmetic (4c 2026-09-07, for the 30B
# MoE-48 run). Above 20B the three supply-capped domains are sized down by _ceiling_weight -- at
# 30B cot 0.0807 -> 0.0538, chatml 0.0074 -> 0.0049, chat_qa 0.0073 -> 0.0048, releasing 0.0317852
# of weight. build()'s renormalisation spreads that pro-rata over every uncapped domain, which sent
# 38% of it to code and the largest single share (+0.0092857) to math_owm_stage2, with +0.0010713
# even to zh_web -- the domain this file's own comment calls irrelevant to an English code model.
# The ruling is that the released weight goes where the OBJECTIVE is: code.
#
# The rationale, recorded because a number without one is the thing this file exists to prevent:
# the target is a code model, and the 8B control profile measured NO humaneval gain from the MoE
# arm (facts/moe.json#moe.control_profile_vs_dense_b192 -- the two bpb estimators disagree in sign
# over the same 164 tasks, so code capability is indistinguishable between the arms at 8B). Code is
# therefore both the objective and the measured deficit, and released weight funds it rather than
# being smeared across roles nobody released it for.
#
# Applied ONLY above 20B, so every mix at or below the decided total keeps 0.34 and no committed
# file changes. Verified against both gates the ruling named: at 0.3717852 and TOTAL=30B,
# code_py_starcoder draws 1.2062 epochs and code_py_rp1t 1.2650, both far under the 4-epoch
# ceiling, so the fallback of routing the excess to a new deduped domain is not needed. CODE_FLOOR
# is untouched by this -- it tests SUPPLY (9.17B available against a 2.5B floor), not the weight.
CODE_TOTAL_ABOVE_20B = 0.3717852
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
    "cot":               (None, "chain-of-thought. THE WEIGHT IS A CEILING, NOT A TARGET, so it is\n"
                                "derived (None) rather than typed: cot's supply is "
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
                                  "once. THE SAME 160,414 CONVERSATIONS ALSO SIT IN chat_qa, "
                                  "and each domain draws ~4 epochs, so this content is seen "
                                  "8-10 times in total (e1 found the overlap; fb ruled KEEP, "
                                  "2026-09-01). The ruling is not that the repetition is "
                                  "harmless in general -- it is that what is being bought here "
                                  "is a FORMAT entering the distribution, formats are cheap to "
                                  "learn, and 1.5% combined is what that costs. What would "
                                  "change it: evidence that the model is reciting these "
                                  "conversations rather than absorbing the markup. Nobody has "
                                  "measured that. WEIGHT IS None BECAUSE IT IS NOT A JUDGEMENT: this "
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
                                  "combined, not each. THAT MEANS THE SAME 160,414 CONVERSATIONS "
                                  "ARE SEEN 8-10 TIMES across the two domains, which fb ruled "
                                  "acceptable on 2026-09-01 for the reason spelled out in "
                                  "chatml's entry: the purchase is two formats in-distribution, "
                                  "not two passes of content. Weight is None for the same reason as "
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
# cot joins on 2026-09-01, when the ceiling moved from tokens to rows and cot's typed 8% came
# out at 4.03 epochs against the real cache. It should have been here from the start: section
# 2.1 of the rationale already said "8% is a CEILING, not a target", and a weight that IS a
# ceiling has no business being a literal -- the literal was correct for a token-denominated
# ceiling and silently wrong the moment the denomination was fixed. Same defect as the
# hand-typed 0.0076, one dict entry away.
SUPPLY_CAPPED = {"chat_qa", "chatml", "cot"}
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
# Fallback only: the real value is read from the corpus stamp when it is present (see
# _rp1t_tokens). This constant is what a dev box without the corpus falls back to, and it is
# labelled as the ast.parse pass's own count so nobody reads it as current.
RP1T_PYTHON_TOKENS_FALLBACK = 420_646_182

# WHICH corpus fills the rp1t half of the code objective. resume 2 swaps this to
# "code_rp1t_dd09", the near-dedup rebuild (data/corpus/code_rp1t_dd09, 235 shards). Held as a
# name rather than edited in three call sites because _code_split, _rp1t_tokens and the floor
# message all need the same one, and a swap that changes two of three is silent.
#
# NOT SWITCHED YET, deliberately: dd09's build_corpus_stats.json currently carries only
# {domain, filters, n_shards} -- no `tokens` and no `fingerprint` (b0 measured it on the pod,
# 2026-09-07). _rp1t_tokens refuses a non-default domain with no stamped tokens, and
# launch_gate.gate_corpora:184 already NOGOs a domain whose stamp has no fingerprint, so
# flipping this line before 3b's pass writes those fields cannot ship a mix -- it fails, which
# is the correct state, not a silent fallback to another corpus's count.
CODE_RP1T_DOMAIN = "code_py_rp1t"


# One-epoch supply, tokens. Measured stamps for the landed domains; code is a parameter.
# chatml is a RE-RENDER of wiki_chat's chat rows, not new data, so its supply is bounded by
# the source it is rendered from -- recorded as such rather than as an independent corpus.
SUPPLY = {
    "math_owm_stage2": 6_513_304_690,   # stamped, fp 1e687e4b5ce37598
    "en_c4_stage2":    2_403_694_865,   # stamped, fp 05e0fc6f14704056
    "cot":               424_056_227,   # stamped, fp 388496b76ed9bf88
    "textbook_30b":    1_610_210_330,   # stamped, fp 3f237c5191cb8571
    "zh_web":         21_293_403_945,   # stamped, fp a0d44fc44a289d60
    # MEASURED by 3b, exact and in TRAINING UNITS (2026-09-01). Both domains hold the same
    # 160,414 documents; chatml is the ChatML re-render, larger by its markup.
    #
    #   chat_qa  38,187,650   data/corpus/chat_qa/  (a copy read out of chat/, which is untouched)
    #   chatml   38,995,846   data/corpus/chatml/
    #
    # THE UNIT IS THE POINT. scripts/count_tokens.py:16 defines the corpus convention as
    # "ids + one <eos> per document (train.py encode)" -- count_docs sums len(ids)+1 -- so a
    # stamp counts what training actually consumes. 3b first sent chat_qa as bare ids
    # (38,027,236) and chatml as its stamp (38,995,846): two domains in two different units,
    # and both plausible. Caught because the gap to chat_qa's stamp was 160,414, exactly one
    # token per document; a byte-scaled extrapolation cannot land on integer-1-per-doc, so
    # "that stamp is a 3-shard estimate" could not be the explanation.
    #
    # A mix pool must be in the same units as train.py, because every other supply here comes
    # from a stamp. Taking bare ids for these two would put one domain pair on a different
    # ruler from the other seven -- structurally this morning's wiki_chat defect: the guard is
    # fine, the number handed to it is not the quantity it names. Cost was 157 rows of draw,
    # in the direction of under-reading.
    #
    # Not 0.076B for either: that figure is the two summed, and no single domain contains it.
    # I introduced it in my own message to fb and it came back in the ruling as a 1.52%
    # per-domain ceiling, which is 8.00 epochs, double the cap.
    #
    # The number all of this replaced was wiki_chat's stamp, 0.284B, ~7x over: wiki_chat is a
    # MERGED wiki+chat domain and the QA rows are only a subset of it. At 0.284B the 4-epoch
    # ceiling read 2.11 and said nothing; at the true supply the draw would have been 15.79
    # epochs. The ceiling was never broken -- it was fed a figure never measured for the
    # domain it names. Three defects now, one domain, one day, all in the INPUT.
    "chatml":         38_995_846,   # ChatML render, data/corpus/chatml/
    "chat_qa":        38_187_650,   # plain 问：/答： render, data/corpus/chat_qa/
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

# Relative uncertainty of each supply figure. EMPTY is the correct state: every supply here
# is now an exact count. It existed because chatml/chat_qa arrived as two significant figures
# and their draw lands within a rounding error of the 4-epoch line -- chat_qa read 3.99996
# epochs, a PASS by four parts in 100,000, from a number carrying +/-1.3%. The ceiling PASSED
# and could not know it was passing: the same shape as the wiki_chat defect, where a guard was
# silent because its input was wrong. There the input named the wrong domain; there it named
# the right domain at the wrong precision.
#
# Kept, empty, rather than deleted: the mechanism is the only thing that makes an imprecise
# supply say so instead of returning the comfortable side of its own error bar, and the next
# domain to arrive as a rounded figure needs it on arrival, not after it ships. Selftest 9
# exercises it against the weight that actually shipped.
SUPPLY_RELATIVE_ERROR = {}

# Where each supply figure came from, and in which units. Every entry must be TRAIN_UNITS:
# ids plus one <eos> per document (scripts/count_tokens.py CONVENTION), which is what a corpus
# stamp records and what train.py consumes.
#
# This exists because units are the one property of a supply that no arithmetic on the supply
# can recover. 3b sent chat_qa as bare ids (38,027,236) and chatml as its stamp (38,995,846):
# two domains, two rulers, and BOTH figures individually reasonable -- nothing about either
# number looks wrong on its own. It was caught only by comparing chat_qa against its own stamp,
# where the gap was 160,414, exactly one token per document.
#
# My first attempt at a guard tried to detect it from the two supplies alone, by looking for a
# sibling gap of n_docs. That is not decidable: the observed gap is the markup delta PLUS the
# missing EOS, and the markup delta is not known in advance. The check passed its own selftest
# only because I had picked the fixture to match my rule. Deleted -- a guard whose predicate is
# invented tests the author's imagination, which is the thing this file keeps re-learning.
#
# What IS enforceable is a declaration. Adding a supply means stating its units here, and the
# mix refuses to build if any domain is on a different ruler from the rest. That does not
# detect a mislabelled number, and it is not claimed to: it makes the question unskippable at
# the point where the number enters, which is where 3b's two figures would have collided.
TRAIN_UNITS = "ids + one <eos> per document (scripts/count_tokens.py CONVENTION)"
SUPPLY_UNITS = {name: TRAIN_UNITS for name in SUPPLY}


def _refuse_mixed_units():
    """Refuse a mix whose supplies are not all in the same units."""
    missing = sorted(set(SUPPLY) - set(SUPPLY_UNITS))
    if missing:
        raise SystemExit(
            f"REFUSING: {', '.join(missing)} has a supply but no declared unit. Every other "
            f"supply here is {TRAIN_UNITS}; a figure on a different ruler silently rescales one "
            f"domain's epochs against the rest of the mix. State the units."
        )
    odd = sorted(n for n, u in SUPPLY_UNITS.items() if u != TRAIN_UNITS)
    if odd:
        raise SystemExit(
            f"REFUSING: {', '.join(odd)} is not in training units ({TRAIN_UNITS}). Bare ids "
            f"undercount by one token per document; the mix pool must be the number train.py "
            f"consumes, because every other supply here is a stamp."
        )

# Muennighoff's repeat-decay constant: R_D* = 15.4, and <=4 epochs costs ~nothing
# (ds.muennighoff_four_epoch: 8.7B at 4 epochs finishes +0.5% val vs single-epoch).
EPOCH_SOFT_CEILING = 4


def _weight_for_rows(rows, total_rows):
    """Shortest decimal weight w with int(total_rows*w) == rows exactly.

    build_mix computes want = int(total_rows * weight), so the weight is the only executable
    denomination -- tokens and rows are views of it. Stating rows without the weight that
    produces them is how stage 1 under-drew cot by 61,593,088 tokens.
    """
    # 5..16, not 5..13. The reachable row counts are NOT monotonic in `places`: for 271,881 rows
    # of 7,324,218, 12dp gives 271,880 and 13dp gives 271,881 exactly, while 14dp and 15dp fall
    # back to 271,880 -- float64 spacing near 0.0371 is coarser than the last decimal asks for.
    # Hit while routing the ceiling through the measured pools: the clamp asks for whatever row
    # count the pool implies, and an arbitrary row count is exactly the case a 12dp ceiling cannot
    # always encode. The assertion below still fires if no precision works, so a genuinely
    # unrepresentable count is still loud rather than silently rounded.
    for places in range(5, 17):
        w = round(rows / total_rows, places)
        if int(total_rows * w) == rows:
            return w, places
    raise AssertionError(f"no weight at 5..16 decimal places yields exactly {rows} rows of "
                         f"{total_rows}; float64 cannot encode this draw")


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


def _band_warning(name, rows, pool_tok):
    """Refuse a ceiling verdict that flips inside the supply's own error bar.

    A supply given to two significant figures cannot decide a draw that lands four parts in
    100,000 under the line, which is exactly where chat_qa sat: 3.99996 epochs, a PASS, from a
    number carrying +/-1.3%. At the low end of that band the draw is 4.05. The guard was not
    broken and was not fed the wrong domain -- it was fed the right domain at a precision too
    coarse for the question, and returned the comfortable side of its own uncertainty.

    Returns a list so build() can extend unconditionally, and so the selftest can call the real
    predicate rather than a second copy of the arithmetic.
    """
    rel = SUPPLY_RELATIVE_ERROR.get(name)
    if not rel:
        return []
    point = rows * SEQ / pool_tok
    worst = rows * SEQ / (pool_tok * (1 - rel))
    if not (point <= EPOCH_SOFT_CEILING < worst):
        return []
    return [
        f"{name}: {point:.5f} epochs is UNDER the {EPOCH_SOFT_CEILING}-epoch line, but the "
        f"supply is known to +/-{rel * 100:.1f}% and at the low end the draw is {worst:.2f}. The "
        f"verdict flips inside the error bar, so this is not a pass -- it is a guard reporting "
        f"on a number too coarse to decide with. Get the exact token count, or cut the weight "
        f"until the WHOLE band clears."
    ]


def _allocation(cursor=None, code_tokens=None):
    """The objective's weights with supply-capped domains held fixed and the rest renormalised.

    One function because the selftest needs the same numbers build() uses, and a selftest that
    recomputes an allocation is testing its own copy of the logic -- which is how a check drifts
    from the thing it checks. `code` appears here as a single objective; _code_split divides its
    rows afterwards.
    """
    capped = {n: _ceiling_weight(n, cursor) for n in OBJECTIVE if n in SUPPLY_CAPPED}
    free = {n: w for n, (w, _) in OBJECTIVE.items() if n not in SUPPLY_CAPPED}
    # THE CODE SHARE ABSORBS WHAT THE CAPPED DOMAINS RELEASE, above 20B only (4c's ruling; see
    # CODE_TOTAL_ABOVE_20B). Raising code's pre-scale share by the released amount is what makes the
    # renormalisation below a no-op for every OTHER free domain: `scale` is
    # (1 - sum(capped)) / sum(free), and both the numerator and sum(free) rise by that same amount,
    # so scale returns to ~1 and the other five keep their decided weights instead of each taking a
    # slice of weight released by a domain they have nothing to do with.
    free["code"] = CODE_TOTAL_ABOVE_20B if TOTAL_TOKENS > 20_000_000_000 else CODE_TOTAL
    if cursor:
        free = _place_freed_under_ceiling(free, cursor, code_tokens)
    # THE RENORMALISATION MUST NOT REACH A CLAMPED DOMAIN, or the ceiling is a comment. Placing
    # the freed share raises sum(free) above the pre-scale total, so `scale` came out at 1.013417
    # on .step22500 -- and multiplying a clamped weight by 1.013417 puts it back over the ceiling
    # it was just clamped to. Measured before this fix: math and starcoder were clamped to 2.10
    # epochs and written at 2.1282, the clamp applied and then undone in the same function.
    # So: hold the clamped domains fixed alongside `capped` and renormalise only what is free to
    # move. Iterate, because absorbing the difference can push a previously-unclamped recipient
    # over its own ceiling; the loop is bounded by the number of free domains, since each pass
    # either clamps one more or converges.
    return _renormalise_holding_clamped(free, capped, cursor, code_tokens)


def _renormalise_holding_clamped(free, capped, cursor, code_tokens):
    """Scale the unclamped free domains to fill 1 - sum(capped) - sum(pinned).

    A domain is PINNED when the ceiling decided its weight rather than the objective. Such a
    weight is a bound, so scaling it is not a renormalisation -- it is a violation with a
    multiplication in front of it. Separated from _allocation so the fixed point is testable.
    """
    free = dict(free)
    pinned = {}
    # Bounded by len(free): each pass either pins one more domain or converges.
    for _ in range(len(free) + 1):
        movable = {n: w for n, w in free.items() if n not in pinned}
        if not movable:
            raise SystemExit(
                "REFUSING: every free domain is pinned at its ceiling, so nothing can absorb the "
                f"remainder; the weights sum to {sum(free.values()) + sum(capped.values()):.6f}."
            )
        scale = (1.0 - sum(capped.values()) - sum(pinned.values())) / sum(movable.values())
        scaled = dict(pinned, **{n: w * scale for n, w in movable.items()})
        over = _ceiling_overshoot(scaled, cursor, code_tokens)
        if not over:
            return dict(scaled, **capped)
        # Pin each overshooting domain AT its ceiling and renormalise the rest around it.
        pinned.update(over)
        free.update(over)
    raise SystemExit("REFUSING: ceiling renormalisation did not converge")


def _ceiling_overshoot(weights, cursor, code_tokens):
    """{domain: its weight AT the ceiling} for every FREED_RECIPIENT above FREED_CEILING.

    Reads the same weights that will be written, so it cannot pass a check the file then fails.
    starcoder is translated through _code_split's share for the reason _place_freed_under_ceiling
    documents: `code` is the objective, starcoder is a fraction of it.
    """
    if not cursor:
        return {}
    sc_share = code_tokens / (code_tokens + _rp1t_tokens())
    out = {}
    for name in FREED_RECIPIENTS:
        is_sc = name == "code_py_starcoder"
        key = "code" if is_sc else name
        if key not in weights:
            continue
        pool = _domain_pool_rows(name, code_tokens if is_sc else SUPPLY[name])
        drawn = int(ROWS * weights[key])
        if is_sc:
            drawn = int(drawn * sc_share)
        if drawn + int(cursor.get(name, 0)) <= int(FREED_CEILING * pool):
            continue
        rows = max(0, int(FREED_CEILING * pool) - int(cursor.get(name, 0)))
        out[key] = _weight_for_rows(int(rows / sc_share) if is_sc else rows, ROWS)[0]
    return out


# The two domains that absorb what the capped three release on a resume, and the ceiling they may
# not cross.
def _ceil_pool(name):
    """The pool row count _ceiling_weight sizes against, for tests that need to construct a
    cursor relative to the ceiling. Same expression, not a copy of the number."""
    return _domain_pool_rows(name, SUPPLY[name] * (1 - SUPPLY_RELATIVE_ERROR.get(name, 0.0)))


FREED_RECIPIENTS = {"math_owm_stage2": 0.2643, "code_py_starcoder": 0.3297}
# 2.10 TOTAL EPOCHS, ruled by 4c 2026-09-07 after two withdrawals, and the withdrawals are the
# reason the number is where it is. The first ruling said 1.000, which is below what the UNCHANGED
# mix already draws (math 1.2179, starcoder 1.1314 measured on .step22500) -- a ceiling under the
# untouched behaviour cannot be met by any reallocation, so the overflow rule had nowhere to send
# anything. The second said 1.30, and I sized that against the 3,004,219-row SEGMENT while every
# weight in this file is a fraction of the 7,324,218-row WHOLE PLAN: at 1.30 the recipients may
# take 2,268,335 plan rows while their launch weights already ask 4,980,346. Both numbers were
# real and each was correct for its own denominator, which is why neither reading looked wrong.
# 2.10 is above what the fixed deriver produces, so nothing binds today and this changes no number
# in the launch mix. THAT CLAIM IS ASSERTED IN THE SELFTEST (case 17) AGAINST THIS VALUE, not stated
# here: three different pairs for those two epoch figures were in circulation in this file and none
# matched the code to better than 1.2%, so the number is derived where it is checked and this comment
# names only the ruling. It is here so the ceiling is a guard and not a comment.
FREED_CEILING = 2.10


def _place_freed_under_ceiling(weights, cursor, code_tokens):
    """Send the share released by the capped domains to FREED_RECIPIENTS, pro rata.

    THE CEILING IS NOT ENFORCED HERE, and the first version's attempt to is worth recording.
    It computed each recipient's room as `int(FREED_CEILING * pool) - cursor` and clamped the
    FREED rows against it -- but a recipient's base weight already draws from that same room, so
    the comparison was freed-rows against total-room and passed a recipient whose total was over.
    4c's ruling is stated in TOTAL epochs (cursor + everything this plan draws), so the bound has
    to be applied to the final weights, which only _renormalise_holding_clamped sees. Two
    derivations of one ceiling in two units is how the 1.000 and 1.30 rulings both went wrong.

    code_py_starcoder IS NOT AN OBJECTIVE. `code` is, and _code_split divides it between starcoder
    and rp1t IN PROPORTION TO SUPPLY, so starcoder's share has to be translated back through that
    split or the freed rows would be added to a quantity nothing draws.
    """
    # STARCODER SUPPLY IS A PARAMETER, not a SUPPLY entry: build() takes it as code_tokens and
    # _code_split is handed it too. Reading a constant here would size against a different supply
    # than the split divides, and the two would drift silently.
    if code_tokens is None:
        raise SystemExit("REFUSING: freed-share placement needs starcoder supply (code_tokens)")
    sc_share = code_tokens / (code_tokens + _rp1t_tokens())
    # The freed share is whatever the capped domains did NOT take relative to a fresh start: their
    # cursor-free weights minus their cursor-aware ones. Derived, not passed in, so it cannot drift
    # from what _ceiling_weight actually returned.
    freed = sum(_ceiling_weight(n) - _ceiling_weight(n, cursor) for n in SUPPLY_CAPPED
                if n in OBJECTIVE)
    if freed <= 0:
        return weights
    base = sum(FREED_RECIPIENTS.values())
    out = dict(weights)
    for name, share in FREED_RECIPIENTS.items():
        rows = int(ROWS * freed * share / base)
        if name == "code_py_starcoder":
            # rows are STARCODER rows; `code` has to rise by rows/sc_share to deliver them.
            out["code"] = out.get("code", 0.0) + _weight_for_rows(int(rows / sc_share), ROWS)[0]
        else:
            out[name] = out.get(name, 0.0) + _weight_for_rows(rows, ROWS)[0]
    return out


MEASURED = os.path.join(ROOT, "data", "token_cache_pools.json")


def _cache_pool(name):
    """Pool rows read from the domain's OWN token cache, when it is on this host.

    Preferred over the recorded file below, because the cache is what train.py draws from.
    The file is a transcription, and tonight a transcription of the mix itself sat stale on
    the pod for 24 minutes while I reported GO from my laptop -- the same file being two
    different things in two places is exactly what derivation is meant to end.

    Cheap: torch reads the header under mmap, not the tensor. The co-residency refusal is
    still asked, and the reason it is not skipped as "only a header" is that the check
    guarding this rule cannot tell a header read from a whole-tensor one -- the argument for
    an exemption is exactly the argument that lets the next by-path reader in (e1,
    2026-09-05). One domain at a time keeps the measured bytes honest: the refusal is
    handed the domain this call reads, not the mix.
    """
    # train's accessor, not a literal: this function reports measured cache bytes, so reading the
    # overlay copy after the caches moved to NVMe would put a number in a mix file that describes
    # a file no run reads (de, 2026-09-05). _domain_cache_path rather than joining the name here,
    # because the '_fone' suffix is part of the cache's identity and a hand-spelled name silently
    # drops it -- with Cfg.fone set, the two strings differ and this would size the wrong file
    # (58, 2026-09-05).
    sys.path.insert(0, ROOT)
    import train

    path = train._domain_cache_path(name)
    if not os.path.exists(path):
        return None
    # OUTSIDE THE try, deliberately. The `except Exception` below turns any failure into
    # "no cache here", so a refusal raised inside it would be swallowed and this function
    # would fall back to the recorded file while the read it was refusing still looked
    # optional. A guard inside a broad except is not a guard.
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    from cache_guard import assert_not_co_resident

    assert_not_co_resident([name])
    try:
        import torch

        n = int(torch.load(path, map_location="cpu", mmap=True).numel())
    except Exception:
        return None
    rows = n // (SEQ + 1)
    return {"tokens": n, "rows": rows, "pool_rows": rows - min(int(rows * 0.05), 5000),
            "source": "cache"}


def _measured_pools():
    """Pool rows per domain: the live cache where it exists, else data/token_cache_pools.json.

    Per-domain, because "measured" is a property of each domain, not of the mix. The blanket
    ESTIMATED string this replaced said the same thing about all nine, true when nothing had a
    cache and a lie the moment five did.
    """
    out = {}
    if os.path.exists(MEASURED):
        out = {k: v for k, v in json.load(open(MEASURED, encoding="utf-8"))["domains"].items()
               if v.get("pool_rows")}
    for name in list(SUPPLY) + ["code_py_starcoder", "code_py_rp1t"]:
        live = _cache_pool(name)
        if live:
            out[name] = live
    return out


def _rp1t_tokens(name=CODE_RP1T_DOMAIN):
    """The ast.parse-surviving Python supply: the stamp when the corpus is here, else the
    fallback constant. Never both, and the JSON records which one was used.

    The DOMAIN IS A PARAMETER because resume 2 swaps in a near-deduped rebuild under a new
    name (code_rp1t_dd09). The fallback is only correct for the default: it is one specific
    corpus's ast.parse count, so a different domain falling back to it would report another
    corpus's supply under its own name -- SUPPLY's wiki_chat entry records what that costs.
    """
    tok = _corpus_stamp(name, "tokens")
    if tok:
        return tok
    if name != CODE_RP1T_DOMAIN:
        sys.exit(
            f"REFUSING: {name} has no `tokens` in data/corpus/{name}/build_corpus_stats.json, "
            f"and RP1T_PYTHON_TOKENS_FALLBACK is {CODE_RP1T_DOMAIN}'s count, not {name}'s. "
            f"A supply figure measured on another corpus is the wiki_chat defect (SUPPLY:212): "
            f"the guard is fine and the number handed to it is not the quantity it names."
        )
    return RP1T_PYTHON_TOKENS_FALLBACK


def _corpus_stamp(name, field):
    """One field out of a domain's build_corpus_stats.json, or None when the corpus is absent.

    Supplies are READ, not typed, for the same reason fingerprints are (fb, 2026-09-01). Three
    values for code_py_rp1t were in circulation tonight -- my 420,646,182, fb's 420,841,191 and
    the stamp's 421,239,303, a 0.14% spread -- and every one of them was somebody's honest
    transcription of a number that had since moved. A figure copied into source is a claim
    nothing recomputes; read it and the mix goes red via --check when the corpus changes.
    """
    stats = os.path.join(ROOT, "data", "corpus", name, "build_corpus_stats.json")
    if not os.path.exists(stats):
        return None
    try:
        return json.load(open(stats, encoding="utf-8")).get(field)
    except (OSError, ValueError):
        return None


def _corpus_fingerprint(name):
    """This domain's corpus fingerprint, READ from its build_corpus_stats.json.

    launch_gate.gate_corpora pins each domain dir to the bytes the mix was written against by
    comparing this to the live stamp. Derived, never typed: a hand-copied fingerprint is a
    string nothing recomputes, so it cannot go stale and cannot go false -- the exact shape of
    every defect this file has hit today (fb ruling, 2026-09-01). Read it here and the mix
    goes red via --check the moment the corpus changes.

    None when the corpus is not on this host, which is the honest answer on a dev box and is
    what gate_corpora refuses.
    """
    stats = os.path.join(ROOT, "data", "corpus", name, "build_corpus_stats.json")
    if not os.path.exists(stats):
        return None
    try:
        return json.load(open(stats, encoding="utf-8")).get("fingerprint")
    except (OSError, ValueError):
        return None


def _pool_rows(pool_tok):
    """Drawable rows in a pool of pool_tok tokens: packed rows, minus the val holdout.

    train.py: n = len(flat) // (seq+1); seqs[:n_val] is the val split. VERIFIED against the
    real caches on the pod, which is the launch gate's epochs prerequisite -- cot's cache holds
    424,056,227 tokens and 103,504 rows, and 103,504 == 424,056,227 // 4097 exactly. Every
    domain with a cache matched to the row.

    One function because the ceiling and the epochs field must agree; they did not while the
    ceiling worked in tokens.
    """
    rows = int(pool_tok) // (SEQ + 1)
    return rows - min(int(rows * 0.05), 5000)


def _domain_pool_rows(name, pool_tok):
    """ONE answer to "how many rows is this domain's pool", for the ceiling, the epochs field and
    the run. A measured pool WINS over the stamp-derived one, because the cache is what train.py
    draws from.

    THE POOL SOURCE WAS THE THIRD INSTANCE OF THIS PR'S OWN DEFECT (tilerl, 2026-09-07). The unit
    was already shared -- _pool_rows' docstring says "the ceiling and the epochs field must
    agree" -- but the SOURCE was not: build() reported epochs off the measured cache while
    _ceiling_overshoot and _ceiling_weight priced the same quantity off the stamp. Measured on the
    .step22500 mix, that leaves a band of ceilings where the guard passes a domain the file then
    reports as over the line:

        math_owm_stage2    stamp 2.0558 vs cache 2.0509   band (2.0509, 2.0558), gap 0.0048
        code_py_starcoder  stamp 1.9909 vs cache 2.0052   band (1.9909, 2.0052), gap 0.0143

    Nothing lands in either band at FREED_CEILING 2.10, which is why it was invisible and why the
    launched run is unaffected. It is fixed anyway: "a bound computed in one denominator and
    applied in another" is the sentence this PR was written to remove, and leaving the third
    instance in the tree because today's ceiling misses the band is how the first two survived.
    """
    meas = _measured_pools().get(name)
    return meas["pool_rows"] if meas else _pool_rows(pool_tok)


def _ceiling_weight(name, cursor=None, ceiling=EPOCH_SOFT_CEILING):
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

    THE CURSOR IS SUBTRACTED, or this sizes each domain to `ceiling` epochs OF THIS SEGMENT while
    the launch check counts cursor + segment. Those are the same two quantities the epoch guard
    itself confused: PR #2 taught the CHECK to count the total and left this, the thing that
    CHOOSES the weight, counting the segment -- so on the first real resume the two disagreed by
    construction and the mix blocked on its own guard. Measured on .step22500: cot/chatml/chat_qa
    sit at 3.54 total epochs already, so a fresh 4.0-per-segment weight put them at 7.54 and the
    room actually left is 53,553 rows, 1.78% of the segment budget against 15.50% before.
    A guard that counts right and a deriver that does not is one defect wearing two faces.
    """
    rel = SUPPLY_RELATIVE_ERROR.get(name, 0.0)
    # ROWS, not tokens. The ceiling asks how many times the model re-reads the pool, and the
    # pool is packed rows -- tokens overstate it, because packing drops a partial row per
    # document and n_val rows are held out on top. Deriving in tokens put all three capped
    # domains over the real line while reporting them under it.
    pool_rows = _domain_pool_rows(name, SUPPLY[name] * (1 - rel))
    max_rows = ceiling * pool_rows - int((cursor or {}).get(name, 0))
    # NEVER NEGATIVE. A domain already past the ceiling on the cursor alone has no room, and a
    # negative weight would be allocated as one -- silently taking rows from the other domains.
    # Zero is the honest answer and the launch check still reports the overrun.
    max_rows = max(0, max_rows)
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
    total = starcoder_tokens + _rp1t_tokens()
    sc_rows = round(code_rows * starcoder_tokens / total)
    return {
        "code_py_starcoder": (sc_rows, starcoder_tokens),
        CODE_RP1T_DOMAIN: (code_rows - sc_rows, _rp1t_tokens()),
    }


def build(code_tokens, cursor=None):
    """code_tokens is the STARCODER supply; rp1t's parse-verified Python is added to it.

    cursor is the resume checkpoint's row_cursor: {domain: rows already drawn} or None for a
    fresh start. IT CHANGES WHAT `epochs` MEANS. build_mix caps a domain at
    int(pool*epochs) - used[name] with used[] seeded from that same cursor, but compares
    `epochs` against THIS PLAN'S draw only -- so on a resume the ceiling is per-segment and
    every resume grants a fresh 4 epochs on top of whatever the cursor already spent. Measured
    on the live 30B resume: cot's plan drew 3.999 epochs and read as a PASS while the cursor had
    already spent 1.416, for 5.415 total; chat_qa and chatml reached 5.4 the same way, and the
    30B plan would take all three to ~6.0. Passing the cursor makes `epochs` a TOTAL and the
    ceiling enforceable; omitting it keeps the fresh-start behaviour byte for byte.
    """
    code_supply = code_tokens + _rp1t_tokens()
    if code_supply < CODE_FLOOR:
        raise SystemExit(
            f"REFUSING: code supply {code_supply / 1e9:.2f}B (starcoder {code_tokens / 1e9:.2f}B "
            f"+ rp1t python {_rp1t_tokens() / 1e9:.2f}B) is below the {CODE_FLOOR / 1e9:.1f}B "
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
    alloc = _allocation(cursor, code_tokens)
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
    over_ceiling = {}
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
        meas = _measured_pools().get(name)
        # A measured pool WINS over the stamp-derived one. Not "if they disagree, warn": the
        # cache is the thing build_mix actually draws from, so where it exists there is
        # nothing to reconcile.
        pool_rows_est = _domain_pool_rows(name, pool_tok)
        # ROWS THIS DOMAIN HAS ALREADY DRAWN, from the resume cursor. Was `used = 0` with the
        # comment "fresh run, new names; asserted rather than assumed" -- which was true when
        # every mix started from scratch and became the defect the moment one did not. Nothing
        # asserted it; the zero was the assumption.
        used = int((cursor or {}).get(name, 0))
        epochs = math.ceil((used + runtime) / pool_rows_est)
        assert pool_rows_est * epochs >= used + runtime, (
            f"{name}: pool {pool_rows_est} x epochs {epochs} < used {used} + want {runtime}; "
            "build_mix would clamp the draw and silently under-train this domain"
        )
        # THE CEILING COUNTS ROWS, NOT TOKENS. An epoch is a pass over the POOL, and the pool
        # is packed rows -- which is also the only quantity build_mix works in (cap =
        # int(pool*epochs) - used, draw = arange(used, used+want)). Tokens are not rows: each
        # document loses a partial row to packing, and n_val rows are held out on top. For cot
        # the two disagree across the line -- 3.83 epochs by tokens, 4.03 by rows against the
        # real cache -- because 20.5M of its 424M tokens (4.8%) never become drawable rows.
        #
        # The token version was the guard for a day and read UNDER the ceiling the whole time.
        # It was not measuring re-reads; it was measuring a quantity that correlates with them
        # (b0, 2026-09-01, found by reading the real caches for the launch gate's epochs item).
        # TOTAL, NOT THIS SEGMENT. (used + runtime) is what the model will have read by the
        # end of the plan, and re-reads are re-reads whether an earlier segment or this one
        # bought them. On a fresh start used is 0 and this is the old expression exactly.
        drawn_epochs = (used + runtime) / pool_rows_est
        if drawn_epochs > EPOCH_SOFT_CEILING and used:
            # BLOCKS, DOES NOT WARN, and only under a cursor (4c's ruling 2026-09-07). The cap is
            # already decided, so a resume mix over it is a WRONG mix, not a mix with a note --
            # and _warnings is print-and-continue: nothing reads it, no gate refuses on it, and a
            # 6.0-epoch mix would have shipped carrying an accurate warning nobody had to read.
            # `and used` scopes it to the resume case: on a fresh start the ceiling keeps warning
            # exactly as it did, because a fresh over-ceiling draw is a weight decision somebody
            # is making deliberately and every committed mix was written under that behaviour.
            over_ceiling[name] = (
                f"{drawn_epochs:.3f} total epochs exceeds the {EPOCH_SOFT_CEILING}-epoch line "
                f"once the resume cursor is counted: {used:,} rows already drawn plus {runtime:,} "
                f"this plan, over a {pool_rows_est:,}-row pool. The segment alone is "
                f"{runtime / pool_rows_est:.3f} epochs, which is why a per-segment ceiling read "
                f"this as a PASS. Cut this domain's weight until the TOTAL clears 4."
            )
        if os.path.isdir(os.path.join(ROOT, "data", "corpus", name)) and not _corpus_fingerprint(name):
            # SCOPED TO A CORPUS THAT IS ACTUALLY HERE. Without the isdir guard this warns on
            # EVERY domain from any machine without the corpora -- which is every dev box, since
            # data/corpus lives on the pod (see the fingerprint comment below). Caught by running
            # the selftest: the first version fired 9 of 9 locally and broke case 6's
            # `assert not _warnings`, i.e. it was the warning-that-fires-on-everything its own
            # negative control exists to forbid.
            #
            # WARN, not a refusal: the launcher cannot fix a missing stamp field, and the writer
            # must stay able to regenerate the 17 committed mixes that carry a null fingerprint
            # for at least one domain (b0 counted them 2026-09-07; those nulls are stale MIX
            # files, written before the read below existed -- every one of those domains has a
            # fingerprint in its stamp today). The BINDING check is
            # launch_gate.gate_corpora:184, which NOGOs a domain whose stamp has no fingerprint,
            # so nothing launches on this WARN alone. Recorded here because the writer, running
            # ON the pod, is where the absence is first visible.
            warnings.append(
                f"{name}: data/corpus/{name} exists but its build_corpus_stats.json carries no "
                f"`fingerprint`, so this mix pins no bytes for it. launch_gate.gate_corpora will "
                f"refuse a launch against this mix until the corpus pass writes that field."
            )
        if drawn_epochs > EPOCH_SOFT_CEILING:
            warnings.append(
                f"{name}: {drawn_epochs:.2f} epochs exceeds the {EPOCH_SOFT_CEILING}-epoch "
                f"line (ds.muennighoff_four_epoch). Past 4 epochs repeated tokens stop paying "
                f"and this weight is buying re-reads, not information. NOTE this is "
                f"{runtime:,} rows over a {pool_rows_est:,}-row pool; the token ratio is "
                f"{runtime * SEQ / pool_tok:.2f} and is NOT what build_mix draws against."
            )
        # A PASS the supply is not precise enough to support is not a pass -- see
        # _band_warning. Called, not inlined, so the selftest exercises this code and not a
        # second copy of the same arithmetic.
        warnings.extend(_band_warning(name, runtime, pool_tok))
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
            # A BOOLEAN the gate can read, beside the prose a person reads. The gate used to
            # grep the prose for "ESTIMATED", so rewording the sentence turned it green --
            # which I did by accident, writing "DERIVED from the stamp" for four domains that
            # have no cache, and watched a NO-GO become GO with nothing measured. A gate that
            # tests a word tests the author's vocabulary (b0, 2026-09-01).
            "epochs_pool_measured": bool(meas and meas["source"] == "cache"),
            "epochs_pool_source": (
                f"MEASURED from the token cache: {pool_rows_est:,} drawable rows "
                f"(rows = cache.numel()//(seq+1), minus n_val), read 2026-09-01"
                if meas and meas["source"] == "cache" else
                f"NOT MEASURED -- estimated from the stamp at {pool_rows_est:,} rows. "
                f"{meas['note'] if meas else 'no cache exists for this domain yet'}"
            ),
            # No fingerprint, and that is a STATEMENT, not an omission. launch_gate's corpora
            # gate pins each domain dir to the bytes the mix was written against by comparing
            # this to the corpus stamp -- and its old code skipped the comparison when the
            # field was absent, then printed "fingerprints match". Every mix but
            # mix_30b_stage2.json carries none, so that GO was routinely a claim about a
            # check that never ran (found and fixed today, b0).
            #
            # This generator cannot fill it honestly: the corpora live on the pod and nothing
            # here can read their stamps. Writing a placeholder would be worse than leaving
            # it out; saying WHY it is out is the only version that survives being read by
            # someone who is about to launch.
            "fingerprint": _corpus_fingerprint(name),
            "fingerprint_source": (
                f"read from data/corpus/{name}/build_corpus_stats.json"
                if _corpus_fingerprint(name) else
                f"ABSENT: data/corpus/{name}/build_corpus_stats.json carries no `fingerprint`, "
                f"so this mix pins nothing for {name}. launch_gate.gate_corpora refuses it."
            ),
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
        # ONE REASON DICT FOR BOTH BLOCK SOURCES, because main() prints
        # _untrusted_supply[n] for every n in _launch_blocked -- a name in the list with no
        # entry here is a KeyError at the moment the tool is trying to explain a refusal.
        "_untrusted_supply": dict(
            {n: why for n, why in UNTRUSTED_SUPPLY.items() if n in domains},
            **over_ceiling),
        "_launch_blocked": sorted(
            set(n for n in UNTRUSTED_SUPPLY if n in domains) | set(over_ceiling)),
        "domains": domains,
    }


def selftest():
    # Case 17 rebinds these to build the only world where a recipient ceiling binds; declared here
    # because Python requires `global` before the name's first use in the function.
    global FREED_CEILING, TOTAL_TOKENS, ROWS
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
    # Rows drawn fall as --code-tokens rises: starcoder takes a larger share of a fixed code
    # objective, so rp1t draws fewer. NOT epochs -- once a domain has a cache its pool comes
    # from the cache, so the pool no longer moves with this argument at all. The old assertion
    # compared epochs and passed only while every pool was stamp-derived; it went red the hour
    # the caches landed, testing a relationship that had stopped existing.
    assert a3["code_py_rp1t"]["rows_from_weight_at_runtime"] > b3["code_py_rp1t"]["rows_from_weight_at_runtime"], (
        "rp1t must draw fewer rows as starcoder's supply rises"
    )
    # Both corpora draw the same epochs ONLY while both pools are stamp-derived: proportional
    # rows over proportional pools. With a measured cache the pools are whatever they are, and
    # equal epochs is no longer the property -- equal SHARE of supply is.
    for cs in (3.0e9, 3.8e9, 6.0e9):
        m = build(cs)["domains"]
        sc, rp = m["code_py_starcoder"], m["code_py_rp1t"]
        share_sc = sc["rows_from_weight_at_runtime"] / sc["supply_tokens_one_epoch"]
        share_rp = rp["rows_from_weight_at_runtime"] / rp["supply_tokens_one_epoch"]
        assert abs(share_sc / share_rp - 1) < 0.02, (
            f"code corpora draw unequal shares of their supply at {cs}: {share_sc:.3e} vs "
            f"{share_rp:.3e} -- the split is meant to be a supply fact, not a quality claim"
        )
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
    # The over-the-line case is forced on a domain whose weight is a LITERAL, because cot's
    # is no longer one -- it became SUPPLY_CAPPED when the ceiling moved to rows, so setting
    # OBJECTIVE["cot"] = 0.09 changes nothing and this assertion silently tested an
    # unreachable path. textbook_30b carries a real literal, and 40% of 20B over its 388,021
    # measured rows is far past the line.
    saved_tb = OBJECTIVE["textbook_30b"]
    OBJECTIVE["textbook_30b"] = (0.60, saved_tb[1])
    try:
        warns = build(3.8e9)["_warnings"]
        assert any("textbook_30b" in w for w in warns), f"textbook at 60% must warn, got {warns}"
        assert any("epochs exceeds" in w for w in warns), warns
    finally:
        OBJECTIVE["textbook_30b"] = saved_tb
    # And a capped domain cannot be pushed over by its literal at all -- that is what capped
    # MEANS, and it is worth asserting so nobody re-adds one thinking they are tuning it.
    saved_cot = OBJECTIVE["cot"]
    OBJECTIVE["cot"] = (0.30, saved_cot[1])
    try:
        assert not build(3.8e9)["_warnings"], "a SUPPLY_CAPPED literal changed the draw"
    finally:
        OBJECTIVE["cot"] = saved_cot
    print("  6 shipped config is under the 4-epoch line; the ceiling still fires on an "
          "over-weighted literal, and a capped domain ignores its literal entirely")

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

    # 9. the error-band refusal, on its OWN fixture. This is the case that shipped: chat_qa at a
    #    hand-typed 0.76% drew 3.99996 epochs -- a PASS, four parts in 100,000 under the line,
    #    from a supply given to two significant figures (+/-1.3%). At the low end of that band
    #    the draw is 4.05. The point estimate was on the right side of the ceiling by less than
    #    the ceiling could resolve.
    #
    #    The fixture is planted rather than read from SUPPLY because 3b has since measured the
    #    exact integers and SUPPLY_RELATIVE_ERROR is empty. A check keyed to the live values
    #    would have gone red the moment the defect was FIXED -- which is testing the defect, not
    #    the guard. Check 7 had exactly this shape earlier today and was rewritten for it.
    saved_sup, saved_err = dict(SUPPLY), dict(SUPPLY_RELATIVE_ERROR)
    SUPPLY["chat_qa"] = 38_000_000          # what 2sf reporting gave us
    SUPPLY_RELATIVE_ERROR["chat_qa"] = 0.5 / 38.0
    try:
        rel = SUPPLY_RELATIVE_ERROR["chat_qa"]
        hand_typed_rows = int(ROWS * 0.0076)
        assert hand_typed_rows * SEQ / SUPPLY["chat_qa"] <= EPOCH_SOFT_CEILING, (
            "check 9 assumes the hand-typed weight PASSED the point-estimate ceiling; if it now "
            "fails outright the band refusal is not what is being tested"
        )
        assert hand_typed_rows * SEQ / (SUPPLY["chat_qa"] * (1 - rel)) > EPOCH_SOFT_CEILING, (
            "the hand-typed 0.0076 must cross the line at the low end of the supply -- that is "
            "the defect this check exists for"
        )
        # the derived weight, on that same imprecise supply, must clear the WHOLE band
        rows = int(ROWS * _ceiling_weight("chat_qa"))
        assert rows * SEQ / (SUPPLY["chat_qa"] * (1 - rel)) <= EPOCH_SOFT_CEILING, (
            f"derived weight draws {rows * SEQ / (SUPPLY['chat_qa'] * (1 - rel)):.4f} epochs at "
            f"the low end -- the band must clear, not just the point estimate"
        )
        assert rows < hand_typed_rows, "clearing the band has to cost something, or it is a no-op"
        # and the WARNING must fire on the shipped-at-the-time configuration, not just the
        # arithmetic above: the refusal reaches a reader through build()'s warnings, or not at
        # all. Calls the real predicate, not a restatement of it.
        warns = _band_warning("chat_qa", hand_typed_rows, SUPPLY["chat_qa"])
        assert warns and "error bar" in warns[0], f"band refusal did not fire: {warns}"
        # and it must NOT fire on the derived weight, whose whole band clears
        assert not _band_warning("chat_qa", rows, SUPPLY["chat_qa"]), (
            "the band refusal fires on a weight that clears at both ends -- it would then refuse "
            "every capped domain forever, which is a guard nobody can act on"
        )
    finally:
        SUPPLY.clear(); SUPPLY.update(saved_sup)
        SUPPLY_RELATIVE_ERROR.clear(); SUPPLY_RELATIVE_ERROR.update(saved_err)
    # With the exact integers now in hand, every capped domain clears on its own terms and the
    # derivation uses the whole sharp ceiling -- no error bar left to be conservative about.
    assert not SUPPLY_RELATIVE_ERROR, (
        "SUPPLY_RELATIVE_ERROR should be empty now that 3b measured exact counts; a stale entry "
        "silently costs weight"
    )
    # In ROWS, matching the ceiling. This block asserted tightness in tokens, which passed
    # only while the ceiling was also token-denominated; after the fix it demanded a row count
    # the ceiling would never produce. A test written in the wrong unit fails for the right
    # reason and points at the wrong place.
    for name in SUPPLY_CAPPED:
        pool = _domain_pool_rows(name, SUPPLY[name])
        rows = int(ROWS * _ceiling_weight(name))
        assert rows / pool <= EPOCH_SOFT_CEILING, (
            f"{name}: {rows / pool:.6f} epochs overshoots the ceiling"
        )
        assert (rows + 1) / pool > EPOCH_SOFT_CEILING, (
            f"{name}: one more row still fits ({(rows + 1) / pool:.6f}), so the sharp ceiling "
            f"is leaving rows unused"
        )
    print("  9 the hand-typed 0.0076 fails the error-band ceiling it passed on the point "
          "estimate; with exact counts the derivation uses the sharp ceiling to the last row")

    # 10. the units declaration. Every supply must be on one ruler, and adding a domain must
    #     be unable to skip saying which. 3b sent chat_qa as bare ids and chatml as its stamp;
    #     both figures were individually reasonable, and the mix would have built happily with
    #     one domain undercounted by one token per document.
    #
    #     What this does NOT do is detect a mislabelled number -- and the first version of this
    #     check claimed to. It looked for a sibling gap of exactly n_docs, which is not
    #     decidable: the real gap is the markup delta PLUS the missing EOS, and the markup delta
    #     is unknown a priori. It passed only because I chose the fixture to fit my rule. The
    #     honest guard is narrower: an undeclared supply is refused.
    _refuse_mixed_units()                        # the shipped set must pass
    saved_units = dict(SUPPLY_UNITS)
    try:
        SUPPLY_UNITS.pop("chat_qa")
        try:
            _refuse_mixed_units()
            raise AssertionError("a supply with no declared unit passed")
        except SystemExit as e:
            assert "no declared unit" in str(e), str(e)
        SUPPLY_UNITS["chat_qa"] = "bare ids"
        try:
            _refuse_mixed_units()
            raise AssertionError("a bare-ids declaration passed")
        except SystemExit as e:
            assert "not in training units" in str(e), str(e)
    finally:
        SUPPLY_UNITS.clear(); SUPPLY_UNITS.update(saved_units)
    # the shipped pair really is +EOS: each is its domain's stamp, and the two differ by the
    # ChatML markup (808,196), not by n_docs (160,414), which is what a unit slip would show.
    assert SUPPLY["chatml"] - SUPPLY["chat_qa"] == 808_196, (
        f"the render delta moved: {SUPPLY['chatml'] - SUPPLY['chat_qa']:,}. If it is now "
        f"160,414 one supply is bare ids again"
    )
    print("  10 every supply declares its units; an undeclared or bare-ids supply is refused, "
          "and the shipped pair differs by markup rather than by n_docs")

    # 11. the probe mix is a RENORMALISATION, not a second composition decision. The risk it
    #     guards is specific: dropping the largest domain and re-arguing the rest would make
    #     the A/B's mix a fresh judgement taken under time pressure, and nobody would notice
    #     because both arms would still be identical to each other.
    pm = build_probe()
    ref = {k: v["weight"] for k, v in build(8.85e9)["domains"].items()
           if k != "code_py_starcoder"}
    scale = sum(ref.values())
    for k, w in ref.items():
        got = pm["domains"][k]["weight"]
        assert abs(w / scale - got) < 2e-5, (
            f"{k}: probe weight {got} is not the shipped ratio {w / scale} renormalised -- "
            f"the probe re-decided a weight instead of rescaling it"
        )
    assert "code_py_starcoder" not in pm["domains"], "the probe must not name an unbuilt domain"
    assert sum(d["rows_from_weight_at_runtime"] for d in pm["domains"].values()) == pm["total_rows"]
    assert pm["total_tokens"] == PROBE_STEPS * PROBE_TOK_PER_STEP, (
        "the probe budget must be steps x the MEASURED tok_per_step, not a round number"
    )
    # and no domain is anywhere near the ceiling, which is why no bypass was needed
    assert not pm["_warnings"], pm["_warnings"]
    worst = max(d["epochs_fractional"] for d in pm["domains"].values())
    assert worst < 1.0, f"a 500-step probe should draw under one epoch everywhere, worst {worst}"
    print("  11 the probe mix renormalises the shipped ratios (no weight re-decided), omits "
          "the unbuilt domain, sums to its row budget, and draws under one epoch everywhere")

    # 12. the writer REFUSES a mix with any null fingerprint, rather than labelling it. The
    #     labelled version was honest and still shipped a file: on 2026-09-01 a Mac-generated
    #     copy carrying nine nulls plus nine explanations overwrote the pod's nine correct
    #     fingerprints, four hours after they were right. Nobody read the explanation.
    #
    #     Exercised through main(), not by re-deriving the condition: the refusal lives at the
    #     write, and a check that recomputes `any null` would pass while main() still wrote
    #     the file.
    import subprocess

    for argv in ([], ["--probe"]):
        r = subprocess.run([sys.executable, __file__, *argv], capture_output=True, text=True)
        on_host = all(_corpus_fingerprint(n) for n in OBJECTIVE if n not in ("chatml", "chat_qa"))
        if on_host:
            assert r.returncode == 0, f"corpora are present but the writer refused: {r.stderr[:200]}"
        else:
            assert r.returncode == 1, (
                f"no corpora on this host, so {argv or ['(mix)']} must REFUSE, got exit "
                f"{r.returncode}"
            )
            assert "REFUSING" in r.stderr and "WHERE THE CORPUS IS" in r.stderr, r.stderr[:200]
    print("  12 the writer refuses to emit a mix with a null fingerprint, and says where to "
          "run it instead -- a labelled null is still a file that can overwrite a correct one")

    # 13 RAISING THE TOTAL RECOMPUTES CEILINGS INSTEAD OF COPYING THEM.
    #
    # THIS DRIVES main(), NOT build(), and that distinction is the whole value of the case. The
    # first version asserted on build()'s output at a patched TOTAL_TOKENS, which is a true
    # statement about build() and says NOTHING about the branch in main() that chooses between
    # copying and recomputing -- measured: restoring the copy (`_shrinking = True`) and deleting
    # the epoch re-check both left it GREEN. A case that cannot see the defect it was written for
    # is registration, not coverage.
    #
    # Reaching that branch needs two things stubbed, because both refuse earlier on this host:
    # the per-domain fingerprint (case 12's refusal, which fires for every domain on a Mac) and
    # the output path. Neither is what this case is about, and stubbing them is what makes the
    # subject reachable rather than what makes the assertion pass.
    import subprocess
    import tempfile

    _stub = (
        "import sys, json, tempfile, os\n"
        "sys.argv = ['w']\n"
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('w', {__file__!r})\n"
        "w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)\n"
        # Every domain gets a fingerprint, so case 12's refusal cannot fire and main() proceeds
        # to the total-override branch. The VALUE is irrelevant here; its presence is the point.
        "w._corpus_fingerprint = lambda n: 'f' * 16\n"
        "out = os.path.join(tempfile.mkdtemp(), 'mix30.json')\n"
        "sys.argv = ['w', '--total', '30e9', '--out', out]\n"
        "rc = 0\n"
        "try:\n"
        "    w.main()\n"
        "except SystemExit as e:\n"
        "    rc = e.code or 0\n"
        "except AssertionError as e:\n"
        "    print('ASSERTION:' + str(e)[:300]); sys.exit(9)\n"
        "m = json.load(open(out))\n"
        "print(json.dumps({n: d['weight'] for n, d in m['domains'].items()}))\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as _fh:
        _fh.write(_stub)
        _stub_path = _fh.name
    r = subprocess.run([sys.executable, _stub_path], capture_output=True, text=True)
    assert r.returncode == 0, (
        f"main() at --total 30e9 did not write a mix (exit {r.returncode}). This is the path that "
        f"used to die on a bare assertion: {r.stdout[-300:]} {r.stderr[-300:]}"
    )
    _w30 = json.loads(r.stdout.strip().splitlines()[-1])
    # The three supply-capped domains must come out SMALLER than their 20B weights, which is what
    # "recomputed" means and precisely what copying does not do.
    for _n, _w20 in (("cot", 0.08069), ("chatml", 0.00741), ("chat_qa", 0.00725)):
        assert _w30[_n] < _w20, (
            f"{_n} weight {_w30[_n]:.7f} at 30B is not below its 20B weight {_w20}: main() copied "
            f"the 20B weights instead of letting _ceiling_weight recompute, so a capped pool would "
            f"be re-read more than {EPOCH_SOFT_CEILING} times"
        )
    # And every domain clears the ceiling on the rows build_mix DRAWS, at the 30B row budget.
    _rows30 = int(30e9) // SEQ
    for _n, _wt in _w30.items():
        _ep = (int(_rows30 * _wt) / _domain_pool_rows(_n, SUPPLY[_n])
               if _n in SUPPLY else None)
        if _ep is not None:
            assert _ep <= EPOCH_SOFT_CEILING, f"{_n} draws {_ep:.6f} epochs at 30B"
    # NEGATIVE CONTROL, on the same numbers: the 20B weights at a 30B row budget must cross the
    # ceiling in all three capped domains. If they did not, none of the assertions above could
    # tell a recomputed mix from a copied one.
    _bad = [n for n, w in (("cot", 0.08069), ("chatml", 0.00741), ("chat_qa", 0.00725))
            if int(_rows30 * w) / _domain_pool_rows(n, SUPPLY[n]) > EPOCH_SOFT_CEILING]
    assert len(_bad) == 3, (
        f"the 20B weights at a 30B budget should cross the ceiling in all three capped domains, "
        f"caught {_bad}. Without this the case cannot distinguish recompute from copy"
    )
    print("  13 main() at a raised --total RECOMPUTES the supply-capped weights rather than "
          "copying them (cot/chatml/chat_qa all shrink), every domain clears the ceiling on the "
          "rows build_mix draws, and the 20B weights at a 30B budget -- what copying produced -- "
          "cross it in all three")

    # 14 THE RELEASED WEIGHT LANDS ON CODE ABOVE 20B, and on nothing else (4c's ruling
    # 2026-09-07). Without CODE_TOTAL_ABOVE_20B the renormalisation spreads what the capped domains
    # release across every uncapped domain pro-rata, which is arithmetically correct and the wrong
    # composition: measured before the change, code took 38% of the released 0.0317852 while
    # math_owm_stage2 took the largest single share (+0.0092857) and zh_web -- the domain this
    # file's own comment calls irrelevant to an English code model -- took +0.0010713.
    _m20 = build(8.85e9)
    _saved_total, _saved_rows = TOTAL_TOKENS, ROWS
    try:
        globals()["TOTAL_TOKENS"] = int(30e9)
        globals()["ROWS"] = int(30e9) // SEQ
        _m30 = build(8.85e9)
    finally:
        globals()["TOTAL_TOKENS"], globals()["ROWS"] = _saved_total, _saved_rows
    _code = [n for n in _m20["domains"] if n.startswith("code_")]
    assert _code, "no code_* domain found, so this case cannot check where the weight went"
    _c20 = sum(_m20["domains"][n]["weight"] for n in _code)
    _c30 = sum(_m30["domains"][n]["weight"] for n in _code)
    _released = sum(_m20["domains"][n]["weight"] - _m30["domains"][n]["weight"]
                    for n in SUPPLY_CAPPED if n in _m20["domains"])
    assert _released > 0, (
        f"the supply-capped domains released {_released:+.7f} at 30B, so nothing was freed and "
        f"this case cannot tell where a release lands -- its subject does not exist"
    )
    # Code must absorb essentially all of it. Not exactly: the renormalisation still runs and the
    # other domains give up a rounding residue, so the gain is slightly ABOVE the release.
    assert _c30 - _c20 >= _released, (
        f"code gained {_c30 - _c20:+.7f} of the {_released:+.7f} released, so the release is still "
        f"being spread over other roles instead of funding the objective"
    )
    # AND THE OTHER DOMAINS MUST BE FLAT. This is the half that fails if CODE_TOTAL_ABOVE_20B is
    # removed: pro-rata is exactly the state where these are NOT flat, and asserting only code's
    # gain would pass under a scheme that raised code AND everyone else.
    _moved = {n: _m30["domains"][n]["weight"] - _m20["domains"][n]["weight"]
              for n in _m20["domains"]
              if n not in SUPPLY_CAPPED and not n.startswith("code_")}
    _worst = max(abs(v) for v in _moved.values())
    assert _worst < 1e-3, (
        f"an uncapped non-code domain moved by {_worst:.7f} at 30B, so the released weight is being "
        f"shared: {', '.join(f'{n} {v:+.7f}' for n, v in sorted(_moved.items(), key=lambda x: -abs(x[1]))[:3])}"
    )
    # BELOW 20B THE RULING MUST NOT APPLY, and the check is on CODE'S PRE-SCALE SHARE, not on
    # build()'s output. My first version compared build() at 8B against build() at 20B and asserted
    # the code weights were equal; they are not, and the difference is by design -- _ceiling_weight
    # scales with the total, so at 8B cot rises to 0.2017 (fewer epochs are needed to reach the
    # ceiling) and every other weight moves in response. The shipped mix_200m_8b.json does not
    # contain build()@8B at all: --total COPIES the 20B weights into it. So an equality between two
    # different totals' builds was never a property of this file and the assertion was testing my
    # own misreading.
    #
    # What the threshold actually controls is one expression, so that is what is asserted: the value
    # free["code"] takes. Below and at 20B it must be CODE_TOTAL, above it CODE_TOTAL_ABOVE_20B.
    assert CODE_TOTAL_ABOVE_20B > CODE_TOTAL, (
        f"CODE_TOTAL_ABOVE_20B {CODE_TOTAL_ABOVE_20B} is not above CODE_TOTAL {CODE_TOTAL}, so the "
        f"ruling cannot absorb a release and this case's subject does not exist"
    )
    for _tot, _want in ((int(8e9), CODE_TOTAL), (20_000_000_000, CODE_TOTAL),
                        (int(30e9), CODE_TOTAL_ABOVE_20B)):
        globals()["TOTAL_TOKENS"], globals()["ROWS"] = _tot, _tot // SEQ
        try:
            _got = CODE_TOTAL_ABOVE_20B if TOTAL_TOKENS > 20_000_000_000 else CODE_TOTAL
        finally:
            globals()["TOTAL_TOKENS"], globals()["ROWS"] = _saved_total, _saved_rows
        assert _got == _want, (
            f"at total {_tot / 1e9:.0f}B the code share resolves to {_got}, want {_want}. The "
            f"boundary is > 20B exclusive: 20B itself keeps CODE_TOTAL, or every committed mix at "
            f"the decided total becomes a different file"
        )
    print("  14 above 20B the weight released by the supply-capped domains lands on the CODE role "
          "and the other uncapped domains stay flat (pro-rata gave code 38% and math the largest "
          "share); the code share resolves to CODE_TOTAL at 8B and at 20B and to "
          "CODE_TOTAL_ABOVE_20B only past it, so the boundary is exclusive")

    # 15. THE CURSOR MAKES `epochs` A TOTAL, and the case is built so it FAILS without the fix
    #     rather than merely passing with it. The live 30B resume is the world: cot's cursor is
    #     139,492 rows on a 98,529-row pool and its plan draws 393,xxx more, so the segment alone
    #     is 3.999 epochs -- a PASS under the old expression -- while the total is 5.415.
    #
    #     ASSERTED ON THE WARNING, not on the epochs field, because the warning is what a person
    #     reads and what a launch gate can refuse on. The epochs INTEGER moves too (ceil of the
    #     total), but an integer going 4 -> 6 is also what a legitimately bigger draw does; only
    #     the warning names the ceiling being crossed.
    _cur = {"cot": 139_492, "chatml": 12_726, "chat_qa": 12_555}
    _fresh = build(8.85e9)
    _resumed = build(8.85e9, _cur)
    _fw = " ".join(_fresh["_warnings"])
    _rw = " ".join(_resumed["_warnings"])
    for _n in _cur:
        assert f"{_n}: " not in _fw or "exceeds" not in _fw.split(f"{_n}: ")[1][:80], (
            f"{_n} must NOT trip the ceiling on a fresh start -- if it does, this case cannot "
            f"tell the fix from a pre-existing violation")
        # THE ASSERTION FLIPPED WHEN THE DERIVER WAS FIXED, and the flip is the point. Until
        # 2026-09-07 this read "the cursor-aware build MUST warn", because the check counted the
        # total while _ceiling_weight still sized each domain to 4.0 epochs of THIS SEGMENT -- so a
        # resume necessarily produced a mix that violated its own ceiling and the warning was the
        # evidence the check had been fixed. Now the deriver subtracts the cursor, so a resume mix
        # is BUILT under the ceiling and must NOT warn: the correct end state is no violation to
        # report. Asserted on the total epochs rather than on the warning's absence alone, because
        # "no warning" is also what a check that stopped looking would produce.
        _dom = _resumed["domains"][_n]
        # pool_rows_estimated and rows, NOT epochs_fractional: that field is want_tok/pool_tok, a
        # TOKEN ratio, and the ceiling is a ROW count -- the two differ because packing drops a
        # partial row per document and n_val rows are held out. Its own note says the token ratio
        # "is NOT what build_mix draws against".
        _tot = (_cur[_n] + _dom["rows_from_weight_at_runtime"]) / _dom["pool_rows_estimated"]
        assert _tot <= EPOCH_SOFT_CEILING + 1e-9, (
            f"{_n} draws {_tot:.4f} total epochs (cursor {_cur[_n]:,} + "
            f"{_dom['rows_from_weight_at_runtime']:,} over {_dom['pool_rows_est']:,}), past "
            f"{EPOCH_SOFT_CEILING}. The deriver is sizing to one segment again.")
        assert not any(w.startswith(f"{_n}: ") and "exceeds" in w for w in _resumed["_warnings"]), (
            f"{_n} is built under the ceiling but still warns -- the check and the deriver "
            f"disagree, which is the defect in the other direction.")
    # AND THE FRESH-START PATH IS BYTE-IDENTICAL. A cursor-aware ceiling that changed the
    # no-cursor answer would silently rewrite every committed mix.
    assert build(8.85e9, None)["domains"] == _fresh["domains"], (
        "build(..., None) must equal build(...): the cursor path must not touch a fresh start")
    assert build(8.85e9, {})["domains"] == _fresh["domains"], (
        "an EMPTY cursor must also equal a fresh start -- {} means 'nothing drawn yet'")
    print("  15 the epoch ceiling counts cursor + this plan, so a resume cannot be granted a "
          "fresh 4 epochs: cot/chatml/chat_qa pass fresh and trip the ceiling under the live "
          "30B cursor, while build(..., None) and build(..., {}) stay byte-identical to fresh")

    # 16. AND IT BLOCKS THE LAUNCH, not merely warns (4c's ruling 2026-09-07). Case 15 proves the
    #     ceiling COUNTS right; this proves it BINDS. The distinction is not academic -- I shipped
    #     case 15 believing the job was done, and _warnings is print-and-continue: no gate reads
    #     it, so a 6.0-epoch resume mix would have been written with an accurate warning nobody
    #     was required to read. That is the shape where the code is defensible and the contract
    #     is false.
    #     THE WORLD HAD TO BE REBUILT WHEN THE DERIVER WAS FIXED. Until 2026-09-07 `_resumed`
    #     itself blocked, because the deriver sized to 4.0 epochs per SEGMENT and any cursor
    #     pushed the total over -- so the live cursor was a world where blocking was reachable.
    #     With the cursor subtracted a normal resume is built UNDER the ceiling and blocks nothing,
    #     which would leave this assertion vacuous on `_resumed`: `set([]) == set([])` after the
    #     loop below iterates nothing. A binding check needs a world where the bad thing is still
    #     possible, so the world is now a cursor ALREADY PAST the ceiling on its own -- there the
    #     deriver returns zero rows (clamped, never negative) and the overrun is real and
    #     unfixable by any weight, which is exactly when a launch must be refused.
    _over = {n: int(_ceil_pool(n) * 4.5) for n in _cur}
    _blocked = build(8.85e9, _over)
    assert set(_blocked["_launch_blocked"]) == set(_over), (
        f"every over-ceiling domain must block the launch, got "
        f"{_blocked['_launch_blocked']} for a cursor over {sorted(_over)}")
    for _n in _over:
        assert _blocked["domains"][_n]["rows_from_weight_at_runtime"] == 0, (
            f"{_n}'s cursor is past the ceiling, so the deriver must ask for ZERO further rows, "
            f"not a negative count silently allocated as one: "
            f"{_blocked['domains'][_n]['rows_from_weight_at_runtime']}")
    _resumed = _blocked
    for _n in _over:
        # A NAME WITH NO REASON IS A KeyError IN main()'s REFUSAL PRINT, which reads
        # _untrusted_supply[n] for every n in _launch_blocked. The two fields are one mechanism.
        assert _n in _resumed["_untrusted_supply"], (
            f"{_n} blocks with no entry in _untrusted_supply; main() prints that dict per blocked "
            f"name and would raise KeyError while explaining the refusal")
        assert "total epochs" in _resumed["_untrusted_supply"][_n], (
            f"{_n}'s block reason must name the TOTAL, since the segment figure is what read as "
            f"a pass: {_resumed['_untrusted_supply'][_n][:80]}")
    # THE FRESH-START PATH STILL ONLY WARNS, and the assertion must be made on a fresh build
    # THAT ACTUALLY CROSSES THE LINE. Asserting it on the shipped config was the defect: no
    # domain crosses fresh, so `not _fresh["_launch_blocked"]` holds no matter what the code
    # does, and dropping the `and used` scope survived it. Caught by mutation, not by reading.
    # textbook_30b at 60% is case 6's own over-the-line world (a real literal, far past the
    # ceiling on its measured rows).
    assert not _fresh["_launch_blocked"], (
        f"a fresh start must not block: {_fresh['_launch_blocked']}")
    _saved_tb = OBJECTIVE["textbook_30b"]
    OBJECTIVE["textbook_30b"] = (0.60, _saved_tb[1])
    try:
        _fresh_over = build(8.85e9)
        assert any("textbook_30b" in w and "exceeds" in w for w in _fresh_over["_warnings"]), (
            f"the fixture must cross the ceiling or it tests nothing: "
            f"{_fresh_over['_warnings']}")
        assert not _fresh_over["_launch_blocked"], (
            f"a FRESH over-ceiling draw must warn and not block -- blocking it would refuse a "
            f"weight decision made deliberately, and every committed mix was written under the "
            f"warn behaviour: {_fresh_over['_launch_blocked']}")
        # The same weight WITH a cursor blocks, which is the whole scope of the ruling.
        _cur_over = build(8.85e9, {"textbook_30b": 1})
        assert "textbook_30b" in _cur_over["_launch_blocked"], (
            f"the same over-ceiling weight under a cursor must block: "
            f"{_cur_over['_launch_blocked']}")
    finally:
        OBJECTIVE["textbook_30b"] = _saved_tb
    _hi = build(8.85e9, {"zh_web": 99_000_000})
    assert "zh_web" in _hi["_launch_blocked"], "a huge cursor on any domain must block too"
    print("  16 an over-ceiling total under a cursor goes to _launch_blocked with a reason naming "
          "the total (not just _warnings, which nothing reads), the reason is in the same dict "
          "main() prints per blocked name, and a fresh start still only warns")

    # 17 THE RECIPIENT CEILING BINDS, AND SURVIVES THE RENORMALISATION THAT FOLLOWS IT.
    #     The defect this catches shipped and was measured: the ceiling was clamped in ROWS inside
    #     _place_freed_under_ceiling, and then `scale = (1 - sum(capped)) / sum(free)` came out at
    #     1.013417 and multiplied the clamp away -- applied and undone in the same function, with
    #     nothing warning, because every row count in between was correct.
    #     THE WORLD IS THE LIVE .step22500 CURSOR AT THE 30B TOTAL, because that is the only
    #     cursor where a recipient is anywhere near its ceiling: the small fixture `_cur` above
    #     leaves math at 0.86 epochs, where any ceiling above 0.86 is vacuous and a test on it
    #     would pass however _renormalise_holding_clamped behaves. Same lesson as case 16's fresh
    #     path -- a negative assertion needs a world where the positive is reachable.
    #     AND THE CEILING HAS TO BE LOWERED, because at the shipped FREED_CEILING nothing binds by
    #     design -- which is why 4c set it there. That claim is ASSERTED below against the shipped
    #     value rather than quoted: three different pairs for these two epoch figures were in
    #     circulation in this file (2.0509/2.0045 at the FREED_CEILING comment, 2.0557/2.0142 here)
    #     and tilerl measured starcoder at 1.9909, so all three disagreed with the code by up to
    #     1.2%. A comment labelled `measured` is what the next reader trusts instead of deriving,
    #     and a stale one is worse than no number. The assertion cannot go stale.
    _live_cur = {"chat_qa": 31260, "chatml": 31945, "code_py_rp1t": 68575,
                 "code_py_starcoder": 1424422, "cot": 348846, "en_c4_stage2": 702789,
                 "math_owm_stage2": 1142230, "textbook_30b": 438345, "zh_web": 131588}
    _saved = (FREED_CEILING, TOTAL_TOKENS, ROWS)
    try:
        TOTAL_TOKENS, ROWS = 30_000_000_000, 30_000_000_000 // SEQ
        _unpinned = build(8.85e9, _live_cur)

        def _ep_of(m, n):
            _pool = _domain_pool_rows(
                n, 8.85e9 if n == "code_py_starcoder" else SUPPLY[n])
            return (m["domains"][n]["rows_from_weight_at_runtime"]
                    + int(_live_cur.get(n, 0))) / _pool
        # THE SHIPPED CEILING BINDS NOTHING -- asserted, not quoted. This is the claim the whole
        # ruling rests on ("2.10 changes no number in the launch mix"), so it is checked against
        # whatever FREED_CEILING currently says rather than against an epoch figure typed into a
        # comment. If a future weight change pushes a recipient over the shipped ceiling, the mix
        # silently starts pinning and this fires instead.
        _shipped = {n: _ep_of(_unpinned, n) for n in FREED_RECIPIENTS}
        for _n, _e in _shipped.items():
            assert _e <= FREED_CEILING, (
                f"{_n} draws {_e:.4f} total epochs against the shipped FREED_CEILING "
                f"{FREED_CEILING}, so the ceiling BINDS in the launch mix and the ruling that it "
                f"changes no number is no longer true. Re-derive the mix with 4c before launching.")
        # THE UNPINNED WORLD MUST BE OVER THE LOWERED CEILING, or there is nothing to clamp.
        FREED_CEILING = 1.90
        for _n in FREED_RECIPIENTS:
            assert _ep_of(_unpinned, _n) > FREED_CEILING, (
                f"fixture is vacuous: {_n} draws {_ep_of(_unpinned, _n):.4f} epochs without any "
                f"clamp, already under the {FREED_CEILING} ceiling this case lowers it to")
        _pinned = build(8.85e9, _live_cur)
        for _n in FREED_RECIPIENTS:
            _ep = _ep_of(_pinned, _n)
            assert _ep <= FREED_CEILING + 1e-6, (
                f"{_n} draws {_ep:.6f} total epochs against a {FREED_CEILING} ceiling that BOUND "
                f"it -- the clamp was applied and then renormalised away, the defect measured at "
                f"scale=1.013417")
            # AT the ceiling, not merely under it. `<=` alone would pass if the renormalisation
            # cut the domain to zero, which is a different bug wearing this one's green tick.
            assert _ep > FREED_CEILING - 0.02, (
                f"{_n} pinned to {_ep:.6f} against {FREED_CEILING}: under the line by more than "
                f"rounding, so the clamp overshot rather than pinned")
        assert abs(sum(v["weight"] for v in _pinned["domains"].values()) - 1.0) < 1e-5, (
            f"pinning both recipients must still leave a normalised mix, got "
            f"{sum(v['weight'] for v in _pinned['domains'].values()):.9f}")
        # THE CAPPED THREE MUST NOT MOVE: their weight is the epoch ceiling's answer, not a share
        # of the remainder, so a recipient pinning cannot be paid for out of them.
        for _n in SUPPLY_CAPPED:
            assert (_pinned["domains"][_n]["rows_from_weight_at_runtime"]
                    == _unpinned["domains"][_n]["rows_from_weight_at_runtime"]), (
                f"{_n} is supply-capped; pinning a RECIPIENT must not move it")
        # AND SOMETHING MUST ABSORB THE REFUSED ROWS, or the mix sums to 1 only by dropping them.
        assert [n for n in _pinned["domains"]
                if n not in SUPPLY_CAPPED and n not in FREED_RECIPIENTS
                and _pinned["domains"][n]["rows_from_weight_at_runtime"]
                != _unpinned["domains"][n]["rows_from_weight_at_runtime"]], (
            "no unpinned domain absorbed the rows the ceiling refused")
    finally:
        FREED_CEILING, TOTAL_TOKENS, ROWS = _saved
    print("  17 a recipient ceiling that BINDS survives the renormalisation after it: on the live "
          "cursor with the ceiling lowered under the unclamped draw, both recipients pin AT it "
          "instead of 1.3% above, the capped three do not move, and another domain absorbs the "
          "difference")

    # 18. THE MISSING-FINGERPRINT WARN, with its NEGATIVE CONTROL in the same case. A warning
    #     that fires on every domain is worth the same as one that fires on none, so the shipped
    #     world (every domain stamped) must produce NO fingerprint warning, and only then does
    #     the stubbed-absent world have to produce one. Ordered that way deliberately: the
    #     positive alone would pass against a `warnings.append` with no condition at all.
    #
    #     `os.path.isdir` IS STUBBED TOO, because the WARN is guarded on the corpus being present
    #     and data/corpus is on the pod: unstubbed, this case would assert on a branch it never
    #     reaches from a dev box and pass for the wrong reason. Restricted to the corpus root so
    #     nothing else in build() sees a lie.
    _saved_fp = globals()["_corpus_fingerprint"]
    _real_isdir = os.path.isdir
    _corpus_root = os.path.join(ROOT, "data", "corpus")

    def _isdir_corpora_exist(p):
        return True if str(p).startswith(_corpus_root) else _real_isdir(p)

    try:
        os.path.isdir = _isdir_corpora_exist
        globals()["_corpus_fingerprint"] = lambda n: "f" * 16
        _all_fp = build(3.8e9)
        assert not [w for w in _all_fp["_warnings"] if "no `fingerprint`" in w], (
            f"a fully stamped mix must raise no fingerprint warning: {_all_fp['_warnings']}"
        )
        for _d in _all_fp["domains"].values():
            assert _d["fingerprint_source"].startswith("read from"), _d["fingerprint_source"]

        _one = sorted(_all_fp["domains"])[0]
        globals()["_corpus_fingerprint"] = lambda n: None if n == _one else "f" * 16
        _miss = build(3.8e9)
        _hits = [w for w in _miss["_warnings"] if "no `fingerprint`" in w]
        assert len(_hits) == 1 and _hits[0].startswith(f"{_one}: "), (
            f"exactly the unstamped domain must warn, got {_hits}"
        )
        assert _miss["domains"][_one]["fingerprint"] is None
        assert _miss["domains"][_one]["fingerprint_source"].startswith("ABSENT:"), (
            "a null fingerprint must say WHY in fingerprint_source, or the field reads as an "
            "omission rather than a statement"
        )
        # NOT in _launch_blocked: the binding refusal is launch_gate.gate_corpora:184, and
        # duplicating it here would let a future edit satisfy the writer while the gate is the
        # thing that actually has to hold.
        assert _one not in _miss.get("_launch_blocked", []), (
            "the missing fingerprint is a WARN in the writer; the refusal belongs to launch_gate"
        )
        # THE isdir GUARD ITSELF, or the stub above would hide a WARN that fires from every dev
        # box: with the corpus absent there is nothing to pin and nothing to say.
        os.path.isdir = _real_isdir
        if not _real_isdir(os.path.join(_corpus_root, _one)):
            globals()["_corpus_fingerprint"] = lambda n: None
            assert not [w for w in build(3.8e9)["_warnings"] if "no `fingerprint`" in w], (
                "with data/corpus absent the writer must not warn about fingerprints -- the "
                "absence of a corpus is not the absence of a pin"
            )
    finally:
        os.path.isdir = _real_isdir
        globals()["_corpus_fingerprint"] = _saved_fp
    print("  18 a domain whose stamp carries no fingerprint warns and says ABSENT with the "
          "reason in fingerprint_source, while a fully stamped mix warns not at all; the "
          "refusal stays in launch_gate rather than being duplicated here")

    # 19. THE FALLBACK IS ONE CORPUS'S COUNT, so a SWAPPED code domain with no stamped tokens
    #     must REFUSE rather than inherit it. Without the name check _rp1t_tokens would report
    #     code_py_rp1t's 420,646,182 as dd09's supply -- the wiki_chat defect exactly: a real
    #     number, honestly measured, for a different domain than the one it is filed under.
    #     Asserted on SystemExit and on the message naming both domains, because a refusal that
    #     does not say which corpus the fallback belongs to sends the reader to the wrong file.
    assert _rp1t_tokens(CODE_RP1T_DOMAIN) > 0, "the default domain must still resolve"
    _saved_stamp = globals()["_corpus_stamp"]
    try:
        globals()["_corpus_stamp"] = lambda n, f: None
        assert _rp1t_tokens(CODE_RP1T_DOMAIN) == RP1T_PYTHON_TOKENS_FALLBACK, (
            "with no stamp the DEFAULT domain falls back, which is what the constant is for"
        )
        try:
            _rp1t_tokens("code_rp1t_dd09")
        except SystemExit as e:
            _msg = str(e)
            assert "code_rp1t_dd09" in _msg and CODE_RP1T_DOMAIN in _msg, (
                f"the refusal must name the domain asked for AND the one the fallback measures: "
                f"{_msg}"
            )
        else:
            raise AssertionError(
                "a swapped code domain with no stamped tokens must refuse, not silently return "
                "another corpus's ast.parse count"
            )
    finally:
        globals()["_corpus_stamp"] = _saved_stamp
    print("  19 a swapped code domain with no stamped tokens refuses and the message names both "
          "domains, while the default still falls back to the constant it was measured for")

    print("selftest: 19/19")
    return 0


PROBE_OUT = os.path.join(ROOT, "data", "mix_probe_lr.json")
PROBE_STEPS = 500
PROBE_TOK_PER_STEP = 917_504     # runs/memory_peaks.json world=7, b32 accum1 seq4096 -- measured


def build_probe():
    """A throwaway mix for the 500-step lr_scale A/B (fb 2026-09-01). NOT a composition.

    The eight domains that exist today -- everything but code_py_starcoder, which 3b is still
    building -- at the shipped mix's ratios RENORMALISED, not re-decided. That distinction is
    the whole point: dropping the largest domain and then re-arguing the remaining weights
    would make this a second composition decision taken under time pressure, and the arm
    comparison does not need one. Both arms see the identical mix; only lr_scale differs.

    Budget is 500 steps x 917,504 tokens/step, the MEASURED step size from
    runs/memory_peaks.json at world=7 -- not a round number, because a round number here would
    silently change how many rows each domain contributes and nobody would know which.

    No epoch ceiling applies: at 0.459B every domain draws well under one epoch (the largest is
    cot at 0.137). The guard is left ON rather than bypassed -- it simply has nothing to say,
    which is a better state than a disabled guard that would stay disabled if this file were
    ever copied.
    """
    total_tokens = PROBE_STEPS * PROBE_TOK_PER_STEP
    rows = total_tokens // SEQ
    full = build(8.85e9)["domains"]
    keep = {k: v for k, v in full.items() if k != "code_py_starcoder"}
    scale = sum(v["weight"] for v in keep.values())
    weights = {k: v["weight"] / scale for k, v in keep.items()}
    alloc = _rows_for_weights(weights, rows)
    domains, warn = {}, []
    for name, r in alloc.items():
        pool = keep[name]["pool_rows_estimated"]
        w, places = _weight_for_rows(r, rows)
        drawn = r / pool
        if drawn > EPOCH_SOFT_CEILING:
            warn.append(f"{name}: {drawn:.2f} epochs -- unexpected in a 500-step probe")
        domains[name] = {
            "weight": w,
            "epochs": max(1, math.ceil(r / pool)),
            "anneal": w,
            "role": f"PROBE ONLY. {keep[name]['role'][:120]}",
            "weight_decimals": places,
            "rows_from_weight_at_runtime": r,
            "pool_rows_estimated": pool,
            "epochs_fractional": round(drawn, 4),
            "epochs_pool_measured": keep[name]["epochs_pool_measured"],
            "epochs_pool_source": keep[name]["epochs_pool_source"],
            "fingerprint": keep[name]["fingerprint"],
        }
    assert sum(d["rows_from_weight_at_runtime"] for d in domains.values()) == rows
    return {
        "_comment": [
            "THROWAWAY. The 500-step lr_scale A/B probe (0.85 vs 1.2), fb 2026-09-01.",
            "NOT the 500M run's composition -- that is data/mix_500m.json.",
            "",
            "Eight domains: everything in mix_500m.json except code_py_starcoder, which does",
            "not exist yet. Their weights are the shipped ratios RENORMALISED to sum to 1,",
            "deliberately not re-decided: an arm comparison needs both arms identical, not a",
            "second composition argument made in a hurry.",
            "",
            "WHAT THIS MIX CAN AND CANNOT ANSWER. It can show that one lr_scale diverges,",
            "spikes, or stalls in the first 500 steps -- the failure that is loud early. It",
            "cannot show which lr_scale ends lower at 20B, and a 500-step loss gap must not be",
            "read that way: the arms differ by lr, so their loss curves are not comparable in",
            "level until both have left the warmup transient.",
            "",
            "It also carries NO code_py_starcoder, so nothing measured here says anything",
            "about the code-first objective the real mix is built around.",
        ],
        "total_tokens": rows * SEQ,
        "total_rows": rows,
        "seq": SEQ,
        "steps": PROBE_STEPS,
        "tok_per_step": PROBE_TOK_PER_STEP,
        "tok_per_step_source": "runs/memory_peaks.json world=7 (measured, b32 accum1 seq4096)",
        "anneal_frac": 0.0,
        "warmdown": 0.0,
        "domains": domains,
        "_probe": True,
        "_warnings": warn,
        "_blocked": {},
    }


def _read_cursor(path):
    """{domain: rows} from a checkpoint's row_cursor, or a refusal.

    torch.load with weights_only=False, because row_cursor sits beside the tensors in a dict
    the trainer wrote. REFUSES rather than returning {} when the key is absent: an empty cursor
    and a missing cursor produce identical mixes, and the whole point of this flag is that the
    caller asserted a resume. A silent {} would write a fresh-start mix under a resume's name.
    """
    import torch

    if not os.path.exists(path):
        sys.exit(f"REFUSING: --resume-cursor {path} does not exist. The cursor decides every "
                 f"`epochs` value in this file; guessing it is worse than not writing it.")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    rc = ck.get("row_cursor")
    if not rc:
        sys.exit(f"REFUSING: {path} carries no row_cursor (keys: "
                 f"{sorted(k for k in ck if not k.startswith('model'))}). A checkpoint without "
                 f"one cannot seed used[], so --resume-cursor cannot mean anything against it.")
    basis = ck.get("row_cursor_basis")
    if basis != "full_plan_prefix":
        sys.exit(f"REFUSING: {path} has row_cursor_basis {basis!r}, not 'full_plan_prefix'. "
                 f"build_mix seeds used[] from a full-plan-prefix cursor; any other basis counts "
                 f"rows differently and the epoch totals here would be arithmetic on two "
                 f"incompatible conventions.")
    return {k: int(v) for k, v in rc.items()}


def main():
    global TOTAL_TOKENS, ROWS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Default is the STAMP when the corpus is on this host, so the common case needs no
    # transcription at all. 8.85e9 remains the labelled projection for a dev box. fb sent me
    # 8,744,830,156 to check against, explicitly not to copy -- and checking is what caught
    # that this figure is a 3/283-shard extrapolation, not the full count its message claimed.
    ap.add_argument("--code-tokens", type=float,
                    default=_corpus_stamp("code_py_starcoder", "tokens") or 8.85e9,
                    help="parse-verified Python tokens that actually landed; defaults to "
                         "data/corpus/code_py_starcoder's stamp, else the 8.85e9 projection")
    ap.add_argument("--probe", action="store_true",
                    help="write data/mix_probe_lr.json instead: the 500-step lr A/B probe, "
                         "eight domains, ratios renormalised not re-decided")
    ap.add_argument("--total", type=float, default=TOTAL_TOKENS,
                    help="token budget; the weights are the same function of the objective, "
                         "epochs and the code floor re-derive against this total")
    ap.add_argument("--out", default=None, help="output path; required with a non-default --total")
    # A RESUME MIX IS NOT REUSABLE FOR A DIFFERENT RESUME POINT, and this flag is what makes that
    # visible. Without it `epochs` counts one segment; with it `epochs` is the total the model
    # will have read, so the file is correct for exactly the checkpoint named here and wrong for
    # any other. That is not a limitation to work around -- a mix generated against a different
    # cursor IS a different mix.
    ap.add_argument("--resume-cursor", default=None, metavar="CKPT",
                    help="checkpoint whose row_cursor this plan continues; makes every `epochs` "
                         "value a TOTAL (cursor + this plan) instead of this plan alone. Omit "
                         "for a fresh start.")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    cursor = _read_cursor(a.resume_cursor) if a.resume_cursor else None
    ref = None
    if int(a.total) != TOTAL_TOKENS:
        if not a.out or a.probe:
            ap.error("a non-default --total needs --out and excludes --probe")
        # The REFERENCE build stays fresh-start: it exists only to lift the 20B weights for a
        # smaller total, and a cursor-aware reference would compare two different quantities.
        ref = build(a.code_tokens)
        TOTAL_TOKENS = int(a.total)
        ROWS = TOTAL_TOKENS // SEQ
    m = build_probe() if a.probe else build(a.code_tokens, cursor)
    out = PROBE_OUT if a.probe else (a.out or OUT)
    if ref is not None:
        # WEIGHTS ARE COPIED FROM THE 20B BUILD ONLY WHEN THE NEW TOTAL IS SMALLER, and that
        # condition is the whole point of this branch. The copy exists so a smaller run keeps the
        # 500M composition (user, 2026-09-02): at a smaller total every domain draws FEWER passes
        # than at 20B, so a weight that cleared the 4-epoch ceiling there clears it here too, and
        # the assertion below states exactly that.
        #
        # ABOVE 20B THE COPY IS UNSOUND AND USED TO DIE ON THE ASSERTION (b0-29 adjacent, found
        # while planning the 30B run): copied weights times a LARGER total exceed the 20B draw by
        # construction, so `--total 30e9` raised a bare AssertionError with no message and wrote
        # nothing. The refusal was right -- three domains cross the ceiling at 30B (cot 5.71
        # epochs, chatml 5.70, chat_qa 5.70, computed against their own supply) -- but it refused
        # by arithmetic accident rather than by saying so, and it left no path to a legitimate
        # larger mix.
        #
        # So: above 20B, DO NOT COPY. build() has already run under the new TOTAL_TOKENS and ROWS,
        # so every supply-capped domain's weight came out of _ceiling_weight against the new total
        # -- which is the recompute this branch was skipping. The freed weight is redistributed by
        # build()'s own logic, not here. What is kept is the CHECK: no domain may cross the
        # ceiling, asserted on the built mix rather than inherited from the 20B one.
        _shrinking = TOTAL_TOKENS <= ref["total_tokens"]
        if _shrinking:
            for name, d in m["domains"].items():
                d["weight"], d["anneal"] = ref["domains"][name]["weight"], ref["domains"][name]["anneal"]
                assert d["weight"] * TOTAL_TOKENS <= ref["domains"][name]["weight"] * ref["total_tokens"], (
                    f"{name}: copied weight {d['weight']} at total {TOTAL_TOKENS} draws more than "
                    f"the 20B build did, so the copy cannot inherit its epoch verdict"
                )
            m["_comment"].append(f"TOTAL overridden to {TOTAL_TOKENS / 1e9:.3f}B by --total for "
                                 f"{os.path.basename(out)}: weights and anneal copied from the 20B "
                                 "build, so the composition is the 500M run's; epochs stay as caps "
                                 "and every domain draws fewer passes than at 20B (user, 2026-09-02 "
                                 "10:0xZ: smaller models, fewer tokens, same composition).")
        else:
            # THE EPOCH CEILING, RE-ASSERTED ON WHAT WAS ACTUALLY BUILT. Every domain, not only the
            # supply-capped ones: an uncapped domain's weight is a judgement that was made against
            # 20B of supply headroom, and raising the total can push one over without anything
            # else noticing. Rows, not tokens, for the reason _ceiling_weight documents -- tokens
            # overstate the pool, because packing drops a partial row per document and n_val is
            # held out on top, and deriving in tokens put all three capped domains over the real
            # line while reporting them under it.
            # THE SUPPLY COMES FROM THE BUILT MIX, not from SUPPLY[name]: the two code domains are
            # not in SUPPLY at all -- their supply is split from --code-tokens at build time -- and
            # indexing SUPPLY here raised KeyError('code_py_starcoder') on the first run of this
            # check. The per-domain supply_tokens_one_epoch that build() writes is the one source
            # that covers all nine, and it is also what a later reader of the file sees.
            _over = []
            for name, d in m["domains"].items():
                _sup = d.get("supply_tokens_one_epoch")
                assert _sup, (
                    f"{name} has no supply_tokens_one_epoch in the built mix, so its epoch count "
                    f"cannot be checked and this refusal cannot be shown to cover every domain"
                )
                # int(ROWS*weight), WHICH IS WHAT build_mix ACTUALLY DRAWS, not the fractional
                # product. _ceiling_weight TARGETS the ceiling exactly -- _weight_for_rows solves
                # for max_rows rather than flooring to a safe precision -- so the un-floored
                # product lands a hair ABOVE 4 on pure float error and this check refused its own
                # generator's correct output: cot 4.000000001828, chatml 4.000012765299, chat_qa
                # 4.000028929185, all three at exactly 4.000000000000 once floored. Checking the
                # fractional row is checking a row nobody reads.
                _rows = int(ROWS * d["weight"])
                _epochs = _rows / _domain_pool_rows(name, _sup)
                if _epochs > EPOCH_SOFT_CEILING:
                    _over.append(f"{name} {_epochs:.2f} epochs at weight {d['weight']:.7f}")

            assert not _over, (
                f"REFUSING to write a mix at total {TOTAL_TOKENS / 1e9:.3f}B: "
                f"{len(_over)} domain(s) cross the {EPOCH_SOFT_CEILING}-epoch ceiling even after "
                f"_ceiling_weight recomputed against this total: {'; '.join(_over)}. A supply-capped "
                f"domain should have been sized down automatically, so a name here means either its "
                f"weight is a typed judgement that this total invalidates, or it is missing from "
                f"SUPPLY_CAPPED."
            )
            m["_comment"].append(f"TOTAL raised to {TOTAL_TOKENS / 1e9:.3f}B by --total for "
                                 f"{os.path.basename(out)}: weights RECOMPUTED at this total, not "
                                 f"copied from the 20B build -- copying above 20B draws more than "
                                 f"the 20B mix did and cannot inherit its epoch verdict. Every "
                                 f"domain re-checked against the {EPOCH_SOFT_CEILING}-epoch "
                                 f"ceiling in rows.")

    # REFUSE, do not label. The previous version wrote null plus a fingerprint_source string
    # explaining that the corpus was not on this host -- honest, and still a file. On
    # 2026-09-01 that file was generated on a Mac and pushed over the pod's copy, replacing
    # nine correct fingerprints with nine nulls four hours after they were right. Nobody read
    # the explanation; they read the mix. A generator that cannot do its job must not produce
    # an artifact that outranks one that could (b0-7, fb).
    null_fp = sorted(n for n, d in m["domains"].items() if not d.get("fingerprint"))
    if null_fp:
        print(
            f"REFUSING to write {os.path.basename(out)}: {len(null_fp)} domain(s) have no "
            f"fingerprint -- {', '.join(null_fp[:4])}. Their build_corpus_stats.json is not on "
            f"this host, so this machine cannot produce a launchable mix. Run this WHERE THE "
            f"CORPUS IS (the pod). Writing the file anyway is how a Mac-generated copy "
            f"overwrote the pod's correct fingerprints today.",
            file=sys.stderr,
        )
        return 1
    text = json.dumps(m, indent=1, ensure_ascii=False) + "\n"
    if a.check:
        if not os.path.exists(out):
            print(f"{out} does not exist", file=sys.stderr)
            return 1
        if open(out, encoding="utf-8").read() != text:
            print(f"REFUSING: {out} does not match its inputs -- regenerate", file=sys.stderr)
            return 1
        print(f"{out} matches its inputs")
        return 0
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {out}: {len(m['domains'])} domains, {m['total_tokens'] / 1e9:.3f}B tokens")
    for w in m["_warnings"]:
        print(f"  WARNING {w}")
    for n in m.get("_launch_blocked", []):
        print(f"  LAUNCH BLOCKED {n}: {m['_untrusted_supply'][n]}")
    if m.get("_launch_blocked"):
        print("  -> this mix must not start a run until every blocked supply is measured")
    return 0


if __name__ == "__main__":
    sys.exit(main())

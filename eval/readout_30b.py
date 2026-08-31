#!/usr/bin/env python3
"""30B milestone readout: the pre-registration as code (t34).

Input: a milestone checkpoint's score_matrix record (generative metrics) plus
per-role domain losses for the milestone and its paired checkpoint, scored on
the SAME held-out heads (domain_loss.py --mix <the same mix> for both).

Output: per metric the three-state verdict -- moved (past threshold, with
direction) / floor (unjudgeable) / flat (readable but under threshold) -- with
n, the threshold from docs/lessons/readout_30b_prereg.md, and the paired value.
Metrics that cannot reach 'flat' at their n (threshold = readability limit)
print that fact rather than omitting the state.

degenerate_rate is printed BESIDE its accuracy, never verdict'd on its own
(v2: 55.8% degenerate at greedy vs 2.2% correct -- the two are not substitutes).

Warmup confound (3.24B milestone only): the paired ladder point used warmup=20,
the 30B run uses 300 (8.5% vs 0.57% of the run at reduced LR). If the milestone
reads WORSE past threshold, the pair is unreadable for the mix question -- the
script flags WARMUP-CONFOUND instead of a clean 'moved (degraded)'. A better
reading is real (drag can only attenuate); a worse one is not.

Known-answer dry runs (--selftest):
  1. p324 against itself -> floor/flat everywhere, never moved (domain-loss path).
  1b. sft_v2 against itself -> same, on the generative-metric path.
  2. p324 against the 0.2b ladder point -> per-role domain losses moved
     (lower, the known ladder direction).

Usage:
  python3 eval/readout_30b.py --milestone ckpt_30b_3p24b.pt --paired ckpt_p324.pt \
      --milestone-tokens 3.24e9 \
      --milestone-domain-loss runs/dl_milestone.json --paired-domain-loss runs/dl_paired.json
  python3 eval/readout_30b.py --selftest
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Thresholds from docs/lessons/readout_30b_prereg.md (frozen before t22 launch).
# Each entry: n (default), threshold, reachable states, and whether the threshold
# is also the readability limit (no flat band).
BINOMIAL_DELTA = 1.4  # 2*delta = 2.8/sqrt(n) at p=0.5 worst case; prereg uses 2*delta
METRICS = {
    "code_500": {
        "field": ("code_500", "code_500"), "n": 500, "unit": "pt",
        "threshold": 12.6, "states": ("moved", "floor"),  # 2delta @ 500
        "better": "higher", "degen_key": "code_500_degeneration",
    },
    "math_hard_pass_at_1": {
        "field": ("pass_at_k", "pass_at_1"), "n": 1036, "unit": "pt",
        "threshold": 8.8, "states": ("moved", "floor"),  # 2delta @ 1036
        "better": "higher", "degen_key": "pass_at_k_degeneration",
    },
    "rl_gap_pass8_minus_pass1": {
        "field": ("pass_at_k", None), "n": 1036, "unit": "pt",
        "threshold": 15.0, "states": ("moved", "flat", "floor"),  # RL gate
        "better": "higher",  # a larger gap = more for RL to amplify
        "floor_limit": 2.0,  # paired-gap readability ~1-2pt @ 1036 (prereg)
    },
    "ceval": {
        "field": ("mc_full", "C-Eval (zh)"), "n": 1300, "unit": "pt",
        "threshold": 5.9, "states": ("moved", "floor"),  # 4.65 * seed SD 1.27
        "better": "higher",
    },
    "per_role_domain_loss": {
        "field": ("domain_loss", None), "n": 262144, "unit": "nat",
        "threshold": 0.1176, "states": ("moved", "floor"),  # 2.28 * sigma-hat
        "better": "lower",
    },
}

WARMUP_CONFOUND_MILESTONE_TOKENS = 3.24e9  # only the 3.24B pair has the 20-vs-300 confound

# Alternative top keys for one metric, tried in order after the declared key. The base
# profile scores ceval alone and writes mc_ceval; the milestone and sft profiles score
# ceval+mmlu+arc-easy and write mc_full. Both hold "C-Eval (zh)". The paired 3.24B readout
# crosses that boundary -- milestone side mc_full, ladder side mc_ceval -- so without the
# alias the tripwire the pre-registration relies on reads ABSENT rather than a number.
FIELD_ALIASES = {"mc_full": ("mc_ceval",), "mc_ceval": ("mc_full",)}

#: A per-role domain-loss delta across two mixes carries a reweighting effect as well as a
#: capability change. Equal domain SETS pass the different-heads guard, so a stage-1 vs
#: stage-2 pair looks valid while three of its seven roles were reweighted 0.74x to 1.75x.
#: 5% relative: comfortably inside the noise on a weight, and at the stage-1 -> A' boundary
#: it separates the two groups cleanly (largest passing move 5%, smallest failing 21%).
#: The partition is a property of THIS pair of mixes and is recomputed per boundary, never
#: memorised (prereg §7.1, fb 2026-08-31).
WEIGHT_CHANGE_LIMIT = 0.05


#: Stage-2 domains bind to new stamped corpus directories, so their mix keys carry a
#: "_stage2" suffix -- check_mix_30b_contract's second invariant REFUSES a mix that reuses
#: a frozen ladder directory name. The suffix is load-bearing there and meaningless when
#: pairing a role across the boundary, so the readout normalises it here. This is the only
#: place that pairs across stages; the mix files must keep the suffix (tilerl, 2026-08-31).
_STAGE_SUFFIXES = ("_stage2", "_stage3")


def _base_role(name):
    """'math_owm_stage2' -> 'math_owm'. The role identity a cross-stage pair shares."""
    for suf in _STAGE_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _mix_entries(mix_path):
    """{base role: entry} for a mix file, reading domains AND _blocked.

    _blocked is included because a mix under construction is the normal state: stage-2
    domains move from _blocked into domains one at a time as they are stamped, and a role
    that has not landed yet still carries the weight and epochs this comparison needs.
    Reading only `domains` made every unlanded role unjudgeable-by-absence, which reads as
    no refusal -- a fail-open on the roles most likely to have been reweighted."""
    if not mix_path or not os.path.exists(mix_path):
        return {}
    try:
        with open(mix_path, encoding="utf-8") as f:
            mix = json.load(f)
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for section in ("domains", "_blocked"):
        for k, v in (mix.get(section) or {}).items():
            if isinstance(v, dict):
                out[_base_role(k)] = v
    return out


def _mix_weights(mix_path):
    """{base role: weight}; {} when the path is absent or unreadable. Absent means
    "cannot check", which reads as no refusal -- the caller treats a missing mix as an
    unknown ratio rather than a zero one."""
    return {k: float(v.get("weight", 0)) for k, v in _mix_entries(mix_path).items()}


def weight_ratio(domain, milestone_mix, paired_mix):
    """milestone weight / paired weight for one role; None when either is unavailable.

    None means unjudgeable-by-absence and must not be read as 1.0: a missing mix file
    would otherwise silently assert the weights are equal, which is the failure this
    whole document is about."""
    mw, pw = _mix_weights(milestone_mix), _mix_weights(paired_mix)
    d = _base_role(domain)
    if d not in mw or d not in pw or not pw[d]:
        return None
    return mw[d] / pw[d]


#: Field names a mix may use for a domain's trainable pool, in preference order. The pool
#: is packed rows of seq+1 minus the validation holdout, so it exists only once a token
#: cache has been built; a stage-2 mix written before its caches land publishes
#: stage1_pool_rows (the pool the stage-1 draw was capped against) and nothing else.
_POOL_FIELDS = ("pool_rows", "stage1_pool_rows")


def _pool_rows(entry):
    for f in _POOL_FIELDS:
        v = entry.get(f)
        if v:
            return int(v)
    return None


def draws_equal(domain, milestone_mix, paired_mix, seq=4096):
    """True when both mixes DRAW the same row count for a role despite different weights.

    The weight ratio is a proxy; the exact condition is post-cap draw equality (44, prereg
    §7.1). build_mix computes want = int(total_rows * weight) and then caps it at
    int(pool_rows * epochs), so two different weights that both exceed the cap produce the
    identical draw and carry no reweighting effect at all. cot is the live case: stage 1
    wants int(3,662,109 x 0.0848) = 310,546 rows and clamps to int(98,504 x 3) = 295,512,
    while stage 2 wants 295,512 and is already at the cap -- both draw 295,512.

    None when the mixes do not carry what this needs, which sends the caller back to the
    conservative weight-ratio test. None is common and must stay visible: the caller says
    so in the output, otherwise the exact test silently degrades and nobody learns the
    field is missing (tilerl, 2026-08-31)."""
    me, pe = _mix_entries(milestone_mix), _mix_entries(paired_mix)
    d = _base_role(domain)
    if d not in me or d not in pe:
        return None
    mw, pw = _mix_weights(milestone_mix), _mix_weights(paired_mix)
    try:
        pool = _pool_rows(me[d]) or _pool_rows(pe[d])
        if not pool:
            return None
        m_tot = json.load(open(milestone_mix, encoding="utf-8")).get("total_tokens")
        p_tot = json.load(open(paired_mix, encoding="utf-8")).get("total_tokens")
        if not m_tot or not p_tot:
            return None
        m_draw = min(int(int(m_tot / seq) * mw[d]), int(pool * me[d].get("epochs", 1)))
        p_draw = min(int(int(p_tot / seq) * pw[d]), int(pool * pe[d].get("epochs", 1)))
    except Exception:  # noqa: BLE001
        return None
    return m_draw == p_draw


def load_score_record(path, ckpt, profile="full"):
    """The score_matrix record (one jsonl line) for a (ckpt, profile). A row
    without a profile reads as 'full' (pre-profile ledger, no migration)."""
    if not path or not os.path.exists(path):
        return None
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("ckpt") == ckpt and r.get("profile", "full") == profile:
            return r
    return None


def load_domain_loss(path, ckpt):
    """domain_loss.py --json output: {ckpt, domains: {name: {loss, tokens}}, unweighted_mean}."""
    if not path or not os.path.exists(path):
        return None
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("ckpt") == ckpt:
            return r
    return None


def get_field(record, field):
    """field = (top_key, sub_key); sub_key None means the metric is derived.

    A metric may be recorded under more than one top key when two profiles measure the same
    thing through different benchmark sets. ceval is scored as mc_ceval by the base profile
    and as mc_full by the milestone profile, both carrying the "C-Eval (zh)" sub-key. Trying
    each alias in turn keeps a paired reading judgeable across a profile boundary: without
    it the 3.24B pair reads ABSENT on ceval, because the milestone side writes mc_full and
    ckpt_p324 was scored under base, which writes mc_ceval and leaves mc_full null."""
    if record is None:
        return None
    top, sub = field
    metrics = record.get("metrics", {})
    for key in (top, *FIELD_ALIASES.get(top, ())):
        v = metrics.get(key)
        if v is None:
            continue
        if sub is None:
            return v
        if isinstance(v, dict) and v.get(sub) is not None:
            return v.get(sub)
    return None


def verdict(name, spec, m_val, p_val, is_3p24b_pair):
    """Three-state verdict. Returns (state, direction, delta, note)."""
    if m_val is None or p_val is None:
        return "absent", None, None, "metric not in both records"
    if spec["better"] == "lower":
        delta = p_val - m_val  # positive = milestone improved (loss went down)
    else:
        delta = m_val - p_val  # positive = milestone improved
    T = spec["threshold"]
    if "flat" in spec["states"]:
        # RL gap: moved >= T, readable-but-under = flat, below readability = floor
        if delta >= T:
            state = "moved"
        elif abs(delta) >= spec.get("floor_limit", 0):
            state = "flat"
        else:
            state = "floor"
    else:
        # threshold = readability limit: no flat band
        if abs(delta) >= T:
            state = "moved"
        else:
            state = "floor"
    direction = None
    note = ""
    if state == "moved":
        direction = "improved" if delta > 0 else "degraded"
        if delta < 0 and is_3p24b_pair:
            note = "WARMUP-CONFOUND: pair unreadable for the mix question (warmup 20 vs 300); clean read at 8B/16B/30B"
    if "flat" not in spec["states"] and state != "moved":
        note = (note + "; " if note else "") + "flat unreachable at this n (threshold = readability limit)"
    return state, direction, delta, note


def print_metric(name, spec, m_val, p_val, state, direction, delta, note, m_degen=None):
    flat_ok = "flat" in spec["states"]
    states_str = "/".join(spec["states"])
    print(f"\n{name}")
    print(f"  n={spec['n']}  threshold={spec['threshold']}{spec['unit']}  reachable: {states_str}")
    if m_val is None or p_val is None:
        print(f"  ABSENT: {note}")
        return
    delta_s = f"{delta:+.2f}{spec['unit']}" if delta is not None else "?"
    print(f"  milestone={m_val:.4f}  paired={p_val:.4f}  delta={delta_s}")
    print(f"  verdict: {state}" + (f" ({direction})" if direction else ""))
    if note:
        print(f"  note: {note}")
    if not flat_ok and state == "floor":
        print(f"  (flat is unreachable for this metric at n={spec['n']})")
    if m_degen is not None:
        t = m_degen.get("temperature", "?")
        print(f"  degenerate_rate beside: {m_degen.get('rate', '?')} (n={m_degen.get('n', '?')}, t={t}) -- never a substitute for the accuracy above")


def param_line(ckpt_name, ckpt_dir=None):
    """'206.1M (tied head; state_dict 239.7M)' for one checkpoint, or None if unreadable.

    The two numbers differ because tok.weight and head.weight are the SAME table: the run
    log reports parameters() (206.1M, each counted once) while the state_dict sums the tied
    32832x1024 table twice (239.7M). Printed side by side so the pairing of a run-log number
    against a checkpoint number never reads as a size change (fb, 2026-08-31).

    COMPUTED, never stamped: a hardcoded '206.1M/239.7M' is the same stale-derived-artifact
    shape as a cache whose key does not cover its content -- it would keep printing the old
    pair after an arch change.

    Tying is detected by storage identity FIRST and value equality only as a fallback, and
    the distinction is load-bearing in both directions. Storage identity alone was wrong: a
    mid-run checkpoint written with optimizer state deserialises tok.weight and head.weight
    into separate storages, so the 3.24B readout printed the milestone as a bare 239.7M
    beside the ladder point's '206.1M (tied head; ...)' -- the exact different-sized-models
    reading this function exists to prevent. Value equality alone is also wrong: an UNTIED
    model whose two tables happen to hold equal values reads as tied, hiding a real doubling
    of the embedding table. So: same storage is tying; different storage with equal values
    on the known tied pair (tok.weight/head.weight) is tying by value, reported as such;
    anything else is two parameters."""
    path = os.path.join(ckpt_dir or ROOT, ckpt_name)
    if not os.path.exists(path):
        return None
    try:
        import torch

        try:
            ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except Exception:  # noqa: BLE001
            ck = torch.load(path, map_location="cpu", weights_only=False)
        model = ck.get("model") if isinstance(ck, dict) else None
        if not isinstance(model, dict):
            return None
        total, uniq, seen = 0, 0, set()
        for t in model.values():
            if not hasattr(t, "numel"):
                continue
            total += int(t.numel())
            key = (t.data_ptr(), int(t.numel()))
            if key not in seen:
                seen.add(key)
                uniq += int(t.numel())
        by_value = False
        if uniq == total:
            tok, head = model.get("tok.weight"), model.get("head.weight")
            if (
                tok is not None
                and head is not None
                and getattr(tok, "shape", None) == getattr(head, "shape", None)
                and torch.equal(tok, head)
            ):
                uniq -= int(head.numel())
                by_value = True
    except Exception:  # noqa: BLE001
        return None
    if not total:
        return None
    if uniq == total:
        return f"{total / 1e6:.1f}M"
    how = "tied head by value" if by_value else "tied head"
    return f"{uniq / 1e6:.1f}M ({how}; state_dict {total / 1e6:.1f}M)"


#: Two checkpoints differenced across different token budgets bias every metric
#: toward "flat" -- the false negative this readout exists to prevent. 5%: the
#: readable-move thresholds are 2-5% effects, so a larger budget gap can produce or
#: erase one on its own (fb, 2026-08-31, after a 2.753B checkpoint was nearly
#: differenced against a 3.24B pair).
BUDGET_MISMATCH_LIMIT = 0.05


def readout(milestone, paired, score_matrix, milestone_dl, paired_dl, milestone_tokens,
            milestone_profile="milestone", paired_profile="full", selftest=False, ckpt_dir=None,
            actual_tokens=None, paired_tokens=None, milestone_mix=None, paired_mix=None):
    is_3p24b = milestone_tokens is not None and abs(milestone_tokens - WARMUP_CONFOUND_MILESTONE_TOKENS) / WARMUP_CONFOUND_MILESTONE_TOKENS < 0.05
    m_rec = load_score_record(score_matrix, milestone, milestone_profile)
    p_rec = load_score_record(score_matrix, paired, paired_profile)
    m_dl = load_domain_loss(milestone_dl, milestone) if milestone_dl else None
    p_dl = load_domain_loss(paired_dl, paired) if paired_dl else None
    # domain loss may also live in the score_matrix record (same heads); prefer the explicit files
    if m_dl is None and m_rec and "domain_loss" in m_rec.get("metrics", {}):
        m_dl = {"ckpt": milestone, "domains": {k: v for k, v in m_rec["metrics"]["domain_loss"].items() if isinstance(v, dict)}}
    if p_dl is None and p_rec and "domain_loss" in p_rec.get("metrics", {}):
        p_dl = {"ckpt": paired, "domains": {k: v for k, v in p_rec["metrics"]["domain_loss"].items() if isinstance(v, dict)}}

    # Budget header, and a refusal when the two are not comparable.
    if actual_tokens or paired_tokens:
        a_t = actual_tokens or milestone_tokens
        p_t = paired_tokens or milestone_tokens
        print(f"budgets: milestone {a_t / 1e9:.3f}B tokens, paired {p_t / 1e9:.3f}B tokens", end="")
        if a_t and p_t:
            gap = abs(a_t - p_t) / max(a_t, p_t)
            print(f", gap {gap:.1%}")
            if gap > BUDGET_MISMATCH_LIMIT:
                print(f"\nREFUSING to judge: the budgets differ by {gap:.1%} (limit "
                      f"{BUDGET_MISMATCH_LIMIT:.0%}). Differencing metrics across different "
                      f"token budgets biases every one toward 'flat' -- the false negative this "
                      f"readout exists to prevent. Score a checkpoint at the pair's budget.")
                return False
        else:
            print()
    print(f"=== 30B readout: {milestone} vs {paired}" + (" (3.24B pair, warmup confound active)" if is_3p24b else "") + " ===")
    # params for BOTH sides, so a run-log 206.1M paired against a state_dict 239.7M does not
    # read as a size change. Silent when a checkpoint is not on this box -- the readout is a
    # verdict table, not a checkpoint inspector.
    for side, name in (("milestone", milestone), ("paired", paired)):
        pl = param_line(name, ckpt_dir)
        if pl:
            print(f"  {side:9s} {name}: params {pl}")
    any_moved = False
    for name, spec in METRICS.items():
        if name == "per_role_domain_loss":
            # per-role: compare each domain present in BOTH loss records
            if m_dl is None or p_dl is None:
                print(f"\n{name}: ABSENT (domain-loss records missing)")
                continue
            # Different heads means the two models were scored on text neither shares.
            # Tonight: stage-1 (code_rp1t/cot/en_c4/math_owm/textbook_30b/wiki_chat/
            # zh_web) against the ladder's (chat/code/en/math/textbook/web_hq/wiki),
            # zero overlap, reported as "6 of 7 domains degraded". A partial overlap is
            # the more dangerous version -- it yields a verdict that looks whole.
            m_heads, p_heads = set(m_dl["domains"]), set(p_dl["domains"])
            if m_heads != p_heads:
                only_m, only_p = sorted(m_heads - p_heads), sorted(p_heads - m_heads)
                print(f"\n{name}: REFUSING -- the pair was scored on DIFFERENT heads. "
                      f"milestone-only {only_m or 'none'}, paired-only {only_p or 'none'}. "
                      f"Domain loss across different corpora measures the corpora, not the "
                      f"models. Rescore both on one mix.")
                continue
            common = sorted(set(m_dl["domains"]) & set(p_dl["domains"]))
            if not common:
                print(f"\n{name}: ABSENT (no shared domains -- score both checkpoints on the same mix heads)")
                continue
            print(f"\n{name}  (threshold={spec['threshold']}{spec['unit']}, states={'/'.join(spec['states'])}, flat unreachable)")
            for d in common:
                mv = m_dl["domains"][d]["loss"]
                pv = p_dl["domains"][d]["loss"]
                # A role whose WEIGHT changed between the two mixes carries a reweighting
                # effect inside its delta. Equal head sets pass the guard above, so this is
                # the case that guard cannot see. Refuse per role rather than per pair: at
                # the stage-1 -> A' boundary four roles move by <5% and three by 21-75%, so
                # refusing the whole metric would discard four honest reads and permitting
                # it would ask a reader to discount 1.75x by eye.
                wr = weight_ratio(d, milestone_mix, paired_mix)
                same_draw = draws_equal(d, milestone_mix, paired_mix)
                if same_draw is None and wr is not None:
                    # Say so. A silent degradation to the proxy means nobody learns the
                    # exact test never ran -- which is how a guard becomes decorative.
                    print(f"  {d:15s} (post-cap draw unknown; using the weight-ratio proxy)")
                if wr is not None and abs(wr - 1.0) > WEIGHT_CHANGE_LIMIT and not same_draw:
                    w_p = _mix_weights(paired_mix).get(d)
                    w_m = _mix_weights(milestone_mix).get(d)
                    print(f"  {d:15s} milestone={mv:.4f} paired={pv:.4f}  "
                          f"floor (reweighted, unjudgeable)  w {w_p:.5f} -> {w_m:.5f} ({wr:.2f}x)")
                    print(f"      note: read within-stage only; the cross-stage delta mixes "
                          f"capability with a {wr:.2f}x weight change")
                    continue
                state, direction, delta, note = verdict(name, spec, mv, pv, is_3p24b)
                if state == "moved":
                    any_moved = True
                flag = f"  {state}" + (f" ({direction})" if direction else "")
                print(f"  {d:15s} milestone={mv:.4f} paired={pv:.4f} delta={delta:+.4f}{flag}")
                if note:
                    print(f"      note: {note}")
            continue
        if name == "rl_gap_pass8_minus_pass1":
            m_pak = get_field(m_rec, ("pass_at_k", None))
            p_pak = get_field(p_rec, ("pass_at_k", None))
            if m_pak is None or p_pak is None or "pass_at_1" not in m_pak or "pass_at_8" not in m_pak:
                print(f"\n{name}: ABSENT (pass_at_k missing)")
                continue
            m_gap = m_pak["pass_at_8"] - m_pak["pass_at_1"]
            p_gap = p_pak["pass_at_8"] - p_pak["pass_at_1"]
            state, direction, delta, note = verdict(name, spec, m_gap, p_gap, is_3p24b)
            if state == "moved":
                any_moved = True
            print(f"\n{name}  (n={spec['n']}, RL gate threshold={spec['threshold']}pt, states={'/'.join(spec['states'])})")
            print(f"  milestone gap={m_gap:.1f}pt (p@1={m_pak['pass_at_1']}, p@8={m_pak['pass_at_8']})  paired gap={p_gap:.1f}pt  delta={delta:+.1f}pt")
            print(f"  verdict: {state}" + (f" ({direction})" if direction else ""))
            if note:
                print(f"  note: {note}")
            if state == "moved" and direction == "improved":
                print("  RL gate: gap >= 15pt -> RL may start (something to amplify)")
            elif state == "flat":
                print("  RL gate: gap readable but < 15pt -> no RL (the 200M 3.5pt reading was flat, not floor)")
            continue
        # generative accuracy metrics
        mv = get_field(m_rec, spec["field"])
        pv = get_field(p_rec, spec["field"])
        state, direction, delta, note = verdict(name, spec, mv, pv, is_3p24b)
        if state == "moved":
            any_moved = True
        m_degen = None
        if spec.get("degen_key") and m_rec:
            m_degen = m_rec.get("metrics", {}).get(spec["degen_key"])
        print_metric(name, spec, mv, pv, state, direction, delta, note, m_degen)

    print(f"\n=== summary: {'at least one metric moved' if any_moved else 'no metric moved (all floor/flat/absent)'} ===")
    return any_moved


def selftest():
    """Known-answer dry runs. A verdict engine that cannot pass these is not an engine.

    Every exit prints the case count, and a skipped case is named. "selftest OK" from a
    directory where ROOT resolves elsewhere would otherwise report success with the
    real-mix case silently skipped -- the §7.3 failure one level up, in how the test is
    invoked rather than in what it reads (aupai-5b, 2026-08-31)."""
    ran, skipped = [], []

    def _done():
        if not ran:
            print(f"\nselftest FAILED: 0 cases ran. A skip is not a pass. ROOT resolved to "
                  f"{ROOT}; run this file from inside the repo, not a copy elsewhere.",
                  file=sys.stderr)
            return 1
        print(f"\nselftest OK -- {len(ran)} case(s) ran: {', '.join(ran)}")
        if skipped:
            print(f"  SKIPPED {len(skipped)}: {', '.join(skipped)}. A skip is not a pass; "
                  f"ROOT resolved to {ROOT}.", file=sys.stderr)
        return 0

    sm = os.path.join(ROOT, "runs", "score_matrix.jsonl")
    # find the ladder 0.2b and 3.24b (p324) records
    rec_0p2 = load_score_record(sm, "ckpt_0830v1_0.2b.pt")
    rec_3p24 = load_score_record(sm, "ckpt_p324.pt")
    if rec_0p2 is None or rec_3p24 is None:
        skipped += ["1", "1b", "2", "2b", "2c", "3", "4", "5", "5b", "6"]
        print(f"selftest SKIP: ladder records not in {sm}", file=sys.stderr)
        return _done()
    # 1. p324 against itself -> never moved (domain-loss path)
    ran.append("1")
    print("--- selftest 1: p324 vs itself (must be floor/flat everywhere, never moved) ---")
    moved1 = readout("ckpt_p324.pt", "ckpt_p324.pt", sm, None, None, None,
                     milestone_profile="full", paired_profile="full", selftest=True)
    assert not moved1, "p324 vs itself read as MOVED -- the engine is broken"
    # 1b. sft_v2 against itself -> never moved (generative-metric path: code_500/pass_at_k/mc_full)
    ran.append("1b")
    print("\n--- selftest 1b: sft_v2 vs itself (generative metrics must be floor/flat, never moved) ---")
    moved1b = readout("ckpt_sft_p324_v2.pt", "ckpt_sft_p324_v2.pt", sm, None, None, None,
                      milestone_profile="full", paired_profile="full", selftest=True)
    assert not moved1b, "sft_v2 vs itself read as MOVED -- the engine is broken"
    # 2. p324 vs 0.2b -> domain losses moved in the known direction (lower at 3.24b)
    ran.append("2")
    print("\n--- selftest 2: p324 vs 0.2b (domain losses must be moved, lower) ---")
    # domain losses live in the score_matrix records for ladder points (same heads)
    dl_0p2 = {"ckpt": "ckpt_0830v1_0.2b.pt", "domains": {k: v for k, v in rec_0p2["metrics"]["domain_loss"].items() if isinstance(v, dict)}}
    dl_3p24 = {"ckpt": "ckpt_p324.pt", "domains": {k: v for k, v in rec_3p24["metrics"]["domain_loss"].items() if isinstance(v, dict)}}
    common = sorted(set(dl_0p2["domains"]) & set(dl_3p24["domains"]))
    assert common, "no shared domains between 0.2b and 3.24b records"
    moved_domains = 0
    for d in common:
        delta = dl_0p2["domains"][d]["loss"] - dl_3p24["domains"][d]["loss"]  # positive = improved
        if abs(delta) >= METRICS["per_role_domain_loss"]["threshold"]:
            assert delta > 0, f"{d}: 3.24b loss HIGHER than 0.2b ({delta:+.4f}) -- ladder direction violated"
            moved_domains += 1
    assert moved_domains >= 1, "no domain loss moved between 0.2b and 3.24b -- engine or data broken"
    print(f"  {moved_domains}/{len(common)} domains moved (all in the known direction: lower at 3.24b)")
    # 2b. param_line distinguishes a TIED head from two equal-but-separate tables. Value
    # Storage identity is primary; value equality on the known tied pair is the fallback.
    # Both are needed: see param_line's docstring for why either alone is wrong.
    ran.append("2b")
    print("\n--- selftest 2b: param_line reads tying by storage, then by value ---")
    try:
        import tempfile

        import torch

        w = torch.zeros(4, 5)
        other = torch.ones(4, 5)  # equal-shaped, DIFFERENT values: genuinely untied
        with tempfile.TemporaryDirectory() as d:
            torch.save({"model": {"tok.weight": w, "head.weight": w}}, os.path.join(d, "tied.pt"))
            torch.save({"model": {"tok.weight": w, "head.weight": other}}, os.path.join(d, "untied.pt"))
            # The shape that broke the 3.24B readout: a mid-run checkpoint deserialises the
            # tied pair into SEPARATE storages, so a storage-only test reads it as untied and
            # prints a bare 239.7M beside the ladder point's 206.1M.
            saved = {"tok.weight": w, "head.weight": w}
            torch.save({"model": saved, "step": 3500}, os.path.join(d, "midrun.pt"))
            reloaded = torch.load(os.path.join(d, "midrun.pt"), weights_only=False)["model"]
            reloaded["head.weight"] = reloaded["head.weight"].clone()  # force distinct storage
            torch.save({"model": reloaded, "step": 3500}, os.path.join(d, "midrun.pt"))
            tied = param_line("tied.pt", d)
            untied = param_line("untied.pt", d)
            midrun = param_line("midrun.pt", d)
        assert tied and tied.startswith("0.0M (tied head; state_dict 0.0M)"), tied
        assert untied and "tied" not in untied and "state_dict" not in untied, (
            f"two DIFFERENT tables reported as tied: {untied} -- this would hide a real "
            "doubling of the embedding table"
        )
        assert midrun and "tied head by value" in midrun, (
            f"a mid-run checkpoint whose tied pair deserialised into separate storages read "
            f"as untied: {midrun}. This is the 3.24B readout defect -- the milestone printed a "
            "bare 239.7M beside the ladder point's 206.1M, i.e. two different-sized models."
        )
        print(f"  tied -> {tied}; untied -> {untied}; mid-run -> {midrun}")
    except ImportError:
        print("  SKIP: no torch on this box", file=sys.stderr)
    # 2c. ceval survives the profile boundary. The milestone profile writes mc_full; the
    # base profile that scored the ladder point writes mc_ceval and leaves mc_full null.
    # Without FIELD_ALIASES the pair reads ABSENT and the tripwire is silently unarmed --
    # ABSENT looks like "not measured", so nobody goes looking for the number that exists.
    ran.append("2c")
    print("\n--- selftest 2c: ceval reads across the milestone/base profile boundary ---")
    spec = METRICS["ceval"]
    milestone_side = {"metrics": {"mc_full": {"C-Eval (zh)": 27.1}}}
    ladder_side = {"metrics": {"mc_ceval": {"C-Eval (zh)": 22.4}, "mc_full": None}}
    mv, pv = get_field(milestone_side, spec["field"]), get_field(ladder_side, spec["field"])
    assert mv == 27.1, f"milestone mc_full not read: {mv}"
    assert pv == 22.4, (
        f"ladder mc_ceval not read through the mc_full alias: {pv}. The 3.24B pair would "
        "read ABSENT on ceval while both numbers exist in the ledger."
    )
    state, direction, delta, _ = verdict("ceval", spec, mv, pv, True)
    assert state != "absent", "ceval judgeable on both sides still read absent"
    print(f"  milestone(mc_full)={mv} paired(mc_ceval)={pv} delta={delta:+.1f}pt -> {state}")

    # 3. a record missing a metric prints ABSENT, never floor (t39 acceptance:
    # a milestone whose score_matrix record lacks a metric is unmeasured, not floor)
    ran.append("3")
    print("\n--- selftest 3: missing metric -> ABSENT, never floor ---")
    rec_full = load_score_record(sm, "ckpt_sft_p324_v2.pt", "full")
    if rec_full is None:
        print("selftest SKIP: sft_v2 record not in runs/score_matrix.jsonl", file=sys.stderr)
        return 0
    import contextlib
    import io
    import tempfile
    rec_missing = json.loads(json.dumps(rec_full))
    rec_missing["profile"] = "milestone"  # the production path: milestone record, one metric dropped
    rec_missing.get("metrics", {}).pop("code_500", None)
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        tf.write(json.dumps(rec_missing) + "\n")
        tmp = tf.name
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            readout("ckpt_sft_p324_v2.pt", "ckpt_sft_p324_v2.pt", tmp, None, None, None)
        out = buf.getvalue()
    finally:
        os.unlink(tmp)
    i = out.index("\ncode_500\n")
    section = out[i:i + 400]
    # A missing metric prints ABSENT and NO verdict line (print_metric returns early);
    # "reachable: moved/floor" is the spec label, not a verdict -- do not grep for "floor".
    assert "ABSENT" in section and "verdict:" not in section, f"missing metric misread:\n{section}"
    print("  code_500 absent from the record -> ABSENT, never floor")
    # 4. tonight's exact pair: disjoint heads must refuse, not report a regression
    ran.append("4")
    print("\n--- selftest 4: different heads -> REFUSE, never a verdict ---")
    import contextlib as _c
    import io as _io
    import tempfile as _t
    m4 = {"ckpt": "m.pt", "profile": "milestone",
          "metrics": {"domain_loss": {k: {"loss": 2.4} for k in
                      ("code_rp1t", "cot", "en_c4", "math_owm", "textbook_30b", "wiki_chat", "zh_web")}}}
    p4 = {"ckpt": "p.pt", "profile": "full",
          "metrics": {"domain_loss": {k: {"loss": 1.6} for k in
                      ("chat", "code", "en", "math", "textbook", "web_hq", "wiki")}}}
    with _t.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        tf.write(json.dumps(m4) + "\n" + json.dumps(p4) + "\n")
        tmp4 = tf.name
    try:
        buf = _io.StringIO()
        with _c.redirect_stdout(buf):
            readout("m.pt", "p.pt", tmp4, None, None, 3.24e9, paired_profile="full")
        out4 = buf.getvalue()
    finally:
        os.unlink(tmp4)
    assert "REFUSING" in out4 and "DIFFERENT heads" in out4, out4[-400:]
    assert "moved (degraded)" not in out4, "disjoint heads must never produce a verdict"
    print("  disjoint heads refuse; no per-domain verdict is printed")
    # 5. A reweighted role refuses; roles whose weight held still print a verdict. The
    # failing case is math_owm at 1.75x (stage 1 0.18333 -> A' 0.32058): equal domain SETS
    # pass the different-heads guard, so without this gate the reweighting shows up as
    # capability. cot is the post-cap case -- its weight moved 0.95x AND both mixes clamp
    # to the same 295,512 rows, so it is judgeable for the stronger reason.
    ran.append("5")
    print("\n--- selftest 5: a reweighted role refuses, an unchanged one is judged ---")
    import tempfile as _t5
    W1 = {"code_rp1t": 0.37200, "math_owm": 0.18333, "cot": 0.08480, "en_c4": 0.20987,
          "zh_web": 0.11000, "textbook_30b": 0.03333, "wiki_chat": 0.00667}
    W2 = {"code_rp1t": 0.29330, "math_owm": 0.32058, "cot": 0.08069, "en_c4": 0.15604,
          "zh_web": 0.10955, "textbook_30b": 0.03320, "wiki_chat": 0.00664}
    with _t5.TemporaryDirectory() as d5:
        m5p, p5p = os.path.join(d5, "s2.json"), os.path.join(d5, "s1.json")
        json.dump({"total_tokens": 15e9, "domains": {k: {"weight": v} for k, v in W2.items()}},
                  open(m5p, "w"))
        json.dump({"total_tokens": 15e9, "domains": {k: {"weight": v} for k, v in W1.items()}},
                  open(p5p, "w"))
        # milestone strictly better on every role, so any judged role reads "moved"
        m5 = {"ckpt": "m5.pt", "profile": "milestone",
              "metrics": {"domain_loss": {k: {"loss": 1.5} for k in W1}}}
        p5 = {"ckpt": "p5.pt", "profile": "full",
              "metrics": {"domain_loss": {k: {"loss": 2.5} for k in W1}}}
        sm5 = os.path.join(d5, "sm.jsonl")
        with open(sm5, "w") as f:
            f.write(json.dumps(m5) + "\n" + json.dumps(p5) + "\n")
        buf5 = _io.StringIO()
        with _c.redirect_stdout(buf5):
            readout("m5.pt", "p5.pt", sm5, None, None, 8e9, paired_profile="full",
                    milestone_mix=m5p, paired_mix=p5p)
        out5 = buf5.getvalue()
    for role, ratio in (("math_owm", 1.75), ("code_rp1t", 0.79), ("en_c4", 0.74)):
        # the role may occupy several lines (proxy note, verdict, note) -- join them
        line = " ".join(ln for ln in out5.splitlines() if ln.strip().startswith(role))
        assert "reweighted, unjudgeable" in line, (
            f"{role} moved {ratio}x and was still judged: {line!r}. Equal domain sets pass "
            "the different-heads guard, so nothing else catches this."
        )
    for role in ("cot", "zh_web", "textbook_30b", "wiki_chat"):
        line = " ".join(ln for ln in out5.splitlines() if ln.strip().startswith(role))
        assert "moved" in line and "reweighted" not in line, (
            f"{role} held its weight and was refused anyway: {line!r}. Refusing the whole "
            "metric would discard four honest reads."
        )
    print("  3 reweighted roles refuse (1.75x/0.79x/0.74x); 4 unchanged roles judged")

    # 5b. The gate must reach the REAL mix files, not only synthetic ones. e178f17 passed
    # its synthetic case and returned None for every role against the live pair, for three
    # reasons a synthetic fixture cannot show: stage-2 keys carry a "_stage2" suffix, most
    # stage-2 roles sit in _blocked until stamped, and the pool field is stage1_pool_rows
    # (tilerl, 2026-08-31). A guard that cannot read production data is decorative.
    real_s1 = os.path.join(ROOT, "data", "mix_15b_stage1.json")
    real_s2 = os.path.join(ROOT, "data", "mix_30b_stage2.json")
    if os.path.exists(real_s1) and os.path.exists(real_s2):
        ran.append("5b")
        print("\n--- selftest 5b: the gate reads the real stage-1/stage-2 pair ---")
        got = {d: (weight_ratio(d, real_s2, real_s1), draws_equal(d, real_s2, real_s1))
               for d in ("code_rp1t", "math_owm", "cot", "en_c4", "zh_web",
                         "textbook_30b", "wiki_chat")}
        unreadable = [d for d, (wr, _) in got.items() if wr is None]
        assert not unreadable, (
            f"weight_ratio is None for {unreadable} against the real mixes -- the gate "
            "cannot see production data and every role would pass unrefused"
        )
        for d in ("math_owm", "code_rp1t", "en_c4"):
            wr, de = got[d]
            assert abs(wr - 1.0) > WEIGHT_CHANGE_LIMIT and not de, f"{d}: ratio {wr}, draws_equal {de}"
        assert got["cot"][1] is True, (
            f"cot post-cap draw equality not detected: {got['cot']}. Both mixes clamp to "
            "295,512 rows, so cot is judgeable for a stronger reason than its 0.95x."
        )
        print(f"  real pair: 3 refuse, cot judged on post-cap draw equality "
              f"(ratio {got['cot'][0]:.3f}, draws_equal True)")
    else:
        skipped.append("5b (real mix pair absent)")
        print("\n--- selftest 5b: SKIP (real mix pair not on this box) ---", file=sys.stderr)

    return _done()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--milestone", help="milestone checkpoint name")
    ap.add_argument("--paired", help="paired checkpoint name (ladder point or previous milestone)")
    ap.add_argument("--score-matrix", default=os.path.join(ROOT, "runs", "score_matrix.jsonl"))
    ap.add_argument("--milestone-domain-loss", help="domain_loss.py --json for the milestone (30B heads)")
    ap.add_argument("--paired-domain-loss", help="domain_loss.py --json for the paired checkpoint (same heads)")
    ap.add_argument("--milestone-tokens", type=float, help="milestone token budget (3.24e9/8e9/16e9/30e9); activates the warmup-confound rule at 3.24B")
    ap.add_argument("--ckpt-dir", default=ROOT, help="where the checkpoints live (for the params header)")
    ap.add_argument("--milestone-mix", help="mix json the milestone trained under (per-role reweight gate, prereg 7.1)")
    ap.add_argument("--paired-mix", help="mix json the paired checkpoint trained under")
    ap.add_argument("--actual-tokens", type=float, default=None,
                    help="tokens the milestone checkpoint actually saw (step x tokens/step)")
    ap.add_argument("--paired-tokens", type=float, default=None,
                    help="tokens the paired checkpoint saw; a >5%% gap refuses the comparison")
    ap.add_argument("--milestone-profile", default="milestone", help="score_matrix profile of the milestone record")
    ap.add_argument("--paired-profile", default="full", help="score_matrix profile of the paired record")
    ap.add_argument("--selftest", action="store_true", help="known-answer dry runs")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.milestone or not a.paired:
        ap.error("--milestone and --paired are required (or --selftest)")
    readout(a.milestone, a.paired, a.score_matrix, a.milestone_domain_loss, a.paired_domain_loss,
            a.milestone_tokens, a.milestone_profile, a.paired_profile, ckpt_dir=a.ckpt_dir,
            milestone_mix=a.milestone_mix, paired_mix=a.paired_mix)


if __name__ == "__main__":
    main()

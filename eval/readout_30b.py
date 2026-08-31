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
    """field = (top_key, sub_key); sub_key None means the metric is derived."""
    if record is None:
        return None
    top, sub = field
    v = record.get("metrics", {}).get(top)
    if v is None:
        return None
    if sub is None:
        return v
    if isinstance(v, dict):
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


def readout(milestone, paired, score_matrix, milestone_dl, paired_dl, milestone_tokens, selftest=False):
    is_3p24b = milestone_tokens is not None and abs(milestone_tokens - WARMUP_CONFOUND_MILESTONE_TOKENS) / WARMUP_CONFOUND_MILESTONE_TOKENS < 0.05
    m_rec = load_score_record(score_matrix, milestone)
    p_rec = load_score_record(score_matrix, paired)
    m_dl = load_domain_loss(milestone_dl, milestone) if milestone_dl else None
    p_dl = load_domain_loss(paired_dl, paired) if paired_dl else None
    # domain loss may also live in the score_matrix record (same heads); prefer the explicit files
    if m_dl is None and m_rec and "domain_loss" in m_rec.get("metrics", {}):
        m_dl = {"ckpt": milestone, "domains": {k: v for k, v in m_rec["metrics"]["domain_loss"].items() if isinstance(v, dict)}}
    if p_dl is None and p_rec and "domain_loss" in p_rec.get("metrics", {}):
        p_dl = {"ckpt": paired, "domains": {k: v for k, v in p_rec["metrics"]["domain_loss"].items() if isinstance(v, dict)}}

    print(f"=== 30B readout: {milestone} vs {paired}" + (" (3.24B pair, warmup confound active)" if is_3p24b else "") + " ===")
    any_moved = False
    for name, spec in METRICS.items():
        if name == "per_role_domain_loss":
            # per-role: compare each domain present in BOTH loss records
            if m_dl is None or p_dl is None:
                print(f"\n{name}: ABSENT (domain-loss records missing)")
                continue
            common = sorted(set(m_dl["domains"]) & set(p_dl["domains"]))
            if not common:
                print(f"\n{name}: ABSENT (no shared domains -- score both checkpoints on the same mix heads)")
                continue
            print(f"\n{name}  (threshold={spec['threshold']}{spec['unit']}, states={'/'.join(spec['states'])}, flat unreachable)")
            for d in common:
                mv = m_dl["domains"][d]["loss"]
                pv = p_dl["domains"][d]["loss"]
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
    """Two known-answer dry runs. A verdict engine that cannot pass these is not an engine."""
    sm = os.path.join(ROOT, "runs", "score_matrix.jsonl")
    # find the ladder 0.2b and 3.24b (p324) records
    rec_0p2 = load_score_record(sm, "ckpt_0830v1_0.2b.pt")
    rec_3p24 = load_score_record(sm, "ckpt_p324.pt")
    if rec_0p2 is None or rec_3p24 is None:
        print("selftest SKIP: ladder records not in runs/score_matrix.jsonl", file=sys.stderr)
        return 0
    # 1. p324 against itself -> never moved (domain-loss path)
    print("--- selftest 1: p324 vs itself (must be floor/flat everywhere, never moved) ---")
    moved1 = readout("ckpt_p324.pt", "ckpt_p324.pt", sm, None, None, None, selftest=True)
    assert not moved1, "p324 vs itself read as MOVED -- the engine is broken"
    # 1b. sft_v2 against itself -> never moved (generative-metric path: code_500/pass_at_k/mc_full)
    print("\n--- selftest 1b: sft_v2 vs itself (generative metrics must be floor/flat, never moved) ---")
    moved1b = readout("ckpt_sft_p324_v2.pt", "ckpt_sft_p324_v2.pt", sm, None, None, None, selftest=True)
    assert not moved1b, "sft_v2 vs itself read as MOVED -- the engine is broken"
    # 2. p324 vs 0.2b -> domain losses moved in the known direction (lower at 3.24b)
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
    # 3. a record missing a metric prints ABSENT, never floor (t39 acceptance:
    # a milestone whose score_matrix record lacks a metric is unmeasured, not floor)
    print("\n--- selftest 3: missing metric -> ABSENT, never floor ---")
    rec_full = load_score_record(sm, "ckpt_sft_p324_v2.pt")
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
    print("\nselftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--milestone", help="milestone checkpoint name")
    ap.add_argument("--paired", help="paired checkpoint name (ladder point or previous milestone)")
    ap.add_argument("--score-matrix", default=os.path.join(ROOT, "runs", "score_matrix.jsonl"))
    ap.add_argument("--milestone-domain-loss", help="domain_loss.py --json for the milestone (30B heads)")
    ap.add_argument("--paired-domain-loss", help="domain_loss.py --json for the paired checkpoint (same heads)")
    ap.add_argument("--milestone-tokens", type=float, help="milestone token budget (3.24e9/8e9/16e9/30e9); activates the warmup-confound rule at 3.24B")
    ap.add_argument("--selftest", action="store_true", help="known-answer dry runs")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.milestone or not a.paired:
        ap.error("--milestone and --paired are required (or --selftest)")
    readout(a.milestone, a.paired, a.score_matrix, a.milestone_domain_loss, a.paired_domain_loss, a.milestone_tokens)


if __name__ == "__main__":
    main()

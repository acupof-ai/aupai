#!/usr/bin/env python3
"""data/mix_1.5b-a0.2b-e48_30b_v2.json must stay resume-1 with ONE value restated.

v2 changes the architecture; the prereg compares it against the unlooped control at equal
tokens, which means equal DATA. Any weight that drifts into this file makes the comparison
two-variable. The file is a hand copy, so nothing recomputes it -- this is what does.

Only the diff is asserted, not the values: a copy of the weights here would be a second
source of truth free to drift from the file, which is the defect write_mix_500m's own
comments keep re-learning.

    python3 scripts/test_v2_mix_is_resume1.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROL = os.path.join(ROOT, "data", "mix_1.5b-a0.2b-e48_30b_resume1.json")
V2 = os.path.join(ROOT, "data", "mix_1.5b-a0.2b-e48_30b_v2.json")

# The one restatement, and its source. facts/corpus_supply.json#cs.math_owm_landed.
RESTATED = ("math_owm_stage2", "supply_tokens_one_epoch", 6_513_304_690, 6_528_546_115)
# supply is a denominator in this file, so exactly these fields may move with it -- and ONLY
# in the restated domain. Kept deliberately tight: the first version of this list was scoped
# per-domain but its negative controls all edited OTHER domains, so widening it to
# {"weight", "fingerprint"} left the test green. A mutation caught that, not a reading.
ALLOWED = {"supply_tokens_one_epoch", "epochs_fractional", "supply_restated"}
# Fields that are never a consequence of a supply restatement, in ANY domain, including the
# restated one. Listing them is what makes ALLOWED's scope testable.
NEVER_MOVES = {"weight", "fingerprint", "rows_from_weight_at_runtime", "want_tokens",
               "epochs", "anneal", "pool_rows_estimated", "cursor_used_rows"}


def check(control, v2):
    """[] when v2 is control plus the one restatement, else a list of complaints."""
    bad = []
    a, b = control["domains"], v2["domains"]
    if set(a) != set(b):
        bad.append(f"domain set differs: only-control {set(a) - set(b)}, only-v2 {set(b) - set(a)}")
        return bad
    for k in ("total_tokens", "total_rows", "seq", "anneal_frac", "warmdown"):
        if control.get(k) != v2.get(k):
            bad.append(f"{k}: {control.get(k)} -> {v2.get(k)}")
    dom, field, old, new = RESTATED
    if ALLOWED & NEVER_MOVES:
        bad.append(f"ALLOWED lets through fields a supply restatement cannot move: "
                   f"{sorted(ALLOWED & NEVER_MOVES)}")
    for name in sorted(a):
        for f in sorted(set(a[name]) | set(b[name])):
            if a[name].get(f) == b[name].get(f):
                continue
            if name == dom and f in ALLOWED:
                continue
            bad.append(f"{name}.{f}: {a[name].get(f)} -> {b[name].get(f)}")
    if a[dom][field] != old:
        bad.append(f"control's {dom}.{field} is {a[dom][field]}, expected {old}")
    if b[dom][field] != new:
        bad.append(f"v2's {dom}.{field} is {b[dom][field]}, expected {new}")
    want = b[dom]["want_tokens"] / new
    if abs(b[dom]["epochs_fractional"] - round(want, 4)) > 1e-9:
        bad.append(f"{dom}.epochs_fractional {b[dom]['epochs_fractional']} != want/supply {want:.4f}")
    if round(a[dom]["epochs_fractional"], 2) != round(b[dom]["epochs_fractional"], 2):
        bad.append(f"{dom} epochs moved at 2dp: {a[dom]['epochs_fractional']:.2f} -> "
                   f"{b[dom]['epochs_fractional']:.2f}; the restatement was declared 2dp-neutral")
    return bad


def _demo():
    control, v2 = json.load(open(CONTROL, encoding="utf-8")), json.load(open(V2, encoding="utf-8"))
    assert not check(control, v2), check(control, v2)

    # NEGATIVE CONTROLS. Without these the check passes on a comparator that returns [] --
    # and "the files match" is exactly the answer a broken comparator gives for free.
    import copy

    w = copy.deepcopy(v2)
    w["domains"]["cot"]["weight"] = 0.0538          # what the generator would produce
    assert any("cot.weight" in c for c in check(control, w)), check(control, w)

    w = copy.deepcopy(v2)
    w["domains"]["code_py_rp1t"]["fingerprint"] = "0" * 16
    assert any("fingerprint" in c for c in check(control, w)), check(control, w)

    w = copy.deepcopy(v2)
    w["total_tokens"] = 20_000_000_000
    assert any("total_tokens" in c for c in check(control, w)), check(control, w)

    # the restated field itself must be checked for VALUE, not just waved through
    w = copy.deepcopy(v2)
    w["domains"]["math_owm_stage2"]["supply_tokens_one_epoch"] = 7_000_000_000
    assert check(control, w), "any supply value passes: the restatement is unchecked"

    # and epochs_fractional must track it rather than being free text
    w = copy.deepcopy(v2)
    w["domains"]["math_owm_stage2"]["epochs_fractional"] = 9.99
    assert any("epochs_fractional" in c for c in check(control, w)), check(control, w)

    # THE 2DP-NEUTRALITY CLAIM NEEDS ITS OWN CASE. The prereg wording says the restatement
    # leaves epochs unchanged to 2 decimals; 9.99 above trips the want/supply check first, so
    # it does not exercise that line. This value is consistent with want/supply for a supply
    # that would move the 2dp reading, which is the only way to reach it.
    w = copy.deepcopy(v2)
    _m = w["domains"]["math_owm_stage2"]
    _m["supply_tokens_one_epoch"] = int(_m["want_tokens"] / 1.40)
    _m["epochs_fractional"] = round(_m["want_tokens"] / _m["supply_tokens_one_epoch"], 4)
    assert any("2dp" in c for c in check(control, w)), (
        f"a restatement that moves epochs at 2dp is not caught: {check(control, w)}"
    )

    # a domain APPEARING (the dd09_full flip) must be caught, not silently allowed
    w = copy.deepcopy(v2)
    w["domains"]["code_rp1t_dd09_full"] = dict(w["domains"]["code_py_rp1t"])
    assert any("domain set differs" in c for c in check(control, w)), check(control, w)

    # AND THE RESTATED DOMAIN GETS NO EXEMPTION BEYOND ITS SUPPLY. Every control above edits
    # some OTHER domain, so all of them pass with ALLOWED widened to {"weight", "fingerprint"}
    # -- measured, that mutation survived the first version of this test. Two cases close it:
    # a weight edit inside math_owm_stage2 itself, and the list check.
    w = copy.deepcopy(v2)
    w["domains"][RESTATED[0]]["weight"] = 0.2176
    assert any(f"{RESTATED[0]}.weight" in c for c in check(control, w)), (
        f"a weight edit in the restated domain is swallowed: {check(control, w)}"
    )
    _saved = set(ALLOWED)
    try:
        ALLOWED.update({"weight", "fingerprint"})
        assert any("ALLOWED lets through" in c for c in check(control, v2)), (
            "widening ALLOWED to weight/fingerprint is not caught"
        )
    finally:
        ALLOWED.clear()
        ALLOWED.update(_saved)
    assert not check(control, v2), "ALLOWED was not restored"

    print("v2 mix self-test OK (one restated field, 9 negative controls)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _demo()
    else:
        c, v = json.load(open(CONTROL, encoding="utf-8")), json.load(open(V2, encoding="utf-8"))
        bad = check(c, v)
        if bad:
            sys.exit("v2 mix has drifted from resume 1:\n  " + "\n  ".join(bad))
        print("v2 mix == resume 1 + the math_owm supply restatement")

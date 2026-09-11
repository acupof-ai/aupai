#!/usr/bin/env python3
"""L3-shortfall reweight for data/mix_v41_gate.json (fb ruling 2026-09-11).

The L3 aggregate is the slow pole. If fewer L3 tokens land than the mix reads, the
file must not over-read the supply. Rule (fb), applied at 23:00Z 2026-09-12 to 0e's
kept_L3 count and a no-op when L3 is full:

  MAIN window (30B):
    w3  = min(0.30, kept_L3 / 27e9)
    short = 0.30 - w3
    w2  = 0.45 + min(short, kept_L2/27e9 - 0.45)     # L2 up to epoch ratio 1.0
    w_starcoder = 0.07 + (short - absorbed_by_L2)    # any remainder
    every other main weight unchanged.
  ANNEAL window (0.10 * 30B = 3B) follows the SAME shift on residual supply:
    a3  = min(0.40, (kept_L3 - 27e9*w3) / 3e9)
    a2  grows only within L2 supply not already consumed by the main window;
    the rest goes to starcoder anneal.
  Total L2 scheduled across both windows never exceeds kept_L2 (epoch ratio 1.0).

Supply cap matches train.py: scheduled POOL tokens (weight*budget +
anneal*anneal_budget, epochs NOT multiplied -- epochs is repetition of the same
pool) must be <= supply * epochs. If the shifted starcoder weight cannot fit its
supply the plan is INFEASIBLE: the dry-run exits 2 and --apply refuses.

Default is DRY-RUN: it prints the per-domain tokens/supply table and exits 0 only
for a feasible plan. Only --apply rewrites the mix (and refuses outside the gate
mix by basename).

# restartable: no shard loop -- default run is read-only and --apply is one atomic
# json rewrite of a single tracked file; an interrupt before that writes nothing and
# the command reruns deterministically from the same kept_l3 number.

    python3 scripts/reweight_l3_shortfall.py --kept_l3 9000000000
    python3 scripts/reweight_l3_shortfall.py --kept_l3 N --apply
"""
import argparse
import json
import os
import sys

# train.py build_mix splits total_tokens into (1-anneal_frac) main + anneal_frac
# anneal -- the SAME 30B row pool, not 30B+3B: 27B main + 3B anneal. See
# `rows = mix["total_tokens"] / Cfg.seq` and the phase loop `want = int(rows * frac
# * d.get(key, d["weight"]))`, with the per-phase cap `cap = int(len(pool) *
# d.get("epochs", 1)) - used[name]` (build_mix; grep these symbols, line numbers drift).
BUDGET = 30e9
MAIN_BUDGET = BUDGET * 0.90   # 27e9
ANNEAL_FRAC = 0.10
ANNEAL_BUDGET = BUDGET * ANNEAL_FRAC  # 3e9
MIX_REL = "data/mix_v41_gate.json"
# L2 kept tokens 0e reported 2026-09-11 (post keep-rules yield); override --kept_l2.
DEFAULT_KEPT_L2 = 16.57e9
# measured packed _dc supply (facts/corpus_supply.json
# #cs.gate_domains_decontaminated_tokenized_0911); used for the printed ratio.
SUPPLY_NONULTRA = {
    "code_py_starcoder_dc": 7_948_925_654,
    "math_owm_stage2_dc": 5_857_968_443,
    "code_keep_p1_dc": 2_626_975_915,
    "en_c4_stage2_dc": 1_985_209_544,
    "cot_dc": 399_994_207,
    "code_py_rp1t_dc": 379_599_341,
}
L2, L3, STAR = "code_ultra_l2_dc", "code_ultra_l3_dc", "code_py_starcoder_dc"


def reweight(kept_l3, kept_l2, mix):
    d = {k: dict(v) for k, v in mix["domains"].items()}

    # main window
    w3 = min(0.30, kept_l3 / MAIN_BUDGET)
    short = 0.30 - w3
    l2_main_cap = max(0.0, kept_l2 / MAIN_BUDGET - 0.45)
    to_l2 = min(short, l2_main_cap)
    w2 = 0.45 + to_l2
    w_star_extra = short - to_l2
    d[L3]["weight"] = round(w3, 6)
    d[L2]["weight"] = round(w2, 6)
    d[STAR]["weight"] = round(0.07 + w_star_extra, 6)

    # anneal window: residual L3 after the main reads; L2 is bounded by its COMBINED
    # schedule across both windows (main L2 + anneal L2 <= kept_l2, epoch ratio 1.0).
    rem_l3 = max(0.0, kept_l3 - MAIN_BUDGET * w3)
    a3 = min(0.40, rem_l3 / ANNEAL_BUDGET)
    rem_l2 = max(0.0, kept_l2 - MAIN_BUDGET * w2)
    a2 = min(0.30, rem_l2 / ANNEAL_BUDGET)
    # the other five domains' anneal sums to 0.25, so star closes whatever the two
    # ultra domains cannot fill (baseline: 0.75 - 0.30 - 0.40 = 0.05).
    a_star = 0.75 - a2 - a3
    d[L3]["anneal"] = round(a3, 6)
    d[L2]["anneal"] = round(a2, 6)
    d[STAR]["anneal"] = round(a_star, 6)

    out = dict(mix)
    out["domains"] = d
    return out


def supply_map(kept_l3, kept_l2):
    supply = dict(SUPPLY_NONULTRA)
    supply[L3] = kept_l3
    supply[L2] = kept_l2
    return supply


def infeasible(mix, kept_l3, kept_l2):
    """Oversubscribed domains under train.py's pool cap: scheduled POOL tokens
    (epochs NOT multiplied; epochs repeats the same pool) vs supply*epochs."""
    supply = supply_map(kept_l3, kept_l2)
    bad = []
    for k, v in mix["domains"].items():
        pool = v["weight"] * MAIN_BUDGET + v["anneal"] * ANNEAL_BUDGET
        cap = supply[k] * v.get("epochs", 1)
        if pool > cap * 1.0001:
            bad.append((k, pool, cap))
    return bad


def table(mix, kept_l3, kept_l2):
    supply = supply_map(kept_l3, kept_l2)
    ws = sum(v["weight"] for v in mix["domains"].values())
    asum = sum(v["anneal"] for v in mix["domains"].values())
    lines = ["domain                          w      main_tok     ann    ann_tok   scheduled    supply      ratio ep",
             "-" * 112]
    for k, v in mix["domains"].items():
        mt = v["weight"] * MAIN_BUDGET
        atok = v["anneal"] * ANNEAL_BUDGET
        tot = mt + atok
        ep = v.get("epochs", 1)
        ratio = f"{tot/(supply[k]*ep):.3f}" if supply[k] else "n/a"
        lines.append(f"{k:30s} {v['weight']:.4f} {mt/1e9:8.3f}B {v['anneal']:.4f} "
                     f"{atok/1e9:6.3f}B {tot/1e9:8.3f}B {supply[k]/1e9:8.3f}B {ratio:>7s} {ep}")
    lines.append(f"sum weights {ws:.4f}  sum anneal {asum:.4f}")
    return "\n".join(lines)


def run(kept_l3, kept_l2, root, apply):
    """Pure driver: returns (table_text, exit_code). main() and the selftest share it."""
    path = os.path.join(root, MIX_REL)
    mix = json.load(open(path, encoding="utf-8"))
    out = reweight(kept_l3, kept_l2, mix)
    text = table(out, kept_l3, kept_l2)
    ws = sum(v["weight"] for v in out["domains"].values())
    ans = sum(v["anneal"] for v in out["domains"].values())
    assert abs(ws - 1.0) < 1e-6 and abs(ans - 1.0) < 1e-6, (ws, ans)
    bad = infeasible(out, kept_l3, kept_l2)
    if bad:
        text += "\nINFEASIBLE: pool tokens exceed supply*epochs for: " + ", ".join(
            f"{k} ({pool/1e9:.3f}B>{cap/1e9:.3f}B)" for k, pool, cap in bad)
        return text, out, 2, path
    if apply:
        assert os.path.basename(path) == "mix_v41_gate.json", "refuse outside the gate mix"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)
        text += f"\napplied -> {path}"
    else:
        text += "\nfeasible dry-run; pass --apply to rewrite"
    return text, out, 0, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--kept_l3", type=float, required=True)
    ap.add_argument("--kept_l2", type=float, default=DEFAULT_KEPT_L2)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    text, _out, code, _path = run(a.kept_l3, a.kept_l2, a.root, a.apply)
    print(text)
    sys.exit(code)


_BASE_MIX = {"domains": {
    "code_ultra_l2_dc": {"weight": 0.45, "anneal": 0.30, "epochs": 1},
    "code_ultra_l3_dc": {"weight": 0.30, "anneal": 0.40, "epochs": 1},
    "code_py_starcoder_dc": {"weight": 0.07, "anneal": 0.05, "epochs": 1},
    "code_keep_p1_dc": {"weight": 0.03, "anneal": 0.03, "epochs": 1},
    "code_py_rp1t_dc": {"weight": 0.01, "anneal": 0.01, "epochs": 1},
    "math_owm_stage2_dc": {"weight": 0.08, "anneal": 0.10, "epochs": 1},
    "en_c4_stage2_dc": {"weight": 0.045, "anneal": 0.04, "epochs": 1},
    "cot_dc": {"weight": 0.015, "anneal": 0.07, "epochs": 3},
}}


def _selftest():
    import tempfile

    def sums(o):
        return (round(sum(v["weight"] for v in o["domains"].values()), 6),
                round(sum(v["anneal"] for v in o["domains"].values()), 6))

    # 1. full L3 (>= 8.1B main + 1.2B anneal = 9.3B): exact no-op, feasible.
    full = reweight(10.2e9, DEFAULT_KEPT_L2, _BASE_MIX)
    assert sums(full) == (1.0, 1.0), sums(full)
    for k in _BASE_MIX["domains"]:
        assert full["domains"][k]["weight"] == _BASE_MIX["domains"][k]["weight"], k
        assert full["domains"][k]["anneal"] == _BASE_MIX["domains"][k]["anneal"], k
    assert infeasible(full, 10.2e9, DEFAULT_KEPT_L2) == []

    # 2. zero L3 on the 27/3 basis (known answers): main shortfall .30 goes to L2 up
    #    to kept_l2 (w2 0.613704), the rest to star (ws .206296); no L3 in either
    #    phase, L2 main consumes all 16.57B so a2 0, star anneal .75.
    zero = reweight(0.0, DEFAULT_KEPT_L2, _BASE_MIX)["domains"]
    assert zero[L3]["weight"] == 0.0 and zero[L3]["anneal"] == 0.0
    assert abs(zero[L2]["weight"] - round(DEFAULT_KEPT_L2 / MAIN_BUDGET, 6)) < 1e-6
    assert zero[L2]["anneal"] == 0.0
    assert abs(zero[STAR]["weight"]
               - round(0.07 + 0.30 - (DEFAULT_KEPT_L2 / MAIN_BUDGET - 0.45), 6)) < 1e-6
    assert abs(zero[STAR]["anneal"] - 0.75) < 1e-6

    # 3. 6B kept on 27/3: w3 .222222, L2 absorbs .077778 of the main shift
    #    (ws stays .07); a3 0; main L2 14.25B leaves 2.32B -> a2 .30, star .45.
    mid = reweight(6e9, DEFAULT_KEPT_L2, _BASE_MIX)["domains"]
    assert abs(mid[L3]["weight"] - round(6e9 / 27e9, 6)) < 1e-6
    assert mid[L3]["anneal"] == 0.0
    assert abs(mid[L2]["weight"] - 0.527778) < 1e-6
    assert mid[L2]["anneal"] == 0.30
    assert mid[STAR]["weight"] == 0.07
    assert mid[STAR]["anneal"] == 0.45
    assert sums(reweight(6e9, DEFAULT_KEPT_L2, _BASE_MIX)) == (1.0, 1.0)

    # 4. cot never trips infeasible at epochs=3: pool .615B vs cap .4B*3=1.2B; the
    #    displayed ratio is pool/(supply*epochs) = .513 (27/3 basis), not the old 4.95.
    assert "0.513 3" in table(full, 10.2e9, DEFAULT_KEPT_L2)

    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "data"))
        mix_path = os.path.join(root, "data", "mix_v41_gate.json")
        json.dump(_BASE_MIX, open(mix_path, "w"))

        # 5. end-to-end --apply at kept_l3=6B (fb case): exit 0, file rewritten,
        #    vectors sum 1.0 and every domain within supply*epochs on the 27/3 basis.
        _t, o6, code6, p6 = run(6e9, DEFAULT_KEPT_L2, root, apply=True)
        assert code6 == 0, _t
        assert infeasible(o6, 6e9, DEFAULT_KEPT_L2) == []
        on_disk = json.load(open(p6))
        assert on_disk["domains"][L3]["weight"] == o6["domains"][L3]["weight"]
        assert round(sum(v["weight"] for v in on_disk["domains"].values()), 6) == 1.0
        assert round(sum(v["anneal"] for v in on_disk["domains"].values()), 6) == 1.0

        # 6. --apply at kept_l3=0 is feasible on 27/3 (star pool 7.82B < 7.949B):
        #    exit 0, and rerunning the same number is idempotent.
        _t2, o0, code0, _p0 = run(0.0, DEFAULT_KEPT_L2, root, apply=True)
        assert code0 == 0, _t2
        assert infeasible(o0, 0.0, DEFAULT_KEPT_L2) == []
        _t3, o0b, code0b, _ = run(0.0, DEFAULT_KEPT_L2, root, apply=True)
        assert code0b == 0 and o0b["domains"] == o0["domains"]

        # 7. forced infeasibility exits 2 and --apply does not touch the file:
        #    shrink starcoder supply below its zero-L3 pool (7.82B). Patch THIS
        #    module's globals (a fresh `import` here would load a second copy when
        #    the file runs as __main__).
        orig_star = SUPPLY_NONULTRA[STAR]
        SUPPLY_NONULTRA[STAR] = 4e9
        try:
            _t4, _o4, code4, _ = run(0.0, DEFAULT_KEPT_L2, root, apply=False)
            assert code4 == 2, _t4
            before = open(mix_path).read()
            _t5, _o5, code5, _ = run(0.0, DEFAULT_KEPT_L2, root, apply=True)
            assert code5 == 2 and open(mix_path).read() == before
        finally:
            SUPPLY_NONULTRA[STAR] = orig_star
    print("reweight_l3_shortfall selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()

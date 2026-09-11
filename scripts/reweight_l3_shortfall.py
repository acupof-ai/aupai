#!/usr/bin/env python3
"""L3-shortfall reweight for data/mix_v41_gate.json (fb ruling 2026-09-11).

The L3 aggregate is the slow pole. If fewer L3 tokens land than the mix reads, the
file must not over-read the supply. Rule (fb), applied at 23:00Z 2026-09-12 to 0e's
kept_L3 count and a no-op when L3 is full:

  MAIN window (30B):
    w3  = min(0.30, kept_L3 / 30e9)
    short = 0.30 - w3
    w2  = 0.45 + min(short, kept_L2/30e9 - 0.45)     # L2 up to epoch ratio 1.0
    w_starcoder = 0.07 + (short - absorbed_by_L2)    # any remainder
    every other main weight unchanged.
  ANNEAL window (0.10 * 30B = 3B) follows the SAME shift on residual supply:
    a3  = min(0.40, (kept_L3 - 30e9*w3) / 3e9)
    a2  grows only within L2 supply not already consumed by the main window;
    the rest goes to starcoder anneal.
  Total L2 scheduled across both windows never exceeds kept_L2 (epoch ratio 1.0).

Default is DRY-RUN: it prints the per-domain tokens/supply table and exits. Only
--apply rewrites the mix (and it refuses outside the gate mix by basename).

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

BUDGET = 30e9
ANNEAL_FRAC = 0.10
ANNEAL_BUDGET = BUDGET * ANNEAL_FRAC
MIX_REL = "data/mix_v41_gate.json"
# L2 kept tokens 0e reported 2026-09-11 (post keep-rules yield); override --kept_l2.
DEFAULT_KEPT_L2 = 16.57e9
# measured packed _dc supply (facts/corpus_supply.json
# #cs.gate_domains_decontaminated_tokenized_0911); used only for the printed ratio.
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
    w3 = min(0.30, kept_l3 / BUDGET)
    short = 0.30 - w3
    l2_main_cap = max(0.0, kept_l2 / BUDGET - 0.45)
    to_l2 = min(short, l2_main_cap)
    w2 = 0.45 + to_l2
    w_star_extra = short - to_l2
    d[L3]["weight"] = round(w3, 6)
    d[L2]["weight"] = round(w2, 6)
    d[STAR]["weight"] = round(0.07 + w_star_extra, 6)

    # anneal window: residual L3 after the main reads; L2 is bounded by its COMBINED
    # schedule across both windows (main L2 + anneal L2 <= kept_l2, epoch ratio 1.0).
    rem_l3 = max(0.0, kept_l3 - BUDGET * w3)
    a3 = min(0.40, rem_l3 / ANNEAL_BUDGET)
    rem_l2 = max(0.0, kept_l2 - BUDGET * w2)
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


def table(mix, kept_l3):
    supply = dict(SUPPLY_NONULTRA)
    supply[L3] = kept_l3
    rows = []
    ws = sum(v["weight"] for v in mix["domains"].values())
    asum = sum(v["anneal"] for v in mix["domains"].values())
    for k, v in mix["domains"].items():
        main_tok = v["weight"] * BUDGET
        ann_tok = v["anneal"] * ANNEAL_BUDGET
        tot = main_tok + ann_tok
        sup = supply.get(k)
        if k == L2:
            sup = DEFAULT_KEPT_L2
        reads = tot * v.get("epochs", 1)
        ratio = f"{reads/sup:.3f}" if sup else "n/a"
        rows.append((k, v["weight"], main_tok, v["anneal"], ann_tok, tot,
                     sup or 0, ratio, v.get("epochs", 1)))
    lines = ["domain                          w      main_tok     ann    ann_tok   scheduled    supply      ratio ep",
             "-" * 112]
    for k, w, mt, a, at, tot, sup, ratio, ep in rows:
        lines.append(f"{k:30s} {w:.4f} {mt/1e9:8.3f}B {a:.4f} {at/1e9:6.3f}B "
                     f"{tot/1e9:8.3f}B {sup/1e9:8.3f}B {ratio:>7s} {ep}")
    lines.append(f"sum weights {ws:.4f}  sum anneal {asum:.4f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--kept_l3", type=float, required=True, help="0e kept L3 tokens")
    ap.add_argument("--kept_l2", type=float, default=DEFAULT_KEPT_L2)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    path = os.path.join(a.root, MIX_REL)
    mix = json.load(open(path, encoding="utf-8"))
    out = reweight(a.kept_l3, a.kept_l2, mix)
    print(table(out, a.kept_l3))
    if a.apply:
        assert os.path.basename(path) == "mix_v41_gate.json", "refuse to rewrite anything but the gate mix"
        ws = sum(v["weight"] for v in out["domains"].values())
        ans = sum(v["anneal"] for v in out["domains"].values())
        assert abs(ws - 1.0) < 1e-6 and abs(ans - 1.0) < 1e-6, (ws, ans)
        # epoch-ratio guard: scheduled main+anneal must not exceed supply, except cot
        # whose epochs=3 is the deliberate repeat (ruling). Refuse, never over-read.
        supply = dict(SUPPLY_NONULTRA)
        supply[L3] = a.kept_l3
        supply[L2] = a.kept_l2
        for k, v in out["domains"].items():
            scheduled = (v["weight"] * BUDGET + v["anneal"] * ANNEAL_BUDGET) * v.get("epochs", 1)
            assert scheduled <= supply[k] * 1.0001, (k, scheduled, supply[k])
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)
        print(f"applied -> {path}")
    else:
        print("dry-run only; pass --apply to rewrite")


def _selftest():
    mix = {"domains": {
        "code_ultra_l2_dc": {"weight": 0.45, "anneal": 0.30, "epochs": 1},
        "code_ultra_l3_dc": {"weight": 0.30, "anneal": 0.40, "epochs": 1},
        STAR: {"weight": 0.07, "anneal": 0.05, "epochs": 1},
        "code_keep_p1_dc": {"weight": 0.03, "anneal": 0.03, "epochs": 1},
        "code_py_rp1t_dc": {"weight": 0.01, "anneal": 0.01, "epochs": 1},
        "math_owm_stage2_dc": {"weight": 0.08, "anneal": 0.10, "epochs": 1},
        "en_c4_stage2_dc": {"weight": 0.045, "anneal": 0.04, "epochs": 1},
        "cot_dc": {"weight": 0.015, "anneal": 0.07, "epochs": 3},
    }}

    def sums(o):
        return (round(sum(v["weight"] for v in o["domains"].values()), 6),
                round(sum(v["anneal"] for v in o["domains"].values()), 6))

    # 1. full L3 (>= 9B main + 1.2B anneal headroom): exact no-op
    full = reweight(10.2e9, DEFAULT_KEPT_L2, mix)
    assert sums(full) == (1.0, 1.0), sums(full)
    for k in mix["domains"]:
        assert full["domains"][k]["weight"] == mix["domains"][k]["weight"], k
        assert full["domains"][k]["anneal"] == mix["domains"][k]["anneal"], k

    # 2. zero L3: main shortfall .30; L2 main absorbs to 16.57B cap, rest to star.
    #    Combined L2 (main+anneal) <= kept_l2, epoch ratio 1.0: main already at
    #    16.57B leaves zero L2 anneal, so all anneal goes to star.
    zero = reweight(0.0, DEFAULT_KEPT_L2, mix)["domains"]
    assert zero[L3]["weight"] == 0.0 and zero[L3]["anneal"] == 0.0
    assert abs(zero[L2]["weight"] - round(DEFAULT_KEPT_L2 / BUDGET, 6)) < 1e-6
    assert zero[L2]["anneal"] == 0.0, zero[L2]["anneal"]
    assert abs(zero[STAR]["weight"] - (0.07 + 0.30 - (DEFAULT_KEPT_L2 / BUDGET - 0.45))) < 1e-6
    assert abs(zero[STAR]["anneal"] - 0.75) < 1e-6

    # 3. 6B kept: main w3 .20 (all 6B in main, 0 L3 anneal); main L2 16.5B leaves
    #    0.07B -> a2 .0233; star closes the rest at .7267. Combined L2 <= supply.
    mid = reweight(6e9, DEFAULT_KEPT_L2, mix)["domains"]
    assert mid[L3]["weight"] == 0.20 and mid[L3]["anneal"] == 0.0
    assert abs(mid[L2]["weight"] - 0.55) < 1e-6
    assert abs(mid[L2]["anneal"] - round(0.07 / 3.0, 6)) < 1e-6
    assert abs(mid[STAR]["weight"] - 0.07) < 1e-6  # all main shortfall fit in L2
    assert abs(mid[STAR]["anneal"] - (0.75 - 0.07 / 3.0)) < 1e-6
    assert sums(reweight(6e9, DEFAULT_KEPT_L2, mix)) == (1.0, 1.0)
    print("reweight_l3_shortfall selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()

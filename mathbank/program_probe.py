#!/usr/bin/env python3
"""Per-program solve-rate probe: sample instances, later score model generations.

Pipeline (no checkpoint needed for step 1):
  1. sample: draw K verified instances per program -> data/rl/program_probe.jsonl
     rows: {program_id, level, instruction, answer}  (RL format, no reference solution)
  2. on a GPU box: generate one greedy + k sampled answers per row, write
     jsonl {instruction, greedy: str, gens: [str, ...]} (or one row per
     generation with greedy_gen on the greedy row),
  3. score: -> data/rl/program_rates.jsonl (per-program diagnostics) plus
     data/rl/instance_rates.jsonl (per-instance pass@k — the SELECTION file:
     RL keeps instances in the 20-80% band, never whole programs).
     program_rates rows: {program_id, level, n, pass_at_1, pass_at_k,
     inst_var, all_or_none}; pass_at_1 uses the greedy gen (falls back to
     first sampled with a warning); inst_var flags programs whose parameter
     range is too wide (all_or_none==n => every instance degenerate).

The 20-80% solve-rate band (LFM-1.3B-Math / DAPO) is the RL-usable range;
programs at 0% or 100% carry ~no gradient.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_math_short import load_programs, num, verify  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE_PATH = os.path.join(ROOT, "data", "rl", "program_probe.jsonl")
RATES_PATH = os.path.join(ROOT, "data", "rl", "program_rates.jsonl")
INST_PATH = os.path.join(ROOT, "data", "rl", "instance_rates.jsonl")


def sample(k, seed):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from holdout import is_holdout  # noqa: E402

    bank = load_programs()
    seen, n_out, n_held = set(), 0, 0
    os.makedirs(os.path.dirname(PROBE_PATH), exist_ok=True)
    with open(PROBE_PATH, "w", encoding="utf-8") as f:
        for lev in sorted(bank):
            for name, fn in bank[lev]:
                got = 0
                for i in range(k * 20):  # headroom for rejects/dups/holdout
                    if got >= k:
                        break
                    rng = random.Random(f"probe-{seed}-{name}-{i}")
                    try:
                        ins, lines, ans = fn(rng)
                    except Exception:
                        continue
                    if not verify(ins, lines, ans)[1] or ins in seen:
                        continue
                    if is_holdout(ins):
                        n_held += 1
                        continue
                    seen.add(ins)
                    f.write(json.dumps({"program_id": name, "level": lev,
                                        "instruction": ins, "answer": num(ans)},
                                       ensure_ascii=False) + "\n")
                    got += 1
                    n_out += 1
    print(f"sampled {n_out} instances from {sum(len(v) for v in bank.values())} programs "
          f"({n_held} holdout hits filtered) -> {PROBE_PATH}")


def score(gen_path):
    sys.path.insert(0, ROOT)
    from algorithms.rlvr_reward import reward_fn, extract_boxed  # noqa: E402
    from eval.math_zh import ANS_RE  # noqa: E402

    def is_ok(gen, gold):
        if extract_boxed(gen) is not None:
            return reward_fn(gen, gold)
        m = ANS_RE.search(gen)
        return reward_fn(f"\\boxed{{{m.group(1).strip()}}}", gold) if m else 0.0

    gold, gens, greedy = {}, defaultdict(list), {}
    for line in open(PROBE_PATH, encoding="utf-8"):
        r = json.loads(line)
        gold[r["instruction"]] = (r["program_id"], r["level"], str(r["answer"]))
    n_warn = 0
    for line in open(gen_path, encoding="utf-8"):
        r = json.loads(line)
        if r["instruction"] not in gold:
            continue
        if r.get("greedy") is not None:
            greedy[r["instruction"]] = r["greedy"]
        elif r.get("greedy_gen"):
            greedy[r["instruction"]] = r["greedy_gen"]
        gs = r.get("gens")
        if gs:
            gens[r["instruction"]].extend(gs)
        elif "gen" in r:
            gens[r["instruction"]].append(r["gen"])
        if r["instruction"] not in greedy:
            n_warn += 1  # pass@1 falls back to first sampled gen (noise, see 7d34ac4)

    # per program: [greedy_correct, n_inst, rates(list of per-instance pass@k)]
    # per instance rows are the SELECTION quantity (RL filters instances, not
    # programs — a wide-range program can have some instances in-band and some
    # degenerate); program-level aggregates are diagnostics only.
    agg = defaultdict(lambda: [0, 0, []])
    inst_rows = []
    for ins, (pid, lev, ans) in gold.items():
        gs = gens.get(ins, [])
        if not gs:
            continue
        oks = [is_ok(g, ans) for g in gs]
        g = greedy.get(ins)
        g_ok = is_ok(g, ans) if g is not None else oks[0]
        a = agg[(pid, lev)]
        a[0] += g_ok
        a[1] += 1
        a[2].append(sum(oks) / len(oks))
        inst_rows.append({"program_id": pid, "level": lev, "instruction": ins,
                          "answer": ans, "greedy_ok": g_ok,
                          "pass_at_k": round(sum(oks) / len(oks), 4)})
    with open(RATES_PATH, "w", encoding="utf-8") as f:
        for (pid, lev), (p1, n, rates) in sorted(agg.items()):
            mean = sum(rates) / len(rates)
            var = sum((r - mean) ** 2 for r in rates) / len(rates)
            f.write(json.dumps({
                "program_id": pid, "level": lev, "n": n,
                "pass_at_1": round(p1 / n, 4),
                "pass_at_k": round(mean, 4),
                "inst_var": round(var, 4),
                "all_or_none": sum(1 for r in rates if r in (0.0, 1.0)),
            }, ensure_ascii=False) + "\n")
    with open(INST_PATH, "w", encoding="utf-8") as f:
        for r in inst_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    scored = sum(1 for a in agg.values() if a[1])
    print(f"scored {scored} programs, {len(inst_rows)} instances "
          f"-> {RATES_PATH} + {INST_PATH}")
    if n_warn:
        print(f"[warn] {n_warn} instances lack a greedy gen; pass@1 used first sampled gen")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--k", type=int, default=8)
    s.add_argument("--seed", type=int, default=11)
    g = sub.add_parser("score")
    g.add_argument("gens", help="jsonl with instruction + gen (one per row) or gens (list)")
    a = ap.parse_args()
    if a.cmd == "sample":
        sample(a.k, a.seed)
    else:
        score(a.gens)

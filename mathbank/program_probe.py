#!/usr/bin/env python3
"""Per-program solve-rate probe: sample instances, later score model generations.

Pipeline (no checkpoint needed for step 1):
  1. sample: draw K verified instances per program -> data/rl/program_probe.jsonl
     rows: {program_id, level, instruction, answer}  (RL format, no reference solution)
  2. on a GPU box: generate K answers per row with any harness, append as
     `gens: [str, ...]` (or write one row per generation as {program_id, gen}),
  3. score: aggregate per program -> data/rl/program_rates.jsonl
     rows: {program_id, level, n, pass_at_1, pass_at_k}  (k = gens per instance)

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


def sample(k, seed):
    bank = load_programs()
    seen, n_out = set(), 0
    os.makedirs(os.path.dirname(PROBE_PATH), exist_ok=True)
    with open(PROBE_PATH, "w", encoding="utf-8") as f:
        for lev in sorted(bank):
            for name, fn in bank[lev]:
                got = 0
                for i in range(k * 4):  # 4x headroom for rejects/dups
                    if got >= k:
                        break
                    rng = random.Random(f"probe-{seed}-{name}-{i}")
                    try:
                        ins, lines, ans = fn(rng)
                    except Exception:
                        continue
                    if not verify(ins, lines, ans)[1] or ins in seen:
                        continue
                    seen.add(ins)
                    f.write(json.dumps({"program_id": name, "level": lev,
                                        "instruction": ins, "answer": num(ans)},
                                       ensure_ascii=False) + "\n")
                    got += 1
                    n_out += 1
    print(f"sampled {n_out} instances from {sum(len(v) for v in bank.values())} programs -> {PROBE_PATH}")


def score(gen_path):
    sys.path.insert(0, ROOT)
    from algorithms.rlvr_reward import reward_fn, extract_boxed  # noqa: E402
    from eval.math_zh import ANS_RE  # noqa: E402

    def is_ok(gen, gold):
        if extract_boxed(gen) is not None:
            return reward_fn(gen, gold)
        m = ANS_RE.search(gen)
        return reward_fn(f"\\boxed{{{m.group(1).strip()}}}", gold) if m else 0.0

    gold, gens = {}, defaultdict(list)
    for line in open(PROBE_PATH, encoding="utf-8"):
        r = json.loads(line)
        gold[r["instruction"]] = (r["program_id"], r["level"], str(r["answer"]))
    for line in open(gen_path, encoding="utf-8"):
        r = json.loads(line)
        if r["instruction"] not in gold:
            continue
        gens[r["instruction"]].extend(r.get("gens") or [r["gen"]])

    agg = defaultdict(lambda: [0, 0, 0, 0])  # program -> [p1, pk, n, ksum]
    for ins, (pid, lev, ans) in gold.items():
        gs = gens.get(ins, [])
        if not gs:
            continue
        oks = [is_ok(g, ans) for g in gs]
        a = agg[(pid, lev)]
        a[0] += oks[0]
        a[1] += int(any(oks))
        a[2] += 1
        a[3] += len(oks)
    with open(RATES_PATH, "w", encoding="utf-8") as f:
        for (pid, lev), (p1, pk, n, ksum) in sorted(agg.items()):
            f.write(json.dumps({"program_id": pid, "level": lev, "n": n,
                                "k": round(ksum / n, 1),
                                "pass_at_1": round(p1 / n, 4),
                                "pass_at_k": round(pk / n, 4)},
                               ensure_ascii=False) + "\n")
    scored = sum(1 for a in agg.values() if a[2])
    print(f"scored {scored} programs -> {RATES_PATH}")


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

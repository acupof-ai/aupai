#!/usr/bin/env python3
"""Short-solution batch: minimal-but-complete solutions for SFT (ref 2502.12143).

Loads only math_programs_short_l{3,4}.py, draws verified instances with the
same per-program cap as the main bank. "Short" means every arithmetic step is
present but nothing else: no restating the question, no meta commentary —
the chain label carries the justification (e.g. 假设全是兔的腿数 = ...).
Key steps stay two-operand so scripts/eqcheck.py covers them (it skips
3+-operand chains). No instruction-length floor: 速算 problems are inherently
short. Output rows match the main-batch schema {instruction, output, level}
so build_corpus needs no special case.

Usage: python run_short_sol.py <N> <out.jsonl> [--seed S]
"""

import argparse
import importlib
import json
import random
import sys
from collections import Counter

from run_math_short import MAX_INST, verify

MODULES = ["math_programs_short_l3", "math_programs_short_l4"]


def load_short():
    out = {}
    for stem in MODULES:
        try:
            mod = importlib.import_module(stem)
        except Exception as e:
            print(f"[warn] skip {stem}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        for level, name, fn in mod.PROGRAMS:
            out.setdefault(level, []).append((f"{stem}:{name}", fn))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    bank = load_short()
    rng = random.Random(args.seed)
    rows, seen, rejected = [], set(), 0
    for lev in sorted(bank):
        progs = bank[lev]
        rng.shuffle(progs)
        counts = {name: 0 for name, _ in progs}
        keep = draw = 0
        target = args.n // 2  # equal split L3/L4
        while keep < target:
            progressed = False
            for name, fn in progs:
                if counts[name] >= MAX_INST or keep >= target:
                    continue
                srng = random.Random(f"{args.seed}-short-{lev}-{name}-{draw}")
                draw += 1
                try:
                    ins, lines, ans = fn(srng)
                except Exception:
                    rejected += 1
                    continue
                out, ok = verify(ins, lines, ans)
                # no instruction-length floor: 速算/简算 problems ("用简便方法计算：101×61")
                # are inherently short; the floor was a main-bank anti-triviality rule.
                if not ok:
                    rejected += 1
                    continue
                if ins in seen:
                    continue
                seen.add(ins)
                counts[name] += 1
                rows.append({"instruction": ins, "output": out, "level": lev})
                keep += 1
                progressed = True
            if not progressed:
                print(f"[warn] {lev}: stalled at {keep}/{target}", file=sys.stderr)
                break

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dist = Counter(r["level"] for r in rows)
    print(f"wrote {len(rows)} rows -> {args.out} | level_dist={dict(dist)} "
          f"reject_rate={rejected / (rejected + len(rows)):.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()

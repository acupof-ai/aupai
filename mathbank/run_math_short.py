#!/usr/bin/env python3
"""Instantiate the math program bank -> verified short-format jsonl batches.

Each program is `fn(rng) -> (instruction, lines[list[str]], ans:int|Fraction)`.
lines are per-line equations `X op Y = Z` (Z may carry a Chinese unit). Every line
is independently verified: LHS is safely evaluated and must equal the RHS value.
The LAST line's value must equal `num(ans)` — tying the chain to the exact answer.
Rows are tagged `level`, deduped by instruction, written with a stats row.

Usage:
  python run_math_short.py <N> <out.jsonl> [--seed S]
  python run_math_short.py --list       # program counts per level
"""

import argparse
import importlib
import json
import random
import re
import sys
from collections import Counter
from fractions import Fraction

from mathcommon import eval_lhs, num

RATIOS = {"L1": 0.15, "L2": 0.35, "L3": 0.35, "L4": 0.15}
MAX_INST = 150          # peer hard line: instance cap per program
TOK = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def load_programs():
    import glob, os
    base = ["math_programs_l1", "math_programs_l2", "math_programs_l3", "math_programs_l4"]
    ext = [os.path.splitext(os.path.basename(f))[0]
           for f in glob.glob(os.path.join(os.path.dirname(__file__), "math_programs_l*_ext*.py"))]
    out = {}
    for stem in base + sorted(ext):
        try:
            mod = importlib.import_module(stem)
        except Exception as e:  # don't let one broken/half-written module kill loading
            print(f"[warn] skip {stem}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        for level, name, fn in mod.PROGRAMS:
            out.setdefault(level, []).append((f"{stem}:{name}", fn))
    return out


def verify(instruction, lines, ans):
    """Return (output_string, ok). Every equation verified; last must == num(ans).

    Also enforces prose<->equation consistency: any integer >=3 mentioned in the
    instruction (excluding percentages) must appear somewhere among the equation
    numbers/substrings. Catches the 'prose says /4 but equations divide by 3'
    class of bug the numeric check alone misses."""
    if not lines:
        return None, False
    # prose numbers (>=3), excluding percentages, must appear among equation values
    pct_clipped = re.sub(r"\d+\s*(?:％|%)", "", instruction)
    inst_nums = {int(m) for m in re.findall(r"-?\d+", pct_clipped) if int(m) >= 3}
    line_tokens = " ".join(tok for ln in lines for tok in TOK.findall(ln))
    for n in inst_nums:
        if str(n) not in line_tokens:
            return None, False  # prose number absent from any equation -> suspect
    last_numeric = None
    for ln in lines:
        if "=" not in ln:
            return None, False
        parts = [p.strip() for p in ln.split("=")]
        if len(parts) == 2:          # "X op Y = Z[u]"
            lhs, rhs = parts
        elif len(parts) == 3:        # "label = X op Y = Z[u]"
            lhs, rhs = parts[1], parts[2]
        else:
            return None, False
        m = TOK.search(rhs.replace(" ", ""))
        if not m:
            return None, False
        rhs_val = float(Fraction(m.group(0)))
        try:
            lhs_val = eval_lhs(lhs)
        except Exception:
            return None, False
        if abs(lhs_val - rhs_val) > 1e-6 * max(1, abs(lhs_val)):
            return None, False
        last_numeric = m.group(0)
    if Fraction(last_numeric) != Fraction(ans):
        return None, False
    return "\n".join(lines).rstrip() + f"\n答案是：\\boxed{{{num(ans)}}}", True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    bank = load_programs()
    if args.list:
        for lev in sorted(bank):
            print(f"{lev}: {len(bank[lev])} programs")
        return
    if not args.n or not args.out:
        ap.error("n and out are required (unless --list)")

    rng = random.Random(args.seed)
    rows, seen, rejected = [], set(), 0
    used = {lev: set() for lev in RATIOS}
    attempts = {lev: 0 for lev in RATIOS}

    for lev, frac in RATIOS.items():
        target = int(args.n * frac)
        progs = bank.get(lev, [])
        if not progs:
            print(f"[warn] no programs for {lev}", file=sys.stderr)
            continue
        rng.shuffle(progs)
        counts = {name: 0 for name, _ in progs}
        keep, draw = 0, 0
        while keep < target:
            progressed = False
            for name, fn in progs:
                if counts[name] >= MAX_INST or keep >= target:
                    continue
                srng = random.Random(f"{args.seed}-{lev}-{name}-{draw}")
                draw += 1
                attempts[lev] += 1
                try:
                    ins, lines, ans = fn(srng)
                except Exception:
                    rejected += 1
                    continue
                out, ok = verify(ins, lines, ans)
                if not ok:
                    rejected += 1
                    continue
                if len(out) < (30 if lev == "L1" else 40 if lev == "L2" else 50):  # peer ruling: L1 >=30, L2 >=40, L3+ >=50
                    rejected += 1
                    continue
                if ins in seen:
                    continue
                seen.add(ins)
                counts[name] += 1
                used[lev].add(name)
                rows.append({"instruction": ins, "output": out, "level": lev})
                keep += 1
                progressed = True
            if not progressed:
                print(f"[warn] {lev}: stalled at {keep}/{target} "
                      f"(all programs at cap or rejecting)", file=sys.stderr)
                break

    level_dist = Counter(r["level"] for r in rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(rows) or 1
    nprogs = sum(len(used[lev]) for lev in RATIOS)
    print(f"wrote {len(rows)} rows -> {args.out}", file=sys.stderr)
    print(f"STATS programs={nprogs} "
          f"inst_avg={total / max(1, nprogs):.1f} "
          f"level_dist={dict(level_dist)} "
          f"reject_rate={rejected / (rejected + len(rows)):.4f} "
          f"attempts={sum(attempts.values())}", file=sys.stderr)


if __name__ == "__main__":
    main()
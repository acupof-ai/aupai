#!/usr/bin/env python3
"""The M1 decomposition table: which region carries the arm's missing throughput points.

Reads the rows scripts/mem_decomp_run.sh writes and prints ms per region per cell, each
against the `off` cell. Charter: docs/standards/memory_layers_0905.md.

THE QUESTION. M1 ran 64K tok/s/gpu against the control's 82K -- 225 ms of extra step at
1024 vs 799 ms -- while the isolated lookup measures 51.6 ms. 173 ms is unaccounted, and a
share of a step is not a place, so this splits the excess by region and by what it scales
with.

WHAT THE CELLS SEPARATE. One pool is shared by layers 3, 6, 9 (model.py:451), which is what
makes l1 informative rather than merely smaller: per-layer terms -- the lookup, the k x k
grid, the touched/key_hits bookkeeping -- divide by three when only layer 6 is pooled, while
the table's own terms -- dense gradient, Adagrad state, the optimizer's read -- do not move
at all, because they cover the same 1,048,576 rows however many layers wrote them. m2 then
shrinks the table with the layer count held fixed. Read together:

    l1 ~ off, m2 ~ m1     the cost is per-layer
    l1 ~ m1, m2 ~ off     the cost is the table
    both intermediate     both terms are real, and the table gives their sizes

k16 halves top_k, which moves the gather and the candidate grid and nothing else.

    python3 scripts/mem_decomp_table.py runs/mem_decomp_0905.jsonl
    python3 scripts/mem_decomp_table.py --selftest    # known answers, no run needed
"""
import argparse
import json
import os
import sys

# The cells, in the order the runner writes them, keyed by their memory config. A row is
# matched on (mem_values, mem_top_k, mem_layers) rather than on file order: a rerun of one
# cell appends, and reading by position would then label every row after it wrongly.
CELLS = [
    ("off", 0, 32, "3,6,9"),
    ("m1", 1048576, 32, "3,6,9"),
    ("k16", 1048576, 16, "3,6,9"),
    ("l1", 1048576, 32, "6"),
    ("m2", 262144, 32, "3,6,9"),
]
REGIONS = ("forward", "backward", "opt_step", "loader_wait", "step_total")


def key_of(row):
    return (row.get("mem_values", 0), row.get("mem_top_k", 32), str(row.get("mem_layers", "")))


def pick(rows, values, top_k, layers):
    """The LAST row matching a cell's config. A rerun appends rather than replacing, and the
    later row is the one whose conditions the report describes."""
    hits = [r for r in rows if key_of(r) == (values, top_k, str(layers))]
    return hits[-1] if hits else None


def med(row, region):
    if not row:
        return None
    st = row.get(region)
    return st["median_ms"] if st else None


def table(rows):
    out = []
    ref = pick(rows, 0, 32, "3,6,9")
    out.append(f"{'cell':6s} " + " ".join(f"{r:>12s}" for r in REGIONS) + f" {'tok/s/gpu':>10s}")
    for name, v, k, ly in CELLS:
        row = pick(rows, v, k, ly)
        if row is None:
            out.append(f"{name:6s} MISSING -- the cell did not write a row (check its rc in the log)")
            continue
        cols = []
        for r in REGIONS:
            m = med(row, r)
            if m is None:
                cols.append(f"{'--':>12s}")
                continue
            base = med(ref, r) if ref is not None and name != "off" else None
            cols.append(f"{m:8.1f}{'':4s}" if base is None else f"{m:8.1f}{m - base:+5.0f}")
        out.append(f"{name:6s} " + " ".join(cols) + f" {row.get('tok_s_per_gpu') or 0:10,d}")

    if ref is None:
        out.append("\nNO off CELL: every delta above is absent, and an absolute ms figure "
                   "answers no question this table was built to ask.")
        return "\n".join(out)

    # The verdict, stated as an arithmetic the reader can check rather than a label. Every
    # delta is arm-minus-off on the same shape, so it is the memory's cost in that region.
    m1 = pick(rows, 1048576, 32, "3,6,9")
    l1 = pick(rows, 1048576, 32, "6")
    m2 = pick(rows, 262144, 32, "3,6,9")
    tot = med(m1, "step_total")
    off = med(ref, "step_total")
    if tot and off:
        out.append(f"\nM1 costs {tot - off:.1f} ms/step ({100 * (1 - off / tot):.1f}% of throughput). "
                   f"The isolated lookup measured 51.6 ms for three pools.")
        for r in ("forward", "backward", "opt_step"):
            d = (med(m1, r) or 0) - (med(ref, r) or 0)
            out.append(f"  {r:10s} {d:+7.1f} ms")
        # PER-LAYER vs PER-TABLE, from the two cells built to separate them. Reported as the
        # measured fractions, not as a verdict word: "per-layer" is a claim about how a cost
        # scales, and the numbers are what say it.
        if l1 and med(l1, "step_total"):
            f = (med(l1, "step_total") - off) / (tot - off)
            out.append(f"\nl1 keeps {100 * f:.0f}% of M1's cost with one pooled layer instead of "
                       f"three. A purely per-layer cost would keep ~33%, a purely per-table one "
                       f"~100%.")
        if m2 and med(m2, "step_total"):
            f = (med(m2, "step_total") - off) / (tot - off)
            out.append(f"m2 keeps {100 * f:.0f}% of M1's cost at a quarter of the table. Its own "
                       f"ratio to the control is {(m2.get('tok_s_per_gpu') or 0) / 82000:.2f} "
                       f"(readout 5 wants >= 0.85 to launch).")
    return "\n".join(out)


def _selftest():
    """Known answers on synthetic rows. No run needed."""
    bad = 0

    def row(v, k, ly, fwd, bwd, opt, tot, tps):
        return {"mem_values": v, "mem_top_k": k, "mem_layers": ly,
                "forward": {"median_ms": fwd}, "backward": {"median_ms": bwd},
                "opt_step": {"median_ms": opt}, "step_total": {"median_ms": tot},
                "loader_wait": {"median_ms": 1.0}, "tok_s_per_gpu": tps}

    rows = [row(0, 32, "3,6,9", 300, 450, 40, 800, 82000),
            row(1048576, 32, "3,6,9", 360, 560, 60, 1024, 64000)]
    t = table(rows)
    ok = "+224.0 ms" not in t and "M1 costs 224.0 ms/step" in t and "21.9% of throughput" in t
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the excess and the throughput cost come out of the rows")

    ok = ("forward      +60.0 ms" in t and "backward    +110.0 ms" in t
          and "opt_step     +20.0 ms" in t)
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} each region's delta is arm minus off")

    # A cell that did not run must SAY so. A missing row silently skipped would leave a table
    # whose absent line reads as "no cost" -- the one reading that is never true.
    ok = "k16    MISSING" in t and "l1     MISSING" in t and "m2     MISSING" in t
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} a cell with no row is named, never omitted")

    # The l1/m2 fractions, on rows built so the answer is known: l1 keeps a third (per-layer),
    # m2 keeps all of it (per-table).
    rows2 = rows + [row(1048576, 32, "6", 320, 487, 60, 875, 74000),
                    row(262144, 32, "3,6,9", 360, 560, 60, 1024, 64000)]
    t2 = table(rows2)
    ok = "l1 keeps 33%" in t2 and "m2 keeps 100%" in t2
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} l1 and m2 fractions read off the step totals ({ok})")

    # A rerun appends; the LAST matching row wins. Reading by position would report the first.
    rows3 = rows + [row(1048576, 32, "3,6,9", 360, 560, 60, 900, 72000)]
    ok = "M1 costs 100.0 ms/step" in table(rows3)
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} a rerun of a cell supersedes the earlier row")

    # No control: the deltas cannot exist, and the table must refuse rather than print
    # absolute milliseconds that look like costs.
    ok = "NO off CELL" in table([rows[1]])
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} without the off cell the table says so instead of "
          "printing bare absolutes")

    print(f"mem_decomp_table selftest: {6 - bad}/6 pass")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="runs/mem_decomp_0905.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not os.path.exists(a.path):
        print(f"no such file: {a.path}", file=sys.stderr)
        return 1
    with open(a.path, encoding="utf-8") as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()]
    print(table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())

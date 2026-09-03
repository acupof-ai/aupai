#!/usr/bin/env python3
"""Twelve held-out items, side by side, so the aggregate can be checked against what it is made of.

# restartable: reads runs/e1_29_per_item.jsonl and the held-out text. No GPU, no model, seconds.

WHY A CASE TABLE AT ALL. Section 5.3c says the floor gap is 1.51x on English and 2.50x on Chinese.
Those are byte-weighted sums over 10,421 items and they are not falsifiable by reading them. Twelve
items with both arms' per-item loss are: a reader can see whether "the control is worse on Chinese"
means what it sounds like, or whether it is an artifact of, say, every Chinese item being short.

THE SELECTION IS FIXED IN CODE WITH A FIXED SEED, and it is stratified so it cannot be a highlight
reel: 4 where WE do best relative to the control, 4 where we do WORST relative to it, 4 uniformly
at random. The middle group is the one that matters -- an all-random sample of 12 from a population
where we win on average would mostly show us winning, which teaches nothing about the failure mode.

"OUR 4 WORST" IS NAMED AS SUCH. It is not "the control wins most": on a population where the
aggregate gap is 2.00x, the four items where our advantage is smallest may still be items we win.
Whether the control actually wins any item is a fact to be read off the table, not assumed from the
group's name -- and the printed table says which arm won each row.

THE CLASSES ARE MINE, from measured content (scripts/e1_29_floor_by_class.py). The data's own `src`
field labels 99.8% code_general while 8.3% of answers contain code, so it cannot carry a
stratification. Every row prints the measured class, never `src`.

PER-ITEM NLL IS PER-BYTE HERE. The raw per-item nll is a sum over that item's tokens, so it is
mostly a length measurement -- item A having twice the nll of item B usually means A is twice as
long. Dividing by the item's own supervised bytes makes rows comparable to each other and to the
1.51x / 2.50x aggregates, which are also per-byte.
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

SEED = 20260903
N_PER_GROUP = 4


def load_per_item(path):
    """{id: {arm: {"nll", "tokens", "cls"}}} from the jsonl both arms wrote."""
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.setdefault(int(d["id"]), {})[d["arm"]] = d
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_item", default="runs/e1_29_per_item.jsonl")
    ap.add_argument("--text", default="data/sft/control_sft_text_heldout.jsonl")
    ap.add_argument("--out", default="runs/e1_30_case_table.json")
    ap.add_argument("--md_out", default="runs/e1_30_case_table.md")
    ap.add_argument("--chars", type=int, default=180)
    a = ap.parse_args()

    import eval_heldout as E

    pi_path = os.path.join(ROOT, a.per_item)
    if not os.path.exists(pi_path):
        sys.exit(f"REFUSING: {a.per_item} absent. It is written by "
                 f"scripts/e1_29_floor_by_class.py --per_item_out; the first run of that script "
                 f"kept only class sums, which is why this table could not be built from it.")
    per = load_per_item(pi_path)
    both = {i for i, v in per.items() if "ours" in v and "control" in v}
    if not both:
        sys.exit("REFUSING: no id has rows for both arms")

    rows = {int(r[0]): (r[1], r[2]) for r in E.read_text(os.path.join(ROOT, a.text))}
    missing = both - set(rows)
    if missing:
        sys.exit(f"REFUSING: {len(missing)} scored ids are absent from {a.text}, e.g. "
                 f"{sorted(missing)[:5]} -- the table would print a loss without its text")

    # PER-BYTE, AND THE BYTES ARE EACH ARM'S OWN COMPLETION. The two arms format the pair
    # differently (ChatML for the control), so a single byte count for both would be wrong for one
    # of them -- and the cross-arm ratio would inherit that error.
    items = []
    for i in sorted(both):
        q, ans = rows[i]
        ob = len(E.format_pair("ours", q, ans)[1].encode("utf-8"))
        cb = len(E.format_pair("control", q, ans)[1].encode("utf-8"))
        if not ob or not cb:
            continue
        onb = per[i]["ours"]["nll"] / ob
        cnb = per[i]["control"]["nll"] / cb
        items.append({"id": i, "cls": per[i]["ours"].get("cls"),
                      "ours_nll_per_byte": onb, "ctrl_nll_per_byte": cnb,
                      "ratio": cnb / onb if onb else None,
                      "ours_bytes": ob, "ctrl_bytes": cb,
                      "ours_tokens": per[i]["ours"]["tokens"],
                      "ctrl_tokens": per[i]["control"]["tokens"],
                      "question": q, "answer": ans})

    # RATIO, NOT DIFFERENCE, because the aggregates being illustrated are ratios. Sorting by
    # difference would rank long items first: a 0.1 nats/byte edge on a 4 KB item and on a 200 B
    # item are the same ratio and very different differences.
    by_ratio = sorted(items, key=lambda r: r["ratio"])
    best = by_ratio[-N_PER_GROUP:][::-1]          # our advantage largest
    worst = by_ratio[:N_PER_GROUP]                # our advantage smallest
    picked = {r["id"] for r in best} | {r["id"] for r in worst}
    pool = [r for r in items if r["id"] not in picked]
    rnd = random.Random(SEED).sample(pool, min(N_PER_GROUP, len(pool)))

    groups = [("our advantage largest", best),
              ("our advantage smallest", worst),
              (f"uniform random (seed {SEED})", sorted(rnd, key=lambda r: r["id"]))]

    def clip(s):
        s = " ".join(s.split())
        return s[:a.chars] + ("..." if len(s) > a.chars else "")

    out = {"seed": SEED, "n_per_group": N_PER_GROUP, "source": a.per_item,
           "population": len(items),
           "classes_defined_by": "e1, measured content -- NOT the data's `src` field, which labels "
                                 "99.8% code_general while 8.3% of answers contain code",
           "note": "nll per SUPERVISED BYTE, each arm over its own formatting. The raw per-item nll "
                   "is a sum over that item's tokens and would mostly measure length.",
           "groups": {}}
    md = ["# e1-30 case table: 12 held-out items with both arms' per-item loss", "",
          f"Population {len(items):,} items scored by both arms. Selection fixed in "
          f"`scripts/e1_30_case_table.py`, seed {SEED}. **nll per supervised byte**, each arm over "
          f"its own formatting.", "",
          "`ratio` = ctrl / ours: above 1 means WE did better on that item. "
          "\"Our advantage smallest\" is not \"the control wins\" -- read the `winner` column.", ""]

    ctrl_wins = 0
    for label, rs in groups:
        out["groups"][label] = []
        md += [f"## {label}", "",
               "| id | class | ours | ctrl | ratio | winner | bytes (ours/ctrl) |",
               "|---|---|---|---|---|---|---|"]
        for r in rs:
            win = "ours" if r["ratio"] > 1 else "control"
            if win == "control":
                ctrl_wins += 1
            out["groups"][label].append({k: v for k, v in r.items() if k != "answer"}
                                        | {"answer_head": clip(r["answer"]),
                                           "question_head": clip(r["question"]), "winner": win})
            md.append(f"| {r['id']} | {r['cls']} | {r['ours_nll_per_byte']:.4f} | "
                      f"{r['ctrl_nll_per_byte']:.4f} | {r['ratio']:.3f} | {win} | "
                      f"{r['ours_bytes']:,} / {r['ctrl_bytes']:,} |")
        md.append("")
        for r in rs:
            md += [f"**{r['id']}** ({r['cls']}, ratio {r['ratio']:.3f}, winner {win})",
                   "", f"> Q: {clip(r['question'])}", "", f"> A: {clip(r['answer'])}", ""]

    out["control_wins_in_table"] = ctrl_wins
    out["control_wins_in_population"] = sum(1 for r in items if r["ratio"] <= 1)
    out["population_share_control_wins"] = out["control_wins_in_population"] / len(items)
    md += ["## What the table does not show", "",
           f"The control wins **{out['control_wins_in_population']:,} of {len(items):,}** items "
           f"({100 * out['population_share_control_wins']:.1f}%) in the full population, and "
           f"{ctrl_wins} of the 12 rows here. A 12-row table cannot establish that share -- it is "
           f"computed over everything and stated here so the table is not read as the population.",
           ""]

    with open(os.path.join(ROOT, a.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    with open(os.path.join(ROOT, a.md_out), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"population {len(items):,} items; control wins "
          f"{out['control_wins_in_population']:,} ({100 * out['population_share_control_wins']:.1f}%)")
    for label, rs in groups:
        print(f"  {label}: " + ", ".join(f"{r['id']}({r['cls']},{r['ratio']:.2f})" for r in rs))
    print(f"wrote {a.out} and {a.md_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

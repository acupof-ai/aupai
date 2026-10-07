#!/usr/bin/env python3
"""v42 SFT recipe 1007 §3.1/§3.2 bucket 3: code_if 30k + aqua_rat MC short-answer 20k.

# restartable: pure-CPU single pass over two local files, ~1-2 min; an interrupt only
# costs the rerun (writes go to <name>.part and atomically rename, so a partial output
# never survives as the artifact).

Two ChatML-ready (bare {instruction,output,source}; the packer renders ChatML) files:

  code_if_mc/code_if_30k.jsonl   30,000 code instruction->body pairs sampled from the
                                 already-decontaminated data/sft/code_if_pairs_dc_train.jsonl.
                                 Length-stratified so the answer-length distribution follows
                                 recipe §3.3 (small 614M-active student: mostly short).

  code_if_mc/mc_aqua_20k.jsonl  20,000 A-E multiple-choice math questions from MathInstruct's
                                 aqua_rat, rewritten to a SHORT letter answer ("The answer is
                                 X.") with the choices kept in the instruction. This bucket
                                 teaches "give the option letter" format for the EN MC gates
                                 (ARC/OpenBookQA). It deliberately keeps the letter answer --
                                 unlike d3's math_en, which drops letter-MC because it cannot
                                 restore a numeric answer; here the letter IS the target.

Hard rules from the recipe:
  - answer length capped short (MC is the cheap format-leader; no long R1 chains);
  - one terminal answer form per bucket, letter for MC ("The answer is X.");
  - exact-dedup on normalised instruction; deterministic seed; no ChatML tokens in the jsonl.

Output dir default data/sft_open_preview/code_if_mc (handed to d3 for mix aggregation).
"""
import argparse
import json
import os
import random
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_NORM = re.compile(r"\s+")
_CHATML = re.compile(r"<\|[A-Za-z0-9_]+\|>")
_LETTER = re.compile(r"answer is\s*(?:option\s*)?\(?([A-E])\)?\.?\s*$", re.IGNORECASE)
# split "question ... Answer Choices: (A) .. (B) .." into stem + choices
_CHOICES = re.compile(r"answer choices\s*:\s*", re.IGNORECASE)
SEED = 42


def norm(s):
    return _NORM.sub(" ", s).strip()


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_code_if(src, n, seed):
    """Length-stratified sample of code pairs. Stratify by OUTPUT line count so the
    draw is dominated by short/medium bodies (§3.3: 70% in the short band), not the
    rare long function. Within a stratum choose uniformly (deterministic seed)."""
    rows = []
    seen = set()
    for d in load_jsonl(src):
        q = (d.get("prompt") or "").strip()
        a = (d.get("output") or "").strip()
        if not q or not a or _CHATML.search(q) or _CHATML.search(a):
            continue
        if len(a) > 3000 or len(a) < 20:  # short bucket for the small student; drop tails
            continue
        k = norm(q)
        if k in seen:
            continue
        seen.add(k)
        rows.append({"instruction": q, "output": a, "source": "code_if_pairs_dc"})
    rng = random.Random(seed)
    rng.shuffle(rows)
    # size buckets by output chars, weighted to short
    bands = [(0, 240), (240, 700), (700, 1500), (1500, 3001)]
    weights = [0.45, 0.35, 0.15, 0.05]
    pools = {i: [] for i in range(len(bands))}
    for r in rows:
        L = len(r["output"])
        for i, (lo, hi) in enumerate(bands):
            if lo <= L < hi:
                pools[i].append(r)
                break
    out, used = [], set()
    for i, w in enumerate(weights):
        want = round(n * w)
        cand = [r for r in pools[i] if norm(r["instruction"]) not in used]
        take = cand[:want]
        for r in take:
            used.add(norm(r["instruction"]))
        out.extend(take)
    # top up to n from any unused pool if a band ran short
    if len(out) < n:
        rest = [r for r in rows if norm(r["instruction"]) not in used]
        out.extend(rest[: n - len(out)])
    out = out[:n]
    return out


def build_mc_aqua(mathinstruct, n, seed):
    """aqua_rat rows -> (question with choices, short 'The answer is X.')."""
    rng = random.Random(seed + 1)
    pool = []
    seen = set()
    for d in load_jsonl(mathinstruct):
        if "aqua_rat" not in d.get("source", ""):
            continue
        instr = (d.get("instruction") or "").strip()
        out = (d.get("output") or "").strip()
        m = _LETTER.search(out)
        if not m:
            continue
        parts = _CHOICES.split(instr, maxsplit=1)
        if len(parts) != 2:
            continue
        stem, choices = parts[0].strip(), parts[1].strip()
        letter = m.group(1).upper()
        # require the chosen letter to actually be present among the choices
        if f"({letter})" not in choices:
            continue
        if not stem or len(stem) > 1200 or len(choices) > 900:
            continue
        k = norm(stem)
        if k in seen or _CHATML.search(instr):
            continue
        seen.add(k)
        q = f"{stem}\nAnswer Choices: {choices}"
        pool.append({"instruction": q, "output": f"The answer is {letter}.",
                     "source": "aqua_rat_mc_short"})
    rng.shuffle(pool)
    return pool[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_if_src", default=os.path.join(ROOT, "data/sft/code_if_pairs_dc_train.jsonl"))
    ap.add_argument("--mathinstruct", default=os.path.join(ROOT, "data/raw/sft_open/mathinstruct/data.jsonl"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "data/sft_open_preview/code_if_mc"))
    ap.add_argument("--n_code_if", type=int, default=30000)
    ap.add_argument("--n_mc", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    ci = build_code_if(a.code_if_src, a.n_code_if, a.seed)
    mc = build_mc_aqua(a.mathinstruct, a.n_mc, a.seed)
    if len(ci) < a.n_code_if:
        raise SystemExit(f"only {len(ci)} code_if rows after filtering (want {a.n_code_if})")
    if len(mc) < a.n_mc:
        raise SystemExit(f"only {len(mc)} aqua MC rows after filtering (want {a.n_mc})")

    p1 = os.path.join(a.out_dir, "code_if_30k.jsonl")
    p2 = os.path.join(a.out_dir, "mc_aqua_20k.jsonl")
    write_jsonl(p1, ci)
    write_jsonl(p2, mc)

    import statistics
    for name, rows in (("code_if_30k", ci), ("mc_aqua_20k", mc)):
        al = [len(r["output"]) for r in rows]
        letters = {}
        for r in mc if name == "mc_aqua_20k" else []:
            L = r["output"][-2]
            letters[L] = letters.get(L, 0) + 1
        print(f"{name}: n={len(rows)} out_chars med={statistics.median(al):.0f} "
              f"p90={sorted(al)[int(0.9*len(al))]} max={max(al)}"
              + (f" letter_dist={dict(sorted(letters.items()))}" if letters else ""))
    print(f"wrote {p1}\nwrote {p2}")


if __name__ == "__main__":
    main()

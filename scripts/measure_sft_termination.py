#!/usr/bin/env python3
"""fb's SFT-termination underdetermination measurement (2026-08-30).

SFT drives degenerate repetition 24.7% -> 55.8% (8-gram repeat >=3x); the model
emits boilerplate then loops, never reaching the answer. Hypothesis: the SFT
pack teaches the SAME (near-dup) question multiple answer TERMINATION shapes, so
termination is underdetermined -- the FoNE rule "one format per prompt" checked
on number answers, never verified on code/math SFT text.

Measure (CPU, over the source SFT jsonl pairs):
  1. MinHash near-dup cluster the INSTRUCTIONS (union-find over LSH band keys,
     so we get cluster ids, not just a seen-bool).
  2. For each answer, extract a structural SHAPE (fence? latex-env? display-math?
     preamble prose? head marker? terminal family).
  3. Per cluster: count distinct shapes. A cluster with >1 shapes taught that
     question multiple terminations -> underdetermined.

Pre-registered read (fb): high fraction of clusters with shapes>1 -> termination
underdetermined, fix = normalize each cluster to one shape at pack time; ~0 -> rule
it out (termination is not data-underdetermination).

Usage: python3 scripts/measure_sft_termination.py [--limit N]
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datagen.build_corpus import MinHashLSH  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = [
    os.path.join(ROOT, "data", "alpaca_gpt4_zh.jsonl"),
    os.path.join(ROOT, "data", "coig.jsonl"),
    os.path.join(ROOT, "data", "openo1_sft.jsonl"),
    os.path.join(ROOT, "data", "gsm8k_zh.jsonl"),
    os.path.join(ROOT, "data", "school_math_r1_zh.jsonl"),
    os.path.join(ROOT, "data", "s1k.jsonl"),
    os.path.join(ROOT, "data", "synthetic", "code_python_zh.jsonl"),
    os.path.join(ROOT, "data", "synthetic", "knowledge_qa_zh.jsonl"),
    os.path.join(ROOT, "data", "synthetic", "math_gsm8k_zh.jsonl"),
]

_FENCE = re.compile(r"```|~~~")
_LATEX_ENV = re.compile(r"\\(begin|end)\{")
_DISPLAY_MATH = re.compile(r"\\\[|\\\]|\$\$")
_PREAMBLE = re.compile(r"^(首先|让我|需要|这道|这题|这是一|解题|答案|解析|思路|先|我们)")
_STAR_HEAD = re.compile(r"^\*\*")


def terminal_family(s):
    s2 = s.rstrip()
    if _FENCE.search(s2[-6:]):
        return "fence-close"
    m = re.search(r"\\end\{(.*?)\}\s*$", s2)
    if m:
        return "env-close:" + m.group(1)
    if re.search(r"\d+\s*[元个米只条千米角分岁天件台张]$", s2):
        return "unit"
    if re.search(r"\d+(\.\d+)?\s*$", s2):
        return "number"
    if s2.endswith(("。", "！", "？", "．")):
        return "cjk-punct"
    if s2.endswith("**"):
        return "bold-close"
    if re.search(r"(答案|解答|参考答案)[：:]\s*$", s2):
        return "answer-lbl"
    if s2.endswith((">", ")")):
        return "close-delim"
    return "other"


def shape(a):
    return (
        bool(_FENCE.search(a)),
        bool(_LATEX_ENV.search(a)),
        bool(_DISPLAY_MATH.search(a)),
        bool(_PREAMBLE.match(a)),
        bool(_STAR_HEAD.match(a)),
        terminal_family(a),
    )


def cluster_instructions(questions):
    """Union-find cluster over MinHash band keys -> list[cid] per question."""
    lsh = MinHashLSH(perms=128, bands=16)
    parent = list(range(len(questions)))
    # band_key -> (question_idx, band_idx) representative
    key_owner = defaultdict(set)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # store signatures hashed by band
    sig_of = []
    for i, q in enumerate(questions):
        import struct

        sig = lsh.signature(q)
        bands = [
            struct.pack(f"{lsh.rows}Q", *sig[lsh.rows * b : lsh.rows * (b + 1)])
            for b in range(lsh.bands)
        ]
        sig_of.append(bands)
    for i, bands in enumerate(sig_of):
        for b, key in enumerate(bands):
            if key in key_owner:
                union(i, next(iter(key_owner[key])))
            key_owner[key].add(i)
    return [find(i) for i in range(len(questions))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    pairs = []  # (q, out)
    for path in SOURCES:
        if not os.path.exists(path):
            print(f"SKIP missing {path}", file=sys.stderr)
            continue
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                q = (d.get("instruction") or "").strip()
                out = (d.get("output") or "").strip()
                if d.get("input"):
                    q = f"{q}\n{d['input']}"
                if not q or not out:
                    continue
                pairs.append((q, out))
                n += 1
                if a.limit and len(pairs) >= a.limit:
                    break
        print(f"  {os.path.basename(path)}: {n}", flush=True)

    print(f"total pairs: {len(pairs)}", flush=True)
    qs = [q for q, _ in pairs]
    cids = cluster_instructions(qs)
    print(f"clustered: {len(cids)} questions, {len(set(cids))} clusters", flush=True)

    # keep only clusters with >=2 members (near-dup families; singletons are not underdetermined)
    fam = defaultdict(list)
    for i, c in enumerate(cids):
        fam[c].append(i)
    families = {c: m for c, m in fam.items() if len(m) >= 2}

    shapes_per = {}
    for c, members in families.items():
        shapes = set(shape(pairs[i][1]) for i in members)
        shapes_per[c] = (len(members), shapes)

    # aggregates
    under = {c: v for c, v in shapes_per.items() if len(v[1]) > 1}
    sizes = [v[0] for v in shapes_per.values()]
    mean_shapes = sum(len(v[1]) for v in shapes_per.values()) / max(1, len(shapes_per))
    print(f"near-dup families (>=2): {len(shapes_per)} ({len(shapes_per)/max(1,len(set(cids))):.1%} of clusters)")
    print(f"  members total: {sum(sizes)}, mean cluster size {sum(sizes)/max(1,len(sizes)):.1f}")
    print(f"  families with >1 answer shape: {len(under)} = {len(under)/max(1,len(shapes_per)):.1%}")
    print(f"  mean distinct shapes per family: {mean_shapes:.2f}")
    for c in list(under)[:8]:
        m, sh = shapes_per[c]
        print(f"    cluster {c} ({m} members): {len(sh)} shapes; shapes=",
              sorted((f"fence={s[0]} latex={s[1]} disp={s[2]} pre={s[3]} star={s[4]} term={s[5]}" for s in sh)))
    # per-feature diversity
    term_div = sum(1 for v in shapes_per.values() if len({s[5] for s in v[1]}) > 1)
    print(f"  families with >1 terminal family: {term_div}")


if __name__ == "__main__":
    main()
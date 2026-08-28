#!/usr/bin/env python3
"""Measure duplicates / template reuse in web text — the content-farm signal.

Two failure classes aupai-fb hand-tagged on 180 web samples:
  - content-farm paragraphs: the same paragraph (or near-) repeats inside one doc
  - cross-doc template reuse: 洗稿/模板 farms reuse identical paragraphs across
    many documents

Both are measurable surface-statistically (character-bigram jaccard), the same
way dist_check measures the eval-adjacent axes. Per project law the probes are
self-validated on hand-built cases BEFORE the full counts are trusted.

  python scripts/repeat_check.py --dir data/corpus/web   # full run, reject histogram
  python scripts/repeat_check.py --selftest              # validate the probes first
"""
import collections
import json
import os
import re
import sys
import argparse

def _paragraphs(text):
    """Split into blocks on blank-ish lines; a doc with no blank lines stays one block."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n|\n", text) if b.strip()]
    return blocks if len(blocks) > 1 else [text.strip()]

def _shingles(s):
    s = "".join(s.split())
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else (set(s) if s else set())

def _jac(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def doc_internal_dup_ratio(text, span=None, thr=0.7):
    """Fraction of paragraph pairs that are near-duplicates of each other.

    A normal document has near-zero crossing-topic paragraph similarity; a
    content-farm page repeats its block, so a big share of pairs collide.
    """
    paras = _paragraphs(text)
    if len(paras) < 2:
        return 0.0
    sh = [_shingles(p) for p in paras]
    n = len(paras)
    pairs = 0
    dupe = 0
    # cap the cross-product when a doc has many paragraphs
    for i in range(n):
        js = range(i + 1, n)
        if span and span < n:
            js = range(i + 1, min(n, i + span + 1))
        for j in js:
            pairs += 1
            if _jac(sh[i], sh[j]) >= thr:
                dupe += 1
    return dupe / max(1, pairs)


def doc_paragraph_templates(pargs):
    """Bigram shingles per paragraph, for the cross-doc pass."""

def cross_doc_template(texts, thr=0.8):
    """global paragraph pool -> docs sharing near-identical paragraphs (template reuse)."""
    pool = []
    doc_of = []
    for di, t in enumerate(texts):
        for p in _paragraphs(t):
            pool.append(_shingles(p))
            doc_of.append(di)
    # near-dup search across docs (skip same doc)
    hit_docs = collections.Counter()  # doc -> how many shared-near-dup partners
    for i in range(len(pool)):
        if not pool[i]:
            continue
        for j in range(i + 1, len(pool)):
            if doc_of[i] == doc_of[j]:
                continue
            if _jac(pool[i], pool[j]) >= thr:
                hit_docs[doc_of[i]] += 1
                hit_docs[doc_of[j]] += 1
    return hit_docs


def scan_dir(path, max_docs=None):
    rows = []
    n = 0
    for fn in sorted(os.listdir(path)):
        if not fn.endswith(".jsonl"):
            continue
        for line in open(os.path.join(path, fn), encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                text = d.get("content") or d.get("text")
            except Exception:
                continue
            if not text:
                continue
            rows.append((fn, text))
            n += 1
            if max_docs and n >= max_docs:
                return rows
    return rows


def _selftest():
    # a normal, coherent document: no internal repeats
    normal = "".join(
        "这是一个自然段，介绍某个事情的背景。\n\n这是接着讲的第二段，继续陈述原因。\n\n"
        "第三段总结结论，与前文连贯。\n\n最后一段收尾。"
    )
    # content farm: same 段落 thrice
    farm = "某商品促销信息，限时抢购，一律八折优惠，不要错过。\n\n" * 4
    r_n = doc_internal_dup_ratio(normal, span=2, thr=0.7)
    r_f = doc_internal_dup_ratio(farm, thr=0.7)
    print(f"selftest normal dup_ratio={r_n:.2f} (want near 0), farm={r_f:.2f} (want high)")
    assert r_n < 0.3, f"normal doc mis-flagged as duplicated: {r_n}"
    assert r_f > 0.5, f"content farm not caught: {r_f}"
    # cross-doc: two docs sharing an identical paragraph
    shared = "这是一个被多家站点重复使用的模板段落，用于站群推广与 SEO 堆砌。"
    docA = "A 的开头\n\n" + shared
    docB = "B 的开头\n\n" + shared
    hits = cross_doc_template([docA, docB], thr=0.8)
    print(f"selftest cross_doc hits={dict(hits)} (want both docs to have a partner)")
    assert hits[0] >= 1 and hits[1] >= 1, "cross-doc template reuse not caught"
    print("repeat_check self-test OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/corpus/web")
    ap.add_argument("--max_docs", type=int, default=None)
    ap.add_argument("--max_docs_prose", type=int, default=2000)
    a = ap.parse_args()

    rows = scan_dir(a.dir, a.max_docs)
    print(f"scanned {len(rows)} docs")
    dup_r = [doc_internal_dup_ratio(t, span=4, thr=0.7) for _, t in rows]
    dup_r = sorted(dup_r)
    med = dup_r[len(dup_r) // 2]
    p90 = dup_r[int(len(dup_r) * 0.9)]
    hi = sum(1 for v in dup_r if v >= 0.3)
    print(f"doc-internal dup_ratio: median {med:.2f} p90 {p90:.2f} | docs with dup_ratio>=0.3: {hi} ({hi / len(rows):.1%})")
    # cross-doc template on a sample
    from random import Random
    rng = Random(0)
    sample = rng.sample(rows, min(a.max_docs_prose, len(rows)))
    hits = cross_doc_template([t for _, t in sample], thr=0.8)
    shared_docs = sum(1 for v in hits.values() if v >= 1)
    print(f"cross-doc template reuse (sample {len(sample)}): docs sharing a near-dup paragraph: {shared_docs} ({shared_docs / len(sample):.1%})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
    else:
        main()
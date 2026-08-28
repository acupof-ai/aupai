#!/usr/bin/env python3
"""Audit opencsg/Fineweb-Edu-Chinese-V2.1 parquet — do NOT trust the data card.

Same metrics aupai-fb wants measured on every candidate corpus, our web as the
reference. One parquet path is enough to see the shape; extra parquets expand
the source/score spread.

  python3 audit_ocsg.py /work/fwe/000000.parquet
"""
import collections, json, os, re, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.repeat_check import doc_internal_dup_ratio  # noqa: E402

SPAM = re.compile(
    r"(彩票|賭場|赌场|赌博|博彩|真人娱乐|北京赛车|时时彩|老虎机|六合彩|百家乐|开户送|注册送"
    r"|威廉希尔|德赢vwin|杏彩|凯发k8|明陞|m88asia|BOSS真人|森林舞会游戏|助赢|大智彩票"
    r"|加微信|QQ[:：]?\d{6,}|微信[:：]?[a-zA-Z0-9_]{5,}|电话[:：]?1[3-9]\d{9}"
    r"|阴道|裸体|情趣用品|一夜情|约炮)"
)
SPLIT = re.compile(r"[\s，,。；;]")
SENT = re.compile(r"[。！？；!?;]")


def repetition(text):
    segs = [s.strip() for s in SPLIT.split(text) if len(s.strip()) > 4]
    return 0.0 if len(segs) < 8 else 1 - len(set(segs)) / len(segs)


def fragmented(text):
    sents = [s for s in SENT.split(text) if s.strip()]
    if len(sents) < 6:
        return False, 0.0
    mean = sum(len(s) for s in sents) / len(sents)
    return mean <= 10.0, mean


def main():
    import pyarrow.parquet as pq
    rows = pq.read_table(sys.argv[1] if len(sys.argv) > 1 else "/work/fwe/000000.parquet").to_pylist()
    n = len(rows)
    print(f"== audit {sys.argv[1]} ({n} rows)")
    scores = sorted(float(r["score"]) for r in rows)
    print(f"score  med {scores[n // 2]:.3f} p90 {scores[int(n * .9)]:.3f} | >=0.5 {(sum(1 for s in scores if s >= .5) / n):.1%} | >=0.7 {(sum(1 for s in scores if s >= .7) / n):.1%}")
    print(f"source {dict(collections.Counter(r['source'] for r in rows).most_common(4))}")
    lens = sorted(len(r["text"]) for r in rows)
    print(f"doclen med {lens[n // 2]} p90 {lens[int(n * .9)]}")
    spam = sum(1 for r in rows if SPAM.search(r["text"])) / n
    rep = sorted(repetition(r["text"]) for r in rows)
    frags = [fragmented(r["text"]) for r in rows]
    frag_n = sum(1 for ok, _ in frags if ok) / n
    scent = [mean for _, mean in frags if _]
    print(f"SPAM          {spam:.2%}")
    print(f"repetition    med {rep[n // 2]:.2f} p90 {rep[int(n * .9)]:.2f}")
    print(f"fragmented(short-sent) {frag_n:.2%} | sentlen med {statistics.median(scent):.1f}")
    # traditional-char rate (t2s table)
    tab = {int(k) for k in json.load(open("/work/aupai/data/t2s_table.json"))}
    tc = 0; cc = 0
    for r in rows[:5000]:
        t = r["text"]; cc += len(t); tc += sum(1 for ch in t if ord(ch) in tab)
    print(f"traditional-char {tc / max(1, cc):.2%}")
    # internal dup ratio sample
    dr = [doc_internal_dup_ratio(r["text"], span=6, thr=0.7) for r in rows[:2000]]
    dr = sorted(dr)
    print(f"doc-internal-dup med {dr[len(dr) // 2]:.2f} p90 {dr[int(len(dr) * .9)]:.2f}")
    # contamination vs eval holdsets: index OpenCSG shingles, query eval problems >=0.85
    from scripts.repeat_check import _shingles, _jac
    ev = []
    for p in ("/work/aupai/data/synthetic/math_hard_eval_1k.jsonl", "/work/aupai/data/eval/math_test_500.jsonl"):
        for l in open(p, encoding="utf-8"):
            try: ev.append(json.loads(l).get("instruction"))
            except: pass
    inv = collections.defaultdict(list)
    for i, r in enumerate(rows):
        for g in _shingles(r["text"]): inv[g].append(i)
    contam = 0; ex = []
    for q in ev:
        if not q: continue
        qs = _shingles(q); cnt = collections.Counter()
        for g in qs:
            for i in inv[g]: cnt[i] += 1
        cand = [i for i, c in cnt.items() if c >= 4]
        best = max((_jac(qs, _shingles(rows[i]["text"])), i) for i in cand) if cand else (0, None)
        if best[0] >= 0.85:
            contam += 1
            if len(ex) < 5: ex.append((round(best[0], 2), q[:30]))
    print(f"contam: {contam}/{len(ev)} eval problems have a ~verbatim match in OpenCSG text (>=0.85)")
    for j, q in ex: print(f"   J{j} {q}")
    print("AUDIT_DONE")


if __name__ == "__main__":
    main()
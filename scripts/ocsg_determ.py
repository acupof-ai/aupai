#!/usr/bin/env python3
"""Deterministic quality axes of an opencsg Fineweb-Edu-Chinese parquet, no heavy contam.

Same surface metrics as audit_ocsg but without the O(n) contamination cross-join,
so it runs to completion quickly and doesn't fight the trainer for CPU. Contamination
is a separate tiny pass (each eval problem as a query, ~seconds).
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
FLUSH = dict(flush=True)


def repetition(text):
    segs = [s.strip() for s in SPLIT.split(text) if len(s.strip()) > 4]
    return 0.0 if len(segs) < 8 else 1 - len(set(segs)) / len(segs)


def sentlen(text):
    ss = [s for s in SENT.split(text) if s.strip()]
    return (sum(len(s) for s in ss) / len(ss)) if len(ss) >= 6 else 0.0


def main():
    import pyarrow.parquet as pq

    rows = pq.read_table(sys.argv[1]).to_pylist()
    n = len(rows)
    print(f"== {sys.argv[1]} ({n} rows)", **FLUSH)
    scores = sorted(float(r["score"]) for r in rows)
    print(
        f"score  med {scores[n // 2]:.3f} p90 {scores[int(n * 0.9)]:.3f} >=0.5 {(sum(1 for s in scores if s >= 0.5) / n):.1%} >=0.7 {(sum(1 for s in scores if s >= 0.7) / n):.1%}",
        **FLUSH,
    )
    print(f"source {dict(collections.Counter(r['source'] for r in rows).most_common(3))}", **FLUSH)
    lens = sorted(len(r["text"]) for r in rows)
    print(f"doclen  med {lens[n // 2]} p90 {lens[int(n * 0.9)]}", **FLUSH)
    print(f"SPAM    {(sum(1 for r in rows if SPAM.search(r['text'])) / n):.2%}", **FLUSH)
    rep = sorted(repetition(r["text"]) for r in rows)
    print(f"repetition med {rep[n // 2]:.2f} p90 {rep[int(n * 0.9)]:.2f}", **FLUSH)
    sent = sorted(sentlen(r["text"]) for r in rows)
    print(
        f"sentlen med {statistics.median(sent):.1f} | short(<6sent) {sum(1 for s in sent if not s) / n:.1%}",
        **FLUSH,
    )
    tab_path = "/work/aupai/data/t2s_table.json"
    tab = {int(k) for k in json.load(open(tab_path))}
    tc = cc = 0
    for r in rows:
        t = r["text"]
        cc += len(t)
        tc += sum(1 for ch in t if ord(ch) in tab)
    print(f"traditional-char {tc / max(1, cc):.2%}", **FLUSH)
    dr = sorted(doc_internal_dup_ratio(r["text"], span=6, thr=0.7) for r in rows)
    print(f"doc-internal-dup med {dr[len(dr) // 2]:.2f} p90 {dr[int(len(dr) * 0.9)]:.2f}", **FLUSH)
    print("DETERM_DONE", **FLUSH)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rebuild data/corpus/web into a cleaned, quality-ranked corpus.

Four filters, in order of how certain each one is. The certain ones run
unconditionally; the classifier only decides the last cut, and its threshold is
set from a measured AUC rather than a wish.

1. Traditional -> Simplified. 59.4% of the fineweb2 Chinese slice is traditional
   and the corpus was never converted, so the model learns two scripts for one
   language and the tokenizer wastes slots on both. scripts/t2s_corpus.py has
   existed the whole time and was never applied to web.
2. Gambling / contact / adult spam by keyword. Hits 2.5% of documents. Small,
   but every hit is unambiguous, and several are keywords injected mid-sentence
   into otherwise ordinary text, which is worse than an obvious ad.
3. Within-document repetition. Product sheets and content farms repeat blocks
   verbatim; forum-fragment splices repeat nothing but are made of very short
   pieces. Both are measured here, both are structural, neither needs a model.
4. The educational-quality classifier, last and softest.

    python datagen/clean_web.py --scores data/web_scores.npy --keep 0.40 \\
        --out data/corpus/web_hq
    python datagen/clean_web.py --dry --limit 20000     # rejects histogram only
"""

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

SPAM = re.compile(
    r"(彩票|賭場|赌场|赌博|博彩|真人娱乐|北京赛车|时时彩|老虎机|六合彩|百家乐|开户送|注册送"
    r"|威廉希尔|德赢vwin|杏彩|凯发k8|明陞|m88asia|BOSS真人|森林舞会游戏|助赢|大智彩票"
    r"|加微信|QQ[:：]?\d{6,}|微信[:：]?[a-zA-Z0-9_]{5,}|电话[:：]?1[3-9]\d{9}"
    r"|阴道|裸体|情趣用品|一夜情|约炮)"
)
SPLIT = re.compile(r"[\s，,。；;]")
SENT = re.compile(r"[。！？；!?;]")


def repetition(text):
    """Fraction of segments that are exact duplicates of an earlier segment."""
    segs = [s.strip() for s in SPLIT.split(text) if len(s.strip()) > 4]
    if len(segs) < 8:
        return 0.0
    return 1 - len(set(segs)) / len(segs)


def fragmented(text):
    """A splice of unrelated forum posts has many sentences and all of them short.
    A real article has a mean sentence length well above 10 characters."""
    sents = [s for s in SENT.split(text) if s.strip()]
    if len(sents) < 6:
        return False
    return sum(len(s) for s in sents) / len(sents) < 9.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=os.path.join(ROOT, "data", "corpus", "web", "*.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "corpus", "web_hq"))
    ap.add_argument("--scores", help="npy of classifier scores, one per doc in glob order")
    ap.add_argument("--keep", type=float, default=1.0, help="fraction to keep by classifier score")
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--max_rep", type=float, default=0.30)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    # t2s_corpus exposes the opencc-derived {codepoint: simplified} table, not a
    # convert function; str.translate over it is idempotent, so running this after
    # scripts/t2s_corpus.py has already converted a shard changes nothing.
    from t2s_corpus import table  # noqa: E402

    t2s = table()

    files = sorted(glob.glob(a.glob))

    cut = None
    if a.scores and a.keep < 1.0:
        import numpy as np

        s = np.load(a.scores)
        # score[i] must be the score OF document i in this same glob order. The
        # scorer walks the identical sorted glob and its parallel workers take
        # contiguous blocks concatenated in worker order for exactly this reason.
        # A silent misalignment here attaches every score to a different document
        # and still produces a perfectly ordinary-looking distribution, so the
        # count is checked rather than assumed.
        n_docs = sum(1 for f in files for line in open(f, encoding="utf-8") if line.strip())
        assert len(s) == n_docs, (
            f"{a.scores} holds {len(s)} scores but {len(files)} shards hold {n_docs} documents. "
            "Scores and documents are matched by position; rescore with the same glob."
        )
        cut = float(np.quantile(s, 1 - a.keep))
        print(f"classifier cut at {cut:.3f} keeps the top {a.keep:.0%} of {len(s)} scored documents")
    if not a.dry:
        os.makedirs(a.out, exist_ok=True)
    rej = {"spam": 0, "short": 0, "repetitive": 0, "fragmented": 0, "low score": 0}
    kept = seen = 0
    si = 0
    import contextlib

    for f in files:
        outp = os.path.join(a.out, os.path.basename(f))
        ctx = contextlib.nullcontext(None) if a.dry else open(outp, "w", encoding="utf-8")  # noqa: SIM115
        with ctx as oh, open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                t = d.get("content", "").translate(t2s)
                seen += 1
                score = None
                if cut is not None:
                    score = s[si] if si < len(s) else 0.0
                si += 1
                if SPAM.search(t):
                    rej["spam"] += 1
                elif len(t) < a.min_chars:
                    rej["short"] += 1
                elif repetition(t) > a.max_rep:
                    rej["repetitive"] += 1
                elif fragmented(t):
                    rej["fragmented"] += 1
                elif cut is not None and score < cut:
                    rej["low score"] += 1
                else:
                    kept += 1
                    if oh:
                        d["content"] = t
                        oh.write(json.dumps(d, ensure_ascii=False) + "\n")
                if a.limit and seen >= a.limit:
                    break
        if a.limit and seen >= a.limit:
            break

    print(f"{seen} documents -> {kept} kept ({kept / max(1, seen):.1%})")
    for k, v in sorted(rej.items(), key=lambda kv: -kv[1]):
        print(f"  rejected {k:<12}{v:>9} ({v / max(1, seen):5.1%})")


def _demo():
    """Each filter must fire on its own case and stay quiet on a clean article."""
    from t2s_corpus import table

    t2s = table()
    assert "台湾的国际关系".translate(t2s) == "台湾的国际关系"
    assert "臺灣的國際關係".translate(t2s) == "台湾的国际关系", "臺灣的國際關係".translate(t2s)
    good = (
        "唐代的两税法把租庸调合并为夏秋两次征收，以资产为宗而不以丁身为本。"
        "这项改革由杨炎在建中元年提出，是中国赋税史上的一次重要转变，"
        "它承认了土地兼并的既成事实，把征税依据从人丁转移到财产。"
    )
    assert not SPAM.search(good) and repetition(good) < 0.3 and not fragmented(good)
    assert SPAM.search("北京赛车官网 您只需要一个电话，足不出户轻松办理社保。")
    assert repetition("移动空调很好。" * 12) > 0.3, repetition("移动空调很好。" * 12)
    assert fragmented("好。真的。对啊。我也是。哈哈。同意。可以。不行。为什么。谁知道。")
    assert not fragmented(good), "a real article was called fragmented"
    print("clean_web self-test OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _demo()
    else:
        main()

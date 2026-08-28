#!/usr/bin/env python3
"""Score web documents for educational value with the 27B served by tileRL.

This is FineWeb-Edu's actual method (arXiv 2406.17557): a strong model labels a
sample, a cheap classifier learns from those labels, the cheap classifier scores
the whole corpus. Everything weaker was tried first and measured, so the reason
for reaching this far is on the record rather than assumed:

    gambling/contact-spam regex alone             AUC 0.50
    hashed character 2-4 grams, 180 hand labels   AUC 0.60
    structural features (tables, phones, quotes)  AUC 0.62
    Qwen3-0.6B, 0-5 rubric                        AUC 0.539
    Qwen3-0.6B, binary yes/no prompt              AUC 0.647

Two things that sweep taught, both of which apply here:

  * Character n-grams rank by TOPIC; the labels split on REGISTER. A page about
    air conditioners is a technical explainer or a product sheet and its
    n-grams barely differ.
  * A small model cannot hold a six-level rubric. The SAME 0.6B went from 0.539
    to 0.647 when the six levels became one yes/no question. So this script
    supports both `--rubric binary` and `--rubric five`, and --check measures
    which one the 27B actually does better on rather than assuming the bigger
    model can take the finer scale.

`--check` scores 180 hand-labelled documents and prints AUC. Nothing downstream
runs until that number is clearly above 0.62.

    python datagen/score_web_27b.py --check data/web_labels.jsonl --rubric binary
    python datagen/score_web_27b.py --glob 'data/corpus/web/*.jsonl' --n 100000 \\
        --out data/web_27b_labels.jsonl
"""

import argparse
import concurrent.futures as cf
import glob
import json
import random
import urllib.error
import urllib.request

BINARY = """下面这段网页文本，是不是有教育价值、值得用来训练语言模型的内容？

有教育价值：科普、技术说明、教材、学习材料、有事实和推理的分析文章。
没有教育价值：广告软文、产品参数和联系方式、赌博和医疗推广、网络小说、娱乐八卦、语句不连贯的碎片拼接、机器翻译。

文本：
{t}

只回答一个字：是 或 否。"""

FIVE = """给下面这段网页文本的教育价值打分，0 到 5 分。

0：赌博/色情/医疗广告、SEO 关键词堆砌、联系方式和产品参数。
1：语句不连贯的碎片拼接、机器翻译、同义词替换的洗稿文。
2：连贯但没有信息量，如娱乐八卦、体育赛报、网络小说。
3：连贯且有信息量的新闻报道、经验分享、产品评测。
4：清晰讲解某个主题的科普、技术说明、学习材料。
5：结构完整、讲解透彻的教材或专业内容。

文本：
{t}

只输出一个 0 到 5 的数字。"""


def ask(url, model, prompt, timeout=120):
    """One completion, one token. Returns the raw string the model emitted."""
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"].strip()


def to_score(raw, rubric):
    """Map the model's text to a number, or None if it did not answer the question.

    Unparseable answers are dropped rather than defaulted: a default silently
    turns 'the model refused' into a real-looking label, which is how a bad
    annotation set gets built without anyone noticing.
    """
    if rubric == "binary":
        if raw.startswith("是"):
            return 1.0
        if raw.startswith("否"):
            return 0.0
        return None
    for ch in raw:
        if ch.isdigit() and int(ch) <= 5:
            return float(ch)
    return None


def score_many(texts, url, model, rubric, workers=16, chars=1200):
    tpl = BINARY if rubric == "binary" else FIVE
    out = [None] * len(texts)

    def one(i):
        try:
            return i, to_score(ask(url, model, tpl.format(t=texts[i][:chars])), rubric)
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
            return i, None

    with cf.ThreadPoolExecutor(workers) as ex:
        for i, s in ex.map(one, range(len(texts))):
            out[i] = s
    return out


def auc(y, s):
    import numpy as np

    y, s = np.asarray(y, float), np.asarray(s, float)
    order = np.argsort(s)
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    npos, nneg = y.sum(), len(y) - y.sum()
    if not npos or not nneg:
        return float("nan")
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8077")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--rubric", choices=["binary", "five"], default="binary")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--check", help="hand-labelled jsonl ({t,y}); report AUC and stop")
    ap.add_argument("--glob", default="data/corpus/web/*.jsonl")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--out", default="data/web_27b_labels.jsonl")
    a = ap.parse_args()

    if a.check:
        with open(a.check, encoding="utf-8") as fh:
            rows = [json.loads(x) for x in fh if x.strip()]
        s = score_many([r["t"] for r in rows], a.url, a.model, a.rubric, a.workers)
        ok = [(r["y"], v) for r, v in zip(rows, s, strict=True) if v is not None]
        print(f"{len(rows)} hand labels, {len(ok)} answered ({len(ok) / len(rows):.0%})")
        if len(ok) < 20:
            print("  too few parseable answers to judge; check the prompt or the server")
            return
        y, v = zip(*ok, strict=True)
        print(f"  rubric={a.rubric}  AUC {auc(y, v):.3f}   (cheap features reached 0.62)")
        import numpy as np

        y, v = np.array(y, float), np.array(v, float)
        print(f"  mean score: hand-keep {v[y == 1].mean():.2f}, hand-drop {v[y == 0].mean():.2f}")
        for t in sorted(set(v)):
            k = v >= t
            if 0.02 < k.mean() < 0.99:
                print(
                    f"  cut at {t:.1f}: keeps {k.mean():5.1%} of docs, {y[k].mean():5.1%} hand-labelled keep"
                )
        return

    files = sorted(glob.glob(a.glob))
    docs = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    docs.append(json.loads(line).get("content", ""))
        if len(docs) >= a.n * 4:
            break
    random.Random(0).shuffle(docs)
    docs = docs[: a.n]
    print(f"scoring {len(docs)} documents", flush=True)

    with open(a.out, "w", encoding="utf-8") as o:
        for i in range(0, len(docs), 512):
            b = docs[i : i + 512]
            for t, s in zip(b, score_many(b, a.url, a.model, a.rubric, a.workers), strict=True):
                if s is not None:
                    o.write(json.dumps({"t": t[:1200], "s": s}, ensure_ascii=False) + "\n")
            print(f"  {i + len(b)}/{len(docs)}", flush=True)
    print(f"saved {a.out}")


def _demo():
    """The parser must reject a non-answer instead of turning it into a label."""
    assert to_score("是", "binary") == 1.0
    assert to_score("否。", "binary") == 0.0
    assert to_score("我认为这段文本", "binary") is None, "a refusal became a label"
    assert to_score("4", "five") == 4.0
    assert to_score("评分：3 分", "five") == 3.0
    assert to_score("无法判断", "five") is None
    assert to_score("9", "five") is None, "an out-of-range digit became a label"
    print("score_web_27b self-test OK")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _demo()
    else:
        main()

#!/usr/bin/env python3
"""Score web documents for educational value with the 27B served by tileRL.

Stage one of FineWeb-Edu's method: a strong model labels a sample, train_quality_head.py
distils it. Everything cheaper was measured first and failed (spam regex AUC 0.50, char
n-grams 0.60, structural features 0.62, Qwen3-0.6B 0.539-0.647), so both rubrics stay and
`--check` measures which the 27B does better on. Nothing downstream runs until that AUC is
clearly above 0.62.

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


def ask(url, model, prompt, timeout=300, max_tokens=400):
    """One completion; the string after the model's think block.

    max_tokens must cover the WHOLE think block: a truncated block makes the parser
    pick a stray digit, and truncation biases toward long high-quality documents.
    """
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return strip_think(d["choices"][0]["message"]["content"])


def strip_think(txt):
    """The answer is what follows the think block; None if the block never closed (truncated
    mid-reasoning is a failure to answer, and must never reach the parser)."""
    if txt.lstrip().startswith("<think>") and "</think>" not in txt:
        return None
    return txt.split("</think>", 1)[-1].strip()


def to_score(raw, rubric):
    """Model text -> number, or None. Unparseable answers are dropped, never defaulted: a default
    turns 'the model refused' into a real-looking label."""
    if raw is None:  # truncated mid-think
        return None
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


def pad_to_shape(prompt, tok, n_tokens):
    """Truncate or newline-pad so every prompt is EXACTLY n_tokens long.

    tileLang JIT-compiles a kernel per sequence shape (~15s each), so mixed lengths pin the
    server at 0% GPU. Newlines pad because they are one token each and ignored at a tail.
    """
    if tok is None:
        return prompt
    ids = tok.encode(prompt).ids if hasattr(tok, "encode") else tok(prompt)["input_ids"]
    if len(ids) > n_tokens:
        # cut the document, never the instructions: the tail of the prompt is the question
        head, tail = prompt.rsplit("\n\n", 1)
        over = len(ids) - n_tokens
        return head[: max(0, len(head) - over * 2)] + "\n\n" + tail
    return prompt + "\n" * (n_tokens - len(ids))


def score_many(texts, url, model, rubric, workers=16, chars=1200, tok=None, pad=0, max_tokens=400):
    tpl = BINARY if rubric == "binary" else FIVE
    urls = [u.strip() for u in url.split(",") if u.strip()]
    out = [None] * len(texts)

    def one(i):
        try:
            p = tpl.format(t=texts[i][:chars])
            if pad:
                p = pad_to_shape(p, tok, pad)
            # round-robin: keeps each endpoint's KV pool inside its 4096 tokens.
            return i, to_score(ask(urls[i % len(urls)], model, p, max_tokens=max_tokens), rubric)
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
    ap.add_argument(
        "--url",
        default="http://127.0.0.1:8077",
        help="comma-separated endpoints. One 27B on one H20 answers 0.76 documents a second "
        "and its KV pool caps useful concurrency at 4, so scoring 100K documents means "
        "running one server per card and spreading requests across them.",
    )
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--rubric", choices=["binary", "five"], default="binary")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--pad",
        type=int,
        default=1024,
        help="pad/truncate every prompt to exactly this many tokens. tileLang compiles a "
        "kernel per sequence shape; without this the server spends all its time in the JIT "
        "and sits at 0%% GPU. 0 disables.",
    )
    ap.add_argument("--tokenizer", default="/work/Qwen3.8-27B-NVFP4/tokenizer.json")
    ap.add_argument("--max_tokens", type=int, default=400, help="must cover the whole think block")
    ap.add_argument("--check", help="hand-labelled jsonl ({t,y}); report AUC and stop")
    ap.add_argument("--glob", default="data/corpus/web/*.jsonl")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--out", default="data/web_27b_labels.jsonl")
    a = ap.parse_args()

    tok = None
    if a.pad:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(a.tokenizer)

    if a.check:
        with open(a.check, encoding="utf-8") as fh:
            rows = [json.loads(x) for x in fh if x.strip()]
        s = score_many(
            [r["t"] for r in rows],
            a.url,
            a.model,
            a.rubric,
            a.workers,
            tok=tok,
            pad=a.pad,
            max_tokens=a.max_tokens,
        )
        import numpy as np

        ok = [(r["y"], v) for r, v in zip(rows, s, strict=True) if v is not None]
        print(f"{len(rows)} hand labels, {len(ok)} answered ({len(ok) / len(rows):.0%})")
        # Unanswered is not a neutral loss: truncated thinking tracks length, length tracks
        # quality, so a low answer rate biases the sample toward what we mean to REMOVE.
        got = np.array([v is not None for v in s])
        yy = np.array([r["y"] for r in rows], float)
        ll = np.array([len(r["t"]) for r in rows], float)
        if (~got).any():
            print(
                f"  UNANSWERED BIAS: hand-keep {yy[got].mean():.1%} among answered vs "
                f"{yy[~got].mean():.1%} among failed (overall {yy.mean():.1%}); "
                f"median chars {np.median(ll[got]):.0f} vs {np.median(ll[~got]):.0f}"
            )
            if got.mean() < 0.9:
                print("  answer rate below 90%; raise --max_tokens before trusting any number below")
        if len(ok) < 20:
            print("  too few parseable answers to judge; check the prompt or the server")
            return
        y, v = zip(*ok, strict=True)
        print(f"  rubric={a.rubric}  AUC {auc(y, v):.3f}   (cheap features reached 0.62)")
        y, v = np.array(y, float), np.array(v, float)
        print(f"  mean score: hand-keep {v[y == 1].mean():.2f}, hand-drop {v[y == 0].mean():.2f}")
        for t in sorted(set(v)):
            k = v >= t
            if 0.02 < k.mean() < 0.99:
                print(
                    f"  cut at {t:.1f}: keeps {k.mean():5.1%} of docs, {y[k].mean():5.1%} hand-labelled keep"
                )
        return

    # Stride EVERY shard: shards are not interchangeable (a prefix draw read 8.5%
    # positive where a shard-stratified draw reads 21.8%).
    files = sorted(glob.glob(a.glob))
    per = max(1, a.n // max(1, len(files)) + 1)
    rng = random.Random(0)
    docs = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            lines = [x for x in fh if x.strip()]
        for x in rng.sample(lines, min(per, len(lines))):
            docs.append(json.loads(x).get("content", ""))
    rng.shuffle(docs)
    docs = docs[: a.n]
    print(f"sampled from {len(files)} shards, up to {per} each", flush=True)
    print(f"scoring {len(docs)} documents", flush=True)

    with open(a.out, "w", encoding="utf-8") as o:
        for i in range(0, len(docs), 512):
            b = docs[i : i + 512]
            scored = score_many(
                b, a.url, a.model, a.rubric, a.workers, tok=tok, pad=a.pad, max_tokens=a.max_tokens
            )
            for t, s in zip(b, scored, strict=True):
                if s is not None:
                    o.write(json.dumps({"t": t[:1200], "s": s}, ensure_ascii=False) + "\n")
            print(f"  {i + len(b)}/{len(docs)}", flush=True)
    print(f"saved {a.out}")


def _demo():
    """The parser must reject a non-answer instead of turning it into a label."""
    assert to_score("是", "binary") == 1.0
    assert to_score("否。", "binary") == 0.0
    assert to_score("我认为这段文本", "binary") is None, "a refusal became a label"
    assert strip_think("<think>\n分析中，这段文本讲的是 3 个要点") is None, (
        "a think block truncated mid-reasoning was treated as an answer"
    )
    assert to_score(strip_think("<think>\n讲了 3 点"), "five") is None, (
        "a digit inside truncated reasoning became a score -- this produced AUC 0.407"
    )
    assert to_score("4", "five") == 4.0
    assert to_score("评分：3 分", "five") == 3.0
    assert to_score("无法判断", "five") is None
    assert to_score("9", "five") is None, "an out-of-range digit became a label"
    assert to_score(strip_think("<think>\n\n</think>\n\n是"), "binary") == 1.0
    assert to_score(strip_think("否"), "binary") == 0.0, "an answer with no think block was lost"
    print("score_web_27b self-test OK")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _demo()
    else:
        main()

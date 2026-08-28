#!/usr/bin/env python3
"""Educational-quality classifier for the Chinese web corpus.

Why. `data/mix.json` gives web 88% of an 11.5B-token pretrain, and hand-reading
140 random web documents says roughly a quarter of them are worth training on.
The rest are gambling-SEO pages, product spec sheets, hospital ads, serialized
web novels, machine translation, and forum fragments spliced together. The
existing filters in datagen/build_corpus.py are heuristics and they let all of
this through: a keyword scan for gambling/contact spam fires on only 2.5% of
480,952 documents, because 业配文 and 企业产品页 have no keyword signature.

Method, following FineWeb-Edu (arXiv 2406.17557): label documents for
educational value, fit a cheap classifier on the labels, keep the top slice.
Their threshold of 3 removed 92% of FineWeb and reached the same MMLU with 10x
fewer tokens, which is the licence to train on less data after filtering.

Where we differ, and it is a real weakness: they labelled 500K documents with
Llama-3-70B. There is no LLM on this pod, so the labels here are hand-assigned
a few hundred at a time and the classifier is a hashed character n-gram logistic
regression rather than an embedding model. Expect it to catch the obvious 60%
and be unreliable near the boundary. Cross-validated AUC is printed on every
fit; do not raise the threshold past what that number supports.

    python datagen/quality_clf.py fit    --labels data/web_labels.jsonl
    python datagen/quality_clf.py score  --glob 'data/corpus/web/*.jsonl' --out data/web_scores.npy
    python datagen/quality_clf.py filter --keep 0.25 --out data/corpus/web_hq
"""

import argparse
import glob
import hashlib
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 2**18
MODEL = os.path.join(ROOT, "data", "quality_clf.npz")

# Hard negatives with a keyword signature. These are not the classifier -- they are
# the cases so unambiguous that spending classifier capacity on them is waste, and
# they double as label sanity checks. Gambling brand names appear INSIDE otherwise
# ordinary sentences (SEO injection), so a hit anywhere in the document is enough.
SPAM = re.compile(
    r"(彩票|賭場|赌场|赌博|博彩|真人娱乐|北京赛车|时时彩|老虎机|六合彩|百家乐|开户送|注册送"
    r"|威廉希尔|德赢vwin|杏彩|凯发k8|明陞|m88asia|BOSS真人|森林舞会游戏|助赢|大智彩票|必威"
    r"|加微信|QQ[:：]?\d{6,}|微信[:：]?[a-zA-Z0-9_]{5,}|电话[:：]?1[3-9]\d{9})"
)


def features(text, dim=DIM):
    """Hashed character 2-, 3- and 4-grams over the first 1000 characters.

    First 1000 characters because that is the window FineWeb-Edu's annotator saw,
    and because a page's nature is settled in its opening: a product sheet opens
    with specifications, an ad opens with a brand, an article opens with a claim.
    Character n-grams and not words: Chinese has no spaces, and a word segmenter
    is another dependency to be wrong about.
    """
    t = text[:1000]
    v = np.zeros(dim, dtype=np.float32)
    for n in (2, 3, 4):
        for i in range(len(t) - n + 1):
            h = hashlib.blake2b(t[i : i + n].encode(), digest_size=8).digest()
            v[int.from_bytes(h, "little") % dim] += 1.0
    # Length-normalize so a long document is not scored simply for being long.
    norm = np.linalg.norm(v)
    return v / norm if norm else v


def _fit_lr(X, y, epochs=2000, lr=5.0, l2=1e-4):
    """Logistic regression by plain gradient descent.

    scikit-learn is not installed on the pod and this is 20 lines. ponytail: full
    batch, no early stopping -- at a few hundred rows the whole fit is milliseconds.
    """
    w = np.zeros(X.shape[1], dtype=np.float32)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b


def _auc(y, s):
    """Rank AUC. With ~100 labels this is the only honest quality number available;
    accuracy at a fixed threshold would hide how bad the ranking is near the boundary."""
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    npos, nneg = y.sum(), len(y) - y.sum()
    if not npos or not nneg:
        return float("nan")
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def load_labels(path):
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    X = np.stack([features(r["t"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows], dtype=np.float32)
    return X, y, rows


def cmd_fit(a):
    X, y, rows = load_labels(a.labels)
    print(f"{len(y)} labels, {int(y.sum())} keep ({y.mean():.0%})")
    # 5-fold CV first: the number that says whether this classifier is worth applying.
    idx = np.arange(len(y))
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    scores = np.zeros(len(y))
    for f in range(5):
        te = idx[f::5]
        tr = np.setdiff1d(idx, te)
        w, b = _fit_lr(X[tr], y[tr])
        scores[te] = X[te] @ w + b
    print(f"5-fold CV AUC {_auc(y, scores):.3f}  (0.5 = useless, 1.0 = perfect)")
    w, b = _fit_lr(X, y)
    np.savez(MODEL, w=w, b=b, dim=DIM)
    print(f"saved {MODEL}")


def cmd_score(a):
    m = np.load(MODEL)
    w, b = m["w"], float(m["b"])
    files = sorted(glob.glob(a.glob))
    out = []
    for fi, f in enumerate(files):
        with open(f, encoding="utf-8") as fh:
            lines = fh.readlines()
        for line in lines:
            if not line.strip():
                continue
            t = json.loads(line).get("content", "")
            s = float(features(t) @ w + b)
            if SPAM.search(t):
                s -= 5.0  # hard negatives, kept as a score penalty so the cut stays one number
            out.append(s)
        if fi % 10 == 0:
            print(f"  {fi}/{len(files)} files, {len(out)} docs", flush=True)
    arr = np.array(out, dtype=np.float32)
    np.save(a.out, arr)
    q = np.percentile(arr, [1, 10, 25, 50, 75, 90, 99])
    print(f"{len(arr)} docs scored -> {a.out}")
    print("  percentiles 1/10/25/50/75/90/99: " + " ".join(f"{v:.2f}" for v in q))


def _demo():
    """The classifier must separate the two kinds of document it was built to
    separate. Trained on six hand-written examples of each, it must rank a held-out
    pair correctly -- if it cannot do that, nothing downstream is worth running."""
    pos = [
        "物理学是描述世上质量与能量交互作用的学问。我们会从电磁学出发，进展到光学，并说明近代物理对人类观念的突破。",
        "产后抑郁症是指产妇在分娩后出现抑郁、悲伤、易激怒等一系列症状为特征的心理障碍，通常在产后2-4周出现。",
        "光缆交接箱用于光缆接入网中主干光缆与配线光缆交接处，结构由箱体、内部金工件、光纤活动连接器组成。",
        "唐代的两税法把租庸调合并为夏秋两次征收，以资产为宗而不以丁身为本，是中国赋税史上的一次重要转变。",
        "牙齿矫正一般需要两年左右。矫正前患者应做一些咨询，了解治疗的大概过程、疗程和费用。",
        "凯度零售咨询发布报告显示，2016年B2C电商渠道贡献了电子商务总量中59%的份额，占据主导地位。",
    ]
    neg = [
        "威廉希尔娱乐怎么样 然而外面的世界即使再新鲜诱人，那个名叫家的地方始终是你前进力量的源泉。",
        "本产品源头厂家直销，质量保证。联系人：王元华 手机：13807477200 销售热线：0734-8558096",
        "这几个月为了拼专案奖金可以说没什么时间休息，还好我的努力有了收获！收到后当然是直接开箱使用啦！！！",
        "他决定亲手挖掘出小镇中已经被尘封了的血腥过去，剑神眼中一亮，这一刻她的目光无比的温柔。",
        "北京赛车官网 关键词：助赢北京赛车官网 您只需要一个电话，足不出户轻松办理社保。",
        "|品牌彩虹||软件名称彩虹图纸管理软件| |版本语言简体中文版||系统平台要求Windows 7|",
    ]
    X = np.stack([features(t) for t in pos[:-1] + neg[:-1]])
    y = np.array([1.0] * 5 + [0.0] * 5, dtype=np.float32)
    w, b = _fit_lr(X, y)
    sp = features(pos[-1]) @ w + b
    sn = features(neg[-1]) @ w + b
    assert sp > sn, f"held-out educational {sp:.3f} did not outrank held-out junk {sn:.3f}"
    assert SPAM.search(neg[0]) and SPAM.search(neg[4]), "the hard-negative regex misses its own cases"
    assert not any(SPAM.search(t) for t in pos), "the hard-negative regex fires on educational text"
    print(f"quality_clf self-test OK (held-out educational {sp:+.3f} > junk {sn:+.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--labels", default=os.path.join(ROOT, "data", "web_labels.jsonl"))
    f.set_defaults(fn=cmd_fit)
    s = sub.add_parser("score")
    s.add_argument("--glob", default=os.path.join(ROOT, "data", "corpus", "web", "*.jsonl"))
    s.add_argument("--out", default=os.path.join(ROOT, "data", "web_scores.npy"))
    s.set_defaults(fn=cmd_score)
    d = sub.add_parser("selftest")
    d.set_defaults(fn=lambda a: _demo())
    a = ap.parse_args()
    a.fn(a)

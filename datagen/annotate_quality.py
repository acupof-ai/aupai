#!/usr/bin/env python3
"""Score web documents 0-5 for educational value with a small instruct model.

This is the step FineWeb-Edu (arXiv 2406.17557) does with Llama-3-70B on 500K
documents. Two cheaper substitutes were tried first and both failed, which is
why this exists:

    hashed character 2-4 grams, 180 hand labels   5-fold CV AUC 0.60
    structural features (tables, phones, quotes)  5-fold CV AUC 0.62
    the gambling/contact-spam regex alone         5-fold CV AUC 0.50

n-grams rank by TOPIC and the labels split on REGISTER: a page about air
conditioners is a good technical explainer or a product sheet, and its character
n-grams are nearly the same either way. Structural features get the product
sheets and miss the content farms. Judging the difference needs a model that
reads the text.

Qwen3-0.6B is small enough to fetch through the tunnel and run on one H20 at a
few hundred documents a second. It is much weaker than Llama-3-70B, so the
annotations are checked against 180 hand labels before anything downstream uses
them -- `--check data/web_labels.jsonl` prints the agreement and the AUC of the
model's own score against those labels. If that AUC is not clearly above the
0.62 the cheap features reached, this route is not worth taking either.

    python datagen/annotate_quality.py --model /work/models/qwen3-0.6b \\
        --check data/web_labels.jsonl                        # validate first
    python datagen/annotate_quality.py --model ... --glob 'data/corpus/web/*.jsonl' \\
        --n 50000 --out data/web_llm_labels.jsonl            # then annotate
"""

import argparse
import glob
import json
import random

PROMPT = """以下是一段从网页上抓取的中文文本。请按它对训练一个语言模型的教育价值打分，0 到 5 分。

评分标准：
0 分：赌博/色情/医疗广告、SEO 关键词堆砌、联系方式和产品参数、纯广告软文。
1 分：语句不连贯的碎片拼接、机器翻译痕迹明显、同义词替换的洗稿文。
2 分：连贯但没有信息量，如娱乐八卦、体育赛报、网络小说、日常流水账。
3 分：连贯且有一定信息量的新闻报道、经验分享、产品评测。
4 分：清晰讲解某个主题的科普、技术说明、学习材料，有事实和推理。
5 分：结构完整、讲解透彻的教材、学术或专业内容。

文本：
{text}

只输出一个 0 到 5 的数字，不要输出别的。"""


def build(model_dir, device="cuda:0"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to(device).eval()
    # Score by comparing the logits of the six digit tokens at the first generated
    # position rather than by generating: one forward pass, no sampling, and the
    # answer cannot be an unparseable string.
    digit_ids = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(6)]
    return tok, model, digit_ids


def score_batch(texts, tok, model, digit_ids, device="cuda:0", max_len=1024):
    import torch

    msgs = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(text=t[:1200])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for t in texts
    ]
    enc = tok(msgs, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        logits = model(**enc).logits[:, -1, :]
    d = logits[:, digit_ids].float().softmax(-1)
    # Expected value, not argmax: the expectation is a continuous score, which
    # gives a threshold to move later instead of six buckets to argue about.
    return (d * torch.arange(6, device=d.device)).sum(-1).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--glob", default="data/corpus/web/*.jsonl")
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="data/web_llm_labels.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--check", help="hand-labelled jsonl ({t,y}); report agreement and stop")
    a = ap.parse_args()

    tok, model, digit_ids = build(a.model, a.device)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if a.check:
        import numpy as np

        with open(a.check, encoding="utf-8") as fh:
            rows = [json.loads(x) for x in fh if x.strip()]
        s = []
        for i in range(0, len(rows), a.batch):
            s += score_batch([r["t"] for r in rows[i : i + a.batch]], tok, model, digit_ids, a.device)
        s = np.array(s)
        y = np.array([float(r["y"]) for r in rows])
        order = np.argsort(s)
        ranks = np.empty(len(s))
        ranks[order] = np.arange(1, len(s) + 1)
        npos, nneg = y.sum(), len(y) - y.sum()
        auc = (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
        print(f"{len(rows)} hand labels ({int(npos)} keep)")
        print(f"  model score: keep mean {s[y == 1].mean():.2f}, drop mean {s[y == 0].mean():.2f}")
        print(f"  AUC against the hand labels {auc:.3f}   (cheap features reached 0.62)")
        for t in (2.0, 2.5, 3.0):
            k = s >= t
            if k.any():
                print(
                    f"  cut at {t}: keeps {k.mean():.0%} of docs, {y[k].mean():.0%} of them hand-labelled keep"
                )
        return

    files = sorted(glob.glob(a.glob))
    random.seed(0)
    docs = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    docs.append(json.loads(line).get("content", ""))
        if len(docs) >= a.n * 3:
            break
    random.shuffle(docs)
    docs = docs[: a.n]
    print(f"annotating {len(docs)} documents from {len(files)} files", flush=True)

    with open(a.out, "w", encoding="utf-8") as o:
        for i in range(0, len(docs), a.batch):
            b = docs[i : i + a.batch]
            for t, s in zip(b, score_batch(b, tok, model, digit_ids, a.device), strict=True):
                o.write(json.dumps({"t": t[:1200], "s": round(s, 3)}, ensure_ascii=False) + "\n")
            if i % (a.batch * 20) == 0:
                print(f"  {i}/{len(docs)}", flush=True)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Same-logic, multi-narrative pretraining augmentation via local MLX seed model.

For each source doc (title+content), prompt LFM2.5-8B-A1B-MLX to rewrite the same
causal chain in N narrative styles. Each (doc, style) -> one output JSON line.

Usage:
  python gen_narrative.py data/augmented/level1_all.jsonl \
      data/augmented_narrative.jsonl --limit 2
"""

import argparse
import json
import os
import re
import time

from mlx_lm import generate, load

STYLES = {
    "textbook": "一本严谨的教科书条目",
    "dialogue": "师生/两人对话,你来我往推进",
    "story": "一个具体的小故事,把逻辑藏在情节里",
    "qa": "问答体:一个问题,一个解释",
    "case": "一个真实案例复盘,从现象到结论",
}

SYSTEM = (
    "你负责把同一段逻辑改写成不同叙事风格的中文短篇,用于高质量预训练数据。"
    "要求:保留原文的完整因果/推理链,不能丢失任何关键环节;表达自然、有信息密度,"
    "无语法错误,无AI腔(禁止'首先其次最后''总而言之''值得注意的是'这类套话),不重复与原文无关的内容。"
    "只输出正文,不要标题、不要前言。控制在200-800字。"
)


def bad(s):
    """Quality filter: reject empty, too-long, or AI-slang-heavy output."""
    if not s or len(s) < 80 or len(s) > 1600:
        return True
    n = sum(1 for w in ("首先", "其次", "最后", "总而言之", "综上", "值得注意", "正如", "综上所述") if w in s)
    return n >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--styles", default=",".join(STYLES))
    args = ap.parse_args()

    model, tok = load("LiquidAI/LFM2.5-8B-A1B-MLX-bf16")
    styles = args.styles.split(",")

    docs = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    if args.limit:
        docs = docs[: args.limit]

    out_exists = os.path.exists(args.out)
    out = open(args.out, "a", encoding="utf-8")
    if not out_exists:
        out.write("")  # create
    seen = set()
    if out_exists:
        for l in open(args.out, encoding="utf-8"):
            try:
                d = json.loads(l)
                seen.add((d["source_title"], d["style"]))
            except Exception:
                pass

    n_tok, t0 = 0, time.time()
    if not os.path.exists(args.out):
        open(args.out, "w").close()
        out = open(args.out, "a", encoding="utf-8")

    produced = 0
    for doc in docs:
        title = doc.get("title", "")
        if not title:
            continue
        for st in styles:
            if (title, st) in seen:
                continue
            prompt = (
                f"以下是原始素材(标题:{title}):\n\n{doc['content']}\n\n"
                f"请把这段逻辑用【{STYLES[st]}】的风格重写成一则完整短文,"
                f"保持因果链不变。直接输出正文。"
            )
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            resp = generate(model, tok, prompt=msgs, max_tokens=args.max_tokens, temp=0.7, top_p=0.9)
            n_tok += len(resp)
            body = re.sub(r"^#.*$", "", resp, flags=re.M).strip()
            if bad(body):
                print(f"  [reject] {title} / {st} ({len(body)}c)", file=sys.stderr)
                continue
            out.write(
                json.dumps(
                    {
                        "content": body,
                        "style": st,
                        "logic_chain": f"{title}",
                        "source": "LFM2.5-8B-A1B-MLX-bf16",
                        "source_title": title,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()
            produced += 1
    dt = time.time() - t0
    print(
        f"produced={produced} tokens={n_tok} {n_tok / max(dt, 1e-6):.1f} tok/s elapsed={dt:.0f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    import sys

    main()

#!/usr/bin/env python3
"""27B zero-shot category discovery over an OpenAI-compatible endpoint.

Role (fb 2026-08-30, user ruling): pattern discovery ONLY — not a production
scorer, not a labeler, no agreement test. The loop is: stratified sample ->
27B names categories -> cluster other_desc -> write regexes -> validate on the
150 dev set, re-test once on the locked 400. Distillation is dead (three
student families all failed at AUC 0.54-0.57); hand-written patterns are the
only measured-effective filter (+11.9% recall at 100% precision).

Output per doc: {"category": <known|other>, "other_desc": <phrase if other>}.
Thinking ON (user ruling): naming novel categories needs the reasoning.

Lock discipline: the 400 stays locked — no 400-derived text in the prompt.
The four *_ad/stock categories below are names from the 400-handread taxonomy
(category names are not 400 text; no 400 document enters the prompt).

Modes:
  --pairs FILE   score pairs (shard\\tid[\\tanything]), report distribution
  --bench N      throughput probe on N dev docs, then exit

    python datagen/score_27b_zero.py --base-url http://127.0.0.1:8100/v1,http://127.0.0.1:8101/v1 \
        --model qwen38-27b --pairs /tmp/pairs_discovery.txt
"""
import argparse, json, re, sys, time, urllib.request, urllib.error
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

PROMPT_VERSION = "v3"
PROMPT_SYS = """你是中文网页语料的内容分析员。我们在建语言模型训练语料,需要你识别文档的内容类型,尤其是对训练无用的垃圾类型。

对每篇文档输出一个 JSON:
{"category": "<类别之一>", "other_desc": "<category=other 时,用一个短语命名它的类型;否则空字符串>"}

已知类别:
- essay: 作文/范文/读后感/观后感/演讲稿
- exam: 题库/试卷/练习题
- lesson: 教案/课件/导学案/复习资料
- toc: 图书目录/章节列表/书籍元数据页
- farm: 内容农场/SEO 软文
- spun: 洗稿/搬运/聚合
- ads: 广告/导航/堆砌页
- medical_ad: 医疗软文
- stock: 股评/荐股
- training_ad: 培训营销
- product: 产品植入/带货
- other: 以上都不是

只输出 JSON,不要输出别的。"""
CAP = 8000          # chars per doc sent to the model
WORKERS = 4         # conservative: the serve has 256 KV blocks; 16 exhausted it
MAX_TOKENS = 1024   # thinking on: measured 274-661 tokens for the JSON answer


def call_model(base_urls, model, text, think):
    """Returns parsed dict, or None on failure. Round-robins endpoints;
    a failed endpoint is skipped on retry."""
    user = f"文档:\n{text[:CAP]}"
    if think == "off":
        user += "\n/no_think"
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT_SYS},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }).encode()
    for attempt in range(len(base_urls) * 2):
        base = base_urls[attempt % len(base_urls)]
        try:
            req = urllib.request.Request(
                f"{base}/chat/completions", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.loads(r.read())
            out = resp["choices"][0]["message"]["content"]
            out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip().strip("`")
            i, j = out.find("{"), out.rfind("}")
            if i < 0 or j <= i:
                continue
            d = json.loads(out[i:j + 1])
            if "category" not in d:
                continue
            d["_usage"] = resp.get("usage", {})
            return d
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError, TimeoutError):
            continue
    return None


def score_all(base_urls, model, texts, think):
    with ThreadPoolExecutor(WORKERS) as ex:
        return list(ex.map(lambda t: call_model(base_urls, model, t, think), texts))


def pull_texts(raw_dir, pairs):
    by_shard = {}
    for shard, doc_id in pairs:
        by_shard.setdefault(shard, set()).add(doc_id)
    out, total = {}, len(pairs)
    for shard, want in by_shard.items():
        with open(f"{raw_dir}/{shard}") as f:
            for line in f:
                d = json.loads(line)
                if d["id"] in want:
                    out[d["id"]] = d["text"]
                    if len(out) == total:
                        return out
    return out


def report(name, results, texts):
    ok = [d for d in results if d]
    print(f"\n== {name} ==  ({len(ok)}/{len(results)} parsed)")
    if not ok:
        print("ALL CALLS FAILED — service down?")
        return
    cats = Counter(str(d.get("category", "?")) for d in ok)
    print(f"categories: {dict(cats.most_common())}")
    others = Counter(str(d.get("other_desc", "")).strip()
                     for d in ok if d.get("category") == "other")
    others.pop("", None)
    if others:
        print(f"\nother_desc candidates ({sum(others.values())} docs, "
              f"{len(others)} distinct):")
        for desc, n in others.most_common(30):
            print(f"  {n:3d}  {desc}")

    # truncation check — long docs are judged on their head only
    tlen = [len(t) for t in texts]
    trunc = sum(1 for L in tlen if L > CAP)
    print(f"\ntruncated (> {CAP} chars): {trunc}/{len(tlen)} ({trunc / len(tlen):.1%})")
    if trunc >= 5:
        tc = Counter(str(d.get("category", "?")) if d else "FAILED"
                     for d, L in zip(results, tlen) if L > CAP)
        print(f"category mix among truncated: {dict(tc.most_common())}")

    ct = [d["_usage"].get("completion_tokens", 0) for d in ok]
    pt = [d["_usage"].get("prompt_tokens", 0) for d in ok]
    print(f"\ntokens: prompt mean {sum(pt) / len(pt):.0f}, "
          f"completion mean {sum(ct) / len(ct):.0f}, total {sum(pt) + sum(ct)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True,
                    help="comma-separated for round-robin across endpoints")
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", help="pairs file: shard\\tid[\\tlabel]")
    ap.add_argument("--raw-dir", default="/work/aupai/data/raw/cci3_hq")
    ap.add_argument("--think", choices=["on", "off"], default="on")
    ap.add_argument("--bench", type=int, help="throughput probe on N dev docs")
    ap.add_argument("--out", help="write results JSON here")
    a = ap.parse_args()
    base_urls = [u.strip() for u in a.base_url.split(",")]

    config = {"prompt_version": PROMPT_VERSION, "prompt_sys": PROMPT_SYS,
              "cap_chars": CAP, "workers": WORKERS, "max_tokens": MAX_TOKENS,
              "temperature": 0, "model": a.model, "think": a.think,
              "endpoints": base_urls,
              "think_handling": "default (thinking enabled)" if a.think == "on"
                                else "user-turn /no_think",
              "parser": "strip <think> block, extract first {...} JSON"}
    print(f"config: {json.dumps({k: v for k, v in config.items() if k != 'prompt_sys'}, ensure_ascii=False)}", flush=True)
    result = {"config": config}

    if a.bench:
        pairs = []
        for line in open(a.pairs):
            pairs.append(line.rstrip("\n").split("\t")[:2])
            if len(pairs) == a.bench:
                break
        texts = pull_texts(a.raw_dir, pairs)
        docs = [texts[i] for _, i in pairs]
        print(f"bench: {len(docs)} docs, think={a.think} ...", flush=True)
        t0 = time.time()
        res = score_all(base_urls, a.model, docs, a.think)
        dt = time.time() - t0
        ok = [d for d in res if d]
        if not ok:
            print("ALL CALLS FAILED — service down?")
            sys.exit(1)
        pt = sum(d["_usage"].get("prompt_tokens", 0) for d in ok)
        ct = sum(d["_usage"].get("completion_tokens", 0) for d in ok)
        print(f"wall {dt:.1f}s | prefill {pt / dt:.0f} tok/s | decode {ct / dt:.0f} tok/s")
        print(f"prompt tok: mean {pt / len(ok):.0f}/doc | completion mean {ct / len(ok):.0f}/doc")
        print(f"300-doc estimate: {300 * (pt / len(ok)) / 1e6:.1f}M prefill tok, "
              f"~{300 * dt / len(ok) / 60:.0f} min wall at this concurrency")
        result["bench"] = {"docs": len(ok), "wall_s": dt,
                           "prefill_tok_s": pt / dt, "decode_tok_s": ct / dt,
                           "mean_prompt_tok": pt / len(ok),
                           "mean_completion_tok": ct / len(ok)}

    elif a.pairs:
        pairs = []
        for line in open(a.pairs):
            pairs.append(line.rstrip("\n").split("\t")[:2])
        texts = pull_texts(a.raw_dir, pairs)
        docs = [texts[i] for _, i in pairs]
        print(f"scoring {len(docs)} docs (think={a.think})...", flush=True)
        res = score_all(base_urls, a.model, docs, a.think)
        report("DISCOVERY", res, docs)
        result["docs"] = [
            {"shard": s, "id": i, "chars": len(t),
             "truncated": len(t) > CAP,
             "category": d and d.get("category"),
             "other_desc": d and d.get("other_desc"),
             "completion_tokens": d and d.get("_usage", {}).get("completion_tokens")}
            for (s, i), d, t in zip(pairs, res, docs)]

    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()

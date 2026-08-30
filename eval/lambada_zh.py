#!/usr/bin/env python3
"""LAMBADA-zh last-token prediction (panel metric 3).

Two readings from one item set:

  open_acc1/5 -- the panel metric. Floor ~= 1/vocab ~= 0. Given the full left
  context, does the model's top-1 (top-5) next token equal the passage's real
  final token? Open-vocabulary, exact match -- the LAMBADA reading.

  two_way_acc -- the known-answer instrument. Real final token vs a random
  same-length token (another item's target, char-length matched), scored by
  continuation log-likelihood via the shared score_mc. Floor 50%. Panel freeze
  rule: normal-vs-swapped spread >= 60pt on the strongest checkpoint.

The target is ONE BPE token by construction (the sentence's last token), so
there is no multi-token-target confound (panel false signal #1). Distractors
are real tokens from the same source, frequency-matched by construction.

Data: build with --build from a prose jsonl ({"content": ...} per line).
The panel metric needs HELD-OUT text (memorization is a listed false signal);
training-domain text is only usable for runner calibration and the output
flags its provenance.

Usage:
    python eval/lambada_zh.py --build --src /path/to/prose.jsonl --out data/eval/lambada_zh.jsonl
    python eval/lambada_zh.py --ckpt ckpt_0830v1_3.24b.pt.ep1 --data data/eval/lambada_zh.jsonl
    python eval/lambada_zh.py --ckpt ... --swap
    python eval/lambada_zh.py --selftest
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ["FLA_FLASH_KDA"] = "0"

from scripts.loader import EOS_ID, load_checkpoint, load_tokenizer  # noqa: E402

TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
DEFAULT_DATA = os.path.join(ROOT, "data", "eval", "lambada_zh.jsonl")

_CJK = re.compile(r"[一-鿿]+")
_SENT_SPLIT = re.compile(r"(?<=[。！？；])|[\n]+")


def build_items(tok, src_path, n=1000, seed=20260830):
    """Carve last-token items from prose. Returns (items, stats).

    Skip reasons in stats: short (sentence too short), no_cjk_target,
    roundtrip (context+target does not re-tokenize identically), dup.
    """
    items, seen_ctx, stats = [], set(), defaultdict(int)
    for line in open(src_path, encoding="utf-8"):
        if len(items) >= n:
            break
        try:
            content = json.loads(line).get("content", "")
        except json.JSONDecodeError:
            continue
        for sent in _SENT_SPLIT.split(content):
            if len(items) >= n:
                break
            # the split keeps the sentence-final punctuation; the target is the
            # last word BEFORE it
            sent = sent.rstrip("。！？；.!?").strip()
            if not 12 <= len(sent) <= 80:
                stats["short"] += 1
                continue
            ids = tok.encode(sent, add_special_tokens=False).ids
            if len(ids) < 10:
                stats["short"] += 1
                continue
            target_id = ids[-1]
            target_str = tok.decode([target_id]).strip()
            if not _CJK.fullmatch(target_str) or len(target_str) > 4:
                stats["no_cjk_target"] += 1
                continue
            # context = everything before the target; verify round-trip.
            if sent.endswith(target_str):
                context_str = sent[: -len(target_str)]
            else:
                context_str = tok.decode(ids[:-1])
            if tok.encode(context_str, add_special_tokens=False).ids + [target_id] != ids:
                stats["roundtrip"] += 1
                continue
            if context_str in seen_ctx:
                stats["dup"] += 1
                continue
            seen_ctx.add(context_str)
            items.append({
                "context": context_str, "target": target_str,
                "target_id": target_id, "src": os.path.basename(src_path),
            })
    # Distractors: rotate within char-length buckets (same token length by
    # construction -- every target is one token; char length matches frequency).
    buckets = defaultdict(list)
    for k, it in enumerate(items):
        buckets[len(it["target"])].append(k)
    for L, idxs in buckets.items():
        if len(idxs) < 2:
            for k in idxs:
                items[k]["distractor"] = items[k]["target"]  # degenerate; scored as skip
            stats["singleton_bucket"] = stats.get("singleton_bucket", 0) + len(idxs)
            continue
        for pos, k in enumerate(idxs):
            items[k]["distractor"] = items[idxs[(pos + 1) % len(idxs)]]["target"]
    random.Random(seed).shuffle(items)
    return items, dict(stats)


def score_open_vocab(model, items, device, batch=32):
    """Top-1/top-5 exact match of the model's next-token prediction."""
    enc = [it["ctx_ids"] for it in items]
    tgt = [it["target_id"] for it in items]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    hits1 = hits5 = 0
    for s in range(0, len(order), batch):
        idx = order[s : s + batch]
        max_len = max(len(enc[i]) for i in idx)
        x = torch.full((len(idx), max_len), EOS_ID, dtype=torch.long, device=device)
        for b, i in enumerate(idx):
            x[b, : len(enc[i])] = torch.tensor(enc[i], device=device)
        logits = model(x)[0].float()
        last = logits[torch.arange(len(idx)), [len(enc[i]) - 1 for i in idx]]
        top5 = last.topk(5, dim=-1).indices
        for b, i in enumerate(idx):
            hits1 += top5[b, 0].item() == tgt[i]
            hits5 += tgt[i] in top5[b].tolist()
    return hits1 / len(items), hits5 / len(items)


def score_two_way(model, tok, cfg, items, device, swap=False, batch=32):
    """Real target vs same-length distractor via continuation log-likelihood."""
    from eval.run_eval import score_mc
    num_id = getattr(cfg, "num_id", None) if getattr(cfg, "fone", False) else None
    mc_items = [
        {"prompt": it["context"],
         "options": [it["distractor"], it["target"]] if swap else [it["target"], it["distractor"]],
         "label": 1 if swap else 0}
        for it in items if it["distractor"] != it["target"]
    ]
    return score_mc(model, tok, mc_items, device, batch_size=batch, num_id=num_id)


def load_items(path, tok):
    items = []
    for d in (json.loads(l) for l in open(path, encoding="utf-8")):
        if "context" not in d:
            continue
        ids = tok.encode(d["context"], add_special_tokens=False).ids
        items.append({**d, "ctx_ids": ids})
    return items


def _selftest():
    import types
    tok = load_tokenizer(TOK_PATH, types.SimpleNamespace(vocab=None, vocab_id=None))
    prose = [
        json.dumps({"content": "他推开门，看见桌上放着一杯还冒着热气的茶。"}, ensure_ascii=False),
        json.dumps({"content": "雨下了一整夜，早上院子里落满了黄色的叶子。"}, ensure_ascii=False),
        json.dumps({"content": "她把信折好，轻轻放进了上衣最里面的口袋。"}, ensure_ascii=False),
        json.dumps({"content": "火车开动的时候，他才想起自己忘了带那本旧书。"}, ensure_ascii=False),
    ]
    tmp = os.path.join("/tmp", "_lzh_selftest.jsonl")
    open(tmp, "w", encoding="utf-8").write("\n".join(prose))
    items, stats = build_items(tok, tmp, n=10)
    assert items, f"no items built: {stats}"
    for it in items:
        ids = tok.encode(it["context"] + it["target"], add_special_tokens=False).ids
        assert ids[-1] == it["target_id"], it
        assert _CJK.fullmatch(it["target"]), it
        assert it["distractor"] != it["target"], it
        assert len(it["distractor"]) == len(it["target"]), it
    print(f"lambada_zh self-test OK ({len(items)} items, {stats})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out")
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--src", help="prose jsonl for --build")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return
    if a.build:
        if not a.src:
            ap.error("--build needs --src")
        import types
        tok = load_tokenizer(TOK_PATH, types.SimpleNamespace(vocab=None, vocab_id=None))
        items, stats = build_items(tok, a.src, n=a.n)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"built {len(items)} items -> {a.out} (skipped {stats})")
        return
    if not a.ckpt:
        ap.error("--ckpt required (or --build / --selftest)")
    if not os.path.exists(a.data):
        sys.exit(f"lambada_zh data not built: {a.data} (build with --build --src <held-out prose>)")

    model, cfg = load_checkpoint(a.ckpt, device=a.device, dtype=torch.bfloat16)
    tok = load_tokenizer(TOK_PATH, cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded {a.ckpt}: {n_params:.1f}M params, device={a.device}", flush=True)

    items = load_items(a.data, tok)
    print(f"items: {len(items)}", flush=True)
    acc1, acc5 = score_open_vocab(model, items, a.device)
    two_way = score_two_way(model, tok, cfg, items, a.device, swap=a.swap)
    result = {
        "ckpt": a.ckpt, "n_params": n_params, "swap": a.swap,
        "n_items": len(items),
        "provenance": sorted({it["src"] for it in items}),
        "open_acc1": acc1, "open_acc5": acc5,
        "two_way_acc": two_way,
    }
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(result, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

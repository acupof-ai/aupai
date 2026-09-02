#!/usr/bin/env python3
"""Likelihood-only eval matrix for base (non-instructed) checkpoints.

No generation anywhere: every metric compares sequence log-likelihoods. A base
LM reads zero on every generation eval (0/500 math-500, 0.8% boxed, 165-colon
loops on math-hard v2 -- facts/base_eval.json), and four zeroes look identical
to "measured" in the ledger. Likelihood is the standard base-model probe.

Metrics:
  minimal pairs  controlled zh pairs differing at one controlled point; the
                 model ranks the well-formed sentence higher = correct (floor 50%,
                 so a small model leaves the floor early -- denser signal than MC)
  chid           10-way zh idiom cloze by continuation log-likelihood (eval/chid_probe.py)
  MC tripwire    the existing MC suite, downgraded to a regression tripwire,
                 reported as z over chance, not as capability

Every metric carries a known-answer pair (correct labels vs swapped labels);
the two readings must differ by >= 60 points on the strongest checkpoint or the
metric is uncalibrated and gets cut (repo rule: tokenizer_report, 38af944).

Resolution: run on the 0830v1 ladder (0.2b->3.24b, 16x data span). A metric that
does not move across the ladder has no resolution at this scale and is cut.

Usage:
  python eval/base_matrix.py --ckpt ckpt_0830v1_0.2b.pt --out runs/bm_0.2b.json
  python eval/base_matrix.py --ckpt ckpt_0830v1_3.24b.pt --swap --out runs/bm_3.24b_swap.json
  python eval/base_matrix.py --summarize runs/bm_*.json
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("FLA_FLASH_KDA", "0")

import torch  # noqa: E402

from scripts.loader import EOS_ID, load_checkpoint, load_tokenizer  # noqa: E402

TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
KNOWN_ANSWER_SPREAD = 0.60  # correct vs swapped must differ by this on the best ckpt


# --- minimal-pair construction -------------------------------------------------
# Every pair is (dimension, correct, wrong). Builders only propose; the
# tokenization alignment check below decides what survives -- a pair that
# differs at more than one controlled point is skipped, not scored.

def _pairs_word_order(rng):
    """SVO vs SOV: 我吃了饭。 vs 我饭吃了。 -- both grammatical, identical token
    multiset; the pair tests order preference, not well-formedness."""
    subs = ["我", "你", "他", "她", "我们", "老师", "妈妈", "小猫", "小狗", "弟弟"]
    verbs = ["吃", "喝", "买", "看", "写", "洗", "关", "开", "找", "喂", "种", "画"]
    objs = ["饭", "水", "书", "字", "手", "门", "灯", "猫", "鱼", "苹果", "衣服", "花"]
    out = []
    for _ in range(100):
        s, v, o = rng.choice(subs), rng.choice(verbs), rng.choice(objs)
        out.append(("word_order", f"{s}{v}了{o}。", f"{s}{o}{v}了。"))
    return out


def _pairs_classifier(rng):
    """这本书。 vs 这张书。 -- the classifier is the only variable."""
    nouns = [("书", "本"), ("狗", "条"), ("猫", "只"), ("马", "匹"), ("牛", "头"),
             ("花", "朵"), ("树", "棵"), ("纸", "张"), ("灯", "盏"), ("衣服", "件"),
             ("鱼", "条"), ("笔", "支"), ("桌子", "张"), ("椅子", "把"), ("车", "辆"),
             ("飞机", "架"), ("信", "封"), ("诗", "首"), ("电影", "部"), ("比赛", "场"),
             ("帽子", "顶"), ("袜子", "双"), ("豆腐", "块"), ("剪刀", "把"), ("镜子", "面"),
             ("门", "扇"), ("山", "座"), ("桥", "座")]
    wrong_pool = ["张", "条", "只", "件", "本", "个", "把", "双", "朵", "支"]
    out = []
    for noun, cl in nouns:
        w = rng.choice([c for c in wrong_pool if c != cl])
        # 五/几, not 这/三: 这本 and 三本 merge into one token while 这本/三张
        # do not, so 这/三 pairs fail the same-length alignment check.
        out.append(("classifier", f"五{cl}{noun}。", f"五{w}{noun}。"))
        out.append(("classifier", f"几{cl}{noun}。", f"几{w}{noun}。"))
    return out


def _pairs_function_word(rng):
    """的/地/得 and 把/被 -- one function word swapped. 在/再 is absent:
    代词+在 merges (他在 one token) while 再 stays split, so those pairs fail
    the same-length alignment check."""
    out = []
    for adj, v in [("飞快", "跑"), ("高兴", "笑了"), ("认真", "学习"), ("慢慢", "走回家"),
                   ("轻轻", "关上门"), ("大声", "朗读课文"), ("安静", "坐着"), ("努力", "工作"),
                   ("仔细", "检查作业"), ("兴奋", "跳了起来"), ("耐心", "讲解"), ("偷偷", "溜走了"),
                   ("顺利", "通过了考试"), ("勉强", "答应了")]:
        out.append(("function_word", f"他{adj}地{v}。", f"他{adj}的{v}。"))
    for s, v in [("他", "跑"), ("她", "字写"), ("他", "汉语说"), ("她", "歌唱"), ("他", "饭吃"),
                 ("他", "球踢"), ("她", "舞跳")]:
        out.append(("function_word", f"{s}{v}得很快。", f"{s}{v}的很快。"))
    for s, o, v in [("他", "饭", "吃完了"), ("她", "衣服", "洗干净了"), ("我", "作业", "写完了"),
                    ("他", "窗户", "打开了"), ("风", "树", "吹倒了"), ("他", "信", "寄出去了"),
                    ("她", "房间", "收拾好了"), ("我", "杯子", "打碎了")]:
        out.append(("function_word", f"{s}把{o}{v}。", f"{s}被{o}{v}。"))
    return out


_FACTUAL = [
    ("北京是中国的首都。", "上海是中国的首都。"),
    ("中国的首都是北京。", "中国的首都是上海。"),
    ("长江是中国最长的河流。", "黄河是中国最长的河流。"),
    ("李白是唐代诗人。", "李白是宋代诗人。"),
    ("杜甫被称为诗圣。", "杜甫被称为诗仙。"),
    ("孔子是儒家学派的创始人。", "孔子是道家学派的创始人。"),
    ("企鹅生活在南极。", "企鹅生活在北极。"),
    ("太阳从东方升起。", "太阳从西方升起。"),
    ("水在零度结冰。", "水在十度结冰。"),
    ("一年有十二个月。", "一年有十三个月。"),
    ("人有五只手指。", "人有六只手指。"),
    ("汉语是中国的官方语言。", "日语是中国的官方语言。"),
    ("造纸术是中国的四大发明之一。", "地动仪是中国的四大发明之一。"),
    ("俄罗斯是世界上面积最大的国家。", "加拿大是世界上面积最大的国家。"),
    ("《红楼梦》的作者是曹雪芹。", "《红楼梦》的作者是罗贯中。"),
    ("圆周率约等于三点一四。", "圆周率约等于二点一四。"),
    ("中华人民共和国成立于一九四九年。", "中华人民共和国成立于一九五零年。"),
    ("《西游记》的主角是孙悟空。", "《西游记》的主角是贾宝玉。"),
]


def _pairs_factual(rng):
    return [("factual", c, w) for c, w in _FACTUAL]


def _pairs_numeric(rng):
    """Arithmetic templates with the computed result vs a same-width wrong one."""
    def two_digit(v):
        return 10 <= v <= 98

    out = []
    while len(out) < 100:
        t = rng.randrange(6)
        if t == 0:
            a, b = rng.randrange(11, 40), rng.randrange(1, 9)
            c, tmpl = a - b, "我有{a}个苹果,吃了{b}个,还剩{c}个。"
        elif t == 1:
            p, k = rng.randrange(2, 9), rng.randrange(3, 12)
            c, tmpl = p * k, "每支铅笔{p}元,买{k}支一共需要{c}元。"
        elif t == 2:
            a, b = rng.randrange(3, 20), rng.randrange(2, 15)
            c, tmpl = 2 * (a + b), "长方形的长是{a}米,宽是{b}米,它的周长是{c}米。"
        elif t == 3:
            a, b = rng.randrange(6, 30), rng.randrange(20, 40)
            c, tmpl = a + b, "小明今年{a}岁,爸爸比他大{b}岁,爸爸今年{c}岁。"
        elif t == 4:
            a, b = rng.randrange(10, 50), rng.randrange(3, 15)
            c, tmpl = a + b, "教室里有{a}张桌子,又搬来{b}张,现在有{c}张。"
        else:
            a, b = rng.randrange(20, 80), rng.randrange(5, 18)
            c, tmpl = a - b, "一本书有{a}页,已经看了{b}页,还剩{c}页没看。"
        e = rng.choice([2, 3, 5])
        w = c + e if (c + e <= 98 and not two_digit(c) or two_digit(c + e)) else c - e
        if not (two_digit(c) and two_digit(w) and w > 0 and w != c):
            continue
        d = dict(a=a, b=b, c=c, p=p if t == 1 else 0, k=k if t == 1 else 0)
        dw = dict(d, c=w)
        out.append(("numeric", tmpl.format(**d), tmpl.format(**dw)))
    return out


BUILDERS = [_pairs_word_order, _pairs_classifier, _pairs_function_word,
            _pairs_factual, _pairs_numeric]


def build_pairs(seed=20260830):
    """All proposed pairs, (dim, correct, wrong)."""
    rng = random.Random(seed)
    pairs = []
    for fn in BUILDERS:
        pairs.extend(fn(rng))
    return pairs


def align_pairs(pairs, tok):
    """Keep pairs whose tokenization differs at exactly one controlled point.

    substitution dims: same length, differing positions form one contiguous span
    word_order:       same length, identical token multiset (one permutation)
    Returns (kept, skipped_per_dim).
    """
    kept, skipped = [], Counter()
    for dim, c, w in pairs:
        ic, iw = tok.encode(c, add_special_tokens=False).ids, tok.encode(w, add_special_tokens=False).ids
        ok = False
        if len(ic) == len(iw):
            if dim == "word_order":
                ok = Counter(ic) == Counter(iw) and ic != iw
            else:
                diffs = [i for i, (a, b) in enumerate(zip(ic, iw)) if a != b]
                ok = bool(diffs) and diffs == list(range(diffs[0], diffs[-1] + 1))
        if ok:
            kept.append((dim, c, w))
        else:
            skipped[dim] += 1
    return kept, skipped


# --- likelihood scorer ----------------------------------------------------------

@torch.no_grad()
def sentence_logprobs(model, tok, sentences, device, batch=32):
    """Sum log p(token | preceding) per sentence, teacher-forced, EOS-prepended.

    Pairs are length-matched by construction, so raw sums are comparable; no
    length normalization (it would dilute the one-point difference).
    """
    enc = [tok.encode(s, add_special_tokens=False).ids for s in sentences]
    jobs = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    scores = [0.0] * len(enc)
    for s in range(0, len(jobs), batch):
        idx = jobs[s : s + batch]
        max_len = max(len(enc[i]) for i in idx) + 1  # +1 for the prepended EOS
        x = torch.full((len(idx), max_len), EOS_ID, dtype=torch.long, device=device)
        for b, i in enumerate(idx):
            x[b, 1 : 1 + len(enc[i])] = torch.tensor(enc[i], device=device)
        logits = model(x)[0].float()
        lp = logits[:, :-1].log_softmax(-1)  # predict positions 1..T from 0..T-1
        tgt = x[:, 1:]
        token_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        real = tgt != EOS_ID  # score real tokens only; EOS-prefix and padding excluded
        for b, i in enumerate(idx):
            scores[i] = token_lp[b][real[b]].sum().item()
    return scores


def score_pairs(model, tok, pairs, device, swap=False):
    """Per-dimension accuracy: correct sentence out-scores the wrong one."""
    sentences = [s for _, c, w in pairs for s in ((w, c) if swap else (c, w))]
    lp = sentence_logprobs(model, tok, sentences, device)
    by_dim, margins = {}, []
    for k, (dim, _c, _w) in enumerate(pairs):
        good, bad = lp[2 * k], lp[2 * k + 1]
        by_dim.setdefault(dim, [0, 0])
        by_dim[dim][0] += good > bad
        by_dim[dim][1] += 1
        margins.append(good - bad)
    return by_dim, sum(margins) / len(margins)


# --- CHID ------------------------------------------------------------------------

def score_chid(model, tok, cfg, device, n_items=500, seed=20260830):
    """10-way idiom cloze via the shared continuation-LH scorer. Data must be
    local (pod has no HF proxy): point --chid at a clue/chid parquet/jsonl."""
    from eval.chid_probe import load_chid
    from eval.run_eval import score_mc

    items = load_chid(split="dev")
    rng = random.Random(seed)
    rng.shuffle(items)
    items = items[:n_items]
    num_id = getattr(cfg, "num_id", None) if getattr(cfg, "fone", False) else None
    acc = score_mc(model, tok, items, device, batch_size=32, num_id=num_id)
    return {"acc": acc, "n": len(items), "chance": 0.10}


# --- MC tripwire ------------------------------------------------------------------

MC_TRIPWIRE = ["arc-easy", "arc-challenge", "winogrande", "piqa", "openbookqa",
               "boolq", "hellaswag", "mmlu", "ceval"]


def _constant_baseline(items):
    """Accuracy of the best always-answer-slot-i strategy, measured on these items.

    The decision line for any MC or short-answer eval is the STRONGEST CONSTANT
    strategy, never random (44's ruling 2026-09-01, docs/lessons/honest_measurement_prereg.md,
    after math_test_500 scored 9.78% by always answering '2' and three L1 points came in
    BELOW it while reading z=8.42 over shuffle). Measured on arc-easy, n=2376: constant
    26.64% against random 25.00%, so the +2se line moves 26.78% -> 28.45%. An accuracy in
    that 1.68pt band is 2 sigma over random and below a strategy that reads nothing.

    Returns None when the gold labels are not slot indices, rather than guessing: an
    unmeasurable baseline must not silently become a favourable one."""
    from collections import Counter
    if not items:
        return None
    golds = [it.get("answer", it.get("label")) for it in items]
    if any(not isinstance(g, int) for g in golds):
        return None
    return max(Counter(golds).values()) / len(items)


def z_over_chance(acc, chance, n):
    """z of the accuracy against the floor -- a tripwire reads in sigmas,
    not percent (3/5 pinned at 25% is invisible as a raw number).

    `chance` is the floor the CALLER chose; score_mc_tripwire passes
    max(random, constant), so this stays the arithmetic and the floor stays a decision
    made where the items are visible."""
    import math
    se = math.sqrt(chance * (1 - chance) / n)
    return (acc - chance) / se if se else 0.0


def score_mc_tripwire(model, tok, cfg, device, benchmarks=MC_TRIPWIRE):
    from eval.run_eval import MC_BENCHMARKS, score_mc

    num_id = getattr(cfg, "num_id", None) if getattr(cfg, "fone", False) else None
    out = {}
    for key in benchmarks:
        if key not in MC_BENCHMARKS:
            continue
        display, loader = MC_BENCHMARKS[key]
        try:
            items = loader()
            if not items:
                out[key] = {"error": "no items (dataset absent?)"}
                continue
            acc = score_mc(model, tok, items, device, batch_size=32, num_id=num_id)
        except Exception as e:
            out[key] = {"error": f"{type(e).__name__}: {str(e)[:90]}"}
            continue
        random_floor = 1.0 / len(items[0]["options"])
        const = _constant_baseline(items)
        # max(random, constant): the floor is whichever a no-capability strategy reaches.
        floor = max(random_floor, const) if const is not None else random_floor
        out[key] = {"acc": acc, "z": z_over_chance(acc, floor, len(items)),
                    "chance": floor, "random": random_floor, "constant": const,
                    "floor_is": ("constant" if const is not None and const > random_floor
                                 else "random" if const is not None else "random (constant unmeasurable)"),
                    "n": len(items)}
    return out


# --- resolution summary ------------------------------------------------------------

def summarize(runs):
    """Ladder table + resolution verdict per metric. A metric whose best-worst
    spread across the ladder is < 0.10 is declared no-resolution and cut."""
    data = [json.load(open(p)) for p in runs]
    data.sort(key=lambda d: d.get("n_params", 0))
    dims = sorted({d for d_ in data for d in d_["dimensions"]})
    print(f"{'ckpt':>28} " + " ".join(f"{d[:10]:>10}" for d in dims) + "  overall")
    for d in data:
        row = [d["dimensions"].get(k, {}).get("acc") for k in dims]
        print(f"{os.path.basename(d['ckpt']):>28} "
              + " ".join(f"{v:>10.1%}" if v is not None else f"{'-':>10}" for v in row)
              + f"  {d['overall']:>7.1%}")
    print("\nresolution (best - worst across ladder; <10pt = no resolution, cut):")
    for k in dims:
        vals = [d["dimensions"][k]["acc"] for d in data if k in d["dimensions"]]
        spread = max(vals) - min(vals)
        verdict = "KEEP" if spread >= 0.10 else "CUT"
        print(f"  {k:16s} spread {spread:+.1%}  {verdict}")
    if any("swap" in d for d in data):
        print("\nknown-answer spread (correct vs swapped, same ckpt):")
        by_ckpt = {d["ckpt"]: d for d in data}
        for d in data:
            if d.get("swap") and d["ckpt"] in by_ckpt:
                base = by_ckpt[d["ckpt"]]
                for k in dims:
                    if k in d["dimensions"] and k in base["dimensions"]:
                        spread = base["dimensions"][k]["acc"] - d["dimensions"][k]["acc"]
                        verdict = "OK" if spread >= KNOWN_ANSWER_SPREAD else "UNCALIBRATED"
                        print(f"  {k:16s} {spread:+.1%}  {verdict}")
                break


# --- self-check ---------------------------------------------------------------------

def _selftest():
    """Construction invariants + metric plumbing with a stub scorer (no model)."""
    # The floor assertions FIRST, because they need neither the tokenizer nor a dataset.
    # Everything below needs data/tokenizer.json, which is gitignored -- on a machine
    # without it the whole selftest died at line 1, so an assertion placed after this
    # point is never reached and never protects anything.
    _selftest_floor()
    tok = load_tokenizer(TOK_PATH, __import__("types").SimpleNamespace(
        vocab=None, vocab_id=None))
    pairs = build_pairs()
    assert len(pairs) >= 250, len(pairs)
    kept, skipped = align_pairs(pairs, tok)
    assert kept, "no pair survived tokenization alignment"
    dims = Counter(d for d, _, _ in kept)
    print(f"pairs: {len(pairs)} proposed, {len(kept)} aligned, "
          f"skipped {dict(skipped)}")
    for d, n in sorted(dims.items()):
        print(f"  {d:16s} {n}")
    # plumbing: a stub scorer that prefers the correct string by a keyword must
    # read 100% normally and 0% swapped.
    class StubTok:
        def encode(self, s, add_special_tokens=False):
            class E:
                ids = [ord(c) % 97 + 1 for c in s]
            return E()

    correct_kw = {c for _, c, _ in kept}
    def stub_score(sentences):
        return [1.0 if s in correct_kw else 0.0 for s in sentences]
    import eval.base_matrix as bm  # noqa: F401  (ensure importable)
    sentences = [s for _, c, w in kept for s in (c, w)]
    lp = stub_score(sentences)
    n_ok = sum(lp[2 * k] > lp[2 * k + 1] for k in range(len(kept)))
    assert n_ok == len(kept), "stub plumbing broken"
    sentences_swap = [s for _, c, w in kept for s in (w, c)]
    lp_swap = stub_score(sentences_swap)
    n_ok_swap = sum(lp_swap[2 * k] > lp_swap[2 * k + 1] for k in range(len(kept)))
    assert n_ok_swap == 0, "swap plumbing broken"
    print("base_matrix self-test OK")


def _selftest_floor():
    """The floor is the strongest constant strategy, not random.

    Constructed to FAIL if the floor reverts: at n=2376 an accuracy of 27.2% is 2.48
    sigma over random 25.00% and 0.62 sigma over the constant 26.64% measured on
    arc-easy -- signal under the old floor, nothing under the correct one. No dataset
    and no tokenizer needed; the arithmetic is what the ruling constrains."""
    skew = ([{"answer": 0}] * 633 + [{"answer": 1}] * 581
            + [{"answer": 2}] * 581 + [{"answer": 3}] * 581)
    const = _constant_baseline(skew)
    assert abs(const - 633 / 2376) < 1e-9, const
    floor = max(0.25, const)
    assert z_over_chance(0.272, 0.25, 2376) > 2, "random floor no longer calls 27.2% signal"
    assert z_over_chance(0.272, floor, 2376) < 2, "constant floor must NOT call 27.2% signal"
    assert _constant_baseline([{"answer": "A"}]) is None, "string golds must be unmeasurable, not 1.0"
    assert _constant_baseline([]) is None
    print(f"floor: constant {const:.4f} > random 0.2500; 27.2% reads "
          f"{z_over_chance(0.272, 0.25, 2376):.2f}s over random, "
          f"{z_over_chance(0.272, floor, 2376):.2f}s over constant")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--out", help="write per-metric JSON here")
    ap.add_argument("--swap", action="store_true", help="known-answer: invert labels")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chid", action="store_true", help="also score CHID (needs local data)")
    ap.add_argument("--mc", action="store_true", help="also score the MC tripwire")
    ap.add_argument("--summarize", nargs="+", metavar="JSON", help="resolution summary over ladder runs")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-nodata", action="store_true",
                    help="the assertions that need no tokenizer and no dataset (hook-runnable)")
    a = ap.parse_args()

    if a.selftest_nodata:
        _selftest_floor()
        print("base_matrix nodata self-test OK")
        return
    if a.selftest:
        _selftest()
        return
    if a.summarize:
        summarize(a.summarize)
        return
    if not a.ckpt:
        ap.error("--ckpt required (or --summarize / --selftest)")

    model, cfg = load_checkpoint(a.ckpt, device=a.device, dtype=torch.bfloat16)
    tok = load_tokenizer(TOK_PATH, cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded {a.ckpt}: {n_params:.1f}M params, bf16, device={a.device}", flush=True)

    pairs = build_pairs()
    kept, skipped = align_pairs(pairs, tok)
    print(f"pairs: {len(kept)} aligned ({dict(skipped)} skipped)", flush=True)
    by_dim, margin = score_pairs(model, tok, kept, a.device, swap=a.swap)

    result = {
        "ckpt": a.ckpt, "n_params": n_params, "swap": a.swap,
        "n_pairs": len(kept), "skipped": dict(skipped),
        "dimensions": {d: {"acc": c / n, "n": n} for d, (c, n) in sorted(by_dim.items())},
        "overall": sum(c for c, _ in by_dim.values()) / sum(n for _, n in by_dim.values()),
        "mean_margin_nats": margin,
    }
    if a.chid:
        result["chid"] = score_chid(model, tok, cfg, a.device)
    if a.mc:
        result["mc_tripwire"] = score_mc_tripwire(model, tok, cfg, a.device)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(result, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

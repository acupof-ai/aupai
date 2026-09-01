#!/usr/bin/env python3
"""Copy-rate probe: is the model reproducing its context instead of predicting?

Three decoding arms on ONE checkpoint and ONE prompt set (fb P0, 2026-09-01;
pre-registered in docs/lessons/copy_hypothesis_prereg.md before this ran):

  greedy      temperature 0
  sampled     temperature 0.8
  fewshot3    3 unrelated shots prepended, greedy

Copy rate = fraction of the generation's 8-grams that already appear in its
context. The DISTRIBUTION is the output; a mean hides a bimodal mix of copiers
and non-copiers, which is a different finding from uniform mild copying.

TWO THINGS THAT WOULD SILENTLY DECIDE THE ANSWER, both handled here:

1. rep_stop=False. train.generate_batch defaults to stopping when a whitespace
   8-gram repeats 3x -- i.e. it truncates exactly the behaviour under test. Left
   on, a copier's generation is cut short and its copy rate is measured over the
   surviving prefix. The stop is a product feature; here it is a confound.
2. The null. A base model under greedy decoding repeats by construction, so a
   high greedy copy rate alone proves nothing. That is why the sampled and
   few-shot arms are not optional extras -- they are the arms that can falsify.

Usage: python probes/t69_copy_rate.py --ckpt ckpt_pretrain_30b_s2.pt.step24000 \
           --out runs/copy_arms_step24000.json [--n 64] [--max_new 128]
"""
import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("FLA_FLASH_KDA", "0")

N = 8  # n-gram width, in TOKENS; reported explicitly because it is not word-level


def ngrams(seq, n=N):
    return {tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)}


def copy_rate(ctx_ids, gen_ids, n=N):
    """Fraction of the generation's n-grams that already occur in the context.

    Returns None when the generation is shorter than n tokens: a 3-token
    generation has no 8-gram and scoring it 0.0 would report 'not copying' for
    a model that produced nothing. Absent, not zero -- the count of these is
    reported alongside, because a checkpoint that mostly emits nothing is its
    own finding and must not hide inside a copy-rate distribution.
    """
    g = ngrams(gen_ids, n)
    if not g:
        return None
    return len(g & ngrams(ctx_ids, n)) / len(g)


def selftest():
    """Known answers. Copy rate is one line of set arithmetic and still has three
    ways to be silently wrong: an off-by-one in the window, scoring the context
    against itself, and returning 0.0 for a too-short generation."""
    ctx = list(range(100))
    assert copy_rate(ctx, list(range(20, 40))) == 1.0, "verbatim slice must score 1.0"
    assert copy_rate(ctx, list(range(500, 540))) == 0.0, "disjoint text must score 0.0"
    half = list(range(20, 32)) + list(range(900, 912))
    r = copy_rate(ctx, half)
    assert 0.2 < r < 0.8, f"half-copied text must land strictly between: got {r}"
    assert copy_rate(ctx, [1, 2, 3]) is None, "a sub-n generation is ABSENT, not 0.0"
    assert copy_rate(ctx, []) is None, "an empty generation is ABSENT, not 0.0"
    # the window itself: 8 tokens is exactly one 8-gram, 7 is none
    assert copy_rate(ctx, list(range(50, 58))) == 1.0, "exactly n tokens is one n-gram"
    assert copy_rate(ctx, list(range(50, 57))) is None, "n-1 tokens is no n-gram"
    print("selftest OK: 7 known answers (verbatim, disjoint, half, sub-n, empty, n, n-1)")
    return 0


def summarise(rates):
    """Distribution, not a mean. Deciles because the pre-registration reads a median
    and the shape decides whether a median means anything."""
    r = sorted(x for x in rates if x is not None)
    if not r:
        return {"n": 0}
    return {
        "n": len(r),
        "median": round(statistics.median(r), 4),
        "mean": round(statistics.fmean(r), 4),
        "p10": round(r[len(r) // 10], 4),
        "p25": round(r[len(r) // 4], 4),
        "p75": round(r[3 * len(r) // 4], 4),
        "p90": round(r[9 * len(r) // 10], 4),
        "min": round(r[0], 4),
        "max": round(r[-1], 4),
        "frac_above_0.9": round(sum(x >= 0.9 for x in r) / len(r), 4),
        "frac_below_0.3": round(sum(x <= 0.3 for x in r) / len(r), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "copy_arms.json"))
    ap.add_argument("--n", type=int, default=64, help="prompts")
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--prompt_tokens", type=int, default=256, help="context length per prompt")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "eval", "code_holdout_500.jsonl"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if not a.ckpt:
        ap.error("--ckpt required (unless --selftest)")

    selftest()  # never measure with an unverified metric

    import torch

    from scripts.loader import load_checkpoint, load_tokenizer
    from train import generate_batch

    model, cfg = load_checkpoint(a.ckpt, device=a.device)
    model = model.to(torch.bfloat16)
    tok = load_tokenizer(a.tokenizer, cfg)

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    texts = [(r.get("instruction") or r.get("question") or r.get("content") or "").strip()
             for r in rows]
    texts = [t for t in texts if t][: a.n]

    # ONE prompt set across all three arms: prompt choice can manufacture either
    # answer, so the arms must differ only in decoding.
    ctxs = [tok.encode(t).ids[: a.prompt_tokens] for t in texts]
    ctxs = [c for c in ctxs if len(c) >= N]

    # The few-shot arm's shots are UNRELATED -- drawn from the tail of the file so
    # they share no content with the prompts, which come from the head. If a copier
    # copies the shots instead of the prompt, that still counts as copying and the
    # rate is scored against the FULL context including shots.
    shot_pool = [t for t in
                 [(r.get("instruction") or r.get("question") or r.get("content") or "").strip()
                  for r in rows[-200:]] if t][:3]
    shot_ids = []
    for s in shot_pool:
        shot_ids.extend(tok.encode(s).ids[:128])

    arms = {
        "greedy": dict(temperature=0.0, shots=False),
        "sampled_t0.8": dict(temperature=0.8, shots=False),
        "fewshot3_greedy": dict(temperature=0.0, shots=True),
    }
    out = {
        "probe": "t69_copy_rate",
        "ckpt": os.path.basename(a.ckpt),
        "ngram_tokens": N,
        "n_prompts": len(ctxs),
        "max_new": a.max_new,
        "prompt_tokens": a.prompt_tokens,
        "data": os.path.relpath(a.data, ROOT),
        "rep_stop": False,
        "rep_stop_note": ("generate_batch defaults rep_stop=True, which truncates a "
                          "repeating generation -- the exact behaviour under test. "
                          "Disabled here; leaving it on measures the copy rate over a "
                          "truncated prefix and biases every arm toward the same answer."),
        "arms": {},
    }

    for name, spec in arms.items():
        prompts = [(shot_ids + c) if spec["shots"] else c for c in ctxs]
        with torch.no_grad():
            gens = generate_batch(model, prompts, a.max_new, a.device,
                                  spec["temperature"], None, tokenizer=tok, rep_stop=False)
        rates, short = [], 0
        for ctx, g in zip(prompts, gens, strict=True):
            r = copy_rate(ctx, list(g))
            if r is None:
                short += 1
            rates.append(r)
        s = summarise(rates)
        s["generations_too_short_to_score"] = short
        s["mean_gen_tokens"] = round(statistics.fmean(len(g) for g in gens), 1)
        out["arms"][name] = s
        print(f"  {name:18s} median {s.get('median')}  p10 {s.get('p10')}  p90 {s.get('p90')}  "
              f"n={s['n']}  short={short}  mean_len={s['mean_gen_tokens']}")

    g = out["arms"]["greedy"].get("median")
    for other in ("sampled_t0.8", "fewshot3_greedy"):
        o = out["arms"][other].get("median")
        if g is not None and o is not None:
            out["arms"][other]["drop_from_greedy"] = round(g - o, 4)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {a.out}")
    print("NO VERDICT from this script: the pre-registered thresholds are read by a human.")


if __name__ == "__main__":
    main()

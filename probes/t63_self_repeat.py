#!/usr/bin/env python3
"""Self-repetition probe: the alternative mechanism to context-copying.

t62 measured copy-from-context and found it near zero at greedy. That refutes
the copy hypothesis but leaves the reported symptom -- "generations are verbatim
repetition" -- unexplained, and a refutation that explains nothing is half a
result. There are two distinct behaviours that both look like "it repeats":

  copy-from-context   the generation reproduces the PROMPT      (t62, ~0.0)
  self-repetition     the generation reproduces ITSELF          (this probe)

They have different causes and different fixes, so the distinction is the
finding. This probe also dumps raw generations, because every number here is a
summary of text nobody has actually looked at yet.

Usage: python probes/t63_self_repeat.py --ckpt ckpt_... --out runs/self_repeat.json
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

N = 8


def self_repeat_rate(ids, n=N):
    """1 - (distinct n-grams / total n-grams). 0.0 = never repeats itself,
    approaching 1.0 = says the same thing over and over. None if too short."""
    total = len(ids) - n + 1
    if total <= 0:
        return None
    grams = [tuple(ids[i : i + n]) for i in range(total)]
    return 1.0 - len(set(grams)) / total


def max_run(ids, n=N):
    """Longest number of consecutive positions whose n-gram was already seen --
    a single degenerate loop at the tail, versus scattered repetition."""
    total = len(ids) - n + 1
    if total <= 0:
        return None
    seen, run, best = set(), 0, 0
    for i in range(total):
        g = tuple(ids[i : i + n])
        if g in seen:
            run += 1
            best = max(best, run)
        else:
            run = 0
            seen.add(g)
    return best


def selftest():
    a = list(range(100))
    assert self_repeat_rate(a) == 0.0, "all-distinct must be 0.0"
    b = list(range(20)) * 5
    r = self_repeat_rate(b)
    assert r > 0.7, f"a 5x repeated block must score high: {r}"
    assert self_repeat_rate([7] * 50) > 0.8, "a constant run must score high"
    assert self_repeat_rate([1, 2, 3]) is None, "sub-n is ABSENT not 0.0"
    assert max_run(a) == 0, "all-distinct has no run"
    assert max_run([7] * 50) > 30, "a constant run is one long run"
    print("selftest OK: 6 known answers (distinct, block, constant, sub-n, runs)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "self_repeat.json"))
    ap.add_argument("--dump", default=None, help="write raw generations here")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--prompt_tokens", type=int, default=256)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "eval", "code_holdout_500.jsonl"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if not a.ckpt:
        ap.error("--ckpt required")
    selftest()

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
    ctxs = [tok.encode(t).ids[: a.prompt_tokens] for t in texts]
    ctxs = [c for c in ctxs if len(c) >= N]

    out = {"probe": "t63_self_repeat", "ckpt": os.path.basename(a.ckpt),
           "ngram_tokens": N, "n_prompts": len(ctxs), "max_new": a.max_new,
           "rep_stop": False, "arms": {}}
    dump = []

    for name, temp in (("greedy", 0.0), ("sampled_t0.8", 0.8)):
        with torch.no_grad():
            gens = generate_batch(model, ctxs, a.max_new, a.device, temp, None,
                                  tokenizer=tok, rep_stop=False)
        rates = [self_repeat_rate(list(g)) for g in gens]
        runs = [max_run(list(g)) for g in gens]
        r = sorted(x for x in rates if x is not None)
        ru = sorted(x for x in runs if x is not None)
        out["arms"][name] = {
            "n": len(r),
            "self_repeat_median": round(statistics.median(r), 4) if r else None,
            "self_repeat_mean": round(statistics.fmean(r), 4) if r else None,
            "self_repeat_p10": round(r[len(r) // 10], 4) if r else None,
            "self_repeat_p90": round(r[9 * len(r) // 10], 4) if r else None,
            "frac_above_0.5": round(sum(x >= 0.5 for x in r) / len(r), 4) if r else None,
            "max_run_median": ru[len(ru) // 2] if ru else None,
            "max_run_p90": ru[9 * len(ru) // 10] if ru else None,
            "mean_gen_tokens": round(statistics.fmean(len(g) for g in gens), 1),
        }
        s = out["arms"][name]
        print(f"  {name:14s} self-repeat median {s['self_repeat_median']}  "
              f"p90 {s['self_repeat_p90']}  frac>=0.5 {s['frac_above_0.5']}  "
              f"max_run median {s['max_run_median']}")
        if a.dump:
            for i in range(min(6, len(gens))):
                dump.append({"arm": name, "i": i,
                             "prompt": tok.decode(ctxs[i])[:400],
                             "generation": tok.decode(list(gens[i]))[:800],
                             "self_repeat": rates[i]})

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {a.out}")
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as f:
            for d in dump:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"wrote {a.dump} ({len(dump)} raw generations -- read them)")


if __name__ == "__main__":
    main()

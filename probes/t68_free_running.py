#!/usr/bin/env python3
"""Free-running per-token agreement vs teacher-forced, on one run and one prompt set.

Pre-registered in docs/lessons/free_running_prereg.md before this ran.

t66 measured gold top-1 at 72.7% TEACHER-FORCED -- every position scored given a
correct prefix the model would not have produced. That is an upper bound on what
the model can do on its own prefix, and it biases toward the decoding-deficit
conclusion it produced. This measures the same quantity free-running.

Also carries the CORRECTED form of a vacuous row from the t66 pre-registration:
gold log-prob against SAMPLED sequence log-probs, not against the argmax. The
old comparison could only ever read 0.0 -- greedy is the per-position maximum by
construction, so gold can never beat it unless it IS the argmax. A sampled
sequence is not the maximum, so the comparison is real.

Usage: python probes/t68_free_running.py --ckpt ckpt_... --out runs/free_running.json
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

SETS = {
    "code_500": ("data/eval/code_holdout_500.jsonl", "instruction", "reference_code"),
    "math_500": ("data/eval/math_test_500.jsonl", "question", "answer"),
}


def load_pairs(root, path, qk, ak, limit):
    out = []
    for line in open(os.path.join(root, path), encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        q = (r.get(qk) or r.get("instruction") or r.get("question") or "").strip()
        a = r.get(ak)
        if a is None:
            for k in ("answer", "reference_code", "solution", "output"):
                if r.get(k):
                    a = r[k]
                    break
        a = (a or "").strip()
        if q and a:
            out.append((q, a))
        if len(out) >= limit:
            break
    return out


def agreement(gen, gold):
    """Position-by-position match over the overlap. Naive: index i vs index i."""
    n = min(len(gen), len(gold))
    if n == 0:
        return None
    return sum(gen[i] == gold[i] for i in range(n)) / n


def best_shift_agreement(gen, gold, max_shift=4):
    """Best agreement over small alignment offsets.

    A single extra token early desynchronises every later position, so the naive
    number cannot tell 'wrong tokens' from 'shifted tokens'. If this is much
    higher than naive, the naive number was measuring desynchronisation.
    """
    best = 0.0
    for s in range(-max_shift, max_shift + 1):
        g = gen[s:] if s > 0 else gen
        d = gold[-s:] if s < 0 else gold
        a = agreement(g, d)
        if a is not None:
            best = max(best, a)
    return best


def selftest():
    """Agreement has two silent failure modes: comparing different lengths as if
    aligned, and a shift search that cannot actually find a shift."""
    assert agreement([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert agreement([1, 2, 3, 4], [9, 9, 9, 9]) == 0.0
    assert agreement([1, 2, 9, 9], [1, 2, 3, 4]) == 0.5
    assert agreement([], [1, 2]) is None, "empty generation is ABSENT, not 0.0"
    # a one-token insertion destroys the naive number and the shift search recovers it
    gold = list(range(20))
    shifted = [999] + gold
    assert agreement(shifted, gold) < 0.1, "naive must collapse on a 1-token insertion"
    assert best_shift_agreement(shifted, gold) > 0.9, "shift search must recover it"
    print("selftest OK: exact, disjoint, half, empty-absent, insertion collapses naive & shift recovers")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "free_running.json"))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--k", type=int, default=8, help="samples for the gold-vs-sampled comparison")
    ap.add_argument("--max_new", type=int, default=192)
    ap.add_argument("--batch", type=int, default=32,
                    help="sequence budget per generate call in the gold-vs-sampled arm")
    ap.add_argument("--gvs_problems", type=int, default=40,
                    help="problems scored in the gold-vs-sampled arm")
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

    from probes.t66_gold_reachability import gold_ranks
    from scripts.loader import load_checkpoint, load_tokenizer
    from train import generate_batch

    model, cfg = load_checkpoint(a.ckpt, device=a.device)
    model = model.to(torch.bfloat16)
    model.eval()
    tok = load_tokenizer(a.tokenizer, cfg)

    res = {"probe": "t68_free_running", "ckpt": os.path.basename(a.ckpt),
           "limit": a.limit, "k": a.k, "max_new": a.max_new, "batch": a.batch,
           "gvs_problems": a.gvs_problems, "rep_stop": False, "sets": {}}

    def seq_logprob(prompt_ids, cont_ids):
        """log P(cont | prompt) under the model -- used for gold vs sampled."""
        s = prompt_ids + cont_ids
        x = torch.tensor([s], dtype=torch.long, device=a.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            lg = model(x)
        if isinstance(lg, tuple):
            lg = lg[0]
        lp = torch.log_softmax(lg[0, len(prompt_ids) - 1 : len(s) - 1].float(), dim=-1)
        tgt = torch.tensor(cont_ids, device=lp.device)
        return float(lp.gather(1, tgt[:, None]).sum())

    for name, (p, qk, ak) in SETS.items():
        pairs = load_pairs(ROOT, p, qk, ak, a.limit)
        prompts = [tok.encode(q).ids[:512] for q, _ in pairs]
        golds = [tok.encode(g).ids for _, g in pairs]

        tf = gold_ranks(model, tok, pairs, a.device)
        tf_top1 = statistics.median(r["top1"] for r in tf) if tf else None

        arms = {}
        for arm, temp in (("greedy", 0.0), ("sampled_t0.8", 0.8)):
            with torch.no_grad():
                gens = generate_batch(model, prompts, a.max_new, a.device, temp, None,
                                      tokenizer=tok, rep_stop=False)
            naive, shifted = [], []
            for g, gold in zip(gens, golds, strict=True):
                x = agreement(list(g), gold)
                if x is not None:
                    naive.append(x)
                    shifted.append(best_shift_agreement(list(g), gold))
            arms[arm] = {
                "n": len(naive),
                "fr_naive_median": round(statistics.median(naive), 4) if naive else None,
                "fr_shift_median": round(statistics.median(shifted), 4) if shifted else None,
                "fr_naive_mean": round(statistics.fmean(naive), 4) if naive else None,
            }

        # The CORRECTED row: gold vs SAMPLED sequences (not vs the argmax).
        #
        # BATCHED. The first version issued one generate call per sample -- 40
        # problems x k=8 = 320 serial single-sequence generations per set, which
        # dominated the whole probe while the other two arms were batched. The
        # cost was self-inflicted, not intrinsic to the measurement, so the
        # committed version carries the fixed cost rather than mine.
        beats = tot = 0
        n_probs = min(len(pairs), a.gvs_problems)
        # a.batch is the SEQUENCE budget per generate call; each problem costs k
        # sequences, so step problems at a time.
        step = max(1, a.batch // max(1, a.k))
        for lo in range(0, n_probs, step):
            hi = min(lo + step, n_probs)
            flat = [prompts[i] for i in range(lo, hi) for _ in range(a.k)]
            with torch.no_grad():
                samp = generate_batch(model, flat, a.max_new, a.device,
                                      0.8, None, tokenizer=tok, rep_stop=False)
            for j, i in enumerate(range(lo, hi)):
                gold_ids = golds[i][:a.max_new]
                gp = seq_logprob(prompts[i], gold_ids)
                gnorm = gp / max(1, len(gold_ids))
                for s in samp[j * a.k : (j + 1) * a.k]:
                    if len(s) == 0:
                        continue
                    sp = seq_logprob(prompts[i], list(s))
                    # length-normalise: raw sums would just report "shorter wins"
                    if gnorm >= sp / max(1, len(s)):
                        beats += 1
                    tot += 1

        res["sets"][name] = {
            "teacher_forced_top1_median": round(tf_top1, 4) if tf_top1 else None,
            "free_running": arms,
            "gold_beats_sampled_frac": round(beats / tot, 4) if tot else None,
            "gold_vs_sampled_n": tot,
            "ratio_fr_over_tf": (round(arms["greedy"]["fr_naive_median"] / tf_top1, 4)
                                 if tf_top1 and arms["greedy"]["fr_naive_median"] is not None else None),
        }
        s = res["sets"][name]
        print(f"  {name}: TF top1 {s['teacher_forced_top1_median']}  "
              f"FR greedy {arms['greedy']['fr_naive_median']} (shift {arms['greedy']['fr_shift_median']})  "
              f"FR t0.8 {arms['sampled_t0.8']['fr_naive_median']}  "
              f"gold>=sampled {s['gold_beats_sampled_frac']}  ratio {s['ratio_fr_over_tf']}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"wrote {a.out}")
    print("NO VERDICT: readings are in docs/lessons/free_running_prereg.md")


if __name__ == "__main__":
    main()

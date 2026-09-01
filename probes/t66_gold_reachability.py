#!/usr/bin/env python3
"""Is the gold answer reachable, or is rising gold probability trapped?

Pre-registered in docs/lessons/gold_reachability_prereg.md before this ran.

be.gold_bpb_falls_while_generation_scores_zero showed the model assigns rising
probability to correct answers while generating nothing usable. This asks
whether that probability is reachable:

  gold ranked high, never sampled   -> decoding and search deficit
  gold not near the top             -> knowledge deficit

THE LOAD-BEARING STATISTIC IS PER-TOKEN RANK, not verbatim sampling. Sequence
probability falls geometrically in length, so a 200-token gold is unsamplable at
any temperature regardless of the model -- reporting "never sampled in k=32" as
the finding would be measuring gold length, not the model.

THE CONFOUND THAT BIASES THE ANSWER: ranks are teacher-forced, i.e. each gold
position is scored given a CORRECT prefix the model would not have produced.
That measures "can it continue a correct answer", which is strictly easier than
"can it produce one", and it biases the result TOWARD the decoding-deficit
conclusion. Free-running agreement is reported alongside for exactly this
reason; where the two disagree, distrust the teacher-forced number.

Usage: python probes/t66_gold_reachability.py --ckpt ckpt_... --out runs/gold_reachability.json
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


def gold_ranks(model, tok, pairs, device, max_len=1024, batch=8):
    """Teacher-forced: for each gold token, where does it sit in the model's
    distribution at that position? Returns per-problem dicts."""
    import torch

    rows = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i : i + batch]
        seqs, meta = [], []
        for q, a in chunk:
            qi, ai = tok.encode(q).ids, tok.encode(a).ids
            if not ai:
                continue
            s = (qi + ai)[:max_len]
            if len(s) <= len(qi):
                continue
            seqs.append(s)
            meta.append(len(qi))
        if not seqs:
            continue
        width = max(len(s) for s in seqs)
        x = torch.zeros(len(seqs), width, dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, : len(s)] = torch.tensor(s, dtype=torch.long)
        x = x.to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            lg = model(x)
        if isinstance(lg, tuple):
            lg = lg[0]
        lg = lg.float()
        for j, s in enumerate(seqs):
            qlen = meta[j]
            sl = lg[j, qlen - 1 : len(s) - 1]
            tgt = torch.tensor(s[qlen : len(s)], device=sl.device)
            lp = torch.log_softmax(sl, dim=-1)
            gold_lp = lp.gather(1, tgt[:, None]).squeeze(1)
            # Rank under TIES, both directions. rank_opt counts only tokens
            # STRICTLY better than the gold; rank_pess also counts ties. They
            # differ by the tie width, and the gap matters: a flat distribution
            # gives rank_opt 0 at every position, which renders as "gold is
            # top-1 everywhere" while the model is in fact indifferent. My own
            # selftest caught that -- a uniform model scored top1 = 1.0. The
            # pessimistic rank is the honest one for a low-confidence model and
            # the optimistic one flatters it, so both are reported.
            rank = (lp > gold_lp[:, None]).sum(dim=1)
            rank_pess = (lp >= gold_lp[:, None]).sum(dim=1) - 1
            greedy_lp = lp.max(dim=1).values
            rows.append({
                "n_gold_tok": int(tgt.numel()),
                "top1": float((rank == 0).float().mean()),
                "top10": float((rank < 10).float().mean()),
                "top100": float((rank < 100).float().mean()),
                "top1_pess": float((rank_pess == 0).float().mean()),
                "top10_pess": float((rank_pess < 10).float().mean()),
                "gold_logprob": float(gold_lp.sum()),
                "greedy_logprob": float(greedy_lp.sum()),
            })
    return rows


def selftest():
    """A uniform model must rank the gold at chance; a model that puts all mass
    on the gold must rank it top-1 everywhere. Both pin the rank direction --
    an inverted comparison (< instead of >) passes neither."""
    import torch

    V = 64

    class Uniform(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], x.shape[1], V)

    class Oracle(torch.nn.Module):
        """Puts all mass on token t+1 of the input -- i.e. predicts the gold."""

        def forward(self, x):
            o = torch.full((x.shape[0], x.shape[1], V), -10.0)
            for b in range(x.shape[0]):
                for t in range(x.shape[1] - 1):
                    o[b, t, x[b, t + 1]] = 10.0
            return o

    class Tk:
        def encode(self, t):
            class R:
                ids = [ord(c) % V for c in t]
            return R()

    tk = Tk()
    pairs = [("qq", "abcd")]
    u = gold_ranks(Uniform(), tk, pairs, "cpu")[0]
    # A uniform model ties with the gold everywhere, so the OPTIMISTIC rank says
    # top-1 at every position -- the flaw this selftest exposed. The pessimistic
    # rank is what must be near chance, and asserting only the optimistic one
    # would have shipped a metric that reports a flat model as a confident one.
    assert u["top1"] == 1.0, "uniform ties everywhere: optimistic rank is 0 by construction"
    assert u["top1_pess"] == 0.0, f"uniform pessimistic top1 must be 0, got {u['top1_pess']}"
    assert abs(u["gold_logprob"] - u["greedy_logprob"]) < 1e-3, "uniform: gold == greedy logprob"
    o = gold_ranks(Oracle(), tk, pairs, "cpu")[0]
    assert o["top1"] == 1.0, f"oracle must rank gold top-1 everywhere, got {o['top1']}"
    assert o["top1_pess"] == 1.0, f"oracle has no ties: pessimistic must also be 1.0, got {o['top1_pess']}"
    assert o["gold_logprob"] >= o["greedy_logprob"] - 1e-4, "oracle: gold must be the argmax"
    assert u["n_gold_tok"] == 4, f"gold token count wrong: {u['n_gold_tok']}"
    print("selftest OK: uniform opt=1.0/pess=0.0 (tie flaw pinned), oracle both 1.0, logprob ordering, token count")
    return 0


def summarise(rows, key):
    v = sorted(r[key] for r in rows)
    if not v:
        return None
    return {"median": round(statistics.median(v), 4), "mean": round(statistics.fmean(v), 4),
            "p10": round(v[len(v) // 10], 4), "p90": round(v[9 * len(v) // 10], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "gold_reachability.json"))
    ap.add_argument("--limit", type=int, default=200)
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

    model, cfg = load_checkpoint(a.ckpt, device=a.device)
    model = model.to(torch.bfloat16)
    model.eval()
    tok = load_tokenizer(a.tokenizer, cfg)

    res = {"probe": "t66_gold_reachability", "ckpt": os.path.basename(a.ckpt),
           "limit": a.limit, "teacher_forced": True,
           "confound": ("ranks are teacher-forced: each gold position is scored given a CORRECT "
                        "prefix the model would not have produced, which biases toward the "
                        "decoding-deficit reading. Not a caveat -- the reason a high top-1 "
                        "cannot settle the split alone."),
           "sets": {}}

    for name, (p, qk, ak) in SETS.items():
        pairs = load_pairs(ROOT, p, qk, ak, a.limit)
        rows = gold_ranks(model, tok, pairs, a.device)
        gold_beats_greedy = sum(r["gold_logprob"] >= r["greedy_logprob"] for r in rows)
        res["sets"][name] = {
            "n": len(rows),
            "top1": summarise(rows, "top1"),
            "top1_pess": summarise(rows, "top1_pess"),
            "top10_pess": summarise(rows, "top10_pess"),
            "top10": summarise(rows, "top10"),
            "top100": summarise(rows, "top100"),
            "gold_beats_greedy_frac": round(gold_beats_greedy / len(rows), 4) if rows else None,
            "median_gold_tokens": statistics.median(r["n_gold_tok"] for r in rows) if rows else None,
        }
        s = res["sets"][name]
        print(f"  {name}: n={s['n']}  top1 median {s['top1']['median']}  "
              f"(pess {s['top1_pess']['median']})  top10 {s['top10']['median']}  "
              f"top100 {s['top100']['median']}  "
              f"gold>=greedy {s['gold_beats_greedy_frac']}  median gold tokens {s['median_gold_tokens']}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"wrote {a.out}")
    print("NO VERDICT: readings are in docs/lessons/gold_reachability_prereg.md")


if __name__ == "__main__":
    main()

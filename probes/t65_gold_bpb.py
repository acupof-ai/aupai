#!/usr/bin/env python3
"""Gold-answer BPB across checkpoints: does the model assign rising probability
to correct answers, even where it cannot generate them?

Pre-registered in docs/lessons/gold_bpb_prereg.md before this ran.

WHY THIS METRIC. Every instrument used on 2026-09-01 passes through a decoder,
and the decoder is broken in a way that manufactures zeros: 74-80% of greedy
generations loop, and the repetition guard truncates the metric that would have
measured it. Gold BPB has no decoder -- conditional NLL of the gold string given
the prompt, over the gold's UTF-8 byte count. No sampling, no stopping, no fence
parsing. None of today's confounds reach it.

BYTES, NOT TOKENS, is the whole point: it makes the number comparable across
tokenizers and immune to a tokenizer that splits the gold differently. Per-token
loss is not comparable across checkpoints if anything about tokenization moved.

Usage: python probes/t65_gold_bpb.py --ckpt A.pt --ckpt B.pt --out runs/gold_bpb.json
"""
import argparse
import json
import os
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
            for k in ("answer", "reference_code", "solution", "output", "expected_output"):
                if r.get(k):
                    a = r[k]
                    break
        a = (a or "").strip()
        if q and a:
            out.append((q, a))
        if len(out) >= limit:
            break
    return out


def gold_bpb(model, tok, pairs, device, batch=8, max_len=1024):
    """Sum NLL of the GOLD tokens only (prompt tokens masked), divided by total
    gold UTF-8 bytes. Returns (bpb, n_scored, gold_bytes, gold_tokens).

    The prompt is masked because this asks how well the model predicts the
    ANSWER given the question -- scoring the prompt too would mix in how
    predictable the question is, which is a property of the dataset and not of
    the checkpoint.
    """
    import torch

    tot_nll = 0.0
    tot_bytes = 0
    tot_gold_tok = 0
    n = 0
    for i in range(0, len(pairs), batch):
        chunk = pairs[i : i + batch]
        seqs, splits = [], []
        for q, a in chunk:
            qi = tok.encode(q).ids
            ai = tok.encode(a).ids
            if not ai:
                continue
            s = (qi + ai)[:max_len]
            if len(s) <= len(qi):
                continue  # gold entirely truncated away: skip, do not score as 0
            seqs.append(s)
            splits.append((len(qi), a))
        if not seqs:
            continue
        width = max(len(s) for s in seqs)
        x = torch.zeros(len(seqs), width, dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, : len(s)] = torch.tensor(s, dtype=torch.long)
        x = x.to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.float()
        for j, s in enumerate(seqs):
            qlen, gold_text = splits[j]
            # predict position t+1 from position t: gold tokens are s[qlen:]
            lp = torch.log_softmax(logits[j, qlen - 1 : len(s) - 1], dim=-1)
            tgt = torch.tensor(s[qlen : len(s)], device=lp.device)
            nll = -lp.gather(1, tgt[:, None]).sum().item()
            tot_nll += nll
            tot_bytes += len(gold_text.encode("utf-8"))
            tot_gold_tok += len(tgt)
            n += 1
    if not tot_bytes:
        return None, 0, 0, 0
    # nats -> bits, per UTF-8 byte
    import math

    return tot_nll / math.log(2) / tot_bytes, n, tot_bytes, tot_gold_tok


def selftest():
    """The metric has three silent failure modes: scoring the prompt as well as
    the gold, an off-by-one in the logits slice, and dividing by tokens instead
    of bytes. A uniform model over a known vocab pins all three."""
    import torch

    V = 256

    class Uniform(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], x.shape[1], V)

    class Tk:
        def encode(self, t):
            class R:
                ids = [ord(c) % V for c in t]
            return R()

    m, tk = Uniform(), Tk()
    # Under a uniform model every token costs log2(V) = 8 bits. The golds here are
    # 1-byte-per-char ASCII with 1 token per char, so BPB must be exactly 8.0.
    pairs = [("qq", "abcd"), ("zz", "efgh")]
    b, n, by, tokn = gold_bpb(m, tk, pairs, "cpu", batch=2, max_len=64)
    assert n == 2, f"scored {n} pairs, expected 2"
    assert by == 8, f"gold bytes {by}, expected 8 (two 4-char golds)"
    assert tokn == 8, f"gold tokens {tokn}, expected 8"
    assert abs(b - 8.0) < 1e-4, f"uniform BPB {b}, expected log2(256)=8.0"
    # A 2-byte-per-char gold at 1 token per char must HALVE the bits per byte.
    b2, _, by2, tk2 = gold_bpb(m, tk, [("qq", "éééé")], "cpu", max_len=64)
    assert by2 == 8 and tk2 == 4, f"expected 8 bytes / 4 tokens, got {by2}/{tk2}"
    assert abs(b2 - 4.0) < 1e-4, f"2-byte chars must give BPB 4.0, got {b2}"
    print("selftest OK: uniform=8.0 bits/byte, 2-byte chars=4.0, prompt masked, byte-normalised")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "gold_bpb.json"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if not a.ckpt:
        ap.error("--ckpt required (repeatable)")
    selftest()

    import hashlib

    import torch

    from scripts.loader import load_checkpoint, load_tokenizer

    tok_file_fp = hashlib.sha256(open(a.tokenizer, "rb").read()).hexdigest()[:16]
    data = {k: load_pairs(ROOT, p, qk, ak, a.limit) for k, (p, qk, ak) in SETS.items()}
    for k, v in data.items():
        print(f"  {k}: {len(v)} (question, gold) pairs")

    res = {"probe": "t65_gold_bpb", "tokenizer_file_fp": tok_file_fp, "limit": a.limit,
           "sets": {k: len(v) for k, v in data.items()}, "checkpoints": []}

    for ck in a.ckpt:
        model, cfg = load_checkpoint(ck, device=a.device)
        model = model.to(torch.bfloat16)
        model.eval()
        tok = load_tokenizer(a.tokenizer, cfg)
        row = {"ckpt": os.path.basename(ck)}
        for name, pairs in data.items():
            b, n, by, gt = gold_bpb(model, tok, pairs, a.device)
            row[name] = {"bpb": round(b, 5) if b else None, "n": n,
                         "gold_bytes": by, "gold_tokens": gt}
            print(f"  {os.path.basename(ck):46s} {name:9s} BPB {b:.5f}  n={n}")
        res["checkpoints"].append(row)
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"wrote {a.out}")
    print("NO VERDICT: the pre-registered readings are in docs/lessons/gold_bpb_prereg.md")


if __name__ == "__main__":
    main()

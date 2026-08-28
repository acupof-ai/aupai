#!/usr/bin/env python3
"""Distil the 27B's judgement into a classifier head on our own 200M model.

Stage two of FineWeb-Edu's architecture. The 27B answers 0.76 documents a second
on one H20; `data/corpus/web` holds 1.97M documents, so scoring them all with it
would take four days on the whole pod. It labels a sample instead, and a cheap
model learns the mapping and scores everything.

The cheap model is our own pretrained checkpoint with a 2-way head on the mean
hidden state. Two reasons over the obvious alternatives:

  * Hashed character n-grams were measured at AUC 0.60 against hand labels and
    the diagnosis was a representation ceiling, not a data one -- they rank by
    TOPIC and the labels split on REGISTER, so a page about air conditioners
    scores the same whether it is a technical explainer or a product sheet. More
    labels do not fix that; features that carry meaning do.
  * opencsg trained exactly this classifier for Chinese and published the
    dataset card but not the weights on HuggingFace, so it cannot be reused.

Scoring 1.97M documents with a 200M model at 512 tokens is minutes, not days.

    python datagen/train_quality_head.py --labels data/web_27b_labels.jsonl \\
        --ckpt ckpt_k5_clean_0827.pt --tokenizer data/tokenizer_k5.json \\
        --check data/web_labels.jsonl --out data/quality_head.pt
    python datagen/train_quality_head.py --score 'data/corpus/web/*.jsonl' \\
        --head data/quality_head.pt --out data/web_scores.npy
"""

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

SEQ = 512


def _encode(texts, tok, seq=SEQ, pad=0):
    import torch

    rows = []
    for e in tok.encode_batch([t[:1500] for t in texts]):
        ids = e.ids[:seq]
        rows.append(ids + [pad] * (seq - len(ids)))
    return torch.tensor(rows, dtype=torch.long)


def _features(model, x, device, batch=32):
    """Mean hidden state over non-padding positions, in fp32."""
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i : i + batch].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                _, h = model(xb, return_hidden=True)
            m = (xb != 0).unsqueeze(-1).float()
            out.append(((h.float() * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
    return torch.cat(out)


def auc(y, s):
    import numpy as np

    y, s = np.asarray(y, float), np.asarray(s, float)
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos, nneg = y.sum(), len(y) - y.sum()
    return float("nan") if not npos or not nneg else (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/web_27b_labels.jsonl")
    ap.add_argument("--ckpt", default="ckpt_k5_clean_0827.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer_k5.json")
    ap.add_argument("--check", help="hand-labelled jsonl; the honest held-out set")
    ap.add_argument("--out", default="data/quality_head.pt")
    ap.add_argument("--score", help="glob to score instead of training")
    ap.add_argument("--head", default="data/quality_head.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()

    import torch

    from loader import load_checkpoint, load_tokenizer

    model, cfg = load_checkpoint(a.ckpt, device=a.device)
    # Non-contiguous parameters make cublasGemmEx fail outright on this checkpoint;
    # eval/ppl.py carries the same line for the same reason.
    for pmt in model.parameters():
        pmt.data = pmt.data.contiguous()
    tok = load_tokenizer(a.tokenizer, cfg)

    if a.score:
        w = torch.load(a.head, map_location="cpu", weights_only=True)
        import numpy as np

        scores, n = [], 0
        for f in sorted(glob.glob(a.score)):
            with open(f, encoding="utf-8") as fh:
                texts = [json.loads(x).get("content", "") for x in fh if x.strip()]
            for i in range(0, len(texts), 512):
                feats = _features(model, _encode(texts[i : i + 512], tok), a.device)
                scores.append((feats @ w["w"] + w["b"]).numpy())
            n += len(texts)
            print(f"  {n} documents", flush=True)
        arr = np.concatenate(scores)
        np.save(a.out, arr)
        q = np.percentile(arr, [10, 25, 50, 75, 90])
        print(f"{len(arr)} scored -> {a.out}; deciles " + " ".join(f"{v:.2f}" for v in q))
        return

    with open(a.labels, encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    y = torch.tensor([float(r["s"]) for r in rows])
    print(f"{len(rows)} teacher labels, {y.mean():.1%} positive", flush=True)
    X = _features(model, _encode([r["t"] for r in rows], tok), a.device)

    # Logistic head on frozen features. The backbone is not fine-tuned: with 20K
    # labels from a teacher that is itself only 2.9x better than chance, fine-tuning
    # would fit the teacher's mistakes as readily as its judgement.
    d = X.shape[1]
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05, weight_decay=1e-3)
    n_val = max(1, len(X) // 10)
    Xtr, ytr, Xva, yva = X[n_val:], y[n_val:], X[:n_val], y[:n_val]
    for ep in range(a.epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, ytr)
        loss.backward()
        opt.step()
        if (ep + 1) % 10 == 0:
            with torch.no_grad():
                va = auc(yva.numpy(), (Xva @ w + b).numpy())
            print(f"  epoch {ep + 1} loss {loss.item():.4f} held-out-teacher AUC {va:.3f}", flush=True)

    torch.save({"w": w.detach(), "b": b.detach()}, a.out)
    print(f"saved {a.out}")

    if a.check:
        # The teacher's own held-out AUC only says the head copied the teacher. This
        # says whether the pair agrees with a human, which is the question.
        with open(a.check, encoding="utf-8") as fh:
            hand = [json.loads(x) for x in fh if x.strip()]
        hf = _features(model, _encode([r["t"] for r in hand], tok), a.device)
        with torch.no_grad():
            hs = (hf @ w + b).numpy()
        hy = [r["y"] for r in hand]
        print(f"  against {len(hand)} HAND labels: AUC {auc(hy, hs):.3f}   (27B itself reached 0.739)")
        import numpy as np

        hy, hs = np.array(hy, float), np.array(hs)
        for keep in (0.2, 0.3, 0.4, 0.5):
            t = np.quantile(hs, 1 - keep)
            k = hs >= t
            print(f"  keep top {keep:.0%}: {hy[k].mean():.1%} hand-labelled keep (base {hy.mean():.1%})")


if __name__ == "__main__":
    main()

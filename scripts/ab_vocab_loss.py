#!/usr/bin/env python
"""A/B sanity for the vocab 32773 -> 32776 alignment change: same seed, same data,
per-step loss. The 32776 arm adds 3 alignment-padding columns to the FLCE softmax;
they are never targets, so the expected result is a small step-1 delta that shrinks
as their logits train down. A diverging or growing delta means the padding leaks.

Also prints the effective dynamo cache_size_limit next to the AttnRes source count,
the guard train.py asserts at startup.

Usage: E2E_GPU=0 AB_STEPS=10 python scripts/ab_vocab_loss.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # repo root, for `import train`
sys.path.insert(0, HERE)
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import train  # noqa: E402
from train import SOFTCAP, Cfg, HybridLM, build_mix, build_optimizers, doc_cu_seqlens  # noqa: E402

STEPS = int(os.environ.get("AB_STEPS", "10"))
LR_SCALE = float(os.environ.get("AB_LR_SCALE", "1.0"))
DEVICE = f"cuda:{os.environ.get('E2E_GPU', '0')}"


def run_arm(vocab, Xtr, Ytr, eos_id):
    Cfg.vocab = vocab
    torch.manual_seed(Cfg.seed)
    model = HybridLM(Cfg).to(DEVICE)
    opts = build_optimizers(model, Cfg)
    if LR_SCALE != 1.0:  # base LRs assume ~1.77M tokens/step; a parity run at batch 4 needs them cut
        for opt in opts:
            for g in opt.param_groups:
                g["lr"] *= LR_SCALE
    model.train()
    losses = []
    for step in range(STEPS):
        idx = torch.arange(step * Cfg.batch, (step + 1) * Cfg.batch)
        xb, yb = Xtr[idx].to(DEVICE), Ytr[idx].to(DEVICE)
        cu = doc_cu_seqlens(xb, eos_id) if Cfg.doc_mask else None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden, _ = model(xb, yb, cu, None)
        B, T, D = hidden.shape
        weight = model.head.weight[:vocab]
        loss = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, softcap=SOFTCAP)(
            weight, hidden.to(weight.dtype).reshape(-1, D), yb.reshape(-1)
        )
        for opt in opts:
            opt.zero_grad()
        loss.backward()
        for opt in opts:
            opt.step()
        losses.append(loss.item())
    del model
    torch.cuda.empty_cache()
    return losses


def main():
    Cfg.mix = "data/mix_sample.json"
    Cfg.batch = 4
    Cfg.seq = 512
    Cfg.compile = False  # the parity question is the FLCE GEMM, not compile
    tok = Tokenizer.from_file(os.path.join(train.ROOT, "data/tokenizer.json"))
    tr, _ = build_mix(os.path.join(train.ROOT, Cfg.mix), tok, True, False, 0, 1)
    seqs = (tr[0] if isinstance(tr, tuple) else tr).long()
    Xtr, Ytr = seqs[:, :-1].contiguous(), seqs[:, 1:].contiguous()
    eos_id = tok.token_to_id("<eos>")
    assert len(Xtr) >= STEPS * Cfg.batch, f"sample mix has {len(Xtr)} seqs, need {STEPS * Cfg.batch}"

    results = {v: run_arm(v, Xtr, Ytr, eos_id) for v in (32773, 32776)}
    print(f"{'step':>4} {'v=32773':>10} {'v=32776':>10} {'delta':>10}")
    for i in range(STEPS):
        a, b = results[32773][i], results[32776][i]
        print(f"{i:>4} {a:>10.4f} {b:>10.4f} {b - a:>+10.4f}")

    import torch._dynamo as _dynamo

    need = 1 + 2 * Cfg.layers
    # train.py's compile path sets cache_size_limit=64 and asserts >= need before compiling;
    # this script runs eager, so it prints the values the guard enforces in a real run.
    print(f"dynamo guard (real runs): cache_size_limit=64 (AttnRes Full needs {need}); "
          f"this script runs eager (limit here: {_dynamo.config.cache_size_limit})")


if __name__ == "__main__":
    main()

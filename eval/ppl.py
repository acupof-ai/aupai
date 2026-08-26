#!/usr/bin/env python3
"""PPL evaluation: computes perplexity on held-out validation set."""

import math
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import HybridLM

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/work/aupai/ckpt.pt.step4000"
TOKEN_CACHE = "/data00/pretrain_1b_tokens.pt"
VAL_FRAC = 0.05
BATCH = 8
SEQ = 4096

device = "cuda:0"
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = SimpleNamespace(**ck["cfg"])
cfg.grad_ckpt = False
model = HybridLM(cfg).to(device)
model.load_state_dict(ck["model"])
for p in model.parameters():
    p.data = p.data.contiguous()
model.eval()

# Load validation data (same split as training: first 5%)
data = torch.load(TOKEN_CACHE, map_location="cpu", weights_only=True).long()
n_seq = len(data) // (SEQ + 1)
data = data[: n_seq * (SEQ + 1)]
seqs = data.view(-1, SEQ + 1)
X, Y = seqs[:, :-1], seqs[:, 1:]
n_val = max(1, int(len(X) * VAL_FRAC))
Xva, Yva = X[:n_val], Y[:n_val]
print(f"val seqs: {len(Xva):,} ({len(Xva) * SEQ:,} tokens)")

total_loss = 0.0
total_tokens = 0
n_batches = 0

with torch.no_grad():
    for i in range(0, len(Xva) - BATCH + 1, BATCH):
        xb = Xva[i : i + BATCH].to(device)
        yb = Yva[i : i + BATCH].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(xb)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), yb.reshape(-1), ignore_index=-100
            )
        ntok = (yb != -100).sum().item()
        total_loss += loss.item() * ntok
        total_tokens += ntok
        n_batches += 1
        if n_batches % 20 == 0:
            avg = total_loss / total_tokens
            print(f"batch {n_batches}: running ppl = {math.exp(avg):.2f}", flush=True)

avg_loss = total_loss / total_tokens
ppl = math.exp(avg_loss)
print("\n=== PPL Results ===")
print(f"checkpoint: {CKPT}")
print(f"val tokens: {total_tokens:,}")
print(f"avg loss: {avg_loss:.4f}")
print(f"perplexity: {ppl:.2f}")

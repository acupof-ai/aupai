#!/usr/bin/env python3
"""Remote inference: takes token IDs as CLI arg, writes output token IDs to stdout."""
import json
import sys

import torch
from types import SimpleNamespace
from train import HybridLM

ck = torch.load("/work/aupai/ckpt.pt.step2000", map_location="cuda:0", weights_only=False)
cfg = SimpleNamespace(**ck["cfg"])
cfg.grad_ckpt = False
model = HybridLM(cfg).cuda(0)
model.load_state_dict(ck["model"])
for p in model.parameters():
    p.data = p.data.contiguous()
model.eval()

ids = json.loads(open(sys.argv[1]).read())
x = torch.tensor([ids], device="cuda:0")
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for _ in range(256):
        logits = model(x[:, -cfg.seq:])[0][:, -1]
        nxt = logits.argmax(-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
        if nxt.item() == 1:
            break
print(json.dumps(x[0].tolist()))

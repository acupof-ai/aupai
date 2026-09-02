#!/usr/bin/env python3
"""Muon/AdamW optimizer phase timing only (CUDA events, 5 iters)."""
import json, os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
import train as T

VOCAB, B, S, DEV, EOS = T.Cfg.vocab, T.Cfg.batch, T.Cfg.seq, "cuda", 1

def make_batch():
    x = torch.randint(10, VOCAB, (B, S), device=DEV)
    for _ in range(8):
        x[torch.randint(0, B, (1,)), torch.randint(0, S, (1,))] = EOS
    y = torch.cat([x[:, 1:], torch.full((B, 1), EOS, device=DEV)], dim=1).contiguous()
    return x.contiguous(), y, T.doc_cu_seqlens(x, EOS)

raw = T.HybridLM(T.Cfg).to(DEV).to(torch.bfloat16)
T.convert_to_fp8_compute(raw)
opts = T.build_optimizers(raw, T.Cfg)
torch._dynamo.config.cache_size_limit = 64
m = torch.compile(raw, dynamic=False)
batch = make_batch()

def fwd_bwd():
    x, y, cu = batch
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden, _ = m(x, y, cu, None)
    w = raw.head.weight[:VOCAB]
    loss = T.LigerFusedLinearCrossEntropyLoss(ignore_index=-100, softcap=T.SOFTCAP)(
        w, hidden.to(w.dtype).reshape(-1, hidden.shape[-1]), y.reshape(-1))
    loss.backward()

def opt_step():
    torch.nn.utils.clip_grad_norm_(raw.parameters(), T.Cfg.clip)
    for o in opts: o.step()
    for o in opts: o.zero_grad(set_to_none=True)

# warmup (includes compile of both model and muon)
for _ in range(15):
    fwd_bwd(); opt_step()
torch.cuda.synchronize()

# time opt phase only: per-iter events around opt_step (no per-iter sync)
pairs = []
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(10):
    fwd_bwd()
    s2, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s2.record(); opt_step(); e2.record()
    pairs.append((s2, e2))
e.record(); torch.cuda.synchronize()
opt = sum(s2.elapsed_time(e2) for s2, e2 in pairs) / len(pairs)
full = s.elapsed_time(e) / 10

print(json.dumps({"full_ms": full, "opt_ms": opt, "fwd_bwd_ms": full - opt,
                  "tok_per_s": B*S/(full/1000)}))

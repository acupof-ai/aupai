"""fp8 LM head activation range: per-CHUNK absmax distribution pre-softcap (t57 gate).

The weight side is benign (max/p50 1.71x, eff.lm_head_is_compute_bound). The activation side
is the gating unknown: Liger FLCE splits BT into 64 chunks and an fp8 per-tensor scale is set
per chunk, so ONE outlier logit fixes the scale for 32784 columns. b0's gate: report the
per-chunk absmax DISTRIBUTION, and p99 against e4m3's 448 decides per-tensor vs per-chunk.

Runs the real head projection on real batches from the live run's own checkpoint and mix.
"""
import glob
import json
import os
import sys

import torch

sys.argv = ["absmax"]
sys.path.insert(0, "/work/aupai/scripts")
sys.path.insert(0, "/work/aupai")
import train  # noqa: E402

E4M3_MAX = 448.0
CHUNK = 2048  # Liger's chunk_size at BT=16*4096, H=1024, V=32784

ck = sorted(glob.glob("/work/aupai/ckpt_pretrain_15b_s1.pt.step*"),
            key=lambda p: int(p.rsplit("step", 1)[1]))[-1]
c = torch.load(ck, map_location="cpu", weights_only=False)
sd = c.get("model") or c.get("state_dict")
W = sd["head.weight"].cuda()                      # (padded_vocab, H)
print(f"ckpt {os.path.basename(ck)}  head {tuple(W.shape)} {W.dtype}", flush=True)

# Real hidden states: run the model body on real rows from the stage-1 mix.
train.Cfg.vocab, train.Cfg.fone = 32784, False
tok = train.build_tokenizer(None)
X = train._domain_seqs("cot", tok, is_main=True, ddp=False)   # smallest warm domain
print(f"cot rows {tuple(X.shape)}", flush=True)

model = train.HybridLM(train.Cfg).cuda().to(torch.bfloat16).eval()
model.load_state_dict({k: v for k, v in sd.items()}, strict=False)

rows = []
g = torch.Generator().manual_seed(0)
for it in range(4):                                # 4 batches of 16 rows
    idx = torch.randperm(len(X), generator=g)[:16]
    xb = X[idx, :-1].cuda().to(torch.int32)
    cu = train.doc_cu_seqlens(xb, train.Cfg.eos_id if hasattr(train.Cfg, "eos_id") else 1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h, _ = model(xb, xb, cu, None)
    h2 = h.reshape(-1, h.shape[-1])                # BT x H
    for s in range(0, h2.shape[0], CHUNK):
        hc = h2[s:s + CHUNK]
        if hc.shape[0] < 16:
            continue
        logits = (hc.float() @ W.float().t())      # PRE-softcap, as fp8 would see it
        rows.append(logits.abs().max().item())
    del h, h2
    print(f"  batch {it}: {len(rows)} chunks so far", flush=True)

t = torch.tensor(rows)
q = lambda p: t.quantile(torch.tensor(p)).item()
print(f"\nper-chunk pre-softcap absmax over {len(t)} chunks:")
print(f"  min {t.min():.2f}  p50 {q(0.50):.2f}  p90 {q(0.90):.2f}  p99 {q(0.99):.2f}  max {t.max():.2f}")
print(f"  e4m3 max {E4M3_MAX}")
print(f"  p99/p50 spread {q(0.99)/q(0.50):.2f}x   max/p50 {t.max()/q(0.50):.2f}x")
clip = (t > E4M3_MAX).float().mean().item() * 100
print(f"  chunks whose absmax EXCEEDS 448 unscaled: {clip:.1f}%")
print(f"  VERDICT: per-tensor scaling {'SUFFICIENT' if q(0.99)/q(0.50) < 4 else 'NOT sufficient -> per-chunk or amax-history'}")
json.dump({"n_chunks": len(t), "p50": q(0.50), "p90": q(0.90), "p99": q(0.99),
           "max": t.max().item(), "p99_over_p50": q(0.99)/q(0.50),
           "ckpt": os.path.basename(ck)}, open("/work/aupai/runs/t57_absmax.json", "w"), indent=1)
print("wrote runs/t57_absmax.json")

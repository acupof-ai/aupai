#!/usr/bin/env python3
"""Arm B's first real run: one forward+backward of the head_mixed model on a card.

WHY A SEPARATE SCRIPT AND NOT A TRAINING STEP. Nothing has ever executed DeltaRecurrence at
inner < cfg.d -- chunk_kda is a Triton kernel, absent on CPU, so the local shape check could only
verify GatedMLA's halves. The KDA half at inner=768 h=6 reaches the kernel with a q/k/v width
that no run in this repo has produced. If that combination is wrong, it is wrong at step 0 of a
7-card launch, and the launch is the expensive way to find out.

WHAT IT ASSERTS, and each one exists because its absence has burned this repo:
  loss finite         -- the ordinary check
  EVERY grad finite   -- not "the loss looked fine": forward-correct/backward-wrong is measured
                         here (flash_attn mask_mod gradients were 21x off at SM 9.0 while the
                         forward matched, docs/lessons/forward_check_hides_gradient_error.md)
  every param HAS a grad -- a mixer whose output never reaches the loss trains silently at zero.
                         With two mixers summed, one contributing nothing is exactly the failure
                         mode that looks like a working model.
  both halves' grads NONZERO -- stronger than "not None". A grad of exactly 0.0 everywhere means
                         that half is decorative, and the sum makes it invisible.
  tok/s               -- reported, not asserted: arm B changes the parameter count (+1.18%), so a
                         throughput number is needed before the launch, not a pass/fail.

Run: CUDA_VISIBLE_DEVICES=<granted card> python3 runs/audit_0904/b0_headmix_first_run.py
"""
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

B, T = 2, 4096

if not torch.cuda.is_available():
    raise SystemExit("REFUSING: no CUDA device. The KDA half is a Triton kernel; a CPU run "
                     "would prove nothing about the path the launch takes.")

import copy  # noqa: E402

import model as M  # noqa: E402
from train import Cfg  # noqa: E402

cfg = copy.copy(Cfg)
cfg.head_mixed = 3
m = M.HybridLM(cfg).cuda()
mx = m.blocks[0].mixer
if type(mx).__name__ != "HeadMix":
    raise SystemExit(f"REFUSING: block 0 holds {type(mx).__name__}, not HeadMix -- head_mixed=3 "
                     f"did not take effect and this would have measured arm A.")
n_par = sum(p.numel() for p in m.parameters())
print(f"head_mixed=3 at d{cfg.d} L{cfg.layers} h{cfg.heads} ffn{cfg.ffn_hidden}: {n_par:,} params")
print(f"  block 0: kda h={mx.kda.h} inner={mx.kda.d} hd={mx.kda.hd} | "
      f"mla h={mx.mla.h} inner={mx.mla.d} hd={mx.mla.hd} latent={mx.mla.latent}")
print(f"  all {len(m.blocks)} blocks HeadMix: "
      f"{all(type(b.mixer).__name__ == 'HeadMix' for b in m.blocks)}")

torch.manual_seed(904)
x = torch.randint(0, int(cfg.vocab_real), (B, T), device="cuda")
y = torch.randint(0, int(cfg.vocab_real), (B, T), device="cuda")

torch.cuda.synchronize()
t0 = time.time()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out = m(x)
    if isinstance(out, tuple):
        out = out[0]
    loss = torch.nn.functional.cross_entropy(
        out[..., : cfg.vocab_real].float().reshape(-1, cfg.vocab_real), y.reshape(-1))
loss.backward()
torch.cuda.synchronize()
dt = time.time() - t0

print(f"\nloss {loss.item():.6f}  finite {torch.isfinite(loss).item()}")
if not torch.isfinite(loss):
    raise SystemExit("REFUSING: loss is not finite on the first forward.")

missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
nonfinite = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
print(f"params {sum(1 for _ in m.parameters())}  no grad {len(missing)}  non-finite grad {len(nonfinite)}")
if missing:
    raise SystemExit(f"REFUSING: {len(missing)} parameter(s) got no gradient, first 5: {missing[:5]} "
                     f"-- a mixer whose output never reaches the loss trains at zero silently.")
if nonfinite:
    raise SystemExit(f"REFUSING: {len(nonfinite)} parameter(s) have non-finite gradients, first 5: "
                     f"{nonfinite[:5]}")

# BOTH HALVES MUST ACTUALLY LEARN. Summing two mixers hides a dead one: the output shape is
# right, the loss is finite, every grad exists, and one half is decorative.
for half in ("kda", "mla"):
    gs = [(n, p.grad.abs().max().item()) for n, p in m.blocks[0].mixer.named_parameters()
          if n.startswith(half + ".") and p.grad is not None]
    mx_g = max(g for _, g in gs)
    print(f"  block0.{half}: {len(gs)} tensors, max|grad| {mx_g:.6e}")
    if mx_g == 0.0:
        raise SystemExit(f"REFUSING: every gradient in block0.{half} is exactly 0.0 -- that half "
                         f"contributes nothing and the sum makes it invisible.")

tok = B * T
print(f"\nfwd+bwd {dt:.3f}s for {tok:,} tokens -> {tok / dt:,.0f} tok/s "
      f"(one step, one card, no DDP -- a floor, not the launch number)")
print(f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
print("\nOK: head_mixed forward+backward runs, loss and all grads finite, both halves learn")

#!/usr/bin/env python3
"""Does grad_ckpt change the AttnRes source count dynamo specialises on? (de, 2026-09-01)

fb's challenge to the cache-limit fix, and it is the right one: `max(64, 2*layers+8)`
is sized against "AttnRes Full builds 1 + 2*layers distinct graphs". If gradient
checkpointing changed what dynamo sees per step, the constant would be sized against
the wrong quantity -- the same defect one level up from the one it fixes.

Reading train.py:772-789 says it should not: the checkpoint wraps `fn = f(norm(t))`,
the sublayer function, and AttnRes is deliberately OUTSIDE it ("only [B,T] logits on
the tape, never [B,T,D]"). So the `ar(done + partial)` call and its source list are
identical either way. But reading is what produced the 12-layer constant in the first
place, so this measures it: AttnRes.forward is instrumented to record how many
sources each call receives, with checkpointing off and on, and the two sequences are
compared element by element.

    python3 scripts/probe_gradckpt_sources.py
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "datagen"))

import train  # noqa: E402

if train.chunk_kda is None:
    train.chunk_kda = lambda q, k, v, **kw: (q * 0 + v, None)

from train import Cfg, HybridLM  # noqa: E402

L = 32
Cfg.d, Cfg.heads, Cfg.layers, Cfg.ffn_hidden = 1024, 8, L, 3072
Cfg.vocab = Cfg.vocab_real = 256
Cfg.seq, Cfg.fone = 8, False
Cfg.attn_res, Cfg.attn_res_blocks, Cfg.attn_res_dyn_q = True, 0, False

_orig = train.AttnRes.forward


def source_counts(grad_ckpt):
    """The length of the source list at every AttnRes call, in order.

    That length is what dynamo specialises a graph on -- a different count is a
    different graph -- so the number of DISTINCT values is the cache pressure.
    """
    seen = []

    def patched(self, srcs, *a, **kw):
        seen.append(len(srcs))
        return _orig(self, srcs, *a, **kw)

    train.AttnRes.forward = patched
    try:
        Cfg.grad_ckpt = grad_ckpt
        m = HybridLM(Cfg)
        m.train()
        x = torch.randint(0, Cfg.vocab, (1, Cfg.seq))
        h, _ = m(x, x)
        h.sum().backward()
    finally:
        train.AttnRes.forward = _orig
    return seen


off = source_counts(False)
on = source_counts(True)

print(f"grad_ckpt OFF: {len(off)} AttnRes calls, {len(set(off))} distinct source counts, max {max(off)}")
print(f"grad_ckpt ON : {len(on)} AttnRes calls, {len(set(on))} distinct source counts, max {max(on)}")

bad = []
if off != on:
    bad.append("the source-count SEQUENCE differs between grad_ckpt off and on -- "
               "cache_size_limit is sized against the wrong quantity")
need = 1 + 2 * L
if len(set(off)) != need:
    bad.append(f"{len(set(off))} distinct counts at L={L}, formula says 1+2*layers={need}")
if max(off) != need:
    bad.append(f"max source count {max(off)}, formula says {need}")

if bad:
    print()
    for b in bad:
        print(f"FAIL: {b}")
    sys.exit(1)
print(f"\nOK: grad_ckpt does not change the count. {need} distinct source counts at "
      f"L={L}, identical with and without checkpointing -- AttnRes sits outside the "
      f"checkpoint (train.py:784), so recompute never re-enters it. max(64, 2*layers+8) "
      f"= {max(64, 2 * L + 8)} covers {need}.")

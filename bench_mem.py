#!/usr/bin/env python3
"""Memory attribution per module + AttnRes A/B. Single GPU, eager (no compile).

Compile doesn't change logical activation size, so eager is fine and hooks
fire reliably. Measures fwd activation (tensors saved for backward) via
forward-hook net allocation deltas. Run on GPU0 after tilerl's bench frees it.
"""
import json, os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
import train as T

VOCAB, DEV, EOS = T.Cfg.vocab, "cuda", 1
T.Cfg.batch = 16
T.Cfg.seq = 4096
B, S = T.Cfg.batch, T.Cfg.seq

def make_batch():
    x = torch.randint(10, VOCAB, (B, S), device=DEV)
    for _ in range(8):
        x[torch.randint(0, B, (1,)), torch.randint(0, S, (1,))] = EOS
    y = torch.cat([x[:, 1:], torch.full((B, 1), EOS, device=DEV)], dim=1).contiguous()
    return x.contiguous(), y, T.doc_cu_seqlens(x, EOS)

def measure(attn_res_on):
    T.Cfg.attn_res = attn_res_on
    torch.manual_seed(0)
    raw = T.HybridLM(T.Cfg).to(DEV).to(torch.bfloat16)
    T.convert_to_fp8_compute(raw)
    x, y, cu = make_batch()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        h, _ = raw(x, y, cu, None)  # warmup
    del h; torch.cuda.empty_cache()

    mem = {}
    def mk(name):
        st = {"pre": 0}
        def pre(mod, inp): st["pre"] = torch.cuda.memory_allocated()
        def post(mod, inp, out):
            mem[name] = mem.get(name, 0.0) + float(torch.cuda.memory_allocated() - st["pre"])
        return pre, post

    for i, b in enumerate(raw.blocks):
        for n, m in (("mixer", b.mixer), ("ffn", b.ffn), ("ar1", b.ar1), ("ar2", b.ar2)):
            if m is not None:
                p, q = mk(f"block{i}.{n}")
                m.register_forward_pre_hook(p); m.register_forward_hook(q)
    if raw.final_ar is not None:
        p, q = mk("final_ar")
        raw.final_ar.register_forward_pre_hook(p); raw.final_ar.register_forward_hook(q)
    p, q = mk("tok"); raw.tok.register_forward_pre_hook(p); raw.tok.register_forward_hook(q)

    base = torch.cuda.memory_allocated()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        h, _ = raw(x, y, cu, None)
    fwd_act = float(torch.cuda.memory_allocated() - base)
    del h

    s = lambda pred: sum(v for k, v in mem.items() if pred(k)) / 1e9
    return {
        "attn_res": attn_res_on,
        "fwd_activation_GB": fwd_act / 1e9,
        "mixer_GB": s(lambda k: ".mixer" in k),
        "ffn_swiglu_GB": s(lambda k: ".ffn" in k),
        "attnres_GB": s(lambda k: ".ar1" in k or ".ar2" in k) + mem.get("final_ar", 0) / 1e9,
        "tok_embed_GB": mem.get("tok", 0) / 1e9,
    }

on = measure(True)
off = measure(False)
on["accounted_GB"] = on["mixer_GB"] + on["ffn_swiglu_GB"] + on["attnres_GB"] + on["tok_embed_GB"]
on["residual_GB"] = on["fwd_activation_GB"] - on["accounted_GB"]
print(json.dumps({"on": on, "off": off,
                  "attnres_delta_GB": on["fwd_activation_GB"] - off["fwd_activation_GB"]}, indent=2))

"""Single-GPU NaN probe: find the FIRST non-finite tensor when grad_ckpt is off.

Runs a few real SFT steps at tiny batch (fits alongside the main run) with
forward/backward hooks that report the first module whose output or grad
goes non-finite, plus the FP8 per-tensor scales.
"""
import os, sys, math
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/work/aupai")
import torch, torch.nn as nn
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from train import Cfg, HybridLM, Muon, convert_to_fp8_compute, FP8Linear

GC = os.environ.get("GC", "0") == "1"
USE_MUON = os.environ.get("MUON", "0") == "1"
STEPS = int(os.environ.get("STEPS", "40"))
dev = "cuda:0"
d = torch.load("/work/aupai/data/sft/sft_v3.pt", map_location="cpu", weights_only=True)
X = d["input_ids"][:4096, :-1].long(); Y = d["labels"][:4096, 1:].long(); del d
ck = torch.load("/work/aupai/ckpt_k3-mla_2b_step2000.pt", map_location="cpu", weights_only=False)
for k, v in ck.get("cfg", {}).items():  # build from the ckpt's Cfg, not the live one
    if not k.startswith("_") and hasattr(Cfg, k):
        setattr(Cfg, k, v)
Cfg.batch = int(os.environ.get("BS", "2"))
Cfg.grad_ckpt = GC
Cfg.compile = os.environ.get("COMPILE", "0") == "1"
m = HybridLM(Cfg).to(dev)
m.load_state_dict(ck["model"]); del ck
m = m.to(torch.bfloat16); convert_to_fp8_compute(m); m.train()
if Cfg.compile:
    torch._dynamo.config.cache_size_limit = 64
    m_run = torch.compile(m, dynamic=False)
else:
    m_run = m

first = {}
def fwd_hook(name):
    def h(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(o) and not torch.isfinite(o).all() and "fwd" not in first:
            first["fwd"] = (name, float(o.abs().max()))
    return h
for n_, mod in m.named_modules():
    mod.register_forward_hook(fwd_hook(n_))

# instrument FP8 scale magnitudes
scales = []


def amax():
    """Largest FP8 input magnitude seen; -1 when the active recipe is not the hand-rolled one."""
    return max(scales) if scales else -1.0


def fp8_kind():
    """Which FP8 path the model actually ended up on."""
    kinds = {type(mod).__name__ for mod in m.modules() if "Float8" in type(mod).__name__ or "FP8" in type(mod).__name__}
    return "+".join(sorted(kinds)) or "none"
orig = FP8Linear.forward
def patched(self, x):
    s = float(x.detach().abs().max())
    scales.append(s)
    if not math.isfinite(s) and "scale" not in first:
        first["scale"] = ("input to FP8Linear non-finite", s)
    return orig(self, x)
FP8Linear.forward = patched

if USE_MUON:
    mp = [p for n_, p in m.named_parameters() if p.ndim == 2 and "tok" not in n_ and "head" not in n_]
    rest = [p for n_, p in m.named_parameters() if not (p.ndim == 2 and "tok" not in n_ and "head" not in n_)]
    opt_list = [Muon(mp, lr=Cfg.muon_lr * 0.1, momentum=Cfg.muon_momentum, ns_steps=Cfg.muon_ns_steps,
                     weight_decay=Cfg.muon_wd),
                torch.optim.AdamW(rest, lr=1e-4, fused=True)]
else:
    opt_list = [torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)]
flce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
w = m.head.weight[: m.cfg.vocab]
for step in range(1, STEPS + 1):
    i = (step * Cfg.batch) % (len(X) - Cfg.batch)
    xb, yb = X[i:i+Cfg.batch].to(dev), Y[i:i+Cfg.batch].to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden, _ = m_run(xb, yb)
    B, T, D = hidden.shape
    loss = flce(w, hidden.to(w.dtype).reshape(-1, D), yb.reshape(-1))
    loss.backward()
    gn = float(nn.utils.clip_grad_norm_(m.parameters(), 1.0))
    bad = [n_ for n_, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    if not math.isfinite(loss.item()) or not math.isfinite(gn) or bad:
        print(f"STEP {step} grad_ckpt={GC} loss={loss.item()} gnorm={gn}")
        print("  first non-finite fwd:", first.get("fwd"))
        print("  first bad grads:", bad[:5], f"({len(bad)} params)")
        print(f"  max FP8 input amax seen: {amax():.4g}")
        break
    for o in opt_list:
        o.step(); o.zero_grad(set_to_none=True)
    if step % 10 == 0:
        print(f"  step {step} loss {loss.item():.3f} gnorm {gn:.3f} max_amax {amax():.4g}", flush=True)
else:
    print(f"grad_ckpt={GC} bs={Cfg.batch} compile={Cfg.compile} muon={USE_MUON} fp8={fp8_kind()}: "
          f"{STEPS} steps clean, max FP8 input amax {amax():.4g}")

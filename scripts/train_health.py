#!/usr/bin/env python3
"""In-training health: every component, every layer, one row per window, alarms by rule.

Called by train.py around opt.step() every Cfg.health_every steps. One row per window goes to
runs/train_health.jsonl and every broken rule prints a `HEALTH ALARM` line in the run log.
It watches, it never stops a run.

  moe      per MoE layer: renormalized top-1 gate share, zero-load experts, load max/mean,
           load gini, expert_bias range, router-logit L2 norm
  opt      per optimizer param group: the lr actually applied, fraction of sampled elements
           the step moved, rms(update)/rms(weight)
  layer    per block: grad norm, weight rms, moved fraction, update ratio
  lens     every Cfg.health_lens_every steps, one val batch: the loss of each block's output
           read through the final norm and head (logit lens)

Why it exists: v41_ced_0923 ran 38K steps with the router collapsed from step 4000 on, and the
one MoE readout that ran (moe_diag, layer 0, load only) read 94% of experts used. Load spread
cannot see a one-hot gate, and nothing read the rows anyway.

    python3 scripts/train_health.py --selftest
"""
# restartable: called in-process by train.py; each row is one appended line, so an interrupt
# loses at most the current window's row and the run itself resumes from its own checkpoint.

import json
import math
import os
import re
import sys

import torch

SAMPLE = 4096  # elements per parameter tensor probed for movement

# Alarm rules: (name, predicate on the row, message). Thresholds are set from the two known
# answers in the selftest and from v41_ced_0923: a uniform router over top-3 gives top-1 ~1/3,
# the 0923 collapse gave ~0.996; a frozen bf16 group moves 0%, mid-run 0923 moved ~67%.
TOP1_MAX = 0.90
ZERO_LOAD_MAX = 0.10  # fraction of routed experts with no token in the window
BIAS_RANGE_MAX = 0.5
ARGMAX_SEL_MIN = 0.5  # rows whose router-preferred expert is actually selected
MOVED_MIN = 0.05  # per group with lr > 0
LAYER_Z_MAX = 6.0  # robust z of a block's grad norm against the other blocks


def _robust_z(xs):
    s = sorted(xs)
    n = len(s)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    d = sorted(abs(x - med) for x in xs)
    mad = d[n // 2] if n % 2 else 0.5 * (d[n // 2 - 1] + d[n // 2])
    sig = 1.4826 * mad
    return [0.0 if sig == 0 else (x - med) / sig for x in xs]


def _gini(c):
    s = c.sort().values
    tot = float(s.sum())
    if tot <= 0:
        return 0.0
    k = s.numel()
    i = torch.arange(1, k + 1, dtype=torch.float32)
    return min(1.0, max(0.0, float(2.0 * (i * s).sum() / (k * tot) - (k + 1) / k)))


def moe_stats(load, sums, bias):
    """One layer's window -> dict. load [E] tokens per expert, sums (top1, lnorm, rows, argmax_sel)."""
    load = load.float().cpu()
    top1, lnorm, rows, amax = (float(v) for v in sums.cpu())
    tot = float(load.sum())
    E = load.numel()
    b = bias.float().cpu()
    return {
        "rows": int(rows),
        "top1": top1 / rows if rows else float("nan"),
        "logit_norm": lnorm / rows if rows else float("nan"),
        "argmax_selected": amax / rows if rows else float("nan"),
        "zero_load": int((load == 0).sum()),
        "zero_load_frac": float((load == 0).sum()) / E,
        "load_max_over_mean": float(load.max()) * E / tot if tot else float("nan"),
        "load_gini": _gini(load),
        "bias_min": float(b.min()),
        "bias_max": float(b.max()),
        "bias_range": float(b.max() - b.min()),
    }


def collect_moe(raw_model, ddp):
    """All MoE layers in one all_reduce, then reset. EVERY rank must call it."""
    mods = [(i, blk.ffn) for i, blk in enumerate(raw_model.blocks) if hasattr(getattr(blk, "ffn", None), "h_sums")]
    if not mods:
        return {}
    buf = torch.stack([torch.cat([m.h_load, m.h_sums]) for _, m in mods])
    if ddp:
        torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
    out = {}
    for (i, m), row in zip(mods, buf):
        E = m.h_load.numel()
        out[str(i)] = moe_stats(row[:E], row[E:], m.expert_bias)
        m.h_load.zero_()
        m.h_sums.zero_()
    return out


def _layer_of(name):
    m = re.match(r"(?:_orig_mod\.)?blocks\.(\d+)\.", name)
    return m.group(1) if m else name.split(".")[0]


def before_step(raw_model):
    """Grad norms and a strided sample of every parameter, taken just before opt.step()."""
    probe = []
    for name, p in raw_model.named_parameters():
        if p.grad is None or p.numel() == 0:
            continue
        flat = p.detach().reshape(-1)
        stride = max(1, flat.numel() // SAMPLE)
        probe.append((name, p, stride, flat[::stride][:SAMPLE].clone(), p.grad.detach().float().norm()))
    return probe


def after_step(probe, optimizers):
    """-> (per-layer dict, per-group dict) from the samples taken before the step."""
    group_of = {}
    for oi, opt in enumerate(optimizers):
        for gi, g in enumerate(opt.param_groups):
            for p in g["params"]:
                group_of[id(p)] = (f"{type(opt).__name__}{oi}.g{gi}", float(g["lr"]))
    layers, groups = {}, {}
    for name, p, stride, old, gnorm in probe:
        new = p.detach().reshape(-1)[::stride][:SAMPLE].float()
        old = old.float()
        moved = int((new != old).sum())
        d2 = float(((new - old) ** 2).sum())
        w2 = float((old**2).sum())
        gkey, lr = group_of.get(id(p), ("unowned", 0.0))
        for key, bucket, extra in ((_layer_of(name), layers, None), (gkey, groups, lr)):
            a = bucket.setdefault(key, {"n": 0, "moved": 0, "d2": 0.0, "w2": 0.0, "g2": 0.0})
            a["n"] += old.numel()
            a["moved"] += moved
            a["d2"] += d2
            a["w2"] += w2
            a["g2"] += float(gnorm) ** 2
            if extra is not None:
                a["lr"] = extra

    def fin(a):
        r = {
            "moved_frac": a["moved"] / a["n"] if a["n"] else float("nan"),
            "update_ratio": math.sqrt(a["d2"] / a["w2"]) if a["w2"] else float("nan"),
            "grad_norm": math.sqrt(a["g2"]),
            "weight_rms": math.sqrt(a["w2"] / a["n"]) if a["n"] else float("nan"),
        }
        if "lr" in a:
            r["lr"] = a["lr"]
        return r

    return {k: fin(v) for k, v in layers.items()}, {k: fin(v) for k, v in groups.items()}


@torch.no_grad()
def logit_lens(raw_model, x, y, cu, loss_fn, amp_dtype):
    """Loss of each block's output through the final norm and head, one batch, eval mode."""
    hs = []
    hooks = [b.register_forward_hook(lambda _m, _i, o: hs.append(o.detach())) for b in raw_model.blocks]
    was = raw_model.training
    raw_model.eval()
    try:
        with torch.autocast(device_type=x.device.type, dtype=amp_dtype, enabled=x.device.type == "cuda"):
            raw_model(x, y, cu)
            return [float(loss_fn(raw_model.norm(h), y)) for h in hs]
    finally:
        for h in hooks:
            h.remove()
        raw_model.train(was)


def alarms(row):
    out = []
    for li, s in row.get("moe", {}).items():
        if not s["top1"] <= TOP1_MAX:
            out.append(f"moe L{li} top1 {s['top1']:.3f} > {TOP1_MAX} (router one-hot)")
        if not s["argmax_selected"] >= ARGMAX_SEL_MIN:
            out.append(f"moe L{li} router's first choice selected in {100 * s['argmax_selected']:.0f}% of rows "
                       f"(bias overriding the router)")
        if s["zero_load_frac"] > ZERO_LOAD_MAX:
            out.append(f"moe L{li} {s['zero_load']} experts got no token")
        if s["bias_range"] > BIAS_RANGE_MAX:
            out.append(f"moe L{li} expert_bias range {s['bias_range']:.3f} > {BIAS_RANGE_MAX}")
    for g, s in row.get("opt", {}).items():
        if s.get("lr", 0) > 0 and not s["moved_frac"] >= MOVED_MIN:
            out.append(f"opt {g} moved {100 * s['moved_frac']:.1f}% at lr {s['lr']:.2e} (updates lost)")
    blocks = {k: v for k, v in row.get("layer", {}).items() if k.isdigit()}
    if len(blocks) >= 4:
        keys = sorted(blocks, key=int)
        for k, z in zip(keys, _robust_z([blocks[k]["grad_norm"] for k in keys])):
            if abs(z) > LAYER_Z_MAX:
                out.append(f"layer {k} grad norm {blocks[k]['grad_norm']:.3g} robust z {z:.1f}")
    for k, v in row.get("layer", {}).items():
        if not all(math.isfinite(x) for x in v.values()):
            out.append(f"layer {k} non-finite stat {v}")
    return out


def write(path, row, runlog=print):
    row["alarms"] = alarms(row)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    moe = row.get("moe", {})
    top1 = max((s["top1"] for s in moe.values()), default=float("nan"))
    zl = sum(s["zero_load"] for s in moe.values())
    mv = min((s["moved_frac"] for s in row.get("opt", {}).values()), default=float("nan"))
    runlog(f"step {row['step']} health | moe top1 max {top1:.3f} zero-load {zl} | "
           f"opt moved min {100 * mv:.1f}% | alarms {len(row['alarms'])}")
    for a in row["alarms"]:
        runlog(f"step {row['step']} HEALTH ALARM {a}")
    return row


def _selftest():
    torch.manual_seed(0)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model import MoEFFN

    cfg = type("C", (), dict(d=32, ffn_hidden=64, layers=2, vocab=100, seq=16, attn_every=4, moe_experts=8,
                             moe_top_k=3, moe_expert_ffn=16, moe_shared=1, moe_bias_gamma=0.001,
                             moe_balance_alpha=1e-4))()

    def window(scale, prep=None, x=None):
        m = MoEFFN(cfg)
        # the training path casts the whole model to bf16 and back; the window counters must
        # come out fp64 or the ratios saturate (v41_ced_fixprobe_0925, logit_norm 32.000)
        m.to(torch.bfloat16)
        assert m.h_sums.dtype == torch.float64 and m.h_load.dtype == torch.float64, (m.h_sums.dtype,)
        m.float()
        with torch.no_grad():
            m.router.weight.mul_(scale)
            if prep is not None:
                prep(m)
        m.train()
        m(torch.randn(4, 16, 32) if x is None else x)
        blk = type("B", (torch.nn.Module,), {})()
        blk.ffn = m
        model = type("M", (), {"blocks": [blk]})()
        return collect_moe(model, ddp=False)["0"], m

    # 1. Known answers through the REAL MoEFFN forward: a small router is near-uniform over
    # top-3, a router scaled 1e4 is one-hot. The healthy world must pass, the collapsed one fail.
    ok, m_ok = window(1.0)
    bad, _ = window(1e4)
    assert ok["rows"] == 64 and bad["rows"] == 64, (ok["rows"], bad["rows"])
    assert ok["top1"] < 0.6, ok
    assert bad["top1"] > 0.99, bad
    assert not alarms({"moe": {"0": ok}}), alarms({"moe": {"0": ok}})
    assert any("one-hot" in a for a in alarms({"moe": {"0": bad}}))
    # the window resets: a second read with no forward in between reports no rows
    assert int(m_ok.h_sums[2]) == 0 and float(m_ok.h_load.sum()) == 0

    # 1b. The bias overriding the router, through the real forward: on an all-ones input expert 0
    # leads every row by a modest margin (gate share ~0.5, not one-hot). Without a bias its first
    # choice is always selected; a -50 bias on it keeps the gate spread and the load flat while
    # the router's preference is never served -- only argmax_selected sees that.
    def prefer0(m):
        m.router.weight.zero_()
        m.router.weight[0].fill_(0.02)

    ones = torch.ones(4, 16, 32)
    pref, _ = window(1.0, prefer0, ones)
    over, _ = window(1.0, lambda m: (prefer0(m), m.expert_bias.__setitem__(0, -50.0)), ones)
    # identical rows pick the same 3 experts, so the zero-load rule fires here by construction;
    # the property under test is only the override rule
    assert pref["argmax_selected"] == 1.0 and pref["top1"] < 0.9, pref
    assert not any("overriding" in a for a in alarms({"moe": {"0": pref}}))
    assert over["argmax_selected"] == 0.0 and over["top1"] < 0.9, over
    assert any("overriding" in a for a in alarms({"moe": {"0": over}})), alarms({"moe": {"0": over}})

    # 2. The 0923 shape: a bias that forces the load flat does not lower the un-biased top-1.
    load = torch.full((48,), 100.0)
    s = moe_stats(load, torch.tensor([0.996 * 4800, 4800.0 * 30, 4800.0, 0.0]), torch.linspace(-0.98, 0.02, 48))
    got = alarms({"moe": {"3": s}})
    assert s["zero_load"] == 0 and any("one-hot" in a for a in got) and any("bias range" in a for a in got), got

    # 3. Movement through a REAL optimizer step: bf16 weight, sub-half-ULP update. Round-to-
    # nearest moves nothing and must alarm; an fp32 weight under the same update must not.
    def moved(dtype):
        w = torch.nn.Parameter((1.0 + 0.5 * torch.rand(64, 64)).to(dtype))
        w.grad = torch.ones_like(w)
        mdl = torch.nn.Module()
        mdl.w = w
        opt = torch.optim.SGD([w], lr=1e-4)
        pr = before_step(mdl)
        opt.step()
        layer, group = after_step(pr, [opt])
        return {"step": 1, "opt": group, "layer": layer}

    frozen, live = moved(torch.bfloat16), moved(torch.float32)
    assert frozen["opt"]["SGD0.g0"]["moved_frac"] == 0.0, frozen
    assert live["opt"]["SGD0.g0"]["moved_frac"] > 0.99, live
    assert any("updates lost" in a for a in alarms(frozen)) and not alarms(live), (alarms(frozen), alarms(live))

    # 4. One block's grad norm far off the others alarms; a flat stack does not.
    flat = {str(i): {"grad_norm": 1.0 + 0.01 * i, "moved_frac": 0.5, "update_ratio": 1e-3, "weight_rms": 0.02}
            for i in range(12)}
    spike = json.loads(json.dumps(flat))
    spike["7"]["grad_norm"] = 50.0
    assert not alarms({"layer": flat}) and any("layer 7" in a for a in alarms({"layer": spike}))

    # 5. Logit lens: hooks every block, restores train mode, one loss per block.
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = torch.nn.Embedding(10, 8)
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(3)])
            self.norm = torch.nn.LayerNorm(8)
            self.head = torch.nn.Linear(8, 10)

        def forward(self, x, y, cu=None):
            h = self.tok(x)
            for b in self.blocks:
                h = b(h)
            return self.norm(h), None

    t = Tiny().train()
    x = torch.randint(0, 10, (2, 5))
    ls = logit_lens(t, x, x, None, lambda h, y: torch.nn.functional.cross_entropy(t.head(h).flatten(0, 1),
                                                                                     y.flatten()), torch.bfloat16)
    assert len(ls) == 3 and all(math.isfinite(v) for v in ls) and t.training
    print("train_health selftest OK: collapsed/uniform router, 0923 bias shape, frozen/live step, layer spike, lens")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        _selftest()
    else:
        raise SystemExit("usage: train_health.py --selftest (train.py calls the functions)")

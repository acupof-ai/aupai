#!/usr/bin/env python3
"""Router-logit saturation stats read OFFLINE from a checkpoint (de, 1e order 2026-09-26).

The runlog logit_norm and online train_health rows of the pre-fp64 builds are not trustworthy,
and the raw logit L2 norm cannot diagnose saturation either: it is dominated by a large
negative COMMON MODE across experts (measured r2 step3000: per-token mean logit -83..-185 while
the max was +217..+636). This tool loads a checkpoint on CPU (never a GPU -- read it on the
login box while another run holds the cards), runs one read-only forward over gate-mix tokens,
and reports the three numbers that actually say whether the router is saturated/dead:

  std_c   mean over tokens of the cross-expert STD of z AFTER subtracting the token's common
          mode (z - mean_E z). Softmax ignores common mode; this is the decision-relevant
          spread. r2 step3000 read 46-98 (saturated); a healthy router is far smaller.
  dead    fraction of (token, expert) with logit < -6  (sigmoid < 0.0025, ~zero gradient).
          r2 step3000 read 0.93-0.96 -- 46 of 48 experts dead per token.
  gateT1  mean max share of the top-k RENORMALIZED sigmoid gate (same column as
          moe_health.gate_top1 / h_sums[0]). 1/k even, 1 one-hot.

    python3 scripts/router_logit_stats.py --ckpt runs/x.pt [--n_seq 24] [--domains a,b]
    # beside a live run (CPU, mmap, only n_seq rows ~ tiny IO):
    AUPAI_ALLOW_CORESIDENT_CACHE=1 python3 scripts/router_logit_stats.py --ckpt runs/x.pt
    python3 scripts/router_logit_stats.py --selftest          # synthetic, no ckpt/GPU/data
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEAD_Z = -6.0  # sigmoid(-6) = 0.00247 -- below this the per-expert gradient is ~0


def router_stats(logits, top_k):
    """Pure stats for a [N, E] logit tensor. No model, no device assumption -- the selftest and
    the checkpoint hook both call this so the numbers cannot drift."""
    z = logits.float()
    zc = z - z.mean(-1, keepdim=True)  # remove the cross-expert common mode
    std_c = float(zc.std(-1).mean().item())
    dead = float((z < DEAD_Z).float().mean().item())
    aff = torch.sigmoid(z)
    top = aff.topk(top_k, dim=-1).values
    gate = top / top.sum(-1, keepdim=True).clamp_min(1e-9)
    gateT1 = float(gate.max(-1).values.mean().item())
    return {"std_c": round(std_c, 4), "dead_frac": round(dead, 4), "gateT1": round(gateT1, 4)}


def _selftest():
    E, k = 8, 3
    # 1. ALL-EQUAL logits: common-mode-removed spread is exactly 0; no dead if positive; gate 1/k.
    eq = torch.full((64, E), 4.0)
    s = router_stats(eq, k)
    assert s["std_c"] == 0.0, s
    assert s["dead_frac"] == 0.0, s
    assert abs(s["gateT1"] - 1.0/k) < 2e-4, s  # rounded 4dp
    # all-equal NEGATIVE common mode: common-mode removal still gives std 0 but every expert is
    # dead -- this is exactly why std alone is insufficient (the huge negative mean the runlog
    # norm hid): dead_frac catches it.
    neg = torch.full((64, E), -10.0)
    sn = router_stats(neg, k)
    assert sn["std_c"] == 0.0 and sn["dead_frac"] == 1.0, sn
    # 2. STRICT ONE-HOT: one winner large-positive, others large-negative -> high spread,
    #    (E-1)/E dead, one-hot gate.
    oh = torch.full((64, E), -20.0)
    oh[:, 0] = 20.0
    so = router_stats(oh, k)
    assert so["std_c"] > 10.0, so
    assert abs(so["dead_frac"] - (E - 1) / E) < 1e-5, so
    assert so["gateT1"] > 0.999, so
    # 3. SOFT near-uniform: tiny spread, no dead, even gate -- the healthy target.
    g = torch.Generator().manual_seed(0)
    soft = torch.randn(128, E, generator=g) * 0.3
    sf = router_stats(soft, k)
    assert sf["std_c"] < 0.5 and sf["dead_frac"] == 0.0 and sf["gateT1"] < 0.5, sf
    # 4. common mode must not inflate std: adding +500 to every expert leaves std_c untouched.
    assert abs(router_stats(eq + 500, k)["std_c"] - 0.0) < 1e-6
    print("router_logit_stats selftest OK: std_c is common-mode-free; dead catches negative-mean; "
          "gateT1 1/k even, ~1 one-hot, soft healthy")


def _probe(ckpt_path, n_seq, domains_override):
    import train  # noqa: F401
    from model import HybridLM
    from train import Cfg

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    for k_, v in cfg.items():
        if not k_.startswith("_"):
            setattr(Cfg, k_, v)
    Cfg.seq = int(cfg.get("seq", 4096))
    if str(getattr(Cfg, "router_score", "softmax")) != "sigmoid":
        print(f"note: checkpoint router_score={getattr(Cfg, 'router_score', 'softmax')}; "
              "gateT1/dead use the sigmoid definition (std_c is score-independent)")
    model = HybridLM(Cfg)
    model.load_state_dict(ck["model"])
    model.eval().bfloat16()
    seq, top_k = Cfg.seq, int(Cfg.moe_top_k)

    if domains_override:
        doms = domains_override
    else:
        with open(cfg["mix"]) as mf:
            doms = list(json.load(mf)["domains"].keys())
    # Co-residency refusal first: this is a CPU-only probe meant to run BESIDE a live 8-card run,
    # but the rule is about the /data00 read, not the GPU -- refuse if a live claim holds cards.
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    from cache_guard import assert_not_co_resident
    dom = next((d for d in doms if os.path.exists(train._domain_cache_path(d))), doms[0])
    assert_not_co_resident([dom])
    data = torch.load(train._domain_cache_path(dom), map_location="cpu", weights_only=True, mmap=True)
    nrows = len(data) // (seq + 1)
    take = min(n_seq, nrows)
    pool = data[: nrows * (seq + 1)].view(nrows, seq + 1)
    rows = torch.linspace(0, nrows - 1, take).long()
    ids = pool[rows][:, :seq].long()
    print(f"{os.path.basename(ckpt_path)} step {ck.get('step')} domain {dom} rows {take} "
          f"router_logit_cap={getattr(Cfg, 'router_logit_cap', 0.0)}")

    stats = {}

    def hook(mod, inp, _out, li):
        with torch.no_grad():
            logits = mod.router(inp[0].reshape(-1, inp[0].shape[-1]).bfloat16()).float()
            stats[li] = logits

    moes = [(i, b.ffn) for i, b in enumerate(model.blocks)
            if hasattr(b, "ffn") and hasattr(b.ffn, "expert_bias")]
    hs = [ffn.register_forward_hook(lambda m, i, o, li=li: hook(m, i, o, li)) for li, ffn in moes]
    with torch.no_grad():
        model(ids, targets=None, no_head=True)
    for h in hs:
        h.remove()
    rec = dict(checkpoint=os.path.basename(ckpt_path), step=int(ck.get("step", -1)),
               domain=dom, n_seq=take,
               router_lr=float(getattr(Cfg, "moe_router_lr", -1.0)),
               router_score=str(getattr(Cfg, "router_score", "softmax")),
               router_logit_cap=float(getattr(Cfg, "router_logit_cap", 0.0)),
               layers={f"L{li}": router_stats(stats[li], top_k) for li in sorted(stats)})
    print(f"{rec['checkpoint']} step {rec['step']} domain {dom} rows {take} "
          f"router_lr={rec['router_lr']} score={rec['router_score']} cap={rec['router_logit_cap']}")
    for li in sorted(stats):
        r = rec["layers"][f"L{li}"]
        print(f"L{li:2d} std_c {r['std_c']:8.2f} dead {r['dead_frac']:.3f} gateT1 {r['gateT1']:.4f}")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt")
    ap.add_argument("--n_seq", type=int, default=24)
    ap.add_argument("--domains", default="")
    ap.add_argument("--out", default="", help="append one JSON line per checkpoint here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.ckpt:
        ap.error("--ckpt is required (or use --selftest)")
    doms = [d for d in a.domains.split(",") if d] if a.domains else None
    rec = _probe(a.ckpt, a.n_seq, doms)
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

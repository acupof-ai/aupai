#!/usr/bin/env python3
"""MoE router-health probe: is the gate balanced or collapsed to top-1? (1e, 2026-09-25)

Reusable acceptance tool for the post-collapse re-run. Point it at a checkpoint and it runs
ONE read-only eval forward over real gate-mix sequences (256 per domain by default), hooks
every MoEFFN, and reports per layer the routing health of the trained router next to a
uniform control:

    top1              mean max un-biased softmax affinity   (1.0 = deterministic top-1)
    top3_entropy      mean H/log(top_k) over the selected 3 (0 = top-3 is really top-1)
    n_zero            # experts never selected by topk(affinity + expert_bias)
    gini              gini of the realized 48-expert selection counts (0 = even)

The same three for a fixed-seed uniform nn.Linear control run on the SAME hidden states, so
the number answers "collapsed router" vs "just skewed finite data": a healthy/untrained
router has top1 ~ 1/top_k and entropy near 1; a saturated softmax router has top1 ~ 1 and
entropy ~ 0.

Output: one JSON line per checkpoint (--out), and a one-line-per-layer table to stdout.

    python3 scripts/moe_health.py --ckpt runs/x.pt --n_seq 256 --out runs/moe_health.jsonl
    python3 scripts/moe_health.py --selftest            # no GPU/ckpt/data needed

The selftest builds SYNTHETIC routers (no checkpoint): a uniform one must read healthy
(top1 low, entropy high) and a deliberately one-hot-saturated one must read collapsed
(top1 ~ 1, entropy ~ 0), through the SAME metric function. It is the known-answer gate that
the metric can actually tell the two states apart.
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn

N_EXPERTS_DEFAULT = 48


def layer_metrics(affinity, bias_or_none, top_k):
    """Routing metrics for one layer from un-biased affinity [n, E].

    Pure tensor function; the selftest calls it on synthetic routers, the checkpoint path
    calls it on hooked hidden states. Selection is topk(affinity + bias); the control passes
    bias_or_none=None. Returns plain Python floats/ints so it is dtype/device independent.
    """
    n, e = affinity.shape
    score = affinity if bias_or_none is None else affinity + bias_or_none
    sel = score.topk(top_k, dim=-1).indices
    counts = torch.bincount(sel.reshape(-1), minlength=e).double()
    mean = counts.mean().clamp_min(1e-12)
    sl, _ = torch.sort(counts)
    idx = torch.arange(1, e + 1, device=affinity.device).double()
    gini = float((2 * (idx * sl).sum() / (e * sl.sum()) - (e + 1) / e).item()) if sl.sum() > 0 else float("nan")
    g3 = affinity.gather(1, sel)
    g3 = g3 / g3.sum(-1, keepdim=True).clamp_min(1e-9)
    ent = float((-(g3 * g3.clamp_min(1e-12).log()).sum(-1).mean() / math.log(top_k)).item())
    return dict(
        top1=float(affinity.max(-1).values.mean().item()),
        top3_entropy_norm=ent,
        n_zero=int((counts == 0).sum().item()),
        gini=gini,
        load_max_over_mean=float((counts.max() / mean).item()),
        load_min_over_mean=float((counts.min() / mean).item()),
    )


# ---- checkpoint path (GPU when available) ------------------------------------------------

def measure_ckpt(ckpt_path, n_seq, batch, domains_override, out_paths_only=False):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import train  # noqa: F401
    from model import HybridLM
    from train import Cfg

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        # Claim the card before moving the model to it. Under `harness launch` AUPAI_CLAIMED_BY is
        # set and claim_my_cards treats the launcher's claim as sufficient; run standalone it
        # claims for this process, so a second job cannot read the card as ORPHAN.
        from loader import claim_my_cards
        claim_my_cards("moe_health", note="read-only router-health probe")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    for k, v in cfg.items():
        if not k.startswith("_"):
            setattr(Cfg, k, v)
    Cfg.seq = int(cfg.get("seq", 4096))
    model = HybridLM(Cfg)
    model.load_state_dict(ck["model"])
    model.eval().to(dev)
    seq, d = Cfg.seq, Cfg.d
    n_exp = int(getattr(Cfg, "moe_experts", N_EXPERTS_DEFAULT))
    top_k = int(getattr(Cfg, "moe_top_k", 3))

    if domains_override:
        domains = domains_override
    else:
        mix = json.load(open(os.path.join("/work/aupai", cfg["mix"]))) if os.path.isdir("/work/aupai") \
            else json.load(open(cfg["mix"]))
        domains = list(mix["domains"].keys())

    moes = [(i, m.ffn) for i, m in enumerate(model.blocks) if hasattr(m, "ffn")
            and hasattr(m.ffn, "expert_bias")]
    torch.manual_seed(1234)
    controls = {li: nn.Linear(d, n_exp, bias=False).to(dev).eval() for li, _ in moes}
    for c in controls.values():
        for p in c.parameters():
            p.requires_grad_(False)
    # Accumulated per-expert COUNTS (exact realized load over the whole run) plus sums for the
    # mean metrics. Kept on device and never expanded per-token, so memory stays O(E).
    acc = {li: dict(cnt=torch.zeros(n_exp, dtype=torch.long, device=dev),
                    r_cnt=torch.zeros(n_exp, dtype=torch.long, device=dev),
                    top1=0.0, ent=0.0, r_top1=0.0, r_ent=0.0,
                    lnorm=0.0, r_lnorm=0.0,
                    # (token, expert) the un-biased router picks as its #1 but selection skips
                    decoupled=0, ntok=0)
           for li, _ in moes}

    handles = []
    for li, moe in moes:
        def hook(mod, inp, _out, li=li):
            with torch.no_grad():
                flat = inp[0].reshape(-1, inp[0].shape[-1]).float()
                n = flat.shape[0]
                A = acc[li]
                logits = mod.router(flat).float()
                aff = torch.softmax(logits, dim=-1)
                sel = (aff + mod.expert_bias.float()).topk(top_k, dim=-1).indices
                A["cnt"] += torch.bincount(sel.reshape(-1), minlength=n_exp)
                A["top1"] += float(aff.max(-1).values.sum())
                A["lnorm"] += float(logits.norm(dim=-1).sum())
                # decoupled: the router's own argmax expert is not among the selected top_k -- the
                # learned preference exists but the balancing bias overrode it.
                want = aff.argmax(-1, keepdim=True)
                chosen = torch.zeros_like(aff, dtype=torch.bool).scatter_(1, sel, True)
                A["decoupled"] += int((~chosen.gather(1, want)).sum().item())
                g3 = aff.gather(1, sel); g3 = g3 / g3.sum(-1, keepdim=True).clamp_min(1e-9)
                A["ent"] += float((-(g3 * g3.clamp_min(1e-12).log()).sum(-1)).sum())
                rlogits = controls[li](flat).float()
                raff = torch.softmax(rlogits, dim=-1)
                rsel = raff.topk(top_k, dim=-1).indices
                A["r_cnt"] += torch.bincount(rsel.reshape(-1), minlength=n_exp)
                A["r_top1"] += float(raff.max(-1).values.sum())
                A["r_lnorm"] += float(rlogits.norm(dim=-1).sum())
                rg3 = raff.gather(1, rsel); rg3 = rg3 / rg3.sum(-1, keepdim=True).clamp_min(1e-9)
                A["r_ent"] += float((-(rg3 * rg3.clamp_min(1e-12).log()).sum(-1)).sum())
                A["ntok"] += n
        handles.append(moe.register_forward_hook(hook))

    # restartable: this is a read-only, append-only probe; an interrupt writes nothing partial
    # (one JSON line is emitted only after all domains finish) and re-running restarts the short
    # forward pass from scratch, so there is no half-written artifact to resume from.
    cache_dir = "/data00" if os.path.isdir("/data00") else "."
    if dev == "cuda" and os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "eval")):
        # Refuse to read a token cache off /data00 beside a live training run (the same co-residency
        # chokepoint as eval/cache_guard.py). Local CPU runs have no card context and are exempt.
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "eval"))
        from cache_guard import assert_not_co_resident
    else:
        assert_not_co_resident = None
    for dom in domains:
        if assert_not_co_resident is not None:
            assert_not_co_resident([dom])
        data = torch.load(os.path.join(cache_dir, f"tokens_{dom}.pt"),
                          map_location="cpu", weights_only=True, mmap=True)
        nrows = len(data) // (seq + 1)
        take = min(n_seq, nrows)
        rows = torch.linspace(0, nrows - 1, take).long()
        pool = data[: nrows * (seq + 1)].view(nrows, seq + 1)
        for s in range(0, take, batch):
            ids = pool[rows[s:s + batch], :seq].to(dev)
            if dev == "cuda":
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model(ids, targets=None, no_head=True)
            else:
                with torch.no_grad():
                    model(ids, targets=None, no_head=True)
        del data, pool
    for h in handles:
        h.remove()

    def gini_counts(counts):
        e = counts.numel()
        sl, _ = torch.sort(counts.double())
        if sl.sum() <= 0:
            return float("nan")
        ii = torch.arange(1, e + 1, device=counts.device).double()
        return float((2 * (ii * sl).sum() / (e * sl.sum()) - (e + 1) / e).item())

    layers = {}
    for li, moe in moes:
        A = acc[li]; n = A["ntok"]
        bias = moe.expert_bias.float()
        # gamma=0.001 step; a bias at ~+-0.98 has walked ~the whole usable range. Flag within 0.05.
        at_boundary = int(((bias.abs() >= 0.93)).sum().item())
        layers[str(li)] = {
            "top1": round(A["top1"] / n, 6),
            "top3_entropy_norm": round(A["ent"] / (n * math.log(top_k)), 6),
            "n_zero": int((A["cnt"] == 0).sum().item()),
            "gini": round(gini_counts(A["cnt"]), 6),
            "logit_norm_mean": round(A["lnorm"] / n, 4),
            "argmax_but_unselected_frac": round(A["decoupled"] / n, 4),
            "bias_at_boundary_n": at_boundary,
            "ctrl_top1": round(A["r_top1"] / n, 6),
            "ctrl_top3_entropy_norm": round(A["r_ent"] / (n * math.log(top_k)), 6),
            "ctrl_logit_norm_mean": round(A["r_lnorm"] / n, 4),
            "ctrl_n_zero": int((A["r_cnt"] == 0).sum().item()),
            "ctrl_gini": round(gini_counts(A["r_cnt"]), 6),
        }
    rec = dict(checkpoint=os.path.basename(ckpt_path), step=int(ck.get("step", -1)),
               n_seq_per_domain=n_seq, n_domains=len(domains), n_experts=n_exp, top_k=top_k,
               domains=domains, layers=layers)
    return rec


# ---- selftest: synthetic known-answer -----------------------------------------------------

COLLAPSED_TOP1_FLOOR = 0.9
HEALTHY_TOP1_CEIL = 0.5


def _selftest():
    torch.manual_seed(0)
    n, e, k = 4096, N_EXPERTS_DEFAULT, 3
    bad = 0

    def check(cond, msg):
        nonlocal bad
        if not cond:
            bad += 1
            print("FAIL:", msg)

    # 1. HEALTHY: near-uniform softmax (tiny random logits, like an untrained nn.Linear) -- every
    #    expert wins some tokens over n rows, top1 stays near 1/E, entropy near 1. EXACT-zero
    #    logits would make topk always pick the first k columns deterministically, which is not a
    #    real router's distribution; the control in measure_ckpt is a random nn.Linear for the
    #    same reason.
    g = torch.Generator().manual_seed(7)
    uni = torch.softmax(torch.randn(n, e, generator=g) * 0.01, dim=-1)
    h = layer_metrics(uni, None, k)
    check(h["top1"] < HEALTHY_TOP1_CEIL, f"uniform top1 should be ~1/{e}, got {h['top1']:.3f}")
    check(h["top3_entropy_norm"] > 0.9, f"uniform entropy should be ~1, got {h['top3_entropy_norm']:.3f}")
    check(h["n_zero"] == 0, f"uniform should select every expert, n_zero={h['n_zero']}")
    check(abs(h["gini"]) < 0.05, f"uniform gini ~0, got {h['gini']:.3f}")

    # 2. COLLAPSED: every token routes to the SAME dominant expert with affinity 1. The other
    #    top_k-1 slots tie at affinity 0 and deterministically fall to the next columns, so
    #    exactly top_k experts are touched and n_zero = E - top_k; the discriminative reading is
    #    top1=1, entropy 0 -- load is not spread by learned preference at all.
    onehot = torch.zeros(n, e)
    onehot[:, 0] = 1.0
    c = layer_metrics(onehot, None, k)
    check(c["top1"] >= COLLAPSED_TOP1_FLOOR, f"one-hot top1 should be 1, got {c['top1']:.3f}")
    check(c["top3_entropy_norm"] < 1e-6, f"one-hot entropy 0, got {c['top3_entropy_norm']:.6f}")
    check(c["n_zero"] == e - k, f"single-hot touches exactly top_k experts, n_zero={c['n_zero']}")

    # 3. SOFTMAX SATURATION under realistic scale: logits with one dominant expert per token.
    logits = torch.randn(n, e) * 0.5
    dom = torch.randint(0, e, (n,))
    logits[torch.arange(n), dom] += 12.0  # softmax -> ~1 on the dominant column
    sat = torch.softmax(logits, dim=-1)
    s = layer_metrics(sat, None, k)
    check(s["top1"] >= COLLAPSED_TOP1_FLOOR, f"saturated top1 should be ~1, got {s['top1']:.3f}")
    check(s["top3_entropy_norm"] < 0.05, f"saturated entropy ~0, got {s['top3_entropy_norm']:.4f}")

    # 4. A balancing BIAS cannot rescue the DISCRIMINATIVE metrics. ONE global expert saturates
    #    every row (the measured shape: load max/mean ~16x, L1 e32 affinity 1.000) and the bias
    #    pushes exactly that column out, so selection moves to the tail while the un-biased
    #    affinity stays one-hot. Per-row dominant experts (world 3) would leave ~92% of rows
    #    untouched by a 4-column bias, and a top1 read from the BIASED score then still cleared
    #    the floor (genB mutant on #716: ALL 5 PASS). Here that mutant reads 0.
    glog = torch.randn(n, e) * 0.5
    glog[:, 0] += 12.0
    gsat = torch.softmax(glog, dim=-1)
    bias = torch.zeros(e)
    bias[0] = -50.0
    bm = layer_metrics(gsat, bias, k)
    check(bm["top1"] >= COLLAPSED_TOP1_FLOOR,
          f"bias must not lower un-biased top1, got {bm['top1']:.3f}")
    check(bm["n_zero"] <= e - k, f"bias should spread selection off the hot column, n_zero={bm['n_zero']}")

    # 5. invariance: metric is deterministic and dtype tolerant (bf16 affinity still classified)
    h2 = layer_metrics(uni.to(torch.bfloat16).float(), None, k)
    check(abs(h2["top1"] - h["top1"]) < 0.01, "bf16/ fp32 uniform top1 disagree")

    print(f"moe_health selftest: {'ALL 5 PASS' if bad == 0 else f'{bad} FAIL'}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", help="checkpoint path (required unless --selftest)")
    ap.add_argument("--n_seq", type=int, default=256, help="packed sequences per gate domain")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="", help="append one JSON line here")
    ap.add_argument("--domains", default="", help="comma-separated domains; default all in the mix")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.ckpt:
        ap.error("--ckpt is required (or use --selftest)")
    rec = measure_ckpt(a.ckpt, a.n_seq, a.batch,
                       a.domains.split(",") if a.domains else None)
    print(f"step{rec['step']} {rec['checkpoint']}: n_seq={rec['n_seq_per_domain']}/domain, "
          f"{rec['n_experts']} experts, top_k={rec['top_k']}")
    print("layer | trained top1 ent nZero gini |L| decouple biasBd || control top1 ent |L|")
    for li in sorted(rec["layers"], key=int):
        x = rec["layers"][li]
        print(f"{int(li):2d} | {x['top1']:.4f} {x['top3_entropy_norm']:.4f} "
              f"{x['n_zero']:3d} {x['gini']:.3f} {x['logit_norm_mean']:6.1f} "
              f"{x['argmax_but_unselected_frac']:.3f} {x['bias_at_boundary_n']:3d} || "
              f"{x['ctrl_top1']:.4f} {x['ctrl_top3_entropy_norm']:.4f} "
              f"{x['ctrl_logit_norm_mean']:6.1f}")
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

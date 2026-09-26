#!/usr/bin/env python3
"""MoE router-health probe: is the gate balanced or collapsed to top-1? (1e, 2026-09-25)

Reusable acceptance tool for the post-collapse re-run. Point it at a checkpoint and it runs
ONE read-only eval forward over real gate-mix sequences (256 per domain by default), hooks
every MoEFFN, and reports per layer the routing health of the trained router next to a
uniform control:

    gate_top1         mean max share of the gate RENORMALIZED within the selected top-k
                      (== model.MoEFFN.h_sums[0], train_health's one-hot criterion). This is
                      the COLLAPSE column and is comparable across softmax and sigmoid routers:
                      ~1 = deterministic top-1, ~1/top_k = even.
    affinity_top1     mean max RAW un-normalized affinity; reference only. Meaningful under
                      softmax (~0.996 = one-hot on v41_ced_0923), but under per-expert sigmoid
                      each expert is independent so a saturated 0.998 says nothing about the
                      routed gate (which can still split 1/top_k).
    top3_entropy      mean H/log(top_k) over the selected 3 (0 = top-3 is really top-1)
    n_zero            # experts never selected by topk(affinity + expert_bias)
    load max/mean     skew of the realized 48-expert selection counts (1 = even); gini also kept
    gini              gini of the selection counts (0 = even)

ALARM if gate_top1 > 0.90 OR argmax_selected < 0.50 (per layer "alarm" bool in the JSON).

Affinity/selection/gate come from the module's own MoEFFN._route, so the probe uses the
checkpoint cfg's router_score (softmax default, or per-expert sigmoid V3 sec 2.1.2) and
cannot hardcode the wrong distribution. The same metrics are reported for a fixed-seed
uniform nn.Linear control run on the SAME hidden states with the SAME score function, so the
number answers "collapsed router" vs "just skewed finite data": a healthy/untrained router
has gate_top1 ~ 1/top_k and entropy near 1; a one-hot router has gate_top1 ~ 1, entropy ~ 0.

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

    TWO top-1 columns -- they are the same under softmax but NOT under per-expert sigmoid:
      gate_top1     max share of the gate RENORMALIZED within the selected top-k. This is the
                    one-hot/collapse criterion and equals model.MoEFFN.h_sums[0] (train_health's
                    top1). A sigmoid router can saturate every affinity to ~0.998 yet split the
                    top-k gate 1/k each (input-independent, but not one-hot); raw affinity cannot
                    show that.
      argmax_selected  fraction of rows where the largest affinity among the SELECTED top-k is
                     >= the row's largest affinity (a VALUE comparison, not index equality):
                     aff.gather(sel).max >= aff.max. A tie counts as selected, so an all-equal
                     (saturated sigmoid) router reads 1 regardless of how topk/argmax break the
                     tie (x86 picks different indices than the laptop). ~1 healthy; ~0 when the
                     balancing bias vetoes the router's STRICT first choice and forces cold
                     experts (v41_ced_0923).
    affinity_top1     mean max RAW un-normalized affinity; reference only. Meaningful under
    Alarm if gate_top1 > 0.9 OR argmax_selected < 0.5.
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
    g3 = g3 / g3.sum(-1, keepdim=True).clamp_min(1e-9)  # gate renormalized within top-k
    ent = float((-(g3 * g3.clamp_min(1e-12).log()).sum(-1).mean() / math.log(top_k)).item())
    # VALUE comparison, tie counts as selected: max affinity inside the selection vs the row max.
    selected_max = affinity.gather(1, sel).max(-1).values
    row_max = affinity.max(-1).values
    argmax_selected = float((selected_max >= row_max).float().mean().item())
    return dict(
        gate_top1=float(g3.max(-1).values.mean().item()),
        argmax_selected=argmax_selected,
        affinity_top1=float(affinity.max(-1).values.mean().item()),
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
                    gate_top1=0.0, aff_top1=0.0, ent=0.0,
                    r_gate_top1=0.0, r_aff_top1=0.0, r_ent=0.0,
                    lnorm=0.0, r_lnorm=0.0,
                    # rows whose max selected affinity >= the row max (VALUE, ties count selected)
                    argmax_sel=0, ntok=0)
           for li, _ in moes}

    handles = []
    for li, moe in moes:
        def hook(mod, inp, _out, li=li):
            with torch.no_grad():
                flat = inp[0].reshape(-1, inp[0].shape[-1]).float()
                n = flat.shape[0]
                A = acc[li]
                logits = mod.router(flat).float()
                # Route through the module's OWN _route: affinity is softmax or per-expert sigmoid
                # (V3 sec 2.1.2) per the checkpoint cfg's router_score, selection is
                # topk(affinity + expert_bias), gate is the un-biased affinity renormalized in the
                # top-k. Recomputing torch.softmax here read a sigmoid checkpoint's top1 wrongly.
                aff, sel, gate = mod._route(logits)
                A["cnt"] += torch.bincount(sel.reshape(-1), minlength=n_exp)
                # gate_top1 = collapse criterion (== model h_sums[0]); affinity_top1 = raw reference
                A["gate_top1"] += float(gate.max(-1).values.sum())
                A["aff_top1"] += float(aff.max(-1).values.sum())
                A["lnorm"] += float(logits.norm(dim=-1).sum())
                # argmax_selected by VALUE (ties count selected): the largest affinity inside the
                # selection vs the row's largest. Index equality would misread an all-equal
                # saturated sigmoid router on x86 where topk and argmax break ties differently.
                A["argmax_sel"] += int((aff.gather(1, sel).max(-1).values >= aff.max(-1).values)
                                       .sum().item())
                g3 = gate  # _route already renormalized the un-biased affinity within top-k
                A["ent"] += float((-(g3 * g3.clamp_min(1e-12).log()).sum(-1)).sum())
                rlogits = controls[li](flat).float()
                # The untrained reference uses the SAME score function as the checkpoint router:
                # comparing a sigmoid checkpoint against a softmax reference would contrast two
                # different affinity distributions, not trained-vs-untrained.
                raff = torch.sigmoid(rlogits) if mod.router_score == "sigmoid" else torch.softmax(
                    rlogits, dim=-1)
                rsel = raff.topk(top_k, dim=-1).indices
                A["r_cnt"] += torch.bincount(rsel.reshape(-1), minlength=n_exp)
                A["r_aff_top1"] += float(raff.max(-1).values.sum())
                rg3 = raff.gather(1, rsel); rg3 = rg3 / rg3.sum(-1, keepdim=True).clamp_min(1e-9)
                A["r_gate_top1"] += float(rg3.max(-1).values.sum())
                A["r_lnorm"] += float(rlogits.norm(dim=-1).sum())
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
        A = acc[li]
        n = A["ntok"]
        bias = moe.expert_bias.float()
        # gamma=0.001 step; a bias at ~+-0.98 has walked ~the whole usable range. Flag within 0.05.
        at_boundary = int(((bias.abs() >= 0.93)).sum().item())
        gate_top1 = round(A["gate_top1"] / n, 6)
        argmax_selected = round(A["argmax_sel"] / n, 6)
        layers[str(li)] = {
            "gate_top1": gate_top1,
            "argmax_selected": argmax_selected,
            "alarm": bool(gate_top1 > GATE_COLLAPSE_MAX or argmax_selected < ARGMAX_SELECTED_MIN),
            "affinity_top1": round(A["aff_top1"] / n, 6),
            "top3_entropy_norm": round(A["ent"] / (n * math.log(top_k)), 6),
            "n_zero": int((A["cnt"] == 0).sum().item()),
            "gini": round(gini_counts(A["cnt"]), 6),
            "load_max_over_mean": round(float(A["cnt"].max().double() / A["cnt"].double().mean().clamp_min(1e-12)), 3),
            "logit_norm_mean": round(A["lnorm"] / n, 4),
            "bias_at_boundary_n": at_boundary,
            "ctrl_gate_top1": round(A["r_gate_top1"] / n, 6),
            "ctrl_affinity_top1": round(A["r_aff_top1"] / n, 6),
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
# Alarm predicate (1e 2026-09-26), must match train_health's gate: gate_top1 one-hot OR the
# balancing bias vetoing the router's own first choice.
GATE_COLLAPSE_MAX = 0.90
ARGMAX_SELECTED_MIN = 0.50


def layer_alarm(m):
    return m["gate_top1"] > GATE_COLLAPSE_MAX or m["argmax_selected"] < ARGMAX_SELECTED_MIN


def _selftest():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    torch.manual_seed(0)
    n, e, k = 4096, N_EXPERTS_DEFAULT, 3
    bad = 0

    def check(cond, msg):
        nonlocal bad
        if not cond:
            bad += 1
            print("FAIL:", msg)

    # 1. HEALTHY: near-uniform softmax (tiny random logits, like an untrained nn.Linear) -- every
    #    expert wins some tokens over n rows. The COLLAPSE column is gate_top1: the top-k gate is
    #    near-even so its max share ~1/k (1/3), well below 1. The RAW affinity max is ~1/E.
    g = torch.Generator().manual_seed(7)
    uni = torch.softmax(torch.randn(n, e, generator=g) * 0.01, dim=-1)
    h = layer_metrics(uni, None, k)
    check(h["gate_top1"] < HEALTHY_TOP1_CEIL, f"uniform gate_top1 should be ~1/{k}, got {h['gate_top1']:.3f}")
    check(abs(h["gate_top1"] - 1.0 / k) < 0.05, f"uniform gate_top1 ~1/{k}, got {h['gate_top1']:.3f}")
    check(h["affinity_top1"] < 0.05, f"uniform raw affinity max ~1/{e}, got {h['affinity_top1']:.4f}")
    check(h["top3_entropy_norm"] > 0.9, f"uniform entropy should be ~1, got {h['top3_entropy_norm']:.3f}")
    check(h["n_zero"] == 0, f"uniform should select every expert, n_zero={h['n_zero']}")
    check(abs(h["gini"]) < 0.05, f"uniform gini ~0, got {h['gini']:.3f}")
    check(not layer_alarm(h), "a healthy uniform router must not alarm")

    # 2. COLLAPSED: every token routes to the SAME dominant expert with affinity 1. The other
    #    top_k-1 slots tie at affinity 0 and deterministically fall to the next columns, so
    #    exactly top_k experts are touched and n_zero = E - top_k; the discriminative reading is
    #    gate_top1=1, entropy 0 -- load is not spread by learned preference at all.
    onehot = torch.zeros(n, e)
    onehot[:, 0] = 1.0
    c = layer_metrics(onehot, None, k)
    check(c["gate_top1"] >= COLLAPSED_TOP1_FLOOR, f"one-hot gate_top1 should be 1, got {c['gate_top1']:.3f}")
    check(c["affinity_top1"] >= COLLAPSED_TOP1_FLOOR, f"one-hot affinity_top1 should be 1, got {c['affinity_top1']:.3f}")
    check(c["top3_entropy_norm"] < 1e-6, f"one-hot entropy 0, got {c['top3_entropy_norm']:.6f}")
    check(c["n_zero"] == e - k, f"single-hot touches exactly top_k experts, n_zero={c['n_zero']}")
    check(c["argmax_selected"] == 1.0 and layer_alarm(c),
          "one-hot collapses via gate_top1 even though its argmax is selected")

    # 3. SOFTMAX SATURATION under realistic scale: logits with one dominant expert per token.
    logits = torch.randn(n, e) * 0.5
    dom = torch.randint(0, e, (n,))
    logits[torch.arange(n), dom] += 12.0  # softmax -> ~1 on the dominant column
    sat = torch.softmax(logits, dim=-1)
    s = layer_metrics(sat, None, k)
    check(s["gate_top1"] >= COLLAPSED_TOP1_FLOOR, f"saturated gate_top1 should be ~1, got {s['gate_top1']:.3f}")
    check(s["top3_entropy_norm"] < 0.05, f"saturated entropy ~0, got {s['top3_entropy_norm']:.4f}")

    # 4. THE v41_ced_0923 FAILURE SHAPE: bias vetoes the router's own first choice. ONE global
    #    expert has raw affinity ~1 every row; the bias pushes that column out of SELECTION.
    #    gate_top1 over the bias-picked runners-up drops to ~1/k and ALONE reads healthy, but the
    #    router's argmax is never selected: argmax_selected ~0. The old single gate_top1 column
    #    missed this; the new column must alarm on it.
    glog = torch.randn(n, e) * 0.5
    glog[:, 0] += 12.0
    gsat = torch.softmax(glog, dim=-1)
    bias = torch.zeros(e)
    bias[0] = -50.0
    bm = layer_metrics(gsat, bias, k)
    check(bm["affinity_top1"] >= COLLAPSED_TOP1_FLOOR,
          f"raw learned preference is still one-hot, got {bm['affinity_top1']:.3f}")
    check(abs(bm["gate_top1"] - 1.0 / k) < 0.12,
          f"routed gate over bias-selected runners-up must be ~1/{k}, got {bm['gate_top1']:.3f}")
    check(bm["argmax_selected"] < ARGMAX_SELECTED_MIN,
          f"router argmax must be vetoed -> argmax_selected < {ARGMAX_SELECTED_MIN}, "
          f"got {bm['argmax_selected']:.4f}")
    check(bm["load_max_over_mean"] < 2.0,
          f"bias should spread selection off the hot column, load max/mean={bm['load_max_over_mean']:.2f}")
    # MUTANT GATE: gate_top1 alone must NOT fire here (that is the blind spot), the full predicate
    # MUST. Deleting the argmax_selected clause makes the second assertion red -- world 4 then
    # passes as healthy, exactly the 0923 miss.
    gate_only_alarm = bm["gate_top1"] > GATE_COLLAPSE_MAX
    check(not gate_only_alarm, "gate_top1 alone must miss the bias-veto shape (the blind spot)")
    check(layer_alarm(bm), "full predicate must alarm on the bias-veto shape via argmax_selected")

    # 5. invariance: metric is deterministic and dtype tolerant (bf16 affinity still classified)
    h2 = layer_metrics(uni.to(torch.bfloat16).float(), None, k)
    check(abs(h2["gate_top1"] - h["gate_top1"]) < 0.01, "bf16/fp32 uniform gate_top1 disagree")

    # 6. SIGMOID ROUTER (DeepSeek-V3 sec 2.1.2) through the REAL MoEFFN._route. Raw per-expert
    #    sigmoid affinities are independent, so their raw max has no collapse meaning; the collapse
    #    criterion is gate_top1 (top-k-renormalized), == model.MoEFFN.h_sums[0]. The old hook both
    #    (a) recomputed softmax for a sigmoid checkpoint and (b) read the RAW affinity max, so it
    #    misread the two shapes below. Drive the production module so the branch cannot drift.
    from model import MoEFFN

    e6, k6, w6 = 8, 3, 16  # (k + 1 shared) * w == ffn_hidden 64
    cfg6 = type(
        "C", (),
        dict(d=64, ffn_hidden=64, layers=4, vocab=100, seq=16, attn_every=4, moe_experts=e6,
             moe_top_k=k6, moe_shared=1, moe_expert_ffn=w6, moe_bias_gamma=0.001,
             moe_balance_alpha=1e-4, router_score="sigmoid"),
    )()
    m6 = MoEFFN(cfg6).eval()
    with torch.no_grad():
        # 6a. INPUT-INDEPENDENT but NOT one-hot: every logit large-positive saturates EVERY
        #     independent sigmoid to ~0.998, so the top-k gate splits 1/k each. gate_top1 must be
        #     1/k (NOT collapsed), while the RAW affinity max is ~0.998. A metric that reads raw
        #     affinity as the collapse column -- exactly the bug -- turns this world red; the two
        #     columns disagreeing is the mutant that must be caught.
        equal_logits = torch.full((256, e6), 6.0)
        aff_e, sel_e, gate_e = m6._route(equal_logits)
        me = layer_metrics(aff_e, None, k6)
        check(abs(me["gate_top1"] - 1.0 / k6) < 1e-4,
              f"equal-saturated sigmoid gate_top1 must be 1/{k6} (not one-hot), got {me['gate_top1']:.3f}")
        check(me["affinity_top1"] > 0.99,
              f"equal-saturated raw affinity max ~0.998, got {me['affinity_top1']:.3f}")
        # TIE MUST COUNT AS SELECTED on every machine: all affinities equal, so the selected top-k
        # max equals the row max exactly. An INDEX-equality definition (argmax idx in topk idx)
        # reads 0 on x86 where topk and argmax break the all-equal tie differently (CI fail, laptop
        # passed by luck). Value comparison is tie-break independent -> exactly 1.
        check(me["argmax_selected"] == 1.0,
              f"all-equal affinity must have argmax_selected 1.0 by value (tie selected), "
              f"got {me['argmax_selected']}")
        # MUTANT GATE: train_health.TOP1_MAX = 0.90 is the one-hot collapse gate. The correct
        # column (gate_top1=1/k ~0.333) passes it; a build that mistakenly sets the collapse column
        # to the RAW affinity max (0.998) trips it and would SIGTERM this healthy-even router.
        from train_health import TOP1_MAX as _GATE
        check(me["gate_top1"] <= _GATE and me["affinity_top1"] > _GATE,
              f"raw-affinity-as-collapse-column mutant must go red: gate {me['gate_top1']:.3f} "
              f"vs raw {me['affinity_top1']:.3f}, gate threshold {_GATE}")
        # tied saturation: argmax col is among the top-k ties, argmax_selected=1, so the full
        # alarm predicate stays SILENT (this world is input-independent but not one-hot/bias-veto;
        # the two columns judged together).
        check(not layer_alarm(me), f"equal-saturated sigmoid must not alarm, {me}")
        # softmax of the same logits is forced to 1/E -- why a softmax recomputation misreads it
        check(abs(float(torch.softmax(equal_logits, -1).max(-1).values.mean()) - 1.0 / e6) < 1e-6,
              "softmax of equal logits must be 1/E (documents the wrong-recompute distribution)")

        # 6b. TRUE one-hot under sigmoid: one expert large-positive, the rest large-negative. The
        #     selected gate is ~0.998 vs two ~0.0025, so gate_top1 ~1 (collapsed).
        one_logits = torch.full((256, e6), -6.0)
        one_logits[:, 0] = 6.0
        aff_o, _, gate_o = m6._route(one_logits)
        mo = layer_metrics(aff_o, None, k6)
        check(mo["gate_top1"] > 0.99, f"one-hot sigmoid gate_top1 must be ~1, got {mo['gate_top1']:.3f}")
        check(torch.allclose(gate_o.max(-1).values.mean().reshape(1),
                             torch.tensor([mo["gate_top1"]]), atol=1e-5),
              "layer_metrics gate_top1 must equal the routed gate's max share (h_sums[0])")

        # 6c. expert_bias enters SELECTION only: force col 0 out of a tied router, its gate weight
        #     stays the un-biased sigmoid renormalized to 1/k.
        m6.expert_bias.zero_()
        m6.expert_bias[0] = -50.0
        zlogits = torch.zeros(1, e6)  # all sigmoids tie at 0.5; argmax tie is col 0
        _, selz, gatez = m6._route(zlogits)
        check(0 not in selz.tolist(), "selection-only bias must move selection off col 0")
        check(torch.allclose(gatez, torch.full_like(gatez, 1.0 / k6), atol=1e-5),
              "gate stays the un-biased sigmoid (0.5 renorm -> 1/k)")

    print(f"moe_health selftest: {'ALL 6 PASS' if bad == 0 else f'{bad} FAIL'}")
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
    print("layer | gateT1 argMaxSel affT1 nZero loadMx | ent gini |L| biasBd || "
          "ctrlGateT1 ctrlAffT1 ctrlNZero (control has no bias -> argMaxSel=1)")
    for li in sorted(rec["layers"], key=int):
        x = rec["layers"][li]
        print(f"{int(li):2d} | {x['gate_top1']:.4f} {x['argmax_selected']:.4f} "
              f"{x['affinity_top1']:.4f} {x['n_zero']:3d} {x['load_max_over_mean']:5.2f} | "
              f"{x['top3_entropy_norm']:.4f} {x['gini']:.3f} {x['logit_norm_mean']:6.1f} "
              f"{x['bias_at_boundary_n']:3d} || {x['ctrl_gate_top1']:.4f} "
              f"{x['ctrl_affinity_top1']:.4f} {x['ctrl_n_zero']:3d}")
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

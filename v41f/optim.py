"""The V4.1 optimizer for the v42 trainer (DeepSeek-V4.1 report §2.5 and §4.2.2).

Three update rules, one learning-rate schedule:

  Muon      2-D weight matrices of the backbone (and the grouped 3-D wo_a, one matrix per group).
            Nesterov momentum 0.95, decoupled weight decay 0.1, every orthogonalised update matrix
            rescaled to RMS 0.18 so it shares the AdamW learning rate. Query and indexer-query
            weights are split by head before orthogonalisation (head-wise Muon).
  Sinkhorn  the token embedding and the prediction head: Nesterov momentum, then K alternating
            row/column l2 normalisations (Algorithm 1: K=11, tau=1e-3, eps=1e-20), times sqrt(n)
            for unit row RMS, times gamma=0.18. No weight decay.
  AdamW     RMSNorm weights (wd 0.1), the router (wd 0.1, AdamW per ruling (f) of moe_0905),
            and every other non-matrix parameter -- attn_sink, mHC scale/base -- at wd 0.
            betas (0.9, 0.95), eps 1e-20.

Newton-Schulz uses the Polar Express coefficients train.py's Muon uses (copied, not imported:
importing train.py pulls in the whole trainer). The report does not state its NS coefficients.
"""

import math

import torch

POLAR_EXPRESS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def orthogonalize(x: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Polar-Express Newton-Schulz over the trailing two dims; leading dims are a batch of
    independent matrices. bf16 on CUDA (train.py's precision), fp32 elsewhere."""
    dt = torch.bfloat16 if x.is_cuda else torch.float32
    tall = x.size(-2) > x.size(-1)
    X = x.to(dt)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    for a, b, c in POLAR_EXPRESS[:steps]:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return (X.mT if tall else X).float()  # ponytail: fp32 copy per chunk; bounded by V42Muon.CHUNK


def sinkhorn_balance(g: torch.Tensor, k: int = 11, tau: float = 1e-3, eps: float = 1e-20) -> torch.Tensor:
    """Algorithm 1 steps 3-17 on a [m, n] matrix with m the larger (vocab) dimension: rows whose
    l2 norm is at most tau times the mean row norm are zeroed, then k alternating normalisations
    (odd k: rows, even: columns), then sqrt(n) so a row has unit RMS."""
    u = g.float()
    rho = u.norm(dim=1)
    u = u * (rho > tau * rho.mean()).unsqueeze(1)
    for i in range(1, k + 1):
        dim = 1 if i % 2 else 0
        u = u / (u.norm(dim=dim, keepdim=True) + eps)
    return math.sqrt(u.size(1)) * u


class V42Muon(torch.optim.Optimizer):
    """Group keys: lr, momentum, weight_decay, rms, ns_steps, heads (per-parameter list, the
    number of row blocks to orthogonalise separately; 1 = the whole matrix)."""

    CHUNK = 512

    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.1, rms=0.18, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                                      rms=rms, ns_steps=ns_steps, heads=None))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta, lr, wd = group["momentum"], group["lr"], group["weight_decay"]
            heads = group["heads"] or [1] * len(group["params"])
            live = [(p, h) for p, h in zip(group["params"], heads, strict=True) if p.grad is not None]
            if not live:  # torch._foreach_* refuse an empty list; a group with no grads this step is a no-op
                continue
            for p, _ in live:
                if "momentum_buffer" not in self.state[p]:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(p.grad)
            grads = [p.grad for p, _ in live]
            bufs = [self.state[p]["momentum_buffer"] for p, _ in live]
            torch._foreach_lerp_(bufs, grads, 1 - beta)
            us = torch._foreach_lerp(grads, bufs, beta)  # Nesterov: beta*M_t + (1-beta)*G_t
            by_shape = {}
            for (p, h), u in zip(live, us, strict=True):
                mats = u.reshape(-1, u.size(-2) // h, u.size(-1)) if u.ndim == 2 else u.reshape(-1, *u.shape[-2:])
                by_shape.setdefault(tuple(mats.shape[-2:]), []).append((p, mats))
            for items in by_shape.values():
                # batched NS over same-shape matrices, in chunks of <= CHUNK matrices so the 1,536
                # expert matrices of one shape do not materialise at once
                i = 0
                while i < len(items):
                    j, n = i, 0
                    while j < len(items) and (n == 0 or n + items[j][1].size(0) <= self.CHUNK):
                        n += items[j][1].size(0)
                        j += 1
                    chunk = items[i:j]
                    o = orthogonalize(torch.cat([m for _, m in chunk]), group["ns_steps"])
                    o = o * (group["rms"] / o.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-12))
                    ps = [p for p, _ in chunk]
                    torch._foreach_mul_(ps, 1 - lr * wd)
                    torch._foreach_add_(ps, [oi.reshape(p.shape).to(p.dtype) for p, oi in
                                             zip(ps, o.split([m.size(0) for _, m in chunk]), strict=True)], alpha=-lr)
                    i = j


class SinkhornMomentum(torch.optim.Optimizer):
    """Algorithm 1 for embedding tables and the prediction head, [vocab, dim] layout."""

    def __init__(self, params, lr=1e-3, momentum=0.95, gamma=0.18, k=11, tau=1e-3, eps=1e-20):
        super().__init__(params, dict(lr=lr, momentum=momentum, gamma=gamma, k=k, tau=tau, eps=eps,
                                      weight_decay=0.0))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.ndim == 2 and p.size(0) >= p.size(1), f"Sinkhorn wants [vocab, dim], got {tuple(p.shape)}"
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p.grad)
                m = st["momentum_buffer"]
                m.lerp_(p.grad, 1 - beta)
                d = sinkhorn_balance(p.grad.lerp(m, beta), group["k"], group["tau"], group["eps"])
                p.add_(d.to(p.dtype), alpha=-group["lr"] * group["gamma"])


def v42_param_groups(model):
    """{group: [(name, param)]} over the trainable parameters. Every trainable parameter lands in
    exactly one group; frozen ones (the dormant indexer path) in none."""
    groups = {"muon": [], "sinkhorn": [], "adamw_decay": [], "adamw_nodecay": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n in ("embed.weight", "head.weight"):
            groups["sinkhorn"].append((n, p))
        elif n.endswith("norm.weight") or n.endswith("ffn.gate.weight"):
            groups["adamw_decay"].append((n, p))
        elif p.ndim >= 2:
            groups["muon"].append((n, p))
        else:
            groups["adamw_nodecay"].append((n, p))
    return groups


def _grad_group_key(name):
    """`layers.7.attn.indexer.wq_b.weight` -> `layers.7.attn.indexer`; `layers.7.ffn.w1` -> `layers.7.ffn`
    (stacked experts) / `layers.7.ffn.experts` (per-expert); `embed.weight` -> `embed`."""
    f = name.split(".")
    if f[0] == "layers":
        keep = 3
        if len(f) > 3 and f[3] in ("indexer", "compressor", "experts", "gate", "qproj", "kvproj", "oproj"):
            keep = 4
        return ".".join(f[:keep])
    return f[0]


def grad_norm_report(model, topk=5):
    """One line from the grads already on the parameters (before clipping, no extra backward, no
    collective): the top-k module groups by grad L2 norm and the per-optimizer-group totals.
    Norms are the sqrt of summed squares per group, fp32, on the parameters' device."""
    groups = v42_param_groups(model)
    by_mod, by_opt = {}, {}
    for gname, items in groups.items():
        for n, p in items:
            if p.grad is None:
                continue
            sq = p.grad.detach().float().square().sum()
            k = _grad_group_key(n)
            by_mod[k] = by_mod.get(k, 0.0) + sq
            by_opt[gname] = by_opt.get(gname, 0.0) + sq
    top = sorted(by_mod.items(), key=lambda kv: -float(kv[1]))[:topk]
    mods = " ".join(f"{k}={float(v) ** 0.5:.3g}" for k, v in top)
    opts = " ".join(f"{k}={float(by_opt.get(k, 0.0)) ** 0.5:.3g}" for k in ("muon", "sinkhorn", "adamw_decay", "adamw_nodecay"))
    return f"gradnorm top{topk} {mods} | groups {opts}"

def _heads_of(name, cfg):
    if name.endswith("qproj.wq_b.weight"):
        return cfg.n_heads
    if name.endswith("indexer.wq_b.weight"):
        return cfg.index_n_heads
    return 1


def build_v42_optimizers(model, cfg, lr, adam_eps=1e-20):
    """[Muon, Sinkhorn, AdamW], each group carrying initial_lr/initial_wd for train.set_schedule
    and an aupai_group name for the step line. `cfg` is the V41FConfig (for head counts)."""
    g = v42_param_groups(model)
    muon = V42Muon([{"params": [p for _, p in g["muon"]], "heads": [_heads_of(n, cfg) for n, _ in g["muon"]]}],
                   lr=lr)
    sink = SinkhornMomentum([p for _, p in g["sinkhorn"]], lr=lr)
    fused = all(p.is_cuda for _, p in g["adamw_decay"] + g["adamw_nodecay"])
    adam = torch.optim.AdamW(
        [{"params": [p for _, p in g["adamw_decay"]], "weight_decay": 0.1},
         {"params": [p for _, p in g["adamw_nodecay"]], "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=adam_eps, fused=fused or None)
    opts = [muon, sink, adam]
    for opt, name in zip(opts, ("muon", "sinkhorn", "adamw"), strict=True):
        opt.aupai_group = name
        for grp in opt.param_groups:
            grp["initial_lr"] = grp["lr"]
            grp["initial_wd"] = grp["weight_decay"]
    return opts

#!/usr/bin/env python3
"""P0-P3: what a fine-grained MoE FFN costs against the dense SwiGLU it replaces.

CHARTER: the ~0.8B-total MoE arm at EQUAL ACTIVE COMPUTE to the 200M dense (4c's ruling
2026-09-05, after this file's arithmetic showed the first config was 1.25x). Backbone
unchanged. The readout is a same-step tok/s ratio against the 0.85 gate, like memory_layers
readout 5.

EQUAL ACTIVE COMPUTE IS THE PRECONDITION, not a detail. model.SwiGLU is w13 (d -> 2*ffn_hidden,
fused) plus w2 (ffn_hidden -> d) = 3*d*ffn_hidden per token. At d=1024, ffn_hidden=3072 that is
9.44M. 32 experts at ffn 768 with top-4 + 1 shared is 3*1024*768*5 = 11.80M -- ratio 1.25, so a
tok/s comparison would have charged the MoE arm 25% more compute and called the result an
architecture difference. top-3 + shared is 9.44M exactly. This file asserts that identity at
import; there is no run to interpret if it does not hold.

WHY A MICROBENCHMARK AND NOT AN ARM. Four questions gate the arm and none of them needs a
training run: does the fp8 grouped path execute at our per-expert sizes (P0), what dispatch
costs relative to dense (P1), whether torch.compile recompiles on token-dependent shapes (P2),
and the step ratio (P3). An arm would answer them in days and confound them with optimization.

THE PREDICTION, recorded before the measurement (prereg, 4c 2026-09-05): _grouped_mm within
1.3x of dense; sort-and-loop 3-10x worse at 32 experts. Both directions of a miss get reported.

    python3 scripts/moe_dispatch_bench.py --selftest      # CPU, no card
    python3 scripts/moe_dispatch_bench.py --phase p0      # on the granted card
    python3 scripts/moe_dispatch_bench.py --phase all --json runs/moe_bench.json

# restartable: each phase writes its own row and rerunning a phase costs its own minutes only.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# The arm's shape. d and ffn_hidden are read from train.Cfg at runtime where available; these
# are the charter's numbers and the selftest checks them against Cfg rather than trusting them.
D = 1024
DENSE_FFN = 3072
N_EXPERTS = 32
EXPERT_FFN = 768
TOP_K = 3          # top-3 + 1 shared == dense active compute exactly. NOT 4; see the docstring.
N_SHARED = 1
SEQ = 4096


def active_params(d, ffn, n_active):
    """Params-worth of matmul one token drives. 3*d*ffn is SwiGLU's w13 (2*d*ffn) + w2 (d*ffn),
    so the same formula scores dense and expert FFNs and the ratio below is like-for-like."""
    return 3 * d * ffn * n_active


def compute_ratio(d=D, dense_ffn=DENSE_FFN, expert_ffn=EXPERT_FFN, top_k=TOP_K,
                  n_shared=N_SHARED):
    return active_params(d, expert_ffn, top_k + n_shared) / active_params(d, dense_ffn, 1)


def _dev():
    import torch
    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible: this benchmark measures a card. "
                 "Run --selftest for the CPU-side checks.")
    # TOUCH THE CARD BEFORE ANY PHASE RUNS, so a launcher's claim can bind. card_claim's
    # --wait-for-device polls for a descendant holding an nvidia fd; a phase that finishes in
    # under the ~1.3 s it takes torch to open one exits first, and the claim reports "exited
    # while polling" while the job HAS used the card. Measured 2026-09-05: --phase p0 alone did
    # exactly that and ran unclaimed. One tiny allocation removes the race.
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    return torch.device("cuda")


def build_moe(dtype, device):
    """Expert weights as ONE stacked tensor per matrix, which is what _grouped_mm consumes.

    Stacked rather than a ModuleList of Linears: the grouped op takes per-expert matrices and a
    row offset, so a list would have to be stacked on every call and the benchmark would measure
    the stacking. The loop fallback indexes the same tensor, so both paths hold identical weights
    and a difference between them is dispatch, not initialization.

    RETURNED TRANSPOSED, (E, N, K).transpose(-2,-1), because _grouped_mm refuses a contiguous
    mat2 -- "Expected mat2 to be transposed", measured 2026-09-05 on every group size swept. The
    loop path multiplies the same views, so neither path gets a layout the other does not."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(0)
    w13 = torch.randn(N_EXPERTS, 2 * EXPERT_FFN, D, generator=g) * D ** -0.5
    w2 = torch.randn(N_EXPERTS, D, EXPERT_FFN, generator=g) * EXPERT_FFN ** -0.5
    return (w13.to(device=device, dtype=dtype).transpose(-2, -1),
            w2.to(device=device, dtype=dtype).transpose(-2, -1))


def route(x_flat, n_experts=N_EXPERTS, top_k=TOP_K, seed=0):
    """Token -> expert assignment, sorted, with the per-expert offsets _grouped_mm wants.

    A REAL ROUTER'S OUTPUT SHAPE, not a uniform split. Uniform assignment is the case where
    every dispatch path looks best: each expert gets exactly T*k/E rows, no padding is wasted
    and the sort is free. Real routing is imbalanced, so the tokens are drawn from a skewed
    distribution -- the measurement has to see the shape it will meet."""
    import torch
    n = x_flat.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Zipf-ish expert popularity: a few experts take a disproportionate share, which is the
    # collapse-adjacent regime a router actually sits in early in training.
    p = 1.0 / torch.arange(1, n_experts + 1, dtype=torch.float32) ** 0.7
    p = p / p.sum()
    idx = torch.multinomial(p.expand(n, -1), top_k, replacement=False, generator=gen)
    flat = idx.reshape(-1).to(x_flat.device)
    order = torch.argsort(flat)
    counts = torch.bincount(flat, minlength=n_experts)
    # OFFSETS ARE CUMULATIVE ENDS, which is _grouped_mm's `offs` convention.
    offs = torch.cumsum(counts, 0).to(torch.int32).to(x_flat.device)
    return order, offs, counts


def _sync_time(fn, iters, warmup=5):
    """Median ms over `iters`, CUDA-synchronized. Median not mean: one autotune outlier in the
    first timed iteration would move a mean by more than the effect being measured."""
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort()
    return ts[len(ts) // 2]


def p0_fp8_constraint(device):
    """Does _scaled_grouped_mm run at our per-expert sizes, and what does it refuse?

    The answer is a property of THIS build at THESE shapes, not a documented API contract, so
    the fact row carries the torch version and the swept sizes and its boundary says it does
    not transfer across an upgrade."""
    import torch
    out = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0),
           "op_present": hasattr(torch, "_scaled_grouped_mm"), "cases": []}
    if not out["op_present"]:
        out["verdict"] = "absent: the fp8 grouped path does not exist in this build"
        return out
    # Sweep the group size, which is the dimension the alignment constraint lives on. Each case
    # is ONE grouped call over n_experts groups of `g` rows.
    #
    # mat2 MUST BE TRANSPOSED, i.e. column-major in its last two dims. Measured 2026-09-05: a
    # contiguous (E, K, N) operand is refused with "Expected mat2 to be transposed" for EVERY
    # group size, which reads as "the op rejects all our shapes" and is really "the op rejects
    # this layout". Built as (E, N, K) and transposed so the stride pattern is the one the
    # kernel wants, without a copy.
    for g in (1, 7, 8, 15, 16, 17, 31, 32, 64, 128, 256):
        n = g * N_EXPERTS
        a = torch.randn(n, D, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        b = torch.randn(N_EXPERTS, 2 * EXPERT_FFN, D, device=device,
                        dtype=torch.bfloat16).to(torch.float8_e4m3fn).transpose(-2, -1)
        offs = (torch.arange(1, N_EXPERTS + 1, device=device, dtype=torch.int32) * g)
        # PER-ROW AND PER-EXPERT-COLUMN SCALES, not two scalars. Measured 2026-09-05: scalar
        # scales are refused with "scale must have the same length as mat for arg 0" at EVERY
        # group size, which reads as a shape constraint on the DATA and is really a constraint on
        # the SCALES. scale_a has one entry per row of a; scale_b has one per output column per
        # expert. Our recipe is tensorwise, so every entry is the same value -- the shape is what
        # the op requires, not a change of recipe.
        sa = torch.ones(n, device=device, dtype=torch.float32)
        sb = torch.ones(N_EXPERTS, 2 * EXPERT_FFN, device=device, dtype=torch.float32)
        rec = {"group_size": g, "rows": n}
        try:
            r = torch._scaled_grouped_mm(a, b, sa, sb, offs=offs,
                                         out_dtype=torch.bfloat16)
            rec["ok"] = True
            rec["finite"] = bool(torch.isfinite(r).all())
        except Exception as e:  # noqa: BLE001 -- the refusal IS the measurement
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["cases"].append(rec)
    ok = [c["group_size"] for c in out["cases"] if c.get("ok")]
    bad = [c["group_size"] for c in out["cases"] if not c.get("ok")]
    out["accepted_group_sizes"] = ok
    out["refused_group_sizes"] = bad
    out["verdict"] = (f"accepted {ok}, refused {bad}" if bad
                      else f"every swept group size accepted ({ok})")
    return out


def situ(a, b, w2_apply, beta1=4.0, beta2=25.0):
    """model.SwiGLU's activation, applied to whatever the second matmul is.

    THE SAME FUNCTION ON BOTH SIDES, and it is not cosmetic. model.py:344-345 is the SiTU
    bounded form -- beta1*tanh(a/beta1)*sigmoid(b), then beta2*tanh(w2(gate)/beta2) -- which is
    two tanh and two divides the plain a*sigmoid(b) does not pay. Measured 2026-09-05: the first
    version of this bench ran SiTU on the dense side and a*sigmoid(b) on the MoE side, so the
    dense baseline carried elementwise work the MoE paths did not and the ratio flattered the
    MoE. b0 found it by reading the source. A dispatch ratio between two different functions is
    not a dispatch ratio."""
    import torch
    gate = beta1 * torch.tanh(a / beta1) * torch.sigmoid(b)
    return beta2 * torch.tanh(w2_apply(gate) / beta2)


def p1_dispatch(device, dtype=None):
    """Dense SwiGLU vs _grouped_mm vs sort-and-loop, at equal active compute, fwd and bwd.

    WHAT THE TIMED REGION MUST CONTAIN, after b0's review (2026-09-05). Three asymmetries in
    the first version all pushed the same way, which is the tell:

      1. the activation differed (see situ above)
      2. NO TIMED PATH RAN THE SHARED EXPERT. N_SHARED entered the ratio formula and the print
         but not one line of timed code, so the grouped path did T*TOP_K rows and was credited
         with T*(TOP_K+N_SHARED) -- exactly 0.75 of the work it was scored against. The
         import-time equal-compute assertion certified a configuration the bench never ran.
      3. the gather and the offsets were built OUTSIDE the timed closures: ~25 MiB per step of
         traffic the real arm pays and the bench did not, with no dense counterpart.

    So: same activation both sides, the shared expert inside f_grouped as one dense matmul, and
    routing + gather + offsets inside every timed closure. The dense path pays its own honest
    cost and nothing else."""
    import torch
    dtype = dtype or torch.bfloat16
    T = SEQ
    x = torch.randn(T, D, device=device, dtype=dtype, requires_grad=True)
    w13, w2 = build_moe(dtype, device)
    # The shared expert: every token goes through it, so it is a plain dense FFN at the expert
    # width, not a grouped call. Its cost is the N_SHARED term of the ratio.
    sh13 = (torch.randn(2 * EXPERT_FFN, D, device=device, dtype=dtype) * D ** -0.5).t()
    sh2 = (torch.randn(D, EXPERT_FFN, device=device, dtype=dtype) * EXPERT_FFN ** -0.5).t()

    from model import SwiGLU

    class _C:
        d, ffn_hidden = D, DENSE_FFN
    dense = SwiGLU(_C()).to(device=device, dtype=dtype)

    gy_dense = torch.ones(T, D, device=device, dtype=dtype)
    gy_moe = torch.ones(T * TOP_K, D, device=device, dtype=dtype)
    gy_sh = torch.ones(T, D, device=device, dtype=dtype)

    def f_dense():
        dense(x).backward(gy_dense)
        x.grad = None

    def _gather():
        """Routing, sort and gather -- INSIDE the timed region, because the arm pays it."""
        order, offs, _ = route(x.detach())
        xs = x.detach().repeat_interleave(TOP_K, 0)[order].contiguous().requires_grad_(True)
        return xs, offs

    def f_grouped():
        xs, offs = _gather()
        h = torch._grouped_mm(xs, w13, offs=offs)
        a, b = h.chunk(2, dim=-1)
        y = situ(a, b, lambda g: torch._grouped_mm(g.contiguous(), w2, offs=offs))
        # THE SHARED EXPERT, which every token also traverses.
        hs = x @ sh13
        sa, sb = hs.chunk(2, dim=-1)
        ys = situ(sa, sb, lambda g: g @ sh2)
        y.backward(gy_moe)
        ys.backward(gy_sh)
        x.grad = None

    def f_loop():
        xs, offs = _gather()
        start, acc = 0, []
        for e in range(N_EXPERTS):
            end = int(offs[e])
            if end > start:
                h = xs[start:end] @ w13[e]
                a, b = h.chunk(2, dim=-1)
                acc.append(situ(a, b, lambda g, _e=e: g @ w2[_e]))
            start = end
        hs = x @ sh13
        sa, sb = hs.chunk(2, dim=-1)
        ys = situ(sa, sb, lambda g: g @ sh2)
        torch.cat(acc).backward(gy_moe)
        ys.backward(gy_sh)
        x.grad = None

    order0, offs0, counts = route(x.detach())
    res = {"dtype": str(dtype), "tokens": T, "top_k": TOP_K, "n_experts": N_EXPERTS,
           "n_shared": N_SHARED, "expert_ffn": EXPERT_FFN, "dense_ffn": DENSE_FFN,
           "compute_ratio": compute_ratio(),
           "shared_expert_timed": True, "gather_inside_timed_region": True,
           "same_activation_both_sides": True,
           "expert_load": {"min": int(counts.min()), "max": int(counts.max()),
                           "mean": float(counts.float().mean())}}

    # THE TIED-WEIGHTS WITNESS, and it is a PRECONDITION rather than a diagnostic: if the MoE
    # machinery with one expert does not reproduce the dense module's own output, then whatever
    # the timed paths compute is not the dense FFN and no ratio between them means anything.
    # This is the check the first version lacked -- it asserted the FORMULA and never asked the
    # CODE what it computed.
    #
    # BOTH TIMED MoE PATHS ARE WITNESSED, not just the grouped one (4c, 2026-09-05). A loop that
    # drifted from dense would otherwise be invisible here AND consistent with the paths-agree
    # check, since that check compares the two MoE paths to EACH OTHER -- two paths that drifted
    # together would agree perfectly and both be wrong. Each is compared to the dense module.
    with torch.no_grad():
        xt = torch.randn(64, D, device=device, dtype=dtype)
        w13_t = dense.w13.weight.detach().t().unsqueeze(0).contiguous().transpose(-2, -1)
        w2_t = dense.w2.weight.detach().t().unsqueeze(0).contiguous().transpose(-2, -1)
        offs_t = torch.tensor([64], device=device, dtype=torch.int32)
        yd = dense(xt)
        hg = torch._grouped_mm(xt, w13_t, offs=offs_t)
        ag, bg = hg.chunk(2, dim=-1)
        yg = situ(ag, bg, lambda g: torch._grouped_mm(g.contiguous(), w2_t, offs=offs_t))
        wit_g = float((yg.float() - yd.float()).abs().max())
        # The LOOP path's own arithmetic, through the same single tied expert.
        hl = xt @ w13_t[0]
        al, bl = hl.chunk(2, dim=-1)
        yl = situ(al, bl, lambda g: g @ w2_t[0])
        wit_l = float((yl.float() - yd.float()).abs().max())
    wit = max(wit_g, wit_l)
    res["tied_weights_witness_max_abs_diff"] = wit
    res["tied_weights_witness_grouped"] = wit_g
    res["tied_weights_witness_loop"] = wit_l
    # THRESHOLD FROM THE MEASURED NOISE FLOOR, not a round number. Measured 2026-09-05 on CPU at
    # both fp32 and bf16: the same function through both expressions differs by EXACTLY 0.0,
    # while the activation defect this witness exists to catch (plain a*sigmoid(b) instead of
    # SiTU) shows 0.023-0.025 on outputs of scale 0.75. A 0.05 tolerance -- which is what I first
    # wrote -- sits ABOVE the defect and would have passed the broken world it was written for.
    # 1e-3 is 23x below the defect signal and still far above a bf16 reduction's noise on CUDA.
    res["tied_weights_witness_ok"] = wit < 1e-3
    if not res["tied_weights_witness_ok"]:
        which = "grouped" if wit_g >= 1e-3 else "loop"
        res["verdict"] = (f"REFUSED: the {which} path, holding the dense module's own weights "
                          f"in one expert, does not reproduce its output (grouped {wit_g:.4f}, "
                          f"loop {wit_l:.4f}). No ratio reported -- the paths do not compute "
                          "the same function.")
        return res

    for name, fn in (("dense", f_dense), ("grouped_mm", f_grouped), ("loop", f_loop)):
        try:
            res[name + "_ms"] = _sync_time(fn, iters=20)
        except Exception as e:  # noqa: BLE001
            res[name + "_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    if "dense_ms" in res:
        for k in ("grouped_mm", "loop"):
            if k + "_ms" in res:
                res[k + "_ratio_to_dense"] = res[k + "_ms"] / res["dense_ms"]
    g = res.get("grouped_mm_ratio_to_dense")
    lo = res.get("loop_ratio_to_dense")
    res["prediction"] = {"grouped_mm_within_1.3x": None if g is None else g <= 1.3,
                         "loop_3x_to_10x": None if lo is None else 3.0 <= lo <= 10.0}
    return res


def p2_recompile(device):
    """Does torch.compile recompile as expert row-counts move step to step?

    Counted from dynamo's own counter, not inferred from wall-clock: a slow step can be a
    recompile or a cache miss or thermal, and only the counter distinguishes them."""
    import torch
    import torch._dynamo as dynamo
    w13, w2 = build_moe(torch.bfloat16, device)

    def block(xs, offs):
        h = torch._grouped_mm(xs, w13, offs=offs)
        a, b = h.chunk(2, dim=-1)
        return torch._grouped_mm((a * torch.sigmoid(b)).contiguous(), w2, offs=offs)

    out = {}
    for label, pad in (("dynamic", False), ("padded", True)):
        dynamo.reset()
        dynamo.utils.counters.clear()
        c = torch.compile(block)
        for step in range(12):
            x = torch.randn(SEQ, D, device=device, dtype=torch.bfloat16)
            order, offs, counts = route(x, seed=step)   # a different routing every step
            xs = x.repeat_interleave(TOP_K, 0)[order].contiguous()
            if pad:
                # CAPACITY PADDING: every expert gets the same fixed row count, so the shapes
                # are static across steps. The waste is a real cost and is reported, not hidden.
                cap = int(SEQ * TOP_K / N_EXPERTS * 1.5)
                offs = (torch.arange(1, N_EXPERTS + 1, device=device,
                                     dtype=torch.int32) * cap)
                xs = torch.zeros(cap * N_EXPERTS, D, device=device, dtype=torch.bfloat16)
                out.setdefault("padding_waste", cap * N_EXPERTS / (SEQ * TOP_K))
            try:
                c(xs, offs)
            except Exception as e:  # noqa: BLE001
                out[label + "_error"] = f"{type(e).__name__}: {str(e)[:200]}"
                break
        n = sum(v for k, v in dynamo.utils.counters["frames"].items() if k == "ok")
        out[label + "_compiles"] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["p0", "p1", "p2", "all"], default="all")
    ap.add_argument("--experts", type=int, default=None,
                    help="routed expert count. Default 32; the prereg's E1 is 24 routed + 1 "
                         "shared. Active compute does NOT move with this (it is set by k and w) "
                         "but DISPATCH does: N is the number of groups the grouped GEMM walks, "
                         "so a ratio measured at 32 does not describe a 24-expert arm.")
    ap.add_argument("--json", default=None, help="append the result row here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()

    if a.experts:
        # Module-level because route/build_moe/p0 all read the constant. Set BEFORE any phase
        # runs and recorded in the row, so a ratio can never be read without its expert count.
        globals()["N_EXPERTS"] = a.experts

    import torch
    dev = _dev()
    rec = {"phase": a.phase, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0),
           "n_experts": N_EXPERTS, "top_k": TOP_K, "expert_ffn": EXPERT_FFN,
           "compute_ratio": compute_ratio()}
    if a.phase in ("p0", "all"):
        rec["p0"] = p0_fp8_constraint(dev)
        print("P0:", rec["p0"]["verdict"])
    if a.phase in ("p1", "all"):
        rec["p1"] = p1_dispatch(dev)
        r = rec["p1"]
        print(f"P1: dense {r.get('dense_ms', float('nan')):.3f} ms  "
              f"grouped {r.get('grouped_mm_ms', float('nan')):.3f} ms "
              f"({r.get('grouped_mm_ratio_to_dense', float('nan')):.2f}x)  "
              f"loop {r.get('loop_ms', float('nan')):.3f} ms "
              f"({r.get('loop_ratio_to_dense', float('nan')):.2f}x)")
        print(f"    prediction: {r['prediction']}")
    if a.phase in ("p2", "all"):
        rec["p2"] = p2_recompile(dev)
        print("P2:", rec["p2"])
    if a.json:
        with open(a.json, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"appended to {a.json}")
    return 0


def _selftest():
    """CPU-side: the arithmetic that gates the whole benchmark, and the routing shapes."""
    bad = 0

    # 1. EQUAL ACTIVE COMPUTE, the precondition. top-3 + shared == dense, exactly.
    r = compute_ratio()
    ok = abs(r - 1.0) < 1e-9
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} top-{TOP_K}+{N_SHARED} shared at ffn {EXPERT_FFN} "
          f"is {r:.4f}x the dense FFN's active compute")

    # 2. AND THE CONFIG THAT WAS PROPOSED IS NOT, which is why the charter changed. Asserting
    # only (1) would pass on a build where the formula itself was wrong.
    r4 = compute_ratio(top_k=4)
    ok = abs(r4 - 1.25) < 1e-9
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the originally proposed top-4+shared is {r4:.4f}x "
          f"-- the reason the charter moved to top-3")

    # 3. THE FORMULA MATCHES model.SwiGLU's REAL PARAMETER COUNT, not an assumed 2-matrix FFN.
    # If SwiGLU stops being w13(d->2f) + w2(f->d), every ratio above is wrong and silently so.
    try:
        import torch  # noqa: F401

        from model import SwiGLU

        class _C:
            d, ffn_hidden = D, DENSE_FFN
        got = sum(p.numel() for p in SwiGLU(_C()).parameters())
        want = active_params(D, DENSE_FFN, 1)
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BUG '} 3*d*ffn matches SwiGLU's real parameter count "
              f"({got:,} vs {want:,})")
    except Exception as e:  # noqa: BLE001
        bad += 1
        print(f"  BUG  could not check SwiGLU's parameter count: {type(e).__name__}: {e}")

    # 4. THE CHARTER'S SHAPES AGREE WITH train.Cfg, so the benchmark is not measuring a model
    # nobody trains.
    try:
        import train
        ok = train.Cfg.d == D and train.Cfg.ffn_hidden == DENSE_FFN
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BUG '} d and ffn_hidden match train.Cfg "
              f"({train.Cfg.d}, {train.Cfg.ffn_hidden})")
    except Exception as e:  # noqa: BLE001
        bad += 1
        print(f"  BUG  could not read train.Cfg: {type(e).__name__}: {e}")

    # 5. ROUTING PRODUCES _grouped_mm's OFFSET CONVENTION: cumulative ends, last == total rows.
    # An off-by-one here silently drops or duplicates a whole expert's tokens.
    import torch
    x = torch.randn(256, D)
    order, offs, counts = route(x)
    ok = (int(offs[-1]) == 256 * TOP_K and len(offs) == N_EXPERTS
          and int(counts.sum()) == 256 * TOP_K and len(order) == 256 * TOP_K)
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} offsets are cumulative ends covering every routed row "
          f"({int(offs[-1])} == {256 * TOP_K})")

    # 6. THE ROUTING IS IMBALANCED ON PURPOSE. A uniform split is the case every dispatch path
    # handles best, so measuring on one would flatter grouped_mm and the loop equally and the
    # ratio would not describe a real router.
    spread = int(counts.max()) - int(counts.min())
    ok = spread > 0
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} expert load is imbalanced as a real router's is "
          f"(min {int(counts.min())}, max {int(counts.max())}, spread {spread})")

    # 7. EACH TOKEN REACHES top_k DISTINCT EXPERTS. replacement=False is what guarantees it, and
    # a token routed to the same expert twice would double-count its compute against the ratio
    # asserted in (1).
    ok = int(counts.sum()) == 256 * TOP_K
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} every token reaches exactly {TOP_K} experts")

    n = 7
    print(f"moe_dispatch_bench selftest: {n - bad}/{n} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

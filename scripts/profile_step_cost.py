#!/usr/bin/env python3
"""Host-side per-step cost on the 200M config, timing train.py's OWN step (de sprint).

fb asked for four numbers with their measurement config: loader wait, save, val, NCCL share.
The run's own log gives one -- wall time per 10-step window (train.py:2736-2760) -- so a
decomposition needs instrumentation train.py does not have. This measures from outside
instead of editing a file that is about to be relaunched.

THE STEP IS train.py's, NOT A REWRITE (fb's condition, 2026-09-02). The first version of this
file timed an index_select, a torch.save, a val pass and a bare all_reduce -- no optimizer, no
FLCE, no FP8, no compile. Four numbers describing a different program. What is assembled here
is the real thing, in train.py's own order (:2322 model, :2445 DDP, :2450-2491 compile):

    HybridLM(Cfg)                     :2322, the same class
    build_optimizers(raw, Cfg)        :1092, Muon for 2D + AdamW for the rest
    LigerFusedLinearCrossEntropyLoss  the same loss with the same SOFTCAP
    DDP(bucket_cap_mb, ...)           :2445, the same wrapper and flags
    torch.compile(dynamic=False)      :2491, when Cfg.compile and amp
    autocast(bfloat16) + --fp8        run_ddp.sh:44 passes --fp8; the flag is honoured here

Shape flags come from the p200m launch line (e19eeb7): d1024 L12 heads 8 ffn 3072, batch 32,
accum 1, --no-grad_ckpt, seq 4096 from Cfg. Anything not on that line is Cfg's default, which
is the point of naming the commit rather than restating the values.

WHAT EACH NUMBER IS:
  step_total    a full step: loader + forward + backward + clip + optimizer. The denominator
                for every share below.
  loader_wait   index_select into pinned memory plus the H2D copy, bracketed by
                cuda.synchronize -- the copy is async and a wall-clock read there returns
                launch cost, not transfer.
  save          torch.save of this model's real state dict, at the run's --save_every cadence.
  val           train.validate at the run's own Cfg.val_batches, on the same shapes.
  nccl_floor    an all_reduce of the whole gradient set, timed alone. A FLOOR on the exposed
                cost, not the overlap: DDP overlaps reduction with backward, so the real cost
                is at most this. The overlap is not separable from outside the loop, and
                calling this "the NCCL share" would be a number without its basis.

    torchrun --nproc_per_node=8 scripts/profile_step_cost.py --mix data/mix_200m_4b.json \
        --dim 1024 --layers 12 --heads 8 --ffn_hidden 3072 --batch 32 --accum 1 --fp8
    python3 scripts/profile_step_cost.py --selftest   # arithmetic + reporting rules, no card

# restartable: writes one JSON line at the end and a scratch checkpoint it removes; an
# interrupt costs the steps already timed and leaves no state.
"""

import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# The p200m launch line, e19eeb7. Kept as data so --selftest can assert the defaults match it
# without a card: a profiler configured differently from the run measures another program, and
# that is exactly fb's condition.
P200M = {"dim": 1024, "layers": 12, "heads": 8, "ffn_hidden": 3072, "batch": 32, "accum": 1,
         "grad_ckpt": False, "fp8": True, "mix": "data/mix_200m_4b.json"}


def _stats(xs):
    """(median, min, max, n) in ms. Median, not mean: one page fault or one allocator growth
    skews a mean over 20 samples, and the question is what a typical step costs."""
    if not xs:
        return None
    ms = [x * 1000 for x in xs]
    return {"median_ms": round(statistics.median(ms), 2), "min_ms": round(min(ms), 2),
            "max_ms": round(max(ms), 2), "n": len(ms)}


def fmt_row(name, st, world, total=None):
    if st is None:
        why = ("world=1, so there is no reduction to time -- run under torchrun"
               if name == "nccl_floor" else "not measured in this run")
        return f"{name:12s} NOT MEASURED ({why})"
    share = ""
    if total and name != "step_total":
        share = f"  = {100 * st['median_ms'] / total:.1f}% of a step"
    return (f"{name:12s} median {st['median_ms']:9.2f} ms  "
            f"[{st['min_ms']:.2f}..{st['max_ms']:.2f}]  n={st['n']}{share}")


def _selftest():
    """Known answers for the arithmetic, the reporting rules, and the shape. No card."""
    bad = 0
    s = _stats([0.010, 0.012, 0.011, 0.100])
    ok = s["median_ms"] == 11.5 and s["max_ms"] == 100.0 and s["n"] == 4
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} median is robust to one outlier ({s})")

    ok = _stats([]) is None
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} no samples reports None, never 0")

    line = fmt_row("nccl_floor", None, 1)
    ok = "NOT MEASURED" in line and "0.0" not in line
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} an unmeasured phase says so, never prints 0")

    line = fmt_row("save", {"median_ms": 500.0, "min_ms": 1.0, "max_ms": 2.0, "n": 3}, 8,
                   total=1000.0)
    ok = "50.0% of a step" in line
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} a share is reported against the measured step total")

    # fb's condition, as an assertion: the step this file builds must be the run's step.
    # Checked against train.py's source, so a change to either side shows up here.
    with open(os.path.join(ROOT, "train.py"), encoding="utf-8") as fh:
        src = fh.read()
    for frag, why in (
        ("raw_model = HybridLM(Cfg).to(device)", "the model class"),
        ("optimizers = build_optimizers(", "the optimizer construction"),
        ("LigerFusedLinearCrossEntropyLoss(ignore_index=-100, softcap=SOFTCAP)", "the loss"),
        ("model = torch.compile(model, dynamic=False", "the compile call"),
    ):
        ok = frag in src
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BUG '} train.py still uses {why}"
              + ("" if ok else f" -- {frag!r} is gone, re-read before trusting a number"))

    ok = P200M["fp8"] and not P200M["grad_ckpt"] and P200M["batch"] == 32
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the recorded p200m shape matches e19eeb7's line")

    # Every train.* name this file resolves must exist, checked by AST rather than by import
    # (train pulls in CUDA-only modules). Three of my drafts named symbols that do not exist
    # -- build_model, an EOS_ID constant, build_tokenizer(is_main) -- and each would have died
    # in setup on a card, after the queue. Names that resolve only at runtime are listed
    # separately with the reason, because the scan cannot see them and silence is not proof.
    import ast

    tree = ast.parse(src)
    top = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top.add(n.name)
        elif isinstance(n, ast.Assign):
            top.update(t.id for t in n.targets if isinstance(t, ast.Name))
    need = ("Cfg", "HybridLM", "build_optimizers", "build_tokenizer", "build_mix",
            "setup_ddp", "validate", "doc_cu_seqlens", "SOFTCAP", "vocab_fingerprint",
            "VOCAB_ID")
    absent = [n for n in need if n not in top]
    ok = not absent
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} every train.* name resolves"
          + ("" if ok else f" -- MISSING {absent}, this file would die in setup on a card"))
    # Runtime-only: imported inside a try at train.py:129, so not a top-level binding an AST
    # scan lists. Asserted by its import line instead.
    ok = "from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss" in src
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} FLCE is still imported into train's namespace")
    n = 6 + 3 + 2
    print(f"profile_step_cost selftest: {n - bad}/{n} pass")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mix", default=P200M["mix"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5,
                    help="steps discarded before timing: compile traces on the first few")
    for k in ("dim", "layers", "heads", "ffn_hidden", "batch", "accum"):
        ap.add_argument(f"--{k}", type=int, default=P200M[k])
    ap.add_argument("--fp8", action="store_true", default=P200M["fp8"])
    ap.add_argument("--grad_ckpt", action="store_true", default=P200M["grad_ckpt"])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.selftest:
        return _selftest()

    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP

    import train

    # Cfg first, before anything reads it: the model, the plan and the loss all take their
    # shape from it, and setting it after construction would time a different model than the
    # flags say.
    for k in ("dim", "layers", "heads", "ffn_hidden", "batch", "accum"):
        if hasattr(train.Cfg, k):
            setattr(train.Cfg, k, getattr(a, k))
    train.Cfg.grad_ckpt = a.grad_ckpt

    ddp, rank, world, local = train.setup_ddp()
    is_main = rank == 0
    torch.cuda.set_device(local if ddp else 0)
    dev = "cuda"

    tok = train.build_tokenizer([])
    # build_mix now RAISES when VOCAB_ID is unset (de-23, this same window), so setting it is
    # not optional here. No hasattr fallback: leaving it None would hit that guard, and the
    # fallback would read as "this is fine on hosts without vocab_fingerprint" when in fact
    # there are none -- the function is at train.py's top level.
    train.VOCAB_ID = train.vocab_fingerprint(tok)
    tr, va = train.build_mix(os.path.join(ROOT, a.mix), tok, is_main, ddp, rank, world)
    seqs = (tr[0] if train.Cfg.fone else tr).long()
    X, Y = seqs[:, :-1], seqs[:, 1:]
    vseqs = (va[0] if train.Cfg.fone else va).long()
    Xva, Yva = vseqs[:, :-1], vseqs[:, 1:]

    raw = train.HybridLM(train.Cfg).to(dev)
    optimizers = train.build_optimizers(raw, train.Cfg)
    model = raw
    if ddp:
        model = DDP(model, device_ids=[local], bucket_cap_mb=25, gradient_as_bucket_view=True,
                    static_graph=True)
    amp = True
    if train.Cfg.compile and amp:
        torch._dynamo.config.cache_size_limit = max(64, 2 * train.Cfg.layers + 8)
        model = torch.compile(model, dynamic=False)

    # train.LigerFusedLinearCrossEntropyLoss is a module-level import inside a try (train.py:129,
    # None at :131 when liger is absent), so it resolves at runtime even though an AST scan of
    # train.py does not list it. Refuse rather than substitute another loss: FLCE with a softcap
    # is what the run computes, and a different loss changes the backward being timed.
    if train.LigerFusedLinearCrossEntropyLoss is None:
        print("FAIL: liger_kernel is absent, so the run's FLCE loss cannot be built here. "
              "Timing another loss would describe another program.", file=sys.stderr)
        return 1
    flce = train.LigerFusedLinearCrossEntropyLoss(ignore_index=-100, softcap=train.SOFTCAP)
    B, SEQ = train.Cfg.batch, train.Cfg.seq
    xb_pin = torch.empty((B, SEQ), dtype=X.dtype).pin_memory()
    yb_pin = torch.empty((B, SEQ), dtype=Y.dtype).pin_memory()
    # From the tokenizer, exactly as train.py:2256 does it. My first version fell back to a
    # literal 1 behind a hasattr -- train has no EOS_ID module constant, so the fallback was
    # the only branch, and a wrong eos id changes the document mask and therefore the step
    # being timed. A default that is always taken is not a default.
    eos = tok.token_to_id("<eos>")
    assert eos is not None, "the tokenizer has no <eos>; the document mask would be wrong"

    loader, steps, nccl = [], [], []
    n_par = sum(p.numel() for p in raw.parameters())
    if is_main:
        print(f"built {n_par / 1e6:.2f}M params, compile={train.Cfg.compile and amp}, "
              f"fp8={a.fp8}, grad_ckpt={a.grad_ckpt}, warmup {a.warmup} steps discarded",
              flush=True)

    for st in range(a.steps + a.warmup):
        idx = torch.arange(st * B, st * B + B) % len(X)
        torch.cuda.synchronize()
        t_step = time.perf_counter()
        torch.index_select(X, 0, idx, out=xb_pin)
        torch.index_select(Y, 0, idx, out=yb_pin)
        xb = xb_pin.to(dev, non_blocking=True)
        yb = yb_pin.to(dev, non_blocking=True)
        torch.cuda.synchronize()
        t_fwd = time.perf_counter()

        cu = train.doc_cu_seqlens(xb, eos) if train.Cfg.doc_mask else None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            hidden, _ = model(xb, yb, cu, None)
        Bt, Tt, D = hidden.shape
        weight = raw.head.weight[: raw.cfg.vocab]
        loss = flce(weight, hidden.to(weight.dtype).reshape(-1, D), yb.reshape(-1))
        (loss / train.Cfg.accum).backward()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), train.Cfg.clip)
        for opt in optimizers:
            opt.step()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if st >= a.warmup:
            loader.append(t_fwd - t_step)
            steps.append(time.perf_counter() - t_step)

        if ddp and st >= a.warmup:
            g = torch.zeros(n_par, device=dev)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.distributed.all_reduce(g)
            torch.cuda.synchronize()
            nccl.append(time.perf_counter() - t0)
            del g

    saves = []
    if is_main:
        for _ in range(3):
            t0 = time.perf_counter()
            torch.save({"model": raw.state_dict(), "cfg": vars(train.Cfg),
                        "opt": [o.state_dict() for o in optimizers]}, "/tmp/_prof_ckpt.pt")
            saves.append(time.perf_counter() - t0)
        os.remove("/tmp/_prof_ckpt.pt")

    vals = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        train.validate(model, raw, Xva, Yva, train.Cfg.batch, dev, torch.bfloat16,
                       max_batches=train.Cfg.val_batches)
        torch.cuda.synchronize()
        vals.append(time.perf_counter() - t0)

    rec = {"mix": a.mix, "world": world, "params_m": round(n_par / 1e6, 2),
           "shape": "step = e19eeb7's p200m launch line", "batch": B, "accum": train.Cfg.accum,
           "seq": SEQ, "layers": train.Cfg.layers, "dim": train.Cfg.dim,
           "fp8": a.fp8, "grad_ckpt": a.grad_ckpt,
           "compile": bool(train.Cfg.compile and amp), "steps_timed": len(steps),
           "step_total": _stats(steps), "loader_wait": _stats(loader),
           "nccl_floor": _stats(nccl) if ddp else None,
           "save": _stats(saves) if is_main else None, "val": _stats(vals)}
    if is_main:
        tot = rec["step_total"]["median_ms"] if rec["step_total"] else None
        print(f"\n200M host-side per-step cost  (mix {a.mix}, world {world}, "
              f"{rec['params_m']}M, batch {B} x accum {train.Cfg.accum} x seq {SEQ}, "
              f"fp8={a.fp8} grad_ckpt={a.grad_ckpt} compile={rec['compile']})")
        for k in ("step_total", "loader_wait", "nccl_floor", "save", "val"):
            print("  " + fmt_row(k, rec[k], world, tot))
        if a.json:
            with open(a.json, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

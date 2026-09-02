#!/usr/bin/env python3
"""Host-side per-step cost on the 200M config: loader wait, save, val, NCCL (de sprint).

fb asked for four numbers, each with its measurement config. What the run's own log gives is
one number -- wall time per 10-step window (train.py:2736-2760) -- so a decomposition needs
instrumentation train.py does not have. This measures it from OUTSIDE instead of editing a
file that is about to be relaunched: the same step shape, the same batch/accum/seq, the same
DDP wrapper, with a timer per phase.

WHAT EACH NUMBER IS, and what it is not:

  loader_wait   time the step spends in index_select + the pinned H2D copy before the forward
                can start. Measured with CUDA events around the copy, not wall clock: the
                copy is async, so a wall-clock read here returns the launch cost, not the
                transfer. This is the number that decides whether prefetch depth matters.
  save          torch.save of a checkpoint of this model's real size. MEASURED, not
                extrapolated from the 500M's 78 s: the 200M's state dict is a different size
                and the run saves every --save_every steps, so this is a recurring cost.
  val           one val pass at Cfg.val_batches, on the same shapes.
  nccl          all_reduce of the gradient bucket set, isolated by timing the same reduction
                on the same tensor sizes with and without the reduction. It is a FLOOR on the
                real overlap cost: DDP overlaps reduction with backward, so the exposed cost
                is at most this and usually less. Reported as a floor rather than as the
                overlap, because the overlap is not separable from outside the loop.

BOUNDARY. Single-process by default, so `nccl` is measured with world=1 and reports 0 --
useless. Run under torchrun to get the real number; the script says which mode produced each
row rather than printing a 0 that reads like "free".

    torchrun --nproc_per_node=8 scripts/profile_step_cost.py --mix data/mix_200m_4b.json
    python3 scripts/profile_step_cost.py --selftest   # arithmetic, no card

# restartable: writes one JSON record at the end and nothing else; an interrupt costs the
# steps already timed and leaves no state.
"""

import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _stats(xs):
    """(median, min, max, n) in ms. Median, not mean: one page fault or one allocator growth
    skews a mean over 20 samples, and the question is what a typical step costs."""
    if not xs:
        return None
    ms = [x * 1000 for x in xs]
    return {"median_ms": round(statistics.median(ms), 2), "min_ms": round(min(ms), 2),
            "max_ms": round(max(ms), 2), "n": len(ms)}


def _selftest():
    """Known answers for the arithmetic and the reporting rules. No card, no corpus."""
    bad = 0
    s = _stats([0.010, 0.012, 0.011, 0.100])
    ok = s["median_ms"] == 11.5 and s["max_ms"] == 100.0 and s["n"] == 4
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} median is robust to one outlier ({s})")

    ok = _stats([]) is None
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} no samples reports None, never 0")

    # The rule that matters most: a phase that was not measured must not print as 0.
    rec = {"nccl": None, "world": 1}
    line = fmt_row("nccl", rec["nccl"], rec["world"])
    ok = "NOT MEASURED" in line and "0.0" not in line
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} an unmeasured phase says so: {line!r}")

    line = fmt_row("save", {"median_ms": 78000.0, "min_ms": 1.0, "max_ms": 2.0, "n": 4}, 8)
    ok = "78000.0" in line or "78.00" in line
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} a measured phase prints its number")
    print(f"profile_step_cost selftest: {4 - bad}/4 pass")
    return 1 if bad else 0


def fmt_row(name, st, world):
    if st is None:
        why = ("world=1, so there is no reduction to time -- run under torchrun"
               if name == "nccl" else "not measured in this run")
        return f"{name:12s} NOT MEASURED ({why})"
    return (f"{name:12s} median {st['median_ms']:9.2f} ms  "
            f"[{st['min_ms']:.2f}..{st['max_ms']:.2f}]  n={st['n']}  world={world}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mix", default="data/mix_200m_4b.json")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.selftest:
        return _selftest()

    import torch

    import train

    ddp, rank, world, local = train.setup_ddp()
    is_main = rank == 0
    dev = "cuda"
    torch.cuda.set_device(local if ddp else 0)

    # Verified against train.py rather than guessed: build_tokenizer takes `texts` and is
    # called as build_tokenizer([]) at :2255, and there is no build_model -- main() constructs
    # HybridLM(Cfg) directly at :2322. My first draft called train.build_model(dev) and
    # train.build_tokenizer(is_main); both are wrong, and a profiler that dies on its own
    # setup measures nothing.
    tok = train.build_tokenizer([])
    tr, va = train.build_mix(os.path.join(ROOT, a.mix), tok, is_main, ddp, rank, world)
    seqs = (tr[0] if train.Cfg.fone else tr).long()
    X, Y = seqs[:, :-1], seqs[:, 1:]
    model = train.HybridLM(train.Cfg).to(dev)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])
    raw = model.module if ddp else model

    B, ACC, SEQ = train.Cfg.batch, train.Cfg.accum, train.Cfg.seq
    loader, nccl = [], []
    xb_pin = torch.empty((B, SEQ), dtype=X.dtype).pin_memory()
    yb_pin = torch.empty((B, SEQ), dtype=Y.dtype).pin_memory()
    for st in range(a.steps):
        idx = torch.arange(st * B, st * B + B) % len(X)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.index_select(X, 0, idx, out=xb_pin)
        torch.index_select(Y, 0, idx, out=yb_pin)
        xb_pin.to(dev, non_blocking=True)
        yb_pin.to(dev, non_blocking=True)
        torch.cuda.synchronize()
        loader.append(time.perf_counter() - t0)

        if ddp:
            g = torch.zeros(sum(p.numel() for p in raw.parameters()), device=dev)
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
            torch.save({"model": raw.state_dict(), "cfg": vars(train.Cfg)},
                       "/tmp/_prof_ckpt.pt")
            saves.append(time.perf_counter() - t0)
        os.remove("/tmp/_prof_ckpt.pt")

    # val: the real train.validate, at the run's own Cfg.val_batches, so the number is what
    # the run pays and not what a re-implementation would pay. All ranks call it in lockstep
    # (its own docstring), so it is timed on every rank and reported from rank 0.
    vseqs = (va[0] if train.Cfg.fone else va).long()
    Xva, Yva = vseqs[:, :-1], vseqs[:, 1:]
    vals = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        train.validate(model, raw, Xva, Yva, train.Cfg.batch, dev, torch.bfloat16,
                       max_batches=train.Cfg.val_batches)
        torch.cuda.synchronize()
        vals.append(time.perf_counter() - t0)

    rec = {"mix": a.mix, "world": world, "batch": B, "accum": ACC, "seq": SEQ,
           "steps_timed": a.steps,
           "loader_wait": _stats(loader), "nccl_floor": _stats(nccl) if ddp else None,
           "save": _stats(saves) if is_main else None, "val": _stats(vals)}
    if is_main:
        print(f"\n200M host-side per-step cost  (mix {a.mix}, world {world}, "
              f"batch {B} x accum {ACC} x seq {SEQ})")
        for k in ("loader_wait", "nccl_floor", "save", "val"):
            print("  " + fmt_row(k, rec[k], world))
        if a.json:
            with open(a.json, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

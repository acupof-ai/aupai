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

    # tok/s/gpu must count the WHOLE step -- batch * accum * seq -- and the peak must be reset
    # before the loop. Both read from main()'s AST, never by substring: a needle like
    # "tok_step = B * train.Cfg.accum * SEQ" written into this check's own table MATCHES ITSELF,
    # so the check stays green after the code is broken. That is the same self-satisfying shape
    # as the first fp8 check below, and it happened again here -- caught only because the
    # mutation harness reported GREEN on a real regression.
    #
    # The A/B's two arms differ ONLY in how batch*accum is split, so a rate over the micro-batch
    # reports the b8a4 arm at 4x its real throughput and picks the wrong config outright. And a
    # peak that includes setup carries the plan, the optimizers and the pinned buffers -- which
    # both arms pay identically -- so the number deciding "does this arm fit" would be mostly
    # cost that is not the arm's.
    import ast as _a0

    with open(os.path.abspath(__file__), encoding="utf-8") as fh:
        src_me = fh.read()
    _me_tree = _a0.parse(src_me)
    _me_main = next((n for n in _me_tree.body
                     if isinstance(n, _a0.FunctionDef) and n.name == "main"), None)
    tok_expr = reset_line = loop_line = None
    for n in _a0.walk(_me_main) if _me_main else ():
        if isinstance(n, _a0.Assign) and any(
                isinstance(t, _a0.Name) and t.id == "tok_step" for t in n.targets):
            tok_expr = _a0.unparse(n.value).replace(" ", "")
        elif isinstance(n, _a0.Call) and isinstance(n.func, _a0.Attribute) \
                and n.func.attr == "reset_peak_memory_stats":
            reset_line = n.lineno
        elif isinstance(n, _a0.For) and _a0.unparse(n.iter).replace(" ", "") \
                == "range(a.steps+a.warmup)":
            loop_line = n.lineno
    for ok, why, hint in (
        (tok_expr == "B*train.Cfg.accum*SEQ", "tok/s/gpu counts batch*accum*seq",
         f"tok_step is {tok_expr!r}; a rate over one micro-batch reports the b8a4 arm at 4x"),
        (reset_line is not None, "the peak is reset in main()",
         "the peak would carry setup both arms pay identically, not the step's own cost"),
        (None not in (reset_line, loop_line) and reset_line < loop_line,
         "the peak reset comes BEFORE the timed loop",
         f"reset at line {reset_line}, loop at {loop_line} -- resetting after measures nothing"),
    ):
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BUG '} {why}" + ("" if ok else f" -- {hint}"))

    # A loss recorded per step, warmup INCLUDED. The arms' equivalence check compares them step
    # by step, and they diverge earliest -- if at all -- in the steps the timing window drops.
    # AST again: the append must NOT sit inside the `if st >= a.warmup` gate.
    gated, appends = set(), []
    for n in _a0.walk(_me_main) if _me_main else ():
        if isinstance(n, _a0.If) and _a0.unparse(n.test).replace(" ", "") == "st>=a.warmup":
            gated.update(x.lineno for b in n.body for x in _a0.walk(b) if hasattr(x, "lineno"))
        elif isinstance(n, _a0.Call) and isinstance(n.func, _a0.Attribute) \
                and n.func.attr == "append" and _a0.unparse(n.func.value) == "losses":
            appends.append(n.lineno)
    ok = bool(appends) and not (set(appends) & gated)
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the loss is recorded before the warmup gate"
          + ("" if ok else " -- the arms' first steps, where a divergence is largest, are dropped"))

    # Every --flag this file promises to apply must name a REAL Cfg field, and every train.Cfg.X
    # it reads must exist. Both halves come from train.py's own class body, so a rename there
    # fails here rather than 8 minutes into a compile on four cards.
    #
    # This is the arm-a crash, and it had two layers. Cfg's width field is `d`, the CLI flag is
    # --dim, and the old loop skipped unknown names behind hasattr -- so --dim 1024 set NOTHING
    # silently, and only the record's train.Cfg.dim raised, after the run. A flag that configures
    # nothing while the record claims it did is two different programs wearing one number. The
    # torch.save crash fixed earlier was masking this one: both are in the same tail block.
    import ast as _a1

    with open(os.path.join(ROOT, "train.py"), encoding="utf-8") as fh:
        _tsrc = fh.read()
    _cfg_cls = next((n for n in _a1.parse(_tsrc).body
                     if isinstance(n, _a1.ClassDef) and n.name == "Cfg"), None)
    _cfg_fields = {t.id for n in (_cfg_cls.body if _cfg_cls else [])
                   if isinstance(n, _a1.Assign) for t in n.targets if isinstance(t, _a1.Name)}
    _nodes = list(_a1.walk(_me_main)) if _me_main else []
    _cfg_reads = {n.attr for n in _nodes
                  if isinstance(n, _a1.Attribute) and isinstance(n.value, _a1.Attribute)
                  and n.value.attr == "Cfg"}
    _missing_reads = sorted(r for r in _cfg_reads if r not in _cfg_fields)
    ok = bool(_cfg_fields) and not _missing_reads
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} every train.Cfg.X this file reads exists in Cfg"
          + ("" if ok else f" -- MISSING {_missing_reads}; it raises after the run, not before"))

    # The flag->field map must be literal pairs, and every field in it must be a real Cfg field.
    # Reading the pairs from the AST rather than restating them: a restated copy is green while
    # the code guesses `getattr(a, k)` into `setattr(Cfg, k)`, which is the bug being guarded.
    _pairs = set()
    for n in _nodes:
        if isinstance(n, _a1.For) and isinstance(n.iter, _a1.Tuple):
            for el in n.iter.elts:
                if isinstance(el, _a1.Tuple) and len(el.elts) == 2 \
                        and all(isinstance(x, _a1.Constant) for x in el.elts):
                    _pairs.add((el.elts[0].value, el.elts[1].value))
    _bad_fields = sorted({f for _, f in _pairs if f not in _cfg_fields})
    ok = bool(_pairs) and not _bad_fields and ("dim", "d") in _pairs
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} every --flag maps to a REAL Cfg field "
          f"({len(_pairs)} pairs)"
          + ("" if ok else f" -- {_bad_fields or 'no literal pairs found'}; --dim would set "
                           "nothing while the record claims it did"))

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

    # fp8 must be APPLIED, not just recorded. The first GPU attempt wrote fp8=True while
    # calling neither of train.py's two fp8 steps, so the model stayed fp32, every activation
    # doubled, and 32x4096 OOMed at 93 GiB of a 95 GiB card. Same shape as the EOS_ID default
    # that was always taken: a field describing a premise the code never established.
    #
    # Checked by walking main()'s AST, NOT by a substring scan of this file. The substring
    # version could not fail: the needle "train.convert_to_fp8_compute(raw)" sat in the
    # check's own data table, so it matched itself even after the real call was deleted --
    # a check whose subject includes the check is self-satisfying. Verified RED by deleting
    # the call from a copy of this file (/tmp/break_fp8.py).
    import ast as _ast

    with open(os.path.abspath(__file__), encoding="utf-8") as fh:
        me = fh.read()
    _main = next((n for n in _ast.parse(me).body
                  if isinstance(n, _ast.FunctionDef) and n.name == "main"), None)
    calls, casts, loops, dicts = set(), set(), [], []
    for n in _ast.walk(_main) if _main else ():
        if isinstance(n, _ast.Call):
            f = n.func
            if isinstance(f, _ast.Attribute):
                calls.add(f.attr)
                if f.attr == "to":
                    casts.update(_ast.unparse(x) for x in n.args)
        elif isinstance(n, _ast.For):
            loops.append(_ast.unparse(n.iter))
        elif isinstance(n, _ast.Dict):
            dicts.append({_ast.unparse(k): _ast.unparse(v)
                          for k, v in zip(n.keys, n.values) if k is not None})
    # The record's fp8 field must carry the APPLIED value (the local `fp8`), not the requested
    # flag (`a.fp8`). Read from the AST, not by substring: a literal '"fp8": fp8,' in this
    # check's own data table would match itself, which is how the first version of the check
    # below stayed green after the real call was deleted.
    fp8_field = next((d["'fp8'"] for d in dicts if "'fp8'" in d), None)
    for ok, why, hint in (
        (_main is not None, "has a main() to check at all", "main() is gone"),
        ("convert_to_fp8_compute" in calls, "calls convert_to_fp8_compute (train.py:2020)",
         "fp8 would be recorded but the linears never converted"),
        (any("bfloat16" in c for c in casts), "casts the model to bf16 (train.py:2019)",
         "the model stays fp32 and every activation doubles -- this is what OOMed"),
        (fp8_field == "fp8", "records the APPLIED fp8, not the requested flag",
         f"the record's fp8 field is {fp8_field!r}; a.fp8 is the request, fp8 is what ran"),
    ):
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BUG '} main() {why}" + ("" if ok else f" -- {hint}"))

    # The cfg handed to torch.save must be a plain dict WITH the dunder keys dropped. Two
    # separate unpicklables live in a class's vars(): the mappingproxy itself, and the
    # __dict__/__weakref__ getset_descriptors inside it -- so `dict(vars(cls))` still raises,
    # and the underscore filter is load-bearing rather than cosmetic. This selftest found that
    # second layer; the first draft of the fix asserted only dict() and would have shipped a
    # save that raises exactly as before. train.py:964 does not hit either half because it
    # already writes {k: v for k, v in cfg.items() if not k.startswith("_")}; copying train.py's
    # save shape while dropping that comprehension is the whole bug.
    #
    # Two halves, because either alone passes for the wrong reason: pickle actually refusing
    # both forms and accepting the filtered one (the known answer -- if a torch/python version
    # starts pickling proxies, the hazard is gone and the AST half should be deleted with it),
    # and main() not passing a bare vars() as the save's cfg (what would regress).
    import pickle as _pickle

    class _C:
        a = 1

    def _refuses(obj):
        try:
            _pickle.dumps(obj)
            return False
        except TypeError:
            return True

    filtered = {k: v for k, v in dict(vars(_C)).items() if not k.startswith("_")}
    ok = (_refuses(vars(_C)) and _refuses(dict(vars(_C)))
          and _pickle.loads(_pickle.dumps(filtered)) == {"a": 1})
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} pickle refuses vars(cls) AND dict(vars(cls)); the "
          "underscore filter is what makes it picklable"
          + ("" if ok else " -- the hazard is gone; drop the AST half too"))

    save_cfgs = [d.get("'cfg'") for d in dicts if "'cfg'" in d]
    ok = bool(save_cfgs) and not any(c.startswith("vars(") or c.startswith("dict(vars(")
                                     for c in save_cfgs)
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} main() saves a filtered plain dict, not vars(train.Cfg)"
          + ("" if ok else f" -- cfg is {save_cfgs}; torch.save raises after the run finishes"))

    # A step is `accum` micro-batches, not one. The first version ran a single forward per
    # timed step, so step_total did not depend on accum at all -- and the b32a1-vs-b16a2 A/B
    # is a comparison whose ONLY variable is accum, so the b16a2 arm would have reported half
    # its work and won. Checked structurally: some loop in main() must iterate over Cfg.accum,
    # and the DDP no_sync that train.py:2246 applies to every micro-batch but the last must be
    # called, or the high-accum arm all-reduces accum times per step and pays what the run
    # does not.
    for ok, why, hint in (
        (any(it.replace(" ", "") == "range(train.Cfg.accum)" for it in loops),
         "loops over range(Cfg.accum) micro-batches",
         f"no loop in main() iterates range(Cfg.accum) (loops: {loops}); step_total would be "
         "independent of accum, deciding the b32a1/b16a2 A/B backwards"),
        ("no_sync" in calls, "calls no_sync for all but the last micro-batch (train.py:2246)",
         "DDP would all-reduce accum times per step, inflating the high-accum arm"),
    ):
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BUG '} the timed step {why}" + ("" if ok else f" -- {hint}"))

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
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            # A RE-EXPORTED name is a top-level binding too. Missing this branch made the scan
            # report HybridLM and SOFTCAP absent the moment b0-8 moved them to model.py and
            # re-exported them from train.py -- `train.HybridLM` still resolves at runtime, so
            # the scan was answering "is it DEFINED here" while claiming to answer "does it
            # RESOLVE here". Those differ exactly when a module re-exports, which is the whole
            # mechanism the split relies on.
            top.update(a.asname or a.name.split(".")[0] for a in n.names)
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
    n = 6 + 3 + 2 + 2 + 4 + 2 + 4 + 2
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
    ap.add_argument("--trace", nargs="?", const=True, default=None,
                    metavar="PATH",
                    help="export a Chrome trace of the last --trace-steps steps for "
                         "scripts/trace_classes.py (default runs/trace_step_cost.json)")
    ap.add_argument("--trace-steps", type=int, default=3,
                    help="how many steps the trace covers. Keep it small: record_shapes holds "
                         "per-kernel shape metadata for the whole window and OOMed the 206M "
                         "shape at 93.8/95.2 GiB over a longer one")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    # Argument validation BEFORE any setup. This guard first sat beside the profiler
    # construction, after setup_ddp and build_mix -- so a bad flag pair would have refused
    # only after ~2.5 min of loading 156 GB of token caches, on cards someone else is queued
    # for. A refusal that costs the queue what a run costs is not a refusal.
    if a.trace and a.trace_steps > a.steps:
        print(f"FAIL: --trace-steps {a.trace_steps} exceeds --steps {a.steps}. The trace window "
              f"must fit inside the timed steps, or prof.start() is never reached and the "
              f"export writes an empty trace -- a silent nothing that reads as 'traced'.",
              file=sys.stderr)
        return 1

    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP

    import train

    # Cfg first, before anything reads it: the model, the plan and the loss all take their
    # shape from it, and setting it after construction would time a different model than the
    # flags say.
    #
    # The CLI name is --dim; Cfg's field is `d`. The hasattr guard turned that mismatch into
    # SILENCE: --dim 1024 set nothing, the model built at Cfg.d whatever was asked, and the record
    # then read train.Cfg.dim and raised AttributeError after 8 minutes of compile -- the arm-a
    # crash at 13:37Z. A guard that skips an unknown flag is the wrong shape here: every name in
    # this map is one THIS file promises to apply, so an absent one is a bug in the map, and the
    # assert says which. Same class as the EOS_ID default that was always taken.
    for flag, field in (("dim", "d"), ("layers", "layers"), ("heads", "heads"),
                        ("ffn_hidden", "ffn_hidden"), ("batch", "batch"), ("accum", "accum")):
        assert hasattr(train.Cfg, field), (
            f"Cfg has no {field!r}, which --{flag} is supposed to set. The model would build at "
            f"Cfg's own value while the record claims the flag's -- two different programs."
        )
        setattr(train.Cfg, field, getattr(a, flag))
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
    # NOT .long(): build_mix returns int32 and the plan is 976,552 x 4097 = 14.9 GiB, so a
    # widening copy costs 29.8 GiB of host RAM for rows the step never reads. train.py keeps
    # the plan as returned and indexes into pinned buffers, which is what is timed below.
    seqs = tr[0] if train.Cfg.fone else tr
    X, Y = seqs[:, :-1], seqs[:, 1:]
    vseqs = va[0] if train.Cfg.fone else va
    Xva, Yva = vseqs[:, :-1], vseqs[:, 1:]

    raw = train.HybridLM(train.Cfg).to(dev)
    # train.py:2016-2020 -- fp8 is `args.fp8 and amp`, and it does TWO things: casts the
    # module to bf16 and then converts the linears. Recording fp8=True while doing neither
    # is a number without its premise: the model stays fp32, every activation doubles, and
    # 32x4096 OOMs at 93 GiB on a 95 GiB card (measured, this run's first attempt). The
    # cast is also not cosmetic -- it is what makes the timed step the run's step.
    amp = True
    fp8 = a.fp8 and amp
    if fp8:
        raw = raw.to(torch.bfloat16)
        train.convert_to_fp8_compute(raw)
    optimizers = train.build_optimizers(raw, train.Cfg)
    model = raw
    if ddp:
        model = DDP(model, device_ids=[local], bucket_cap_mb=25, gradient_as_bucket_view=True,
                    static_graph=True)
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
    losses = []
    n_par = sum(p.numel() for p in raw.parameters())
    if is_main:
        print(f"built {n_par / 1e6:.2f}M params, compile={train.Cfg.compile and amp}, "
              f"fp8={fp8}, grad_ckpt={a.grad_ckpt}, warmup {a.warmup} steps discarded",
              flush=True)

    # A Chrome trace of the LAST --trace-steps timed steps, for scripts/trace_classes.py.
    # record_shapes is REQUIRED by trace_classes: it derives a GEMM's ideal time from
    # 2*M*N*K, and a kernel whose shapes the trace does not carry is reported as unknown
    # rather than given a guessed ideal. It is also what OOMed b0's trace at the 206M shape
    # -- per-kernel shape metadata is held for the whole profiled window, 93.8 of 95.2 GiB on
    # a config that trains fine. So the window is the last few steps only, and with_flops
    # stays OFF: trace_classes computes FLOPs itself from the shapes, so with_flops buys a
    # column nothing reads and pays for it in the same memory (b0, measured).
    prof = None
    trace_from = a.steps + a.warmup - a.trace_steps
    if a.trace:
        from torch.profiler import ProfilerActivity, profile
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       record_shapes=True, with_flops=False, with_stack=False)

    # Reset before the loop, not after setup: the peak must describe the STEP, and building the
    # model, the optimizers and the pinned buffers all allocate. An arm that OOMs is decided by
    # this number, so it may not carry setup that both arms pay identically.
    torch.cuda.reset_peak_memory_stats()

    for st in range(a.steps + a.warmup):
        if prof is not None and st == trace_from:
            prof.start()
        torch.cuda.synchronize()
        t_step = time.perf_counter()
        t_fwd = None
        step_loss = torch.zeros((), device=dev)
        # ACCUM micro-batches per optimizer step, as train.py:2212 does: its loop strides by
        # Cfg.batch and steps the optimizer every Cfg.accum-th iteration, so one "step" is
        # accum forward/backward passes. Timing a single pass and calling it a step made
        # step_total independent of accum -- which would have decided fb's b32a1 vs b16a2 A/B
        # backwards, since accum IS the variable there: the b16a2 arm would report half its
        # real work and win on a number that describes half a step.
        for micro in range(train.Cfg.accum):
            idx = torch.arange(st * B * train.Cfg.accum + micro * B,
                               st * B * train.Cfg.accum + (micro + 1) * B) % len(X)
            torch.index_select(X, 0, idx, out=xb_pin)
            torch.index_select(Y, 0, idx, out=yb_pin)
            xb = xb_pin.to(dev, non_blocking=True)
            yb = yb_pin.to(dev, non_blocking=True)
            if micro == 0:
                torch.cuda.synchronize()
                t_fwd = time.perf_counter()

            cu = train.doc_cu_seqlens(xb, eos) if train.Cfg.doc_mask else None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                hidden, _ = model(xb, yb, cu, None)
            Bt, Tt, D = hidden.shape
            weight = raw.head.weight[: raw.cfg.vocab]
            loss = flce(weight, hidden.to(weight.dtype).reshape(-1, D), yb.reshape(-1))
            loss = loss / train.Cfg.accum
            step_loss += loss.detach()
            # train.py:2246 -- no_sync on every micro-batch but the last, or DDP all-reduces
            # accum times per step and the NCCL cost of the b16a2 arm doubles for no reason.
            if ddp and train.Cfg.accum > 1 and micro + 1 != train.Cfg.accum:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), train.Cfg.clip)
        for opt in optimizers:
            opt.step()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        # EVERY step's loss, warmup included. The A/B's correctness condition is that the two
        # arms compute the same thing, and the arms differ from step 0 -- comparing only the
        # timed window would skip the steps where a divergence is largest and easiest to see.
        losses.append(round(float(step_loss.item()), 6))
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

    trace_path = None
    if prof is not None:
        prof.stop()
        # Rank 0 only: eight ranks writing eight traces of the same step answers nothing
        # extra and the files are large. trace_classes.py consumes one.
        if is_main:
            trace_path = a.trace if isinstance(a.trace, str) else "runs/trace_step_cost.json"
            if not os.path.isabs(trace_path):
                trace_path = os.path.join(ROOT, trace_path)
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            prof.export_chrome_trace(trace_path)
            sz = os.path.getsize(trace_path) / 1e6
            print(f"trace: {trace_path} ({sz:.1f} MB, {a.trace_steps} steps, record_shapes=True) "
                  f"-> python3 scripts/trace_classes.py {os.path.relpath(trace_path, ROOT)} "
                  f"--steps {a.trace_steps}", flush=True)

    # vars() on a CLASS returns a mappingproxy, which pickle refuses -- train.py never hits
    # this because train.py:964 passes vars(cfg) through a dict comprehension first. The save
    # is the LAST thing this script does, so the timings and the trace were already complete
    # and correct when it raised; the crash cost the record, not the measurement.
    cfg_dict = {k: v for k, v in dict(vars(train.Cfg)).items() if not k.startswith("_")}
    saves = []
    if is_main:
        for _ in range(3):
            t0 = time.perf_counter()
            torch.save({"model": raw.state_dict(), "cfg": cfg_dict,
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

    # tok/s/gpu from the MEDIAN step, and peak from the loop only. These two are the A/B's whole
    # output: a config that is faster per GPU and fits is better, and nothing else in this record
    # decides that. Per-GPU, so an arm run on a different card count is still comparable -- the
    # 300M A/B runs on 4 while the p200m numbers came from 4 and the ladder from 7.
    med = _stats(steps)
    tok_step = B * train.Cfg.accum * SEQ
    rec = {"mix": a.mix, "world": world, "params_m": round(n_par / 1e6, 2),
           "shape": "step = e19eeb7's p200m launch line", "batch": B, "accum": train.Cfg.accum,
           "seq": SEQ, "layers": train.Cfg.layers, "dim": train.Cfg.d,
           "fp8": fp8, "grad_ckpt": a.grad_ckpt,
           "compile": bool(train.Cfg.compile and amp), "steps_timed": len(steps),
           "tokens_per_step_per_gpu": tok_step,
           "tok_s_per_gpu": round(tok_step / (med["median_ms"] / 1000.0)) if med else None,
           "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "loss_per_step": losses,
           "step_total": med, "loader_wait": _stats(loader),
           "nccl_floor": _stats(nccl) if ddp else None,
           "save": _stats(saves) if is_main else None, "val": _stats(vals)}
    if is_main:
        tot = rec["step_total"]["median_ms"] if rec["step_total"] else None
        print(f"\n{rec['params_m']}M host-side per-step cost  (mix {a.mix}, world {world}, "
              f"batch {B} x accum {train.Cfg.accum} x seq {SEQ}, "
              f"fp8={fp8} grad_ckpt={a.grad_ckpt} compile={rec['compile']})")
        for k in ("step_total", "loader_wait", "nccl_floor", "save", "val"):
            print("  " + fmt_row(k, rec[k], world, tot))
        print(f"  {'tok/s/gpu':16s} {rec['tok_s_per_gpu']:,}   "
              f"({tok_step:,} tok/step/gpu over the median step)")
        print(f"  {'peak':16s} {rec['peak_gib']:.2f} GiB   (loop only; setup excluded)")
        print(f"  {'loss':16s} first {losses[:3]} last {losses[-3:]}")
        if a.json:
            with open(a.json, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

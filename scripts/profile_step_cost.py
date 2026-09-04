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
  forward       model + FLCE, summed over the step's accum micro-batches. CUDA events, so no
                synchronize is added and step_total measures the same step it did before.
  backward      loss.backward(), summed the same way. Under DDP this INCLUDES the overlapped
                gradient reduction -- it is what backward costs, not what compute costs, and
                nccl_floor below is the separate bound on the reduction.
  opt_step      clip_grad_norm_ + every optimizer's step() and zero_grad().
                forward + backward + opt_step is less than step_total by the loader wait and
                the host-side gaps between regions; the residual is not attributed.
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


def _mark(marks):
    """Record a CUDA event into `marks`, or nothing at all when marks is None (a warmup
    step). None rather than a flag so the call sites carry no condition. torch is imported
    here, as everywhere in this file above main(): --selftest runs on a machine without it."""
    if marks is None:
        return
    import torch  # noqa: PLC0415

    e = torch.cuda.Event(enable_timing=True)
    e.record()
    marks.append(e)


def _regions(marks, n_mb):
    """(forward, backward, opt_step) in SECONDS from one step's marks: 3 per micro-batch
    (enter fwd, enter bwd, leave bwd) then 1 after the optimizer. Forward and backward are
    summed over micro-batches -- one step is accum of each, and reporting a single
    micro-batch's forward against a whole step's total understates it by exactly accum,
    which is the same error that would have decided the b32a1 vs b16a2 A/B backwards.
    A function, not inline arithmetic, so the selftest reads THIS and not a copy of it."""
    assert len(marks) == 3 * n_mb + 1, f"{len(marks)} marks for {n_mb} micro-batches"
    f = sum(marks[3 * i].elapsed_time(marks[3 * i + 1]) for i in range(n_mb))
    b = sum(marks[3 * i + 1].elapsed_time(marks[3 * i + 2]) for i in range(n_mb))
    o = marks[3 * n_mb - 1].elapsed_time(marks[3 * n_mb])
    return f / 1000.0, b / 1000.0, o / 1000.0


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
    tok_expr = reset_line = loop_line = loop_node = None
    for n in _a0.walk(_me_main) if _me_main else ():
        if isinstance(n, _a0.Assign) and any(
                isinstance(t, _a0.Name) and t.id == "tok_step" for t in n.targets):
            tok_expr = _a0.unparse(n.value).replace(" ", "")
        elif isinstance(n, _a0.Call) and isinstance(n.func, _a0.Attribute) \
                and n.func.attr == "reset_peak_memory_stats":
            reset_line = n.lineno
        elif isinstance(n, _a0.For) and _a0.unparse(n.iter).replace(" ", "") \
                == "range(a.steps+a.warmup)":
            loop_line, loop_node = n.lineno, n
    def _loop_end():
        return max((getattr(x, "lineno", 0) for x in _a0.walk(loop_node)), default=0) if loop_node else 0

    def _startup_before_reset():
        """The STARTUP read exists only if it happens before the reset. Same shape as the
        reset check itself: a read on the wrong side of one line turns two numbers into one,
        and both print fine."""
        if loop_node is None or reset_line is None:
            return False
        for n in _a0.walk(loop_node):
            if isinstance(n, _a0.Assign) and any(
                    isinstance(t, _a0.Name) and t.id == "startup_res" for t in n.targets):
                return n.lineno < reset_line
        return False

    def _reset_ok():
        """Before the loop, or inside it under `if st == a.warmup` -- both mean the peak
        describes the timed steps. The previous form was `reset_line < loop_line`, a line
        comparison that named the property but tested position: it rejected the placement
        that excludes compile and warmup, which is the STRICTER of the two, while claiming
        to be about what the peak carries (fb, resolving the b0-split merge 2026-09-03)."""
        if None in (reset_line, loop_line):
            return False
        if reset_line < loop_line:
            return True
        if reset_line > _loop_end():
            return False
        for n in _a0.walk(loop_node):
            if isinstance(n, _a0.If) and _a0.unparse(n.test).replace(" ", "") == "st==a.warmup":
                if any(getattr(x, "lineno", None) == reset_line for b in n.body for x in _a0.walk(b)):
                    return True
        return False

    for ok, why, hint in (
        (tok_expr == "B*train.Cfg.accum*SEQ", "tok/s/gpu counts batch*accum*seq",
         f"tok_step is {tok_expr!r}; a rate over one micro-batch reports the b8a4 arm at 4x"),
        (reset_line is not None, "the peak is reset in main()",
         "the peak would carry setup both arms pay identically, not the step's own cost"),
        (_startup_before_reset(), "the startup peak is read BEFORE the reset",
         "max_memory_reserved for STARTUP must be read before reset_peak_memory_stats, or it "
         "reports the steady-state number twice and the launch gate loses the transient it exists to catch"),
        (_reset_ok(), "the peak reset covers the timed steps and nothing else",
         f"reset at line {reset_line}, loop {loop_line}-{_loop_end()}: it must sit before the "
         "loop or inside it under `if st == a.warmup`. After the loop measures nothing; "
         "under any later gate the first timed steps' peak is lost"),
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

    # The RNG must be seeded BEFORE the model is built, as train.py:1835 does. Without it two
    # arms of an A/B get different initial weights and their loss-parity gate fires on the
    # harness rather than on the variable -- measured, 1.53 nat apart at step 5. Line order is
    # the whole assertion: seeding after construction leaves the init already drawn.
    seed_line = model_line = None
    for n in _nodes:
        if isinstance(n, _a1.Call) and _a1.unparse(n.func) == "torch.manual_seed":
            seed_line = n.lineno
        elif isinstance(n, _a1.Call) and _a1.unparse(n.func) == "train.HybridLM":
            model_line = n.lineno
    ok = None not in (seed_line, model_line) and seed_line < model_line
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the RNG is seeded before the model is built"
          + ("" if ok else f" -- manual_seed at {seed_line}, HybridLM at {model_line}; two arms "
                           "would init differently and the loss-parity gate would test nothing"))

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

    # _regions on a fake clock. accum=2 with distinguishable durations, so summing over
    # micro-batches is separable from taking one of them: forward is 1+3=4 ms and NOT 1, and
    # opt_step reads from the LAST backward's end mark rather than the first's.
    class _E:
        def __init__(self, t):
            self.t = t

        def elapsed_time(self, o):
            return o.t - self.t
    # fwd 1, bwd 10 | fwd 3, bwd 30 | opt 7  -> marks at 0,1,11, 14,17,47, 54
    m = [_E(x) for x in (0, 1, 11, 14, 17, 47, 54)]
    f, b, o = _regions(m, 2)
    ok = (round(f, 6), round(b, 6), round(o, 6)) == (0.004, 0.040, 0.007)
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} regions sum over micro-batches "
          f"(fwd {f * 1000:.0f} bwd {b * 1000:.0f} opt {o * 1000:.0f} ms, want 4/40/7)")

    try:
        _regions(m, 1)
        ok = False
    except AssertionError:
        ok = True
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} a mark count that disagrees with accum raises "
          "rather than reporting a shifted region")

    n = 6 + 3 + 2 + 2 + 4 + 2 + 4 + 2 + 1 + 2
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
    ap.add_argument("--skip-save-val", action="store_true",
                    help="skip the save and val timings and still write the JSON row. Measured "
                         "on the memory arm: save 33.6 s and val 5.7 s per run against a 2.0 s "
                         "step, and a multi-config sweep pays that per cell for numbers no cell "
                         "reads. --peak-only also skips them but returns before the row is "
                         "written, so it cannot feed a table.")
    ap.add_argument("--peak-only", action="store_true",
                    help="memory probe: run --steps steps, report peak GiB and tok/s/gpu, "
                         "skip save/val/nccl. For the L18 feasibility question.")
    # SPARSE MEMORY, so --peak-only can answer the M3 question before a card is committed to it.
    # M3 is 2048^2 x 1024 = 4,294,967,296 parameters: 8.00 GiB of bf16 table plus a DENSE fp32
    # Adagrad moment of 16 GiB, against a 95.58 GiB card whose control peak is 49.53 GiB. Whether
    # that fits is a measurement, not an addition -- activations and the allocator's fragmentation
    # are what the arithmetic cannot predict, and this is the only tool that reads the high-water
    # mark of live tensors rather than nvidia-smi's reservation.
    ap.add_argument("--mem_values", type=int, default=0,
                    help="sparse memory pool size, a perfect square (0 = no memory)")
    ap.add_argument("--mem_top_k", type=int, default=32)
    ap.add_argument("--mem_layers", type=str, default="3,6,9")
    ap.add_argument("--mem_sparse", action=argparse.BooleanOptionalAction, default=False,
                    help="COO grads for the value table. Default FALSE, matching the arms: NCCL "
                         "raises on all_reduce of a sparse tensor (tilerl, 2026-09-05)")
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
    # Same map, same assert, for the memory fields: a name here is one this file PROMISES to
    # apply, so an absent field is a bug in the map and the assert names it. Without this the
    # probe would build a model with no memory and report its peak as M3's -- the number would
    # look like a comfortable fit and be an answer to a different question.
    for _f in ("mem_values", "mem_top_k", "mem_layers", "mem_sparse"):
        assert hasattr(train.Cfg, _f), f"Cfg has no {_f!r}, which --{_f} is supposed to set"
        setattr(train.Cfg, _f, getattr(a, _f))
    # mem_arm is train.py's launch-time requirement, not this probe's: nothing here writes a
    # memory_diag row, so there is no arm label to get wrong. Set explicitly rather than left
    # empty so a future reader does not read the empty string as an oversight.
    if hasattr(train.Cfg, "mem_arm"):
        train.Cfg.mem_arm = "probe"

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

    # train.py:1835 seeds before building the model, and this file did not -- so two arms of an
    # A/B initialized DIFFERENT weights and their per-step losses diverged by up to 1.53 nat,
    # failing the <=1e-3 parity gate for a reason that has nothing to do with the variable under
    # test (grad_ckpt and the batch/accum split). MEASURED 2026-09-02: the b16a2 and b8a4 arms
    # read 10.596 vs 10.580 at step 0, already 0.016 apart before any of the arms' own arithmetic
    # could differ. A parity gate that fires on an unseeded init tests the harness, not the arms.
    torch.manual_seed(train.Cfg.seed)
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
    # AFTER the cast, BEFORE DDP and compile -- train.py's order (:2016 cast, :2445 DDP,
    # :2491 compile). Building the optimizers first would hand Muon fp32 parameter references
    # that the cast then replaces, so the optimizer would step tensors the model no longer uses.
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
    fwds, bwds, opts = [], [], []
    losses = []
    mem_grad = None  # (bytes per element, layout) of the table's gradient, read inside the loop
    n_par = sum(p.numel() for p in raw.parameters())
    if is_main:
        print(f"built {n_par / 1e6:.2f}M params, compile={train.Cfg.compile and amp}, "
              f"fp8={fp8}, grad_ckpt={a.grad_ckpt}, warmup {a.warmup} steps discarded",
              flush=True)

    # A Chrome trace of the LAST --trace-steps timed steps, for scripts/trace_classes.py.
    # record_shapes is REQUIRED by trace_classes: it derives a GEMM's ideal time from
    # 2*M*N*K, and a kernel whose shapes the trace does not carry is reported as unknown
    # rather than given a guessed ideal. with_flops stays OFF: trace_classes computes FLOPs
    # itself from the shapes, so it buys a column nothing reads (b0, measured).
    #
    # THE MEMORY COST OF record_shapes IS UNKNOWN, NOT MEASURED (fb retraction, 2026-09-02).
    # The trace OOMed at 93.8/95.22 GiB with batch 32 accum 1 and I blamed the profiler.
    # eff.microbatch_32_oom measured b32a1 OOMing at the same 93.8/95.2 GB with no profiler
    # attached at all, and p200m OOMed at 95.1 GiB the same way. Three OOMs at one number,
    # one of them profiler-free: the batch was the variable. Trace at the RUN's shape (b16a2)
    # first; only if that does not fit is the batch reduced, and then the batch becomes part
    # of the trace's provenance, because a share measured at a smaller batch is not the run's.
    prof = None
    trace_from = a.steps + a.warmup - a.trace_steps
    if a.trace:
        from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       record_shapes=True, with_flops=False, with_stack=False)

    wall0 = startup_res = startup_alloc = None
    for st in range(a.steps + a.warmup):
        if prof is not None and st == trace_from:
            prof.start()
        # Peak over the TIMED steps only. Resetting here rather than before the loop keeps
        # construction, compile and warmup's transient allocations out of a number that decides
        # whether a shape FITS -- the question is whether the steady-state step fits.
        if st == a.warmup:
            # READ BEFORE THE RESET. Construction, compile and warmup allocate more than a
            # steady-state step does -- b0 watched nvidia-smi hit 72.5 GiB on a d1536 b8 probe
            # whose steady-state peak predicts 38.87 -- and that transient OOMs exactly like a
            # step does. The reset below is correct for "does the STEP fit"; without this read
            # the run reports a number that says GO on a shape that dies during compile.
            startup_res = torch.cuda.max_memory_reserved() / 1024**3
            startup_alloc = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.reset_peak_memory_stats()
            wall0 = time.perf_counter()
        torch.cuda.synchronize()
        t_step = time.perf_counter()
        t_fwd = None
        # CUDA events, not perf_counter: a wall-clock read between forward and backward returns
        # launch cost, and inserting a synchronize to fix that changes the step being measured.
        # Events are recorded in-stream and read after the step's existing synchronize, so the
        # regions cost nothing that step_total does not already pay.
        timed = st >= a.warmup
        marks = [] if timed else None
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
            _mark(marks)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                hidden, _ = model(xb, yb, cu, None)
            Bt, Tt, D = hidden.shape
            weight = raw.head.weight[: raw.cfg.vocab]
            loss = flce(weight, hidden.to(weight.dtype).reshape(-1, D), yb.reshape(-1))
            loss = loss / train.Cfg.accum
            step_loss += loss.detach()
            _mark(marks)
            # train.py:2246 -- no_sync on every micro-batch but the last, or DDP all-reduces
            # accum times per step and the NCCL cost of the b16a2 arm doubles for no reason.
            if ddp and train.Cfg.accum > 1 and micro + 1 != train.Cfg.accum:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()
            _mark(marks)
        torch.nn.utils.clip_grad_norm_(raw.parameters(), train.Cfg.clip)
        # THE TABLE'S GRADIENT, CAPTURED HERE BECAUSE NOTHING CAN READ IT LATER: the
        # zero_grad(set_to_none=True) two lines down drops it, so a report written after the
        # loop sees grad=None and would print 0 bytes for the largest single tensor the
        # memory allocates. Itemsize and layout only -- no tensor is retained.
        if mem_grad is None and train.Cfg.mem_values:
            _g = raw.memory.values.weight.grad
            mem_grad = (0, "absent") if _g is None else (
                _g.element_size(), "sparse COO" if _g.is_sparse else "dense")
        for opt in optimizers:
            opt.step()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        _mark(marks)
        torch.cuda.synchronize()
        # EVERY step's loss, warmup included. The A/B's correctness condition is that the two
        # arms compute the same thing, and the arms differ from step 0 -- comparing only the
        # timed window would skip the steps where a divergence is largest and easiest to see.
        losses.append(round(float(step_loss.item()), 6))
        if st >= a.warmup:
            loader.append(t_fwd - t_step)
            steps.append(time.perf_counter() - t_step)
            n_mb = train.Cfg.accum
            f, b, o = _regions(marks, n_mb)
            fwds.append(f)
            bwds.append(b)
            opts.append(o)

        # NOT under --peak-only: this allocates n_par floats (1.75 GB at 438M) inside the
        # window whose peak is the answer, so the probe would measure itself.
        if ddp and st >= a.warmup and not a.peak_only:
            g = torch.zeros(n_par, device=dev)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.distributed.all_reduce(g)
            torch.cuda.synchronize()
            nccl.append(time.perf_counter() - t0)
            del g

    wall = time.perf_counter() - wall0
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    peak_res_gib = torch.cuda.max_memory_reserved() / 1024**3
    tok_s_gpu = a.steps * B * SEQ / wall
    if is_main:
        print(f"\nPEAK {peak_gib:.2f} GiB allocated, {peak_res_gib:.2f} GiB reserved | "
              f"{tok_s_gpu / 1e3:.1f}K tok/s/gpu over {a.steps} timed steps "
              f"(L={train.Cfg.layers} d={train.Cfg.d} batch={B} accum={train.Cfg.accum} "
              f"grad_ckpt={a.grad_ckpt} fp8={a.fp8} world={world}"
              # THE MEMORY CONFIG ON THE SAME LINE AS THE PEAK IT PRODUCED. A peak reported
              # without the pool size that produced it is the shape of number that gets copied
              # into a decision about a different shape.
              + (f" mem_values={train.Cfg.mem_values} top_k={train.Cfg.mem_top_k}"
                 f" layers={train.Cfg.mem_layers} sparse={train.Cfg.mem_sparse}"
                 if train.Cfg.mem_values else " mem=off")
              + ")", flush=True)
        print("  reserved is the number that decides whether it FITS -- allocated omits the "
              "caching allocator's fragmentation, and OOM is raised against reserved.",
              flush=True)
        # THE TABLE'S BYTES PER PARAMETER, READ OFF THE LIVE TENSORS. A peak answers "does
        # THIS table fit"; extrapolating to another size multiplies bytes-per-parameter, and
        # that figure was derived by reading the construction site -- nn.Embedding in the
        # default dtype, so fp32, 12 B/param -- rather than the model that trains. `raw.to(
        # torch.bfloat16)` at :610 casts every floating parameter, the embedding included, so
        # the premise does not survive the cast. PRINTED HERE, after the timed steps, because
        # Adagrad allocates its state on the FIRST step: reading it before would have to
        # assume the dtype instead of measuring it, which is the error this line exists to
        # correct.
        if train.Cfg.mem_values:
            tw = raw.memory.values.weight
            wb = tw.element_size()
            gb, glayout = mem_grad if mem_grad else (0, "never read")
            st = [s for o in optimizers if isinstance(o, torch.optim.Adagrad)
                  for p, s in o.state.items() if p is tw]
            sb = st[0]["sum"].element_size() if st and "sum" in st[0] else 0
            print(f"TABLE dtype={tw.dtype} weight={wb}B grad={gb}B ({glayout}) "
                  f"opt_state={sb}B{'' if sb else ' (no Adagrad state found)'} "
                  f"-> {wb + gb + sb} B/param x {tw.numel()} params = "
                  f"{tw.numel() * (wb + gb + sb) / 2**30:.2f} GiB of table tensors", flush=True)
        print(f"STARTUP {startup_alloc:.2f} GiB allocated, {startup_res:.2f} GiB reserved "
              f"(construction + compile + {a.warmup} warmup steps, before the reset)", flush=True)
        print("  TWO NUMBERS, TWO QUESTIONS. STARTUP decides whether the shape can be LAUNCHED "
              "at all; PEAK decides whether each step fits once it is running. STARTUP is the "
              "larger of the two and a gate reading only PEAK says GO on a shape that dies "
              "during compile.", flush=True)

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

    if a.peak_only:
        # The regions print HERE too, not only in the full report below: --peak-only is the
        # mode the memory arms run in, and a peak without the forward/backward split cannot
        # say whether a slow arm is the lookup or the world. Same numbers, printed earlier.
        if is_main:
            tot = _stats(steps)
            tot_ms = tot["median_ms"] if tot else None
            for k, v in (("step_total", tot), ("loader_wait", _stats(loader)),
                         ("forward", _stats(fwds)), ("backward", _stats(bwds)),
                         ("opt_step", _stats(opts))):
                print("  " + fmt_row(k, v, world, tot_ms), flush=True)
        if ddp:
            torch.distributed.destroy_process_group()
        return 0

    # SKIP SAVE AND VAL, and skip them by measurement rather than by taste: save measured
    # 33.6 s and val 5.7 s per config on the memory arm, against a 2.0 s step. A five-config
    # decomposition pays 3+ minutes of that per cell for two numbers no cell reads, and the
    # 33.6 s save is where the m1 cell's ranks desynchronised -- rank 0 saving while rank 1
    # waits is a two-minute gap with no collective in it.
    saves = vals = []
    if not a.skip_save_val:
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
           # The memory config IN THE ROW, for the same reason the print line carries it:
           # rows that differ only by --mem_values are otherwise indistinguishable in the JSONL,
           # and a decomposition table keyed on nothing is a table of one config repeated.
           "mem_values": train.Cfg.mem_values, "mem_top_k": train.Cfg.mem_top_k,
           "mem_layers": train.Cfg.mem_layers, "mem_sparse": train.Cfg.mem_sparse,
           "compile": bool(train.Cfg.compile and amp), "steps_timed": len(steps),
           "tokens_per_step_per_gpu": tok_step,
           "tok_s_per_gpu": round(tok_step / (med["median_ms"] / 1000.0)) if med else None,
           "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "loss_per_step": losses,
           "step_total": med, "loader_wait": _stats(loader),
           "forward": _stats(fwds), "backward": _stats(bwds), "opt_step": _stats(opts),
           "nccl_floor": _stats(nccl) if ddp else None,
           "save": _stats(saves) if is_main else None, "val": _stats(vals)}
    if is_main:
        tot = rec["step_total"]["median_ms"] if rec["step_total"] else None
        print(f"\n{rec['params_m']}M host-side per-step cost  (mix {a.mix}, world {world}, "
              f"batch {B} x accum {train.Cfg.accum} x seq {SEQ}, "
              f"fp8={fp8} grad_ckpt={a.grad_ckpt} compile={rec['compile']})")
        for k in ("step_total", "loader_wait", "forward", "backward", "opt_step",
                  "nccl_floor", "save", "val"):
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

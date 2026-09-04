#!/usr/bin/env python3
"""The nobk cell: is backward's +315 ms the scatter-add, or compile fragmenting around
the diagnostics block?

4c's hypothesis (2026-09-05): the index_put on a 1M-element bool buffer plus two bincounts
at model.py:538 sit inside the compiled forward, and a graph break there splits the backward
too -- which would inflate backward far beyond the block's own 38 ms of forward cost. This
runs M1 with that block removed and nothing else changed.

EVIDENCE ALREADY AGAINST IT, stated so the cell has something to disagree with:
probes/mem_boundary_audit.py ran dynamo.explain on ProductKeyMemory with the block live
(training=True, grad enabled, so the condition is True) and reported graphs=1 breaks=0
ops=36. Zero breaks at the toy shape. The gap that keeps the cell worth running is that the
audit compiles the memory ALONE at D=128 side=64, not inside the full model where inductor
sees the surrounding blocks.

Prediction on the record before it runs (tilerl): forward drops ~38 ms, backward drops
under 50, the scatter-add stays the finding. Backward dropping more than 100 ms means 4c is
right and the fix is a step-gate outside the compiled region.

HOW THE BLOCK IS REMOVED. The If node is located in model.py's OWN ast by its test
expression -- `not self.training or torch.is_grad_enabled()` -- and deleted, then the class
is recompiled and the method rebound. Reading the live source rather than carrying a copy of
forward is what keeps this from silently measuring an older version of the function: a
hand-pasted body would still run after model.py changed under it and would report the delta
of a function nobody trains. Nothing is written to model.py.

    torchrun --nproc_per_node=2 scripts/mem_nobk_cell.py --json runs/mem_decomp_0905.jsonl
"""
import argparse
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def strip_bookkeeping(verbose=True, root=None):
    """Delete the diagnostics If from ProductKeyMemory.forward. Returns the number of
    statements removed; refuses rather than silently changing nothing."""
    import model

    with open(os.path.join(root or ROOT, "model.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    cls = next((n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == "ProductKeyMemory"), None)
    assert cls is not None, "model.py has no ProductKeyMemory"
    fn = next((n for n in cls.body
               if isinstance(n, ast.FunctionDef) and n.name == "forward"), None)
    assert fn is not None, "ProductKeyMemory has no forward"

    # Matched on the TEST, not on an index. A positional match reads correct and silently
    # deletes the wrong statement the moment anything above it moves.
    victims = [i for i, n in enumerate(fn.body)
               if isinstance(n, ast.If) and "is_grad_enabled" in ast.unparse(n.test)]
    assert len(victims) == 1, (
        f"expected exactly one diagnostics If in forward, found {len(victims)}. "
        f"The block moved or split; this probe would remove the wrong thing."
    )
    removed = fn.body.pop(victims[0])
    # The statements INSIDE the no_grad `with`, not the If's own body: the If holds a single
    # With, so counting its body reports 1 for an eight-statement block and a caller checking
    # "did anything real get removed" would be satisfied by a stub.
    n_removed = sum(len(getattr(s, "body", [s])) for s in removed.body)
    if verbose:
        print(f"  removed: if {ast.unparse(removed.test)}  "
              f"({n_removed} statements)", flush=True)

    # Recompile ONLY this class, in model's own namespace, so names it closes over resolve
    # exactly as they do in the module.
    mod = ast.Module(body=[cls], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = dict(vars(model))
    exec(compile(mod, "<model.py: ProductKeyMemory, bookkeeping stripped>", "exec"), ns)
    model.ProductKeyMemory.forward = ns["ProductKeyMemory"].forward
    return n_removed


def _selftest():
    """The strip works on model.py as it stands, and the result no longer touches the
    buffers. No card."""
    bad = 0
    import model

    with open(os.path.join(ROOT, "model.py"), encoding="utf-8") as fh:
        before = ast.unparse(ast.parse(fh.read()))
    n = strip_bookkeeping(verbose=False)
    ok = n == 8
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the diagnostics block was found and removed "
          f"({n} statements inside the no_grad, want 8)")

    # THE BYTECODE, not inspect.getsource: the patched method is compiled from a synthetic
    # filename with no source on disk, so getsource raises -- and a source check would be
    # answering about text rather than about what the function will execute anyway.
    import dis
    import io
    buf = io.StringIO()
    dis.dis(model.ProductKeyMemory.forward, file=buf)
    code = buf.getvalue()
    ok = ("touched" not in code and "bincount" not in code
          and "key_hits" not in code and "values" in code)
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the patched forward's bytecode drops "
          f"touched/bincount/key_hits and keeps the lookup")

    # THE FILE IS UNCHANGED. A probe that edits model.py would leave the tree in a state a
    # later run inherits without knowing it.
    with open(os.path.join(ROOT, "model.py"), encoding="utf-8") as fh:
        after = ast.unparse(ast.parse(fh.read()))
    ok = before == after
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} model.py on disk is untouched")

    # A forward with no diagnostics block must REFUSE, not report a no-op as done. Calling
    # strip twice does not test that -- it re-parses model.py from disk each time and
    # legitimately finds the block again. Point it at an already-stripped copy of model.py
    # instead, which is the state that would otherwise measure nothing and say it measured
    # the fix. The check calls strip_bookkeeping ITSELF: re-deriving the victim filter here
    # would test a copy of the rule and pass while the real one was broken.
    import tempfile
    with open(os.path.join(ROOT, "model.py"), encoding="utf-8") as fh:
        stripped = ast.parse(fh.read())
    c2 = next(n for n in ast.walk(stripped)
              if isinstance(n, ast.ClassDef) and n.name == "ProductKeyMemory")
    f2 = next(n for n in c2.body
              if isinstance(n, ast.FunctionDef) and n.name == "forward")
    f2.body = [n for n in f2.body
               if not (isinstance(n, ast.If) and "is_grad_enabled" in ast.unparse(n.test))]
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "model.py"), "w", encoding="utf-8") as fh:
            fh.write(ast.unparse(stripped))
        try:
            strip_bookkeeping(verbose=False, root=td)
            ok = False
        except AssertionError:
            ok = True
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} a forward with no diagnostics block makes "
          f"strip_bookkeeping refuse, never report a no-op as done")

    print(f"mem_nobk_cell selftest: {4 - bad}/4 pass")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a, rest = ap.parse_known_args()
    if a.selftest:
        return _selftest()

    print("stripping the diagnostics block from ProductKeyMemory.forward", flush=True)
    strip_bookkeeping()

    # profile_step_cost's main() with M1's flags, in THIS process, so the patched class is
    # what it builds. Importing and calling rather than spawning: a subprocess would import
    # a clean model.py and measure the arm as it already ran.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "psc", os.path.join(ROOT, "scripts", "profile_step_cost.py"))
    psc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(psc)
    sys.argv = ["profile_step_cost.py",
                "--mix", "data/mix_200m_8b.json",
                "--dim", "1024", "--layers", "12", "--heads", "8", "--ffn_hidden", "3072",
                "--batch", "16", "--accum", "2", "--steps", "20", "--warmup", "8",
                "--mem_values", "1048576", "--mem_top_k", "32", "--mem_layers", "3,6,9",
                "--no-mem_sparse"] + rest
    return psc.main()


if __name__ == "__main__":
    sys.exit(main())

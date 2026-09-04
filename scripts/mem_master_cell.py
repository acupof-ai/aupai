#!/usr/bin/env python3
"""The master-off half of the table-master pair: what does TableMaster cost per step?

train.py builds TableMaster for any mem_values > 0 (:2603), and profile_step_cost mirrors
that unconditionally, so the probe as it stands IS the master-on half. This runs the same
probe with the master absent, which is the world before amendment 8's fix.

WHAT THE PAIR MEASURES, stated precisely because it is more than the copies. 4c's prediction
is about pull_grads + push -- ~12 GB of fp32 traffic per step, predicted under 1 point. But
the master also changes WHICH tensor the optimizer steps: with it, Adagrad holds the fp32
master and its `sum` state is fp32 over 1,048,576 rows; without it, Adagrad holds the bf16
table and `sum` is bf16. That is a real difference in the same direction and it is not copy
traffic.

The regions separate them without any extra instrumentation, which is why this is one pair
and not two: `opt_step` carries the optimizer's dtype change, and the copies land in
`opt_step` too (pull_grads runs before the optimizers, push after), so the split to report is
opt_step's delta against the 19.2 ms the decomposition already measured for a bf16 Adagrad
over this table. Anything appearing in forward or backward is neither and would want
explaining.

HOW THE MASTER IS TURNED OFF: train.TableMaster is replaced with a factory returning None,
which is the one value the probe already handles -- `table_master = train.TableMaster(raw) if
raw.memory is not None else None`, then `build_optimizers(..., None)` and both copy calls
guarded on `is not None`. So the off-half runs the probe's own no-master path rather than a
second code path written for this measurement. Nothing is written to train.py.

    torchrun --nproc_per_node=2 scripts/mem_master_cell.py --mem_layers 6 ...
    python3 scripts/mem_master_cell.py --selftest
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def disable_table_master():
    """Replace train.TableMaster with a factory returning None. Returns the original, and
    refuses if the name is not what this probe expects to be disabling."""
    import train

    orig = getattr(train, "TableMaster", None)
    assert orig is not None, (
        "train has no TableMaster. This probe exists to turn it off; with the name gone, "
        "running it would measure the master-on world and report it as master-off."
    )
    assert isinstance(orig, type), f"train.TableMaster is {type(orig)}, not a class"
    assert hasattr(orig, "pull_grads") and hasattr(orig, "push"), (
        "train.TableMaster has no pull_grads/push -- the thing being disabled is not the "
        "thing the pair is measuring."
    )
    train.TableMaster = lambda model: None
    return orig


def _selftest():
    """The patch reaches the probe's own no-master path. No card."""
    bad = 0
    import train

    orig = disable_table_master()
    ok = train.TableMaster(object()) is None
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the patched factory returns None, which is the "
          f"value the probe's no-master path takes")

    # THE PROBE MUST ACTUALLY BRANCH ON IT. A patch that returns None buys nothing if the
    # probe then calls pull_grads unguarded -- it would raise, not measure, and the check
    # that only looks at the patch would still be green.
    import ast
    with open(os.path.join(ROOT, "scripts", "profile_step_cost.py"), encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src)
    main_fn = next((n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    calls = [ast.unparse(n) for n in ast.walk(main_fn)
             if isinstance(n, ast.Call) and "table_master." in ast.unparse(n.func)]
    guarded = [n for n in ast.walk(main_fn)
               if isinstance(n, ast.If) and "table_master is not None" in ast.unparse(n.test)]
    ok = len(calls) >= 2 and len(guarded) >= 2
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} profile_step_cost guards every table_master call "
          f"({len(calls)} call(s), {len(guarded)} guard(s))")

    # And it must pass the master to build_optimizers, or the off-half would differ from the
    # on-half only in the copies while both optimizers held the same tensor.
    ok = "table_master.map if table_master else None" in src
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the optimizer takes the master's map, so the pair "
          f"differs in what Adagrad steps as well as in the copies")

    train.TableMaster = orig
    ok = train.TableMaster is orig
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the original class is restored")

    print(f"mem_master_cell selftest: {4 - bad}/4 pass")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a, rest = ap.parse_known_args()
    if a.selftest:
        return _selftest()

    print("disabling train.TableMaster for the master-off half", flush=True)
    disable_table_master()

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "psc", os.path.join(ROOT, "scripts", "profile_step_cost.py"))
    psc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(psc)
    sys.argv = ["profile_step_cost.py"] + rest
    return psc.main()


if __name__ == "__main__":
    sys.exit(main())

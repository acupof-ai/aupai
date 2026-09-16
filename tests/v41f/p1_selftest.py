#!/usr/bin/env python3
"""P1 allclose runner — assembled modules (Block, then Model).

Same discovery/contract as p0_selftest: runs every tests/v41f/test_p1_*.py test
function against the vendored reference, CPU bf16. P1 proves the LEAF MODULES are
wired together correctly (whole-Block / whole-Model forward), which per-module P0
cannot. Tolerates the --selftest argument the hook appends.

Run: python tests/v41f/p1_selftest.py [--selftest]
"""
import importlib.util
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    test_files = sorted(HERE.glob("test_p1_*.py"))
    if not test_files:
        print("no test_p1_*.py found")
        return 1
    failures = 0
    n_cases = 0
    for tf in test_files:
        spec = importlib.util.spec_from_file_location(tf.stem, tf)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            n_cases += 1
            try:
                fn()
                print(f"ok   {tf.name}::{name}")
            except Exception:
                failures += 1
                print(f"FAIL {tf.name}::{name}")
                traceback.print_exc()
    print(f"P1 allclose: {n_cases - failures}/{n_cases} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""P0 allclose runner — the single selftest entry the pre-commit hook gates.

Discovers and runs every tests/v41f/test_p0_*.py test function against the vendored
upstream reference. Tolerates the --selftest argument the hook appends. New P0 files
need no hook registration: drop a test_p0_*.py next to the others.

Run: python tests/v41f/p0_selftest.py [--selftest]
"""
import importlib.util
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    test_files = sorted(HERE.glob("test_p0_*.py"))
    if not test_files:
        print("no test_p0_*.py found")
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
    print(f"P0 allclose: {n_cases - failures}/{n_cases} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

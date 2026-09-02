#!/usr/bin/env python3
"""3b-6 m1b: mine runnable impl+test pairs from code_supply (The-Stack github dump).

fb ruling: mine, don't synthesize. Output feeds SFT corpus v1 (code-with-tests) and
de-28's code-execution reward ground truth. Per-pair format (aligned with de-28):
  {repo, impl_path, impl, test_path, tests, passed: true}

Per repo:
  1. test files by path heuristic; impl = a same-repo file whose basename matches a
     module the test imports (bare or dotted), found by ast-import analysis across
     the repo's impl-file basenames.
  2. drop pairs importing third-party (sandbox is stdlib-only).
  3. assemble sandbox blob = impl + module-alias (sys.modules[mod]=__main__ so the
     test's imported module resolves to the inlined impl) + the test source.
  4. run_sandboxed(blob); keep iff rc == 0 and the test actually calls unittest asserts.
  5. emit {repo, impl_path, impl, test_path, tests, passed: true}.

Requires root: pod-only. Writes per shard to a NEW dir (frozen-domain rule).
"""
import argparse
import ast
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyarrow.parquet as pq  # noqa: E402
import sandbox_exec  # noqa: E402

TEST_RE = re.compile(r"(/[Tt]est[s]?/|/t/?ests/|(^|[/._-])(test|tests|spec)[/._-]|test[a-zA-Z0-9_]*\.py$|_test\.py$|\.test\.)", re.I)
THIRD_PARTY = {"numpy", "pandas", "scipy", "sklearn", "torch", "tensorflow", "tqdm",
               "pytest", "requests", "flask", "django", "matplotlib", "sympy", "click", "yaml"}


def imports(mod):
    """All module names a test imports, with dotted parts (bare 'a' and full 'a.b')."""
    try:
        tree = ast.parse(mod)
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            out.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for n in node.names:
                out.add(n.name)
                out.add(n.name.split(".")[0])
    return out


def impl_matches(imported, impl_by_base):
    """Return an impl basename X such that 'X' or 'a.b.X' is imported."""
    for base in impl_by_base:
        for name in imported:
            if name == base or name.endswith("." + base):
                return base
    return None


def detects_unittest(src):
    return "unittest." in src or "assertTrue" in src or "self.assertEqual" in src


def has_unskippable_assert(src):
    """True iff some test method has a real assert and is not skip-decorated.
    de-28b: a fully-skippable/assert-free test returns rc 0 even with a wrong impl
    (reward 1.0). Guard at the miner (cheapest), not in the reward parse."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test"):
            continue
        if any(isinstance(d, ast.Name) and d.id == "skip" for d in node.decorator_list):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assert):
                return True
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and \
               sub.func.attr in ("assertEqual", "assertTrue", "assertIs", "assertIn"):
                return True
    return False


def stdlib_core_ok(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in THIRD_PARTY:
                return False
        elif isinstance(node, ast.Import):
            for n in node.names:
                if n.name.split(".")[0] in THIRD_PARTY:
                    return False
    return True


def test_runner_suffix():
    """Appended to the blob so the test ACTUALLY RUNS (not just imports/parses).
    Runs unittest-discovered TestCases in the inlined module; exit 1 on failure
    so a wrong impl cannot yield rc 0. Mirrors de-28's pytest semantics for the
    unittest-pattern tests this miner accepts."""
    return (
        "\n\nimport unittest as _un\n"
        "_suite = _un.defaultTestLoader.loadTestsFromModule(__import__('__main__'))\n"
        "_res = _un.TextTestRunner(verbosity=0).run(_suite)\n"
        "raise SystemExit(0 if _res.wasSuccessful() else 1)\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard")
    ap.add_argument("--out_dir", default="/work/aupai/data/sft/code_with_tests")
    ap.add_argument("--max_pairs", type=int, default=0)
    a = ap.parse_args()

    pf = pq.ParquetFile(a.shard)
    by_repo = defaultdict(lambda: defaultdict(str))
    for batch in pf.iter_batches(batch_size=5000):
        d = batch.to_pydict()
        for i in range(len(d["path"])):
            by_repo[d["repo_name"][i]][d["path"][i]] = d["content"][i]

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, os.path.basename(a.shard).replace(".parquet", "_mined.jsonl"))
    counts = {"repos": len(by_repo), "test_files": 0, "stdlib_only": 0,
              "matched_impl": 0, "passed": 0}
    npairs = 0
    with open(out, "w", encoding="utf-8") as fo:
        for repo, files in by_repo.items():
            impl_by_base = {}
            for p in files:
                if not TEST_RE.search(p):
                    impl_by_base.setdefault(os.path.basename(p)[:-3], p)  # strip .py
            for tp, src in files.items():
                if not TEST_RE.search(tp):
                    continue
                counts["test_files"] += 1
                if not detects_unittest(src) or not has_unskippable_assert(src):
                    continue
                if not stdlib_core_ok(src):
                    continue
                counts["stdlib_only"] += 1
                target = impl_matches(imports(src), impl_by_base)
                if target is None:
                    continue
                counts["matched_impl"] += 1
                ip = impl_by_base[target]
                impl = files[ip]
                if not impl or not stdlib_core_ok(impl):
                    continue
                blob = impl + "\n\nimport sys as _s; _s.modules[%r] = _s.modules['__main__']\n\n" % target + src + test_runner_suffix()
                rc, outt, err = sandbox_exec.run_sandboxed(blob, timeout=15)
                if rc == 0:
                    fo.write(json.dumps({"repo": repo, "impl_path": ip, "impl": impl,
                                         "test_path": tp, "tests": src, "passed": True},
                                        ensure_ascii=False) + "\n")
                    counts["passed"] += 1
                    npairs += 1
                if a.max_pairs and npairs >= a.max_pairs:
                    fo.flush()
                    print(json.dumps(counts))
                    print(f"WROTE {out} {npairs} pairs")
                    return
    print(json.dumps(counts))
    print(f"WROTE {out} {npairs} pairs")


if __name__ == "__main__":
    main()
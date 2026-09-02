#!/usr/bin/env python3
"""3b-6 m1b: mine runnable impl+test pairs from code_supply (The-Stack github dump).

fb ruling: mine, don't synthesize. Output feeds SFT corpus v1 (code-with-tests) and
de-28's code-execution reward ground truth. Per-pair format (aligned with de-28):
  {repo, impl_path, impl, test_path, tests, passed: true}

Per repo:
  1. test files by path heuristic (/tests/ or test_|_test stem); impl = the module
     the test imports under the same name, found by ast-import analysis + stem match.
  2. drop pairs importing third-party (sandbox is stdlib-only).
  3. assemble sandbox blob = impl + module-alias (so `import <mod>` resolves to the
     inlined impl via sys.modules) + the test.
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
    """Set of all module top-names a test file imports (stdlib + its target)."""
    try:
        tree = ast.parse(mod)
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for n in node.names:
                out.add(n.name.split(".")[0])
    return out


def detects_unittest(src):
    return "unittest." in src or "assertTrue" in src or "self.assertEqual" in src


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


def impl_stem_for_test(tp):
    base = os.path.basename(tp)
    if base.startswith("test_"):
        return base[len("test_"):]
    if base.endswith("_test.py"):
        return base[:-len("_test.py")] + ".py"
    # tests/test_x.py -> sibling x.py
    if base.startswith("test") and base.endswith(".py") and base != "test.py":
        return base[len("test"):]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard")
    ap.add_argument("--out_dir", default="/work/aupai/data/sft/code_with_tests")
    ap.add_argument("--max_pairs", type=int, default=0)
    a = ap.parse_args()

    pf = pq.ParquetFile(a.shard)
    docs = pf.to_pydict()
    nrows = len(docs["path"])
    by_repo = defaultdict(dict)
    for i in range(nrows):
        by_repo[docs["repo_name"][i]][docs["path"][i]] = docs["content"][i]

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, os.path.basename(a.shard).replace(".parquet", "_mined.jsonl"))
    counts = {"repos": len(by_repo), "test_files": 0, "passed": 0}
    npairs = 0
    with open(out, "w", encoding="utf-8") as fo:
        for repo, files in by_repo.items():
            test_paths = [p for p in files if TEST_RE.search(p)]
            impl_paths = [p for p in files if not TEST_RE.search(p)]
            for tp in test_paths:
                counts["test_files"] += 1
                src = files[tp]
                if not detects_unittest(src):
                    continue
                imported = imports(src) - {"unittest", "sys", "os"}
                stem = impl_stem_for_test(tp)
                target = (stem[:-3] if stem and stem.endswith(".py") else stem)
                if not target or target not in imported:
                    continue
                if not stdlib_core_ok(src):
                    continue
                cands = [p for p in impl_paths if os.path.basename(p) == (target + ".py")]
                for ip in cands:
                    impl = files[ip]
                    if impl is None or not stdlib_core_ok(impl):
                        continue
                    mod = target
                    blob = impl + "\n\nimport sys as _s; _s.modules[%r] = _s.modules['__main__']\n\n" % mod + src
                    rc, outt, err = sandbox_exec.run_sandboxed(blob, timeout=15)
                    if rc == 0:
                        fo.write(json.dumps({"repo": repo, "impl_path": ip, "impl": impl,
                                             "test_path": tp, "tests": src, "passed": True},
                                            ensure_ascii=False) + "\n")
                        counts["passed"] += 1
                        npairs += 1
                        break
                    if a.max_pairs and npairs >= a.max_pairs:
                        fo.flush()
                        print(json.dumps(counts)); print(f"WROTE {out} {npairs} pairs")
                        return
    print(json.dumps(counts))
    print(f"WROTE {out} {npairs} pairs")


if __name__ == "__main__":
    main()
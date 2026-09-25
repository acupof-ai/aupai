#!/usr/bin/env python3
"""3b 2026-09-25: build the post-SFT executable CODE RL problem pool.

This is DATA BUILDING, not training. Output is one jsonl of code problems whose
reference solution actually passes generated unit tests, aligned to
algorithms/code_reward.py's reward_fn(code, tests) contract. Each row:

    {id, source, prompt, impl, tests, difficulty, entry, n_tests, license,
     attribution}

  prompt = HumanEval-shaped RAW CONTINUATION (def signature + docstring), no
           ChatML -- same shape as SFT-A and the --rstrip_nl eval arm.
  impl   = reference solution written to solution.py. Class `Solution.method`
           references are normalised to a module-level def (self removed only
           when the body is self-free; self.* bodies are rejected, not patched).
           `from __future__ import annotations` is prepended so typing
           annotations (List[int]) do not need imports at run time.
  tests  = pytest parametrised source `from solution import fn`, asserting
           fn(*args) == expected, written to test_solution.py.

A row is emitted only when a reference solution passes ALL generated tests
(reward verdict rc0 + >=1 passed). Up to a few distinct solutions are tried in
length order; the first that passes is kept.

Gates in order, each counted independently:
  1. license/source filter (TACO: exclude HackerRank and rights-unknown; keep
     permissive + CC BY crawled with attribution; APPS MIT),
  2. call-style only (input_output.fn_name + per-call arg lists). stdin/stdout
     problems counted and excluded (code_reward has no subprocess stdin driver),
  3. HumanEval/MBPP 13-gram DROP on prompt and impl (filters/decontam_ngram),
  4. two-layer SFT-A dedup, both layers reported: normalised-question-text hash
     OR normalised signature+fn_name hash matching an APPS problem SFT-A used,
  5. reference solution executes and passes its tests.

CPU only. Trusted references, so on the digest box run with ALLOW_UNISOLATED=1
(rlimits/timeout only); pin cores and nice. Requires pytest in the interpreter.
# restartable: per-source streaming over local jsonl/parquet; an interrupt loses
# only the in-flight source and reruns to the same rows (rendering deterministic;
# pass/fail execution is the gate).
"""
import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "filters"))

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

from decontam_ngram import Decontaminator, decontam_fp  # noqa: E402

HE = os.path.join(ROOT, "data", "eval", "humaneval", "humaneval_164.jsonl")
MBPP = os.path.join(ROOT, "data", "eval", "mbpp_holdouts.jsonl")
_WS = re.compile(r"\s+")

# TACO sources whose rights are unclear / excluded by 1e order 2026-09-25.
TACO_EXCLUDE_SOURCES = {"HackerRank"}
# everything else in TACO is retained: permissive datasets and CC BY crawled,
# with attribution carried on every row.

FUTURE = "from __future__ import annotations\n"
# Standard-library import header prepended before the reference solution (1e
# order 2026-09-25). APPS/TACO references frequently use math/collections/etc.
# without importing them (the upstream evaluator injected them); this is a
# fixed, non-behaviour-changing header of stdlib modules only -- it does not
# patch the solution's logic, unlike rewriting self.* bodies. A solution that
# still fails after this header is dropped.
STD_IMPORTS = ("import math\n"
               "import re\n"
               "import string\n"
               "import bisect\n"
               "import heapq\n"
               "import itertools\n"
               "import functools\n"
               "import collections\n"
               "from collections import Counter, defaultdict, deque\n"
               "from functools import lru_cache, reduce, partial, cached_property\n"
               "from heapq import heappush, heappop, heapify, nlargest, nsmallest\n"
               "from itertools import permutations, combinations, product, accumulate, chain, groupby\n")
IMPL_HEADER = FUTURE + STD_IMPORTS


def norm_text(s):
    return _WS.sub(" ", (s or "").strip()).lower()


def qhash(s):
    import hashlib
    return hashlib.sha1(norm_text(s).encode("utf-8")).hexdigest()


def dedent(code):
    lines = code.split("\n")
    indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    cut = min(indents) if indents else 0
    return "\n".join(l[cut:] if l.strip() else "" for l in lines)


def _uses_self(method_body):
    tree = ast.parse(method_body)
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id == "self":
            return True
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self":
            return True
    return False


def normalize_solution(sol):
    """(impl_without_future, entry_short) or (None, None). entry_short='fn'."""
    try:
        tree = ast.parse((sol or "").strip("\n"))
    except SyntaxError:
        return None, None
    # top-level matching def
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            seg = ast.get_source_segment((sol or "").strip("\n"), node)
            if seg:
                return dedent(seg), node.name
    # class method -> module fn, only when the body is genuinely self-free
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and not m.name.startswith("_"):
                    seg = ast.get_source_segment((sol or "").strip("\n"), m)
                    if not seg:
                        continue
                    body = dedent(seg)
                    try:
                        if _uses_self(body):
                            continue
                    except SyntaxError:
                        continue
                    body = re.sub(r"(\bdef\s+" + re.escape(m.name) + r"\s*\(\s*)self\s*,?\s*",
                                  r"\1", body)
                    return body, m.name
    return None, None


def signature_from_starter(starter, fn_name):
    """Real `def fn(params):` header from starter_code, else def fn(*args).

    TACO codewars starters carry the true signature (with parameter names);
    APPS call-style starters are `class Solution: def fn(...)`, in which case
    the method header is extracted. Tabs are valid in the source; we keep them
    only inside the signature line, the docstring below is space-indented.
    """
    if starter:
        m = re.search(r"def\s+" + re.escape(fn_name) + r"\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:",
                      starter)
        if m:
            # class-method starter carries `self`; the normalised impl and the
            # tests expose a module function, so drop it from the signature too.
            return re.sub(r"(\(\s*)self\s*,?\s*", r"\1", m.group(0))
    return f"def {fn_name}(*args):"


def render_prompt(question, signature):
    q = (question or "").strip()
    doc = "\n".join("    " + ln for ln in q.split("\n"))
    return signature + "\n" + f'    """\n{doc}\n    """\n'


def render_tests(fn, inputs, outputs, max_cases):
    cases = [(a, e) for a, e in zip(inputs, outputs) if isinstance(a, list)][:max_cases]
    if not cases:
        return None
    lines = ["import pytest", "", "from solution import " + fn, "",
             "@pytest.mark.parametrize('args,expected', ["]
    for a, e in cases:
        try:
            lines.append(f"    ({a!r}, {e!r}),")
        except Exception:
            return None
    lines += ["])",
              f"def test_{fn}(args, expected):",
              f"    assert {fn}(*args) == expected", ""]
    return "\n".join(lines)


def execute(impl, tests, timeout):
    d = tempfile.mkdtemp(prefix="rlpool.")
    with open(os.path.join(d, "solution.py"), "w") as f:
        f.write(IMPL_HEADER + impl)
    with open(os.path.join(d, "test_solution.py"), "w") as f:
        f.write(tests)
    try:
        p = subprocess.run(
            [sys.executable, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider",
             "--no-header", "test_solution.py"],
            cwd=d, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    out = p.stdout
    if p.returncode != 0:
        return False, "rc%d" % p.returncode
    import re as _re
    m = _re.search(r"(\d+)\s+passed", out)
    if not m or int(m.group(1)) < 1:
        return False, "no_passed"
    if _re.search(r"\d+\s+(failed|error)", out):
        return False, "failed_or_error"
    return True, "%s passed" % m.group(1)


def candidate_impls(solutions_json, fn_name, limit):
    seen = set()
    out = []
    try:
        sols = json.loads(solutions_json or "[]")
    except (json.JSONDecodeError, TypeError):
        return out
    parsed = []
    for s in sols:
        impl, entry = normalize_solution(s)
        if not impl or entry != fn_name:
            continue
        if impl in seen:
            continue
        seen.add(impl)
        parsed.append(impl)
    parsed.sort(key=len)
    return parsed[:limit]


def sfta_dedup_sets(sfta_apps_jsonl, apps_raw):
    """Two exclusion hash layers from what SFT-A consumed (1e order 2026-09-25)."""
    import hashlib
    starters = set()
    if os.path.exists(sfta_apps_jsonl):
        for line in open(sfta_apps_jsonl):
            starters.add(norm_text(json.loads(line)["prompt"]))
    q_for_starter = {}
    if os.path.exists(apps_raw):
        for line in open(apps_raw):
            r = json.loads(line)
            st = (r.get("starter_code") or "").strip()
            if st:
                q_for_starter.setdefault(norm_text(st), norm_text(r.get("question")))
    sig_layer = {hashlib.sha1(s.encode()).hexdigest() for s in starters}
    q_layer = {hashlib.sha1(q.encode()).hexdigest() for q in q_for_starter.values() if q}
    return sig_layer, q_layer


def process_rows(rows, source, stats, args, dcon, sig_layer, q_layer, writer):
    for r in rows:
        stats["problems_total"] += 1
        src_label = r.get("source") or source
        if source == "taco" and src_label in TACO_EXCLUDE_SOURCES:
            stats["license_excluded"] += 1
            continue
        try:
            io = json.loads(r.get("input_output") or "{}")
        except (json.JSONDecodeError, TypeError, ValueError):
            stats["bad_io_json"] += 1
            continue
        fn = io.get("fn_name")
        inputs, outputs = io.get("inputs"), io.get("outputs")
        if not fn or not inputs or not isinstance(inputs[0], list):
            stats["stdin_stdout_excluded"] += 1
            continue
        stats["call_style"] += 1
        question = r.get("question") or r.get("instruction") or ""
        starter = (r.get("starter_code") or "").strip()
        signature = signature_from_starter(starter, fn)
        prompt = render_prompt(question, signature)
        impls = candidate_impls(r.get("solutions"), fn, args.max_solutions)
        if not impls:
            stats["no_normalizable_solution"] += 1
            continue
        # 13-gram on prompt and every impl; gate the prompt once and accept the
        # first clean impl.
        if dcon.hit(prompt):
            stats["ngram_prompt_drop"] += 1
            continue
        tests = render_tests(fn, inputs, outputs, args.max_cases)
        if tests is None:
            stats["unrenderable_tests"] += 1
            continue
        if dcon.hit(tests):
            stats["ngram_tests_drop"] += 1
            continue
        # two-layer SFT-A dedup: layer1 normalised question text, layer2 the
        # normalised starter signature+fn_name. Either hit excludes the problem.
        if qhash(question) in q_layer or qhash(starter) in sig_layer:
            stats["sfta_dedup"] += 1
            continue
        passed_impl = None
        tried = 0
        for impl in impls:
            if dcon.hit(IMPL_HEADER + impl):
                stats["ngram_impl_drop"] += 1
                continue
            tried += 1
            if args.verify:
                ok, why = execute(impl, tests, args.timeout)
                if not ok:
                    stats["ref_failed"] += 1
                    continue
            passed_impl = impl
            break
        if passed_impl is None:
            stats["no_passing_reference"] += 1
            continue
        stats["kept"] += 1
        diff = r.get("difficulty") or ""
        stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1
        row = {"id": f"{source}/{stats['problems_total']}",
               "source": source, "source_dataset": src_label,
               "prompt": prompt, "impl": IMPL_HEADER + passed_impl, "tests": tests,
               "difficulty": diff, "entry": fn,
               "n_tests": len([1 for a in inputs if isinstance(a, list)][:args.max_cases]),
               "license": args.license,
               "attribution": args.attribution.format(source=src_label)}
        writer.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_apps(path):
    for line in open(path):
        yield json.loads(line)


def iter_taco(raw_dir):
    import pyarrow.parquet as pq
    files = sorted(os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
                   if f.startswith("train-") and f.endswith(".parquet"))
    for f in files:
        tbl = pq.read_table(f)
        cols = {n: tbl.column(n).to_pylist() for n in tbl.column_names}
        for i in range(tbl.num_rows):
            yield {n: cols[n][i] for n in cols}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["apps", "taco", "both"], default="both")
    ap.add_argument("--apps-raw", default="")
    ap.add_argument("--taco-dir", default="")
    ap.add_argument("--sfta-apps", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-cases", type=int, default=20)
    ap.add_argument("--max-solutions", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dcon = Decontaminator.load_default(ROOT)
    sig_layer = q_layer = set()
    if args.sfta_apps and args.apps_raw:
        sig_layer, q_layer = sfta_dedup_sets(args.sfta_apps, args.apps_raw)

    plans = []
    if args.source in ("apps", "both"):
        plans.append(("apps", args.apps_raw, iter_apps,
                      "MIT", "codeparrot/apps train (MIT)"))
    if args.source in ("taco", "both"):
        plans.append(("taco", args.taco_dir, iter_taco,
                      "Apache-2.0 + source-specific (CC BY crawled kept; HackerRank excluded)",
                      "BAAI/TACO, source={source}"))

    for source, path, it, lic, attr in plans:
        stats = {"problems_total": 0, "call_style": 0, "stdin_stdout_excluded": 0,
                 "license_excluded": 0, "bad_io_json": 0, "no_normalizable_solution": 0,
                 "unrenderable_tests": 0, "ngram_prompt_drop": 0, "ngram_impl_drop": 0,
                 "ngram_tests_drop": 0, "sfta_dedup": 0, "ref_failed": 0,
                 "no_passing_reference": 0, "kept": 0, "by_difficulty": {}}
        a2 = argparse.Namespace(**vars(args), license=lic, attribution=attr)
        out_path = os.path.join(args.out_dir, f"rl_code_{source}.jsonl")
        with open(out_path, "w") as w:
            process_rows(it(path), source, stats, a2, dcon, sig_layer, q_layer, w)
        stats["_fingerprint"] = {
            "builder_sha256": hashlib.sha256(
                open(os.path.abspath(__file__), "rb").read()).hexdigest(),
            "decontam_ngram_n": 13,
            "decontam_fp_inputs": decontam_fp(HE, MBPP),
            "sfta_sig_layer_size": len(sig_layer),
            "sfta_question_layer_size": len(q_layer),
            "verify": bool(args.verify),
            "max_cases": args.max_cases,
        }
        with open(os.path.join(args.out_dir, f"rl_code_{source}_stats.json"), "w") as f:
            json.dump(stats, f, indent=1, ensure_ascii=False)
        print(source, json.dumps(stats, ensure_ascii=False), flush=True)


def _selftest():
    # Pure-function checks, no data files, no pytest execution: the transform
    # contract a later reader depends on. Execution pass/fail is the build gate
    # itself and is not faked here.
    # 1. class-method signature drops self and keeps the params/return
    sig = signature_from_starter(
        "\nclass Solution:\n    def f(self, a: int, b) -> int:\n        pass\n", "f")
    assert sig == "def f(self, a: int, b) -> int:".replace("self, ", ""), sig
    assert "self" not in sig and "a: int" in sig and "-> int" in sig, sig
    # 2. top-level starter keeps its params
    sig2 = signature_from_starter("def g(x, y):\n\t", "g")
    assert sig2 == "def g(x, y):", sig2
    # 3. missing starter falls back to *args
    assert signature_from_starter("", "h") == "def h(*args):"
    # 4. a self-free class method normalises to a module def without self
    impl, entry = normalize_solution(
        "class Solution:\n    def add(self, a, b):\n        return a + b\n")
    assert entry == "add" and impl.startswith("def add(a, b):"), impl
    # 5. a method whose body uses self.* is REJECTED, not patched
    bad, be = normalize_solution(
        "class S:\n    def f(self):\n        return self.x\n")
    assert bad is None and be is None, (bad, be)
    # 6. generated tests import the entry and parametrise over arg lists
    t = render_tests("add", [[[1, 2], [3, 4]]], [3, 7], 20)
    assert "from solution import add" in t and "add(*args) == expected" in t, t
    # 7. non-list (stdin-style) args produce no test file
    assert render_tests("f", ["4\n"], ["3"], 20) is None
    print("rl_code_pool selftest OK: signature/self/normalise/test-render contract")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()

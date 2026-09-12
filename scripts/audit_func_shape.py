#!/usr/bin/env python3
"""Audit how functions appear in the four gate code domains (ae-7, fb 2026-09-12).

Per domain, over a 20k-document deterministic stratified sample (one jittered
index per equal-size bin over the concatenation of all shards, seed 7):
  - function docstring rate (ast.get_docstring) and doctest rate within them
  - document shape: problem-statement -> complete-solution (L3 exercise form)
    vs raw source.

L3 documents are a prose statement followed by UNFENCED code, so the whole text
does not parse: for function metrics we parse the longest parseable code suffix
(starting at a def/class/import/assignment line) that contains a function def.
Shape classification is text-level (independent of parse): ps_solution needs an
imperative task instruction in the leading text, a `def`, and asserts/tests.

CPU only; reads JSONL {"content": ...}; writes JSON to --out.
"""
# restartable: read-only single pass over corpus shards; cost is IO, not compute
# (~20 min CPU, no GPU/card claim), and a rerun reproduces the same sample (seed 7).
import argparse
import ast
import contextlib
import glob
import hashlib
import json
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor

SAMPLE = 20000
SEED = 7
ROOT = os.environ.get("AUPAI_ROOT", "/work/aupai")

_TASK_RE = re.compile(
    r"\b(write|implement|create|define|build|complete|fix|refactor)\b[^.\n]{0,80}"
    r"\b(function|method|program|script|class|routine|solution|algorithm)\b",
    re.IGNORECASE,
)
_TASK_RE2 = re.compile(
    r"\b(your task|task:|complete the function|fill in)\b",
    re.IGNORECASE,
)
_DEF_RE = re.compile(r"(?m)^[ \t]*(?:async[ \t]+)?def[ \t]+\w+")
_CODE_START_RE = re.compile(r"(?m)^[ \t]*(?:async[ \t]+)?def[ \t]+\w+|^(?:class |import |from \w)")
_TEST_RE = re.compile(r"(?m)^\s*assert\b|# Test|def test_")
_DOCTEST_RE = re.compile(r"^\s*>>>", re.MULTILINE)


def parse_functions(content):
    """Return list of FunctionDef nodes: whole-doc parse, else longest code suffix."""
    try:
        tree = ast.parse(content)
        return [n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))], True
    except SyntaxError:
        pass
    best = []
    starts = [m.start() for m in _CODE_START_RE.finditer(content)]
    # try candidates last -> first; first parseable suffix wins (code runs to EOF)
    for s in reversed(starts):
        try:
            tree = ast.parse(content[s:])
        except SyntaxError:
            continue
        funcs = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if funcs:
            best = funcs
        break
    return best, False


def analyze(content):
    out = {"py_parseable_full": 0, "n_funcs": 0, "n_funcs_docstr": 0,
           "n_docstr_doctest": 0, "ps_solution": 0, "raw_source": 0}
    funcs, full = parse_functions(content)
    out["py_parseable_full"] = int(full)
    for fn in funcs:
        ds = ast.get_docstring(fn)
        if ds is not None:
            out["n_funcs_docstr"] += 1
            if _DOCTEST_RE.search(ds):
                out["n_docstr_doctest"] += 1
    out["n_funcs"] = len(funcs)

    head = "\n".join(content.splitlines()[:30])
    is_task = bool(_TASK_RE.search(head) or _TASK_RE2.search(head))
    has_def = bool(_DEF_RE.search(content))
    has_test = bool(_TEST_RE.search(content))
    if is_task and has_def and has_test:
        out["ps_solution"] = 1
    else:
        out["raw_source"] = 1
    return out


def _file_sample(args):
    """Single read: count rows, reservoir-keep up to K uniformly (per-file seed)."""
    path, k = args
    seed = int(hashlib.blake2b(os.path.basename(path).encode(), digest_size=4).hexdigest(), 16) ^ SEED
    rng = random.Random(seed)
    pool, n = [], 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n += 1
            if len(pool) < k:
                pool.append(line)
            else:
                j = rng.randrange(n)
                if j < k:
                    pool[j] = line
    return path, n, pool, seed


def iter_docs(domain, want, workers=32, cap=128):
    """Uniform sample over all rows: per-file reservoir cap, then proportional pick."""
    files = sorted(glob.glob(os.path.join(ROOT, "data", "corpus", domain, "*.jsonl")))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_file_sample, [(f, cap) for f in files]))
    total = sum(n for _, n, _, _ in parts)
    if total == 0:
        return
    if total <= want:
        for _, _, pool, _ in parts:
            for line in pool:
                with contextlib.suppress(ValueError):
                    yield json.loads(line).get("content", "") or ""
        return
    # exact per-file quotas summing to want, largest remainder
    raw = [(p, n, pool, seed, n * want / total) for p, n, pool, seed in parts]
    quotas = [int(x[4]) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i][4] - quotas[i], reverse=True)[:want - sum(quotas)]:
        quotas[i] += 1
    for (_, _, pool, seed, _), q in zip(raw, quotas, strict=True):
        if q > cap:
            raise SystemExit(f"{domain}: needs {q} > cap {cap}; raise cap")
        rng = random.Random(seed ^ 0x5EED)
        rng.shuffle(pool)
        for line in pool[:q]:
            with contextlib.suppress(ValueError):
                yield json.loads(line).get("content", "") or ""


def pct(x, y):
    return round(100 * x / y, 3) if y else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+",
                    default=["code_ultra_l2_dc", "code_ultra_l3_noexec_dc",
                             "code_py_starcoder_dc", "code_keep_p1_dc"])
    ap.add_argument("--sample", type=int, default=SAMPLE)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "func_shape_audit_0912.json"))
    a = ap.parse_args()
    report = {"n_sample_per_domain": a.sample, "seed": SEED, "domains": {}}
    for dom in a.domains:
        agg = dict(py_parseable_full=0, n_funcs=0, n_funcs_docstr=0,
                   n_docstr_doctest=0, ps_solution=0, raw_source=0, n_docs=0)
        for content in iter_docs(dom, a.sample):
            agg["n_docs"] += 1
            for k, v in analyze(content).items():
                agg[k] += v
        d, nf, nd = agg, agg["n_funcs"], agg["n_docs"]
        report["domains"][dom] = {
            "docs_sampled": nd,
            "docs_parsed_full_python_pct": pct(d["py_parseable_full"], nd),
            "functions_total": nf,
            "functions_with_docstring_pct": pct(d["n_funcs_docstr"], nf),
            "docstrings_with_doctest_pct": pct(d["n_docstr_doctest"], d["n_funcs_docstr"]),
            "functions_with_doctest_pct": pct(d["n_docstr_doctest"], nf),
            "docs_problem_solution_shape_pct": pct(d["ps_solution"], nd),
            "docs_raw_source_pct": pct(d["raw_source"], nd),
            "_counts": {"funcs_docstring": d["n_funcs_docstr"],
                        "docstrings_doctest": d["n_docstr_doctest"],
                        "ps_solution_docs": d["ps_solution"],
                        "raw_source_docs": d["raw_source"],
                        "full_parse_docs": d["py_parseable_full"]},
        }
        print(json.dumps({dom: report["domains"][dom]}, indent=1), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

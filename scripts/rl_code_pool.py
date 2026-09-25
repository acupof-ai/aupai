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
  2. call-style only by default (input_output.fn_name + per-call arg lists).
     With --include-stdin, stdin/stdout problems are built into a separate
     rl_code_<src>_stdin.jsonl (mode kind="stdin"): prompt is the statement as
     a module docstring continuation (plan A), impl is the reference run whole
     as a script, tests are [{input,output}] scored by code_reward.score_stdin
     (exact per-line compare after trailing-whitespace strip; per-case
     rel/abs float tolerance is a separate field, never fuzzy matching).
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
sys.path.insert(0, os.path.join(ROOT, "algorithms"))
sys.path.insert(0, os.path.join(ROOT, "filters"))

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

from decontam_ngram import Decontaminator, decontam_fp  # noqa: E402

HE = os.path.join(ROOT, "data", "eval", "humaneval", "humaneval_164.jsonl")
MBPP = os.path.join(ROOT, "data", "eval", "mbpp_holdouts.jsonl")
_WS = re.compile(r"\s+")

# TACO sources whose rights are unclear / excluded by 1e order 2026-09-25.
TACO_EXCLUDE_SOURCES = {"hackerrank"}  # TACO stores source labels lowercase
# everything else in TACO is retained: permissive datasets and CC BY crawled,
# with attribution carried on every row.

FUTURE = "from __future__ import annotations\n"
# Standard-library import header prepended before the reference solution (1e
# order 2026-09-25). APPS/TACO references frequently use math/collections/etc.
# without importing them (the upstream evaluator injected them); this is a
# fixed, non-behaviour-changing header of stdlib modules only -- it does not
# patch the solution's logic, unlike rewriting self.* bodies. A solution that
# still fails after this header is dropped.
STD_IMPORTS = (
    "import math\n"
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
    "from itertools import permutations, combinations, product, accumulate, chain, groupby\n"
)
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
                    body = re.sub(r"(\bdef\s+" + re.escape(m.name) + r"\s*\(\s*)self\s*,?\s*", r"\1", body)
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
        m = re.search(r"def\s+" + re.escape(fn_name) + r"\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:", starter)
        if m:
            # class-method starter carries `self`; the normalised impl and the
            # tests expose a module function, so drop it from the signature too.
            return re.sub(r"(\(\s*)self\s*,?\s*", r"\1", m.group(0))
    return f"def {fn_name}(*args):"


def render_prompt(question, signature):
    q = (question or "").strip()
    doc = "\n".join("    " + ln for ln in q.split("\n"))
    return signature + "\n" + f'    """\n{doc}\n    """\n'


# stdin/stdout prompt (1e plan A, 2026-09-25): the problem statement -- which
# carries the Input/Output format section and the statement's OWN sample I/O --
# is the module docstring, followed by a comment directing the continuation to
# a stdin->stdout program. No test cases are injected here: the docstring sample
# is the one the problem statement itself ships, never one chosen from the
# hidden input_output cases (that would leak evaluation data into the prompt).
_STDIN_LEAD = (
    "Read integers/strings from standard input in the input format "
    "described in the docstring and print the required answer to "
    "standard output."
)


def render_stdin_prompt(question):
    q = (question or "").strip()
    doc = "\n".join(q.split("\n"))
    return f'"""{doc}\n\n{_STDIN_LEAD}\n"""\n'


def stdin_cases(io, max_cases):
    """[{input,output}] for stdin-style problems; None when no usable string IO."""
    ins, outs = io.get("inputs"), io.get("outputs")
    if not ins or not isinstance(ins[0], str):
        return None
    cases = []
    for a, e in list(zip(ins, outs))[:max_cases]:
        if not isinstance(a, str) or not isinstance(e, str):
            return None
        cases.append({"input": a, "output": e})
    return cases or None


def render_tests(fn, inputs, outputs, max_cases):
    cases = [(a, e) for a, e in zip(inputs, outputs) if isinstance(a, list)][:max_cases]
    if not cases:
        return None
    lines = [
        "import pytest",
        "",
        "from solution import " + fn,
        "",
        "@pytest.mark.parametrize('args,expected', [",
    ]
    for a, e in cases:
        try:
            lines.append(f"    ({a!r}, {e!r}),")
        except Exception:
            return None
    lines += ["])", f"def test_{fn}(args, expected):", f"    assert {fn}(*args) == expected", ""]
    return "\n".join(lines)


def execute(impl, tests, timeout):
    d = tempfile.mkdtemp(prefix="rlpool.")
    with open(os.path.join(d, "solution.py"), "w") as f:
        f.write(IMPL_HEADER + impl)
    with open(os.path.join(d, "test_solution.py"), "w") as f:
        f.write(tests)
    try:
        p = subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--no-header",
                "test_solution.py",
            ],
            cwd=d,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
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


def execute_stdin(impl, cases, timeout):
    """(ok, reason, used_tol). Mirrors execute() but drives a full script with
    stdin and compares stdout via code_reward.score_stdin. Tries exact first;
    on a purely numeric mismatch retries with per-case float tolerance."""
    from code_reward import score_stdin

    r = score_stdin(IMPL_HEADER + impl, cases, timeout=timeout)
    if r["reward"] == 1.0:
        return True, r["reason"], False
    rt = score_stdin(
        IMPL_HEADER + impl, [dict(c, rel_tol=1e-6, abs_tol=1e-6) for c in cases], timeout=timeout
    )
    if rt["reward"] == 1.0:
        return True, rt["reason"] + " (numeric tolerance)", True
    return False, rt["reason"], False


def first_stdin_solution(solutions_json, limit):
    """Distinct parseable raw scripts up to limit; stdin refs are run whole
    (no def-main wrapping per plan A), so the source segment is the full file."""
    seen = set()
    out = []
    try:
        sols = json.loads(solutions_json or "[]")
    except (json.JSONDecodeError, TypeError):
        return out
    for s in sols:
        s = (s or "").strip("\n")
        if not s or s in seen:
            continue
        try:
            ast.parse(s)
        except SyntaxError:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= limit:
            break
    return out


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
        if source == "taco" and (src_label or "").lower() in TACO_EXCLUDE_SOURCES:
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
        row = {
            "id": f"{source}/{stats['problems_total']}",
            "source": source,
            "source_dataset": src_label,
            "prompt": prompt,
            "impl": IMPL_HEADER + passed_impl,
            "tests": tests,
            "difficulty": diff,
            "entry": fn,
            "n_tests": len([1 for a in inputs if isinstance(a, list)][: args.max_cases]),
            "license": args.license,
            "attribution": args.attribution.format(source=src_label),
        }
        writer.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_stdin_rows(rows, source, stats, args, dcon, sig_layer, q_layer, writer):
    """stdin/stdout problems: full-script prompt + score_stdin verification."""
    for r in rows:
        stats["problems_total"] += 1
        src_label = r.get("source") or source
        if source == "taco" and (src_label or "").lower() in TACO_EXCLUDE_SOURCES:
            stats["license_excluded"] += 1
            continue
        try:
            io = json.loads(r.get("input_output") or "{}")
        except (json.JSONDecodeError, TypeError, ValueError):
            stats["bad_io_json"] += 1
            continue
        if io.get("fn_name"):
            # owned by the call-style output; the subset whose inputs are strings
            # is excluded there as stdin_stdout_excluded (list args required)
            stats["fn_name_present"] += 1
            continue
        cases = stdin_cases(io, args.max_cases)
        if not cases:
            stats["no_string_io"] += 1
            continue
        stats["stdin_style"] += 1
        question = r.get("question") or r.get("instruction") or ""
        starter = (r.get("starter_code") or "").strip()
        prompt = render_stdin_prompt(question)
        impls = first_stdin_solution(r.get("solutions"), args.max_solutions)
        if not impls:
            stats["no_parseable_solution"] += 1
            continue
        if dcon.hit(prompt):
            stats["ngram_prompt_drop"] += 1
            continue
        if dcon.hit(json.dumps(cases)):
            stats["ngram_tests_drop"] += 1
            continue
        # two-layer SFT-A dedup (same two hashes as call-style)
        if qhash(question) in q_layer or qhash(starter) in sig_layer:
            stats["sfta_dedup"] += 1
            continue
        passed_impl = used_tol = None
        for impl in impls:
            if dcon.hit(IMPL_HEADER + impl):
                stats["ngram_impl_drop"] += 1
                continue
            ok, why, tol = execute_stdin(impl, cases, args.timeout)
            if ok:
                passed_impl, used_tol = impl, tol
                break
            stats["ref_failed"] += 1
        if passed_impl is None:
            stats["no_passing_reference"] += 1
            continue
        stats["kept"] += 1
        diff = r.get("difficulty") or ""
        stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1
        if used_tol:
            stats["numeric_tolerance"] += 1
        row = {
            "id": f"{source}_stdin/{stats['problems_total']}",
            "source": source,
            "source_dataset": src_label,
            "kind": "stdin",
            "prompt": prompt,
            "impl": IMPL_HEADER + passed_impl,
            "cases": cases,
            "numeric_tolerance": bool(used_tol),
            "difficulty": diff,
            "n_tests": len(cases),
            "license": args.license,
            "attribution": args.attribution.format(source=src_label),
        }
        writer.write(json.dumps(row, ensure_ascii=False) + "\n")


def _pool_files(out_dir):
    """The four pool outputs in canonical tie-break priority. A cross-source
    exact-prompt tie on test count keeps the EARLIER file here: call-style over
    stdin, APPS over TACO (APPS is the primary of the republished pair)."""
    names = [
        "rl_code_apps.jsonl",
        "rl_code_taco.jsonl",
        "rl_code_apps_stdin.jsonl",
        "rl_code_taco_stdin.jsonl",
    ]
    return [os.path.join(out_dir, n) for n in names if os.path.exists(os.path.join(out_dir, n))]


def dedupe_pool(out_dir):
    """Global exact-prompt dedup across every pool file (1e order 2026-09-25).

    Key = sha1 of norm_text(prompt), the SAME normalisation as the SFT-A hash
    layer. Within-file and cross-file duplicates both collapse; the survivor is
    the row with MORE tests (len(cases)/n_tests), ties broken by _pool_files
    priority then original line order. Files are rewritten in place with the
    survivors in original order; per-file *_stats.json gains dedup counters and
    one rl_code_global_dedup_stats.json records the rule and final unique count.
    Returns the global stats dict.
    """
    files = _pool_files(out_dir)

    def ntests(row):
        return len(row["cases"]) if row.get("kind") == "stdin" else row["n_tests"]

    # pass 1: winning (file_idx, line_idx) per normalised-prompt key; more tests
    # wins, a count tie falls back to _pool_files priority (call-style, apps,
    # taco), then first line.
    winner = {}
    for fi, path in enumerate(files):
        with open(path) as fh:
            for li, line in enumerate(fh):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = qhash(row["prompt"])
                cur = winner.get(key)
                if cur is None or ntests(row) > cur[2]:
                    winner[key] = (fi, li, ntests(row))
    win_ids = {(fi, li) for fi, li, _ in winner.values()}
    # pass 2: rewrite each file keeping its winners in original order; a dropped
    # row is within-file when another row of the SAME file won its key, else it
    # was lost to a different file (cross-source republish).
    final_counts, within_dropped, cross_dropped = {}, {}, {}
    for fi, path in enumerate(files):
        keep, wd, cd = [], 0, 0
        with open(path) as fh:
            for li, line in enumerate(fh):
                if not line.strip():
                    continue
                if (fi, li) in win_ids:
                    keep.append(line if line.endswith("\n") else line + "\n")
                else:
                    key = qhash(json.loads(line)["prompt"])
                    if winner[key][0] == fi:
                        wd += 1
                    else:
                        cd += 1
        with open(path, "w") as fh:
            fh.writelines(keep)
        base = os.path.basename(path)
        final_counts[base] = len(keep)
        within_dropped[base] = wd
        cross_dropped[base] = cd
        sp = path.replace(".jsonl", "_stats.json")
        stats = json.load(open(sp)) if os.path.exists(sp) else {}
        stats["global_dedup"] = {
            "within_file_dropped": wd,
            "lost_to_other_file": cd,
            "final_rows": len(keep),
            "rule": "sha1(norm_text(prompt)) exact; keep more tests; "
            "tie: call-style, apps-stdin, taco-stdin, then line order",
        }
        json.dump(stats, open(sp, "w"), indent=1, ensure_ascii=False)
    gstats = {
        "rule": "global exact-prompt dedup across rl_code_* pool files; "
        "key=sha1(norm_text(prompt)); survivor=max(test count); "
        "tie priority=call-style, apps, taco",
        "builder_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "files": final_counts,
        "within_file_dropped": sum(within_dropped.values()),
        "cross_file_dropped": sum(cross_dropped.values()),
        "per_file_within_dropped": within_dropped,
        "per_file_cross_dropped": cross_dropped,
        "unique_total": len(winner),
    }
    json.dump(
        gstats,
        open(os.path.join(out_dir, "rl_code_global_dedup_stats.json"), "w"),
        indent=1,
        ensure_ascii=False,
    )
    print("global dedup:", json.dumps(gstats, ensure_ascii=False), flush=True)
    return gstats


def _cooked_cases(row):
    """The case list with tolerance applied to a numeric-tolerance row."""
    if row.get("kind") == "stdin" and row.get("numeric_tolerance"):
        return [dict(c, rel_tol=1e-6, abs_tol=1e-6) for c in row["cases"]]
    return row.get("cases") if row.get("kind") == "stdin" else None


def _nondet_worker(job):
    """Module-level (picklable) probe of one pool row: run its reference twice,
    one case at a time so the passing path exposes stdout. Returns
    (bad, reason)."""
    row, timeout = job
    from code_reward import score, score_stdin

    if row.get("kind") == "stdin":
        outs = []
        for _ in range(2):
            one = []
            for c in _cooked_cases(row):
                r = score_stdin(row["impl"], [c], timeout=timeout, generated=False)
                if r["reward"] != 1.0:
                    return True, "replay_not_1.0:" + r["reason"]
                one.append(r["stdout"])
            outs.append(one)
        if outs[0] != outs[1]:
            return True, "stdout_drifts_between_runs"
        return False, ""
    r1 = score(row["impl"], row["tests"], timeout=timeout, generated=False)
    r2 = score(row["impl"], row["tests"], timeout=timeout, generated=False)
    if r1["reward"] != 1.0 or r2["reward"] != 1.0:
        return True, "pytest_replay_not_1.0"
    return False, ""


def nondet_screen(out_dir, timeout, workers):
    """Determinism gate for the binary reward (code_reward doc: nondeterminism
    is the reward's enemy). Each surviving reference runs TWICE, one case at a
    time so stdout is observable on the passing path. A row stays only when both
    runs reward 1.0 AND print byte-identical stdout on every case: a reference
    emitting different valid answers (set/hash/random iteration, e.g. TACO
    multi-solution problems) makes a correct rollout's score a coin flip. Dropped
    rows are quarantined to rl_<name>_nondet.jsonl, not deleted. imap keeps
    order, so result index is the row index. Re-dedupes at the end: a dropped
    survivor cannot resurrect a duplicate."""
    from multiprocessing import Pool

    files = [p for p in _pool_files(out_dir) if os.path.getsize(p) > 0]
    per_file = {}
    for p in files:
        rows = [json.loads(l) for l in open(p) if l.strip()]
        drops = {}
        with Pool(processes=workers) as pool:
            for idx, (bad, why) in enumerate(
                pool.imap(_nondet_worker, ((r, timeout) for r in rows), chunksize=16)
            ):
                if bad:
                    drops[idx] = why
        keep, quar = [], []
        for idx, row in enumerate(rows):
            if idx in drops:
                row["_nondet_reason"] = drops[idx]
                quar.append(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                keep.append(json.dumps(row, ensure_ascii=False) + "\n")
        with open(p, "w") as fh:
            fh.writelines(keep)
        with open(p.replace(".jsonl", "_nondet.jsonl"), "w") as fh:
            fh.writelines(quar)
        per_file[os.path.basename(p)] = {"dropped": len(quar), "kept": len(keep)}
    gstats = {
        "rule": "reference executed twice; kept iff both reward 1.0 with byte-identical "
        "stdout per case; nondeterministic references quarantined",
        "builder_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "timeout_s": timeout,
        "workers": workers,
        "files": per_file,
        "dropped_total": sum(v["dropped"] for v in per_file.values()),
    }
    json.dump(
        gstats,
        open(os.path.join(out_dir, "rl_code_nondet_screen_stats.json"), "w"),
        indent=1,
        ensure_ascii=False,
    )
    dedupe_pool(out_dir)
    print("nondet screen:", json.dumps(gstats, ensure_ascii=False), flush=True)
    return gstats


def iter_apps(path):
    for line in open(path):
        yield json.loads(line)


def iter_taco(raw_dir):
    import pyarrow.parquet as pq

    files = sorted(
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.startswith("train-") and f.endswith(".parquet")
    )
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
    ap.add_argument(
        "--include-stdin",
        action="store_true",
        help="also build the stdin/stdout pool (rl_code_<src>_stdin.jsonl)",
    )
    ap.add_argument(
        "--dedupe-only",
        action="store_true",
        help="skip the build; global-dedupe the existing pool files in --out-dir",
    )
    ap.add_argument(
        "--nondet-only",
        action="store_true",
        help="skip the build; run the two-run determinism screen on the existing pool",
    )
    ap.add_argument("--nondet-workers", type=int, default=8)
    ap.add_argument(
        "--nondet-screen",
        action="store_true",
        help="after build+dedupe, run each surviving reference twice and drop stdout-drifters",
    )
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.dedupe_only:
        dedupe_pool(args.out_dir)
        return
    if args.nondet_only:
        nondet_screen(args.out_dir, args.timeout, args.nondet_workers)
        return
    dcon = Decontaminator.load_default(ROOT)
    sig_layer = q_layer = set()
    if args.sfta_apps and args.apps_raw:
        sig_layer, q_layer = sfta_dedup_sets(args.sfta_apps, args.apps_raw)

    plans = []
    if args.source in ("apps", "both"):
        plans.append(("apps", args.apps_raw, iter_apps, "MIT", "codeparrot/apps train (MIT)"))
    if args.source in ("taco", "both"):
        plans.append(
            (
                "taco",
                args.taco_dir,
                iter_taco,
                "Apache-2.0 + source-specific (CC BY crawled kept; HackerRank excluded)",
                "BAAI/TACO, source={source}",
            )
        )

    for source, path, it, lic, attr in plans:
        stats = {
            "problems_total": 0,
            "call_style": 0,
            "stdin_stdout_excluded": 0,
            "license_excluded": 0,
            "bad_io_json": 0,
            "no_normalizable_solution": 0,
            "unrenderable_tests": 0,
            "ngram_prompt_drop": 0,
            "ngram_impl_drop": 0,
            "ngram_tests_drop": 0,
            "sfta_dedup": 0,
            "ref_failed": 0,
            "no_passing_reference": 0,
            "kept": 0,
            "by_difficulty": {},
        }
        a2 = argparse.Namespace(**vars(args), license=lic, attribution=attr)
        out_path = os.path.join(args.out_dir, f"rl_code_{source}.jsonl")
        with open(out_path, "w") as w:
            process_rows(it(path), source, stats, a2, dcon, sig_layer, q_layer, w)
        stats["_fingerprint"] = {
            "builder_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
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

        if args.include_stdin:
            sstats = {
                "problems_total": 0,
                "stdin_style": 0,
                "fn_name_present": 0,
                "license_excluded": 0,
                "bad_io_json": 0,
                "no_string_io": 0,
                "no_parseable_solution": 0,
                "ngram_prompt_drop": 0,
                "ngram_impl_drop": 0,
                "ngram_tests_drop": 0,
                "sfta_dedup": 0,
                "ref_failed": 0,
                "no_passing_reference": 0,
                "numeric_tolerance": 0,
                "kept": 0,
                "by_difficulty": {},
            }
            spath = os.path.join(args.out_dir, f"rl_code_{source}_stdin.jsonl")
            with open(spath, "w") as w:
                process_stdin_rows(it(path), source, sstats, a2, dcon, sig_layer, q_layer, w)
            sstats["_fingerprint"] = stats["_fingerprint"]
            with open(os.path.join(args.out_dir, f"rl_code_{source}_stdin_stats.json"), "w") as f:
                json.dump(sstats, f, indent=1, ensure_ascii=False)
            print(source, "stdin", json.dumps(sstats, ensure_ascii=False), flush=True)

    dedupe_pool(args.out_dir)


def _selftest():
    # Pure-function checks, no data files, no pytest execution: the transform
    # contract a later reader depends on. Execution pass/fail is the build gate
    # itself and is not faked here.
    # 1. class-method signature drops self and keeps the params/return
    sig = signature_from_starter("\nclass Solution:\n    def f(self, a: int, b) -> int:\n        pass\n", "f")
    assert sig == "def f(self, a: int, b) -> int:".replace("self, ", ""), sig
    assert "self" not in sig and "a: int" in sig and "-> int" in sig, sig
    # 2. top-level starter keeps its params
    sig2 = signature_from_starter("def g(x, y):\n\t", "g")
    assert sig2 == "def g(x, y):", sig2
    # 3. missing starter falls back to *args
    assert signature_from_starter("", "h") == "def h(*args):"
    # 4. a self-free class method normalises to a module def without self
    impl, entry = normalize_solution("class Solution:\n    def add(self, a, b):\n        return a + b\n")
    assert entry == "add" and impl.startswith("def add(a, b):"), impl
    # 5. a method whose body uses self.* is REJECTED, not patched
    bad, be = normalize_solution("class S:\n    def f(self):\n        return self.x\n")
    assert bad is None and be is None, (bad, be)
    # 6. generated tests import the entry and parametrise over arg lists
    t = render_tests("add", [[[1, 2], [3, 4]]], [3, 7], 20)
    assert "from solution import add" in t and "add(*args) == expected" in t, t
    # 7. non-list (stdin-style) args produce no test file
    assert render_tests("f", ["4\n"], ["3"], 20) is None
    # 8. stdin prompt is a module docstring continuation carrying the statement
    sp = render_stdin_prompt("Sum two ints.\n\nInput: a b\nOutput: a+b")
    assert sp.startswith('"""Sum two ints.') and sp.rstrip().endswith('"""'), sp
    assert "standard input" in sp and "standard output" in sp, sp
    # 9. stdin_cases keeps only all-string IO and caps the count
    cs = stdin_cases({"inputs": ["1 2\n", "3 4\n"], "outputs": ["3\n", "7\n"]}, 20)
    assert cs == [{"input": "1 2\n", "output": "3\n"}, {"input": "3 4\n", "output": "7\n"}], cs
    assert stdin_cases({"inputs": [["1", "2"]], "outputs": [["3"]]}, 20) is None
    assert stdin_cases({"inputs": ["1\n", "x\n"], "outputs": ["1\n", 2]}, 20) is None
    assert len(stdin_cases({"inputs": ["1\n"] * 9, "outputs": ["1\n"] * 9}, 3)) == 3
    # 10. first_stdin_solution parses/dedups raw whole scripts, rejects syntax errors
    fs = first_stdin_solution(json.dumps(["import sys\nprint(1)", "import sys\nprint(1)", "def f(:\n"]), 5)
    assert fs == ["import sys\nprint(1)"], fs
    print("rl_code_pool selftest OK: signature/self/normalise/test-render/stdin contract")


def _selftest_dedupe():
    import tempfile

    d = tempfile.mkdtemp(prefix="rlpool_dedupe_test.")

    def w(name, rows):
        with open(os.path.join(d, name), "w") as f:
            for i, (prompt, n, kind) in enumerate(rows):
                row = {"id": f"{name}:{i}", "prompt": prompt, "n_tests": n}
                if kind == "stdin":
                    row["kind"] = "stdin"
                    row["cases"] = [{"input": str(i), "output": str(i)} for i in range(n)]
                f.write(json.dumps(row) + "\n")
        json.dump({"kept": len(rows)}, open(os.path.join(d, name.replace(".jsonl", "_stats.json")), "w"))

    # within-file dup (keep more tests); cross-file dup (taco-stdin loses to apps-stdin)
    w("rl_code_apps_stdin.jsonl", [("Problem A", 5, "stdin"), ("Problem B", 3, "stdin")])
    w(
        "rl_code_taco_stdin.jsonl",
        [
            ("problem a", 9, "stdin"),  # within-taco dup of A, more tests -> wins over apps copy
            ("PROBLEM  a", 2, "stdin"),  # whitespace-normalises to same A, fewer tests -> dropped
            ("Problem C", 1, "stdin"),
        ],
    )  # unique
    g = dedupe_pool(d)
    ta = [json.loads(l) for l in open(os.path.join(d, "rl_code_taco_stdin.jsonl"))]
    ap = [json.loads(l) for l in open(os.path.join(d, "rl_code_apps_stdin.jsonl"))]
    # A: 9-test taco copy is the global winner -> apps copy becomes cross-file loss
    assert len(ta) == 2 and {r["id"] for r in ta} == {
        "rl_code_taco_stdin.jsonl:0",
        "rl_code_taco_stdin.jsonl:2",
    }, ta
    assert len(ap) == 1 and ap[0]["id"] == "rl_code_apps_stdin.jsonl:1", ap
    assert g["unique_total"] == 3, g
    assert g["per_file_within_dropped"]["rl_code_taco_stdin.jsonl"] == 1, g
    assert g["per_file_cross_dropped"]["rl_code_apps_stdin.jsonl"] == 1, g
    # idempotent
    g2 = dedupe_pool(d)
    assert g2["unique_total"] == 3 and g2["within_file_dropped"] == 0 and g2["cross_file_dropped"] == 0, g2
    print("rl_code_pool dedupe selftest OK: within/cross/tie/idempotent")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        _selftest_dedupe()
    else:
        main()

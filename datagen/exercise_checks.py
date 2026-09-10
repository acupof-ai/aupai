#!/usr/bin/env python3
"""Acceptance checks for the synthetic exercise set (docs/standards/p1_data_recipe.md, 44's line).

Four checks, each with a recorded result:
  execution   every exercise's solution passes its own tests; failures are discarded and the
              discard rate is recorded (criterion 1)
  decontam    exact + containment match against HumanEval 164 and MBPP holdouts; a planted
              HumanEval problem MUST be caught -- the known-positive control is the load-bearing
              half, because a decontamination that reports "0 hits" is indistinguishable from one
              that never ran (criterion 2; facts/contamination.json#cont.split is the prior hit)
  diversity   topic distribution over the docstrings; one topic above 30% is a FAIL -- 180M tokens
              of sorting/fibonacci is the failure this exists to name (criterion 3)
  spotcheck   a seeded 50-sample draw to a review sheet; --agreement scores two filled sheets
              (criterion 4)

Records are code_if-shaped: {"prompt": "def f(...):\n    \"\"\"...\"\"\"\n", "output": "    body",
"tests": "def check():\n    assert f(...) == ..."}. "solution" is accepted as an alias for
"output" so teacher-generated records need no rename.

The execution scorer is the same shape as eval/humaneval_gen.py::judge -- in-process exec with a
6s SIGALRM ceiling -- so an exercise that passes here passes the eval harness's judge. The ceiling
is the same as the judge's: a record that segfaults or os._exits takes the batch with it, which is
acceptable for a teacher-generated corpus and not for untrusted input; chunked subprocess workers
are the upgrade if that changes.

restartable: yes -- read-only over the input, writes only the sheet named by --spotcheck-out.
"""
import argparse
import contextlib
import io
import json
import os
import random
import re
import signal
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# restartable: read-only over the input jsonl; the only write is the sheet named by
# --spotcheck-out, which is a fresh seeded draw and safe to overwrite by re-running.
HUMANEVAL = os.path.join(ROOT, "data", "eval", "humaneval", "humaneval_164.jsonl")
MBPP = os.path.join(ROOT, "data", "eval", "mbpp_holdouts.jsonl")
EXEC_TIMEOUT_S = 6
DOMINANCE = 0.30


class _TO(Exception):
    pass


def _alarm(*_a):
    raise _TO()


signal.signal(signal.SIGALRM, _alarm)


def _solution(rec):
    return rec.get("solution") if rec.get("solution") is not None else rec.get("output", "")


def judge(rec):
    """prompt + solution + tests + check(); pass iff clean exit. Same shape as humaneval_gen.judge."""
    src = rec["prompt"] + _solution(rec) + "\n" + rec["tests"] + "\ncheck()\n"
    g = {"__name__": "__main__"}
    signal.alarm(EXEC_TIMEOUT_S)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(src, g)
        return True
    except BaseException:
        return False
    finally:
        signal.alarm(0)


def check_execution(records):
    """(passed, failed_idx) -- every record's solution must pass its own tests."""
    failed = [i for i, r in enumerate(records) if not judge(r)]
    return len(records) - len(failed), failed


def _norm(s):
    return re.sub(r"\s+", " ", s).strip()


def _load_benchmarks(humaneval_path, mbpp_path):
    """[(source, task_id, key_text, match_text)] -- key_text for containment, match_text for exact."""
    bench = []
    for path, source in ((humaneval_path, "humaneval"), (mbpp_path, "mbpp")):
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if source == "humaneval":
                prompt, sol = r["prompt"], r.get("canonical_solution", "")
                bench.append((source, r["task_id"], _norm(prompt), _norm(prompt + sol)))
            else:
                bench.append((source, r["task_id"], _norm(r.get("text", "")), _norm(r.get("code", ""))))
    return bench


def check_decontam(records, bench, planted_idx=None):
    """(hits, control_caught) -- exact or containment match against the benchmarks.

    Exact: normalized prompt+solution equals a benchmark's normalized prompt+solution.
    Containment: a benchmark prompt (>=100 chars normalized) appears inside the exercise's
    normalized prompt, or an MBPP problem statement (>=40 chars) inside its docstring.
    A planted known-positive (an exercise copied from a benchmark problem) MUST be caught.
    """
    hits = []
    for i, r in enumerate(records):
        body = _norm(r["prompt"] + _solution(r))
        doc = _norm(r["prompt"])
        for source, task_id, key, match in bench:
            if match and body == match:
                hits.append((i, source, task_id, "exact"))
                break
            if key and (len(key) >= 100 and key in doc or len(key) >= 40 and source == "mbpp" and key in doc):
                hits.append((i, source, task_id, "containment"))
                break
    control_caught = None
    if planted_idx is not None:
        control_caught = any(i == planted_idx for i, *_ in hits)
    return hits, control_caught


_TOPICS = [
    ("sorting", r"sort|order|merge|quick|heap|rank"),
    ("string", r"string|substring|palindrome|anagram|char|regex|pattern match"),
    ("math", r"prime|factor|gcd|lcm|fibonacci|factorial|power|sqrt|matrix|geometry|triangle|circle"),
    ("recursion_dp", r"recurs|dynamic|memoiz|subsequence|knapsack|longest|optimal"),
    ("tree_graph", r"tree|graph|node|traversal|dfs|bfs|binary|heap|queue|stack|linked"),
    ("hash", r"dict|hash|map|set|count|frequency|unique|duplicate"),
    ("list_array", r"list|array|matrix|grid|index|slice|rotate|reverse|flatten"),
    ("io_text", r"file|read|write|parse|csv|json|format|print|input"),
    ("datetime", r"date|time|year|month|day|calendar|timezone"),
    ("class_oop", r"class |object|inherit|method|constructor|__init__"),
]
_TOPIC_RE = [(name, re.compile(pat, re.I)) for name, pat in _TOPICS]


def _topic(rec):
    doc = rec["prompt"]
    for name, rx in _TOPIC_RE:
        if rx.search(doc):
            return name
    return "misc"


def check_diversity(records):
    """(table, max_share) -- topic distribution; FAIL is the caller's (max_share > DOMINANCE)."""
    table = {}
    for r in records:
        t = _topic(r)
        table[t] = table.get(t, 0) + 1
    n = len(records) or 1
    return {t: c for t, c in sorted(table.items(), key=lambda kv: -kv[1])}, max(c / n for c in table.values())


def make_sheet(records, seed, out_path, n=50):
    """A seeded n-sample draw with blank verdict fields for two readers."""
    rng = random.Random(seed)
    sample = rng.sample(range(len(records)), min(n, len(records)))
    with open(out_path, "w", encoding="utf-8") as fh:
        for i in sample:
            r = dict(records[i])
            r["_idx"] = i
            r["verdict"] = ""
            r["notes"] = ""
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(sample)


def agreement(a_path, b_path):
    """(n, agree, pct, kappa) over two filled sheets, joined by _idx."""
    def load(p):
        return {r["_idx"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8"))}
    a, b = load(a_path), load(b_path)
    keys = sorted(set(a) & set(b))
    if not keys:
        return 0, 0, 0.0, 0.0
    agree = sum(1 for k in keys if a[k].get("verdict", "").strip() == b[k].get("verdict", "").strip())
    n = len(keys)
    p0 = agree / n
    labels = sorted({a[k].get("verdict", "").strip() for k in keys}
                    | {b[k].get("verdict", "").strip() for k in keys})
    # Cohen's kappa against the observed marginals.
    pe = sum(sum(a[k].get("verdict", "").strip() == l for k in keys) / n
             * sum(b[k].get("verdict", "").strip() == l for k in keys) / n for l in labels)
    kappa = (p0 - pe) / (1 - pe) if pe < 1 else 1.0
    return n, agree, p0, kappa


# ---------------------------------------------------------------------------------------------
# Selftest: synthetic worlds for each check. Run with --selftest.

def _rec(prompt, solution, tests):
    return {"prompt": prompt, "output": solution, "tests": tests}


def _selftest():
    # execution: a passing exercise, a wrong solution, a timeout.
    good = _rec("def add(a, b):\n    \"\"\"Return a+b.\"\"\"\n", "    return a + b\n",
                "def check():\n    assert add(1, 2) == 3\n")
    bad = _rec("def add(a, b):\n    \"\"\"Return a+b.\"\"\"\n", "    return a - b\n",
               "def check():\n    assert add(1, 2) == 3\n")
    slow = _rec("def f():\n    \"\"\"hang\"\"\"\n", "    while True:\n        pass\n",
                "def check():\n    assert f() is None\n")
    passed, failed = check_execution([good, bad, slow])
    assert passed == 1 and failed == [1, 2], f"execution: {passed} {failed}"

    # decontamination: a planted HumanEval problem must be caught; a clean exercise must not be.
    # The bench prompt is long on purpose: real HumanEval prompts carry a docstring, and the
    # containment floor exists to stop a 20-char signature matching half the corpus.
    he_prompt = ("def has_close_elements(numbers: list, threshold: float) -> bool:\n"
                 "    \"\"\"Check if in given list of numbers, are any two closer than the threshold.\n"
                 "    >>> has_close_elements([1.0, 2.0], 0.5)\n"
                 "    False\n"
                 "    \"\"\"\n")
    bench = [("humaneval", "HumanEval/0", _norm(he_prompt), _norm(he_prompt + "    return True\n")),
             ("mbpp", "mbpp-train-0", "find the longest chain which can be formed from the given set of pairs",
              "class Pair: pass")]
    planted = _rec(he_prompt, "    return True\n", "def check(): pass\n")
    clean = _rec("def totally_unrelated(x):\n    \"\"\"A function nobody benchmarked.\"\"\"\n", "    return x\n",
                 "def check():\n    assert totally_unrelated(1) == 1\n")
    hits, control = check_decontam([planted, clean], bench, planted_idx=0)
    assert control is True, f"known-positive control not caught: {hits}"
    assert any(i == 1 for i, *_ in hits) is False, f"clean exercise flagged: {hits}"
    # containment half: the benchmark prompt inside a longer exercise prompt, different body.
    cont = _rec(he_prompt + "\n# adapted with an extra note\n", "    return len(numbers) > 0\n",
                "def check(): pass\n")
    hits_c, _ = check_decontam([cont], bench)
    assert any(kind == "containment" for *_rest, kind in hits_c), f"containment missed: {hits_c}"
    # exact half: prompt+solution equal to a benchmark entry, even under a different idx.
    exact = _rec(he_prompt, "    return True\n", "def check(): pass\n")
    hits2, _ = check_decontam([exact], bench)
    assert any(kind == "exact" for *_rest, kind in hits2), f"exact match missed: {hits2}"
    # a missing benchmark file is not a clean pass.
    assert _load_benchmarks("/nonexistent", "/nonexistent") == [], "missing benchmarks must be empty, not skipped"

    # diversity: a one-topic set trips the dominance guard; a mixed set does not.
    mono = [_rec(f"def f{i}(x):\n    \"\"\"sort the list\"\"\"\n", "    pass\n", "def check(): pass\n")
            for i in range(10)]
    _, mono_max = check_diversity(mono)
    assert mono_max > DOMINANCE, f"monoculture not flagged: {mono_max}"
    mixed = (mono[:3]
             + [_rec("def g(x):\n    \"\"\"parse the file\"\"\"\n", "    pass\n", "def check(): pass\n")] * 3
             + [_rec("def h(x):\n    \"\"\"is the number prime\"\"\"\n", "    pass\n", "def check(): pass\n")] * 2
             + [_rec("def i(x):\n    \"\"\"walk the tree node\"\"\"\n", "    pass\n", "def check(): pass\n")] * 2)
    _, mixed_max = check_diversity(mixed)
    assert mixed_max <= DOMINANCE, f"mixed set flagged: {mixed_max}"

    # spotcheck: the draw is seeded and reproducible; agreement scores two filled sheets.
    with tempfile.TemporaryDirectory() as d:
        recs = [_rec(f"def f{i}():\n    \"\"\"doc\"\"\"\n", "    pass\n", "def check(): pass\n") for i in range(100)]
        s1, s2 = os.path.join(d, "s1.jsonl"), os.path.join(d, "s2.jsonl")
        assert make_sheet(recs, 7, s1) == make_sheet(recs, 7, s2) == 50
        rows = [json.loads(l) for l in open(s1)]
        for r in rows:
            r["verdict"] = "ok"
        with open(s1, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        with open(s2, "w") as fh:
            for r in rows:
                r2 = dict(r)
                if r2["_idx"] % 10 == 0:
                    r2["verdict"] = "bad"
                fh.write(json.dumps(r2) + "\n")
        n, agree, pct, kappa = agreement(s1, s2)
        flipped = sum(1 for r in rows if r["_idx"] % 10 == 0)
        assert n == 50 and agree == 50 - flipped, f"agreement: {n} {agree}, flipped {flipped}"
        assert abs(pct - agree / 50) < 1e-9, f"pct {pct}"

    print("exercise_checks selftest OK: execution (pass/fail/timeout), decontamination "
          "(planted control caught, clean clean, exact caught, missing benchmarks empty), "
          "diversity (monoculture flagged, mixed clean), spotcheck (seeded, agreement 45/50)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("exercises", nargs="?", help="jsonl of exercise records")
    ap.add_argument("--checks", default="execution,decontam,diversity",
                    help="comma list; spotcheck runs via --spotcheck-out")
    ap.add_argument("--humaneval", default=HUMANEVAL)
    ap.add_argument("--mbpp", default=MBPP)
    ap.add_argument("--spotcheck-out", help="write a seeded 50-sample review sheet and exit")
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--agreement", nargs=2, metavar=("A", "B"), help="score two filled sheets")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if args.agreement:
        n, agree, pct, kappa = agreement(*args.agreement)
        print(f"agreement: {agree}/{n} = {pct:.1%}, Cohen's kappa {kappa:.3f}")
        return 0 if pct >= 0.8 else 1
    if not args.exercises:
        ap.error("exercises jsonl required (or --selftest / --agreement)")

    records = [json.loads(l) for l in open(args.exercises, encoding="utf-8")]
    print(f"loaded {len(records)} exercises")
    want = set(args.checks.split(","))
    rc = 0

    if "execution" in want:
        passed, failed = check_execution(records)
        rate = len(failed) / len(records) if records else 0.0
        print(f"execution: {passed}/{len(records)} pass, discard rate {rate:.2%}")
        if failed:
            print(f"  discarded idx (first 10): {failed[:10]}")
        records = [r for i, r in enumerate(records) if i not in set(failed)]
        print(f"  {len(records)} exercises survive to the next check")

    if "decontam" in want:
        bench = _load_benchmarks(args.humaneval, args.mbpp)
        if not bench:
            print("decontam: FAIL -- no benchmark problems loaded; a 0-hit report would be meaningless")
            return 1
        hits, _ = check_decontam(records, bench)
        print(f"decontam: {len(hits)} hit(s) against {len(bench)} benchmark problems")
        for i, source, task_id, kind in hits[:20]:
            print(f"  idx {i}: {source} {task_id} ({kind})")
        if hits:
            rc = 1

    if "diversity" in want:
        table, max_share = check_diversity(records)
        top = max(table, key=table.get)
        print(f"diversity: {len(table)} topics, top {top} {max_share:.1%}")
        for t, c in table.items():
            print(f"  {t}: {c} ({c / len(records):.1%})")
        if max_share > DOMINANCE:
            print(f"  FAIL: one topic above {DOMINANCE:.0%} -- the corpus is not diverse")
            rc = 1

    if args.spotcheck_out:
        n = make_sheet(records, args.seed, args.spotcheck_out)
        print(f"spotcheck: sheet of {n} written to {args.spotcheck_out} (seed {args.seed}); "
              f"two readers fill verdict, then --agreement A B")

    return rc


if __name__ == "__main__":
    sys.exit(main())

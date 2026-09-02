#!/usr/bin/env python3
"""Code-execution reward: (code, tests) -> float in {0.0, 1.0}. Stage 4(a) of de-28.

The contract, and it is deliberately the whole contract:

    1.0 if every test passes inside the timeout, 0.0 otherwise.

Binary, not partial credit by test count. A fraction would reward a rollout that makes 9 of
10 tests pass by deleting the tenth's assertion, and partial credit on a reward the model
optimises against is a gradient toward exactly that. GRPO normalises within the group, so a
binary reward still separates K rollouts as long as they do not all agree.

INTERFACE. `(code, tests) -> float` and nothing else: no Cfg, no tokenizer, no checkpoint,
no import from train.py. tileRL's 27B GRPO takes this module as `--reward` (tilerl-19) and
that repo has none of ours. The executor is injectable for the same reason -- tileRL may
have a different sandbox than either of our two hosts.

WHY THE TESTS RUN AND THE OUTPUT IS NOT PARSED. A reward that greps stdout for "ok" is a
reward the model games by printing "ok". The exit code of the test runner is the signal:
python -m pytest and python -m unittest both exit non-zero on any failure, error or
collection error. What the model writes to stdout cannot change it.

WHAT A NON-ZERO REWARD DOES NOT MEAN. Tests passing means the tests passed, which is not
the same as the implementation being correct: the model can rewrite the tests. The caller
prevents that, not this function -- the runner writes `tests` to disk AFTER `code`, so a
rollout that edits the test file has its edit overwritten. That ordering is the whole
defence and it is asserted in the suite.

NON-DETERMINISM is the reward's enemy: the same (code, tests) must give the same float, or
the advantage within a group is noise. Three sources, all handled here rather than hoped
about:
  time      a test asserting on wall-clock or dates flips between rollouts. Detected, not
            fixed: `nondeterminism_risk(tests)` names the pattern found, and the round trip
            reports the rate rather than silently averaging over it.
  random    unseeded random in the TEST is the same problem; seeded random in the code is
            fine. Same detector.
  float     exact float equality in a test is platform-dependent. Same detector.
Anything the detector flags is reported by --roundtrip as a separate bucket, so a
ground-truth pair that cannot give a stable reward is visible instead of averaged in.

    python3 algorithms/code_reward.py --selftest
    python3 algorithms/code_reward.py --roundtrip <pairs.jsonl>   # the 0830v1 gate

# restartable: the executor writes only inside a mkdtemp it removes. --roundtrip is a pure
# read of its input plus per-pair execution; an interrupt costs the pairs already scored.
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "algorithms"))

# The default executor. Injectable: tileRL passes its own.
try:
    from isolate import Unisolated, detect_level
    from isolate import run as _isolate_run
except ImportError:  # pragma: no cover - only when called from outside the repo
    _isolate_run = None
    detect_level = None

    class Unisolated(RuntimeError):
        pass


IMPL_NAME = "solution.py"
TEST_NAME = "test_solution.py"

# Patterns whose presence in a TEST makes the reward unstable across rollouts. Matched on
# the test, not the implementation: seeded randomness inside the code under test is fine,
# an unseeded draw in the assertion is not.
NONDETERMINISM = (
    (r"\btime\.time\(|\bdatetime\.now\(|\bdate\.today\(|\btime\.monotonic\(",
     "wall clock: the assertion depends on when it runs"),
    (r"\brandom\.(?!seed)\w+\(|\bnp\.random\.(?!seed)\w+\(",
     "unseeded random in the test"),
    (r"assert\s+[\w.\[\]()]+\s*==\s*[-+]?\d+\.\d{6,}",
     "exact equality against a long float literal"),
    (r"\bos\.environ\b|\bsocket\.|\brequests\.|\burllib\b",
     "environment or network dependence"),
    (r"\bid\(|\bhash\(",
     "identity or hash: PYTHONHASHSEED varies"),
)


def nondeterminism_risk(tests):
    """[(pattern_label, ...)] for every instability found in the test source. [] is clean."""
    return [label for pat, label in NONDETERMINISM if re.search(pat, tests)]


def _runner_argv(tests):
    """How to invoke the test file. pytest when it is importable, else unittest.

    Both exit non-zero on any failure. unittest is stdlib, so the fallback always exists;
    pytest is preferred because a pytest-style file (bare `assert`, no TestCase) collects
    under unittest as zero tests, and zero tests passing would read as success.
    """
    try:
        import pytest  # noqa: F401

        return [sys.executable, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                "--no-header", TEST_NAME]
    except ImportError:
        return [sys.executable, "-I", "-m", "unittest", "-q", TEST_NAME.removesuffix(".py")]


# pytest's last summary line. TWO SHAPES, and the first version of this regex only knew one:
#   default   `===== 1 failed, 2 passed, 1 skipped in 0.31s =====`
#   -q        `1 passed in 0.00s`                  <- no `=` decoration at all
# `-q` is what _runner_argv passes, so requiring the `=` matched NOTHING in production while
# every case in the suite passed -- because I wrote all of them with the decoration. The
# suite caught it on the pod (every genuine pass scored 0.0), which is the only reason this
# comment exists. MEASURED raw stdout: '.  [100%]\n1 passed in 0.00s\n' (2026-09-02).
# The `=` is therefore optional, and the count regex does the real work.
_SUMMARY = re.compile(r"^=*\s*(?P<body>[^=\n]*?\d+\s+\w+[^=\n]*?)\s+in\s+[\d.]+s[^=\n]*=*\s*$",
                      re.M)
_NOTESTS = re.compile(r"^=*\s*no tests ran\s+in\s+[\d.]+s.*$", re.M)
_COUNT = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed|deselected)")


def parse_summary(stdout):
    """{kind: n} from pytest's summary line, or None when there is no parseable line.

    None is "cannot tell", and verdict() scores it 0 rather than reading it as "nothing
    failed" -- an unparseable run is not evidence of a pass. A line carrying no counts also
    answers None, because the regex requires at least one `<n> <word>`; that requirement is
    what keeps `-q`'s progress line (`. [100%]`) from matching.
    """
    m = None
    for m in _SUMMARY.finditer(stdout):  # noqa: B007 -- the LAST match is what we want
        pass
    if m is None:
        return None
    return {kind: int(n) for n, kind in _COUNT.findall(m.group("body"))}


def verdict(rc, timed_out, stdout):
    """(passed: bool, reason: str). fb's ruling (c), 2026-09-02.

    PASS = rc 0 AND at least one `passed` AND no failed/error. The exit code stays the
    authority for FAILURE -- a reward that greps stdout for "ok" is a reward the model games
    by printing "ok" -- and the summary is read only to catch the shapes where rc 0 is a LIE
    about a pass having happened:

        2 skipped in 0.30s      rc 0, nothing ran. The exploit I measured and reported: a
                                wrong implementation earns 1.0 when every test skips, and it
                                survives the write-order defence because the model never has
                                to touch the test file.
        no tests ran            rc 5 already, but belt and braces.

    `summary_mismatch` is reserved for genuinely unreadable or self-contradictory output --
    rc 0 with a failure counted, an INTERNALERROR, or no summary line at all -- and it scores
    0. The asymmetry is deliberate: an unparseable run is not evidence of a pass.

    What this does NOT do is treat rc-vs-summary disagreement as the criterion. It cannot:
    `1 passed, 1 skipped` at rc 0 is a real pass and `2 skipped` at rc 0 is not, and both
    "agree" with their exit code. The question is whether a test actually passed.
    """
    if timed_out:
        return False, "timeout"
    counts = parse_summary(stdout)
    if "INTERNALERROR" in stdout:
        return False, "summary_mismatch: pytest INTERNALERROR"
    if rc != 0:
        return False, f"rc {rc}"
    if counts is None:
        return False, "summary_mismatch: rc 0 but no pytest summary line"
    bad = counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0)
    if bad:
        return False, f"summary_mismatch: rc 0 but summary counts {bad} failed/error"
    if not counts.get("passed", 0):
        skipped = counts.get("skipped", 0)
        return False, (f"rc 0 with no test passed ({skipped} skipped)" if skipped
                       else "rc 0 with no test passed")
    return True, f"{counts['passed']} passed"


def reward_fn(code, tests, timeout=30, executor=None, level=None, workdir=None):
    """1.0 if every test passes, else 0.0. See the module docstring for the contract."""
    return score(code, tests, timeout=timeout, executor=executor, level=level,
                 workdir=workdir)["reward"]


def score(code, tests, timeout=30, executor=None, level=None, workdir=None):
    """reward_fn plus the evidence: {reward, level, rc, stdout, stderr, timed_out, risk}.

    The dict is what a rollout record carries. `level` in particular: a reward earned under
    rlimits_only and one earned under a namespace are different measurements (the vocab_id
    reasoning), so the level travels with the number instead of being logged separately.
    """
    ex = executor or _isolate_run
    if ex is None:
        raise RuntimeError(
            "no executor: algorithms/isolate.py could not be imported and none was passed. "
            "Untrusted code does not run unisolated by default -- pass executor=... with "
            "the same signature (code, workdir, timeout, level, argv) -> dict."
        )
    own = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="reward.")
    try:
        # code FIRST, tests SECOND. A rollout that rewrites the test file loses the rewrite,
        # because `tests` lands after whatever the model produced. This ordering is the only
        # thing standing between "the tests pass" and "the model made the tests pass", and
        # the suite asserts it directly.
        with open(os.path.join(workdir, IMPL_NAME), "w", encoding="utf-8") as f:
            f.write(code)
        with open(os.path.join(workdir, TEST_NAME), "w", encoding="utf-8") as f:
            f.write(tests)
        r = ex(None, workdir=workdir, timeout=timeout, level=level, argv=_runner_argv(tests))
        passed, reason = verdict(r["rc"], r["timed_out"], r["stdout"])
        return {"reward": 1.0 if passed else 0.0, "level": r["level"], "rc": r["rc"],
                "reason": reason,
                "stdout": r["stdout"][-2000:], "stderr": r["stderr"][-2000:],
                "timed_out": r["timed_out"], "risk": nondeterminism_risk(tests)}
    finally:
        if own:
            shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------- known answers
# SHOULD: reward must equal expected; a mismatch is a BUG (exit 1).
# GAP:    known behaviour that is arguably wrong, documented; a change is a flag.
# Same two classes as algorithms/test_rlvr_reward_suite.py.

_GOOD_IMPL = "def add(a, b):\n    return a + b\n"
_GOOD_TEST = ("from solution import add\n\n\n"
              "def test_add():\n    assert add(2, 3) == 5\n    assert add(-1, 1) == 0\n")

SHOULD = [
    (_GOOD_IMPL, _GOOD_TEST, 1.0, "correct implementation passes its tests"),
    ("def add(a, b):\n    return a - b\n", _GOOD_TEST, 0.0, "wrong implementation fails"),
    ("def add(a, b):\n    return a + b\n", "from solution import add\n\n\n"
     "def test_add():\n    assert add(2, 3) == 6\n", 0.0, "test disagrees with correct code"),
    ("def add(a, b:\n    return a\n", _GOOD_TEST, 0.0, "syntax error is not a pass"),
    ("", _GOOD_TEST, 0.0, "empty implementation: ImportError is not a pass"),
    (_GOOD_IMPL, "", 0.0, "empty test file collects zero tests, which cannot pass"),
    (_GOOD_IMPL, "from solution import add\n\n\ndef test_a():\n    assert add(1, 1) == 2\n\n\n"
     "def test_b():\n    assert add(1, 1) == 3\n", 0.0,
     "one of two tests fails -> 0.0, no partial credit"),
    ("import time\n\n\ndef add(a, b):\n    time.sleep(120)\n    return a + b\n",
     _GOOD_TEST, 0.0, "a hang is a failure, not a hang"),
    (_GOOD_IMPL + "\nraise SystemExit(0)\n", _GOOD_TEST, 0.0,
     "import-time SystemExit(0) does not buy a pass"),
    # The defence, as a case: the model rewrites the test file, and it does not help.
    ("import pathlib\n"
     "pathlib.Path('test_solution.py').write_text('def test_x():\\n    assert True\\n')\n"
     "def add(a, b):\n    return a - b\n", _GOOD_TEST, 0.0,
     "a rollout that overwrites the tests at import time still fails: tests are written "
     "after code, and the runner reads them from disk"),
    # And the subtler form: sabotage the assertion by shadowing what it compares.
    ("def add(a, b):\n    return a - b\n\n\n"
     "class _Always:\n"
     "    def __eq__(self, other):\n        return True\n",
     _GOOD_TEST, 0.0, "defining an always-equal class does not make a wrong add pass"),
    # A test file that exits at module level is not a pass. MEASURED, and the measurement
    # corrected me: I first wrote this as a GAP documenting reward 1.0 on the reasoning that
    # sys.exit(0) yields rc 0. pytest actually raises INTERNALERROR during collection and
    # exits 3, so the reward is 0.0 and the case belongs here. The GAP entry would have
    # documented an exploit that does not exist and, worse, would have flagged as CHANGED
    # the first time anyone read it -- a documented non-fact is a false alarm with a
    # citation. Run the case before writing down what it does.
    (_GOOD_IMPL,
     "import sys\nfrom solution import add\n\n\n"
     "def test_add():\n    assert add(2, 3) == 5\n\n\nsys.exit(0)\n",
     0.0, "sys.exit(0) at test-module level: pytest INTERNALERRORs, rc 3, no tests ran"),
    # THE EXPLOIT, closed by fb ruling (c) 2026-09-02 and kept as a SHOULD so it cannot come
    # back. It was a GAP documenting reward 1.0: every test skipping gives `2 skipped in
    # 0.30s` at rc 0, so a WRONG implementation earned full reward -- and it survived the
    # write-order defence entirely, because the model never has to touch the test file if the
    # task ships tests that skip. rc alone cannot see it; the summary line can.
    ("def add(a, b):\n    return a - b\n",
     "from solution import add\n\n\n"
     "def test_a():\n    import pytest\n    pytest.skip('hard')\n\n\n"
     "def test_b():\n    import pytest\n    pytest.skip('also hard')\n",
     0.0, "EVERY test skipped -> rc 0 but nothing passed -> 0.0, not 1.0"),
    # And the other side of that line, which is why the criterion is "at least one passed"
    # and not "no skips": a real pass beside a skip is a pass.
    (_GOOD_IMPL,
     "from solution import add\n\n\n"
     "def test_add():\n    assert add(2, 3) == 5\n\n\n"
     "def test_skipped():\n    import pytest\n    pytest.skip('not implemented')\n",
     1.0, "1 passed, 1 skipped -> 1.0: a skip beside a real pass does not void it"),
]

GAP = [
    # Both former entries moved to SHOULD when fb's ruling (c) closed the all-skip hole.
    # This class must stay non-empty (the suite asserts it), and this is a real one: a test
    # that passes for a reason the implementation did not earn. xfail/xpass is the shape --
    # `1 xpassed` is not counted as `passed`, so an implementation that unexpectedly satisfies
    # an xfail-marked test scores 0.0. Defensible (the task declared it expected to fail) and
    # arguably wrong (the code worked). NOT decided here: whether a miner should keep such
    # pairs at all is 3b's call, and inventing a rule for it in the reward would be the
    # improvisation fb's ruling on the skip hole deliberately avoided.
    (_GOOD_IMPL,
     "import pytest\nfrom solution import add\n\n\n"
     "@pytest.mark.xfail\ndef test_add():\n    assert add(2, 3) == 5\n",
     0.0, 0.0,
     "an xfail-marked test that PASSES reports `1 xpassed`, not `1 passed`, so the reward "
     "is 0.0 on a correct implementation. Defensible -- the task declared the test expected "
     "to fail -- but a miner shipping xfail markers would silently zero good rollouts. "
     "Raise with 3b before the first RL step rather than special-casing it here"),
    (_GOOD_IMPL,
     "from solution import add\n\n\n"
     "def test_add():\n    assert add(2, 3) == 5\n\n\n"
     "def test_deselected():\n    assert add(1, 1) == 2\n",
     1.0, 1.0,
     "a plain two-test file passes; recorded beside the xfail case as the control that the "
     "summary parser is not simply rejecting multi-test files"),
]


def _detector_cases():
    """The non-determinism detector and the summary parser on their own known answers.

    No sandbox, no subprocess: both are pure functions over text, so the commit hook can run
    this half on any host.
    """
    print("== summary parser + verdict (fb ruling c) ==")
    bugs = 0
    # (rc, stdout, want_pass, note).
    #
    # THE FIRST HALF ARE REAL `-q` OUTPUT, captured from the pod, because the first version of
    # this list was entirely invented with `===== ... =====` decoration -- which `-q` does not
    # print. Every case passed and the parser matched nothing in production, scoring 0.0 on
    # every genuine pass. Invented fixtures test the fixture writer's assumptions; the shape
    # that reaches the code has to come from the code.
    vcases = [
        (0, ".                                     [100%]\n1 passed in 0.00s\n", True,
         "REAL -q output: no `=` decoration at all"),
        (0, "..                                    [100%]\n2 passed in 0.01s\n", True,
         "REAL -q, two tests"),
        (0, "ss                                    [100%]\n2 skipped in 0.30s\n", False,
         "REAL -q, ALL skipped at rc 0 -> NOT a pass"),
        (1, ".F                                    [100%]\n1 failed, 1 passed in 0.12s\n",
         False, "REAL -q, rc 1 -> fail"),
        (0, ".s                                    [100%]\n1 passed, 1 skipped in 0.02s\n",
         True, "REAL -q, a skip beside a real pass is still a pass"),
        # And the decorated shape, which is what pytest prints without -q -- kept so the
        # parser kegs working if _runner_argv ever drops -q.
        (0, "===== 1 passed in 0.01s =====", True, "decorated (non-q) shape still parses"),
        (0, "===== 2 passed, 1 skipped in 0.31s =====", True, "decorated, passed beside skip"),
        (0, "===== 1 failed, 1 passed in 0.12s =====", False,
         "rc 0 contradicting its own summary -> summary_mismatch"),
        (5, "no tests ran in 0.01s\n", False, "no tests ran -> fail"),
        (0, "nothing that looks like a summary", False,
         "rc 0 with no summary line -> summary_mismatch, never an implied pass"),
        (0, "INTERNALERROR> boom\n1 passed in 0.01s\n", False,
         "INTERNALERROR outweighs a passed count"),
        (0, "1 xpassed in 0.01s\n", False, "xpassed is not passed"),
    ]
    for rc, out, want, note in vcases:
        got, reason = verdict(rc, False, out)
        ok = got == want
        bugs += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BUG '} pass={got!s:5} want={want!s:5} {note} [{reason}]")
    got, reason = verdict(0, True, "===== 1 passed in 0.01s =====")
    ok = got is False and reason == "timeout"
    bugs += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} a timeout is a failure even with a passed count")
    # A summary line with no counts no longer parses at all, because the regex now REQUIRES
    # at least one `<n> <word>` -- that is what stops `-q`'s progress line from matching. So
    # the distinction that matters is None (no line) vs a dict with counts; both no-line and
    # countless-line answer None, and neither is ever read as "nothing failed" because
    # verdict() treats None as summary_mismatch and scores 0.
    ok = parse_summary("no summary here") is None and parse_summary("==== in 0.1s ====") is None
    bugs += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} an unparseable summary is None, and None scores 0")

    print("== nondeterminism detector ==")
    det = [        ("def test_x():\n    assert 1 == 1\n", [], "clean"),
        ("import time\ndef test_x():\n    assert time.time() > 0\n", 1, "wall clock"),
        ("import random\ndef test_x():\n    assert random.random() < 1\n", 1, "unseeded random"),
        ("import random\nrandom.seed(0)\ndef test_x():\n    assert True\n", 0,
         "random.seed alone is NOT flagged"),
        ("def test_x():\n    assert compute() == 0.333333333\n", 1, "long float equality"),
    ]
    bugs = 0
    for tests, want, note in det:
        got = nondeterminism_risk(tests)
        n = len(got)
        ok = (n == 0) if want == 0 or want == [] else (n >= 1)
        bugs += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BUG '} {n} risk(s)  {note}")
    return bugs, len(vcases) + 2 + len(det)


def _run_suite(level=None):
    bugs = flags = 0
    print("== SHOULD (mismatch = bug) ==")
    for code, tests, exp, note in SHOULD:
        got = reward_fn(code, tests, timeout=20, level=level)
        ok = got == exp
        bugs += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BUG '} {got:.1f} want {exp:.1f}  {note}")
    print("== GAP (change = flag) ==")
    for code, tests, cur, _should, note in GAP:
        got = reward_fn(code, tests, timeout=20, level=level)
        ok = got == cur
        flags += 0 if ok else 1
        print(f"  {'as-documented' if ok else 'CHANGED      '} {got:.1f} "
              f"(documented {cur:.1f}) {note[:70]}")
    dbugs, dn = _detector_cases()
    return bugs + dbugs, flags, dn


def _selftest(detector_only=False):
    if detector_only:
        # The assertions that need no isolation and no subprocess: pure regex over test
        # source. Split out because the rest is 42s of sandboxed execution on a host that
        # HAS isolation and a refusal on one that does not, so the hook can run this half
        # everywhere. The exemption belongs to the assertions that need a sandbox, not to
        # the file -- the same granularity the hook's own NEEDS_DATA comment argues for.
        bugs, n = _detector_cases()
        print(f"{n} sandbox-free cases: {bugs} bug(s) (reward cases skipped: they need "
              f"process isolation)")
        return 1 if bugs else 0
    lvl = None
    if detect_level is not None:
        lvl = detect_level()
        print(f"executor level: {lvl}\n")
    bugs, flags, dn = _run_suite(level=lvl)
    # A suite where nothing ever fails is not a suite: both classes must be non-empty and
    # the negative cases must actually have produced 0.0 above.
    assert any(e == 0.0 for *_, e, _ in ((c, t, e, n) for c, t, e, n in SHOULD)), \
        "SHOULD has no negative case"
    assert len(SHOULD) >= 10 and len(GAP) >= 2, "the suite shrank"
    print(f"\n{len(SHOULD)} SHOULD, {len(GAP)} GAP, {dn} sandbox-free cases: "
          f"{bugs} bug(s), {flags} flag(s)")
    if bugs:
        print("FAIL: a SHOULD case did not match. The reward is what the model optimises "
              "against, so a blind spot here becomes training signal.")
    return 1 if bugs else 0


def _roundtrip(path, limit=None, level=None):
    """The 0830v1 gate: a checker that becomes a reward must score its ground truth ~1.0.

    Input: JSONL with {impl, tests} (3b's miner writes {repo, impl_path, impl, test_path,
    tests, passed}). Every pair is one the miner already ran and recorded as passing, so the
    expected reward is 1.0 and anything else is this reward function's defect -- not the
    data's. Target 99.9%.

    Reported in three buckets, never one number: passed, failed, and unstable (the
    non-determinism detector fired). Averaging an unstable pair into the rate would hide
    exactly the pairs whose reward cannot be trusted twice.
    """
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    if not rows:
        print(f"FAIL: {path} holds no pairs")
        return 1
    ok = bad = 0
    unstable = []
    failures = []
    for i, r in enumerate(rows):
        code, tests = r.get("impl"), r.get("tests")
        if not code or not tests:
            failures.append((i, r.get("impl_path", "?"), "missing impl or tests field"))
            bad += 1
            continue
        risk = nondeterminism_risk(tests)
        res = score(code, tests, timeout=30, level=level)
        if res["reward"] == 1.0:
            ok += 1
        else:
            bad += 1
            failures.append((i, r.get("impl_path", "?"),
                             ("TIMEOUT" if res["timed_out"] else f"rc {res['rc']}: "
                              + (res["stderr"] or res["stdout"]).strip()[-160:])))
        if risk:
            unstable.append((i, r.get("impl_path", "?"), risk))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)}  {ok} pass  {bad} fail  "
                  f"{len(unstable)} unstable", flush=True)
    rate = ok / len(rows)
    print(f"\nround trip: {ok}/{len(rows)} = {rate:.4%}")
    print(f"unstable (detector fired, reward may differ between rollouts): "
          f"{len(unstable)}/{len(rows)} = {len(unstable) / len(rows):.2%}")
    for i, p, why in failures[:15]:
        print(f"  FAIL row {i} {p}: {why}")
    for i, p, risk in unstable[:10]:
        print(f"  UNSTABLE row {i} {p}: {', '.join(risk)}")
    if rate < 0.999:
        print(f"\nBELOW THE 0830v1 GATE (99.9%). Reporting the number, not moving the "
              f"threshold: {len(rows) - ok} pairs the miner ran green score 0.0 here, and "
              f"each is either a defect in this reward or a pair the miner should not have "
              f"kept. Read the failures above before the first RL step.")
        return 1
    print("\nGATE PASSED: >=99.9% of ground-truth pairs score 1.0.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-detector", action="store_true",
                    help="only the assertions that need no sandbox (the commit hook runs this)")
    ap.add_argument("--roundtrip", metavar="PAIRS_JSONL")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--level", default=None, help="force an isolation level (default: detect)")
    a = ap.parse_args()
    if a.selftest_detector:
        return _selftest(detector_only=True)
    if a.selftest:
        return _selftest()
    if a.roundtrip:
        return _roundtrip(a.roundtrip, limit=a.limit, level=a.level)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

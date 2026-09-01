#!/usr/bin/env python3
"""The record `launch_gate.gate_arch_tests` reads, written by the test that ran.

A launch-test record typed by hand is exactly the defect the gate exists to catch: a
claim about what ran, not derived from what ran. So the shape and the kernel come from
the values the test actually used -- the test passes them from its own locals -- and
nothing here accepts a shape as a string to be believed.

    from launch_tests import record_launch_test
    record_launch_test(__file__, "pass", {"d": D, "layers": L, ...}, real_kernel=True)

One row per test path, last write wins: rerunning a test replaces its verdict rather
than appending a second one, because the question is "did it pass on this shape now",
not "has it ever passed".

    python3 scripts/launch_tests.py --selftest
"""
import contextlib
import hashlib
import io
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "runs", "launch_tests.json")
SHAPE_KEYS = ("d", "layers", "heads", "ffn_hidden")


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _launch_shape():
    """The gate's expectation, imported rather than restated.

    One definition, in launch_gate.py:191. A second copy here would drift, and the drift
    would be invisible in exactly the case this warning exists for -- the two would
    disagree while each looked internally consistent. Imported lazily: launch_gate is the
    heavier module and record_launch_test is called from tests that do not otherwise need
    it. This reads the expectation to WARN about a mismatch, never to choose a shape to
    run at -- a test that picked its shape from the gate's expectation would make the
    gate's comparison a tautology, and launch_gate.py:727's deliberately-wrong world
    could not be constructed at all (de).
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from launch_gate import LAUNCH_SHAPE
    return LAUNCH_SHAPE


def record_launch_test(test_file, result, shape, real_kernel, path=PATH, root=ROOT):
    """Write this test's verdict. `test_file` is __file__; the key is its repo path."""
    missing = [k for k in SHAPE_KEYS if k not in shape]
    if missing:
        raise ValueError(f"shape is missing {missing}: a row that does not state its "
                         f"full shape cannot be compared against the launch shape")
    key = os.path.relpath(os.path.abspath(test_file), root)
    rows = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                rows = json.load(f)
        except ValueError:
            rows = {}
    rows[key] = {
        "result": result,
        "shape": {k: shape[k] for k in SHAPE_KEYS},
        "real_kernel": bool(real_kernel),
        "recorded": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
        "host": os.uname().nodename,
        # The fingerprint of what produced it. Without this the row stays valid after
        # the test changes, which is the failure this repo has bought three times
        # (vocab_id on checkpoints, .srcfp on token caches, filters_fp on shards). Both
        # of these files changed three times on the day the record format was written.
        # Content hash, not a git sha: an uncommitted edit changes what runs and a sha
        # cannot see it.
        "test_sha256": _sha256(os.path.abspath(test_file)),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, sort_keys=True)
        f.write("\n")
    # Say it here, where the run just happened, rather than leaving it for the gate to
    # discover later. A pass at the wrong shape is a real pass -- the run happened, the
    # test succeeded -- so the row is still written; what it must not do is look like
    # evidence for the launch. NOT a refusal to write: "ran at another shape" and "never
    # ran" need different actions, and collapsing them loses the only fact that says
    # which (de). Measured 2026-09-02: an e2e pass at layers 12 was carried toward the
    # gate as though it cleared arch_tests, and nothing between the run and the gate said
    # otherwise.
    off = {k: (v, shape.get(k)) for k, v in _launch_shape().items() if shape.get(k) != v}
    warn = ("  [NOT THE LAUNCH SHAPE: " +
            ", ".join(f"{k} is {got}, launch is {want}" for k, (want, got) in sorted(off.items()))
            + "]") if off else ""
    print(f"  recorded {key}: {result} at "
          f"d{shape['d']} L{shape['layers']} "
          f"{'real kernel' if real_kernel else 'STAND-IN kernel'} -> "
          f"{os.path.relpath(path, root)}{warn}")
    return rows[key]


def _selftest():
    import shutil
    import tempfile

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from launch_gate import ARCH_TESTS, LAUNCH_SHAPE, gate_arch_tests

    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "runs", "launch_tests.json")
        os.makedirs(os.path.join(d, "runs"))
        os.makedirs(os.path.join(d, "scripts"))
        for n in ARCH_TESTS:
            open(os.path.join(d, n), "w").write("")

        for n in ARCH_TESTS:
            record_launch_test(os.path.join(d, n), "pass", dict(LAUNCH_SHAPE),
                               real_kernel=True, path=p, root=d)
        # The join, which is the only thing worth asserting here: what this writes must
        # be what the gate accepts. Two files agreeing on a format by hand is how the
        # format drifts -- sft_math.py read "vocab" for a year while the packer wrote
        # "vocab_id", and the assert that was the sole enforcer of vocabulary identity
        # was unreachable the whole time.
        st, why = gate_arch_tests(d, os.path.join(d, "mix.json"), 7)
        assert st == "GO", f"the gate refuses what this writer produces: {why}"

        # rerun replaces, never appends a second verdict
        record_launch_test(os.path.join(d, ARCH_TESTS[0]), "fail", dict(LAUNCH_SHAPE),
                           real_kernel=True, path=p, root=d)
        rows = json.load(open(p, encoding="utf-8"))
        assert len(rows) == len(ARCH_TESTS), f"rerun appended: {sorted(rows)}"
        assert rows[ARCH_TESTS[0]]["result"] == "fail", "rerun did not replace the verdict"
        st, why = gate_arch_tests(d, os.path.join(d, "mix.json"), 7)
        assert st != "GO", f"a recorded fail must not pass the gate: {why}"

        # an incomplete shape is refused at write time, not discovered at read time
        try:
            record_launch_test(os.path.join(d, ARCH_TESTS[0]), "pass", {"d": 1024},
                               real_kernel=True, path=p, root=d)
            raise AssertionError("a shape missing layers/heads/ffn_hidden was accepted")
        except ValueError:
            pass
    finally:
        shutil.rmtree(d, ignore_errors=True)
    # The warning is the point of the last change, so it has a case: a row recorded at
    # the wrong depth must SAY so on the line the runner sees, and must still be written.
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "scripts"))
        for n in ARCH_TESTS:
            with open(os.path.join(d, n), "w") as f:
                f.write("")
        p2 = os.path.join(d, "runs", "launch_tests.json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            record_launch_test(os.path.join(d, ARCH_TESTS[0]), "pass",
                               dict(LAUNCH_SHAPE, layers=12), real_kernel=True,
                               path=p2, root=d)
        out = buf.getvalue()
        assert "NOT THE LAUNCH SHAPE" in out, f"a wrong-shape pass said nothing: {out!r}"
        assert "layers is 12, launch is 32" in out, f"the warning does not name it: {out!r}"
        with open(p2, encoding="utf-8") as f:
            assert json.load(f)[ARCH_TESTS[0]]["shape"]["layers"] == 12, (
                "the row was not written -- 'ran at another shape' must not be erased "
                "into 'never ran'")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            record_launch_test(os.path.join(d, ARCH_TESTS[0]), "pass", dict(LAUNCH_SHAPE),
                               real_kernel=True, path=p2, root=d)
        assert "NOT THE LAUNCH SHAPE" not in buf.getvalue(), (
            "the launch shape itself was flagged; a warning that fires on the good case "
            "gets ignored on the bad one")

    print("launch_tests selftest OK: what the writer writes is what the gate accepts, "
          "and a wrong-shape pass says so")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else
             print(__doc__.strip()) or 0)

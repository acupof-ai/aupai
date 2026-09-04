#!/usr/bin/env python3
"""The record `launch_gate.gate_arch_tests` reads, written by the test that ran.

A launch-test record typed by hand is exactly the defect the gate exists to catch: a
claim about what ran, not derived from what ran. So the shape and the kernel come from
the values the test actually used -- the test passes them from its own locals -- and
nothing here accepts a shape as a string to be believed.

    from launch_tests import record_launch_test
    record_launch_test(__file__, "pass", {"d": D, "layers": L, ...}, real_kernel=True)

One row per (test path, SHAPE), last write wins WITHIN a shape: rerunning a test at the
same shape replaces its verdict rather than appending a second one, because the question
is "did it pass on this shape now", not "has it ever passed".

THE KEY WAS THE TEST PATH ALONE UNTIL 2026-09-04, and one launch shape at a time is what
that was correct for. N7 Stage E runs an L12 arm and an L16 arm CONCURRENTLY, so b0's L16
certification overwrote the L12 rows and gate_arch_tests then read "ran at L16, launch is
L12" for the arm that had in fact been certified -- Stage E arm 2 blocked on evidence that
existed and had been erased (6e, measured). Two concurrent shapes are two questions and
they need two rows.

The key is `<path>@d<D>L<L>h<H>f<FFN>`. Rows written before this carry the bare path and
are still READ: `rows_for` falls back to the bare key when no shaped key matches, so an
existing certification is not invalidated by the format change -- it just cannot say which
of two shapes it was for, which is exactly what it could never say.

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

    Returns None when launch_gate cannot be imported, and never raises. This runs AFTER
    the row is on disk, so an exception here would leave the record written and the test
    reporting failure -- the record saying pass while its runner says it did not. Not
    hypothetical: both files are in pod_head_manifest.txt and a named single-file push
    happened four times tonight, so a tree holding one and not the other is one push away
    (de). A warning that can break the thing it annotates is worse than the inconsistency
    it reports.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from launch_gate import LAUNCH_SHAPE
        return LAUNCH_SHAPE
    except Exception:
        return None


def _launch_mix():
    """The mix the gate expects, imported for the same reason as the shape above.

    Returns None when launch_gate cannot be imported OR does not define LAUNCH_MIX, and
    never raises: a row must still land when this cannot be answered, and "mix unknown"
    is reported rather than guessed. Deliberately does NOT fall back to a literal --
    _launch_shape's docstring says a second copy would drift invisibly in exactly the case
    the warning exists for, and a hardcoded mix path here would be that copy."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from launch_gate import LAUNCH_MIX
        return LAUNCH_MIX
    except Exception:
        return None


def shape_key(test_rel, shape):
    """The row key for one test at one shape: `<path>@d<D>L<L>h<H>f<FFN>`.

    Derived from the shape the test ACTUALLY used, like every other field here -- a key a
    caller could pass as a string would be a claim about which shape ran."""
    missing = [k for k in SHAPE_KEYS if k not in shape]
    if missing:
        raise ValueError(f"shape is missing {missing}: a key that does not state its full "
                         f"shape cannot distinguish two concurrent arms")
    return (f"{test_rel}@d{shape['d']}L{shape['layers']}"
            f"h{shape['heads']}f{shape['ffn_hidden']}")


def rows_for(rows, test_rel, shape):
    """The row certifying `test_rel` at `shape`, or None.

    Tries the shaped key, then the BARE path for rows written before 2026-09-04. The
    fallback is not leniency about the shape: the bare row still carries its own `shape`
    field and the gate compares that as it always did. What the fallback buys is that a
    format change does not silently un-certify a launch someone already ran (the same
    reason a checkpoint's vocab_id refuses rather than rebuilding)."""
    k = shape_key(test_rel, shape)
    if k in rows:
        return rows[k]
    return rows.get(test_rel)


def record_launch_test(test_file, result, shape, real_kernel, mix=None, stages=None,
                       path=PATH, root=ROOT):
    """Write this test's verdict. `test_file` is __file__; the key is its repo path.

    `mix` is the data the test ran on, and it is the third field a row needs beside shape
    and real_kernel. A row without it cannot answer the gate's question: the 20B launch
    died at step 0 on a KeyError('content') while e2e was green, because e2e ran
    data/mix_sample.json, whose corpus dir holds zero holdout slices. The shape was
    pinned and the DATA was a different question (de-10). Optional only so an older row
    keeps parsing -- a row that omits it is reported as mix unknown, never as the launch
    mix.

    `stages` is HOW MUCH of the test the pass covers, and it exists because a record
    written before the last stage is a claim about a run that has not finished:
    test_e2e.py wrote "pass" before its `finally`, so a stage-11 AssertionError left this
    file certifying a run that exited nonzero (b0 at the Stage E shape, 6e 2026-09-04).
    A partial pass is real evidence and must stay readable as partial -- 6e had to rule
    "arm 2 may launch on stages 1-10" by hand because the row could not say it. Optional
    for the same reason as mix: an older row keeps parsing, and reads as unstated."""
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
    # KEYED BY (path, shape) so two concurrent arms coexist. A bare-path row for this same
    # test is left in place rather than migrated: it may be the certification of a shape
    # nobody is re-running, and rewriting someone else's evidence to a key derived from THIS
    # run's shape would relabel their measurement as mine.
    rows[shape_key(key, shape)] = {
        "result": result,
        "shape": {k: shape[k] for k in SHAPE_KEYS},
        "real_kernel": bool(real_kernel),
        "mix": mix,
        "stages": stages,
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
    want_shape = _launch_shape()
    if want_shape is None:
        warn = "  [launch shape unknown: launch_gate did not import, mismatch not checked]"
    else:
        off = {k: (v, shape.get(k)) for k, v in want_shape.items() if shape.get(k) != v}
        warn = ("  [NOT THE LAUNCH SHAPE: " +
                ", ".join(f"{k} is {got}, launch is {want}"
                          for k, (want, got) in sorted(off.items())) + "]") if off else ""
    # The mix, said here for the same reason as the shape: a pass on the sample mix is a
    # real pass and must still be written, but it must not read as evidence for a launch
    # whose data it never touched.
    want_mix = _launch_mix()
    if mix is None:
        warn += "  [MIX UNRECORDED: this row cannot clear any launch mix]"
    elif want_mix and mix != want_mix:
        warn += f"  [NOT THE LAUNCH MIX: ran {mix}, launch is {want_mix}]"
    print(f"  recorded {shape_key(key, shape)}: {result} at "
          f"d{shape['d']} L{shape['layers']} "
          f"{'real kernel' if real_kernel else 'STAND-IN kernel'}"
          f"{' mix ' + mix if mix else ''} -> "
          f"{os.path.relpath(path, root)}{warn}")
    return rows[shape_key(key, shape)]


def _selftest():
    import shutil
    import tempfile

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from launch_gate import (
        ARCH_TESTS,
        LAUNCH_MIX,
        LAUNCH_SHAPE,
        READS_LAUNCH_DATA,
        gate_arch_tests,
    )

    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "runs", "launch_tests.json")
        os.makedirs(os.path.join(d, "runs"))
        os.makedirs(os.path.join(d, "scripts"))
        for n in ARCH_TESTS:
            open(os.path.join(d, n), "w").write("")

        for n in ARCH_TESTS:
            record_launch_test(os.path.join(d, n), "pass", dict(LAUNCH_SHAPE),
                               real_kernel=True, mix=LAUNCH_MIX, path=p, root=d)
        # The join, which is the only thing worth asserting here: what this writes must
        # be what the gate accepts. Two files agreeing on a format by hand is how the
        # format drifts -- sft_math.py read "vocab" for a year while the packer wrote
        # "vocab_id", and the assert that was the sole enforcer of vocabulary identity
        # was unreachable the whole time.
        st, why = gate_arch_tests(d, os.path.join(d, LAUNCH_MIX), 7)
        assert st == "GO", f"the gate refuses what this writer produces: {why}"

        # THE MIX HALF OF THE SAME JOIN (de-10). This assertion is why the mix field was
        # not a one-line addition: gate_arch_tests began reading `mix`, and this selftest
        # went red because the world above wrote no mix -- the writer and the gate had
        # drifted the moment the gate learned the field. Recording it above closes that,
        # and the world below is the other direction: a row without the field must be
        # refused, or "the gate accepts what the writer produces" is satisfied by a gate
        # that reads nothing.
        #
        # IT MUST BE A TEST THE MIX CHECK COVERS, i.e. one in READS_LAUNCH_DATA, and it
        # was ARCH_TESTS[0] instead. gate_arch_tests only asks about `mix` for the tests
        # that READ the launch data -- test_e2e.py, not test_arch_L32.py -- so writing the
        # mix-less row to ARCH_TESTS[0] produced a GO and this assertion had been RED on
        # main before any of today's changes (measured 2026-09-04 by running HEAD's copy of
        # this file: identical failure). It went unseen because the hook runs a file's
        # selftest only when that file is STAGED, and nobody had staged launch_tests.py
        # since the break -- registration gates the commits that touch it, not the tree.
        mix_test = READS_LAUNCH_DATA[0]
        record_launch_test(os.path.join(d, mix_test), "pass", dict(LAUNCH_SHAPE),
                           real_kernel=True, mix=None, path=p, root=d)
        st, why = gate_arch_tests(d, os.path.join(d, LAUNCH_MIX), 7)
        assert st != "GO", f"a row recording no mix passed the gate: {why}"
        assert "no mix" in why, f"it refused for another reason: {why}"
        record_launch_test(os.path.join(d, mix_test), "pass", dict(LAUNCH_SHAPE),
                           real_kernel=True, mix="data/mix_sample.json", path=p, root=d)
        st, why = gate_arch_tests(d, os.path.join(d, LAUNCH_MIX), 7)
        assert st != "GO", f"a pass on the sample mix cleared the launch mix: {why}"
        assert "mix_sample" in why, f"it refused for another reason: {why}"
        record_launch_test(os.path.join(d, ARCH_TESTS[0]), "pass", dict(LAUNCH_SHAPE),
                           real_kernel=True, mix=LAUNCH_MIX, path=p, root=d)

        # rerun at the SAME SHAPE replaces, never appends a second verdict
        record_launch_test(os.path.join(d, ARCH_TESTS[0]), "fail", dict(LAUNCH_SHAPE),
                           real_kernel=True, mix=LAUNCH_MIX, path=p, root=d)
        rows = json.load(open(p, encoding="utf-8"))
        assert len(rows) == len(ARCH_TESTS), f"rerun appended: {sorted(rows)}"
        assert rows[shape_key(ARCH_TESTS[0], LAUNCH_SHAPE)]["result"] == "fail", \
            "rerun at the same shape did not replace the verdict"
        st, why = gate_arch_tests(d, os.path.join(d, LAUNCH_MIX), 7)
        assert st != "GO", f"a recorded fail must not pass the gate: {why}"

        # TWO SHAPES CERTIFIED IN SEQUENCE MUST BOTH STAY READABLE (6e, 2026-09-04). This
        # is the whole point of the re-key: N7 Stage E runs an L12 arm and an L16 arm
        # concurrently, and with the key being the test path alone, b0's L16 certification
        # erased the L12 rows -- gate_arch_tests then reported "ran at L16, launch is L12"
        # for an arm that HAD been certified, and Stage E arm 2 blocked on evidence that
        # existed and had been overwritten.
        other = dict(LAUNCH_SHAPE, layers=LAUNCH_SHAPE["layers"] + 4)
        for n in ARCH_TESTS:
            record_launch_test(os.path.join(d, n), "pass", dict(LAUNCH_SHAPE),
                               real_kernel=True, mix=LAUNCH_MIX, path=p, root=d)
        for n in ARCH_TESTS:
            record_launch_test(os.path.join(d, n), "pass", other,
                               real_kernel=True, mix=LAUNCH_MIX, path=p, root=d)
        rows = json.load(open(p, encoding="utf-8"))
        assert len(rows) == 2 * len(ARCH_TESTS), (
            f"certifying a second shape did not leave the first readable: {sorted(rows)}. "
            f"That is the erasure this key exists to prevent")
        for n in ARCH_TESTS:
            for sh in (LAUNCH_SHAPE, other):
                got = rows_for(rows, n, sh)
                assert got and got["shape"]["layers"] == sh["layers"], (
                    f"rows_for({n}, L{sh['layers']}) returned "
                    f"{got and got['shape'].get('layers')} -- each shape must resolve to "
                    f"its OWN row, or the lookup reintroduces the overwrite")
        # ...and the gate, which is what actually blocked the arm, still clears the LAUNCH
        # shape while the other shape's rows sit beside it.
        st, why = gate_arch_tests(d, os.path.join(d, LAUNCH_MIX), 7)
        assert st == "GO", f"the launch shape stopped clearing once a second shape existed: {why}"

        # A BARE-PATH ROW STILL READS, or the format change silently un-certifies a launch
        # someone already ran. Written directly because record_launch_test can no longer
        # produce one -- the world has to be the old format, not a claim about it.
        rows["scripts/test_e2e.py"] = dict(rows[shape_key("scripts/test_e2e.py", other)])
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1, sort_keys=True)
        legacy = dict(LAUNCH_SHAPE, d=LAUNCH_SHAPE["d"] * 2)  # a shape with no shaped row
        assert rows_for(rows, "scripts/test_e2e.py", legacy) is not None, (
            "a pre-2026-09-04 bare-path row became unreadable, so the re-key un-certified "
            "an existing launch")

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
            wrong = shape_key(ARCH_TESTS[0], dict(LAUNCH_SHAPE, layers=12))
            assert json.load(f)[wrong]["shape"]["layers"] == 12, (
                "the row was not written -- 'ran at another shape' must not be erased "
                "into 'never ran'")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            record_launch_test(os.path.join(d, ARCH_TESTS[0]), "pass", dict(LAUNCH_SHAPE),
                               real_kernel=True, path=p2, root=d)
        assert "NOT THE LAUNCH SHAPE" not in buf.getvalue(), (
            "the launch shape itself was flagged; a warning that fires on the good case "
            "gets ignored on the bad one")

        # A tree with this file and not launch_gate.py: the import runs after the row is
        # written, so it must degrade to a warning. de built exactly this tree and got a
        # ModuleNotFoundError raised past a pass already on disk -- test says failed,
        # record says pass. One named single-file push produces that tree.
        import unittest.mock
        with unittest.mock.patch.dict("sys.modules", {"launch_gate": None}):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                record_launch_test(os.path.join(d, ARCH_TESTS[0]), "pass",
                                   dict(LAUNCH_SHAPE, layers=12), real_kernel=True,
                                   path=p2, root=d)
            assert "launch shape unknown" in buf.getvalue(), (
                f"a missing launch_gate must warn, not raise: {buf.getvalue()!r}")

        # The mix half of the same rule (de-10). Three cases, and the first is the one the
        # 20B launch actually hit: e2e passed on data/mix_sample.json while the launch mix
        # died at step 0, and the row said nothing either way because it had no mix field.
        for mix_arg, want in ((None, "MIX UNRECORDED"),
                              ("data/mix_sample.json", "NOT THE LAUNCH MIX")):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                record_launch_test(os.path.join(d, ARCH_TESTS[0]), "pass", dict(LAUNCH_SHAPE),
                                   real_kernel=True, mix=mix_arg, path=p2, root=d)
            out = buf.getvalue()
            assert want in out, f"mix={mix_arg!r} did not warn {want!r}: {out!r}"
            with open(p2, encoding="utf-8") as f:
                row = json.load(f)[shape_key(ARCH_TESTS[0], LAUNCH_SHAPE)]
            assert "mix" in row, ("the row has no mix field at all: a record that cannot "
                                  "say which data it ran on counts as no record (de-10)")
            assert row["mix"] == mix_arg, f"the row lost its mix: {row['mix']!r} != {mix_arg!r}"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            record_launch_test(os.path.join(d, ARCH_TESTS[0]), "pass", dict(LAUNCH_SHAPE),
                               real_kernel=True, mix=LAUNCH_MIX, path=p2, root=d)
        out = buf.getvalue()
        assert "NOT THE LAUNCH MIX" not in out and "MIX UNRECORDED" not in out, (
            f"the launch mix itself was flagged; a warning that fires on the good case "
            f"gets ignored on the bad one: {out!r}")

    print("launch_tests selftest OK: what the writer writes is what the gate accepts, "
          "a wrong-shape pass says so, and a missing launch_gate warns instead of raising")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else
             print(__doc__.strip()) or 0)

#!/usr/bin/env python3
"""E0-E6 evaluation ladder: the single entry point.

Contract: docs/standards/eval_ladder_v1_1009.md (from the v42 Engram handover doc,
2026-10-09). One subcommand per stage; the stage module owns its own checks and its
own artifacts under runs/ladder/<tag>/<stage>/.

Usage:
    python3 eval/eval_ladder.py e0 --ckpt ckpt_v42s2.pt
    python3 eval/eval_ladder.py e1 --ckpt ckpt_v42s2.pt --control
    python3 eval/eval_ladder.py selftest          # CPU, no checkpoint, all modules

Exit codes (same for every stage): 0 pass, 2 REFUSE (precondition unmet),
3 gate failed, 1 crashed.
"""
from __future__ import annotations

import argparse
import importlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import ladder_common as lc  # noqa: E402

STAGE_MODULES = {
    "e0": "eval.ladder.e0_identity",
    "e1": "eval.ladder.e1_base_gen",
    "e2": "eval.ladder.e2_chatml",
    "e3": "eval.ladder.e3_arith",
    "e4": "eval.ladder.e4_wordprob",
    "e5": "eval.ladder.e5_code",
    "e6": "eval.ladder.e6_bench",
}


def refuse_path_controls(stage, mod):
    """The error path must exit 2 with a REFUSE line, never die in its own handler.

    Handover doc P0: a verify script whose error path raised NameError reported the
    NameError instead of the missing input. The predicate here is the exit CODE and
    the summary's shape, not a message: a missing checkpoint must produce rc=2 and a
    summary with "refused", not rc=1 and a "crashed" traceback.

    Only `--ckpt` is passed: that flag is the one every stage shares. An argparse
    error is a SystemExit, which is a BaseException and would sail through an
    `except Exception` and out of the whole selftest -- measured 2026-10-09, when
    this control passed `--skip-forward` (an E0-only flag) to E3 and the dispatcher
    exited 2 with no FAIL line printed at all.
    """
    import contextlib
    import io
    import json
    import shutil

    missing = f"/nonexistent/_ladder_selftest_{stage}.pt"
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = mod.main(["--ckpt", missing])
    except SystemExit as exc:  # argparse exits 2 on an unknown flag
        rc = exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:  # noqa: BLE001
        return [f"{stage} refuse path raised {type(exc).__name__}: {exc}"]
    if rc != lc.EXIT_REFUSE:
        return [f"{stage} refuse path rc={rc}, want {lc.EXIT_REFUSE}: {buf.getvalue()[-300:]}"]
    # THE CODE ALONE IS NOT THE PREDICATE. An argparse error also exits 2, writes no
    # summary, and would pass this control while the stage never reached its own
    # refusal (codex, eval/eval_ladder.py:72, 2026-10-09). Require the refusal to be
    # stated: a REFUSE line on stdout, or a summary that says "refused".
    output = buf.getvalue()
    if "REFUSE:" not in output:
        return [f"{stage} refuse path exited 2 without a REFUSE line: {output[-300:]}"]
    tag = lc.ckpt_tag(missing)
    scratch = lc.LADDER / tag
    summary = scratch / stage / "summary.json"
    if not summary.exists():
        return [f"{stage} refuse path wrote no summary at {summary}"]
    got = json.loads(summary.read_text())
    if "refused" not in got:
        return [f"{stage} refuse path wrote {sorted(got)} (no 'refused')"]
    shutil.rmtree(scratch, ignore_errors=True)  # the control's own scratch tree, not an artifact
    print(f"{stage}: refuse path rc=2 + REFUSE line (missing checkpoint)")
    return []


def selftest(argv):
    """CPU-only: every stage module's own controls, plus this dispatcher's own checks.

    A module that does not exist yet is reported SKIP by name -- never silently.
    """
    ap = argparse.ArgumentParser(prog="eval_ladder.py selftest")
    ap.add_argument("--only", default=None, help="comma list of stages to run")
    args = ap.parse_args(argv)

    wanted = args.only.split(",") if args.only else list(STAGE_MODULES)
    failures, skipped = [], []
    # Dispatcher controls: the stop-reason rule and the coverage rule are the two
    # shared predicates every stage leans on, so they are controlled here.
    cases = [
        (lc.stop_reason(32764, 1, 32764, 5, 512), "im_end"),
        (lc.stop_reason(1, 1, 32764, 5, 512), "eos"),
        (lc.stop_reason(45, 1, 32764, 512, 512), "max_new"),
        (lc.stop_reason(45, 1, 32764, 12, 512), "other"),
        (lc.stop_reason(1, 1, 32764, 512, 512), "eos"),  # terminator wins over the cap
    ]
    for got, want in cases:
        if got != want:
            failures.append(f"stop_reason: got {got!r}, want {want!r}")
    ok, detail = lc.verify_coverage([{"id": "a"}, {"id": "b"}], ["a", "b"], label="control")
    if not ok:
        failures.append(f"verify_coverage valid world failed: {detail}")
    for rows, expect, why in (
        ([{"id": "a"}], ["a", "b"], "missing id"),
        ([{"id": "a"}, {"id": "a"}], ["a"], "duplicate id"),
        ([{"id": "a"}, {"id": "z"}], ["a", "b"], "extra id"),
        ([{"id": None}, {"id": "a"}], ["a"], "blank id (used to raise TypeError)"),
    ):
        ok, _ = lc.verify_coverage(rows, expect, label="control")
        if ok:
            failures.append(f"verify_coverage accepted a broken world ({why})")

    for stage in wanted:
        mod_name = STAGE_MODULES[stage]
        mod_file = ROOT / (mod_name.replace(".", "/") + ".py")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            # A MISSING FILE and a BROKEN IMPORT are different facts. Catching both as
            # SKIP let a stage with an unimportable dependency (say a missing numpy)
            # pass the whole selftest silently (codex, eval/eval_ladder.py:120,
            # 2026-10-09). The file on disk decides: absent -> SKIP, present -> FAIL.
            if mod_file.exists():
                failures.append(f"{stage} module {mod_file} exists but does not import: {exc}")
            else:
                skipped.append(f"{stage} (no module file)")
            continue
        fn = getattr(mod, "control_selftest", None)
        if fn is None:
            skipped.append(f"{stage} (no control_selftest)")
            continue
        rc = fn()
        print(f"{stage}: control_selftest rc={rc}")
        if rc != 0:
            failures.append(f"{stage} control_selftest rc={rc}")
        failures.extend(refuse_path_controls(stage, mod))

    for s in skipped:
        print(f"SKIP {s}")
    if failures:
        for f in failures:
            print(f"FAIL {f}", file=sys.stderr)
        return lc.EXIT_GATE
    print(f"PASS dispatcher controls; {len(wanted) - len(skipped)} stage modules ran")
    return lc.EXIT_OK


# The rungs of the ladder, in the order handover doc section 8 fixes. A later stage
# does not start when ANY rung under it failed ("前一级失败，后一级不放行"): every
# earlier rung in the checkpoint's chain must have a summary with exit_code 0.
# The chain is chosen by the checkpoint's own kind, read from E0's summary identity:
# a base checkpoint climbs e0/e1/e3/e4/e5/e6, an SFT checkpoint climbs e0/e2/e3/...
# `--skip-prereq` is the documented escape, for a harness check.
CHAIN_BASE = ("e0", "e1", "e3", "e4", "e5", "e6")
CHAIN_SFT = ("e0", "e2", "e3", "e4", "e5", "e6")


def check_prereq(stage, ckpt):
    """(ok, reason) for every rung under `stage` in this checkpoint's chain."""
    import json

    if stage == "e0":
        return True, ""  # E0 is the bottom rung; it cannot require itself
    tag = lc.ckpt_tag(ckpt)
    e0 = lc.LADDER / tag / "e0" / "summary.json"
    if not e0.exists():
        return False, (f"{stage} needs E0 first: {e0} does not exist -- run e0 on {ckpt}, "
                       f"or pass --skip-prereq for a harness check")
    kind = ((json.loads(e0.read_text()).get("identity") or {}).get("kind") or "").lower()
    chain = CHAIN_SFT if kind == "sft" else CHAIN_BASE
    if stage not in chain:
        return True, ""
    for need in chain[: chain.index(stage)]:
        summary = lc.LADDER / tag / need / "summary.json"
        if not summary.exists():
            return False, (f"{stage} needs {need} on {ckpt}, and {summary} does not exist -- "
                           f"run {need} first, or pass --skip-prereq for a harness check")
        got = json.loads(summary.read_text())
        code = got.get("exit_code")
        if code != lc.EXIT_OK:
            return False, (f"{stage} needs {need} to pass, and {need} recorded exit_code={code} "
                           f"gates={got.get('gates')} -- fix {need} or pass --skip-prereq for a "
                           f"harness check")
        # EXIT 0 IS NOT ENOUGH. `e1 --skip-forward` validated its prompt set, evaluated
        # nothing, and exited 0 with n=0 -- so the rung above it would have climbed onto
        # a stage that never ran (codex, eval/ladder/e1_base_gen.py:855, 2026-10-09).
        n = got.get("n")
        if not isinstance(n, int) or n <= 0:
            return False, (f"{stage} needs {need} to have evaluated something, and {need} "
                           f"recorded n={n!r} -- a stage that ran nothing is not a passed rung")
    return True, ""


def main(argv):
    ap = argparse.ArgumentParser(prog="eval_ladder.py", add_help=True)
    # DECLARED, not intercepted: scripts/hooks/pre-commit invokes every registered file as
    # `<python> <file> --selftest`, and harness check selftest_flags_accepted reads the parser
    # (AST) to prove the flag is accepted. An argv interception runs the selftest and still
    # fails that check -- the flag would be real to the caller and invisible to the gate, which
    # is the "green line that can pass by not running" shape (gate_failure_shapes §279).
    ap.add_argument("--selftest", action="store_true",
                    help="run the dispatcher controls and every stage module's own selftest "
                         "(the hook's spelling; the `selftest` subcommand is the same thing)")
    ap.add_argument("stage", nargs="?", choices=list(STAGE_MODULES) + ["selftest"])
    ap.add_argument("--skip-prereq", action="store_true",
                    help="harness check: run this stage even when the rung under it failed")
    stage, rest = ap.parse_known_args(argv)
    if stage.selftest or stage.stage == "selftest":
        return selftest(rest)
    if stage.stage is None:
        ap.error("a stage is required unless --selftest is passed")
    ckpt = None
    for i, tok in enumerate(rest):
        if tok == "--ckpt" and i + 1 < len(rest):
            ckpt = rest[i + 1]
        elif tok.startswith("--ckpt="):
            ckpt = tok.split("=", 1)[1]
    if ckpt and not stage.skip_prereq:
        ok, why = check_prereq(stage.stage, ckpt)
        if not ok:
            print(f"REFUSE: {why}")
            return lc.EXIT_REFUSE
    mod = importlib.import_module(STAGE_MODULES[stage.stage])
    return mod.main(rest)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

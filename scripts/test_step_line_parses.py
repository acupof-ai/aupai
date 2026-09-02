#!/usr/bin/env python3
# restartable: pure regex + string checks, no model, no GPU. Milliseconds.
"""RunLog._STEP_RE must still parse the step line train.py actually prints.

    python3 scripts/test_step_line_parses.py

WHY THIS EXISTS. train.py:2574 prints the step line; RunLog._STEP_RE (train.py:46) parses that
same line to feed trackio. Nothing connected the two, so editing the line could stop every
trackio metric with no error anywhere -- the log would look RICHER while the dashboard went
flat, which is the worst shape a logging defect can take.

b0-14 walked straight into it. The honest format for "show every optimizer's lr" drops the
privileged bare value:

    | lr muon 7.00e-03 embed 1.00e-01 scalar 7.00e-03 arq 7.00e-03 |

Measured against the real regex: that does NOT match. Neither does a slash-joined variant. The
landed format keeps the bare value first and appends the rest in parens, which does match --
but that constraint lived only in a comment until this file, and a comment does not fail.

This test builds the line the way train.py builds it (same f-string shape, same field order)
rather than pasting a literal from a log, so a reordered field is caught too.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _step_re():
    """The regex as train.py defines it, read from source rather than imported.

    Importing train.py pulls in torch, fla and a CUDA probe -- seconds, and it fails outright on
    a box with no GPU. The regex is two adjacent string literals in a class body, so reading it
    is exact and cheap. If the shape changes enough that this cannot find it, that is a failure:
    silently falling back to a hardcoded copy would test this file against itself.
    """
    src = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
    m = re.search(r"_STEP_RE = re\.compile\(\s*((?:r\"[^\"]*\"\s*)+)\)", src)
    if not m:
        sys.exit("could not find _STEP_RE in train.py: this test can no longer verify the "
                 "parser, which is a failure and not a reason to skip")
    pattern = "".join(re.findall(r'r"([^"]*)"', m.group(1)))
    return re.compile(pattern), pattern


def _step_line(lr_field):
    """The line train.py:2574 prints, with `lr <field>` substituted.

    Field order and separators copy the f-string. A field moved or a `|` dropped there and not
    here means this test passes on a line nobody prints -- so if train.py's line changes shape,
    this function is the thing to update, and the assertions below say what must stay true.
    """
    return (f"step 470/500 94% [main] | loss 2.540 | lr {lr_field} | gnorm 0.21 | 0.25B tok "
            f"| 57K tok/s/gpu | MFU 48% | peak 60.98GiB | ETA 0.0h")


def main():
    # The hook runs every SELFTEST_FILES entry as `<file> --selftest` (scripts/hooks/pre-commit:
    # 808). This file's whole body IS the test, so the flag is accepted and ignored rather than
    # required -- and argv is checked instead of ignored blindly, because a script that exits 0
    # on an unknown argument registers as a pass without running anything, which is the exact
    # failure the hook's own comment at :450 warns about.
    for arg in sys.argv[1:]:
        if arg != "--selftest":
            sys.exit(f"unknown argument {arg!r}: this file takes no arguments (--selftest is "
                     f"accepted so the commit hook can run it uniformly)")
    RE, pattern = _step_re()
    fails = []

    # 1. THE LANDED FORMAT: bare value first, per-group lrs in parens.
    landed = "7.00e-03 (muon 7.00e-03 embed 1.00e-01 scalar 7.00e-03 arq 7.00e-03)"
    m = RE.search(_step_line(landed))
    if not m:
        fails.append(f"the step line train.py prints does NOT parse. Pattern: {pattern}")
    else:
        step, loss, lr, gnorm, tps, mfu = m.groups()
        # Every captured group feeds a trackio metric, so a group that matches the WRONG token
        # is as bad as no match: it would log a plausible number for the wrong quantity.
        got = (step, loss, lr, gnorm, tps, mfu)
        want = ("470", "2.540", "7.00e-03", "0.21", "57", "48")
        if got != want:
            fails.append(f"the parser matched but captured {got}, expected {want} -- a metric "
                         f"is now being logged from the wrong field")

    # 2. THE FORMATS THAT SILENTLY BREAK IT, asserted as broken. Without this, a future edit to
    #    "just label every lr" reads as an improvement and passes every other check in the repo.
    for name, field in [
        ("fully labeled, no bare value", "muon 7.00e-03 embed 1.00e-01 scalar 7.00e-03"),
        ("slash-joined", "muon/embed/scalar 7.00e-03/1.00e-01/7.00e-03"),
    ]:
        if RE.search(_step_line(field)):
            fails.append(f"the {name} lr field now parses. If _STEP_RE was deliberately widened "
                         f"that is fine -- update this test and say so; if it was widened by "
                         f"accident, a metric may be reading the wrong token.")

    # 3. THE LINE train.py PRINTS MUST ACTUALLY CONTAIN THE PARENS FORM. Checks 1 and 2 test the
    #    regex against a line this file builds; if train.py's real f-string drifts to a format
    #    check 2 calls broken, nothing above notices. So read the source.
    src = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
    if "| lr {optimizers[0].param_groups[0]['lr']:.2e} ({lrs}) " not in src:
        fails.append("train.py's step line no longer prints `lr <bare> (<per-group>)`. If the "
                     "format changed, check 2's rejected formats are the ones to re-measure "
                     "against _STEP_RE before trusting trackio.")
    # And every optimizer must be named at the SITE THAT ASSIGNS the name, not merely mentioned.
    # Measured: asserting `"aupai_group" in src` passed with the assignment deleted, because the
    # step line's own getattr("aupai_group") keeps the string present -- the check was reading
    # the reader, not the writer. So match the assignment itself.
    if not re.search(r"\.aupai_group\s*=", src):
        fails.append("build_optimizers no longer ASSIGNS aupai_group, so the step line falls back "
                     "to opt0/opt1/... and the per-group names come from a construction order "
                     "the reader cannot see -- which is the b0-14 misread all over again")

    if fails:
        for f in fails:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("test_step_line_parses OK: RunLog._STEP_RE parses the step line train.py prints and "
          "captures all six metrics from the right fields; the two formats that would have "
          "silently stopped trackio (a fully labeled lr field, a slash-joined one) are asserted "
          "to still NOT match, so 'label every lr' cannot land as an invisible regression; and "
          "the source is checked to still print the parens form and to tag every optimizer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

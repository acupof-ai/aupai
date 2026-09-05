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

    THE TRAILING `s/step` IS PART OF THE REAL LINE (prereg#moe_0905 amendment_12_steady_state)
    and is included here for that reason: the field was APPENDED rather than inserted, because
    _STEP_RE ends with `([\\d.]+)K tok/s/gpu \\| MFU (\\d+)%` and matches on adjacency, so a
    field placed between those two would stop every trackio metric on this line. Building the
    line WITH the new field is what makes check 1 evidence that appending was safe.
    """
    return (f"step 470/500 94% [main] | loss 2.540 | lr {lr_field} | gnorm 0.21 | 0.25B tok "
            f"| 57K tok/s/gpu | MFU 48% | peak 60.98GiB | ETA 0.0h | s/step 2.2824")


def _val_re():
    """RunLog._VAL_RE, read from source for the same reason as _step_re."""
    src = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
    m = re.search(r"_VAL_RE = re\.compile\(r\"([^\"]*)\"\)", src)
    if not m:
        sys.exit("could not find _VAL_RE in train.py: this test can no longer verify the val "
                 "parser, which is a failure and not a reason to skip")
    return re.compile(m.group(1)), m.group(1)


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

    # 4. THE FULL-PRECISION TIMING FIELDS, prereg#moe_0905 amendment_12_steady_state. The adopt
    #    comparison is steady-state seconds, and before these two fields it was computable only
    #    from `{tps/1e3:.0f}K` (+/-0.8%, ~110s over a 3815-step extrapolation) on any DENSE arm --
    #    full-precision tok/s reached moe_diag and memory_diag, and a dense arm writes neither.
    #    So the pair had a precise clock on one side and a rounded one on the other.
    #
    #    ASSERTED AT THE PRINT SITE, not by parsing a log and not by matching the rendered line
    #    this file builds: checks 1-2 would pass unchanged if train.py stopped printing both
    #    fields tomorrow, because _step_line() is this file's own string. The f-string fragment is
    #    the writer.
    if "f\" | s/step {dt / 10:.4f}\"" not in src:
        fails.append("train.py's step line no longer prints the unrounded `s/step` field, so "
                     "steady-state seconds is back to being computable only from the rounded "
                     "{tps:.0f}K -- +/-0.8%, and on a dense arm there is no other timing source "
                     "(moe_diag/memory_diag are MoE- and memory-arm only)")
    if "val_s {_val_s:.2f} val_s_total {_val_s_total:.1f}" not in src:
        fails.append("train.py's val line no longer prints val_s/val_s_total. validate() runs "
                     "INSIDE the tps window (train.py:3340 before the tps compute at :3357 from "
                     "the same `now - t_log`), so without these the validation term has to be "
                     "ESTIMATED out of steady-state seconds instead of measured")
    if not re.search(r"_val_s_total\s*=\s*0\.0", src):
        fails.append("_val_s_total is never initialised, so the val line would raise NameError "
                     "on the first validation pass -- the field exists and the run dies")
    # `s/step` MUST BE LAST, because that is the only position that leaves _STEP_RE's
    # `tok/s/gpu | MFU` adjacency intact. Measured against the real regex rather than argued:
    # the same line with the field moved before MFU must NOT parse.
    moved = ("step 470/500 94% [main] | loss 2.540 | lr 7.00e-03 (muon 7.00e-03) | gnorm 0.21 "
             "| 0.25B tok | 57K tok/s/gpu | s/step 2.2824 | MFU 48% | peak 60.98GiB | ETA 0.0h")
    if RE.search(moved):
        fails.append("a line with `s/step` inserted between tok/s/gpu and MFU now parses, so the "
                     "adjacency this field was appended to preserve is no longer enforced -- the "
                     "next person to insert a field there will not be caught")
    # AND THE ADJACENCY AT THE PRINT SITE, which the two checks above do not cover between them.
    # Measured: moving the field into the middle of train.py's own f-string turns this file red
    # only via the "no longer prints s/step" check above, whose message is then FALSE -- the field
    # is printed, just in the position that silently kills trackio. A refusal that misdescribes
    # the defect sends the next reader looking for a deleted field, which is the failure shape
    # test_moe_module check 11 spent a day in (it reported a guard as ABSENT when the guard had
    # merely grown a term). So assert the two fields are adjacent in the SOURCE.
    if not re.search(r"K tok/s/gpu \| MFU \{mfu", src):
        fails.append("train.py's step line no longer prints `tok/s/gpu | MFU` ADJACENTLY. Some "
                     "field was inserted between them, and _STEP_RE matches on that adjacency, "
                     "so every trackio metric on this line is now silently dropped while the log "
                     "looks richer. Append the field at the end of the line instead.")
    # And the val parser must still read the val line WITH the two new fields appended.
    VAL, vpat = _val_re()
    vm = VAL.match("step 3800/3815 val 2.259 val_s 16.08 val_s_total 305.5")
    if not vm:
        fails.append(f"the val line train.py now prints does NOT parse. Pattern: {vpat}")
    elif vm.groups() != ("3800", "2.259"):
        fails.append(f"_VAL_RE captured {vm.groups()}, expected ('3800', '2.259') -- val_s is "
                     f"being read as the val loss")

    if fails:
        for f in fails:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("test_step_line_parses OK: RunLog._STEP_RE parses the step line train.py prints and "
          "captures all six metrics from the right fields; the two formats that would have "
          "silently stopped trackio (a fully labeled lr field, a slash-joined one) are asserted "
          "to still NOT match, so 'label every lr' cannot land as an invisible regression; the "
          "source is checked to still print the parens form and to tag every optimizer; and the "
          "full-precision timing fields (s/step appended last, val_s/val_s_total with the "
          "counter initialised) are asserted at the PRINT SITE, with a line carrying s/step "
          "between tok/s/gpu and MFU asserted to still NOT parse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

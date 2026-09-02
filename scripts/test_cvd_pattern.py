#!/usr/bin/env python3
# restartable: pure string matching, no state, no card. A rerun is free.
"""What `CUDA_VISIBLE_DEVICES=<x>` assignments harness's device_set_honoured accepts.

    python3 scripts/test_cvd_pattern.py --selftest

WHY THIS FILE. device_set_honoured enforces "never take a card the caller did not give
you" by matching assignment SYNTAX (harness.py:6381 states that ceiling itself). Its
accept-list is one regex, and both directions of a wrong regex are expensive and quiet:

  too narrow   a legitimate form is refused, so the way to get green is to leave the
               variable inherited -- i.e. a pure-CPU script keeps the ability to open
               every visible card. That ambiguity turned a CPU counting job into a
               card-ownership investigation on 2026-09-03.
  too wide     a physical index passes and a shard escapes its lane onto a training
               card, which is the incident the check was written for (2026-08-31,
               eval/code_zh.py onto GPU 0, commit 2f97e4a).

The check's own broken-fixture proves it fails on a bare `=0`. Nothing proved which OTHER
forms it accepts or refuses, so widening the regex for `=""` had no way to show it had not
also let `="$CARDS"` through. This table is that proof, and it is what a future widening
has to keep green.

`=""` and `=-1` are accepted as NO DEVICE. Measured on the pod 2026-09-03, not assumed:
CUDA_VISIBLE_DEVICES="" gives torch.cuda.device_count() 0 and is_available() False; -1
also gives 0; unset gives 8. Asking for zero cards cannot escape a lane, so it is the
strongest compliance with the rule -- while being syntactically indistinguishable from
writing a physical index, which is why the check needed telling.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

#: (assignment as it would appear in a script, accepted?, why it is that way)
CASES = [
    # no device: cannot take a card, so it cannot take the caller's
    ('CUDA_VISIBLE_DEVICES=""', True, "no device (measured: device_count 0)"),
    ("CUDA_VISIBLE_DEVICES=''", True, "no device, single-quoted"),
    ("CUDA_VISIBLE_DEVICES=-1", True, "no device (measured: device_count 0)"),
    # the sanctioned idioms
    ("export CUDA_VISIBLE_DEVICES=${_DEVS[0]}", True, "eval/_devs.sh, the accepted idiom"),
    ("CUDA_VISIBLE_DEVICES=${_DEVS[$i]}", True, "eval/_devs.sh, indexed"),
    ("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}", True, "inherits the caller's set"),
    # escapes: every one of these can land on a card the caller did not grant
    ("CUDA_VISIBLE_DEVICES=0", False, "physical index -- the 2026-08-31 incident"),
    ("CUDA_VISIBLE_DEVICES=7", False, "physical index"),
    ("CUDA_VISIBLE_DEVICES=$i", False, "computed index, escapes when N > caller's count"),
    ('CUDA_VISIBLE_DEVICES="0,1"', False, "physical list"),
    ('CUDA_VISIBLE_DEVICES="$CARDS"', False, "opaque variable: could hold anything"),
    ("CUDA_VISIBLE_DEVICES=${_DEVS[$i]:-$i}", True,
     "accepted, and correctly so: the spill this form once had is now prevented at the "
     "SOURCE -- eval/_devs.sh:24-28 refuses when N exceeds the caller's device count, so "
     "the :-$i branch is unreachable. harness.py:6379 records it as a historical survival, "
     "not a live hole. Kept in the table so a change to _devs.sh that removes that refusal "
     "has a second place that has to be reconsidered."),
]


def run():
    from harness import _CVD_ASSIGN, _CVD_SAFE

    fails = []
    for line, want, why in CASES:
        m = _CVD_ASSIGN.search(" " + line)
        if not m:
            fails.append(f"not even recognised as an assignment: {line!r}")
            continue
        got = bool(_CVD_SAFE.match(m.group(1)))
        if got != want:
            fails.append(f"{line!r} accepted={got}, expected {want} ({why})")
    # The two directions must both be represented, or a regex of `.*` (accept everything)
    # or `$^` (accept nothing) would pass whichever half the table happened to contain.
    if not any(w for _, w, _ in CASES) or not any(not w for _, w, _ in CASES):
        fails.append("the table has only one direction, so it cannot detect an "
                     "accept-everything or refuse-everything regex")
    return fails


def check_devs_still_refuses_overflow():
    """`${_DEVS[$i]:-$i}` is accepted ONLY because eval/_devs.sh refuses the overflow.

    That is the whole reason the table's expectation for it is True. If the refusal in
    _devs.sh goes away, the fallback becomes reachable and the accept becomes a hole -- so
    the dependency is asserted here rather than left as a comment. This reads the source,
    which is a weaker check than running it, and says so: it proves the refusal is still
    written, not that it still fires.
    """
    path = os.path.join(ROOT, "eval", "_devs.sh")
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as e:
        return [f"cannot read eval/_devs.sh ({e}) -- the accept of ${{_DEVS[$i]:-$i}} rests "
                f"on its overflow refusal and that could not be confirmed"]
    body = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    if "refusing:" not in body or "-gt" not in body:
        return ["eval/_devs.sh no longer refuses when N exceeds the caller's device count, "
                "so ${_DEVS[$i]:-$i} can spill to a physical index again and must no longer "
                "be accepted by _CVD_SAFE"]
    return []


def selftest():
    fails = run() + check_devs_still_refuses_overflow()
    # Prove the table can fail: an accept-everything regex must break it. Without this the
    # table's green says nothing about whether it is looking at the regex at all.
    import re

    import harness
    real = harness._CVD_SAFE
    try:
        harness._CVD_SAFE = re.compile(r"^.*$")
        if not run():
            fails.append("an accept-everything regex passed the table -- the table is "
                         "not actually testing _CVD_SAFE")
        harness._CVD_SAFE = re.compile(r"^\0$")
        if not run():
            fails.append("a refuse-everything regex passed the table")
    finally:
        harness._CVD_SAFE = real

    for f in fails:
        print(f"  SELFTEST FAIL {f}")
    if fails:
        print(f"\n{len(fails)} selftest failure(s)")
        return 1
    n_ok = sum(1 for _, w, _ in CASES if w)
    print(f"test_cvd_pattern selftest OK ({n_ok} accepted / {len(CASES)-n_ok} refused forms, "
          f"and the table fails on an accept-everything regex)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    return selftest()


if __name__ == "__main__":
    sys.exit(main())

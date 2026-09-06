"""Attribute the eight causeless domain_bpb rows in runs/score_matrix.jsonl (e1-41).

WHAT THE ROWS SAY. Each records, as its whole diagnosis:

    domain_bpb.py exited 1: /work/aupai/eval/domain_bpb.py:221: UserWarning: checkpoint has
    no vocab_id (old format); cannot cross-check tokenizer | ours_tok = load_tokenizer(...)

A warning and a source line. Neither is the cause, and I opened e1-41 calling them
unrecoverable. THAT WAS WRONG, and this script is the correction: the cause is not in the
record, but it is in the CODE THAT RAN, which git still has.

THE CAUSE. All eight rows were scored under eval/domain_bpb.py at e84fd88b or earlier, which
carried MIN_ROUNDTRIP = 0.98 (:49) and skipped any domain whose decode round-trip fell below it
(:289). Measured in def65db5's own commit message: the id round-trip was 0.9826 overall and
below 0.98 for zh_web (0.9375), cot and chatml (0.9688) -- so the gate skipped ALL NINE domains,
`out` came out empty, and the metric took the `if not out` branch at :316, printing
"REFUSING: no domain produced a number" and returning 1.

WHY THE RECORD LOST IT. That refusal goes to STDOUT. The capture in score_matrix.py at the
time was `(r.stderr or r.stdout)`, which discards stdout entirely whenever stderr holds
anything -- and scripts/loader.py warns on stderr for every old-format checkpoint. So the
warning survived and the refusal did not. Both defects are since fixed (794299f6 keeps both
streams, def65db5 replaced the round-trip fraction with exact text identity), and both fixes
landed AFTER every one of these rows.

WHAT THIS SCRIPT DOES. It proves that attribution against the ledger and the git history
rather than asserting it: every row is checked to predate both fixes, to carry the warning-tail
shape, and to contain no trace of the refusal. It writes nothing. The disposition e1-41 asks
for is a decision, not an edit, and the ledger is append-only: these rows are a true record of
a real attempt that failed, and the honest repair is a re-score, not a rewritten row.

    python3 scripts/e1_41_domain_bpb_attribution.py
"""

import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "runs", "score_matrix.jsonl")

# The two fixes, and what each one would have changed had it been in place.
GATE_FIX = "def65db5"     # text identity replaces MIN_ROUNDTRIP: the domains stop being skipped
CAPTURE_FIX = "794299f6"  # both streams kept: the refusal reaches the record

WARNING_TAIL = "UserWarning: checkpoint has no vocab_id"
# THE OTHER TAIL, and the reason this is not a one-shape check. e1-41 says "eight rows"; the
# warning tail matches only SEVEN. The eighth, ckpt_data_leg_206m_8b.pt.step10000, records
# `domain_bpb.py exited 1:   ours_tok = load_tokenizer(a.tokenizer, None)` -- the source line
# ALONE, with the warning gone. Same defect, one line further along: `(r.stderr or r.stdout)`
# kept stderr, and on that run stderr held only the warning's source echo, not the warning. So a
# predicate written on the warning text would have found 7 and silently confirmed a count of 8.
SOURCE_TAIL = "ours_tok = load_tokenizer"
# If either of these appeared in a row, the refusal WAS captured and the row is not this shape.
REFUSAL_MARKERS = ("REFUSING", "round-trip")


def _commit_time(sha):
    """Author time of sha as 'YYYY-MM-DD HH:MM', or None."""
    r = subprocess.run(["git", "-C", ROOT, "log", "-1", "--format=%ad",
                        "--date=format:%Y-%m-%d %H:%M", sha],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def _row_landed(ckpt):
    """When the ledger line naming ckpt was first committed, oldest first.

    The row's own `ts` is absent on these (they came off the pod through pod_push), so the
    commit that introduced the line is the only timestamp the repo holds for it.
    """
    r = subprocess.run(["git", "-C", ROOT, "log", "--format=%ad", "--date=format:%Y-%m-%d %H:%M",
                        "--all", "-S", ckpt, "--", "runs/score_matrix.jsonl"],
                       capture_output=True, text=True)
    times = [t for t in r.stdout.strip().splitlines() if t.strip()]
    return times[-1] if times else None


def affected():
    """[(ckpt, error)] for every domain_bpb entry whose whole diagnosis is a leaked tail.

    Both tails, not just the warning one: see SOURCE_TAIL. The test is "does this error text
    name a cause", answered as "no" by matching either leak, and a row carrying a real
    exception, a refusal or the card-claim message is excluded by not matching either.
    """
    out = []
    with open(LEDGER, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        v = (row.get("metrics") or {}).get("domain_bpb")
        if not (isinstance(v, dict) and "error" in v):
            continue
        err = str(v["error"])
        if WARNING_TAIL in err or SOURCE_TAIL in err:
            out.append((row.get("ckpt"), err))
    return out


def old_gate_present(sha):
    """Whether eval/domain_bpb.py at sha still carried the round-trip gate.

    THE ASSIGNMENT, not the name. `"MIN_ROUNDTRIP" in src` was here and it passes on def65db5 --
    the commit that DELETED the gate -- because its replacement comment says "MIN_ROUNDTRIP = 0.98
    was here and it was wrong" and three more lines cite the defect by name (4 hits at HEAD). So
    the substring test attributed these rows to a version that could not have produced them, and
    the world built to catch exactly that survived. Matching `MIN_ROUNDTRIP = <number>` at the
    start of a line is the difference between the constant existing and being talked about.
    """
    r = subprocess.run(["git", "-C", ROOT, "show", f"{sha}:eval/domain_bpb.py"],
                       capture_output=True, text=True)
    return bool(re.search(r"^MIN_ROUNDTRIP\s*=\s*[\d.]+", r.stdout, re.M)), r.stdout


def main():
    rows = affected()
    if not rows:
        print("no warning-tail domain_bpb rows: either they were re-scored or the shape changed")
        return 1

    gate_t, cap_t = _commit_time(GATE_FIX), _commit_time(CAPTURE_FIX)
    print(f"the two fixes: {GATE_FIX} (text identity) {gate_t}; "
          f"{CAPTURE_FIX} (both streams) {cap_t}\n")

    bad = []
    print(f"{len(rows)} affected metric entries:")
    for ckpt, err in rows:
        landed = _row_landed(ckpt)
        before_gate = landed is not None and landed < gate_t
        before_cap = landed is not None and landed < cap_t
        captured = [m for m in REFUSAL_MARKERS if m in err]
        flag = "" if (before_gate and before_cap and not captured) else "  <-- NOT THIS SHAPE"
        print(f"  {ckpt:42s} landed {landed}  pre-gate={before_gate} "
              f"pre-capture={before_cap}{flag}")
        if not before_gate:
            bad.append(f"{ckpt}: landed {landed}, NOT before the gate fix at {gate_t} -- "
                       f"the round-trip gate cannot be its cause")
        if not before_cap:
            bad.append(f"{ckpt}: landed {landed}, NOT before the capture fix at {cap_t} -- "
                       f"the refusal should have been recorded and is not")
        if captured:
            bad.append(f"{ckpt}: the record DOES contain {captured} -- the cause was captured "
                       f"after all and this row is not causeless")

    had_gate, src = old_gate_present("e84fd88b")
    print(f"\neval/domain_bpb.py at e84fd88b (the version live when these ran): "
          f"MIN_ROUNDTRIP present = {had_gate}")
    if not had_gate:
        bad.append("e84fd88b does not carry MIN_ROUNDTRIP, so the attribution is wrong")
    else:
        m = re.search(r"MIN_ROUNDTRIP\s*=\s*([\d.]+)", src)
        print(f"  MIN_ROUNDTRIP = {m.group(1) if m else '?'}, and def65db5's message measures the "
              f"id round-trip at 0.9826 overall,\n  0.9375 (zh_web) / 0.9688 (cot, chatml) -- "
              f"below the threshold, so all nine domains were skipped and\n  the metric took its "
              f"`if not out` branch: 'REFUSING: no domain produced a number', return 1.")

    if bad:
        print(f"\n{len(bad)} row(s) do not fit the attribution:")
        for b in bad:
            print(f"  {b}")
        return 1
    print(f"\nATTRIBUTED: all {len(rows)} entries were scored by a domain_bpb carrying the "
          f"round-trip gate, and\nthe refusal it printed went to stdout, which the capture of the "
          f"day discarded. The cause is\nrecoverable from the code, NOT from the record -- "
          f"correcting e1-41's 'unrecoverable'.\n\nDisposition: leave the rows. They are a true "
          f"record of an attempt that failed, the ledger is\nappend-only, and rewriting them to "
          f"name a cause no capture produced would be manufacturing\nevidence. A re-score under "
          f"today's domain_bpb is what replaces them, and that needs a card.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

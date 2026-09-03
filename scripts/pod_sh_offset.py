#!/usr/bin/env python3
"""Is it safe to overwrite a .sh that is EXECUTING on the pod? Answer from its byte offset.

`pod_push.sh` refuses to push a script that is running, because podput writes with `>` --
truncating the same inode -- and bash reads a script incrementally by byte offset, so the
live shell resumes at its old offset inside the new bytes. `POD_PUSH_ALLOW_RUNNING_SH=1`
overrides that refusal, and until now the override was unconditional: the operator asserted
the edit was safe and nothing checked it.

THE SAFETY IS A PROPERTY OF THE DIFF, NOT OF THE FLAG. On 2026-09-04 an edit to
run_ddp.sh's scoring block was pushed under the override while two runs were mid-script,
and it was safe -- every added byte landed after the offset both shells had already read.
The same flag on an edit touching an earlier byte would have been unsafe with no warning.
So the operator's measurement becomes the tool's: this script recomputes it.

    python3 scripts/pod_sh_offset.py --check run_ddp.sh     # exit 0 safe, 2 refuse
    python3 scripts/pod_sh_offset.py --selftest

MEASURED on the pod 2026-09-04, which is where the mechanism comes from: bash holds the
script it is executing on fd 255, and /proc/<pid>/fdinfo/255 reads `pos: 4391` -- the byte
after the newline ending the torchrun line both shells were blocked in.
"""

import argparse
import contextlib
import os
import subprocess
import sys

POD = os.path.expanduser("~/bin/pod")
WORK = "/work/aupai"


def first_diff(old, new):
    """Index of the first byte that differs, or None when the two are identical.

    A pure prefix counts as differing at the shorter length: bash resuming past a new EOF
    is the truncation case, and it is unsafe for exactly the same reason a changed byte is.
    """
    n = min(len(old), len(new))
    for i in range(n):
        if old[i] != new[i]:
            return i
    return None if len(old) == len(new) else n


def parse_shells(text):
    """[(pid, stat, fd, offset)] from the pod probe's output, plus the pids it could not read.

    Returns (shells, unreadable). `unreadable` holds pids that HOLD THE FILE OPEN but whose
    offset could not be read. That list must make the caller REFUSE, never pass: "no offset
    found" and "nothing is running" are different facts, and reading the first as the second
    permits exactly the push this gate exists to stop.

    A pid with no matching fd never reaches here -- _probe drops it, because a process that
    merely mentions the path in its argv (the launcher wrapper, the probe's own shell) holds
    no cursor into the file. The parser still handles a PID line with no FD line, since a
    process can close the fd between the readlink and the fdinfo read.

    ZOMBIES ARE DROPPED, and this is not hypothetical: `pgrep -f run_ddp.sh` returned eight
    pids on the pod 2026-09-04 and FOUR were zombies (2332282, 2557639, 2558388, 2664687,
    state Z, `[run_ddp.sh] <defunct>`). A zombie keeps its /proc entry and its argv but has
    zero fds, so it yields no offset -- and counting it as unreadable would make this gate a
    permanent refusal, while counting it as live-with-no-offset is the same bug wearing the
    other hat. It is neither: it is not running.
    """
    shells, unreadable, pid, stat = [], [], None, None

    def close(p, s):
        if p is not None and s is not None and not s.startswith("Z") and not any(x[0] == p for x in shells):
            unreadable.append((p, s))

    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "PID" and len(parts) >= 3:
            close(pid, stat)
            pid, stat = parts[1], parts[2]
        elif parts[0] == "FD" and len(parts) >= 3 and pid is not None:
            if stat and stat.startswith("Z"):
                continue  # a zombie cannot hold an fd; if one is reported, it is not ours
            # a fd row with no parseable pos is no offset: close() then lands it in
            # `unreadable`, which refuses -- never treated as "nothing is running".
            with contextlib.suppress(ValueError):
                shells.append((pid, stat, parts[1], int(parts[2])))
    close(pid, stat)
    return shells, unreadable


def verdict(old, new, shells, unreadable, rel):
    """(ok, message). Safe only when every differing byte is at or after EVERY live offset.

    The threshold is the SMALLEST offset across live shells, not the first one found: two
    runs can sit at different points in the same script, and a byte safe for the one further
    along is unsafe for the one behind it. Both read 4391 on 2026-09-04, which is precisely
    why taking the first would have looked correct.
    """
    if unreadable:
        who = ", ".join(f"{p} (stat {s})" for p, s in unreadable[:4])
        return False, (
            f"REFUSING {rel}: {len(unreadable)} live shell(s) are executing it but their "
            f"script offset could not be read: {who}. A live shell with no offset is not a "
            f"safe push -- 'no offset found' is not 'nothing is running'. Wait for it to "
            f'finish, or verify by hand with: ~/bin/pod "cat /proc/<pid>/fdinfo/255"'
        )
    if not shells:
        return True, f"{rel} is not executing on the pod (no live shell holds it)"
    diff = first_diff(old, new)
    if diff is None:
        return True, f"{rel} is byte-identical to the pod copy; nothing to corrupt"
    lo = min(off for _p, _s, _fd, off in shells)
    where = ", ".join(f"pid {p} fd {fd} pos {off}" for p, _s, fd, off in shells)
    if diff < lo:
        return False, (
            f"REFUSING {rel}: byte {diff} differs and the earliest live shell has already "
            f"read up to {lo} ({where}). Overwriting would make a running shell resume "
            f"inside bytes it never read as the file it started. This is what "
            f"POD_PUSH_ALLOW_RUNNING_SH cannot know: the safety is a property of the diff, "
            f"not of the flag. Wait for the run to finish."
        )
    return True, (
        f"{rel} is safe to overwrite: first differing byte {diff} is at or after every live "
        f"shell's offset (earliest {lo}; {where}). {len(new) - len(old):+d} bytes."
    )


def _probe(rel):
    """Ask the pod which live shells hold this script OPEN, and at what offset.

    ~/bin/pod, never `tn exec`: /proc and the fd table must be read in the SAME namespace
    the pids came from, and the host view numbers the same process differently (AGENTS, Pod).
    The fd is DISCOVERED by readlink rather than hardcoded to 255 -- 255 is what bash uses
    today, and a gate that assumes it would silently find no offset if that changed, which
    lands in `unreadable` and refuses. Failing safe either way.

    THE POPULATION IS "HOLDS THE FILE OPEN", NOT "MENTIONS IT IN ARGV", and this cost a
    wrong refusal before it was measured. `pgrep -f run_ddp.sh` matched three processes with
    no fd on the file: the LAUNCHER wrapper (`bash -lc cd /work/aupai && ... ./run_ddp.sh
    ...`, pid 4147355), and the probe's own shell, whose command line contains the pattern it
    is searching for. None of them is executing the script -- a launcher that already exec'd
    or forked holds no cursor into it -- so reporting them as live-with-no-offset made the
    gate refuse a push that was safe, for a reason that did not exist. Only a process the
    kernel says has the file open can have a byte offset in it, so the fd table IS the
    population: a pid with no matching fd is dropped, not called unreadable. What stays
    unreadable is a pid WITH a matching fd whose pos cannot be read -- the case that must
    still refuse.
    """
    path = f"{WORK}/{rel}"
    cmd = (
        f"for p in $(pgrep -f '{rel}' 2>/dev/null); do "
        f'  hit=""; '
        f"  for fd in /proc/$p/fd/*; do "
        f'    t=$(readlink "$fd" 2>/dev/null) || continue; '
        f'    [ "$t" = "{path}" ] || continue; '
        f'    hit=1; n=$(basename "$fd"); '
        f'    st=$(ps -o stat= -p $p 2>/dev/null | tr -d " "); echo "PID $p ${{st:-?}}"; '
        f"    echo \"FD $n $(awk '/^pos/{{print $2}}' /proc/$p/fdinfo/$n 2>/dev/null)\"; "
        f"  done; "
        f'  [ -n "$hit" ] || continue; '
        f"done"
    )
    r = subprocess.run([POD, cmd], capture_output=True, text=True, timeout=120)
    return r.stdout


def _pod_bytes(rel):
    """The pod's current copy of the file. base64 so a CRLF or a high byte survives."""
    r = subprocess.run(
        [POD, f"cd {WORK} && base64 -- {rel} 2>/dev/null"], capture_output=True, text=True, timeout=120
    )
    import base64

    try:
        return base64.b64decode(r.stdout)
    except Exception:
        return None


def check(rel):
    if not os.path.exists(POD):
        print(
            f"{POD} not present -- cannot verify; this must run where the pod is reachable", file=sys.stderr
        )
        return 2
    with open(rel, "rb") as fh:
        new = fh.read()
    old = _pod_bytes(rel)
    if old is None:
        print(f"REFUSING {rel}: could not read the pod's copy to diff against", file=sys.stderr)
        return 2
    shells, unreadable = parse_shells(_probe(rel))
    ok, msg = verdict(old, new, shells, unreadable, rel)
    print(msg if ok else msg, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 2


def _selftest():
    bad = 0

    def case(cond, label):
        nonlocal bad
        bad += 0 if cond else 1
        print(f"  {'ok  ' if cond else 'BUG '} {label}")

    # THE FIXTURE IS THE REAL SHAPE, from the real measurement: a shell at 4391 in a 6133-byte
    # script, pushed to 7799 bytes by appending after the offset (the de-47 push).
    old = b"A" * 4391 + b"tail-of-old\n"
    after = b"A" * 4391 + b"tail-of-old\nappended block\n"
    before = b"B" + b"A" * 4390 + b"tail-of-old\n"
    live = [("4147358", "S", "255", 4391)]

    ok, msg = verdict(old, after, live, [], "run_ddp.sh")
    # The byte is 4403 -- 4391 A's plus "tail-of-old\n" -- COMPUTED, not guessed. My first
    # version wrote `ok and "4402" in msg or ok`, which short-circuits to `ok` and never
    # required the substring at all: an assertion whose failure mode is "always true" is the
    # same defect this file exists to prevent, one level up.
    case(
        ok and str(len(old)) in msg,
        f"a diff entirely AFTER the offset is safe, naming byte {len(old)} ({msg[:52]})",
    )
    ok2, msg2 = verdict(old, before, live, [], "run_ddp.sh")
    case(
        not ok2 and "byte 0" in msg2 and "4391" in msg2,
        f"a byte changed BEFORE the offset refuses, naming both ({msg2[:70]})",
    )

    # THE POSITIVE IS NOT OPTIONAL. Without it every assertion above passes on an
    # implementation that refuses everything -- the shape that bit me on the CVD check.
    case(ok and not ok2, "the two verdicts DIFFER (the gate is not a constant)")

    # TRUNCATION: a new file shorter than the offset leaves the shell resuming past EOF.
    ok3, msg3 = verdict(old, b"A" * 100, live, [], "run_ddp.sh")
    case(not ok3 and "100" in msg3, f"truncation below the offset refuses ({msg3[:60]})")

    # THE SMALLEST OFFSET, not the first found. Two shells, the second further behind.
    two = [("4147358", "S", "255", 4391), ("4160562", "S", "255", 10)]
    ok4, _ = verdict(old, before, two, [], "run_ddp.sh")
    ok5, msg5 = verdict(old, b"A" * 4391 + b"X", two, [], "run_ddp.sh")
    case(not ok4, "a byte before the EARLIEST offset refuses even when another shell is past it")
    case(ok5 and "earliest 10" in msg5, f"and the message names the earliest ({msg5[:70]})")

    # ZOMBIES ARE NOT LIVE SHELLS. Measured: four of eight pgrep hits on the pod were Z.
    z = "PID 2332282 Z\nPID 2557639 Z\nPID 4147358 S\nFD 255 4391\n"
    shells, unread = parse_shells(z)
    case(len(shells) == 1 and shells[0][0] == "4147358", f"four pgrep hits, one live shell parsed ({shells})")
    case(unread == [], f"and no zombie lands in `unreadable` ({unread})")

    # A LIVE SHELL WITH NO OFFSET MUST REFUSE. This is the dangerous reading: my first probe
    # hit a zombie, saw an empty /proc/<pid>/fd/, and it looked like fdinfo carried no offset
    # at all. If that had been a LIVE shell, passing would allow the exact push this stops.
    shells2, unread2 = parse_shells("PID 4147358 S\nPID 4160562 S\nFD 255 4391\n")
    case(
        len(shells2) == 1 and len(unread2) == 1 and unread2[0][0] == "4147358",
        f"a live pid with no fd row is unreadable, not absent ({unread2})",
    )
    ok6, msg6 = verdict(old, after, shells2, unread2, "run_ddp.sh")
    case(
        not ok6 and "could not be read" in msg6,
        f"...and an unreadable live shell REFUSES a diff that is otherwise safe ({msg6[:60]})",
    )

    # An fd row with no parseable pos is the same case: no offset, so it must not pass as safe.
    shells3, unread3 = parse_shells("PID 4147358 S\nFD 255 \n")
    ok7, _ = verdict(old, after, shells3, unread3, "run_ddp.sh")
    case(not ok7, "an fd row with no parseable pos refuses rather than passing")

    # AN ARGV-ONLY MATCH IS NOT AN EXECUTING SHELL, and this was a real wrong refusal before
    # it was measured against the pod: `pgrep -f run_ddp.sh` matched the LAUNCHER wrapper
    # (`bash -lc cd /work/aupai && ... ./run_ddp.sh ...`, pid 4147355) and the probe's own
    # shell, neither of which holds an fd on the file. _probe drops a pid with no matching fd,
    # so those never reach the parser -- the population is "holds the file open", because only
    # a process the kernel says has it open can have an offset in it. The gate refused a safe
    # push for a reason that did not exist.
    shells4, unread4 = parse_shells("PID 4147358 S\nFD 255 4391\n")
    case(
        len(shells4) == 1 and unread4 == [],
        f"only fd-holding pids are parsed; argv-only matches are absent by construction "
        f"({shells4}, {unread4})",
    )

    # Nothing running at all: the gate must not block a normal push.
    ok8, msg8 = verdict(old, before, [], [], "run_ddp.sh")
    case(ok8, f"no live shell: a push is allowed even with an early diff ({msg8[:50]})")

    # Identical bytes: podput would be a no-op, so a running shell is irrelevant.
    ok9, _ = verdict(old, old, live, [], "run_ddp.sh")
    case(ok9, "byte-identical content is safe even mid-run")

    # first_diff's own contract, including the boundary the whole gate turns on.
    case(first_diff(b"abc", b"abc") is None, "first_diff: identical -> None")
    case(first_diff(b"abc", b"abd") == 2, "first_diff: last byte -> 2")
    case(first_diff(b"abc", b"abcd") == 3, "first_diff: pure append -> len(old)")
    case(first_diff(b"abcd", b"abc") == 3, "first_diff: truncation -> len(new)")
    case(first_diff(b"", b"x") == 0, "first_diff: empty vs one byte -> 0")

    print(f"pod_sh_offset selftest: {17 - bad}/17 pass")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", metavar="REL", help="repo-relative .sh to verify before a push")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.check:
        ap.error("--check REL or --selftest")
    return check(a.check)


if __name__ == "__main__":
    sys.exit(main())

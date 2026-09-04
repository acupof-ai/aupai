#!/usr/bin/env python3
"""Supply the .rc file and the watchdog for a run that was launched WITHOUT `harness launch`.

WHY THIS EXISTS, and it is remediation, not a feature. `harness launch` does four things:
writes the exp row, acquires the card claim, wraps the command so the shell records $? into
runs/<name>.rc, and arms harness._arm_monitor. On 2026-09-04 I launched the head-hybrid A/B by
calling run_ddp.sh directly and got none of them. The row and the claim can be written after the
fact. The .rc CANNOT: only the shell that reaped the child knows its exit code, and that shell
was never told to record it.

That gap is not cosmetic, because _arm_monitor reads the .rc to decide what the death MEANT:

    rc == 0     -> ok,   "process exited cleanly"
    rc missing  -> fail, "vanished: no exit code recorded"
    rc == N     -> fail, "exit N (signal N-128)"

So arming the harness monitor on a run with no .rc guarantees a `fail | vanished` row even on a
perfect 3815-step finish. Its own docstring records why it fails closed: the 22B milestone was
KILLED at 04:22 and its row read `ok | process exited`, indistinguishable from a completed score
(de, 2026-09-01). Fail-closed is right; supplying a fake `0` would re-open exactly that hole,
which is why the first version of this -- a one-line `printf 0 > runs/<name>.rc` after the wait --
was wrong and was killed by pid before it could write anything.

WHAT THIS DOES INSTEAD. It cannot recover the true exit code, so it does not claim one. It reads
the run's own log for a terminal signal and writes an rc that matches what the log SHOWS:

    the log's last step line reached max_steps, or it printed a completion marker  -> 0
    the log ends with a traceback / CUDA error / OOM                               -> 1
    neither: the process is gone and the log says nothing terminal                 -> NO .rc

The third case is the honest one and it is why this is not just a wrapper: a run that vanished
with a silent log gets no .rc, the monitor says `vanished`, and that is the correct verdict --
the fate really is unknown. This narrows the unknown from "every outcome" to "only the genuinely
ambiguous ones".

AND IT WATCHES THE NEIGHBOUR. Arm B shares card 1 with a foreign process holding 54.7 GB against
arm B's 51.76 GiB peak on a ~95 GiB card (6e, 2026-09-04). An OOM there is not a defect in the
arm, it is death by co-residency, so the OOM/CUDA-error scan writes rc=1 AND names the card in the
finding, because "arm B failed" and "arm B was killed by a neighbour on a granted card" call for
different responses -- the first retires a design, the second reruns it on a clean pair.

Run one per job, detached:
    setsid nohup python3 runs/audit_0904/b0_rc_watchdog.py <name> <pid> </dev/null >/dev/null 2>&1 &
"""
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FATAL = re.compile(r"Traceback \(most recent call last\)|CUDA error|out of memory|"
                   r"CUDA out of memory|NCCL.*(?:error|timeout)|Killed", re.I)
STEP = re.compile(r"^step (\d+)/(\d+)")


def alive(pid):
    """A ZOMBIE IS DEAD. os.kill(pid, 0) succeeds on an exited-but-unreaped child, so the
    signal probe alone waits forever on a finished run -- harness.py:14155 and :6418 both
    record this trap, and it has been hit more than once here."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True  # no procfs: trust the signal probe


def read_log(path, tail_bytes=200_000):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - tail_bytes))
            return f.read()
    except OSError:
        return ""


def verdict(log, ckpt_path=None):
    """(rc, why) from what the log and the on-disk checkpoint SHOW, or (None, why) when they
    show nothing terminal.

    REACHING step N/N IS NOT COMPLETION (6e, 2026-09-04). After the last step the run still has
    to write ckpt_<name>.pt, and a death inside that save leaves a log whose last line says
    3815/3815 -- which the first version of this read as a clean exit. So rc 0 requires an
    artifact, not a step count: either the final checkpoint exists with a size stable across two
    polls (a save still in flight grows between them), or the log printed its own completion
    marker. The step count alone now yields None, which the monitor reports as `vanished`.
    """
    m = FATAL.search(log)
    if m:
        tail = log[max(0, m.start() - 200):m.start() + 400].strip().splitlines()
        return 1, f"log holds {m.group(0)!r}; context: {' | '.join(tail[-3:])[:300]}"
    if re.search(r"run complete|training complete", log, re.I):
        return 0, "the log printed its own completion marker"
    steps = STEP.findall(log)
    reached = bool(steps) and int(steps[-1][0]) >= int(steps[-1][1])
    if reached and ckpt_path:
        # SIZE STABLE ACROSS TWO POLLS, because a torch.save in flight grows. 15 s is far longer
        # than the write takes for a 540 MB checkpoint and costs nothing here: the process is
        # already gone, so nothing is waiting on this answer.
        try:
            a = os.path.getsize(ckpt_path)
            time.sleep(15)
            b = os.path.getsize(ckpt_path)
        except OSError:
            return None, (f"the log reached step {steps[-1][0]}/{steps[-1][1]} but "
                          f"{os.path.basename(ckpt_path)} does not exist -- the run died in or "
                          f"before its final save, which a step count cannot distinguish from a "
                          f"clean finish. No .rc; 'vanished' is the correct verdict.")
        if a == b and a > 0:
            return 0, (f"log reached step {steps[-1][0]}/{steps[-1][1]} AND "
                       f"{os.path.basename(ckpt_path)} is {a} B, stable across two polls 15 s apart")
        return None, (f"the log reached step {steps[-1][0]}/{steps[-1][1]} but "
                      f"{os.path.basename(ckpt_path)} changed size between polls ({a} -> {b}) or "
                      f"is empty: the save was still in flight when the process vanished")
    if reached:
        return None, (f"the log reached step {steps[-1][0]}/{steps[-1][1]} but no checkpoint path "
                      f"was given to check, and a step count alone cannot rule out a death inside "
                      f"the final save")
    return None, ("the process is gone and the log shows neither completion nor a fatal error, "
                  "so the exit code is genuinely unknown -- no .rc is written and the monitor's "
                  "'vanished' verdict is correct")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {os.path.basename(__file__)} <run-name> <pid>")
    name, pid = sys.argv[1], int(sys.argv[2])
    log_path = os.path.join(ROOT, "runs", f"{name}.log")
    rc_path = os.path.join(ROOT, "runs", f"{name}.rc")
    note_path = os.path.join(ROOT, "runs", f"{name}.rc_why")
    oom_flag = os.path.join(ROOT, "runs", f"{name}.OOM_ALERT")

    while alive(pid):
        # The neighbour watch runs WHILE the job lives, not after: an OOM alert that arrives
        # with the post-mortem is too late to be an alert.
        log = read_log(log_path, 40_000)
        if re.search(r"out of memory|CUDA out of memory", log, re.I) and not os.path.exists(oom_flag):
            with open(oom_flag, "w", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {name} pid {pid}: "
                        f"the log reports an out-of-memory. If this is arm B on card 1, the card "
                        f"is shared with a foreign process (54.7 GB at 09:2xZ) and this is death "
                        f"by co-residency on a granted card, not a defect in the arm.\n")
        time.sleep(30)

    rc, why = verdict(read_log(log_path), os.path.join(ROOT, f"ckpt_{name}.pt"))
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} pid {pid} gone. "
                f"rc={rc!r} because {why}\n"
                f"NOTE: this rc is DERIVED FROM THE LOG, not from the shell that reaped the "
                f"process -- this run was launched without `harness launch`, so no wrapper "
                f"recorded $?. Treat it as evidence, not as the exit code.\n")
    if rc is not None:
        with open(rc_path, "w", encoding="utf-8") as f:
            f.write(str(rc))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Divergence watchdog for a training run (v41_ced_0923 resume, prereg amendment_2).

Read-only on the log. Sends SIGTERM to the run's torchrun PIDs when either fires:
  - gnorm > G on N consecutive progress lines, or
  - mean train loss over the last W progress lines > L (needs W records).
A read error, or a progress-shaped line whose numbers do not parse, is a HARD error
(exit 3), never a silent "healthy": a watchdog that cannot read its log is blind.

Exit codes: 0 healthy to EOF (--once); 2 triggered (and signalled unless --dry-run);
3 read/parse error; 4 log stale (no new progress line within --max-stale seconds);
5 target torchrun process not found at trigger time.

The parse/decide layer (parse_step_line / DivergeDetector) is pure and known-answer
tested in scripts/test_ced_diverge_watch.py --selftest (synthetic); the real segment-1
known-answer test runs on the pod against the actual seg1 log via --once.
"""

import argparse
import collections
import contextlib
import os
import re
import signal
import subprocess
import sys
import time

PROGRESS_RE = re.compile(r"^step (\d+)/\d+ .*\[main\].*$")
LOSS_RE = re.compile(r"\bloss ([0-9.eE+-]+)")
GNORM_RE = re.compile(r"\bgnorm ([0-9.eE+-]+)")


class LogParseError(Exception):
    pass


def parse_step_line(line):
    """Return (step, loss, gnorm) for a [main] progress line, None for any other line.

    A line that HAS the progress shape but whose loss/gnorm cannot be read raises:
    matching the shape yet failing to yield numbers is an instrument fault, not data.
    """
    m = PROGRESS_RE.match(line.rstrip("\n"))
    if not m:
        return None
    step = int(m.group(1))
    lm, gm = LOSS_RE.search(line), GNORM_RE.search(line)
    if not lm or not gm:
        raise LogParseError(f"progress line missing loss/gnorm: {line.strip()[:160]}")
    try:
        loss, gnorm = float(lm.group(1)), float(gm.group(1))
    except ValueError as e:
        raise LogParseError(f"non-numeric loss/gnorm: {line.strip()[:160]}") from e
    return step, loss, gnorm


class DivergeDetector:
    def __init__(self, g_thresh=10.0, g_consec=3, loss_window=50, loss_thresh=3.0):
        self.g_thresh = g_thresh
        self.g_consec = g_consec
        self.loss_thresh = loss_thresh
        self.losses = collections.deque(maxlen=loss_window)
        self._g_run = 0
        self.last_step = None

    def feed(self, step, loss, gnorm):
        """Return a reason string on divergence, else None. One trigger, then sticky."""
        import math

        if not (math.isfinite(loss) and math.isfinite(gnorm)):
            return f"nonfinite at step {step} (loss={loss} gnorm={gnorm})"
        if gnorm > self.g_thresh:
            self._g_run += 1
            if self._g_run >= self.g_consec:
                return (
                    f"gnorm {gnorm:g} > {self.g_thresh:g} for {self._g_run} consecutive lines, at step {step}"
                )
        else:
            self._g_run = 0
        self.losses.append(loss)
        if len(self.losses) == self.losses.maxlen:
            mean = sum(self.losses) / len(self.losses)
            if mean > self.loss_thresh:
                return f"{self.losses.maxlen}-step mean loss {mean:.3f} > {self.loss_thresh:g} at step {step}"
        self.last_step = step
        return None


def cmdline_matches(raw_cmdline, name):
    """True for a torchrun launcher whose argv carries `--name <name>` adjacently.

    /proc/<pid>/cmdline separates argv with NUL bytes, never spaces -- matching the
    substring "--name name" with a space fails against every real process. Parse tokens.
    """
    toks = raw_cmdline.split("\x00")
    if not any("torchrun" in t for t in toks):
        return False
    return any(
        toks[i] == "--name" and i + 1 < len(toks) and toks[i + 1] == name for i in range(len(toks) - 1)
    )


def resolve_torchrun_pids(name):
    """Exact PIDs of the torchrun LAUNCHER(s) whose argv carries `--name <name>`.

    pgrep -f narrows to cmdlines containing torchrun; worker processes are `python train.py`
    and do not contain that token, so only the elastic launcher is returned. SIGTERM to the
    launcher makes the elastic agent tear down every rank. [] if none matches.
    """
    out = subprocess.run(["pgrep", "-f", "torchrun"], capture_output=True, text=True)
    pids = []
    for tok in out.stdout.split():
        try:
            pid = int(tok)
            with open(f"/proc/{pid}/cmdline", "rb") as c:
                raw = c.read().decode(errors="replace")
        except (OSError, ValueError):
            continue
        if cmdline_matches(raw, name):
            pids.append(pid)
    return sorted(set(pids))


def _term(pids):
    for pid in pids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)


def scan_lines(lines, det, lo=None, hi=None):
    """Feed an iterable of lines; return (reason_or_None, parsed_count). Raises on bad line.

    Only progress lines with lo <= step < hi are fed (bounds optional), so a real log can
    be replayed over a named step window for the known-answer test.
    """
    n = 0
    for line in lines:
        rec = parse_step_line(line)
        if rec is None:
            continue
        if (lo is not None and rec[0] < lo) or (hi is not None and rec[0] >= hi):
            continue
        reason = det.feed(*rec)
        n += 1
        if reason:
            return reason, n
    return None, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--name", required=True, help="run name token matched in torchrun cmdline")
    ap.add_argument("--g_thresh", type=float, default=10.0)
    ap.add_argument("--g_consec", type=int, default=3)
    ap.add_argument("--loss_window", type=int, default=50)
    ap.add_argument("--loss_thresh", type=float, default=3.0)
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument(
        "--max_stale",
        type=float,
        default=1800.0,
        help="seconds with no new progress line before a STALE (exit 4) report",
    )
    ap.add_argument(
        "--ready_wait",
        type=float,
        default=900.0,
        help="seconds to wait for the log to first appear (launch race)",
    )
    ap.add_argument("--once", action="store_true", help="scan whole file once, do not tail")
    ap.add_argument("--from_step", type=int, default=None, help="once: only feed steps >= N")
    ap.add_argument("--to_step", type=int, default=None, help="once: only feed steps < N")
    ap.add_argument("--dry_run", action="store_true", help="report triggers, send no signal")
    args = ap.parse_args()

    det = DivergeDetector(args.g_thresh, args.g_consec, args.loss_window, args.loss_thresh)

    def act(reason):
        print(f"WATCHDOG TRIGGER {args.name}: {reason}", flush=True)
        if args.dry_run:
            print("dry-run: no signal sent", flush=True)
            return 2
        pids = resolve_torchrun_pids(args.name)
        if not pids:
            print(
                "WATCHDOG ERROR: divergence fired but no torchrun pid matched "
                f"--name {args.name}; run not signalled",
                flush=True,
            )
            return 5
        print(f"WATCHDOG SIGTERM torchrun pids {pids}", flush=True)
        _term(pids)
        return 2

    if args.once:
        try:
            with open(args.log, errors="replace") as f:
                reason, n = scan_lines(f, det, args.from_step, args.to_step)
        except OSError as e:
            print(f"WATCHDOG READ ERROR {args.log}: {e}", flush=True)
            return 3
        except LogParseError as e:
            print(f"WATCHDOG PARSE ERROR {args.log}: {e}", flush=True)
            return 3
        print(f"scanned {n} progress lines; {'TRIGGER: ' + reason if reason else 'silent'}", flush=True)
        if reason:
            return act(reason)
        return 0

    # live tail. Wait for the file to appear (launch race); that wait is not a read fault.
    # On open, seek to EOF: a resumed run reuses its append log, and the file already on
    # disk is HISTORY (it may contain a prior segment's divergence). Tailing from offset 0
    # would feed that history to the detector and SIGTERM a healthy resume for a runaway
    # that happened before this watchdog started. Guard only lines appended after attach.
    f, waited = None, 0.0
    while f is None:
        try:
            f = open(args.log, errors="replace")  # noqa: SIM115 handle outlives this loop, tailed below
            f.seek(0, os.SEEK_END)
        except OSError:
            if waited >= args.ready_wait:
                print(f"WATCHDOG READ ERROR {args.log}: did not appear within {args.ready_wait}s", flush=True)
                return 3
            time.sleep(args.interval)
            waited += args.interval
    # staleness is time since the last PROGRESS line, not since any byte -- a stuck run that
    # keeps emitting non-progress chatter must still cross max_stale. Initialized at attach;
    # if no progress line arrives within max_stale (and ready_wait already passed), it fires.
    last_progress = time.time()
    while True:
        line = f.readline()
        if not line:
            if time.time() - last_progress > args.max_stale:
                print(
                    f"WATCHDOG STALE {args.log}: no progress line for {args.max_stale}s "
                    f"(last step {det.last_step}); reporting, not killing",
                    flush=True,
                )
                return 4
            try:
                time.sleep(args.interval)
            except KeyboardInterrupt:
                return 0
            continue
        try:
            rec = parse_step_line(line)
        except LogParseError as e:
            print(f"WATCHDOG PARSE ERROR {args.log}: {e}", flush=True)
            return 3
        if rec is not None:
            last_progress = time.time()
            reason = det.feed(*rec)
            if reason:
                return act(reason)


if __name__ == "__main__":
    sys.exit(main())

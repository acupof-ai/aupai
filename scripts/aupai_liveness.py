#!/usr/bin/env python3
"""Which aupai panes are working, which are thinking, which are idle.

Replaces ~/bin/aupai_liveness.sh, whose three failure modes are answered by three
signal choices rather than by tuning thresholds:

1. `caffeinate` is excluded from workers. Claude spawns it for every Bash call, so
   counting it made every pane RUNNING forever.
2. The spinner symbol is not read at all. tmux's ✻/✢ appear on both a live turn and
   the finished "✻ Brewed for 1h · done 8:51PM" line, so counting glyphs cannot
   separate the two. Claude Code already maintains the state machine: the session
   file at ~/.claude/sessions/<pid>.json carries `status` (busy | idle | shell) and
   `statusUpdatedAt`. That is a field, not a rendering.
3. A single frame cannot tell a long think from a stopped one, so two clocks are
   read together: how long the status has stood, and how long since the pane's own
   transcript last grew. A busy status with a transcript that stopped growing an
   hour ago is not a think.

THE PANE -> SESSION JOIN IS THE WHOLE CORRECTNESS ARGUMENT, and it is why this does
not scan ~/.claude/projects. Every session file names its own `tmux` pane
("aupai:@5.%5") and its own `sessionId`, so the transcript is derived from the pane
rather than guessed by mtime. Scanning the projects directory was measured and
refuted: it reports 0 for every pane because the newest transcript there belongs to
whichever session last wrote, not to the pane you asked about.

Exit status is informational only: 0 when every pane resolved, 1 when any pane
could not be mapped to a session file.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

RUNNING, THINKING, IDLE, STALE, UNKNOWN = "RUNNING", "THINKING", "IDLE", "STALE", "UNKNOWN"

# Commands that are NOT the pane doing work. caffeinate is the anti-sleep wrapper
# Claude attaches to every Bash call (trap 1); the rest are this script's own
# transport and the shell that runs it. A pane was measured RUNNING purely because
# the liveness probe itself was running inside it.
NOISE_COMMANDS = {
    "caffeinate", "sleep", "ps", "pgrep", "awk", "sed", "grep", "sort", "uniq",
    "wc", "head", "tail", "cat", "date", "cut", "tr", "bash", "sh", "zsh", "sleep",
}

# A busy status older than this, over a transcript that also stopped growing, is
# not a think. 0e was measured in an 18-minute xhigh think, so the floor is above
# that: this reports STALE rather than silently calling a dead busy pane IDLE.
STALE_AFTER_S = 1800.0


def classify(status, status_age, tr_age, workers, stale_after=STALE_AFTER_S):
    """The verdict, as a pure function of the signals -- so the selftest can drive it.

    Order matters and each step is a decision the .sh version got wrong:

    - Real workers first, and independent of `status`: a pane that spawned pytest is
      RUNNING even if a status field has not caught up.
    - `shell` is RUNNING because a shell job IS attached work, and the child may be
      a grandchild this walk cannot see.
    - `busy` with no worker is the long-think case, which is why `busy` cannot be
      the RUNNING test.
    - STALE is keyed on the TRANSCRIPT ALONE, never on max(status_age, tr_age).
      That was the first version and it was measured wrong within the hour: tilerl
      rev-8d showed status_age 2876s beside tr_age 7s and was called STALE while its
      transcript was growing 22KB per 20s -- an active pane. `statusUpdatedAt` moves
      only on a status TRANSITION, so a pane that has been continuously busy has an
      arbitrarily old one, and taking the max of two clocks with different semantics
      reports the oldest as if it were the quietest. The transcript is the one that
      answers "is anything still happening"; the status field only says which MODE it
      is in. A busy pane whose transcript stopped moving is the wedged case; a busy
      pane whose status stamp is old but whose transcript is live is simply busy.
    """
    if workers:
        return RUNNING, f"workers: {', '.join(sorted(set(workers))[:3])}"
    if status == "shell":
        return RUNNING, "status=shell"
    if status == "busy":
        if tr_age is not None and tr_age > stale_after:
            return STALE, (f"status=busy but the transcript has not grown for {tr_age:.0f}s")
        return THINKING, f"status=busy ({status_age:.0f}s), no worker"
    if status == "idle":
        return IDLE, "status=idle"
    if status is None:
        return UNKNOWN, "no session file for this pane"
    return UNKNOWN, f"status={status!r}"


def _children_map():
    """pid -> [(child_pid, comm), ...]. Read once; the walk is over this, not /proc."""
    out = subprocess.run(["ps", "-eo", "pid,ppid,comm"], capture_output=True, text=True)
    ch = {}
    for line in out.stdout.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) == 3:
            ch.setdefault(parts[1], []).append((parts[0], parts[2]))
    return ch


def workers_under(pid, ch, exclude_pids=(), depth=4):
    """Descendant commands that are not transport.

    exclude_pids carries this probe's own ancestry: without it the pane running the
    probe reports the probe. Depth is bounded because the walk is over live process
    state and a reparented grandchild can make it arbitrarily deep.
    """
    found, stack = [], [(str(pid), 0)]
    seen = set()
    while stack:
        p, d = stack.pop()
        if p in seen or d > depth:
            continue
        seen.add(p)
        for cpid, comm in ch.get(p, []):
            if cpid in exclude_pids:
                continue
            base = os.path.basename(comm)
            if base in NOISE_COMMANDS:
                stack.append((cpid, d + 1))
                continue
            found.append(base)
            stack.append((cpid, d + 1))
    return found


def _transcript_age(cwd, session_id, now):
    slug = cwd.replace("/", "-").replace(".", "-")
    path = os.path.expanduser(f"~/.claude/projects/{slug}/{session_id}.jsonl")
    try:
        return now - os.path.getmtime(path), path
    except OSError:
        return None, path


def scan(sessions_dir, tmux_session, now=None, ch=None, exclude_pids=()):
    now = now or time.time()
    ch = ch if ch is not None else _children_map()
    rows = []
    for f in sorted(glob.glob(os.path.join(sessions_dir, "*.json"))):
        try:
            d = json.load(open(f))
        except (OSError, ValueError):
            continue
        tmux = d.get("tmux") or ""
        if not tmux.startswith(tmux_session + ":"):
            continue
        pane = tmux.split(":", 1)[1]
        cwd, sid = d.get("cwd", ""), d.get("sessionId", "")
        status = d.get("status")
        supd = d.get("statusUpdatedAt")
        status_age = (now - supd / 1000.0) if supd else None
        tr_age, tpath = _transcript_age(cwd, sid, now)
        wrk = workers_under(d.get("pid"), ch, exclude_pids=exclude_pids)
        verdict, why = classify(status, status_age, tr_age, wrk)
        rows.append({
            "pane": pane, "name": d.get("name") or os.path.basename(cwd), "pid": d.get("pid"),
            "verdict": verdict, "why": why, "status": status,
            "status_age": status_age, "tr_age": tr_age, "workers": wrk, "transcript": tpath,
        })
    rows.sort(key=lambda r: r["pane"])
    return rows


def render(rows):
    print(f"  {'pane':<8}{'session':<18}{'verdict':<10}{'status':<8}{'stale_s':>8}{'tr_s':>8}  why")
    for r in rows:
        sa = f"{r['status_age']:.0f}" if r["status_age"] is not None else "-"
        ta = f"{r['tr_age']:.0f}" if r["tr_age"] is not None else "-"
        print(f"  {r['pane']:<8}{r['name']:<18}{r['verdict']:<10}{str(r['status']):<8}"
              f"{sa:>8}{ta:>8}  {r['why']}")


def _selftest():
    """Three known-answer worlds, plus mutants that must break them.

    A liveness tool that reads green while blind is the failure this replaces, so the
    gate is the three states fb named -- confirmed IDLE, confirmed THINKING (a long
    xhigh think with no child), confirmed RUNNING (a real worker) -- and each verdict
    is asserted on the SIGNAL that carries it. Removing that signal must move the
    verdict, or the case was passing for a reason other than the one it claims.
    """
    cases = [
        # label,            status,  status_age, tr_age, workers,        want,      mutant_drop
        ("confirmed IDLE",     "idle",  120.0,     120.0,  [],            IDLE,      "status"),
        ("18m xhigh THINKING", "busy",  1080.0,    1.0,    [],            THINKING,  "status"),
        ("THINKING, long gap", "busy",  300.0,     280.0,  [],            THINKING,  "status"),
        # THE MEASURED CASE, and the one the first version failed: statusUpdatedAt moves
        # only on a transition, so a continuously-busy pane carries a stale stamp beside a
        # live transcript. tilerl rev-8d: status_age 2876s, tr_age 7s, +22KB/20s.
        ("busy, stale stamp, live transcript", "busy", 2876.0, 7.0, [],
         THINKING, "transcript"),
        ("RUNNING (python)",   "busy",  60.0,      2.0,    ["python3.11"], RUNNING,   "workers"),
        ("RUNNING (pytest)",   "busy",  60.0,      2.0,    ["pytest"],    RUNNING,   "workers"),
        ("RUNNING (shell)",    "shell", 30.0,      30.0,   [],            RUNNING,   "status"),
        ("stale busy",         "busy",  4000.0,    4000.0, [],            STALE,     "status"),
        ("no session file",    None,    None,      None,   [],            UNKNOWN,   "none"),
    ]
    fails = []
    for label, status, sa, ta, wrk, want, drop in cases:
        got, why = classify(status, sa, ta, wrk)
        if got != want:
            fails.append(f"{label}: got {got}, want {want} ({why})")
            continue
        # MUTATION: drop the signal this case claims to rest on. The verdict must move.
        if drop == "status":
            m_got, _ = classify(None, sa, ta, wrk)
        elif drop == "workers":
            m_got, _ = classify(status, sa, ta, [])
        elif drop == "transcript":
            # Move the signal the case rests on: a transcript that stopped growing.
            m_got, _ = classify(status, sa, sa, wrk)
        else:
            continue
        if m_got == want:
            fails.append(f"{label}: removing the {drop} signal left the verdict at {want} "
                         f"-- the case does not test what it names")

    # The two traps, asserted as behaviour rather than described in a comment.
    fake_ch = {"100": [("200", "claude"), ("201", "caffeinate")],
               "200": [("300", "caffeinate"), ("301", "python3.11")]}
    w = workers_under("100", fake_ch)
    if "caffeinate" in w:
        fails.append(f"caffeinate counted as a worker: {w} (trap 1)")
    if "python3.11" not in w:
        fails.append(f"a real grandchild worker was missed: {w}")
    # The probe's own tree must not appear: that is how a pane running the probe
    # reports itself RUNNING.
    w_self = workers_under("100", fake_ch, exclude_pids={"301"})
    if "python3.11" in w_self:
        fails.append(f"an excluded pid was still counted: {w_self}")

    if fails:
        print("SELFTEST FAILED", file=sys.stderr)
        for x in fails:
            print(f"  {x}", file=sys.stderr)
        return 1
    print(f"liveness selftest ok: {len(cases)} known-answer states judged correctly, each "
          f"reddens when its own signal is removed; caffeinate excluded and a real "
          f"grandchild worker found; an excluded pid stays excluded")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true", help="known-answer gate, no live read")
    ap.add_argument("--sessions-dir", default=os.path.expanduser("~/.claude/sessions"))
    ap.add_argument("--tmux-session", default="aupai")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    ch = _children_map()
    excl = {str(os.getpid()), str(os.getppid())}
    for p in list(excl):
        for cpid, _c in ch.get(p, []):
            excl.add(cpid)
    rows = scan(a.sessions_dir, a.tmux_session, ch=ch, exclude_pids=excl)
    if a.json:
        print(json.dumps(rows, indent=1, default=str))
    else:
        render(rows)
        if rows:
            counts = {}
            for r in rows:
                counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
            print("  " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return 1 if any(r["verdict"] == UNKNOWN for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())

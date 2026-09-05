#!/usr/bin/env python3
"""Card claims: a holder DECLARES the cards it uses; nobody infers ownership (de-30).

WHY. Ownership was read off nvidia-smi, and an instantaneous reading cannot answer "is this
card mine to take". Three failures on 2026-09-02, all from that one gap:

  11:28Z  a probe read 0 MiB on all eight cards, then spent ~2.5 min in build_mix loading
          156 GB of token caches before its first GPU allocation. Another eight-card job took
          every card inside that window. The reading was TRUE and already stale when used.
  11:49Z  the same shape again, in the other direction: a monitor matched occupancy by
          COMMAND LINE, a different job's cmdline did not match, "no match" was read as
          "cards free", and a probe launched onto 95.2 GiB.
  all day free-card's lane is `all_cards - config["cards"]`, so under NGPU=8 it hands back
          card 7 -- which the eight-card run itself is using. A config complement cannot
          describe a run that took the whole machine.

A claim closes the gap because it is a fact about INTENT, written before the first allocation
and removed after the last, whereas memory.used is a fact about the present instant. The
window between "I checked" and "I allocated" stops mattering.

WHAT THIS IS NOT. Not a replacement for reading the cards: a claim says who intends to hold a
card, memory.used says who actually holds one, and the interesting states are the DISAGREEMENTS
-- a claim with no memory yet (starting up), and memory with no claim (an orphan). Both are
reported, neither is silently reconciled.

    python3 scripts/card_claim.py acquire --name p200m --cards 0,1,2,3,4,5,6,7 [--wait 1800]
    python3 scripts/card_claim.py release --name p200m
    python3 scripts/card_claim.py status          # claims, and the disagreements with the cards
    python3 scripts/card_claim.py --selftest      # 9 known answers, no cards needed
"""

import argparse
import errno
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAIM_DIR = os.environ.get("AUPAI_CLAIM_DIR") or os.path.join(ROOT, "runs", "claims")
# Below this, a card counts as free. Idle H20s report a few MiB of context, and the settle
# windows elsewhere in the repo use the same figure.
FREE_MIB = 64
# The seam nvidia_fds reads through, so the selftest can build a /proc-shaped world on macOS.
PROC_ROOT = "/proc"
# How long a launch waits for the job to open a device. Measured 1.33s on the pod in the harness
# wrapper shape; 90s is ~68x that and is NOT waiting for build_mix, which runs after set_device.
# None = poll while the wrapper process lives. NOT a constant: run_ddp.sh's drift gate, mix
# assertion and 155 GiB cache load all precede the ranks, and that launch's own startup gate was
# 793 s -- any ceiling short of the job's own startup abandons a live run's cards (4c, 2026-09-05).
DEVICE_WAIT_S = None


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def _cmdline(pid):
    """The claimant's own cmdline, read in the namespace that will do the killing.

    A PID is only meaningful in the namespace that read it -- the host sees a rank as
    1738493 while the container sees 1382917 -- so the claim records BOTH the pid and the
    cmdline, and cross-boundary readers match on the cmdline (AGENTS, Pod).

    /proc FIRST, `ps -o args=` SECOND. The /proc-only version returned "" on macOS, where there
    is no /proc and where the selftest runs -- so every claim written on a laptop recorded an
    empty cmdline, and the cmdline is the half of the identity that survives a namespace
    crossing. `_alive` already carried the note "this also has to work on a Mac"; its neighbour
    did not. Measured 2026-09-03: /proc raises FileNotFoundError here, `ps -o args=` returns the
    full `bash -lc env ... ; true`.
    """
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            got = fh.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        if got:
            return got
    except OSError:
        pass
    try:
        r = subprocess.run(["ps", "-o", "args=", "-p", str(pid)], capture_output=True, text=True)
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _proc_stat(pid, table=None):
    """The process state letter for pid ('Z', 'S', 'R', ...), or "" if unknown.

    `ps -o stat=` is the reader, because it is the only one that answers the question. See
    _alive: signal 0 and /proc both report a ZOMBIE as present.
    """
    try:
        r = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout.strip().split()[0] if r.stdout.strip() else ""


def _is_zombie(pid):
    """Whether pid is a reaped-pending corpse.

    MEASURED on the pod 2026-09-03 against b0's pid 3891063, `Zs`, etime 07:02: os.kill(pid, 0)
    returned ALIVE and /proc/3891063 was present. Reproduced locally with a real fork+exit.
    So _alive() is True for a zombie and claims() files it as LIVE, which means the card stays
    refused after the job ended -- the same trap as fb's 31-minute wait on `[ -d /proc/<pid> ]`
    (AGENTS.md, Pod). ps's state letter is the only reader that distinguishes the two.
    """
    return _proc_stat(pid).startswith("Z")


def _cvd(pid):
    """pid's CUDA_VISIBLE_DEVICES AS IT WAS AT EXEC, "" if it had none, None if unreadable.

    EXEC-TIME is the whole design constraint, and it is why the original ruling's "refuse when
    absent" half is not implemented. Both readers below see the environment the process was
    started with, so a job that sets the variable ITSELF reads absent here -- MEASURED
    2026-09-04, a child assigning os.environ["CUDA_VISIBLE_DEVICES"] read ABSENT while a child
    given it by its parent read its value. Four scripts in this repo do exactly that
    (scripts/test_e2e.py:37, bench_eff/{bench_eff,bench_eff2,bench_opt}.py's
    os.environ.setdefault), so absent is the normal reading for correct code and only a
    DISAGREEING value is evidence of anything.

    /proc first (the pod), `ps eww` second (macOS, where the selftest runs). The ps path strips
    the command prefix before scanning, because `ps eww` prints command then environment with no
    delimiter and run_ddp.sh's wrapper argv CONTAINS `CUDA_VISIBLE_DEVICES=3,4,5,6`. MEASURED on
    a child given NO such variable but the literal string as argv[1]: the naive scan returned
    '7' out of the command, the strip returned absent. Same trap that made every substring test
    bind to the wrapper rather than the job (_argv0_is_shell).

    A parse failure returns "" rather than a value, which is the safe direction: absent never
    refuses, so an unreadable process gets no opinion instead of a wrong one. The cost is that
    this cannot tell absent from a deliberate CUDA_VISIBLE_DEVICES="" (harness.py's CPU-only
    shape) -- a claim on cards the job cannot see. status()'s idle-card note catches that one's
    consequence.
    """
    try:
        with open(f"/proc/{pid}/environ", "rb") as fh:
            raw = fh.read().decode("utf-8", "replace")
        for item in raw.split("\0"):
            if item.startswith("CUDA_VISIBLE_DEVICES="):
                return item.split("=", 1)[1]
        return ""
    except OSError:
        pass
    try:
        cmd = subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                             capture_output=True, text=True).stdout.strip()
        full = subprocess.run(["ps", "eww", "-p", str(pid)], capture_output=True, text=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if not cmd or not full.strip():
        return None
    i = full.find(cmd)
    env = full[i + len(cmd):] if i >= 0 else ""
    for tok in env.split():
        if tok.startswith("CUDA_VISIBLE_DEVICES="):
            return tok.split("=", 1)[1]
    return ""


def _cvd_mismatch(pid, cards):
    """(value, why) when pid's exec-time CUDA_VISIBLE_DEVICES DISAGREES with `cards`, else None.

    Compared as sets of physical indices, which is what both sides hold: a claim's `cards` are
    the controller's indices and the variable is a list of the same. The claimed pid is the
    torchrun/python descendant harness names (harness.py:12931), and in torchrun's DEFAULT mode
    ranks INHERIT the parent's value -- read 2026-09-04 in
    torch/distributed/elastic/agent/server/local_elastic_agent.py:468, "In traditional mode,
    don't override CUDA_VISIBLE_DEVICES", with virtual_local_rank defaulting to False
    (api.py:105) and run_ddp.sh not passing it. So the descendant's value IS the block's.

    THE BOUNDARY, since it would make this refuse a correct launch: under
    `torchrun --virtual_local_rank` each worker gets a SINGLE card
    (local_elastic_agent.py:460, `worker_env["CUDA_VISIBLE_DEVICES"] = visible_gpu`), so a
    per-rank pid would read one index against a seven-card claim and this would call it a
    mismatch. Nothing in this repo passes that flag today. If a launcher starts to, the fix is
    subset-not-equality for a rank pid, not deleting the check.

    Unreadable and absent both return None. See _cvd: absent is the normal reading for a job
    that sets the variable itself, and refusing there would refuse four correct scripts."""
    val = _cvd(pid)
    if not val:
        return None
    got = {t.strip() for t in val.split(",") if t.strip()}
    want = {str(c).strip() for c in cards}
    if got == want:
        return None
    return val, f"visible {sorted(got) or ['none']} vs claimed {sorted(want)}"


def nvidia_fds(pid):
    """How many nvidia device fds pid holds, or None when that cannot be read.

    None IS NOT ZERO and never refuses, the same rule _cvd follows for an absent
    CUDA_VISIBLE_DEVICES. /proc does not exist on macOS, where the selftest runs, so a predicate
    that refused on an unreadable pid would refuse every claim on a laptop for a reason that has
    nothing to do with cards.

    Why fds and not nvidia-smi: `nvidia-smi --query-compute-apps` inside the pod's container
    reports HOST-namespace pids -- 3796003/3796004 while the container's own ranks were
    842276/842277 (measured 2026-09-04). Matching a claim against that list would refuse every
    correct claim on the pod. /proc/<pid>/fd is namespace-local and agrees with the process it
    describes.

    Measured on the pod, cards 3/4, the harness wrapper shape: `import torch` alone opens 0,
    torch.cuda.set_device opens 34 within 1.3s and 48 shortly after, torchrun itself holds 36,
    a rank holds 52. So >0 means the process has actually touched a card, and 0 on a live
    non-shell process is the b0_mem_m1 defect: at that instant the pid exists, is not a shell,
    and holds no device.

    PROC_ROOT is the seam the selftest moves, like FOREIGN_ROOT -- macOS cannot produce a real
    /proc, so the worlds build one out of real directories and real symlinks."""
    try:
        entries = os.listdir(f"{PROC_ROOT}/{pid}/fd")
    except OSError:
        return None
    n = 0
    for e in entries:
        try:
            if "nvidia" in os.readlink(f"{PROC_ROOT}/{pid}/fd/{e}"):
                n += 1
        except OSError:
            continue  # the fd closed between listdir and readlink
    return n


def wait_for_device(pid, deadline=DEVICE_WAIT_S, interval=0.25):
    """The descendant of pid that holds a GPU device, waiting up to `deadline`. (pid, fds) or None.

    THE WAIT AND THE CHECK ARE ONE CONDITION, which is the whole point. Polling until a
    descendant is "not a shell" returns at t=0.11s on the pod, 1.22s before any device fd exists,
    so it answers a question nobody asked. Polling until a descendant HOLDS a device closes both
    halves: the pid is the job, and it is provably on a card at claim time.

    NO DEADLINE WHILE THE WRAPPER IS ALIVE (4c's ruling, 2026-09-05, after this gave up on a
    live M1 launch). The poll ends when a device appears or when the job ends -- never on a clock.
    A constant cannot be right here: run_ddp.sh runs the drift gate, asserts the mix, and
    build_mix loads token caches (155 GiB on that launch) before the ranks are even spawned, and
    the run's own startup gate was 793 s. A launcher that gives up before the job it launched has
    started is b0_mem_m1's hole in a new form -- the card ends up held with no claim, which is the
    state the whole mechanism exists to prevent. `deadline` is kept for the selftest's bounded
    worlds and for a caller that wants a ceiling; None means "while the wrapper lives".

    A VANISHED PID READS None, THE SAME VALUE AS "no /proc", and conflating them is what actually
    broke that launch -- not the deadline. The first version returned None on the first unreadable
    descendant, and run_ddp.sh spawns short-lived python children (scripts/pod_drift.py --check at
    run_ddp.sh:34) whose /proc entry is gone by the next read. So the poll aborted about a second
    in and printed "no descendant opened a GPU device within 90s", a message naming a timeout that
    had not happened: measured on that run, the ranks held device fds from 22:35:01, two seconds
    after launch, while the claim was refused at 22:35:00-ish and b0 claimed by hand at 22:36:01.
    The distinction is made ONCE, against the wrapper itself, before the loop: if the wrapper's
    own /proc is unreadable this host cannot answer (macOS), and a descendant that goes unreadable
    mid-poll is simply gone and is skipped.

    Returns None when /proc is unreadable, so a caller on macOS falls back rather than polling for
    an answer that cannot arrive. Callers must distinguish that from "nothing opened a device":
    ask nvidia_fds(os.getpid()) is None."""
    if nvidia_fds(os.getpid()) is None:
        return None  # no /proc anywhere here: this predicate has no opinion, decided once
    end = None if deadline is None else time.time() + deadline
    while True:
        for p, _a in _job_descendants(pid):
            n = nvidia_fds(p)
            if n:
                return p, n
            # n == 0: alive, not on a card yet. n is None: the pid vanished between the ps
            # snapshot and this read -- skip it, never end the wait on another process's exit.
        if not _alive(pid) or (end is not None and time.time() >= end):
            return None
        time.sleep(interval)


def _alive(pid):
    """Whether pid exists IN THIS NAMESPACE. signal 0, not /proc: this also has to work on
    a Mac, where the selftest runs.

    True for a ZOMBIE: see _is_zombie. Callers that mean "is the job running" must ask both."""
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno == errno.EPERM  # exists, not ours to signal
    return True


SHELLS = {"bash", "sh", "zsh", "dash", "ksh", "-bash", "-sh", "-zsh"}
# The pod's tree is wrapper bash -> torchrun -> 4 ranks. 6 is that plus slack, and it is a cap
# rather than a guess: a cycle in a ppid table would otherwise loop forever (CLAUDE.md's
# iteration-cap rule).
MAX_DEPTH = 6


def _ps_table():
    """[(pid, ppid, args)] for every process THIS namespace can see, or [] if ps cannot run.

    `ps -eo pid,ppid,args` is the one reader that works on both sides: /proc does not exist on
    macOS (where the selftest runs) and the pod's container has both. Called once per question
    rather than per pid -- a 900-row table is one fork, and pgrep -P per level is not.
    """
    try:
        r = subprocess.run(["ps", "-eo", "pid,ppid,args"], capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for line in r.stdout.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 2:
            continue
        try:
            out.append((int(parts[0]), int(parts[1]), parts[2] if len(parts) > 2 else ""))
        except ValueError:
            continue
    return out


def _argv0_is_shell(args):
    """Is this cmdline's argv[0] a shell?

    NOT "does the cmdline mention python". The pod's wrapper is
        bash -lc cd /work/aupai && setsid env CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun ... train.py
    so it CONTAINS `torchrun`, `python`-adjacent words and the master_port -- measured locally
    too: the wrapper's args gave contains_python=True while argv0 was bash. Every substring test
    matches the wrapper first, which is exactly how both of tonight's claims bound to a shell.
    """
    parts = (args or "").split()
    if not parts:
        return False
    return os.path.basename(parts[0]) in SHELLS


def _descendants(pid, table=None, max_depth=MAX_DEPTH):
    """[(pid, args)] beneath pid, by the real ppid chain, breadth-first, depth-capped."""
    table = _ps_table() if table is None else table
    kids = {}
    for p, ppid, args in table:
        kids.setdefault(ppid, []).append((p, args))
    out, frontier, seen = [], [pid], {pid}
    for _ in range(max_depth):
        nxt = []
        for parent in frontier:
            for child, args in kids.get(parent, []):
                if child in seen:
                    continue
                seen.add(child)
                out.append((child, args))
                nxt.append(child)
        if not nxt:
            break
        frontier = nxt
    return out


def _job_descendants(pid, table=None):
    """The descendants that are the JOB: a python or torchrun process, not another shell.

    CASE-INSENSITIVE, and matched on argv0's basename as well as the whole cmdline. macOS reports
    the interpreter as
        /Library/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python
    with a capital P, so `"python" in args` was False and the selftest's world reported zero
    descendants -- which read as "the negative case passes" for the ORPHAN-SHELL check while
    actually meaning the world had no job in it at all. The positive case still passed, so only
    the world assertions caught it (de, 2026-09-03).
    """
    out = []
    for p, a in _descendants(pid, table):
        if _argv0_is_shell(a):
            continue
        low = a.lower()
        parts = a.split()
        base = os.path.basename(parts[0]).lower() if parts else ""
        if "python" in low or "torchrun" in low or base.startswith(("python", "torchrun")):
            out.append((p, a))
    return out


def _read(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def claims():
    """Every live claim, plus the stale ones and why they are stale.

    A claim whose process is gone is reclaimable -- a crash must not lock the cards forever,
    which is the failure mode that makes people delete lock files by hand and then race.

    A ZOMBIE IS DELIBERATELY NOT STALE HERE, and the reason is `acquire`: it deletes every
    file this function calls stale, inside its wait loop (:425-428). So filing a zombie as
    stale would delete a claim and hand its cards to the next caller, and `_is_zombie` cannot
    distinguish "the job ended" from a corpse observed while a job is still coming up. The
    disagreement is REPORTED instead, by `status()` at :575, which is 6e's ruling of
    2026-09-03 with e1 as reviewer: a human releases, and `acquire` keeps refusing until they
    do. `_is_zombie`'s docstring says a caller meaning "is the job running" must ask both --
    that caller is `status()`, not this one, because this one's answer is also a deletion
    (de-51; corrects DL-9 in runs/audit_0904/instruments_ledgers.md)."""
    live, stale = [], []
    if not os.path.isdir(CLAIM_DIR):
        return live, stale
    for nm in sorted(os.listdir(CLAIM_DIR)):
        if not nm.endswith(".json"):
            continue
        c = _read(os.path.join(CLAIM_DIR, nm))
        if c is None:
            stale.append({"file": nm, "why": "unreadable or truncated"})
            continue
        if not _alive(int(c.get("pid", -1))):
            stale.append(dict(c, why=f"pid {c.get('pid')} is gone", file=nm))
        else:
            # `file` ON EVERY ROW, live as well as stale. acquire excludes its OWN claim from the
            # clash check by filename now, not by name, so a live row without this field would be
            # excluded from nothing and a re-acquire of the same name and cards would clash with
            # itself instead of reaching the rebind path. The stale rows already carried it,
            # because acquire deletes those by filename.
            live.append(dict(c, file=nm))
    return live, stale


FOREIGN_MARKERS = ("NEVER TAKE", "FOREIGN OCCUPANT", "not an aupai job", "another container")
# Where runs/card_assignment.json is read from. Separate from CLAIM_DIR because the two move
# independently: a claim dir belongs to the tree the JOB runs in, the grant file to the repo.
FOREIGN_ROOT = ROOT


def foreign_cards(root=ROOT):
    """{card: why} for cards runs/card_assignment.json says are not ours to take.

    ORPHAN means "memory held by nobody, find it and reclaim it". For a card holding the USER'S
    own job or another container's, that instruction is wrong and acting on it is destructive --
    e1 read "ORPHAN card 7 27,809 MiB" for the user's job on 2026-09-04, and it was the third
    reader of that line (6e). The claim files cannot say this: a claim records OUR intent, and
    another container writes none, so the absence of a claim is exactly what a foreign job looks
    like. The controller's grant file is the only place the distinction is written down.

    Read from the `cards` values' own text rather than a new field, because the text is already
    there and already maintained -- a parallel schema would be a second thing to keep in step
    with the prose, and the prose is what the controller actually edits.
    """
    p = os.path.join(root, "runs", "card_assignment.json")
    try:
        with open(p, encoding="utf-8") as fh:
            obj = json.load(fh)
    except (OSError, ValueError):
        return {}
    out = {}
    for card, why in (obj.get("cards") or {}).items():
        if not isinstance(why, str):
            continue
        hit = next((m for m in FOREIGN_MARKERS if m.lower() in why.lower()), None)
        if not hit:
            continue
        # Keys are single indices ("5") or RANGES ("0-3"): the file uses both, so a plain
        # string compare against an nvidia-smi index silently misses every ranged entry.
        #
        # AND THEY OVERLAP. As of 2026-09-04 the file carries both "0-3" and "0".."3", so one
        # card can be named twice. Overlap resolves FOREIGN-WINS by construction, not by
        # ordering: a key with no marker hits the `continue` above and is never inserted, so a
        # non-foreign entry cannot clear a foreign one whichever order the dict yields them.
        # That is the safe direction -- the failure this function prevents is telling someone
        # to reclaim a card that is not ours, so a false FOREIGN costs a question and a false
        # ORPHAN costs the user's job.
        key = str(card).strip()
        if "-" in key:
            try:
                lo, hi = (int(x) for x in key.split("-", 1))
                for i in range(lo, hi + 1):
                    out[str(i)] = (hit, why)
                continue
            except ValueError:
                pass
        out[key] = (hit, why)
    return out


def grant_lane(card, to, why, by, root=None):
    """Write a lane grant as ONE edit: lane_card, lane_to, cards[card], lane_note, granted_by.

    Five fields state one fact, and hand-editing them is how they diverge: on 2026-09-04
    cards["5"] said the lane was e1's for C11 while lane_note said the same thing again, and
    nothing would have caught either one going stale alone. This is the only writer, so a grant
    cannot be half-written -- and `launch_gate.gate_cards` reads exactly these fields to confirm
    a lane launch, so a grant it cannot read is a grant that refuses.

    `lane_to` is a FIELD and not a phrase in the prose, because the prose cannot answer "whose
    lane is this": measured on the live file, a substring search for the session name `de` hits
    inside "excluded", and adding word boundaries then hits `b0` inside
    "ckpt_b0_sd_equalcompute" and `fb` in the granter's own name. Two-character names are not
    recoverable from free text.

    REFUSES a card marked foreign, since granting our lane onto another container's job is the
    destructive direction. Appends to granted_by rather than replacing it: the list is the
    grant's history and the controller reads it to see who moved the lane last.
    """
    root = root or FOREIGN_ROOT
    p = os.path.join(root, "runs", "card_assignment.json")
    try:
        with open(p, encoding="utf-8") as fh:
            obj = json.load(fh)
    except (OSError, ValueError) as e:
        return False, f"{p}: {e}"
    card = str(card).strip()
    if not card.isdigit():
        return False, f"card {card!r} is not a card index"
    foreign = foreign_cards(root).get(card)
    if foreign:
        return False, (f"refusing: card {card} is marked {foreign[0]!r} -- {foreign[1][:80]}. "
                       f"The lane cannot be granted onto another container's job")
    stamp = _now()
    obj["lane_card"] = card
    obj["lane_to"] = to
    obj.setdefault("cards", {})[card] = f"lane: {to} {why} (granted {stamp} {by})"
    obj["lane_note"] = f"{stamp} {by}: lane card {card} granted to {to} -- {why}"
    hist = obj.get("granted_by")
    obj["granted_by"] = (hist if isinstance(hist, list) else [hist] if hist else []) + [
        f"{by} {stamp} (lane {card} -> {to})"]
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    os.replace(tmp, p)
    return True, (f"lane card {card} -> {to} ({why}); lane_card, lane_to, cards[{card}], "
                  f"lane_note, granted_by written")


def card_memory():
    """{index: MiB} from nvidia-smi, or None when there is no nvidia-smi.

    None is not {}: "no GPUs here" and "GPUs, all idle" are different facts, and returning an
    empty dict for the first would make a dev box look like an idle pod."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    mem = {}
    for line in out.stdout.splitlines():
        if "," in line:
            i, m = line.split(",", 1)
            try:
                mem[i.strip()] = int(m.strip())
            except ValueError:
                continue
    return mem


def held_cards(live):
    """{card: claim_name} over live claims. A card in two live claims cannot happen if acquire
    is the only writer, so it is reported rather than resolved."""
    held = {}
    for c in live:
        for card in c.get("cards", []):
            held.setdefault(card, []).append(c.get("name"))
    return held


def _card_key(c):
    """Sort cards numerically where they are numbers, so a set has one filename.

    Plain string sort puts "10" before "2", which would give one set of cards two possible
    filenames depending on the order the caller passed them -- and the filename is the lock."""
    s = str(c)
    return (0, int(s), "") if s.isdigit() else (1, 0, s)


def claim_file(name, cards):
    """The claim's filename: keyed on the name AND the cards, because the CARDS are the lock.

    The bug this fixes, found by b0 in production 2026-09-04: claim_my_cards("score_matrix")
    wrote runs/claims/score_matrix.json keyed on the name alone, so armB's doc_cu pass on card 2
    was refused while armA's pass held card 4. Disjoint cards, no possible contention, mutual
    exclusion by filename. Two jobs that cannot interfere were serialised, and the message said
    "already holds a live claim" -- true of the file and false of the hardware.

    Cards, not pid, and not a counter. The pid is already in the claim's content and changes on
    every relaunch, so a pid-keyed file cannot be found again by a human running `release`; a
    counter makes the name meaningless. The cards are what acquire is protecting, they are what
    the operator knows, and they make the filename state the same fact the lock enforces.

    An OLD name-only file still reads fine -- claims() globs *.json and never parses the name --
    so claims live on the pod at the moment of this change keep working and are released by the
    `cards=None` path in release(). Nothing needs migrating.
    """
    tag = "-".join(str(c) for c in sorted(cards, key=_card_key))
    return f"{name}.{tag}.json" if tag else f"{name}.json"


def _resolve_to_device_holder(holder, deadline):
    """(pid, None) for the ONE descendant of `holder` on a card, or (None, why).

    Wired to the CLI as `acquire --wait-for-device [SECONDS]`. wait_for_device has existed and been
    selftested since 2026-09-05 and was reachable only from _selftest and harness.py, so every
    hand-run acquire had to reinvent the poll -- and the refusal it hit already printed the pid to
    claim while offering no way to act on it. e1 hit that 60 times in one day.

    ONE DESCENDANT OR NONE. Several holders means the operator has to choose: e1's script had two
    candidates, the probe and the card_claim call itself, and a launcher that picks for them is
    wrong half the time and silent about it. Listing them is the answer.

    THE TIMEOUT MESSAGE IS A DIFFERENT SENTENCE from the immediate refusal, deliberately. Both say
    "this pid is not the job", and if they read alike nobody can tell whether they waited. 60 slow
    refusals are worse than 60 fast ones.
    """
    on_card = [(p, a, nvidia_fds(p)) for p, a in _job_descendants(holder)]
    on_card = [(p, a, n) for p, a, n in on_card if n]
    if len(on_card) > 1:
        return None, (
            f"pid {holder} has {len(on_card)} descendants holding a device, so --wait-for-device "
            f"cannot choose one for you:\n" + "\n".join(
                f"    --pid {p}  ({n} device fds)  {a[:82]}" for p, a, n in on_card[:6])
            + "\nName the one that IS the job with --pid.")
    if len(on_card) == 1:
        return on_card[0][0], None

    got = wait_for_device(holder, deadline=deadline)
    if got is None:
        if nvidia_fds(os.getpid()) is None:
            return None, (
                f"pid {holder} is not the job and this host cannot follow it: /proc is unreadable "
                f"here (macOS), so --wait-for-device has no opinion. Pass --pid explicitly.")
        if not _alive(holder):
            return None, (f"pid {holder} exited while --wait-for-device was polling, so it never "
                          f"had a descendant on a card. Nothing was claimed.")
        return None, (
            f"no descendant of pid {holder} opened a GPU device in {deadline}s -- it is still "
            f"alive, so this is a TIMEOUT and not a dead job. If it never launches one (an "
            f"interactive shell), waiting longer will not help and this is the wrong pid to "
            f"claim; if it is a launcher still in startup, raise the seconds.")
    return got[0], None


def acquire(name, cards, wait=0, note="", pid=None, require_device=False, wait_for_device_s=None):
    """Claim `cards` for `name`, waiting up to `wait` seconds. Returns (ok, message).

    Refuses rather than sharing: this is a lock, not a message. The previous arrangement was a
    message -- two sessions each announced a launch and both proceeded.

    wait_for_device_s: when the named holder is a shell, or holds no device fd, follow it to the
    python/torchrun descendant that IS on a card and claim that instead, polling up to this many
    seconds. None (the default) keeps the refusals exactly as they were.

    THE FLAG IS OFF BY DEFAULT AND THAT IS NOT TIMIDITY. scripts/loader.py's claim_my_cards path
    claims ITSELF before opening the device it named through CVD, so it holds zero device fds by
    construction and has no descendant to follow; a poll there would wait out its deadline and
    then refuse a correct claim. The flag belongs to the hand-run and launcher shapes, where a
    wrapper shell has spawned, or is about to spawn, the job.

    IT RESOLVES ONE DESCENDANT, NEVER TWO. If several descendants hold a device, they are listed
    and the claim refuses: e1's own script had two candidates (the probe and the card_claim call
    itself), and picking for the operator would be picking wrong half the time.

    require_device: refuse a holder that holds no GPU device fd. DEFAULT FALSE, and the default
    is the design. There are two claim shapes and only one of them can honour that predicate:

      harness launch    claims ANOTHER process, AFTER wait_for_device proved it is on a card
      claim_my_cards    claims ITSELF, BEFORE it opens the device it just named through CVD

    The second is scripts/loader.py's acquire point for every GPU entry point that is not
    `harness launch`, and it claims the card it is about to use precisely so nothing else takes
    it in between -- so at claim time it holds zero device fds by construction. Refusing there
    unconditionally broke `python scripts/loader.py selftest` on every Linux host (CI red on main
    at 121a865d for two hours; reproduced on the pod: "demo_set: pid 902861 holds no GPU device
    fd") while passing on macOS, where no /proc means the predicate has no opinion.

    A process claiming ITSELF by CVD cannot claim the wrong pid -- CVD is the only thing it could
    name. The refusal was written for a launcher that bound the wrong pid (b0_mem_m1: the claimed
    842204 held 0 fds while rank 842276 held 52), so it belongs to the launcher path, where
    wait_for_device has already established the fact it asserts."""

    os.makedirs(CLAIM_DIR, exist_ok=True)
    mine = os.path.join(CLAIM_DIR, claim_file(name, cards))
    deadline = time.time() + wait
    while True:
        live, stale = claims()
        for s in stale:
            f = os.path.join(CLAIM_DIR, s.get("file") or f"{s.get('name')}.json")
            if os.path.exists(f):
                os.unlink(f)
        # EXCLUDE THIS CLAIM'S OWN FILE, not every claim sharing its name. The old form was
        # `c.get("name") != name`, and it is the second half of b0's defect: with the name no
        # longer the lock, a same-name claim on DIFFERENT cards is a different lock and its
        # cards must still clash. `score_matrix` live on card 2, then `score_matrix` asking for
        # 2,3 overlaps itself -- the name filter excluded it, O_EXCL saw a new path, and two
        # live claims would both have held card 2. Keyed on the file, a re-acquire of the same
        # name AND cards still skips the clash check and falls through to the O_EXCL/rebind
        # path below, which is where that case belongs.
        held = held_cards([c for c in live if c.get("file") != os.path.basename(mine)])
        clash = {c: held[c] for c in cards if c in held}
        if not clash:
            # WHOSE pid. os.getpid() is THIS process -- the card_claim.py invocation, which
            # exits the moment it has written the file. Every claim was therefore stale on
            # arrival: the next acquire read a dead pid, deleted the file as a crashed claim,
            # and took the card. Measured 2026-09-03: two acquires for the same card both
            # succeeded, and `release --name armA` then reported "no claim" because armB's
            # file had replaced it. That is why `card_claim.py status` reported all eight pod
            # cards as ORPHAN -- not because nobody claimed, but because no claim could
            # survive its own exit.
            #
            # The pid that matters is the JOB's. --pid names it; PPID is the default because
            # the normal caller is a launcher script that outlives this command and dies with
            # the job.
            holder = pid if pid else os.getppid()
            # REFUSE A SHELL. Two claims tonight bound to one, in opposite directions:
            # shapelr bound the launching shell, which exited and left the card ORPHAN; the VE
            # arm bound the pod's wrapper bash (2878732, while torchrun was 2878734), which
            # OUTLIVED nothing but kept the claim alive after training ended, so the card looked
            # held and a rebind was refused with "already holds a live claim". One root cause:
            # `pgrep -f <port>` matches the calling shell first, because the wrapper's own argv
            # contains the port and the word torchrun.
            #
            # The claim must name the process that dies WITH the job. A shell is either about to
            # exec away or about to outlive it, and neither is that process.
            if _argv0_is_shell(_cmdline(holder)):
                # FOLLOW THE SHELL TO ITS JOB when asked. The refusal below already prints the
                # descendant to claim, and e1 hit it 60 times in one day: every refusal named the
                # right pid and there was no way to act on it except retyping it. That is the whole
                # gap -- not the detection.
                if wait_for_device_s is not None:
                    resolved, why = _resolve_to_device_holder(holder, wait_for_device_s)
                    if resolved is None:
                        return False, why
                    holder = resolved
                else:
                    kids = _job_descendants(holder)
                    msg = (f"pid {holder} is a shell, not the job: "
                           f"{_cmdline(holder)[:90]!r}\n"
                           f"A claim on a shell fails both ways -- the shell exits and the card "
                           f"reads ORPHAN, or it lingers and the card reads held after the job is "
                           f"gone (both happened 2026-09-03).")
                    if kids:
                        msg += ("\nIts python/torchrun descendants -- claim one of these, or pass "
                                "--wait-for-device to follow it automatically:\n" + "\n".join(
                                    f"    --pid {p}  {a[:96]}" for p, a in kids[:6]))
                    else:
                        msg += ("\nIt has no python/torchrun descendant yet. Claim after the job "
                                "starts, pass --pid explicitly, or pass --wait-for-device to poll "
                                "until one is on a card.")
                    return False, msg
            claim = {"name": name, "cards": list(cards), "pid": holder,
                     "cmdline": _cmdline(holder), "acquired": _now(), "note": note}
            # REFUSE A DISAGREEING CUDA_VISIBLE_DEVICES. The claim says which cards this job may
            # touch and the variable says which it CAN touch; when they differ the claim protects
            # the wrong cards, and the job runs on cards nobody claimed -- which is how an ORPHAN
            # is produced by a healthy-looking claim. AGENTS.md, harness.py's device_set_honoured:
            # the variable is not additive, a child REPLACES its parent's set.
            #
            # Only a DISAGREEMENT refuses. Absent does not, and that is a correction to the
            # original ruling rather than an omission -- /proc/<pid>/environ and `ps eww` are both
            # exec-time readers, so the four scripts that set the variable themselves
            # (scripts/test_e2e.py:37, bench_eff/*'s setdefault) read absent while being right.
            # Measured 2026-09-04; see _cvd.
            bad_cvd = _cvd_mismatch(holder, cards)
            if bad_cvd:
                val, why = bad_cvd
                return False, (
                    f"pid {holder}'s CUDA_VISIBLE_DEVICES={val!r} does not match the cards being "
                    f"claimed: {why}. The claim would protect cards the job cannot touch while it "
                    f"runs on cards nobody claimed -- an orphan behind a healthy claim. Claim the "
                    f"cards the job can see, or relaunch with the variable set to "
                    f"{','.join(str(c) for c in cards)}. (Read at exec time: a value set inside "
                    f"the process after start is invisible here and never refuses.)")
            # REFUSE A PID THAT HOLDS NO CARD. The shell refusal above is necessary and not
            # sufficient: measured on the pod in the harness wrapper shape, the first non-shell
            # descendant appears at t=0.11s holding ZERO device fds and the first fd appears at
            # t=1.33s -- a 1.22s window in which a pid passes every test above and is not yet on
            # a card. b0_mem_m1's claim landed in exactly that window: the claimed 842204 held 0
            # fds while 842220/842240 held 36 and rank 842276 held 52, so `status` reported STALE
            # plus "ORPHAN card 1 holds 1179 MiB with no claim" for a correctly running arm, and
            # any sweep could have reclaimed two cards from under it.
            #
            # None (no /proc, e.g. macOS) proceeds -- see nvidia_fds.
            devs = nvidia_fds(holder)
            if devs == 0 and (require_device or wait_for_device_s is not None):
                if wait_for_device_s is not None:
                    resolved, why = _resolve_to_device_holder(holder, wait_for_device_s)
                    if resolved is None:
                        return False, why
                    holder = resolved
                    devs = nvidia_fds(holder)
                    # claim was built above with the OLD holder, so both fields are rewritten.
                    # Missing this leaves a claim naming the wrapper while device_fds names the
                    # rank -- a claim that looks verified and protects the wrong process.
                    claim["pid"] = holder
                    claim["cmdline"] = _cmdline(holder)
                else:
                    holders = [(p, a, nvidia_fds(p)) for p, a in _job_descendants(holder)]
                    holders = [(p, a, n) for p, a, n in holders if n]
                    msg = (f"pid {holder} holds no GPU device fd: {_cmdline(holder)[:90]!r}\n"
                           f"A claim on a process that is not on a card protects nothing: the card "
                           f"reads ORPHAN while the job runs (b0_mem_m1, 2026-09-04).")
                    if holders:
                        msg += ("\nIts descendants that DO hold a device -- claim one of these, or "
                                "pass --wait-for-device:\n" + "\n".join(
                                    f"    --pid {p}  ({n} device fds)  {a[:82]}"
                                    for p, a, n in holders[:6]))
                    else:
                        msg += ("\nNo descendant holds one yet either. Wait for the job to reach "
                                "torch.cuda.set_device (~1.3s after exec, measured) and claim "
                                "then, or pass --wait-for-device to poll.")
                    return False, msg
            if devs:
                claim["device_fds"] = devs
            # O_EXCL: two acquirers racing on the same name must not both believe they won.
            try:
                fd = os.open(mine, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                existing = _read(mine)
                old = int(existing.get("pid", -1)) if existing else -1
                if existing and _alive(old):
                    # REBIND, not refuse, when the new pid is a DESCENDANT of the recorded one.
                    # The VE arm hit this: the claim held the wrapper bash and the rebind wanted
                    # torchrun, its own child, so "already holds a live claim" refused the
                    # correction of the very defect above. A descendant is the same job by
                    # construction -- the recorded process is its ancestor -- so this cannot
                    # hand a card to an unrelated claimant.
                    if holder != old and holder in {p for p, _ in _descendants(old)}:
                        os.unlink(mine)
                        continue
                    # SAY IT IS A ZOMBIE WHEN IT IS ONE. e1 hit this in production 2026-09-04:
                    # arm 1's pid went Zs, acquire for arm 2 refused with "already holds a live
                    # claim", and the reader has no way from that message to know the process
                    # has exited -- the advice "release first" is right but reads as "wait for
                    # the running job". _is_zombie had one call site, in status(), so the
                    # classification existed and the refusal path could not see it (6e).
                    # Same no-auto-break policy: a human releases, acquire does not.
                    if _is_zombie(old):
                        return False, (
                            f"{name}'s claim pid {old} is a ZOMBIE (exited, not reaped; "
                            f"state {_proc_stat(old)!r}) -- the job is over and the claim is "
                            f"not. Nothing is auto-broken: `card_claim.py release --name "
                            f"{name}`, then acquire again. os.kill(pid,0) and /proc both "
                            f"report a zombie as alive; `ps -o stat=` is the only reader "
                            f"that does not.")
                    return False, (f"{name} already holds a live claim (pid {existing['pid']}"
                                   f"{', a shell' if _argv0_is_shell(existing.get('cmdline', '')) else ''})"
                                   f". If pid {holder} is the real job and not a descendant of "
                                   f"{old}, release first.")
                os.unlink(mine)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(claim, fh, ensure_ascii=False)
            return True, f"claimed {','.join(cards)} for {name}"
        if time.time() >= deadline:
            # NAME A ZOMBIE HOLDER HERE TOO. 6e's ruling of 2026-09-04 -- acquire's own message
            # must say the holder is a corpse, because "Queue" reads as "wait for the running
            # job" and e1 waited ten minutes on a `Zs` pid -- was implemented only on the O_EXCL
            # path below, which is reached when the SAME name and cards are re-acquired. Every
            # other case arrives here: a different name wanting the card (armB after armA died),
            # and now also the same name on a different card set, since the claim is keyed on
            # both. Measured: without this, a real zombie holding card 4 refused armB with
            # "claimed by {'4': ['armA']}. Queue" and nothing in the message said the job was
            # over.
            corpses = []
            for c in live:
                p = c.get("pid")
                if (isinstance(p, int) and _is_zombie(p)
                        and any(card in clash for card in c.get("cards", []))):
                    corpses.append((c.get("name"), p, _proc_stat(p)))
            if corpses:
                nm, zp, st = corpses[0]
                return False, (
                    f"cards {sorted(clash)} are claimed by {clash}, and {nm}'s pid {zp} is a "
                    f"ZOMBIE (exited, not reaped; state {st!r}) -- that job is over and its claim "
                    f"is not, so this is not a queue to wait in. Nothing is auto-broken: "
                    f"`card_claim.py release --name {nm}`, then acquire again. os.kill(pid,0) and "
                    f"/proc both report a zombie as alive; `ps -o stat=` is the only reader that "
                    f"does not.")
            return False, (f"cards {sorted(clash)} are claimed by {clash}. "
                           f"Queue -- do not spill onto a claimed card.")
        time.sleep(min(15, max(2, wait / 20)))


def release(name, cards=None):
    """Release `name`'s claim. With `cards`, exactly that claim; without, every claim under the name.

    Two forms because there are two callers and they know different things. loader's atexit hook
    holds the card list it acquired, so it releases precisely its own -- one arm's exit must not
    free the other arm's card, which is what a name-wide release would do now that one name can
    hold several disjoint claims. A human typing `release --name score_matrix` after a crash knows
    the name and not the cards, so the bare form stays and clears them all.

    It reports WHAT it released, because "released score_matrix" is no longer unambiguous.
    """
    if cards:
        p = os.path.join(CLAIM_DIR, claim_file(name, cards))
        if not os.path.exists(p):
            # Fall back to the pre-2026-09-04 name-only filename: a claim acquired by an older
            # copy of this file is still on the pod and must still be releasable by its owner.
            legacy = os.path.join(CLAIM_DIR, f"{name}.json")
            got = _read(legacy)
            if got and [str(c) for c in got.get("cards", [])] == [str(c) for c in cards]:
                os.unlink(legacy)
                return True, f"released {name} (legacy name-only claim) on {','.join(cards)}"
            return False, f"no claim for {name} on cards {','.join(str(c) for c in cards)}"
        os.unlink(p)
        return True, f"released {name} on {','.join(str(c) for c in cards)}"
    gone = []
    if os.path.isdir(CLAIM_DIR):
        for nm in sorted(os.listdir(CLAIM_DIR)):
            if not nm.endswith(".json"):
                continue
            # Match on the claim's OWN `name` field, never on the filename prefix: `score_matrix`
            # must not release `score_matrix_probe`, and a filename-prefix test cannot tell the
            # two apart once the cards are appended to the name.
            c = _read(os.path.join(CLAIM_DIR, nm))
            if not c or c.get("name") != name:
                continue
            os.unlink(os.path.join(CLAIM_DIR, nm))
            gone.append(",".join(str(x) for x in c.get("cards", [])) or nm)
    if not gone:
        return False, f"no claim for {name}"
    return True, f"released {name}: {len(gone)} claim(s) on {'; '.join(gone)}"


def status():
    """Claims, and the two disagreements with the cards. Exit code is 0 unless an orphan
    exists -- memory held by nobody is the one state that needs a person."""
    live, stale = claims()
    mem = card_memory()
    held = held_cards(live)
    lines = []
    # ORPHAN-SHELL: the claim's pid is alive, so nothing else here reports a problem, and it is a
    # SHELL with no python/torchrun descendant -- meaning the job it was claimed for has ended
    # while the claim reads healthy. This is the state the VE arm sat in: card looks held, no
    # training on it, and a rebind refused. Judged on argv0 plus the real ppid chain, never on
    # whether the cmdline mentions python -- the pod's wrapper bash mentions torchrun and the
    # master port, which is what made every substring test bind to it.
    table = _ps_table()
    # ZOMBIE: the third claim/card disagreement, and the only one whose cause is INVISIBLE to
    # every other line here. The claim reads healthy -- _alive() is True, the pid resolves, the
    # cmdline is right -- and the job ended minutes ago. MEASURED on the pod 2026-09-03: pid
    # 3891063 `Zs`, os.kill(pid,0) ALIVE, /proc entry present, its card at 3 MiB. e1 waited ten
    # minutes on it and its QUEUE_TIMEOUT would have reported "the holder is still running".
    #
    # NOT auto-broken, and not filed as stale. claims() deletes stale files inside acquire's
    # loop, so classifying a zombie as stale would break a live claim during a slow startup --
    # the disagreement this file explicitly says not to reconcile. A human releases; acquire
    # keeps refusing until they do. Reported per 6e's ruling 2026-09-03, reviewer e1.
    #
    # A zombie is the GUARANTEED outcome of the pod's launch shape, not an anomaly (b0): a
    # `setsid python3 ... &` whose wrapper exits immediately leaves nobody to wait() the child,
    # so every such job ends as an unreapable corpse. Hence a printed state, not an incident.
    zombies = []
    for c in live:
        p = c.get("pid")
        if isinstance(p, int) and _is_zombie(p):
            zombies.append((c.get("name"), p, _proc_stat(p), c.get("acquired"), c.get("cards", [])))
    for name, p, st, since, cards in zombies:
        lines.append(f"ZOMBIE {name} holds cards {','.join(cards)} on pid {p}, state {st!r} -- the "
                     f"process is a corpse, so the job ended; claimed since {since}. "
                     f"os.kill(pid,0) and /proc BOTH report it alive, `ps -o stat=` does not. "
                     f"A human releases: `card_claim.py release --name {name}`")
    # CVD-MISMATCH: the fourth disagreement, and the only one between a claim and the JOB'S OWN
    # VIEW of the cards rather than between a claim and the hardware. The claim names cards the
    # job cannot touch, so the job runs on cards nobody claimed -- an orphan sitting behind a
    # claim that every other line here reads as healthy.
    #
    # NO GRACE PERIOD, which corrects the original ruling's "after a grace period". The variable
    # is fixed at exec and both readers are exec-time readers (see _cvd), so this cannot be a
    # startup transient the way an idle card can -- a wait would only delay a verdict that is
    # already final. The idle-card note below stays as it is: it answers "is the job up yet",
    # a different question with a real transient.
    mismatches = []
    for c in live:
        p = c.get("pid")
        if not isinstance(p, int) or _is_zombie(p):
            continue
        got = _cvd_mismatch(p, c.get("cards", []))
        if got:
            mismatches.append((c.get("name"), p, got[0], got[1]))
    for name, p, val, why in mismatches:
        lines.append(f"CVD-MISMATCH {name} claims cards it cannot touch: pid {p} was started with "
                     f"CUDA_VISIBLE_DEVICES={val!r}, {why}. The job runs on cards nobody claimed "
                     f"while this claim protects cards it never touches. Fix the claim or the "
                     f"launch: `card_claim.py release --name {name}` and acquire the cards the "
                     f"job can see.")
    orphan_shells = []
    for c in live:
        p = c.get("pid")
        if not isinstance(p, int):
            continue
        args = c.get("cmdline") or _cmdline(p)
        if _argv0_is_shell(args) and not _job_descendants(p, table):
            orphan_shells.append((c.get("name"), p))
    for name, p in orphan_shells:
        lines.append(f"ORPHAN-SHELL {name} holds pid {p}, a shell with no live python/torchrun "
                     f"descendant -- the job ended and the claim did not. "
                     f"`card_claim.py release --name {name}`")
    for c in live:
        lines.append(f"CLAIM {c['name']:<20} cards {','.join(c.get('cards', [])):<16} "
                     f"pid {c.get('pid')} since {c.get('acquired')}")
    for s in stale:
        lines.append(f"STALE {s.get('name') or s.get('file')} -- {s.get('why')} (reclaimable)")
    dup = {c: n for c, n in held.items() if len(n) > 1}
    for c, names in dup.items():
        lines.append(f"CONFLICT card {c} claimed by {names} -- acquire should have refused")
    orphans = []
    # ROOT, not a path derived from CLAIM_DIR: the selftest swaps CLAIM_DIR to a temp directory,
    # and deriving the repo root from it would look for card_assignment.json under /tmp and find
    # nothing -- the foreign case would then silently never fire and the world would prove the
    # ORPHAN branch twice. FOREIGN_ROOT is the seam the selftest moves instead.
    foreign = foreign_cards(FOREIGN_ROOT)
    if mem is None:
        lines.append("CARDS  not measured (no nvidia-smi here) -- claims listed above are all "
                     "this can say")
    else:
        for card, m in sorted(mem.items(), key=lambda kv: kv[0]):
            if m > FREE_MIB and card not in held and card in foreign:
                # FOREIGN, not ORPHAN, and it does not go in `orphans`: ORPHAN's instruction is
                # "find it and reclaim it", which for the user's own job or another container's
                # is destructive advice. Exits 0 for this card -- nothing is wrong.
                hit, why = foreign[card]
                lines.append(f"FOREIGN card {card} holds {m} MiB, no claim, and "
                             f"runs/card_assignment.json says {hit!r} -- not ours to take or "
                             f"reclaim: {why[:110]}")
            elif m > FREE_MIB and card not in held:
                orphans.append((card, m))
                lines.append(f"ORPHAN card {card} holds {m} MiB with no claim -- find it with "
                             f"nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory")
            elif card in held and m <= FREE_MIB:
                lines.append(f"note   card {card} claimed by {held[card][0]} but idle "
                             f"({m} MiB) -- starting up, or the job died without releasing")
    if not lines:
        lines.append("no claims, no orphans")
    # An ORPHAN-SHELL goes in `dup`, not `orphans`: `orphans` is (card, MiB) and one caller
    # asserts its exact contents, while both callers exit nonzero on either. Printing a state
    # nobody exits on is the print-and-continue shape -- a card that looks held with no job on it
    # needs a person, same as memory held by nobody.
    if orphan_shells:
        dup = dict(dup)
        for name, p in orphan_shells:
            dup[f"orphan-shell:{name}"] = [f"pid {p}"]
    if zombies:
        dup = dict(dup)
        for name, p, st, _since, _cards in zombies:
            dup[f"zombie:{name}"] = [f"pid {p} {st}"]
    # Same reasoning as the two above: a printed state nobody exits on is print-and-continue. A
    # claim guarding the wrong cards needs a person as much as memory held by nobody does.
    if mismatches:
        dup = dict(dup)
        for name, p, val, _why in mismatches:
            dup[f"cvd-mismatch:{name}"] = [f"pid {p} CUDA_VISIBLE_DEVICES={val}"]
    return orphans, dup, lines


def _selftest():
    """Known answers over a temporary claim dir. No cards."""
    import shutil
    import tempfile
    global CLAIM_DIR
    bad = 0
    # Counted as the cases run, not hardcoded. `n = 10` at the bottom stayed 10 when two cases
    # were added, so the line read "10/10 pass" while running twelve -- a total that cannot
    # notice a case going missing is not a total.
    n = 0

    def _case(good, text):
        nonlocal bad, n
        n += 1
        bad += 0 if good else 1
        print(f"  {'ok  ' if good else 'BUG '} {text}")

    d = tempfile.mkdtemp(prefix="claim_")
    CLAIM_DIR = d

    # pid=os.getpid() ON EVERY IN-PROCESS acquire BELOW, and it is not cosmetic. The default
    # holder is os.getppid(), so these cases bound to whatever invoked the selftest -- python
    # when run from a wrapper (22/22 green) and the SHELL when a human types
    # `python3 scripts/card_claim.py --selftest`, where acquire correctly refuses a shell and
    # the first four cases go red. Measured 2026-09-03 at 8393d579, before this commit's edits:
    # red from zsh, green from python, same code. A selftest whose answer depends on its caller
    # is testing the caller. This process is a python process by construction, so naming it
    # removes the dependency without weakening any case -- the shell-refusal behaviour has its
    # own dedicated world further down, built with a real shell on purpose.
    me = os.getpid()

    ok, msg = acquire("runA", ["0", "1"], pid=me)
    _case(ok, f"a free card set is claimable ({msg})")

    # The whole point: the second acquirer must NOT get overlapping cards.
    ok2, msg2 = acquire("runB", ["1", "2"], wait=0, pid=me)
    good = not ok2 and "claimed by" in msg2
    _case(good, f"an overlapping set is refused, not shared ({msg2})")

    # Disjoint sets coexist -- a lock that blocks unrelated work gets deleted by hand.
    ok3, msg3 = acquire("runC", ["5", "6"], wait=0, pid=me)
    _case(ok3, f"a disjoint set is granted ({msg3})")

    live, stale = claims()
    good = len(live) == 2 and not stale
    _case(good, f"two live claims, no stale ({[c['name'] for c in live]})")

    # THE CLI PATH, which every real caller uses and no case here covered. The in-process
    # acquire() calls above record THIS process's pid, which is alive for the whole selftest --
    # so they passed while the command line was broken: `card_claim.py acquire` recorded its
    # own pid, exited, and the claim was stale before the next command could read it. Run two
    # real subprocesses and assert the second is refused.
    import subprocess as _sp
    env = dict(os.environ, AUPAI_CLAIM_DIR=d)
    here = os.path.abspath(__file__)
    r1 = _sp.run([sys.executable, here, "acquire", "--name", "cliA", "--cards", "3",
                  "--wait", "0"], capture_output=True, text=True, env=env)
    r2 = _sp.run([sys.executable, here, "acquire", "--name", "cliB", "--cards", "3",
                  "--wait", "0"], capture_output=True, text=True, env=env)
    good = r1.returncode == 0 and r2.returncode != 0
    _case(good, f"a claim survives the claiming COMMAND's exit "
          f"(first rc={r1.returncode}, second rc={r2.returncode})")
    r3 = _sp.run([sys.executable, here, "release", "--name", "cliA"],
                 capture_output=True, text=True, env=env)
    good = r3.returncode == 0
    _case(good, f"and the original holder can release it "
          f"({r3.stdout.strip() or r3.stderr.strip()})")

    # SAME NAME, DISJOINT CARDS, BOTH GRANTED. b0's production defect, 2026-09-04: two
    # score_matrix passes -- armA on card 4, armB's doc_cu on card 2 -- and the second was
    # refused. Disjoint cards, no possible contention, mutual exclusion by filename, because the
    # claim was runs/claims/score_matrix.json keyed on the name alone. b0 ran them sequentially
    # for nothing.
    okd1, msgd1 = acquire("sameName", ["7"], wait=0, pid=me)
    okd2, msgd2 = acquire("sameName", ["8"], wait=0, pid=me)
    _case(okd1 and okd2, f"one name holds two DISJOINT claims ({msgd1} / {msgd2})")
    live_sn = [c for c in claims()[0] if c.get("name") == "sameName"]
    _case(sorted(sum((c.get("cards", []) for c in live_sn), [])) == ["7", "8"],
          f"and both are live at once ({[c.get('cards') for c in live_sn]})")

    # SAME NAME, OVERLAPPING CARDS, REFUSED. The other half, and the one that decides whether
    # this change weakened the lock: the old clash filter was `c.get("name") != name`, so with
    # the name no longer the filename a same-name overlapping acquire would have been excluded
    # from its own clash check, taken a new path via O_EXCL, and produced TWO live claims on one
    # card. Reverting the filter to the name form fails this case and leaves the two above green.
    okd3, msgd3 = acquire("sameName", ["8", "9"], wait=0, pid=me)
    _case(not okd3 and "claimed by" in msgd3,
          f"the same name is refused on an OVERLAPPING set ({msgd3})")

    # A NAME-SCOPED RELEASE MUST NOT FREE THE OTHER ARM. loader's atexit hook passes the cards
    # it acquired for exactly this reason: with one name holding two claims, releasing by name
    # at one arm's exit would drop the other arm's card while its job is still running.
    okr, msgr = release("sameName", ["7"])
    left = [c for c in claims()[0] if c.get("name") == "sameName"]
    _case(okr and [c.get("cards") for c in left] == [["8"]],
          f"releasing one arm's cards leaves the other's held ({msgr}; left {[c.get('cards') for c in left]})")
    # ...and the bare form, which is what a human types after a crash, clears them all.
    acquire("sameName", ["7"], wait=0, pid=me)
    okr2, msgr2 = release("sameName")
    _case(okr2 and not [c for c in claims()[0] if c.get("name") == "sameName"],
          f"the bare release clears every claim under the name ({msgr2})")
    # A PREFIX IS NOT A NAME. `sameName` must not release `sameNameProbe`: the filename now
    # begins with the name, so a prefix test would take both and free a card nobody asked about.
    acquire("sameNameProbe", ["7"], wait=0, pid=me)
    release("sameName")
    _case([c.get("cards") for c in claims()[0] if c.get("name") == "sameNameProbe"] == [["7"]],
          "releasing a name leaves a longer name that starts with it alone")
    release("sameNameProbe")

    # A LEGACY NAME-ONLY CLAIM STAYS RELEASABLE. Claims written by the copy of this file that is
    # on the pod right now use `{name}.json`; they must not become unreleasable by their owner
    # the moment the new code lands. Hand-placed on purpose -- this is the only way that filename
    # can exist once acquire has changed.
    with open(os.path.join(d, "legacyClaim.json"), "w", encoding="utf-8") as fh:
        json.dump({"name": "legacyClaim", "cards": ["4"], "pid": me,
                   "cmdline": _cmdline(me), "acquired": _now(), "note": ""}, fh)
    okl, msgl = release("legacyClaim", ["4"])
    _case(okl and not os.path.exists(os.path.join(d, "legacyClaim.json")),
          f"a pre-rename name-only claim is still releasable by its owner ({msgl})")

    # A dead claimant's cards are reclaimable, or a crash locks the machine forever.
    p = os.path.join(d, claim_file("runA", ["0", "1"]))
    c = _read(p)
    c["pid"] = 999999999
    json.dump(c, open(p, "w"))
    live, stale = claims()
    good = [s.get("name") for s in stale] == ["runA"] and len(live) == 1
    _case(good, f"a claim whose pid is gone reads as stale ({stale and stale[0]['why']})")

    ok4, msg4 = acquire("runD", ["0", "1"], wait=0, pid=me)
    _case(ok4, f"a stale claim's cards are re-acquirable ({msg4})")

    ok5, _ = release("runD")
    live, _ = claims()
    good = ok5 and "runD" not in [c["name"] for c in live]
    _case(good, "release removes the claim")

    ok6, msg6 = release("never_existed")
    good = not ok6
    _case(good, f"releasing an unheld name fails loudly ({msg6})")

    # An orphan is memory with no claim. card_memory() is stubbed, because the property under
    # test is the DISAGREEMENT logic, not nvidia-smi.
    global card_memory
    real = card_memory
    card_memory = lambda: {"3": 9000, "5": 12, "6": 40}   # 3 unclaimed+busy, 5/6 claimed+idle
    orphans, dup, lines = status()
    good = orphans == [("3", 9000)] and any("ORPHAN card 3" in x for x in lines) \
        and any("claimed by runC but idle" in x for x in lines)
    _case(good, "busy-and-unclaimed is an ORPHAN; claimed-and-idle is a note")
    if not good:
        print("       " + " | ".join(lines))

    # No nvidia-smi must not read as "all cards idle".
    card_memory = lambda: None
    orphans, dup, lines = status()
    good = not orphans and any("not measured" in x for x in lines)
    _case(good, "no nvidia-smi reports 'not measured', never zero orphans")
    card_memory = real

    # ------------------------------------------------------------------ de-34
    # A claim must name the process that dies WITH the job, never a shell. Two claims on
    # 2026-09-03 bound to a shell in opposite directions: shapelr bound the launching shell,
    # which exited and left the card reading ORPHAN; the VE arm bound the pod's wrapper bash
    # (2878732, torchrun was 2878734), which kept the claim alive after training ended, so the
    # card read held and the rebind was refused. One cause: `pgrep -f <port>` matches the calling
    # shell first, because the wrapper's own argv carries the port and the word torchrun.
    #
    # BUILDING THE WORLD IS THE HARD PART, and getting it wrong reads as "there is no bug":
    #
    #   `bash -lc '<single command>'` EXECS it rather than forking, so the wrapper pid BECOMES
    #   python and there is no shell layer at all -- the first attempt measured argv0='Python'
    #   and the positive case could not exist. The pod's wrapper survives because its command is
    #   a LIST (`cd ... && setsid env ... torchrun`). A trailing `; true` forces the fork.
    #
    #   With the fork, bash waits, and the child inherits the caller's stdout -- so a pipeline
    #   reading this selftest never sees EOF and hangs to its timeout. That is the runaway shape
    #   CLAUDE.md names, produced by the fixture itself. Every layer gets DEVNULL and its own
    #   session.
    #
    #   macOS has no setsid, so the local tree is bash -> python where the pod's is
    #   bash -> torchrun -> ranks. Accepted deliberately: setsid changes process-GROUP membership,
    #   which affects how a kill propagates, while these predicates read argv0 and the ppid chain.
    #   A world that silently substituted an equivalent would be the worse choice.
    tree = None
    try:
        inner = 'python3 -c "import time; time.sleep(30)"'
        if shutil.which("setsid"):
            inner = f"setsid env DE34=1 {inner}"
        # TWO python children, not one. The second exists for de45's ambiguity refusal: the case
        # 4c named is e1's real shape -- a script whose shell had two candidate descendants, the
        # probe and the card_claim call itself -- and with a single child that assertion cannot run
        # at all. Adding it costs one sleep and makes the refusal testable; the de34 cases below
        # take kids[0] and are unaffected by a sibling.
        second = 'python3 -c "import time; time.sleep(30)"'
        tree = subprocess.Popen(
            ["bash", "-lc", f"{second} & {inner}; true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
            start_new_session=True)
        time.sleep(1.5)
        shell_pid = tree.pid
        table = _ps_table()
        kids = _job_descendants(shell_pid, table)

        # The world must actually BE the pod's shape, or the cases below prove nothing.
        _case(_argv0_is_shell(_cmdline(shell_pid)),
              f"world: pid {shell_pid} really is a shell (argv0), not exec'd away")
        _case(bool(kids), f"world: it really has a python descendant ({len(kids)} found)")
        # And the discriminator has to be argv0, because the substring test is TRUE here.
        _case("python" in _cmdline(shell_pid),
              "world: the shell's own argv mentions python -- why a substring test binds to it")

        ok, msg = acquire("de34_shell", ["7"], pid=shell_pid)
        _case(not ok and "is a shell" in msg and "--pid" in msg,
              "acquire refuses a shell pid and lists its python descendants")
        if ok:
            release("de34_shell")
        _case("--wait-for-device" in msg,
              "and the refusal offers the flag that acts on what it just printed")

        # de45: --wait-for-device FOLLOWS the shell to its job instead of refusing. This world has
        # no GPU, so wait_for_device returns None and the resolver takes its no-device branch --
        # which is the honest limit of a laptop world and is asserted as such rather than skipped.
        # On the pod the same call resolves to the rank; here it must produce the TIMEOUT sentence,
        # distinguishable from the immediate refusal, which is the property that keeps 60 refusals
        # from becoming 60 indistinguishable slow ones.
        okw, msgw = acquire("de45_follow", ["7"], pid=shell_pid, wait_for_device_s=2)
        if nvidia_fds(os.getpid()) is None:
            _case(not okw and "cannot follow it" in msgw and "/proc" in msgw,
                  f"--wait-for-device says /proc is unreadable here rather than polling for an "
                  f"answer that cannot arrive: {msgw[:70]}")
        else:
            _case(not okw and ("TIMEOUT" in msgw or "opened a GPU device in 2s" in msgw),
                  f"--wait-for-device times out with its OWN sentence, not the immediate "
                  f"refusal's: {msgw[:70]}")
            _case("is a shell" not in msgw,
                  "and the timeout message is not the shell refusal reworded -- a reader can tell "
                  "which one they got")
        if okw:
            release("de45_follow")
        # NEGATIVE CONTROL, or the case above passes for any flag value: without the flag the same
        # pid gets the immediate shell refusal, which is a DIFFERENT message.
        okn, msgn = acquire("de45_noflag", ["7"], pid=shell_pid)
        _case(not okn and "is a shell" in msgn and msgn != msgw,
              "without the flag the same pid gets the immediate shell refusal, a different message")
        if okn:
            release("de45_noflag")

        # TWO DESCENDANTS ON A CARD MUST REFUSE, not pick. This is e1's actual shape -- their
        # script had two candidates, the probe and the card_claim call itself -- and it cannot be
        # built on a laptop, because it needs two processes holding real device fds. nvidia_fds is
        # stubbed to answer for the pids this world DOES have, so the branch under test runs on the
        # real _job_descendants output rather than on an invented tree.
        if kids:
            _real_fds = nvidia_fds
            try:
                # One holder: resolves. Two: refuses. The stub differs between the two cases only
                # in HOW MANY pids it says are on a card, which is the variable under test.
                _one = {kids[0][0]}
                globals()["nvidia_fds"] = lambda p: (4 if p in _one else
                                                     (0 if p == shell_pid else _real_fds(p)))
                got, why = _resolve_to_device_holder(shell_pid, 2)
                _case(got == kids[0][0] and why is None,
                      f"one descendant on a card resolves to it (pid {got})")
                if len(kids) > 1:
                    _many = {p for p, _a in kids}
                    globals()["nvidia_fds"] = lambda p: (4 if p in _many else
                                                         (0 if p == shell_pid else _real_fds(p)))
                    got2, why2 = _resolve_to_device_holder(shell_pid, 2)
                    _case(got2 is None and "cannot choose one for you" in (why2 or "")
                          and "--pid" in (why2 or ""),
                          "two descendants on a card REFUSE and list them, rather than picking")
                else:
                    _case(False,
                          f"world: only {len(kids)} descendant, so the ambiguity refusal did not "
                          f"run -- the case 4c named (e1's two candidates) is unexercised")
            finally:
                globals()["nvidia_fds"] = _real_fds

        # The job itself is accepted.
        job_pid = kids[0][0] if kids else None
        if job_pid:
            ok, msg = acquire("de34_job", ["7"], pid=job_pid)
            _case(ok, f"acquire accepts the python descendant (pid {job_pid})")

            # A live same-name claim yields to a DESCENDANT of the recorded pid -- the VE rebind.
            # Recorded pid is the shell here (written directly, since acquire now refuses it),
            # and the rebind target is its child.
            #
            # RELEASE de34_job FIRST. Card 7 is still held by it, and held_cards excludes only
            # claims of the SAME name, so a different name holding the card makes the next
            # acquire a CLASH and it never reaches the rebind branch at all. Both rebind cases
            # failed that way while the behaviour was correct -- verified in isolation before
            # touching the code. A world that leaves the previous case's state behind tests the
            # clash check twice and the rebind never.
            release("de34_job")
            with open(os.path.join(CLAIM_DIR, claim_file("de34_rebind", ["7"])), "w", encoding="utf-8") as fh:
                json.dump({"name": "de34_rebind", "cards": ["7"], "pid": shell_pid,
                           "cmdline": _cmdline(shell_pid), "acquired": _now()}, fh)
            ok, msg = acquire("de34_rebind", ["7"], pid=job_pid)
            _case(ok, f"a live claim on an ancestor rebinds to the job (the VE case): {msg[:60]}")

            # An UNRELATED live pid must still be refused: rebinding is justified only because a
            # descendant is the same job by construction.
            release("de34_rebind")
            with open(os.path.join(CLAIM_DIR, claim_file("de34_unrel", ["7"])), "w", encoding="utf-8") as fh:
                json.dump({"name": "de34_unrel", "cards": ["7"], "pid": job_pid,
                           "cmdline": _cmdline(job_pid), "acquired": _now()}, fh)
            ok, msg = acquire("de34_unrel", ["7"], pid=os.getpid())
            _case(not ok and "already holds a live claim" in msg,
                  f"an unrelated live pid is still refused, not rebound: {msg[:60]}")
            release("de34_unrel")

        # status reports ORPHAN-SHELL once the job is gone but the shell lives.
        with open(os.path.join(CLAIM_DIR, claim_file("de34_orphan", ["7"])), "w", encoding="utf-8") as fh:
            json.dump({"name": "de34_orphan", "cards": ["7"], "pid": shell_pid,
                       "cmdline": _cmdline(shell_pid), "acquired": _now()}, fh)
        _, dup_now, lines_now = status()
        _case(not any("ORPHAN-SHELL" in x for x in lines_now),
              "no ORPHAN-SHELL while the job is still running (the negative case)")

        for p, _a in kids:
            try:
                os.kill(p, 9)
            except OSError:
                pass
        time.sleep(1.0)
        _, dup_now, lines_now = status()
        said = [x for x in lines_now if "ORPHAN-SHELL" in x]
        _case(bool(said) and any("de34_orphan" in x for x in said),
              "status reports ORPHAN-SHELL when the shell lives and the job is gone")
        _case(any(k.startswith("orphan-shell:") for k in dup_now),
              "and it drives a nonzero exit, not a printed line nobody acts on")
    finally:
        if tree is not None:
            for p, _a in _descendants(tree.pid):
                try:
                    os.kill(p, 9)
                except OSError:
                    pass
            tree.kill()
            tree.wait()

    # ------------------------------------------------------- 4c item 1, 2026-09-05
    # A CLAIM MUST NAME A PID THAT HOLDS A CARD. b0_mem_m1: the claimed 842204 held 0 device fds
    # while 842220/842240 held 36 and rank 842276 held 52, so status reported STALE plus "ORPHAN
    # card 1 holds 1179 MiB with no claim" for a correctly running arm.
    #
    # macOS HAS NO /proc, so the fd worlds are built out of real directories and real symlinks
    # and read through PROC_ROOT. That is the same seam FOREIGN_ROOT uses. It does mean these
    # cases prove the COUNTING and the REFUSAL, not that /proc/<pid>/fd on Linux looks like this
    # -- the shape was measured on the pod instead (torch import 0 fds, set_device 34 then 48,
    # torchrun 36, a rank 52) and that measurement is what the numbers here stand in for.
    _saved_proc = PROC_ROOT
    proot = tempfile.mkdtemp(prefix="proc_")

    def _fake_proc(pid, nvidia=0, other=0):
        # REBUILT, not appended to. Several cases give the SAME pid a different device count in
        # sequence -- the self-claim world sets `me` to 0 fds and W1 later sets it to 36 -- and an
        # additive version raises FileExistsError on the second call, which reads as a crash in
        # whichever case happens to run second rather than as a fixture defect.
        d = os.path.join(proot, str(pid))
        shutil.rmtree(d, ignore_errors=True)
        fd = os.path.join(d, "fd")
        os.makedirs(fd, exist_ok=True)
        for i in range(nvidia):
            os.symlink(f"/dev/nvidia{i}", os.path.join(fd, str(i)))
        for i in range(other):
            os.symlink("/dev/null", os.path.join(fd, str(100 + i)))

    try:
        globals()["PROC_ROOT"] = proot
        _fake_proc(4242, nvidia=52, other=9)
        _case(nvidia_fds(4242) == 52,
              f"nvidia_fds counts only nvidia links, not every fd ({nvidia_fds(4242)} of 61)")
        _fake_proc(4243, nvidia=0, other=7)
        _case(nvidia_fds(4243) == 0, "a process with fds but no device reads 0, not None")

        # W4, THE NEGATIVE CONTROL, and the load-bearing case of the four. An unreadable pid must
        # read None and None must never refuse -- /proc does not exist on macOS, so a predicate
        # that refused here would refuse every claim on a laptop and turn this selftest red for a
        # reason unrelated to cards. Asserting only "0 refuses" would pass a version that treats
        # unreadable as 0, which is precisely the version that breaks every laptop.
        _case(nvidia_fds(999999) is None, "an unreadable pid reads None, which is not 0")
        me_ok, me_msg = acquire("w4_noproc", ["4"], pid=me, require_device=True)
        _case(me_ok, f"W4: a claim proceeds when the device cannot be read ({me_msg[:52]})")
        if me_ok:
            got = _read(os.path.join(CLAIM_DIR, claim_file("w4_noproc", ["4"])))
            _case("device_fds" not in (got or {}),
                  "and it records no device_fds rather than recording 0")
            release("w4_noproc")

        # THE SELF-CLAIM SHAPE, and it is the world that was missing. CI went red on main for two
        # hours at 121a865d because `python scripts/loader.py selftest` claims from
        # CUDA_VISIBLE_DEVICES BEFORE the process opens the device -- zero fds by construction --
        # and the refusal was unconditional. It passed on macOS, where no /proc means the
        # predicate abstains, so every local gate was green while every Linux host failed. The
        # asymmetry IS the defect: a world that only ever runs where /proc is absent cannot see a
        # refusal that needs /proc to fire.
        #
        # Reproduced on the pod: "demo_set: pid 902861 holds no GPU device fd". So the default
        # must not refuse, and this case is Linux-shaped on purpose -- PROC_ROOT points at a real
        # directory here, so nvidia_fds returns 0 rather than None on macOS too, which is what
        # makes this world exercise the CI path at all.
        _fake_proc(me, nvidia=0, other=5)
        assert nvidia_fds(me) == 0, "world must look like Linux-with-no-device, not like macOS"
        ok, msg = acquire("selfclaim", ["4"], pid=me)
        _case(ok, f"a SELF-CLAIM by a pid not yet on its card is accepted by default ({msg[:44]})")
        if ok:
            release("selfclaim")
        # And the launcher's opt-in still refuses the same pid: the predicate did not weaken, it
        # moved to the path that can honour it. Without this pair the case above would pass for a
        # version that deleted the refusal outright.
        ok, msg = acquire("launchclaim", ["4"], pid=me, require_device=True)
        _case(not ok and "holds no GPU device fd" in msg,
              "and require_device=True still refuses it (the launcher path, unweakened)")

        # W1: the claimed pid holds nothing while a descendant holds a card. THIS process is the
        # holder, so its real descendants are whatever the selftest spawned -- give it a fake
        # /proc entry with 0 and let the refusal name what it found.
        _fake_proc(me, nvidia=0, other=4)
        ok, msg = acquire("w1_nodev", ["4"], pid=me, require_device=True)
        _case(not ok and "holds no GPU device fd" in msg,
              f"W1: a pid holding 0 device fds is refused ({msg.splitlines()[0][:58]})")
        _case(not ok and ("--pid" in msg or "No descendant holds one yet" in msg),
              "and the refusal says what to claim instead, not just that it failed")

        # And the positive: the same pid with a device is accepted, and the count is recorded.
        # Without this case W1 passes for a version that refuses everything.
        _fake_proc(me, nvidia=36)
        ok, msg = acquire("w1_dev", ["4"], pid=me, require_device=True)
        _case(ok, f"W1 positive: the same pid WITH a device is accepted ({msg[:48]})")
        if ok:
            got = _read(os.path.join(CLAIM_DIR, claim_file("w1_dev", ["4"])))
            _case((got or {}).get("device_fds") == 36,
                  f"and the claim records device_fds={(got or {}).get('device_fds')}")
            release("w1_dev")

        # W2: wait_for_device must not bind a pid that exits during the poll. A dead wrapper ends
        # the wait -- the deadline is the backstop, not the mechanism, or a launch whose job died
        # at once would block for the full 90s.
        dead = subprocess.Popen(["bash", "-c", "exit 0"])
        dead.wait()
        t0 = time.time()
        got = wait_for_device(dead.pid, deadline=5.0, interval=0.1)
        el = time.time() - t0
        _case(got is None and el < 2.0,
              f"W2: a pid that is gone ends the wait at once, claiming nothing ({el:.2f}s)")

        # W2b: a live pid whose descendants never open a device times out. Bounded, and it
        # returns None rather than the pid it found -- "not a shell" is no longer sufficient.
        #
        # `; true` FORCES THE FORK. Without it bash EXECs the single command and the wrapper pid
        # BECOMES python, so there is no descendant at all -- and then W2b passes because the
        # world is empty rather than because the timeout works, which is §186 exactly. Measured:
        # 0 descendants, the case green, the accept path below unreachable. The de-34 world
        # documents the same trap; it caught this one 20 lines later.
        idle = subprocess.Popen(["bash", "-c", 'python3 -c "import time; time.sleep(9)"; true'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            time.sleep(1.0)
            kids2 = _job_descendants(idle.pid)
            _case(bool(kids2), f"world: the idle tree has a python descendant ({len(kids2)})")
            for q, _a in kids2:
                _fake_proc(q, nvidia=0, other=3)
            t0 = time.time()
            got = wait_for_device(idle.pid, deadline=1.5, interval=0.1)
            _case(got is None, "W2b: a descendant that never opens a device is not claimed")
            _case(time.time() - t0 >= 1.4, "and the wait actually waited its deadline")
        finally:
            for q, _a in _descendants(idle.pid):
                try:
                    os.kill(q, 9)
                except OSError:
                    pass
            idle.kill()
            idle.wait()

        # THE LIVE M1 DEFECT, 2026-09-05, and it was NOT the deadline the message blamed. The
        # launcher printed "no descendant of 915701 opened a GPU device within 90s" while the ranks
        # had held device fds since 22:35:01, two seconds after launch -- b0 claimed by hand at
        # 22:36:01. Cause: a VANISHED pid reads None from nvidia_fds, the same value as "this host
        # has no /proc", and the first version returned None on the first unreadable descendant.
        # run_ddp.sh runs scripts/pod_drift.py --check (run_ddp.sh:34) and other short-lived
        # children, so a descendant disappears between the ps snapshot and the fd read, about a
        # second in. The poll aborted then and blamed a timeout that had not happened.
        #
        # THE WORLD MUST PUT AN UNREADABLE PID INSIDE THE WALK, and my first attempt did not --
        # it asserted nvidia_fds(9999991) is None beside a healthy tree, which is true of the
        # broken code too. Measured: 3 of 4 mutants GREEN, including the original bug restored
        # verbatim. A world that cannot reach the defect proves nothing about the fix, and the
        # mutation run is the only reason I know that (§186's shape, in my own test).
        #
        # So: patch _job_descendants for the duration, to return a vanished pid FIRST and the
        # device-holding one second -- the real ordering, since ps lists by pid and the transient
        # child is younger. The tree is real, only the walk's answer is staged.
        vlive = subprocess.Popen(["bash", "-c", 'python3 -c "import time; time.sleep(9)"; true'],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        _saved_jd = globals()["_job_descendants"]
        try:
            time.sleep(1.0)
            kids4 = _saved_jd(vlive.pid)
            _case(bool(kids4), f"world: the vanish tree has a real descendant ({len(kids4)})")
            if kids4:
                real = kids4[0][0]
                _fake_proc(real, nvidia=44)
                gone = 9999991
                _case(nvidia_fds(gone) is None and nvidia_fds(real) == 44,
                      "world: one pid reads None (vanished), the other reads 44 (on a card)")

                def _staged(pid, table=None):
                    return [(gone, "python3 -u train.py"), (real, "python3 -u train.py")]

                globals()["_job_descendants"] = _staged
                got = wait_for_device(vlive.pid, deadline=3.0, interval=0.1)
                _case(got == (real, 44),
                      f"a vanished pid EARLIER in the walk does not end the wait: {got}")
                # THE TWO SEPARATIONS THE MUTATION RUN FOUND UNCOVERED. Both were GREEN on the
                # first rebuild, i.e. the code was right and nothing held it there.
                #
                # (a) NO-PROC IS DECIDED ONCE, ON THE WRAPPER, not per descendant. Moving the test
                # inside the loop makes it fire on any vanished pid again -- the original bug with
                # an extra step. Staged with a walk whose ONLY entry is unreadable and a wrapper
                # that IS readable: the answer must be a timeout (None after the deadline), not an
                # instant None, and the difference is visible only in the elapsed time.
                globals()["_job_descendants"] = lambda pid, table=None: [(gone, "python3 x.py")]
                _t0 = time.time()
                got2 = wait_for_device(vlive.pid, deadline=1.0, interval=0.1)
                _el = time.time() - _t0
                _case(got2 is None and _el >= 0.9,
                      f"an all-vanished walk TIMES OUT rather than abstaining instantly "
                      f"({_el:.2f}s of a 1.0s deadline)")

                # (b) deadline=None means "while the wrapper lives", not "90 seconds". A constant
                # reinstated anywhere reads identically until the wait outlives it, so the world
                # has to outlive it -- impossible in a selftest at 90s. Assert the CONTRACT
                # instead, on a dead wrapper where an unbounded wait must still return at once:
                # with a constant ceiling this passes too, so pair it with the module default,
                # which is the thing a reinstated constant would have to change.
                _dead = subprocess.Popen(["bash", "-c", "exit 0"])
                _dead.wait()
                globals()["_job_descendants"] = _saved_jd
                _t0 = time.time()
                got3 = wait_for_device(_dead.pid, deadline=None, interval=0.1)
                _case(got3 is None and time.time() - _t0 < 2.0,
                      "deadline=None still returns at once when the wrapper is gone")
                _case(DEVICE_WAIT_S is None,
                      f"and the module default carries no ceiling ({DEVICE_WAIT_S!r}) -- a "
                      f"constant here abandoned a live launch's cards")

                # (a2) THE OTHER HALF OF "decided once, on the wrapper": with the wrapper
                # UNREADABLE the answer is None even though a descendant holds 44 fds. Case (a)
                # above cannot see this -- it keeps the wrapper readable, so deleting the check
                # only removes an early return nothing reaches, and the mutation run read GREEN.
                # This is the macOS contract, stated as a case: no /proc means the predicate
                # abstains and the caller falls back, rather than polling for an answer that
                # cannot arrive.
                shutil.rmtree(os.path.join(proot, str(me)), ignore_errors=True)
                globals()["_job_descendants"] = _staged
                _t0 = time.time()
                got4 = wait_for_device(vlive.pid, deadline=3.0, interval=0.1)
                _el4 = time.time() - _t0
                _case(got4 is None and _el4 < 0.5,
                      f"an unreadable wrapper abstains AT ONCE even with a 44-fd descendant in "
                      f"the walk ({got4}, {_el4:.2f}s)")
                _fake_proc(me, nvidia=36)

                # (b2) deadline=0 MEANS CHECK ONCE. `deadline or 90.0` reads identically to
                # `None if deadline is None` for every value except 0 -- and 0 is the one a caller
                # uses for a non-blocking probe, where the falsy version waits 90 s instead. The
                # 90 s ceiling itself cannot be tested in a selftest; this boundary can, and it is
                # the same bug in a reachable place.
                globals()["_job_descendants"] = lambda pid, table=None: [(gone, "python3 x.py")]
                _t0 = time.time()
                got5 = wait_for_device(vlive.pid, deadline=0, interval=0.1)
                _el5 = time.time() - _t0
                _case(got5 is None and _el5 < 0.5,
                      f"deadline=0 checks once and returns, it is not falsy-promoted to a "
                      f"default wait ({_el5:.2f}s)")
        finally:
            globals()["_job_descendants"] = _saved_jd
            for q, _a in _descendants(vlive.pid):
                try:
                    os.kill(q, 9)
                except OSError:
                    pass
            vlive.kill()
            vlive.wait()

        # W3, AND IT DOES NOT BEHAVE THE WAY THE SPEC ASSUMED. "keep walking the tree from the
        # wrapper" is not available: when the intermediate shell exits, the job REPARENTS to init
        # and leaves the wrapper's subtree entirely. MEASURED on this laptop -- outer bash 79367,
        # its python at ppid 1, _job_descendants(79367) == [] with the python still running.
        # A ppid walk cannot see a process that is no longer a descendant, on either platform.
        #
        # So the honest behaviour is a TIMEOUT that claims nothing, which the launcher already
        # reports. That is safe (no wrong pid is bound) and it is a real limit, not a fix: a job
        # launched through a shell that exits gets no claim at all. The pod's own launches do not
        # have this shape -- harness's wrapper waits for torchrun, and the setsid form detaches
        # deliberately before the claim is attempted -- which is why this is recorded rather than
        # worked around.
        orph = subprocess.Popen(
            ["bash", "-lc", 'bash -c \'python3 -c "import time; time.sleep(8)" & exit 0\'; sleep 8'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
            start_new_session=True)
        try:
            time.sleep(1.5)
            reparented = [(q, pp) for q, pp, a in _ps_table() if "time.sleep(8)" in a and pp == 1]
            _case(bool(reparented),
                  f"world: the job really did reparent out of the tree (ppid 1): {reparented[:1]}")
            _case(not _job_descendants(orph.pid),
                  "W3: the ppid walk cannot see it -- so the poll finds nothing to claim")
            for q, _pp in reparented:
                _fake_proc(q, nvidia=48)
            got = wait_for_device(orph.pid, deadline=1.0, interval=0.1)
            _case(got is None,
                  "and it claims nothing even though a process on a card exists (a limit, "
                  "not a fix: an exited intermediate shell yields no claim)")
        finally:
            for q, _pp, a in _ps_table():
                if "time.sleep(8)" in a:
                    try:
                        os.kill(q, 9)
                    except OSError:
                        pass
            orph.kill()
            orph.wait()

        # The accept path end to end: a wrapper whose live descendant holds a device is found and
        # returned. Without it every case above passes for a wait_for_device that returns None
        # unconditionally (§186: an absence assertion is satisfied by code that never runs).
        good = subprocess.Popen(["bash", "-c", 'python3 -c "import time; time.sleep(9)"; true'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            time.sleep(1.0)
            kids3 = _job_descendants(good.pid)
            for q, _a in kids3:
                _fake_proc(q, nvidia=52)
            got = wait_for_device(good.pid, deadline=3.0, interval=0.1)
            _case(got is not None and got[1] == 52,
                  f"the accept path returns the device-holding descendant: {got}")
            _case(got is not None and got[0] in {q for q, _a in kids3},
                  "and it is a real descendant, not the wrapper")
        finally:
            for q, _a in _descendants(good.pid):
                try:
                    os.kill(q, 9)
                except OSError:
                    pass
            good.kill()
            good.wait()
    finally:
        globals()["PROC_ROOT"] = _saved_proc
        shutil.rmtree(proot, ignore_errors=True)

    # ------------------------------------------------------------------ 6e ruling 2026-09-04
    # FOREIGN: a card runs/card_assignment.json marks as another container's or the user's must

    # NOT print as ORPHAN. ORPHAN's instruction is "find it and reclaim it", and e1 got
    # "ORPHAN card 7 27,809 MiB" for the user's own job -- the third reader of that line would
    # have acted on it. Both directions, and the negative is the load-bearing one: the SAME
    # memory on an UNMARKED card must still read ORPHAN, or the change has simply silenced the
    # warning everywhere.
    _saved_foreign_root = FOREIGN_ROOT
    froot = tempfile.mkdtemp(prefix="foreign_")
    try:
        os.makedirs(os.path.join(froot, "runs"), exist_ok=True)
        with open(os.path.join(froot, "runs", "card_assignment.json"), "w") as fh:
            json.dump({"cards": {
                "3": "lane, but HELD by the user's container job -- NEVER TAKE while held",
                "0-1": "params leg block, ours",
            }}, fh)
        globals()["FOREIGN_ROOT"] = froot
        got = foreign_cards(froot)
        _case(set(got) == {"3"},
              f"only the marked card is foreign, and a ranged ours-key is not: {sorted(got)}")

        # OVERLAPPING KEYS, which the real file has carried since 2026-09-04 ("0-3" alongside
        # "0".."3"). A card named by a foreign range AND by a non-foreign single key must stay
        # foreign whichever order the dict yields, since the wrong direction tells someone to
        # reclaim a card that is not ours.
        with open(os.path.join(froot, "runs", "card_assignment.json"), "w") as fh:
            json.dump({"cards": {
                "2-3": "FOREIGN OCCUPANT: another container",
                "2": " | after the run ends: granted to e1",
                "3": " | after the run ends: granted to e1",
            }}, fh)
        got = foreign_cards(froot)
        _case(set(got) == {"2", "3"},
              f"a non-foreign single key does not clear a foreign range over it: {sorted(got)}")

        # A ranged FOREIGN key must expand, since the file uses both forms and a string
        # compare against an nvidia-smi index would miss every range.
        with open(os.path.join(froot, "runs", "card_assignment.json"), "w") as fh:
            json.dump({"cards": {"4-6": "FOREIGN OCCUPANT: another container's pids"}}, fh)
        got = foreign_cards(froot)
        _case(set(got) == {"4", "5", "6"},
              f"a ranged foreign key expands to every index in it: {sorted(got)}")

        with open(os.path.join(froot, "runs", "card_assignment.json"), "w") as fh:
            json.dump({"cards": {"7": "the user's own job, NEVER TAKE"}}, fh)
        real = card_memory
        globals()["card_memory"] = lambda: {"7": 27809, "2": 27809}
        orphans, dup, lines = status()
        said_foreign = [x for x in lines if x.startswith("FOREIGN")]
        said_orphan = [x for x in lines if x.startswith("ORPHAN")]
        _case(any("card 7" in x and "NEVER TAKE" in x for x in said_foreign),
              "a marked card holding memory prints FOREIGN with the reason, not ORPHAN")
        _case(not any("card 7" in x for x in said_orphan) and ("7", 27809) not in orphans,
              "and it is NOT counted as an orphan, so nothing tells anyone to reclaim it")
        _case(any("card 2" in x for x in said_orphan) and ("2", 27809) in orphans,
              "THE SAME memory on an unmarked card still reads ORPHAN (the negative case)")
        globals()["card_memory"] = real

        # ---------------------------------------------------------- grant_lane (de, 2026-09-04)
        # The four fields must move together, and the gate that reads them must go GREEN on the
        # result. Both halves are asserted here because the writer's whole purpose is to be
        # readable by launch_gate.gate_cards -- a grant written in a shape the gate refuses is
        # the divergence with an extra step.
        with open(os.path.join(froot, "runs", "card_assignment.json"), "w") as fh:
            json.dump({"cards": {"6": "FOREIGN OCCUPANT: another container"},
                       "block_cards": "0-3", "lane_card": "7",
                       "launch_block_granted": True, "granted_by": "fb 2026-09-03"}, fh)
        ok, msg = grant_lane("5", "de", "the gate_cards fixture", "6e", root=froot)
        _case(ok, f"grant_lane writes a lane grant: {msg}")
        got = json.load(open(os.path.join(froot, "runs", "card_assignment.json")))
        _case(got.get("lane_card") == "5" and got.get("lane_to") == "de"
              and "de" in got["cards"].get("5", "") and "de" in got.get("lane_note", "")
              and any("lane 5 -> de" in x for x in got.get("granted_by", [])),
              f"all five fields name the same grant: {json.dumps(got)[:120]}")
        _case(got["cards"].get("6", "").startswith("FOREIGN"),
              "and the other cards' entries are untouched")

        ok, msg = grant_lane("6", "de", "onto a foreign card", "6e", root=froot)
        _case(not ok and "FOREIGN" in msg.upper(),
              f"REFUSES the lane onto a card marked foreign: {msg}")
        _case(json.load(open(os.path.join(froot, "runs", "card_assignment.json")))
              .get("lane_card") == "5",
              "and a refused grant writes nothing (lane_card still the previous grant)")

        # THE GATE READS IT. Positive: de's own lane launch on card 5 is confirmed. Negative,
        # and this is the case the old gate got wrong: the same launch on card 6 refuses.
        import launch_gate
        _saved = launch_gate.LAUNCH_CARDS
        _saved_owner = os.environ.get("LAUNCH_OWNER")
        try:
            os.environ["LAUNCH_OWNER"] = "de"
            launch_gate.LAUNCH_CARDS = ["5"]
            st, why = launch_gate.gate_cards(froot, None, 1)
            _case(st == launch_gate.GO and "de" in why,
                  f"gate_cards confirms the lane it was granted: {st} {why[:90]}")
            launch_gate.LAUNCH_CARDS = ["6"]
            st, why = launch_gate.gate_cards(froot, None, 1)
            _case(st == launch_gate.NOGO and "6" in why,
                  f"and refuses the same launch on an ungranted card: {st} {why[:90]}")
            # The lane granted to SOMEONE ELSE. The gate must refuse b0 on de's lane card --
            # this is the collision the item was raised for, and it is the case an
            # ownership-blind gate reports GO on.
            os.environ["LAUNCH_OWNER"] = "b0"
            launch_gate.LAUNCH_CARDS = ["5"]
            st, why = launch_gate.gate_cards(froot, None, 1)
            _case(st == launch_gate.NOGO and "b0" in why,
                  f"and refuses a lane granted to someone else: {st} {why[:90]}")
            # OWNERSHIP COMES FROM lane_to, NOT FROM THE PROSE, and this world is the reason:
            # the real 2026-09-04 lane_note grants card 5 to e1, and a substring search for the
            # session name "de" hits inside the word "excluded" in it. My first version of the
            # gate did exactly that and reported GO for de on e1's lane. Word boundaries do not
            # rescue it -- "b0" then matches "ckpt_b0_sd_equalcompute" in the same string.
            _prose = ("2026-09-04 05:40Z fb: lane card 5 granted to e1 for C11: one score_matrix "
                      "run over ckpt_b0_sd_equalcompute; domain_bpb excluded (C12)")
            _obj = json.load(open(os.path.join(froot, "runs", "card_assignment.json")))
            _obj["lane_note"] = _prose
            _obj["cards"]["5"] = "lane: e1 C11 doc_cu re-score"
            _obj["lane_to"] = "e1"
            with open(os.path.join(froot, "runs", "card_assignment.json"), "w") as fh:
                json.dump(_obj, fh)
            for _w in ("de", "b0"):
                os.environ["LAUNCH_OWNER"] = _w
                st, why = launch_gate.gate_cards(froot, None, 1)
                _case(st == launch_gate.NOGO and "e1" in why,
                      f"{_w} does not get e1's lane off a prose match: {st} {why[:80]}")
            os.environ["LAUNCH_OWNER"] = "e1"
            st, why = launch_gate.gate_cards(froot, None, 1)
            _case(st == launch_gate.GO, f"and e1, whom lane_to names, does: {st} {why[:80]}")
            # No lane_to at all: UNKNOWN naming the writer, never a guess from the prose.
            _obj.pop("lane_to")
            with open(os.path.join(froot, "runs", "card_assignment.json"), "w") as fh:
                json.dump(_obj, fh)
            st, why = launch_gate.gate_cards(froot, None, 1)
            _case(st == launch_gate.UNKNOWN and "grant-lane" in why,
                  f"a lane grant with no lane_to is UNKNOWN naming the writer: {st} {why[:80]}")
        finally:
            launch_gate.LAUNCH_CARDS = _saved
            if _saved_owner is None:
                os.environ.pop("LAUNCH_OWNER", None)
            else:
                os.environ["LAUNCH_OWNER"] = _saved_owner
    finally:
        globals()["FOREIGN_ROOT"] = _saved_foreign_root
        shutil.rmtree(froot, ignore_errors=True)

    # ------------------------------------------------------------------ 6e ruling 2026-09-03
    # ZOMBIE, the third disagreement. A REAL zombie, made with fork+exit and deliberately not
    # reaped -- NOT a fabricated dead pid, which is the world that cannot tell this defect from
    # its absence: a fabricated pid is gone, reads stale, and the zombie case is precisely the
    # one where every liveness reader says ALIVE. Both directions, live claim first.
    zpid = os.fork()
    if zpid == 0:
        os._exit(0)
    try:
        # NEGATIVE FIRST, on the live process this test process itself is: no ZOMBIE line for a
        # claim whose pid is running. Without this the positive could pass on a status() that
        # printed ZOMBIE for everything.
        with open(os.path.join(CLAIM_DIR, "zlive.json"), "w", encoding="utf-8") as fh:
            json.dump({"name": "zlive", "cards": ["7"], "pid": os.getpid(),
                       "cmdline": _cmdline(os.getpid()), "acquired": _now()}, fh)
        _, dup_z, lines_z = status()
        _case(not any("ZOMBIE" in x for x in lines_z),
              "no ZOMBIE for a claim whose pid is genuinely running (the negative case)")
        release("zlive")

        # The defect this fixes, asserted as a fact about the readers rather than trusted:
        # every liveness test in this file says the corpse is alive.
        time.sleep(0.3)
        st = _proc_stat(zpid)
        _case(st.startswith("Z") and _alive(zpid),
              f"a zombie reads state {st!r} while _alive() says True -- os.kill(pid,0) cannot "
              f"see the difference, which is why claims() filed b0's corpse as live")

        with open(os.path.join(CLAIM_DIR, claim_file("zdead", ["7"])), "w", encoding="utf-8") as fh:
            json.dump({"name": "zdead", "cards": ["7"], "pid": zpid,
                       "cmdline": "python3 rescore.py", "acquired": _now()}, fh)
        live_z, stale_z = claims()
        _case(any(c["name"] == "zdead" for c in live_z)
              and "zdead" not in [s.get("name") for s in stale_z],
              "the zombie claim is still LIVE, not stale -- so acquire keeps refusing the card "
              "and nothing is auto-broken")
        ok_z, msg_z = acquire("zother", ["7"], wait=0, pid=me)
        _case(not ok_z and "claimed by" in msg_z,
              f"and the card is still refused to a second acquirer ({msg_z[:52]})")
        # ...AND THAT REFUSAL NAMES THE ZOMBIE. This is the case a reader actually hits -- armB
        # wanting the card armA died holding -- and it arrives at the CLASH path, not the O_EXCL
        # path the ruling was implemented on. Measured before the fix: "cards ['7'] are claimed
        # by {'7': ['zdead']}. Queue -- do not spill onto a claimed card", which is what e1 read
        # as "wait for the running job" while waiting ten minutes on a corpse. Deleting the
        # corpses loop in acquire fails this case and leaves the same-name one green, which is
        # exactly how the gap survived the ruling.
        _case("ZOMBIE" in msg_z and "release --name zdead" in msg_z,
              f"a DIFFERENT name's refusal names the zombie and the release command too "
              f"({msg_z[:70]})")
        # ACQUIRE'S OWN MESSAGE must name the zombie, not just status's line. e1 got
        # "already holds a live claim" for a Zs pid in production, where the advice reads as
        # "wait for the running job" (6e, 2026-09-04). Asserted on the SAME name, which is the
        # case that arises: arm 2 re-acquiring the name arm 1 left behind.
        ok_same, msg_same = acquire("zdead", ["7"], wait=0, pid=me)
        _case(not ok_same and "ZOMBIE" in msg_same,
              f"acquire on the zombie's own claim name says ZOMBIE, not 'a live claim': "
              f"{msg_same[:60]}")
        _case(not ok_same and "release --name zdead" in msg_same,
              "and it names the release command rather than leaving the reader to guess")

        _, dup_z, lines_z = status()
        said_z = [x for x in lines_z if "ZOMBIE" in x]
        _case(bool(said_z) and any("zdead" in x for x in said_z),
              "status names it: ZOMBIE with the pid state and the claim age")
        _case(any(k.startswith("zombie:") for k in dup_z),
              "and it drives a nonzero exit, not a printed line nobody acts on")
        release("zdead")
    finally:
        os.waitpid(zpid, 0)

    # CVD-MISMATCH, the fourth disagreement. A REAL child with a REAL environment, not a fake:
    # the whole question is what an exec-time reader sees, so a fabricated pid or a hand-written
    # claim would test the comparison and skip the reader -- which is the half that has a trap in
    # it. Runs on macOS via the `ps eww` path (measured 2026-09-04), so this is not a SKIP.
    child = os.path.join(d, "cvd_child.py")
    with open(child, "w", encoding="utf-8") as fh:
        # argv CARRIES the literal string while the environment does NOT, which is the trap: `ps
        # eww` prints command then environment with no delimiter, and run_ddp.sh's wrapper argv
        # contains `CUDA_VISIBLE_DEVICES=3,4,5,6`. A naive scan of that output returns the
        # COMMAND's value. Measured: naive '7', after stripping the command, absent.
        fh.write("import time\ntime.sleep(30)\n")
    env_set = dict(os.environ)
    env_set["CUDA_VISIBLE_DEVICES"] = "2,3"
    env_bare = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    kid_set = subprocess.Popen([sys.executable, child], env=env_set,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    kid_trap = subprocess.Popen([sys.executable, child, "CUDA_VISIBLE_DEVICES=7"], env=env_bare,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.0)
        _case(_cvd(kid_set.pid) == "2,3",
              f"_cvd reads the exec-time value ({_cvd(kid_set.pid)!r} for a child given 2,3)")
        # THE TRAP, asserted directly. Without the command-prefix strip this reads '7' and the
        # check would refuse every run_ddp launch, whose wrapper argv names the variable.
        _case(_cvd(kid_trap.pid) == "",
              f"a cmdline CONTAINING CUDA_VISIBLE_DEVICES=7 with none in the environment reads "
              f"absent, not '7' ({_cvd(kid_trap.pid)!r}) -- the command prefix is stripped")
        _case(_cvd_mismatch(kid_set.pid, ["2", "3"]) is None,
              "a matching claim is not a mismatch")
        _case(_cvd_mismatch(kid_set.pid, ["3", "2"]) is None,
              "and order does not matter -- compared as sets, which is what both sides hold")
        got_m = _cvd_mismatch(kid_set.pid, ["4"])
        _case(got_m is not None and got_m[0] == "2,3" and "claimed ['4']" in got_m[1],
              f"a disagreeing claim IS a mismatch, and says both sides ({got_m and got_m[1]})")
        # ABSENT DOES NOT REFUSE, and this is the case that keeps four correct scripts working.
        # scripts/test_e2e.py:37 and bench_eff/*'s setdefault assign CUDA_VISIBLE_DEVICES INSIDE
        # the process, which no exec-time reader can see. The original ruling said to refuse on
        # absent; measured 2026-09-04, that refuses them.
        _case(_cvd_mismatch(kid_trap.pid, ["0"]) is None,
              "absent never refuses -- a job that sets the variable itself reads absent here "
              "(test_e2e.py:37, bench_eff/*'s setdefault)")
        # CARD 4 ONLY, and unclaimed by every case above: runC holds 5,6 and the de34/zombie cases
        # hold 7. My first fixture claimed 4,5 and the clash check refused it FIRST with "cards
        # ['5'] are claimed by runC" -- correct precedence, and the CVD assertion never ran. A
        # negative that fails for the wrong reason certifies nothing.
        ok_cvd, msg_cvd = acquire("de_cvd", ["4"], pid=kid_set.pid)
        _case(not ok_cvd and "CUDA_VISIBLE_DEVICES" in msg_cvd and "'2,3'" in msg_cvd,
              f"acquire REFUSES a claim on cards the job cannot see: {msg_cvd[:78]}")
        _case(not ok_cvd and "relaunch with the variable set to 4" in msg_cvd,
              "and the message names the cards to relaunch with, not just the disagreement")
        ok_cvd2, msg_cvd2 = acquire("de_cvd_ok", ["2", "3"], pid=kid_set.pid)
        # WITHOUT THIS the negative above would pass on a check that refuses every acquire.
        _case(ok_cvd2, f"a claim on the cards it CAN see is accepted ({msg_cvd2})")
        if ok_cvd2:
            _, dup_c, lines_c = status()
            said_c = [x for x in lines_c if "CVD-MISMATCH" in x]
            _case(not said_c, "a matching claim produces no CVD-MISMATCH line")
            _case(not any(k.startswith("cvd-mismatch:") for k in dup_c),
                  "and does not drive a nonzero exit")
            release("de_cvd_ok")
        # status's own line, on a hand-placed claim: acquire refuses this state, so the only way
        # it exists on disk is a launch that changed the variable after the claim -- which is
        # exactly the state status has to report.
        with open(os.path.join(d, claim_file("de_cvd_bad", ["6", "7"])), "w", encoding="utf-8") as fh:
            json.dump({"name": "de_cvd_bad", "cards": ["6", "7"], "pid": kid_set.pid,
                       "cmdline": _cmdline(kid_set.pid), "acquired": _now(), "note": ""}, fh)
        _, dup_c2, lines_c2 = status()
        said_c2 = [x for x in lines_c2 if "CVD-MISMATCH" in x]
        _case(bool(said_c2) and any("de_cvd_bad" in x for x in said_c2),
              "status names it: CVD-MISMATCH with the value and both card sets")
        _case(any(k.startswith("cvd-mismatch:") for k in dup_c2),
              "and it drives a nonzero exit, not a printed line nobody acts on")
        release("de_cvd_bad")
    finally:
        kid_set.kill()
        kid_set.wait()
        kid_trap.kill()
        kid_trap.wait()

    shutil.rmtree(d, ignore_errors=True)
    print(f"card_claim selftest: {n - bad}/{n} pass")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", nargs="?", choices=["acquire", "release", "status", "grant-lane"])
    ap.add_argument("--name", help="the exp row name of the job claiming the cards")
    ap.add_argument("--cards", help="comma-separated card indices, as CUDA_VISIBLE_DEVICES")
    ap.add_argument("--to", help="grant-lane: who the lane is granted to (a session name)")
    ap.add_argument("--why", default="", help="grant-lane: the one job it is granted for")
    ap.add_argument("--by", default="", help="grant-lane: who is granting (the controller)")
    ap.add_argument("--wait", type=int, default=0, help="seconds to wait for a clash to clear")
    ap.add_argument("--note", default="", help="free text: what this job is")
    ap.add_argument("--pid", type=int, default=None,
                    help="the pid that HOLDS the cards (default: this command's parent, i.e. "
                         "the launcher script). A claim recording this command's own pid is "
                         "stale the instant it is written")
    ap.add_argument("--require-device", action="store_true",
                    help="refuse a --pid that holds no GPU device fd. For a LAUNCHER claiming "
                         "another process after it is on a card (harness launch does this after "
                         "wait_for_device). NOT for a process claiming itself before it opens "
                         "the device it named through CUDA_VISIBLE_DEVICES -- that holds zero "
                         "fds by construction (scripts/loader.py claim_my_cards)")
    ap.add_argument("--wait-for-device", nargs="?", type=int, const=90, default=None,
                    metavar="SECONDS",
                    help="when --pid is a shell or holds no device fd, follow it to the ONE "
                         "python/torchrun descendant that IS on a card and claim that, polling up "
                         "to SECONDS (default 90, matching harness launch's own poll). Refuses, "
                         "listing them, if several descendants hold a device -- choosing for you "
                         "would be wrong half the time. Off unless passed: a process claiming "
                         "ITSELF has no descendant to follow and would just wait out the deadline")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.action:
        ap.error("an action is required (acquire, release, status, grant-lane) unless --selftest")
    if a.action == "grant-lane":
        for f in ("cards", "to", "why", "by"):
            if not getattr(a, f):
                ap.error(f"--{f} is required for grant-lane")
        cards = [c.strip() for c in a.cards.split(",") if c.strip()]
        if len(cards) != 1:
            ap.error("grant-lane takes exactly one card: the lane holds one job at a time")
        ok, msg = grant_lane(cards[0], a.to, a.why, a.by)
        print(msg, file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 1
    if a.action == "status":
        orphans, dup, lines = status()
        for line in lines:
            print(line)
        return 1 if (orphans or dup) else 0
    if not a.name:
        ap.error(f"--name is required for {a.action}")
    if a.action == "release":
        ok, msg = release(a.name)
        print(msg, file=sys.stderr if not ok else sys.stdout)
        return 0 if ok else 1
    if not a.cards:
        ap.error("--cards is required for acquire")
    cards = [c.strip() for c in a.cards.split(",") if c.strip()]
    ok, msg = acquire(a.name, cards, wait=a.wait, note=a.note, pid=a.pid,
                      require_device=a.require_device,
                      wait_for_device_s=a.wait_for_device)
    print(msg, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

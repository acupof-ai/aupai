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


def _alive(pid):
    """Whether pid exists IN THIS NAMESPACE. signal 0, not /proc: this also has to work on
    a Mac, where the selftest runs."""
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
    which is the failure mode that makes people delete lock files by hand and then race."""
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
            stale.append(dict(c, why=f"pid {c.get('pid')} is gone"))
        else:
            live.append(c)
    return live, stale


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


def acquire(name, cards, wait=0, note="", pid=None):
    """Claim `cards` for `name`, waiting up to `wait` seconds. Returns (ok, message).

    Refuses rather than sharing: this is a lock, not a message. The previous arrangement was a
    message -- two sessions each announced a launch and both proceeded."""
    os.makedirs(CLAIM_DIR, exist_ok=True)
    mine = os.path.join(CLAIM_DIR, f"{name}.json")
    deadline = time.time() + wait
    while True:
        live, stale = claims()
        for s in stale:
            f = os.path.join(CLAIM_DIR, s.get("file") or f"{s.get('name')}.json")
            if os.path.exists(f):
                os.unlink(f)
        held = held_cards([c for c in live if c.get("name") != name])
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
                kids = _job_descendants(holder)
                msg = (f"pid {holder} is a shell, not the job: "
                       f"{_cmdline(holder)[:90]!r}\n"
                       f"A claim on a shell fails both ways -- the shell exits and the card "
                       f"reads ORPHAN, or it lingers and the card reads held after the job is "
                       f"gone (both happened 2026-09-03).")
                if kids:
                    msg += "\nIts python/torchrun descendants -- claim one of these:\n" + "\n".join(
                        f"    --pid {p}  {a[:96]}" for p, a in kids[:6])
                else:
                    msg += ("\nIt has no python/torchrun descendant yet. Claim after the job "
                            "starts, or pass --pid explicitly.")
                return False, msg
            claim = {"name": name, "cards": list(cards), "pid": holder,
                     "cmdline": _cmdline(holder), "acquired": _now(), "note": note}
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
            return False, (f"cards {sorted(clash)} are claimed by {clash}. "
                           f"Queue -- do not spill onto a claimed card.")
        time.sleep(min(15, max(2, wait / 20)))


def release(name):
    p = os.path.join(CLAIM_DIR, f"{name}.json")
    if not os.path.exists(p):
        return False, f"no claim for {name}"
    os.unlink(p)
    return True, f"released {name}"


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
    if mem is None:
        lines.append("CARDS  not measured (no nvidia-smi here) -- claims listed above are all "
                     "this can say")
    else:
        for card, m in sorted(mem.items(), key=lambda kv: kv[0]):
            if m > FREE_MIB and card not in held:
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

    ok, msg = acquire("runA", ["0", "1"])
    _case(ok, f"a free card set is claimable ({msg})")

    # The whole point: the second acquirer must NOT get overlapping cards.
    ok2, msg2 = acquire("runB", ["1", "2"], wait=0)
    good = not ok2 and "claimed by" in msg2
    _case(good, f"an overlapping set is refused, not shared ({msg2})")

    # Disjoint sets coexist -- a lock that blocks unrelated work gets deleted by hand.
    ok3, msg3 = acquire("runC", ["5", "6"], wait=0)
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

    # A dead claimant's cards are reclaimable, or a crash locks the machine forever.
    p = os.path.join(d, "runA.json")
    c = _read(p)
    c["pid"] = 999999999
    json.dump(c, open(p, "w"))
    live, stale = claims()
    good = [s.get("name") for s in stale] == ["runA"] and len(live) == 1
    _case(good, f"a claim whose pid is gone reads as stale ({stale and stale[0]['why']})")

    ok4, msg4 = acquire("runD", ["0", "1"], wait=0)
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
        tree = subprocess.Popen(
            ["bash", "-lc", f"{inner}; true"],
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
            with open(os.path.join(CLAIM_DIR, "de34_rebind.json"), "w", encoding="utf-8") as fh:
                json.dump({"name": "de34_rebind", "cards": ["7"], "pid": shell_pid,
                           "cmdline": _cmdline(shell_pid), "acquired": _now()}, fh)
            ok, msg = acquire("de34_rebind", ["7"], pid=job_pid)
            _case(ok, f"a live claim on an ancestor rebinds to the job (the VE case): {msg[:60]}")

            # An UNRELATED live pid must still be refused: rebinding is justified only because a
            # descendant is the same job by construction.
            release("de34_rebind")
            with open(os.path.join(CLAIM_DIR, "de34_unrel.json"), "w", encoding="utf-8") as fh:
                json.dump({"name": "de34_unrel", "cards": ["7"], "pid": job_pid,
                           "cmdline": _cmdline(job_pid), "acquired": _now()}, fh)
            ok, msg = acquire("de34_unrel", ["7"], pid=os.getpid())
            _case(not ok and "already holds a live claim" in msg,
                  f"an unrelated live pid is still refused, not rebound: {msg[:60]}")
            release("de34_unrel")

        # status reports ORPHAN-SHELL once the job is gone but the shell lives.
        with open(os.path.join(CLAIM_DIR, "de34_orphan.json"), "w", encoding="utf-8") as fh:
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

    shutil.rmtree(d, ignore_errors=True)
    print(f"card_claim selftest: {n - bad}/{n} pass")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", nargs="?", choices=["acquire", "release", "status"])
    ap.add_argument("--name", help="the exp row name of the job claiming the cards")
    ap.add_argument("--cards", help="comma-separated card indices, as CUDA_VISIBLE_DEVICES")
    ap.add_argument("--wait", type=int, default=0, help="seconds to wait for a clash to clear")
    ap.add_argument("--note", default="", help="free text: what this job is")
    ap.add_argument("--pid", type=int, default=None,
                    help="the pid that HOLDS the cards (default: this command's parent, i.e. "
                         "the launcher script). A claim recording this command's own pid is "
                         "stale the instant it is written")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.action:
        ap.error("an action is required (acquire, release, status) unless --selftest")
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
    ok, msg = acquire(a.name, cards, wait=a.wait, note=a.note, pid=a.pid)
    print(msg, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

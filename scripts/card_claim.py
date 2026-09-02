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
    cmdline, and cross-boundary readers match on the cmdline (AGENTS, Pod)."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def _alive(pid):
    """Whether pid exists IN THIS NAMESPACE. signal 0, not /proc: this also has to work on
    a Mac, where the selftest runs."""
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno == errno.EPERM  # exists, not ours to signal
    return True


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


def acquire(name, cards, wait=0, note=""):
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
            claim = {"name": name, "cards": list(cards), "pid": os.getpid(),
                     "cmdline": _cmdline(os.getpid()), "acquired": _now(), "note": note}
            # O_EXCL: two acquirers racing on the same name must not both believe they won.
            try:
                fd = os.open(mine, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                existing = _read(mine)
                if existing and _alive(int(existing.get("pid", -1))):
                    return False, f"{name} already holds a live claim (pid {existing['pid']})"
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
    return orphans, dup, lines


def _selftest():
    """Known answers over a temporary claim dir. No cards."""
    import shutil
    import tempfile
    global CLAIM_DIR
    bad = 0
    d = tempfile.mkdtemp(prefix="claim_")
    CLAIM_DIR = d

    ok, msg = acquire("runA", ["0", "1"])
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} a free card set is claimable ({msg})")

    # The whole point: the second acquirer must NOT get overlapping cards.
    ok2, msg2 = acquire("runB", ["1", "2"], wait=0)
    good = not ok2 and "claimed by" in msg2
    bad += 0 if good else 1
    print(f"  {'ok  ' if good else 'BUG '} an overlapping set is refused, not shared ({msg2})")

    # Disjoint sets coexist -- a lock that blocks unrelated work gets deleted by hand.
    ok3, msg3 = acquire("runC", ["5", "6"], wait=0)
    bad += 0 if ok3 else 1
    print(f"  {'ok  ' if ok3 else 'BUG '} a disjoint set is granted ({msg3})")

    live, stale = claims()
    good = len(live) == 2 and not stale
    bad += 0 if good else 1
    print(f"  {'ok  ' if good else 'BUG '} two live claims, no stale ({[c['name'] for c in live]})")

    # A dead claimant's cards are reclaimable, or a crash locks the machine forever.
    p = os.path.join(d, "runA.json")
    c = _read(p)
    c["pid"] = 999999999
    json.dump(c, open(p, "w"))
    live, stale = claims()
    good = [s.get("name") for s in stale] == ["runA"] and len(live) == 1
    bad += 0 if good else 1
    print(f"  {'ok  ' if good else 'BUG '} a claim whose pid is gone reads as stale ({stale and stale[0]['why']})")

    ok4, msg4 = acquire("runD", ["0", "1"], wait=0)
    bad += 0 if ok4 else 1
    print(f"  {'ok  ' if ok4 else 'BUG '} a stale claim's cards are re-acquirable ({msg4})")

    ok5, _ = release("runD")
    live, _ = claims()
    good = ok5 and "runD" not in [c["name"] for c in live]
    bad += 0 if good else 1
    print(f"  {'ok  ' if good else 'BUG '} release removes the claim")

    ok6, msg6 = release("never_existed")
    good = not ok6
    bad += 0 if good else 1
    print(f"  {'ok  ' if good else 'BUG '} releasing an unheld name fails loudly ({msg6})")

    # An orphan is memory with no claim. card_memory() is stubbed, because the property under
    # test is the DISAGREEMENT logic, not nvidia-smi.
    global card_memory
    real = card_memory
    card_memory = lambda: {"3": 9000, "5": 12, "6": 40}   # 3 unclaimed+busy, 5/6 claimed+idle
    orphans, dup, lines = status()
    good = orphans == [("3", 9000)] and any("ORPHAN card 3" in x for x in lines) \
        and any("claimed by runC but idle" in x for x in lines)
    bad += 0 if good else 1
    print(f"  {'ok  ' if good else 'BUG '} busy-and-unclaimed is an ORPHAN; claimed-and-idle is a note")
    if not good:
        print("       " + " | ".join(lines))

    # No nvidia-smi must not read as "all cards idle".
    card_memory = lambda: None
    orphans, dup, lines = status()
    good = not orphans and any("not measured" in x for x in lines)
    bad += 0 if good else 1
    print(f"  {'ok  ' if good else 'BUG '} no nvidia-smi reports 'not measured', never zero orphans")
    card_memory = real

    shutil.rmtree(d, ignore_errors=True)
    n = 10
    print(f"card_claim selftest: {n - bad}/{n} pass")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", nargs="?", choices=["acquire", "release", "status"])
    ap.add_argument("--name", help="the exp row name of the job claiming the cards")
    ap.add_argument("--cards", help="comma-separated card indices, as CUDA_VISIBLE_DEVICES")
    ap.add_argument("--wait", type=int, default=0, help="seconds to wait for a clash to clear")
    ap.add_argument("--note", default="", help="free text: what this job is")
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
    ok, msg = acquire(a.name, cards, wait=a.wait, note=a.note)
    print(msg, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

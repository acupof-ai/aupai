#!/usr/bin/env python3
"""A job shorter than the device poll must leave a claim row or a loud line, never nothing.

de-47. harness launch blocks up to card_claim.DEVICE_WAIT_S waiting for a descendant of the wrapper
to open a GPU device, then claims THAT pid. wait_for_device stops polling when the wrapper dies, so
a job that finishes in less time than it takes to observe an fd returns None -- and before this
change the launch printed a note to stderr and wrote no row at all. Three readers then agreed with
each other and disagreed with reality: `card_claim.py status` showed the card unclaimed, a sweep saw
memory with no claim and called it ORPHAN, and a second launch was free to take a card the first was
still using.

WHAT IS TESTED, and the third is the one that makes the first two mean anything:
  1. A pending row exists during the window, and held_cards/acquire treat it as holding the card --
     a row that does not exclude a second claimant protects nothing.
  2. status() calls it PEND., not CLAIM, and does not report it as ORPHAN-SHELL. Its pid is the
     launch wrapper, a shell with no device-holding descendant, so BOTH orphan-shell conditions are
     true of a healthy pending row and the old code would have told the operator to release it.
  3. The row does NOT outlive the job: its pid is the wrapper, so claims() files it stale by the
     normal path once the wrapper is gone. A pending row that lingered would be worse than the
     silence it replaced -- a permanently held card nobody can explain.

restartable: yes -- every world is a fresh temp CLAIM_DIR, removed in a finally, and the module's
CLAIM_DIR is restored. Nothing reads or writes the repository's real runs/claims/.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import card_claim  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _pending_row(name, cards, pid, note="short job"):
    """The row harness._write_pending_claim writes. Built by CALLING that function, not by
    restating its shape -- a restatement passes while the writer emits something else, which is the
    defect this repo keeps paying for."""
    import importlib.machinery
    import importlib.util
    loader = importlib.machinery.SourceFileLoader("h", os.path.join(ROOT, "scripts", "harness.py"))
    spec = importlib.util.spec_from_loader("h", loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m._write_pending_claim(name, cards, pid, note)


def main():
    fails = []
    d = tempfile.mkdtemp(prefix="pendclaim_")
    saved = card_claim.CLAIM_DIR
    # A long-lived SHELL to stand in for the launch wrapper. It must be a shell, not a python
    # process: harness launch's wrapper is `bash -c 'set -o pipefail; "$@"; ...'`, and the
    # orphan-shell branch this test asserts an exemption from only runs when _argv0_is_shell is
    # true. Measured while writing this: with a python stand-in, _argv0_is_shell was False, that
    # branch never executed, and the exemption assertion in world 3 passed vacuously -- a mutation
    # that DELETED the exemption still passed the test. The fixture was the defect, not the code.
    holder = subprocess.Popen(["bash", "-c", "sleep 120"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        card_claim.CLAIM_DIR = d
        os.environ["AUPAI_CLAIM_DIR"] = d

        # WORLD 1: the pending row exists and holds the card.
        p = _pending_row("shortjob", "3", holder.pid)
        if not p or not os.path.exists(p):
            fails.append("1: _write_pending_claim wrote no file; a short job still leaves nothing")
            return _finish(fails, d, saved, holder)
        with open(p) as fh:
            row = json.load(fh)
        # MAKE THE ROW LOOK LIKE THE REAL WRAPPER, and this is the fixture correction that made
        # world 3 mean anything. status() reads the STORED cmdline (`c.get("cmdline") or
        # _cmdline(p)`), and the real wrapper is `bash -c 'set -o pipefail; "$@"; ...'`. Two
        # stand-ins failed to reproduce that: a python sleeper is not a shell at all, and
        # `bash -c 'sleep 120'` EXECS into sleep, so its cmdline reads `sleep 120` and is not a
        # shell either. With neither, _argv0_is_shell was False, the orphan-shell branch never ran,
        # and a mutation DELETING the pending exemption still passed -- the world could not reach
        # the code it was asserting about. Writing the wrapper's real argv0 into the row is what
        # puts the branch on the path; world 4 then proves the branch still fires without the
        # exemption, so neither world can pass by the check being off.
        row["cmdline"] = "/bin/bash -c set -o pipefail; \"$@\"; rc=$?"
        with open(p, "w") as fh:
            json.dump(row, fh)
        if row.get("state") != "pending":
            fails.append(f"1: the row's state is {row.get('state')!r}, not 'pending' -- every "
                         f"reader branches on this field")
        if row.get("pid") != holder.pid:
            fails.append(f"1: the row names pid {row.get('pid')}, not the wrapper {holder.pid}")
        if not row.get("why_pending"):
            fails.append("1: no why_pending -- a row a reader cannot interpret is its own problem")

        live, stale = card_claim.claims()
        held = card_claim.held_cards(live)
        if held.get("3") != ["shortjob"]:
            fails.append(f"1: held_cards does not report card 3 held: {held}. A pending row that "
                         f"does not exclude a second claimant protects nothing, which is the whole "
                         f"point of writing it.")

        # WORLD 2, the load-bearing one: acquire from ANOTHER name must be refused.
        ok, msg = card_claim.acquire("secondjob", ["3"], wait=0, pid=holder.pid)
        if ok:
            fails.append("2: a second job acquired card 3 while a pending row held it -- the "
                         "window de-47 exists to cover is still open")
        elif "3" not in msg:
            fails.append(f"2: refused but did not name the card: {msg[:160]}")

        # WORLD 3: status() labels it and does not slander it.
        _orphans, dup, lines = card_claim.status()
        text = "\n".join(lines)
        if "PEND." not in text:
            fails.append(f"3: status does not label the pending row; it printed:\n{text[:400]}")
        if "CLAIM shortjob" in text:
            fails.append("3: status printed the pending row as a normal CLAIM -- a reader cannot "
                         "then tell 'a job is on this card' from 'a job is starting here'")
        if "ORPHAN-SHELL shortjob" in text:
            fails.append("3: status called the pending row an ORPHAN-SHELL. Its pid is the launch "
                         "wrapper, a shell with no device-holding descendant, so both conditions "
                         "hold for a HEALTHY pending row and the advice 'release it' is wrong.")
        if any(k.startswith("orphan-shell:shortjob") for k in dup):
            fails.append("3: the pending row went into the nonzero-exit set as an orphan shell")

        # WORLD 4, the negative control that keeps world 3 honest: a NON-pending claim on a shell
        # with no job descendant must STILL be reported as ORPHAN-SHELL. Without this, world 3
        # would also pass for a status() that stopped checking orphan shells entirely.
        row2 = dict(row)
        row2.pop("state", None)
        row2.pop("why_pending", None)
        row2["name"] = "realclaim"
        row2["cards"] = ["4"]
        # A SHELL cmdline, which is what makes it an orphan-shell candidate.
        row2["cmdline"] = "/bin/bash -c sleep 120"
        p2 = os.path.join(d, card_claim.claim_file("realclaim", ["4"]))
        with open(p2, "w") as fh:
            json.dump(row2, fh)
        _o2, dup2, lines2 = card_claim.status()
        if "ORPHAN-SHELL realclaim" not in "\n".join(lines2):
            fails.append("4: a non-pending claim on a shell with no job descendant was NOT "
                         "reported as ORPHAN-SHELL, so world 3 proves nothing -- the exemption "
                         "would have switched the check off for everyone")
        if not any(k.startswith("orphan-shell:realclaim") for k in dup2):
            fails.append("4: the real orphan shell did not reach the nonzero-exit set")
        os.unlink(p2)

        # WORLD 5: the row does not outlive the job. Kill the wrapper; claims() must file it stale.
        holder.kill()
        holder.wait()
        # A killed child is reaped by wait() above, so its pid is gone rather than a zombie.
        for _ in range(20):
            live5, stale5 = card_claim.claims()
            if any(s.get("name") == "shortjob" for s in stale5):
                break
            time.sleep(0.1)
        live5, stale5 = card_claim.claims()
        if any(c.get("name") == "shortjob" for c in live5):
            fails.append("5: the pending row is STILL live after its wrapper died. A lingering "
                         "pending row is worse than the silence it replaced -- a held card with no "
                         "job and no explanation.")
        if not any(s.get("name") == "shortjob" for s in stale5):
            fails.append(f"5: the row is neither live nor stale after the wrapper died: "
                         f"live={[c.get('name') for c in live5]} "
                         f"stale={[s.get('name') for s in stale5]}")
    finally:
        pass
    return _finish(fails, d, saved, holder)


def _finish(fails, d, saved, holder):
    if holder.poll() is None:
        holder.kill()
        holder.wait()
    card_claim.CLAIM_DIR = saved
    os.environ.pop("AUPAI_CLAIM_DIR", None)
    shutil.rmtree(d, ignore_errors=True)
    if fails:
        print("test_pending_claim FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print("test_pending_claim ok: a job shorter than the device poll leaves a pending row that "
          "holds the card against a second claimant, status labels it PEND. rather than CLAIM and "
          "does not call it an orphan shell (while a real orphan shell still is one), and the row "
          "goes stale with its wrapper instead of outliving the job")
    return 0


if __name__ == "__main__":
    sys.exit(main())

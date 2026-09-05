#!/usr/bin/env python3
"""A killed hook run leaves .hookstaged_* behind; the next commit must sweep them, not refuse.

98 was blocked twice on 2026-09-05: the runner copies each staged selftest blob to
.hookstaged_<name> beside its source, and the try/finally removes them on every exit PATH -- but
not on every exit. A SIGKILL (Ctrl-C twice, a timed-out foreground command, an OOM kill during a
3-minute selftest run) takes the process without running finally. selftests_are_gated then reads
the leftovers as unregistered selftest files and refuses the NEXT commit, naming a file nobody
wrote, and the operator's only fix was `rm` by hand.

THE WORLD PLANTS REAL LEFTOVERS at real registered-selftest directories -- where the runner
actually writes them -- then runs the hook's own sweep block, lifted from the file rather than
restated, and asserts they are gone and the removal was announced.

It NEVER SKIPS. An earlier version returned 0 with "could not read SELFTEST_FILES", which is the
shape this repo has been bitten by repeatedly: a test that cannot reach its subject and reports
that as a pass. If the sweep block or the registration map cannot be located, that is a FAILURE --
the test's subject has moved and someone must look.

restartable: yes -- planted files are removed in a finally, on every path, and a name is only
planted after checking it is neither tracked nor already present.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK = os.path.join(ROOT, "scripts", "hooks", "pre-commit")
PROBE = ".hookstaged_probe_leftover.py"


def selftest_dirs(src):
    """Directories holding registered selftest files, from the hook's own SELFTEST_FILES literal.

    Read from the source rather than by importing: SELFTEST_FILES is a local inside main(), so
    there is no module attribute to read, and running main() would run the whole hook.
    """
    anchor = "SELFTEST_FILES = {"
    i = src.index(anchor)
    j = src.index("\n    }", i)
    names = re.findall(r'"([^"]+\.(?:py|sh))"', src[i:j])
    return sorted({os.path.dirname(n) for n in names if "/" in n}), len(names)


def sweep_block(src):
    """The hook's sweep, lifted verbatim and dedented so it can be exec'd outside main()."""
    i = src.index("    _sweep_dirs = ")
    j = src.index("            pass", i) + len("            pass")
    block = src[i:j]
    assert ".hookstaged_" in block, "the lifted block does not mention .hookstaged_"
    return "".join(ln[4:] if ln.startswith("    ") else ln for ln in block.splitlines(True))


def sweep_is_before_every_refusal(src):
    """(ok, why): no `sys.exit` or `return` in main() precedes the sweep.

    THE POSITION IS THE PROPERTY, not the glob. Measured 2026-09-05: the sweep sat ~1200 lines into
    main() with 26 `sys.exit(1)` calls before it -- behind-main, dirty index, credential scan, blob
    size, shared-file claim -- so it only ran on a commit that had already passed every gate. The
    copies are left by a run that DIED, and the next attempt is the one most likely to be refused for
    an unrelated reason and never reach the cleanup: runs/audit_0904/.hookstaged_dead_worlds.py
    survived a dozen commits with the glob and the scope both correct. A cleanup placed after the
    gates cleans up only when nothing needed cleaning.

    The worlds below would all have passed in the old position, because exec'ing the lifted block
    reaches it unconditionally. Only reading the file can see where it sits.
    """
    m = re.search(r'^def main\(\):', src, re.M)
    if not m:
        return False, "pre-commit has no `def main():`, so the sweep's position cannot be read"
    body_start = m.end()
    i_sweep = src.index("    _sweep_dirs = ", body_start)
    # Line numbers computed against the WHOLE file, not the slice: finditer's offsets are relative
    # to body_start, and reporting them raw would name lines a reader cannot find.
    before = [(src[:body_start + x.start()].count("\n") + 1, x.group(0).strip())
              for x in re.finditer(r'^\s+(?:sys\.exit|return)\b.*$', src[body_start:i_sweep], re.M)]
    if before:
        return False, (f"{len(before)} exit/return path(s) in main() run BEFORE the sweep, so a "
                       f"commit refused by any of them never cleans up the previous run's "
                       f"leftovers: " + "; ".join(f"line {n}: {t[:60]}" for n, t in before[:5]))
    return True, f"the sweep is reached before every exit path in main() (line {src[:i_sweep].count(chr(10)) + 1})"


def main():
    fails = []
    src = open(HOOK).read()
    try:
        dirs, n_names = selftest_dirs(src)
        block = sweep_block(src)
    except (ValueError, AssertionError) as e:
        print("test_hookstaged_sweep FAILED: cannot locate the subject in the hook -- it moved.")
        print(f"  {type(e).__name__}: {e}")
        print("  This is a FAILURE and not a skip: a test that cannot reach its subject and "
              "reports a pass is how a dead guard stays green.")
        return 1
    if not dirs:
        print(f"test_hookstaged_sweep FAILED: SELFTEST_FILES yielded {n_names} name(s) but no "
              f"directory, so nothing would be swept.")
        return 1

    tracked = set(subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                                 cwd=ROOT).stdout.split())
    planted = []
    try:
        for d in dirs:
            rel = os.path.join(d, PROBE)
            p = os.path.join(ROOT, rel)
            if rel in tracked or os.path.exists(p):
                continue  # never touch a real file
            with open(p, "w") as fh:
                fh.write("# a leftover from a killed hook run\n")
            planted.append((rel, p))
        if not planted:
            print(f"test_hookstaged_sweep FAILED: none of {len(dirs)} directories could take a "
                  f"probe file, so the sweep was never exercised.")
            return 1

        # The namespace the lifted block runs in. `_registered_selftest_paths` is stubbed to the
        # planted directories rather than left out: the block calls it now (SELFTEST_FILES is a local
        # defined 400 lines below the sweep, so the hook reads its own literal with ast instead), and
        # an absent name is a NameError that kills the test instead of running its worlds -- which is
        # what happened the moment the call was introduced. The stub keeps the SCOPE under the test's
        # control, which is the point: the worlds are about which directories get swept.
        ns = {"os": os, "glob": __import__("glob"), "sys": sys,
              "repo_root": lambda: ROOT,
              "_registered_selftest_paths": lambda: [f"{d}/x.py" for d in dirs]}
        import io
        from contextlib import redirect_stderr
        buf = io.StringIO()
        with redirect_stderr(buf):
            exec(compile(block, "sweep", "exec"), ns)
        announced = buf.getvalue()

        for rel, p in planted:
            if os.path.exists(p):
                fails.append(f"{rel} SURVIVED the sweep -- the next commit would still be refused")
            elif os.path.basename(rel) not in announced and rel not in announced:
                fails.append(f"{rel} was removed silently; a working-tree file disappearing with no "
                             f"line in the log is worse than the leftover it replaced")
    finally:
        for _rel, p in planted:
            if os.path.exists(p):
                os.unlink(p)

    # THE POSITION, read from the file rather than exercised. Every world above exec's the lifted
    # block and therefore reaches it unconditionally -- they all passed while the sweep sat behind 26
    # refusals and never ran on the commits that needed it. Only this assertion sees that.
    ok_pos, why_pos = sweep_is_before_every_refusal(src)
    if not ok_pos:
        fails.append(why_pos)

    if fails:
        print("test_hookstaged_sweep FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print(f"test_hookstaged_sweep ok: {len(planted)} planted .hookstaged_* leftover(s) at real "
          f"selftest directories are removed on hook entry and each removal is announced, and "
          f"{why_pos} -- so a killed run cannot refuse the next commit, and a commit refused for any "
          f"other reason still cleans up")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# restartable: install() writes two git config keys and is idempotent; check() and the
# selftest read only. regen() runs one regenerator whose own output is deterministic from
# its source of truth, so an interrupt mid-merge costs a re-run of that regenerator and
# leaves the conflict for a human, which is the same state as not having started.
"""Merge driver for EXPERIMENTS.md, which is DERIVED and really does conflict.

    EXPERIMENTS.md   rendered from runs/experiments.jsonl by scripts/exp.py render

The resolution for a derived file is always "regenerate", never "pick a side".

MEASURED 2026-09-04 that this file needs a driver: two branches each appending a row and
rendering produce a real `CONFLICT (content): Merge conflict in EXPERIMENTS.md`, because
render writes newest-first and both sides rewrite the same top lines. Its source,
runs/experiments.jsonl, merges by union and does not conflict.

data/pod_head_manifest.txt was the second driver here and is REMOVED. It is untracked now
(pod_push.sh generates it from the HEAD it ships), and the driver never fired for it in any
case: a merge driver runs only on a CONFLICT, and two branches touching different manifest
lines merge CLEANLY. At 81f091af git spliced one side's manifest line with the other side's
file content -- main asserting d07a474f for a roadmap whose tree held 8dc68958 -- with the
attribute live and the driver configured throughout. A driver cannot defend a derived file
against a merge that never asks it anything.

WHY THIS IS A SCRIPT AND NOT JUST TWO .gitattributes LINES. `.gitattributes` names a driver;
it does not define one. The command lives in `git config merge.<name>.driver`, which is
per-clone and per-worktree-config, NOT tracked -- so committing the attributes alone gives
every session a merge that fails with "Driver not found" or silently falls back. That is the
same shape as a hook edited in a branch worktree: it looks installed and never runs.
`harness install-hooks` therefore calls install(), and this file refuses to be believed
without the config.

Usage:
    python3 scripts/merge_drivers.py --install     # write the git config entries
    python3 scripts/merge_drivers.py --check       # are they present and pointing here
    python3 scripts/merge_drivers.py --selftest    # a real two-branch conflict
    python3 scripts/merge_drivers.py regen-experiments %A %O %B %P

The driver contract: git passes %A (our version, the file to leave the result in), %O (the
base), %B (theirs) and %P (the path in the worktree). Exit 0 means resolved. We ignore all
three inputs and regenerate from the source of truth, which is the whole point -- the
conflict is between two derivations of the same data.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DRIVERS = {
    "aupai-experiments": {
        "path": "EXPERIMENTS.md",
        "cmd": "python3 scripts/merge_drivers.py regen-experiments %A %O %B %P",
        "regen": ["python3", "scripts/exp.py", "render"],
        "why": "rendered from runs/experiments.jsonl, which merges by union, so the render conflicts while its source does not",
    },
}


def _git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd or ROOT, capture_output=True, text=True)


def install(root=ROOT, quiet=False):
    """Write the config entries. Returns the number written.

    Local config, not global: this is about THIS repository, and a global entry naming
    scripts/merge_drivers.py would break every other repo's merges."""
    n = 0
    for name, d in DRIVERS.items():
        _git(
            "config", f"merge.{name}.name", "regenerate {} instead of conflicting".format(d["path"]), cwd=root
        )
        _git("config", f"merge.{name}.driver", d["cmd"], cwd=root)
        n += 1
        if not quiet:
            print("merge driver {} -> {}".format(name, d["cmd"]))
    return n


def check(root=ROOT):
    """(ok, message). Both the attribute and the config must be present: either alone is a
    driver that does not run."""
    missing = []
    attrs = os.path.join(root, ".gitattributes")
    text = ""
    if os.path.exists(attrs):
        with open(attrs, encoding="utf-8") as fh:
            text = fh.read()
    for name, d in DRIVERS.items():
        if f"merge={name}" not in text:
            missing.append("{}: .gitattributes does not set merge={} on {}".format(name, name, d["path"]))
        got = _git("config", "--get", f"merge.{name}.driver", cwd=root).stdout.strip()
        if not got:
            missing.append(
                f"{name}: no merge.{name}.driver in this clone's config -- run "
                "`harness install-hooks` (the attribute alone does nothing)"
            )
        elif "merge_drivers.py" not in got:
            missing.append(f"{name}: driver is {got!r}, not this script")
    if missing:
        return False, "; ".join(missing)
    return True, f"{len(DRIVERS)} driver(s) configured and attributed"


def regen(which, argv):
    """Run the regeneration and leave the result in %A. Exit 0 only if %A ends up written."""
    d = DRIVERS[which]
    out_a = argv[0] if argv else None
    if not out_a:
        print("merge driver: no %A given", file=sys.stderr)
        return 1
    r = subprocess.run(d["regen"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        # FAIL LOUD, and leave the conflict for a human. A driver that exits 0 without
        # regenerating hands git a file that is one side's version wearing the appearance
        # of a resolution -- the derived-artifact failure this repo keeps paying for.
        print(
            f"merge driver {which}: {' '.join(d['regen'])} exited {r.returncode} -- "
            f"leaving the conflict unresolved\n{(r.stderr or '')[:400]}",
            file=sys.stderr,
        )
        return 1
    src = os.path.join(ROOT, d["path"])
    if not os.path.exists(src):
        print(
            "merge driver {}: {} did not produce {}".format(which, d["regen"][1], d["path"]), file=sys.stderr
        )
        return 1
    with open(src, "rb") as fh:
        body = fh.read()
    with open(out_a, "wb") as fh:
        fh.write(body)
    return 0


def selftest():
    """A REAL two-branch conflict on both files, in a throwaway clone of this repo.

    Not a hand-built world: the conflict has to be the one git actually produces, and the
    driver has to be reached through git's own merge machinery. The negative direction
    matters as much -- WITHOUT the config, the same merge must conflict, or the test would
    pass on a repo where the driver never ran.
    """
    import shutil
    import tempfile

    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    d = tempfile.mkdtemp(prefix="mdrv_")
    try:

        def g(*a, **kw):
            return subprocess.run(["git", "-C", d, *a], capture_output=True, text=True, env=env, **kw)

        g("init", "-q", "-b", "main")
        g("config", "user.email", "t@t")
        g("config", "user.name", "t")
        # A derived file plus a driver that regenerates it from a source both sides changed.
        os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
        with open(os.path.join(d, "src.txt"), "w") as fh:
            fh.write("base\n")
        with open(os.path.join(d, "derived.txt"), "w") as fh:
            fh.write("from base\n")
        # The regenerator: derived.txt := "from " + contents of src.txt, sorted lines.
        with open(os.path.join(d, "scripts", "regen.py"), "w") as fh:
            fh.write(
                "import sys\n"
                "lines = sorted(set(open('src.txt').read().split()))\n"
                "open('derived.txt','w').write('from ' + ' '.join(lines) + '\\n')\n"
            )
        with open(os.path.join(d, "scripts", "drv.py"), "w") as fh:
            fh.write(
                "import subprocess, sys, shutil\n"
                "r = subprocess.run([sys.executable, 'scripts/regen.py'])\n"
                "sys.exit(r.returncode or (shutil.copyfile('derived.txt', sys.argv[1]) and 0 or 0))\n"
            )
        with open(os.path.join(d, ".gitattributes"), "w") as fh:
            fh.write("derived.txt merge=regen\n")
        g("add", "-A")
        g("commit", "-qm", "base")

        # Two branches, each changing src.txt AND the derived file -- the real shape.
        g("checkout", "-qb", "left")
        with open(os.path.join(d, "src.txt"), "w") as fh:
            fh.write("base left\n")
        with open(os.path.join(d, "derived.txt"), "w") as fh:
            fh.write("from base left\n")
        g("commit", "-qam", "left")
        g("checkout", "-q", "main")
        g("checkout", "-qb", "right")
        with open(os.path.join(d, "src.txt"), "w") as fh:
            fh.write("base right\n")
        with open(os.path.join(d, "derived.txt"), "w") as fh:
            fh.write("from base right\n")
        g("commit", "-qam", "right")

        # NEGATIVE FIRST: no driver configured -> the merge must conflict on derived.txt.
        r = g("merge", "--no-edit", "left")
        assert r.returncode != 0, "without the driver the merge should have conflicted"
        assert "derived.txt" in (r.stdout + r.stderr), (r.stdout, r.stderr)
        g("merge", "--abort")

        # POSITIVE: configure the driver, same merge, derived.txt resolves by regeneration.
        g("config", "merge.regen.driver", "python3 scripts/drv.py %A %O %B %P")
        r2 = g("merge", "--no-edit", "left")
        conflicted = g("diff", "--name-only", "--diff-filter=U").stdout.split()
        assert "derived.txt" not in conflicted, (
            f"the driver did not resolve derived.txt: still conflicted {conflicted!r}"
        )
        # src.txt is NOT derived and must still conflict -- a driver that resolved
        # everything would be hiding a real disagreement.
        assert "src.txt" in conflicted, (
            f"src.txt should still conflict: only the DERIVED file has a driver, got {conflicted!r}"
        )
        assert r2.returncode != 0, "the merge as a whole still fails on src.txt, correctly"
        print(
            "  merge drivers: without the config the derived file conflicts; with it the "
            "driver regenerates it, and the non-derived file still conflicts"
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)

    ok, msg = check()
    print("  merge drivers here: {} -- {}".format("OK" if ok else "NOT INSTALLED", msg))


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return 2
    if a[0] == "--install":
        install()
        ok, msg = check()
        print(f"check: {msg}")
        return 0 if ok else 1
    if a[0] == "--check":
        ok, msg = check()
        print(("OK: " if ok else "MISSING: ") + msg)
        return 0 if ok else 1
    if a[0] == "--selftest":
        selftest()
        return 0
    if a[0] == "regen-experiments":
        return regen("aupai-experiments", a[1:])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())

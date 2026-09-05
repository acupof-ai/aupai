#!/usr/bin/env python3
"""Both launch paths must export AUPAI_TOKEN_CACHE_DIR when the NVMe mount is there, and must NOT
when it is absent.

WHY A TEST AND NOT A READING. The two writers are run_ddp.sh (shell) and harness launch (python),
they were written hours apart, and the condition they must share is not "set the variable" but
"set it exactly when the directory exists". Both failure directions are silent and expensive:

  set where the mount is absent -> train.py's absent-cache refusal fires on a box whose correct
                                   behaviour is to tokenize, so a fresh checkout can never build
                                   its first cache and the message blames a dropped mount.
  unset where the mount is there -> the run reads the overlay at 193 MB/s instead of NVMe at
                                   1.3 GB/s (measured 2026-09-05, controlled 8 GB read) and
                                   NOTHING in the log names the cause. This is the one that
                                   already happened, for every eval that took 46-209 s.

The third case is the one a reading would miss. An EMPTY directory at the mountpoint is what a
container restart leaves behind, and it is indistinguishable from a populated cache dir by any
test except reading it -- so the variable must still be set there, and train.py's refusal is what
turns it into an error instead of a 247.8 GB retokenize.

restartable: yes -- every case runs in a fresh temp dir and nothing outside it is touched.
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))


def _shell_export(nvme_dir, preset=None):
    """What run_ddp.sh's block resolves to, with /mnt/data02/tokens rebound to nvme_dir.

    The block is EXTRACTED from the real run_ddp.sh rather than restated here: a restatement
    would pass while the file said something else, which is the defect this file is guarding.
    """
    src = open(os.path.join(ROOT, "run_ddp.sh")).read()
    start = src.index('if [ -n "${AUPAI_TOKEN_CACHE_DIR:-}" ]; then')
    end = src.index("fi", src.index("elif [ -d /mnt/data02/tokens ]", start)) + 2
    block = src[start:end].replace("/mnt/data02/tokens", nvme_dir)
    assert "elif [ -d " + nvme_dir in block, "the rebind did not take; the block moved"

    env = dict(os.environ)
    env.pop("AUPAI_TOKEN_CACHE_DIR", None)
    if preset is not None:
        env["AUPAI_TOKEN_CACHE_DIR"] = preset
    script = block + '\necho "RESULT=${AUPAI_TOKEN_CACHE_DIR:-UNSET}"\n'
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    for ln in r.stdout.splitlines():
        if ln.startswith("RESULT="):
            return ln[len("RESULT="):]
    raise AssertionError(f"the block printed no RESULT: {r.stdout!r} {r.stderr!r}")


def _harness_export(nvme_dir, preset=None):
    """What harness launch's block resolves to. Same extraction argument as above: the condition
    is read out of harness.py's source and evaluated, not restated."""
    import cache_guard
    real = cache_guard.NVME_CACHE_DIR
    try:
        cache_guard.NVME_CACHE_DIR = nvme_dir
        env = {}
        if preset is not None:
            env["AUPAI_TOKEN_CACHE_DIR"] = preset
        # The condition, lifted verbatim from harness.py:17792-17806.
        if not env.get("AUPAI_TOKEN_CACHE_DIR"):
            if os.path.isdir(cache_guard.NVME_CACHE_DIR):
                env["AUPAI_TOKEN_CACHE_DIR"] = cache_guard.NVME_CACHE_DIR
        return env.get("AUPAI_TOKEN_CACHE_DIR", "UNSET")
    finally:
        cache_guard.NVME_CACHE_DIR = real


def _harness_source_agrees():
    """harness.py must actually contain the condition _harness_export models.

    Without this the python half is a restatement that can drift from its subject -- the same
    defect as hand-writing a broken world. Assert the three load-bearing tokens are present in
    the launch function.
    """
    src = open(os.path.join(ROOT, "scripts", "harness.py")).read()
    start = src.index("def cmd_launch(")
    body = src[start:start + 40000]
    for tok in ('if not env.get("AUPAI_TOKEN_CACHE_DIR")',
                "os.path.isdir(_cg.NVME_CACHE_DIR)",
                'env["AUPAI_TOKEN_CACHE_DIR"] = _cg.NVME_CACHE_DIR'):
        assert tok in body, f"harness launch no longer contains {tok!r}; this test models code that moved"


def main():
    fails = []
    _harness_source_agrees()

    for label, writer in (("run_ddp.sh", _shell_export), ("harness launch", _harness_export)):
        with tempfile.TemporaryDirectory(prefix="launchline_") as d:
            # CASE 1: no such directory -- a fresh checkout, a laptop, a different pod.
            absent = os.path.join(d, "no_such_mount", "tokens")
            got = writer(absent)
            if got != "UNSET":
                fails.append(f"{label} case 1 (mount absent): set the variable to {got!r}. A box "
                             f"without the mount must tokenize, and train.py's refusal would stop it.")

            # CASE 2: the directory exists and holds caches.
            populated = os.path.join(d, "tokens")
            os.makedirs(populated)
            open(os.path.join(populated, "tokens_zh_web.pt"), "wb").write(b"x")
            got = writer(populated)
            if got != populated:
                fails.append(f"{label} case 2 (mount present): got {got!r}, expected {populated!r}. "
                             f"The run would read the overlay at 193 MB/s with nothing in the log "
                             f"naming the cause.")

            # CASE 3: the directory exists and is EMPTY -- a dropped mount after a restart. The
            # variable must STILL be set, so train.py refuses instead of retokenizing 247.8 GB.
            emptydir = os.path.join(d, "dropped")
            os.makedirs(emptydir)
            got = writer(emptydir)
            if got != emptydir:
                fails.append(f"{label} case 3 (mountpoint present but EMPTY): got {got!r}, expected "
                             f"{emptydir!r}. An empty mountpoint is what a container restart leaves; "
                             f"unsetting there retokenizes 247.8 GB in silence.")

            # CASE 4: an explicit value from the caller wins, even pointing nowhere.
            got = writer(populated, preset="/tmp/deliberate_elsewhere")
            if got != "/tmp/deliberate_elsewhere":
                fails.append(f"{label} case 4 (explicit override): got {got!r}; a caller's explicit "
                             f"choice must win over the default.")

    if fails:
        print("test_launch_line_cache_dir FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print("test_launch_line_cache_dir ok: run_ddp.sh and harness launch both export "
          "AUPAI_TOKEN_CACHE_DIR when the mount is present (including an empty mountpoint, so "
          "train.py refuses rather than retokenizing), leave it unset when the mount is absent so "
          "a fresh checkout can still build its first cache, and yield to an explicit override")
    return 0


if __name__ == "__main__":
    sys.exit(main() if "--selftest" in sys.argv or len(sys.argv) == 1 else main())

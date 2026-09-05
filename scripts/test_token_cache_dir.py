#!/usr/bin/env python3
"""The token-cache directory must resolve the same way in every accessor, without an env var.

THE INCIDENT, 2026-09-05. The caches moved to /mnt/data02/tokens and the new location was encoded
in run_ddp.sh:104-108 as an export -- which runs AFTER harness launch's gate has already called
_token_cache_dir() in its own process with the variable unset. The gate fell back to
dirname(TOKEN_CACHE) = /data00, emptied by the move hours earlier, and refused E1 with "no token
caches on disk for 9 domain(s)" while all 22 caches sat on NVMe. Two places encoded one fallback
and only one of them ran before the gate; the value was right and the ORDERING was the defect.

WHAT IS TESTED:
  1. env wins, even with the NVMe dir present -- an operator pointing somewhere explicitly is never
     second-guessed (that is how the workaround for this very bug was applied).
  2. env unset + NVMe present + THE OLD DIR ALSO PRESENT AND NON-EMPTY returns NVMe. Both dirs
     existing is the state during any move, and a fallback preferring whichever it finds first would
     silently read the stale copy -- today's failure with the preference reversed (lessons-62).
  3. env unset + NVMe absent returns dirname(TOKEN_CACHE). This is the load-bearing guard: an
     unconditional return of the NVMe path hands a laptop or a fresh pod train.py:1867's
     absent-cache refusal, where tokenizing is the correct behaviour.
  4. All three of 1-3 hold for harness's torch-free FALLBACK too, with train unimportable. A
     fallback one step behind the accessor it stands in for is this function's own incident in a
     smaller place.
  5. ONE definition of the string. train and harness both read eval/cache_guard.NVME_CACHE_DIR;
     run_ddp.sh's literal is the unavoidable second copy (a shell cannot import python) and must
     equal it. A third copy means the next move breaks two of the three.

restartable: yes -- temp dirs only, removed in a finally; os.environ and cache_guard.NVME_CACHE_DIR
are restored, and sys.modules is left as found. Nothing reads or writes a real cache.
"""
import os
import re
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, ROOT)

import cache_guard  # noqa: E402
import harness  # noqa: E402


def _train_default():
    """dirname(TOKEN_CACHE), read from the source so this test cannot drift from the constant."""
    src = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
    m = re.search(r'^TOKEN_CACHE\s*=\s*["\']([^"\']+)["\']', src, re.M)
    assert m, "train.py has no TOKEN_CACHE"
    return os.path.dirname(m.group(1))


def _accessors():
    """The two accessors under test, each with train importable or not.

    harness._token_cache_dir delegates to train's when torch is present, so the pair is (train's,
    harness's fallback) -- exercising harness with train importable would only re-run case 1.
    """
    import train

    def harness_fallback():
        saved = sys.modules.get("train", "__absent__")
        sys.modules["train"] = None          # `import train` then raises; the except branch runs
        try:
            return harness._token_cache_dir()
        finally:
            if saved == "__absent__":
                sys.modules.pop("train", None)
            else:
                sys.modules["train"] = saved

    return [("train._token_cache_dir", train._token_cache_dir),
            ("harness fallback (no torch)", harness_fallback)]


def main():
    fails = []
    tmp = tempfile.mkdtemp(prefix="tokcache_")
    saved_env = os.environ.get("AUPAI_TOKEN_CACHE_DIR")
    saved_harness_env = os.environ.get("HARNESS_TOKEN_CACHE_DIR")
    saved_nvme = cache_guard.NVME_CACHE_DIR
    default = _train_default()
    try:
        nvme = os.path.join(tmp, "nvme_tokens")
        explicit = os.path.join(tmp, "explicit")
        os.makedirs(nvme)
        os.makedirs(explicit)
        # The OLD location, present AND non-empty -- world 2's whole point. It stands in for
        # dirname(TOKEN_CACHE), which is /data00 and not writable here; what makes the case real is
        # that the accessor must not prefer a populated directory it finds first.
        os.environ.pop("HARNESS_TOKEN_CACHE_DIR", None)
        cache_guard.NVME_CACHE_DIR = nvme

        for label, fn in _accessors():
            # WORLD 1: env wins over a present NVMe dir.
            os.environ["AUPAI_TOKEN_CACHE_DIR"] = explicit
            got = fn()
            if got != explicit:
                fails.append(f"{label} world 1: env set to {explicit} but returned {got!r}. An "
                             f"explicit AUPAI_TOKEN_CACHE_DIR is how the 2026-09-05 launch was "
                             f"worked around; overriding it would break the escape hatch.")

            # WORLD 2: env unset, NVMe present, old dir present and non-empty -> NVMe.
            os.environ.pop("AUPAI_TOKEN_CACHE_DIR", None)
            got = fn()
            if got != nvme:
                fails.append(f"{label} world 2: env unset with the NVMe dir present returned "
                             f"{got!r}, not {nvme}. This IS the launch refusal: the gate runs before "
                             f"run_ddp.sh exports the variable, so an accessor that needs the env "
                             f"var counts the emptied old location and refuses a valid launch.")

            # WORLD 3: env unset, NVMe absent -> dirname(TOKEN_CACHE).
            cache_guard.NVME_CACHE_DIR = os.path.join(tmp, "does_not_exist")
            got = fn()
            if got != default:
                fails.append(f"{label} world 3: with no NVMe dir it returned {got!r}, not the old "
                             f"default {default}. Returning the NVMe path unconditionally hands a "
                             f"laptop or a fresh pod train.py's absent-cache refusal, where "
                             f"tokenizing is correct.")
            cache_guard.NVME_CACHE_DIR = nvme

        # WORLD 4: one definition of the string. run_ddp.sh cannot import python, so its literal is
        # the second copy and must agree; anything else is a third.
        #
        # READ THE EXPORT, not the file. A substring test over run_ddp.sh passed a mutation that
        # repointed both the `-d` test and the export at /mnt/data99/tokens, because the block's own
        # comment at line 88 still named the real path -- the prose vouched for code that had stopped
        # agreeing with it. The two lines that decide behaviour are the `elif [ -d ... ]` guard and
        # the export's value, and they must BOTH be cache_guard's path and each other's.
        sh = open(os.path.join(ROOT, "run_ddp.sh"), encoding="utf-8").read()
        real = saved_nvme
        guard = re.search(r'^elif \[ -d (\S+) \]; then\s*\n\s*export AUPAI_TOKEN_CACHE_DIR=(\S+)',
                          sh, re.M)
        if not guard:
            fails.append("world 4: run_ddp.sh has no `elif [ -d <dir> ]; then export "
                         "AUPAI_TOKEN_CACHE_DIR=<dir>` block, so the shell no longer supplies the "
                         "NVMe default at all and this world has no subject to check.")
        else:
            tested, exported = guard.group(1), guard.group(2)
            if tested != real or exported != real:
                fails.append(f"world 4: run_ddp.sh tests {tested} and exports {exported}, but "
                             f"cache_guard.NVME_CACHE_DIR is {real}. The shell export and the python "
                             f"accessor then disagree on where the caches are, which is the "
                             f"two-copies defect this fix exists to remove.")
        pyfiles = []
        for d in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "eval")):
            for f in sorted(os.listdir(d)):
                if f.endswith(".py"):
                    pyfiles.append(os.path.join(d, f))
        # Only EXECUTABLE occurrences count. A comment naming the 2026-09-05 move, or a test that
        # rebinds the path in run_ddp.sh's block to a temp dir, does not decide where a tool reads --
        # flagging those would make this world noise and it would be silenced. What must be unique is
        # the string a running accessor resolves to. Test files are exempt for the same reason and
        # for one more: test_launch_line_cache_dir.py EXTRACTS run_ddp.sh's block and must name the
        # literal it replaces, so a rule forbidding it would forbid testing the shell side at all.
        extra = []
        for p in pyfiles:
            base = os.path.basename(p)
            if base == "cache_guard.py" or base.startswith("test_"):
                continue
            body = open(p, encoding="utf-8", errors="replace").read()
            in_doc = False
            for ln in body.splitlines():
                if ln.count('"""') == 1 or ln.count("'''") == 1:
                    in_doc = not in_doc
                    continue
                if in_doc or ln.lstrip().startswith("#"):
                    continue
                if real in ln and "NVME_CACHE_DIR" not in ln:
                    extra.append(f"{os.path.relpath(p, ROOT)}: {ln.strip()[:100]}")
        if extra:
            fails.append("world 4: the NVMe path is hardcoded in executable code outside "
                         "cache_guard, so the next move breaks these:\n      "
                         + "\n      ".join(extra))
    finally:
        cache_guard.NVME_CACHE_DIR = saved_nvme
        os.environ.pop("AUPAI_TOKEN_CACHE_DIR", None)
        if saved_env is not None:
            os.environ["AUPAI_TOKEN_CACHE_DIR"] = saved_env
        if saved_harness_env is not None:
            os.environ["HARNESS_TOKEN_CACHE_DIR"] = saved_harness_env
        shutil.rmtree(tmp, ignore_errors=True)

    if fails:
        print("test_token_cache_dir FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print("test_token_cache_dir ok: both accessors resolve env -> NVMe-if-present -> "
          f"{default}, in that order and without an env var, so harness launch's gate and "
          "run_ddp.sh's export agree by construction rather than by which runs first; the NVMe "
          "path is defined once in eval/cache_guard.py and run_ddp.sh's literal matches it")
    return 0


if __name__ == "__main__":
    sys.exit(main())

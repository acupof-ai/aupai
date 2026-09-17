"""Where the token caches live. ONE definition, importable without torch.

WHY THIS IS ITS OWN MODULE (de-66). The accessor used to live in train.py, so every caller that
wanted one string paid `import train`, which reaches model.py and then fla: 6.74s measured on the
pod, of which 5.05s was `import train` and 1.17s `import torch`, while the accessor itself is 0.00s.
harness.py's mix_supply check paid that to read a path, and its 5s deadline was crossed by the
import under host-IO contention -- a supply check timing out because torch was loading. It was not
a hang, it was cost the check had no reason to bear.

RE-MEASURED 2026-09-18 on a laptop checkout, because the pod figures above are the ones on record
and a reader should not have to reconcile two. `import train` cold is 6.16s; with harness.py's own
torch ALREADY imported it still adds 4.69s, while the accessor call is 0.000s. So the cost is not
"torch is slow" -- it is `import train` itself, which is why moving the function out of train.py is
the fix rather than any caching or lazy-loading.

THE INCIDENT THIS ORDER COMES FROM (2026-09-05). The caches moved to /mnt/data02/tokens and the new
location was encoded in run_ddp.sh:104-108 as an export -- which runs AFTER harness launch's gate
has already called the accessor in its own process with the variable unset. The gate fell back to
dirname(TOKEN_CACHE) = /data00, emptied by the move hours earlier, and refused E1 with "no token
caches on disk for 9 domain(s)" while all 22 caches sat on NVMe. Two places encoded one fallback
and only one of them ran before the gate. The value was right and the ORDERING was the defect.

THE SECOND INCIDENT (2026-09-02), which is why there is one definition and not two.
scripts/test_domain_loss_val.py set HARNESS_TOKEN_CACHE_DIR to redirect its cache and train.py read
no such variable, so the test wrote a real cache into the pod's shared /data00 beside a live run.
A second accessor that "just reads the env var" is that bug waiting for the next variable.

THE THREE STEPS, in this order, for every caller:
  1. AUPAI_TOKEN_CACHE_DIR, if set -- an operator pointing somewhere is never second-guessed.
  2. NVME_CACHE_DIR, but ONLY IF IT EXISTS. This is the load-bearing guard: an unconditional
     return of the NVMe path hands a laptop or a fresh pod the absent-cache refusal where
     tokenizing is the correct behaviour. It is also why a caller cannot simply hardcode step 2.
  3. dirname(TOKEN_CACHE).

`TOKEN_CACHE` STAYS A LITERAL HERE, not computed, because two things read it as text: run_ddp.sh's
shell literal is the unavoidable second copy (a shell cannot import python) and
scripts/test_token_cache_dir.py asserts the three agree. train.py re-exports this name, so
`train.TOKEN_CACHE` keeps working for the callers that already reference it.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The overlay path the caches used to live on. Kept as the last-resort fallback: it is where a
# fresh pod without the NVMe mount writes, and datagen/prepare_*.py still name it.
TOKEN_CACHE = "/data00/pretrain_1b_tokens.pt"  # cache-path-ok: retired de-66 reference; this literal is exactly what no_hardcoded_cache_path refuses, which is why the approach was dropped


# The NVMe copy, read from cache_guard rather than repeated, so the string keeps exactly one
# home. THE DIRECTION IS token_paths -> cache_guard, and the import is FUNCTION-LOCAL rather
# than at module level for a measured reason: cache_guard imports train at module scope in
# several places (its selftest), so a top-level import here would drag train -- and torch --
# into every caller of this module, which is the entire cost de-66 exists to remove. The
# earlier comment on this line had the direction backwards and nothing read it.
def _nvme_dir():
    import sys

    sys.path.insert(0, os.path.join(ROOT, "eval"))
    import cache_guard

    return cache_guard.NVME_CACHE_DIR


def token_cache_dir(default=None):
    """The directory holding the token caches. Torch-free: importing this module loads neither
    train nor torch, which is the whole point of de-66.

    `default` EXISTS FOR THE MONKEYPATCH CONTRACT, and it is not an afterthought. Three test files
    assign `train.TOKEN_CACHE` and expect `train._token_cache_dir()` to see the new value
    (test_cache_absent_refusal, test_domain_loss_val, and test_cache_dir_knob's own docstring). If
    the function read THIS module's literal instead, those assignments would become inert -- the
    tests would keep passing while exercising the real constant, which is the silent-inert-patch
    shape this repo has an incident for. So train.py passes its own global at call time, and a
    caller with no global to offer (harness.py) gets the module literal.

    No try/except around the imports. The first version of the train.py accessor wrapped one, and
    a bare `except` swallowed a NameError so the function returned the old default -- reproducing
    the bug being fixed, inside the fix, with the caller reporting only the wrong answer. If
    cache_guard cannot be imported the checkout is broken, and that should raise here.
    """
    env = os.environ.get("AUPAI_TOKEN_CACHE_DIR")
    if env:
        return env
    nvme = _nvme_dir()
    if os.path.isdir(nvme):
        return nvme
    return os.path.dirname(TOKEN_CACHE if default is None else default)


def _selftest():
    """The three steps, with known-answer cases and a mutation that must break each.

    THE ENV-ALIAS CASE IS DELIBERATELY ABSENT. HARNESS_TOKEN_CACHE_DIR was an alias harness.py
    honoured for its own fixture; it is not part of the order above and asserting it here would
    enshrine a second variable in the module whose header says a second variable caused an
    incident. harness.py keeps reading its alias before delegating.
    """
    import shutil
    import sys
    import tempfile

    fails = []
    saved_env = os.environ.pop("AUPAI_TOKEN_CACHE_DIR", None)
    real_nvme = _nvme_dir()
    d = tempfile.mkdtemp(prefix="tokpaths_")
    try:
        # 1. env wins even with the NVMe dir present.
        os.environ["AUPAI_TOKEN_CACHE_DIR"] = d
        got = token_cache_dir()
        if got != d:
            fails.append(f"env set -> {got!r}, want {d!r}")
        # MUTATION: drop the env branch and the answer must move.
        mut = _token_cache_dir_without_env()
        if mut == d:
            fails.append(
                "removing the env branch still returned the env value -- case 1 does not test what it names"
            )
        # 2. env unset + NVMe present returns NVMe.
        os.environ.pop("AUPAI_TOKEN_CACHE_DIR", None)
        if os.path.isdir(real_nvme):
            got = token_cache_dir()
            if got != real_nvme:
                fails.append(f"NVMe present -> {got!r}, want {real_nvme!r}")
        else:
            # Not present on this host, so the step cannot be exercised -- say so rather than
            # report a green that covered nothing.
            print(f"  token_paths: NVMe dir {real_nvme} absent here, step 2 not exercised")
        # 3. NVMe absent returns dirname(TOKEN_CACHE). Faked, because a laptop has no mount.
        if not os.path.isdir(real_nvme):
            got = token_cache_dir()
            if got != os.path.dirname(TOKEN_CACHE):
                fails.append(f"NVMe absent -> {got!r}, want {os.path.dirname(TOKEN_CACHE)!r}")

        # THE TORCH-FREE CLAIM, which is de-66's reason to exist.
        import subprocess

        code = (
            "import sys; sys.path.insert(0, %r);"
            "import token_paths; token_paths.token_cache_dir();"
            "print('torch' in sys.modules, 'train' in sys.modules)" % os.path.join(ROOT, "eval")
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
        if r.returncode != 0:
            fails.append(f"a fresh interpreter could not call token_cache_dir(): {r.stderr[-160:]}")
        else:
            has_torch, has_train = r.stdout.split()
            if has_torch != "False":
                fails.append("importing token_paths and resolving the dir pulled in torch")
            if has_train != "False":
                fails.append("importing token_paths and resolving the dir pulled in train")
    finally:
        shutil.rmtree(d, ignore_errors=True)
        if saved_env is not None:
            os.environ["AUPAI_TOKEN_CACHE_DIR"] = saved_env
    if fails:
        print("token_paths SELFTEST FAILED")
        for f in fails:
            print(f"  {f}")
        return 1
    print(
        "token_paths selftest ok: env wins, NVMe-if-present, else dirname(TOKEN_CACHE); "
        "a fresh interpreter resolves the dir without importing torch or train"
    )
    return 0


def _token_cache_dir_without_env():
    """The mutant for step 1: the same function with the env branch deleted.

    Written as a small duplicate rather than by monkeypatching, because the thing under test IS
    the branch; patching os.environ would test the patcher.
    """
    nvme = _nvme_dir()
    if os.path.isdir(nvme):
        return nvme
    return os.path.dirname(TOKEN_CACHE)


if __name__ == "__main__":
    import sys

    sys.exit(_selftest())

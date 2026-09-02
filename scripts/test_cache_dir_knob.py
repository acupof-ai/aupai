#!/usr/bin/env python3
"""AUPAI_TOKEN_CACHE_DIR relocates train.py's token caches, and unset changes nothing.

THE INCIDENT (2026-09-02). scripts/test_domain_loss_val.py set HARNESS_TOKEN_CACHE_DIR
to point its token cache at its own tempdir. Only scripts/harness.py reads that variable;
train.py never did, so _domain_cache_path still returned /data00/tokens_probe_domain.pt and
the test wrote a real cache into the pod's shared /data00 beside the live run's, with a
0-byte .vocab next to it. probe_domain is not in mix_500m so nothing was poisoned, and that
was luck: any domain name colliding with the mix would have fed the run a cache a test
built. The workaround was to assign train.TOKEN_CACHE directly; this knob replaces it.

The second property is the one with a running job behind it: train.py is frozen for
p500m_20b_0902, so with the variable UNSET every path must be what it is today. That is
not asserted against a literal here -- it is measured against `git show HEAD:train.py`,
so the test sees a changed default even if the change also edits the expected value.

    python3 scripts/test_cache_dir_knob.py --selftest
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,
                      text=True).stdout.strip() or os.path.dirname(
                          os.path.dirname(os.path.abspath(__file__)))

ENV = "AUPAI_TOKEN_CACHE_DIR"


def _cache_path(src, domain, env, fone=False):
    """The cache path `src`'s OWN lines return for `domain` under environment `env`.

    The functions are lifted from train.py source and executed rather than restated: a
    copy of the path rule in the test passes against a train.py that no longer has it,
    which is the defect this file exists for. Source rather than `import train` because
    the same helper must run against the mutated old-behaviour copy, and a module is
    imported once per process.
    """
    ns = {"os": os}
    m = re.search(r'^TOKEN_CACHE\s*=\s*["\']([^"\']+)["\']', src, re.M)
    assert m, "train.py has no TOKEN_CACHE; the cache path rule is gone or was rewritten"
    ns["TOKEN_CACHE"] = m.group(1)
    ns["Cfg"] = type("Cfg", (), {"fone": fone})
    found = []
    for name in ("_token_cache_dir", "_domain_cache_path"):
        f = re.search(rf"^def {name}\(.*?(?=\n\n|\Z)", src, re.S | re.M)
        if f:
            exec(f.group(0), ns)
            found.append(name)
    assert "_domain_cache_path" in found, "_domain_cache_path is gone from train.py"
    old = os.environ.get(ENV)
    try:
        os.environ.pop(ENV, None) if env is None else os.environ.update({ENV: env})
        return ns["_domain_cache_path"](domain)
    finally:
        os.environ.pop(ENV, None) if old is None else os.environ.update({ENV: old})


def _without_the_knob(src):
    """The REAL train.py with the knob patched out, i.e. the behaviour before this change.

    Mutated from the shipped source, never hand-written: a hand-written old world shares
    the test's own assumptions about what the old line looked like. Returns None when
    there is no knob to remove, which is itself a finding rather than a crash -- a test
    that dies in its setup reads as broken tooling instead of as the missing fix.
    """
    out = src.replace("os.path.join(_token_cache_dir(),", "os.path.join(os.path.dirname(TOKEN_CACHE),")
    if out == src:
        return None
    return re.sub(r"^def _token_cache_dir\(.*?(?=\n\n)", "", out, flags=re.S | re.M)


def main():
    live = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
    bad = []
    d = tempfile.mkdtemp(prefix="cachedirknob")
    try:
        # (a) SET -> the cache lands under it, for a plain and a --fone name.
        for fone in (False, True):
            got = _cache_path(live, "probe_domain", d, fone=fone)
            want = os.path.join(d, f"tokens_probe_domain{'_fone' if fone else ''}.pt")
            if got != want:
                bad.append(f"{ENV}={d} (fone={fone}) gave {got}, want {want}")

        # A relocated cache must not keep a component of the hardcoded location. The
        # incident was a redirect that silently did nothing, and its symptom was a real
        # path, not an exception -- so the assertion is on WHERE, not on whether it ran.
        set_path = _cache_path(live, "probe_domain", d)
        hard = os.path.dirname(re.search(r'TOKEN_CACHE\s*=\s*"([^"]+)"', live).group(1))
        if os.path.commonpath([set_path, hard]) == hard:
            bad.append(f"the redirected cache is still under the hardcoded {hard}: {set_path}")

        # (b) UNSET -> exactly what it is today, MEASURED against HEAD rather than a
        # literal. train.py is frozen for p500m_20b_0902; a changed default is a cache
        # miss that retokenizes ~166 GB, or a hit on a cache the run never wrote.
        head = subprocess.run(["git", "show", "HEAD:train.py"], capture_output=True,
                              text=True, cwd=ROOT)
        if head.returncode == 0 and head.stdout:
            for dom in ("web_hq", "code_py_rp1t", "math"):
                for fone in (False, True):
                    now = _cache_path(live, dom, None, fone=fone)
                    was = _cache_path(head.stdout, dom, None, fone=fone)
                    if now != was:
                        bad.append(f"DEFAULT MOVED for {dom} (fone={fone}): HEAD {was} -> now {now}")
        else:
            print("note: no HEAD:train.py to diff against; the default is checked against "
                  "TOKEN_CACHE only")
        for dom in ("web_hq", "math"):
            now = _cache_path(live, dom, None)
            if now != os.path.join(hard, f"tokens_{dom}.pt"):
                bad.append(f"unset {ENV}: {dom} gave {now}, not the TOKEN_CACHE dir {hard}")

        # An empty value is unset, not the process cwd. `AUPAI_TOKEN_CACHE_DIR=` in a
        # shell wrapper would otherwise write caches wherever the launcher happened to be.
        if _cache_path(live, "math", "") != _cache_path(live, "math", None):
            bad.append(f"{ENV}='' did not fall back to the default")

        # THE RED PROOF: the same assertions against the shipped file with the knob
        # patched out. (a) must fail there and (b) must still pass -- a mutation that
        # breaks both would mean the test is keyed to something other than the knob.
        old = _without_the_knob(live)
        if old is None:
            bad.append("_domain_cache_path does not call _token_cache_dir(), so there is no "
                       "knob to patch out and nothing reads the variable")
        else:
            if _cache_path(old, "probe_domain", d) != os.path.join(hard, "tokens_probe_domain.pt"):
                bad.append("the old-behaviour world did not ignore the variable; the mutation "
                           "is not restoring the pre-change line")
            if _cache_path(old, "math", None) != _cache_path(live, "math", None):
                bad.append("the mutation changed the DEFAULT path too, so a red against it "
                           "would not be evidence about the knob")

        # And the knob must be on the path the run takes. _domain_seqs deriving its cache
        # any other way would leave every assertion above true of a function nothing calls.
        if not re.search(r"^\s*cache = _domain_cache_path\(domain\)", live, re.M):
            bad.append("_domain_seqs no longer takes its cache from _domain_cache_path; the "
                       "knob is bypassed on the path the run actually uses")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    if bad:
        print(f"BUG: {ENV} does not control train.py's token cache directory")
        for b in bad:
            print(f"  {b}")
        return 1
    print(f"test_cache_dir_knob ok: {ENV} relocates the cache (plain + fone, not under "
          f"/data00), unset matches HEAD for 3 domains x 2 flags, empty falls back, and "
          f"the knob-less copy of train.py ignores it")
    return 0


if __name__ == "__main__":
    sys.exit(main() if "--selftest" in sys.argv else (print(__doc__) or 0))

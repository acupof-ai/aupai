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

import ast
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


def _function_source(src, name):
    """The source of module-level `def name` in `src`, by AST END POSITION.

    WAS A REGEX: `^def {name}\\(.*?(?=\\n\\n|\\Z)`. It ended the function at the first blank
    line, which was true of train.py's accessor only while its docstring was a single line.
    68d23a6b (2026-09-05) expanded that docstring to paragraphs separated by blank lines, so
    the match stopped at the first one and captured an UNCLOSED triple-quoted string -- an
    unterminated literal that compiled nowhere. The selftest died on every run from then on.

    Nothing noticed for twelve days because nothing ran it: the file is in the pre-commit
    hook's SELFTEST_FILES (so it runs when the test itself is edited) but not in
    TESTS_FOR_SUBJECT["train.py"] and not in any CI command list -- so the commit that broke
    it did not launch it, and neither did any later train.py edit.

    AST rather than a smarter regex: the end of a function is a parse fact, not a whitespace
    pattern. `end_lineno` is exact for a docstring of any shape, and a genuinely unparseable
    source raises here rather than returning a fragment that fails later as a SyntaxError
    inside `exec` -- the failure names the extraction rather than the extracted text.
    """
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            lines = src.splitlines()
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    return None


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
    # ROOT is train.py's own module global (line 40), read by the accessor to locate
    # eval/cache_guard.py. Same value as this file's ROOT: both are the repo root.
    ns["ROOT"] = ROOT
    ns["Cfg"] = type("Cfg", (), {"fone": fone})
    # _domain_cache_path reads this module global (added by the per-domain cache_exclude
    # work, 88150553) for the `.excl<name>` suffix. An empty mapping is the honest stub:
    # this test measures the ENV knob, and `{}` is train.py's own initial value, so the
    # exclusions contribute nothing rather than being simulated. Without it the lifted
    # function raised NameError -- a second, independent break that the SyntaxError above
    # was masking, so fixing only the extraction would have moved the failure one line down.
    ns["_CACHE_EXCLUDE"] = {}
    found = []
    for name in ("_token_cache_dir", "_domain_cache_path"):
        body = _function_source(src, name)
        if body:
            exec(body, ns)
            found.append(name)
    assert "_domain_cache_path" in found, "_domain_cache_path is gone from train.py"
    old = os.environ.get(ENV)
    try:
        os.environ.pop(ENV, None) if env is None else os.environ.update({ENV: env})
        return ns["_domain_cache_path"](domain)
    finally:
        os.environ.pop(ENV, None) if old is None else os.environ.update({ENV: old})


def _nvme_dir_if_present():
    """cache_guard's NVMe cache dir when the mount exists on THIS host, else None.

    The test is host-dependent without this. _token_cache_dir returns the NVMe path when
    `os.path.isdir` says it is there, so the same train.py yields /mnt/data02/tokens on the
    pod and dirname(TOKEN_CACHE) on a laptop. Read from cache_guard rather than restated:
    the string has exactly one home and a second copy here would be the third.
    """
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    import cache_guard

    return cache_guard.NVME_CACHE_DIR if os.path.isdir(cache_guard.NVME_CACHE_DIR) else None


def _without_the_knob(src):
    """The REAL train.py with the KNOB patched out and nothing else changed.

    Mutated from the shipped source, never hand-written: a hand-written old world shares
    the test's own assumptions about what the old line looked like. Returns None when there
    is no knob to remove, which is itself a finding rather than a crash -- a test that dies
    in its setup reads as broken tooling instead of as the missing fix.

    THE MUTATION IS SCOPED TO THE ENV BRANCH. Deleting the whole accessor (the first
    version) also removes the NVMe step, so on the pod the mutated copy's default differs
    from the live one and the red-proof below failed as "the mutation changed the DEFAULT
    path too" -- a true statement about a mutation that was too wide, reported as if the
    test were keyed to the wrong thing. Stripping the two env lines leaves every other
    step of the order in place, which is what "the knob removed" means.
    """
    out = src.replace("os.path.join(_token_cache_dir(),", "os.path.join(os.path.dirname(TOKEN_CACHE),")
    if out == src:
        return None
    return out.replace(
        '    env = os.environ.get("AUPAI_TOKEN_CACHE_DIR")\n    if env:\n        return env\n', "", 1
    )


def _extractor_probe():
    """Known-answer cases for _function_source, run BEFORE anything execs train.py text.

    IT MUST RUN FIRST AND RETURN, not append to `bad`. The first version appended, and the
    mutation that reinstates the blank-line regex proved why that is useless: the extractor
    returns a fragment of train.py, `exec` raises SyntaxError inside _cache_path, and the
    process dies with a traceback before the `bad` list is ever printed -- the named message
    was collected and never shown. A probe for a defect that crashes is a probe that must
    report ahead of the crash, or it is decorative.

    Returns an error string, or None when the extractor behaves.
    """
    probe = (
        'def f():\n    """One.\n\n    Two.\n    """\n    return 1\n'
        '\n\ndef g():\n    """G."""\n    return f() + 1\n'
    )
    got = _function_source(probe, "f")
    if got != probe.split("\n\n\ndef g")[0]:
        return (
            "_function_source stops at a blank line inside a docstring: it returned "
            f"{got!r}, want the whole of f. A function's end is a parse fact; do not "
            "match it with a whitespace pattern."
        )
    if _function_source(probe, "g") is None:
        return "_function_source missed a function that follows another"
    return None


def main():
    live = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
    bad = []
    err = _extractor_probe()
    if err:
        print("BUG: the train.py source extractor is broken, so nothing below ran")
        print(f"  {err}")
        return 1
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
        head = subprocess.run(["git", "show", "HEAD:train.py"], capture_output=True, text=True, cwd=ROOT)
        if head.returncode == 0 and head.stdout:
            for dom in ("web_hq", "code_py_rp1t", "math"):
                for fone in (False, True):
                    now = _cache_path(live, dom, None, fone=fone)
                    was = _cache_path(head.stdout, dom, None, fone=fone)
                    if now != was:
                        bad.append(f"DEFAULT MOVED for {dom} (fone={fone}): HEAD {was} -> now {now}")
        else:
            print("note: no HEAD:train.py to diff against; the default is checked against TOKEN_CACHE only")
        for dom in ("web_hq", "math"):
            now = _cache_path(live, dom, None)
            if now != os.path.join(hard, f"tokens_{dom}.pt"):
                # ONLY WHERE THAT IS THE RULE. _token_cache_dir prefers the NVMe mount when
                # it EXISTS (68d23a6b), so on the pod -- where /mnt/data02/tokens is real --
                # the unset path is the NVMe one and this literal expectation is simply
                # false. Asserting it unconditionally makes the test host-dependent: green
                # on a laptop, red on the pod, for a reason that is not a defect. Compare
                # against the NVMe dir when the mount is present, which is what the accessor
                # itself does, and SAY which branch ran rather than silently skipping.
                nvme = _nvme_dir_if_present()
                want = (
                    os.path.join(nvme, f"tokens_{dom}.pt") if nvme else os.path.join(hard, f"tokens_{dom}.pt")
                )
                if now != want:
                    bad.append(
                        f"unset {ENV}: {dom} gave {now}, not {want} "
                        f"({'NVMe mount present' if nvme else 'no NVMe mount'})"
                    )

        # An empty value is unset, not the process cwd. `AUPAI_TOKEN_CACHE_DIR=` in a
        # shell wrapper would otherwise write caches wherever the launcher happened to be.
        if _cache_path(live, "math", "") != _cache_path(live, "math", None):
            bad.append(f"{ENV}='' did not fall back to the default")

        # THE RED PROOF: the same assertions against the shipped file with the knob
        # patched out. (a) must fail there and (b) must still pass -- a mutation that
        # breaks both would mean the test is keyed to something other than the knob.
        old = _without_the_knob(live)
        if old is None:
            bad.append(
                "_domain_cache_path does not call _token_cache_dir(), so there is no "
                "knob to patch out and nothing reads the variable"
            )
        else:
            # THE TWO PROPERTIES THAT MAKE THE RED EVIDENCE, both host-independent. The
            # earlier pair compared the mutated and live DEFAULTS and demanded equality,
            # which is only true where no NVMe mount exists: the mutation restores the
            # pre-knob line, and that line was hardcoded dirname(TOKEN_CACHE), so on the pod
            # the two defaults SHOULD differ. Asserting they match called a correct mutation
            # a defect. What actually matters is the pair below -- the old world does not
            # respond to the variable, the live one does.
            _old_on, _old_off = _cache_path(old, "childless", d), _cache_path(old, "childless", None)
            if _old_on != _old_off:
                bad.append(
                    f"the old-behaviour world still responded to the variable: {_old_on!r} "
                    f"with it set vs {_old_off!r} without, so the mutation does not restore "
                    "the pre-change line"
                )
            _live_on, _live_off = _cache_path(live, "childless", d), _cache_path(live, "childless", None)
            if _live_on == _live_off:
                bad.append(
                    f"the SHIPPED train.py ignored the variable ({_live_on!r} either way), "
                    "so the red against the mutation would not be about the knob"
                )

        # And the knob must be on the path the run takes. _domain_seqs deriving its cache
        # any other way would leave every assertion above true of a function nothing calls.
        if not re.search(r"^\s*cache = _domain_cache_path\(domain\)", live, re.M):
            bad.append(
                "_domain_seqs no longer takes its cache from _domain_cache_path; the "
                "knob is bypassed on the path the run actually uses"
            )
    finally:
        shutil.rmtree(d, ignore_errors=True)

    if bad:
        print(f"BUG: {ENV} does not control train.py's token cache directory")
        for b in bad:
            print(f"  {b}")
        return 1
    print(
        f"test_cache_dir_knob ok: {ENV} relocates the cache (plain + fone, not under "
        f"/data00), unset matches HEAD for 3 domains x 2 flags, empty falls back, the "
        f"knob-less copy of train.py ignores it, and the extractor survives a docstring "
        f"with a blank line in it"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main() if "--selftest" in sys.argv else (print(__doc__) or 0))

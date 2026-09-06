#!/usr/bin/env python3
"""_domain_seqs maps the token cache instead of loading it resident.

    python3 scripts/test_cache_mmap.py

WHAT THIS IS FOR. train.py full-loaded every domain cache on every rank before step 0. The E1
mix's caches sum to 166.2 GB, so a --plan run reached VmRSS 101 GB on the pod and was killed
mid-zh_web (task e1-41, found by watching the process). de ruled the training path exempt from
assert_not_co_resident (1fd88227) -- a launch is the job the lane exists for -- so there is no
refusal to lean on and the fix has to be that the read itself is smaller.

THE PROPERTY IS A RATIO, NOT A THRESHOLD, and that is the whole design of this test. "RSS stays
under N MiB" would need a machine-specific N, and the interpreter's own baseline (~200 MiB here)
swamps a small fixture. What separates a mapped read from a full one is that the mapped one's
resident cost tracks the rows DRAWN while the full one's tracks the FILE: so this builds ONE
fixture and draws two different fractions of it, and asserts the deltas differ by roughly the
ratio of the draws. A full load gives the same delta for both draws -- the file is resident
either way -- so it fails on the RATIO with no absolute number anywhere in the assertion.

MEASURED IN A CHILD PROCESS, once per case. ru_maxrss is a high-water MARK, so two cases in one
process cannot both be measured -- the second reads the first's peak. That is not a hypothetical:
it is how the first version of this measurement reported "peak RSS delta 0 MiB" and read as
evidence that mmap changed nothing, and then how `ps` RSS reported 577 MiB unchanged across a
381 MiB full load. The instrument was wrong twice before the finding was right once.

THE SECOND ASSERTION IS IDENTITY, and it is the one that matters more: mapped rows and
full-loaded rows must be the same bytes. A performance change that alters the training data is
not a performance change. Both cases hash the drawn rows and the hashes must match, so a
reshape or dtype difference introduced by the mapping path fails here rather than showing up as
a loss curve nobody can explain.

WHAT THIS DOES NOT TEST: the pod's page cache, DDP (rank 0 tokenizes while others barrier), and
the legacy-format fallback -- that branch needs a file written with
_use_new_zipfile_serialization=False, which no cache here is, and asserting on a file the
fixture had to write in a format the code never produces tests the fixture.
"""
import ast
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

FAILS = []

# The child: build the cache if absent, then load through the REAL _domain_seqs and draw
# `frac` of the pool the way build_mix does. Printed as one json line so the parent parses a
# value rather than a log.
CHILD = r'''
import hashlib, json, os, subprocess, sys, torch
sys.path.insert(0, {root!r})
os.environ["AUPAI_TOKEN_CACHE_DIR"] = sys.argv[1]
frac, mode = float(sys.argv[2]), sys.argv[3]
import train
train.DATA = sys.argv[1]
train.VOCAB_ID = "test-cache-mmap"
train.Cfg.seq = 4096

# THE CONTROL IS THE SAME WORLD WITH ONE FLAG FLIPPED. mode="full" strips mmap=True from the
# call _domain_seqs makes, so the control differs from the treatment in exactly the thing under
# test -- not in a hand-written alternative load path, which would compare two of my own
# functions instead of the one that runs.
if mode == "full":
    _real = torch.load

    def _no_mmap(*a, **k):
        k.pop("mmap", None)
        return _real(*a, **k)

    torch.load = _no_mmap


def rss_mib():
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return int(out) / 1024


class Tok:
    pass


before = rss_mib()
rows = train._domain_seqs("mmapprobe", Tok(), True, False)
after_load = rss_mib()
draw = max(1, int(len(rows) * frac))
g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(rows), generator=g)[:draw]
out = torch.empty((draw, rows.shape[1]), dtype=rows.dtype)
out[:] = rows[idx]
print(json.dumps({{
    "load_delta": after_load - before,
    "total_delta": rss_mib() - before,
    "rows": int(len(rows)),
    "draw": draw,
    "sha": hashlib.sha256(out.numpy().tobytes()).hexdigest()[:16],
}}))
'''


def check(ok, why):
    if not ok:
        FAILS.append(why)
    return ok


def _report():
    for f in FAILS:
        print(f"FAIL: {f}")
    if FAILS:
        return 1
    print("ok  _domain_seqs maps the cache: resident cost tracks the rows drawn, not the file "
          "size, and the drawn rows are byte-identical to a full load's")
    return 0


def _run(child_py, cache_dir, frac, mode):
    r = subprocess.run([sys.executable, child_py, cache_dir, str(frac), mode],
                       capture_output=True, text=True, cwd=ROOT)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("{")), None)
    if line is None:
        raise RuntimeError(f"child at frac={frac} mode={mode} printed no measurement "
                           f"(rc={r.returncode}): {r.stdout[-400:]} {r.stderr[-800:]}")
    return json.loads(line)


def main():
    d = tempfile.mkdtemp(prefix="cache_mmap_")
    try:
        import torch

        import train

        # train.DATA repointed at the fixture (in the child too), so the one-shard corpus dir
        # the freshness test needs is written under the fixture and nothing touches the repo's
        # data/. _corpus_fp raises FileNotFoundError on a missing dir, so the fixture cannot
        # skip it, and .srcfp has to hold the REAL fingerprint of what is there or the cache
        # reads as stale and every case measures a retokenize instead of a load.
        os.environ["AUPAI_TOKEN_CACHE_DIR"] = d
        train.DATA = d
        train.VOCAB_ID = "test-cache-mmap"
        cdir = os.path.join(d, "corpus", "mmapprobe")
        os.makedirs(cdir)
        with open(os.path.join(cdir, "mmapprobe_000.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "x" * 8}) + "\n")
        cache = train._domain_cache_path("mmapprobe")
        if not cache.startswith(d):
            return _skip(f"cache path {cache} escaped the fixture dir {d}")
        # 0.92 GiB: big enough that a full load's delta clears the interpreter's ~200 MiB
        # baseline by ~5x, small enough to write in a few seconds. int32 is the cache's dtype,
        # and FLAT, not pre-shaped: _domain_seqs reshapes with n = len(data) // (seq+1), so a
        # pre-shaped tensor would make len(data) the row count.
        seq, rows = 4097, 60_000
        torch.save(torch.randint(0, 65535, (seq * rows,), dtype=torch.int32), cache)
        for suffix, body in ((".vocab", train.VOCAB_ID),
                             (".srcfp", train._corpus_fp(cdir)),
                             (".seed", str(train._sample_seed()))):
            with open(cache + suffix, "w", encoding="utf-8") as f:
                f.write(str(body))

        child_py = os.path.join(d, "child.py")
        with open(child_py, "w") as f:
            f.write(CHILD.format(root=ROOT))

        small = _run(child_py, d, 0.02, "mmap")
        small2 = _run(child_py, d, 0.02, "mmap")
        big = _run(child_py, d, 0.40, "mmap")
        ctl_small = _run(child_py, d, 0.02, "full")
        file_mib = os.path.getsize(cache) / 2**20

        # A CONTROL ON THE INSTRUMENT, before any assertion about the arms (de, 2026-09-07). RSS
        # counts resident PAGES, so the mapped arm's number depends on how much the kernel has
        # faulted in at the moment it is read -- which depends on page-cache state left by the
        # previous run. Same code and same rows can read differently on a warm cache. That is the
        # layer both earlier wrong readings lived at: ru_maxrss's high-water mark, then ps RSS
        # unchanged across a 381 MiB load. So the identical case runs TWICE and the two readings
        # must agree; if they do not, the number is not a property of the code and no ratio
        # computed from it means anything. 15% of the reading, floored at 32 MiB so a small
        # absolute wobble does not trip it.
        tol = max(32.0, 0.15 * max(small["total_delta"], small2["total_delta"]))
        check(abs(small["total_delta"] - small2["total_delta"]) <= tol,
              f"THE INSTRUMENT IS UNSTABLE, so nothing below can be trusted: two identical "
              f"mmap runs of the same draw read {small['total_delta']:.0f} and "
              f"{small2['total_delta']:.0f} MiB, differing by more than {tol:.0f} MiB. RSS is "
              f"resident pages, so this is page-cache state or another process, not the code")

        check(small["sha"] != big["sha"],
              "the two draws hashed the same, so the fixture is not drawing different row sets "
              "and the ratio below would compare a quantity to itself")
        check(small["sha"] == small2["sha"],
              f"two identical runs drew DIFFERENT rows ({small['sha']} vs {small2['sha']}), so "
              f"the draw is not seeded and the identity check below compares nothing")
        # IDENTITY, and it matters more than the ratio: a performance change that alters the
        # training data is not a performance change. Same draw, mapped vs full, must be the
        # same bytes.
        check(small["sha"] == ctl_small["sha"],
              f"the mapped read and the full load returned DIFFERENT rows for the same draw "
              f"({small['sha']} vs {ctl_small['sha']}), so this changed the training data")

        # THE ASSERTION IS AGAINST THE CONTROL ROW, not a threshold I chose. The load itself is
        # where the two differ: a mapped load touches almost nothing, a full load makes the
        # whole file resident. Measured 2026-09-07: mmap +1 MiB, full +1128 MiB on a 938 MiB
        # file. 10x is the loosest statement of "one of these is the file and the other is not".
        check(ctl_small["load_delta"] >= 10 * max(small["load_delta"], 1.0),
              f"the load's resident cost is not distinguishable between the two: mapped "
              f"+{small['load_delta']:.0f} MiB, full +{ctl_small['load_delta']:.0f} MiB on a "
              f"{file_mib:.0f} MiB file. Either the mmap argument is gone, or the control's "
              f"mmap-stripping no longer reaches the call it is meant to strip -- and a control "
              f"that measures the same thing as the treatment cannot fail")
        check(small["load_delta"] < 0.25 * file_mib,
              f"the load made {small['load_delta']:.0f} MiB of a {file_mib:.0f} MiB file "
              f"resident before a single row was drawn, so it is not a mapped read")

        # AND the resident cost must track the DRAW, which is the property the fix is for:
        # zh_web is 85 GB on disk to draw 0.08 epochs of it. Total delta, not load delta --
        # after the load, both cases pay for the rows they touch, and only the mapped one pays
        # a cost that grows with the draw rather than one already spent.
        ratio = big["total_delta"] / max(small["total_delta"], 1.0)
        check(ratio >= 3.0,
              f"resident cost does not track the rows drawn: a {small['draw']}-row draw cost "
              f"{small['total_delta']:.0f} MiB and a {big['draw']}-row draw cost "
              f"{big['total_delta']:.0f} MiB, a ratio of {ratio:.1f} against a 20x difference "
              f"in rows. A full load gives ~1 -- the file is resident either way")

        _check_premises()
        _check_git_failure_worlds()
    finally:
        import shutil

        shutil.rmtree(d, ignore_errors=True)
    return _report()


def _check_premises():
    """The two premises that make the DELETED legacy fallback safe (de, 2026-09-07).

    _domain_seqs passes mmap=True with no try/except. That raises RuntimeError on a file saved
    with _use_new_zipfile_serialization=False, and the fallback that caught it was deleted rather
    than tested -- because a fixture writing a legacy file would test torch's behaviour on an
    input this tree cannot produce, and would go green forever while telling a reader the branch
    is exercised.

    The deletion is safe only while its premise holds, and it was NOT checked: every cache
    reaching _domain_seqs must be in torch.save's default zip format. If someone writes one another
    way, the deleted branch becomes a crash at step 0 instead of a slow load. Greps, so the premise
    fails loudly the day it dies -- a check with a real subject, unlike the fixture.

    THE FLAG IS FOUND WITH ast, NOT A REGEX, and that is the second thing the first version got
    wrong. Stripping `#` comments does not strip DOCSTRINGS or message strings, so the check went
    red on this very file -- which necessarily discusses the flag in its own prose, including in
    the FAIL text it prints. Exempting the file by name would have been the wrong repair: this file
    IS a cache writer, so an exemption would blind the check to the one writer it can see least.
    `ast` asks the precise question instead -- is there a Call with a keyword argument of that
    name -- which prose cannot satisfy and a real caller cannot evade by formatting.

    THE WRITER REGEX DOES NOT PARSE THE ARGUMENT LIST (de, 2026-09-07). `torch.save\\([^)]*cache`
    cannot cross an inner `)`, so it missed `torch.save(dict(...), cache)` and
    `torch.save(model.state_dict(), cache)`, and a trailing `\\b` after `cache` missed
    `cache_path`. Measured: 4 of 6 real shapes invisible, including the dict-literal one a second
    writer actually looks like. Co-occurrence in the statement instead -- verified on 6 positives
    and 3 negatives (train.py's checkpoint save, readout_30b's fixture, train_quality_head's
    weights), and tree-wide it returns 4 hits, all genuine cache writers, 0 false positives.

    Greps the tracked tree via git, not os.walk: an untracked scratch file is not the tree, and
    including one would make this fail on someone's local copy of a script.
    """
    # Known cache writers, all confirmed to use torch.save's default zip format. A new one trips
    # PREMISE 1 -- not because a second writer is wrong, but because nobody has confirmed its
    # format, and the deleted fallback is what used to make that not matter.
    KNOWN_WRITERS = {
        "train.py",                                # the real one: torch.save(data, cache)
        "scripts/test_mix_val_frac.py",             # fixture: a real cache in a temp dir
        "scripts/test_mix_anneal_and_cursor.py",    # fixture: same
        "scripts/test_cache_mmap.py",               # this file's own 0.92 GiB fixture
    }
    src = subprocess.run(["git", "ls-files", "-z", "*.py"],
                         capture_output=True, text=True, cwd=ROOT)
    if src.returncode != 0:
        # NOT A GIT REPOSITORY IS A SKIP; ANY OTHER GIT FAILURE IS A FAIL. Measured on the pod
        # 2026-09-07: /work/aupai is a file copy with no .git, so `git ls-files` exits 128 with
        # "not a git repository" and the first version reported FAIL there -- a red on a tree
        # whose premise is fine, for a reason that has nothing to do with the premise.
        #
        # WHY THIS IS NOT THE VACUITY SHAPE, since a SKIP that hides a failure is exactly what I
        # keep finding: the premise can only CHANGE where someone edits a file, and that is the
        # laptop, where the pre-commit hook runs this on every train.py commit and git is present.
        # The pod is a read-only copy of main -- nobody writes a new cache writer there -- so the
        # population this check exists to watch is not on the pod at all. What IS on the pod is
        # the measurement half, which needs the real caches and passes there (verified: 1 FAIL,
        # this one, and every RSS/identity assertion green). Reporting FAIL for "cannot check"
        # would teach a reader that the test is broken on the pod and cost the half that works.
        #
        # The reason is PRINTED, not swallowed: a skip nobody sees is indistinguishable from a
        # pass. Any other returncode keeps FAIL, because a git that exists and refuses is a
        # condition about this tree, not about the environment.
        #
        # KEYED ON THE MESSAGE, NOT ON EXIT 128, and de named the hole this closes (2026-09-07):
        # 128 is ALSO how git exits for a locked index (`Unable to create .git/index.lock`), a
        # corrupt index, and a dubious-ownership refusal -- and the last two happen on the LAPTOP,
        # the one machine where the population this check watches actually lives. Keying on the
        # code would make a stale index.lock read as SKIP and silence the check exactly there,
        # which is the vacuity shape again: population present, predicate blind. Driven, all three
        # FAIL: dubious ownership (128), index.lock (128), index corrupt (4).
        if "not a git repository" in src.stderr:
            print("SKIP: no git repository here, so the deleted-fallback premise cannot be "
                  "checked (it is checked on every commit by the hook, where git exists). "
                  "The measurements above ran.")
            return
        FAILS.append(f"cannot list tracked .py files, so the premise is not checked: "
                     f"{src.stderr[-200:]}")
        return
    # torch.save and a cache-ish destination CO-OCCURRING in the statement. Case-insensitive and
    # deliberately loose on the destination: a false positive costs one line in KNOWN_WRITERS, a
    # false negative costs the whole check.
    writer_re = re.compile(r"torch\.save\b[^\n]*(?:cache|tokens_)", re.I)
    writers, flaggers = [], []
    for rel in [p for p in src.stdout.split("\0") if p]:
        try:
            with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as f:
                body = f.read()
        except OSError:
            continue
        try:
            tree = ast.parse(body)
        except SyntaxError:
            # A file that does not parse cannot be cleared, and staying silent about it would be
            # the vacuous-pass shape: report it rather than skipping it.
            FAILS.append(f"{rel} does not parse, so it cannot be cleared of passing "
                         f"_use_new_zipfile_serialization")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and any(
                    kw.arg == "_use_new_zipfile_serialization" for kw in node.keywords):
                flaggers.append(rel)
                break
        code = "\n".join(ln.split("#", 1)[0] for ln in body.splitlines())
        if writer_re.search(code):
            writers.append(rel)
    # THE PREMISE, stated as the FORMAT rather than the writer count. A cache writer that also
    # passes the flag is the one case the deletion cannot survive.
    both = sorted(set(writers) & set(flaggers))
    if both:
        FAILS.append(
            f"PREMISE BROKEN, and this is the case the deletion cannot survive: {both} write a "
            f"token cache AND pass _use_new_zipfile_serialization, so that cache can be in the "
            f"legacy format. `torch.load(..., mmap=True)` in _domain_seqs raises RuntimeError on "
            f"it and has no fallback -- the run dies at step 0. Either write the default format, "
            f"or restore the fallback WITH a test that can fail")
    unknown = sorted(set(writers) - KNOWN_WRITERS)
    if unknown:
        FAILS.append(
            f"PREMISE 1: {unknown} write a token cache and are not in this test's KNOWN_WRITERS. "
            f"That is not wrong by itself -- two test fixtures legitimately do -- but the deleted "
            f"legacy fallback means every cache MUST be torch.save's default zip format, and "
            f"nobody has confirmed these are. Confirm the format and add the path, or route the "
            f"write through train.py's")
    # A flag passed anywhere else is worth reporting: the tree now holds the capability, so the
    # next cache writer can reach for it. Not broken yet, which the message says.
    elsewhere = sorted(set(flaggers) - set(writers))
    if elsewhere:
        FAILS.append(
            f"PREMISE 2: {elsewhere} pass _use_new_zipfile_serialization. No cache write does it "
            f"today, so nothing is broken yet -- but the tree now holds the pattern next to the "
            f"caches, and _domain_seqs has no fallback. Confirm no cache reaches it")


def _check_git_failure_worlds():
    """The SKIP above must key on git's MESSAGE, never on exit 128 (de, 2026-09-07).

    git exits 128 for a repository that IS one: a locked index, a corrupt index, a
    dubious-ownership refusal. Two of those happen on the LAPTOP, which is the only machine where
    the population _check_premises watches actually lives -- so a narrowing keyed on the code
    would make a stale .git/index.lock read as SKIP and silence the check exactly where it is
    load-bearing. Population present, predicate blind: the same vacuity shape as everything else
    found tonight.

    RUN AS A WORLD RATHER THAN LEFT AS A COMMENT, because a comment recording this cannot fail
    when someone simplifies the condition to `returncode == 128`. Each world stubs git on PATH,
    calls the real _check_premises, and asserts on the outcome:

      not a git repository (128)          SKIP   the pod's shape, the one exemption
      dubious ownership (128)             FAIL   de's case, laptop- and CI-shaped
      Unable to create index.lock (128)   FAIL   de's case, laptop-shaped
      index file corrupt (4)              FAIL   a non-128 refusal
    """
    import contextlib
    import io
    import shutil
    import tempfile

    worlds = [
        ("not a git repository", 128, "fatal: not a git repository (or any of the parent "
                                      "directories): .git", False),
        ("dubious ownership", 128, "fatal: detected dubious ownership in repository at '/x'", True),
        ("index.lock", 128, "fatal: Unable to create '/x/.git/index.lock': File exists.", True),
        ("index corrupt", 4, "fatal: index file corrupt", True),
    ]
    d = tempfile.mkdtemp(prefix="gitstub_")
    saved_path, saved_fails = os.environ.get("PATH", ""), list(FAILS)
    try:
        for label, code, msg, want_fail in worlds:
            bindir = os.path.join(d, label.replace(" ", "_").replace(".", "_"))
            os.makedirs(bindir, exist_ok=True)
            stub = os.path.join(bindir, "git")
            with open(stub, "w") as f:
                f.write(f'#!/bin/sh\necho "{msg}" >&2\nexit {code}\n')
            os.chmod(stub, 0o755)
            os.environ["PATH"] = bindir + os.pathsep + saved_path
            FAILS.clear()
            # The no-git world PRINTS its skip line, and on a real laptop run that line would
            # otherwise read as the premise check itself having skipped -- the opposite of what
            # happened. Captured here so the only skip a reader sees on stdout is a real one.
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _check_premises()
            got_fail = bool(FAILS)
            FAILS.clear()
            if got_fail != want_fail:
                FAILS.append(
                    f"the git-failure narrowing is wrong for {label!r} (exit {code}): it "
                    f"{'FAILED' if got_fail else 'SKIPped'} and must "
                    f"{'FAIL' if want_fail else 'SKIP'}. Only 'not a git repository' is an "
                    f"environment fact; every other refusal is a condition about this tree, and "
                    f"keying the exemption on exit 128 silences the check on a laptop with a "
                    f"stale index.lock -- where the population it watches actually lives")
                saved_fails.append(FAILS[-1])
    finally:
        os.environ["PATH"] = saved_path
        FAILS.clear()
        FAILS.extend(saved_fails)
        shutil.rmtree(d, ignore_errors=True)


def _skip(why):
    print(f"SKIP: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

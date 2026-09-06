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
import json
import os
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
        big = _run(child_py, d, 0.40, "mmap")
        ctl_small = _run(child_py, d, 0.02, "full")
        file_mib = os.path.getsize(cache) / 2**20

        check(small["sha"] != big["sha"],
              "the two draws hashed the same, so the fixture is not drawing different row sets "
              "and the ratio below would compare a quantity to itself")
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
    finally:
        import shutil

        shutil.rmtree(d, ignore_errors=True)
    return _report()


def _skip(why):
    print(f"SKIP: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

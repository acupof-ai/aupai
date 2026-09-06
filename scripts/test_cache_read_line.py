#!/usr/bin/env python3
"""build_mix states the cache bytes it is about to read, before the first torch.load.

    python3 scripts/test_cache_read_line.py

WHY A PRINT NEEDS A TEST. de ruled the training path exempt from assert_not_co_resident
(1fd88227): a launch is the job the lane exists for, so refusing it inverts the priority. What
that leaves is a read nobody measures at launch time -- train.py full-loads every domain cache,
and the 166.2 GB figure for the E1 mix was computed offline by e1 and appeared in NO log, so the
101 GB RSS that followed had to be recovered from /proc/<pid>/status after the fact. The line
this tests is the cheapest fix: a number in the run's own output. A print with no check regresses
in silence, and the regression is invisible precisely because nothing downstream consumes it.

THREE PROPERTIES, and the third is the one a reimplementation gets wrong.

(1) THE TOTAL IS RIGHT. Asserted against a total computed HERE from the fixture's own file
sizes, never parsed back out of anything train.py computed -- a test that reads the number under
test as its own expectation passes on any transformation applied to both sides
(test-compared-code-to-itself). Sizes are 1..6 MiB so the sum is arithmetic a reader can verify
by eye.

(2) THE HEADER'S UNIT CANNOT COLLAPSE THE TOTAL. The first version printed `.1f` GiB, so this
fixture's 21 MiB rendered as "0.0 GiB over 6 cache(s)" -- a total reading as nothing beside a
non-zero count. MiB is asserted, and asserted as a NUMBER parsed from the line, because "0.0"
and "21" are both strings that contain a digit and only one of them is the measurement.

(3) AN ABSENT CACHE IS REPORTED, NOT SUMMED AS ZERO. This is the property with teeth. A domain
whose cache does not exist will be TOKENIZED -- minutes of CPU and a cache write -- so folding
it in as 0 bytes both understates the read and hides the expensive case. The fixture builds SOME
caches and leaves others missing, and asserts the count of each appears. A version that skipped
missing files silently would satisfy (1) and (2) exactly.

REACHABILITY IS PART OF THE CLAIM, not a separate concern: the fixture calls the real
train.build_mix through the real guards (VOCAB_ID set as build_tokenizer sets it, anneal_frac
agreeing with the mix as _mix_anneal_frac demands) and stops at the first _domain_seqs. So a
change that moves the print below the first torch.load, or behind a guard that returns earlier,
fails this test rather than passing it while printing nothing on a real launch. The print is
under `if is_main`, so the fixture drives is_main=True; a rank that is not main printing nothing
is correct and not tested here.
"""
import json
import os
import re
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

FAILS = []


def check(ok, why):
    if not ok:
        FAILS.append(why)
    return ok


def _report():
    """Every exit goes through here. The first version had `return 1` on the missing-line
    check, which bypassed the printing loop below: mutating the print out of existence made
    this test exit 1 and say NOTHING, so a reader saw a red test with no reason and the most
    important failure was the least legible one."""
    for f in FAILS:
        print(f"FAIL: {f}")
    if FAILS:
        return 1
    print("ok  build_mix states the cache bytes it is about to read: total in MiB, per-domain "
          "breakdown largest-first, and absent caches named as work rather than summed as zero")
    return 0


def main():
    import io
    import contextlib

    d = tempfile.mkdtemp(prefix="cache_read_line_")
    try:
        os.environ["AUPAI_TOKEN_CACHE_DIR"] = d
        import train

        mix_path = os.path.join(ROOT, "data", "mix_e1_n1.json")
        if not os.path.exists(mix_path):
            print(f"SKIP: {mix_path} absent")
            return 0
        mix = json.load(open(mix_path, encoding="utf-8"))
        names = list(mix["domains"])
        if len(names) < 8:
            print(f"SKIP: mix has {len(names)} domains, need >= 8 to leave some absent")
            return 0

        # PRESENT: six caches at 1..6 MiB. ABSENT: everything else in the mix.
        present, want_bytes = names[:6], 0
        for i, n in enumerate(present):
            nbytes = (i + 1) * 2**20
            with open(os.path.join(d, f"tokens_{n}.pt"), "wb") as f:
                f.write(b"\0" * nbytes)
            want_bytes += nbytes
        absent = names[6:]
        want_mib = want_bytes / 2**20

        # Reach the print through the real guards, then stop before any load.
        def stop(*a, **k):
            raise SystemExit("stop-after-print")

        train._domain_seqs = stop
        train.Cfg.anneal_frac = float(mix.get("anneal_frac", train.Cfg.anneal_frac))
        train.VOCAB_ID = "test-cache-read-line"

        class Tok:
            pass

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                train.build_mix(mix_path, Tok(), True, False)
        except SystemExit:
            pass
        out = buf.getvalue()

        line = next((ln for ln in out.splitlines() if ln.startswith("cache read:")), None)
        if not check(line is not None,
                     "build_mix printed no 'cache read:' line before the first _domain_seqs -- "
                     "either the print is gone or it now sits below the first load, where a "
                     "launch's own log cannot show the read it is about to do"):
            return _report()

        # (1)+(2) the total, as a NUMBER, in MiB.
        m = re.search(r"cache read: ([\d,]+) MiB", line)
        if check(m is not None,
                 f"the header does not state a MiB total: {line!r}. GiB alone rounds this "
                 f"fixture's {want_mib:.0f} MiB to 0.0, which reads as no bytes at all beside a "
                 f"non-zero cache count"):
            got_mib = float(m.group(1).replace(",", ""))
            check(abs(got_mib - want_mib) <= 1,
                  f"total is {got_mib} MiB, the fixture's files are {want_mib:.0f} MiB "
                  f"({want_bytes} bytes over {len(present)} files, sizes 1..{len(present)} MiB)")

        # (1) the count of caches actually read.
        mc = re.search(r"over (\d+) cache\(s\)", line)
        if check(mc is not None, f"the header does not state how many caches it read: {line!r}"):
            check(int(mc.group(1)) == len(present),
                  f"header says {mc.group(1)} cache(s), fixture wrote {len(present)}")

        # (3) THE ABSENT ONES ARE NAMED AS WORK, not folded in as zero.
        ma = re.search(r"(\d+) to tokenize", line)
        if check(ma is not None,
                 f"{len(absent)} domain(s) in this mix have NO cache and the line does not say "
                 f"so: {line!r}. Those will be tokenized -- minutes of CPU and a cache write -- "
                 f"so summing them as 0 bytes understates the read and hides the costly case"):
            check(int(ma.group(1)) == len(absent),
                  f"line says {ma.group(1)} to tokenize, fixture left {len(absent)} absent")

        # The per-domain breakdown exists and names the biggest first, so a reader can see
        # WHICH domain is the read rather than only that the total is large.
        body = [ln for ln in out.splitlines() if re.match(r"^\s+[\d.]+ GiB\s+\S", ln)]
        check(len(body) == len(present),
              f"{len(body)} per-domain line(s) for {len(present)} cache(s)")
        if body:
            check(present[-1] in body[0],
                  f"the largest cache ({present[-1]}, {len(present)} MiB) is not the first "
                  f"per-domain line: {body[0]!r}")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    return _report()


if __name__ == "__main__":
    sys.exit(main())

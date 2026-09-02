#!/usr/bin/env python3
"""de-13 acceptance, the part that needs no GPU: does a row cursor actually seed used[]?

fb's de-13 said the resume path stores row_cursor and never reads it, so every domain
restarts at row 0. Archaeology says the read/pass/apply/save halves all landed in d5ed78c
(2026-09-01 03:19), five hours after the projection that motivated the task -- and b0 has
confirmed the 26/34/92% figures were arithmetic, never a measurement. But fb's standard is
behaviour, not code appearance, and tonight every mistake came from something that looked
right. So run it.

Runs on the pod because data/tokenizer.json and data/corpus/ live there only. CPU and RAM
only: build_mix tokenizes and plans, it builds no model and touches no card. One small
domain (cot, 13 shards) is enough -- train.py:1881-1882 executes per domain independently,
so one domain exercises the same two lines nine would.

Four things are checked, and the fourth is the one the other three cannot see:

  1. a cursor SEEDS used[]           -- the planned ROWS move, measured by content overlap
  2. no domain silently discards it  -- "cursor discarded" must not appear (tilerl)
  3. a mismatched seed IS refused    -- the guard fires when it should, so 2 is not vacuous
  4. Cfg._plan_trimmed becomes true  -- what the LR compensation at train.py:2428 reads

Check 1 was "the plan shrinks" on the first run and that was wrong. Row count comes from
total_tokens (:1919 want = int(rows * frac * weight)); the cursor only moves the window
(:1929 arange(used, used+want)), and it shrinks the plan only where want exceeds
cap = len(pool)*epochs - used. cot's pool clears 11718 rows, so nothing capped and the
count was identical on both sides -- a red that says nothing about the cursor. What the
cursor must change is WHICH rows: baseline plans pool[0:7812], a cursor at 3906 plans
pool[3906:11718], so exactly half the content is shared. Ignored cursor -> 100% shared.
Measured on the returned token rows, not on a printed line: printing is not checking.

Writes nothing under runs/ (the pod is training and pod_push skips runs/ both ways).

    python3 scripts/test_resume_cursor_pod.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile

ROOT = "/work/aupai"
sys.path.insert(0, ROOT)


def build(mix_path, tok, **kw):
    """build_mix, with its chatter captured. Returns (rows, log, trimmed)."""
    from train import Cfg, build_mix

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = build_mix(mix_path, tok, True, False, 0, 1, **kw)
    rows = out[0] if isinstance(out, tuple) else out
    if isinstance(rows, tuple):
        rows = rows[0]
    return rows, buf.getvalue(), bool(getattr(Cfg, "_plan_trimmed", False))


def fingerprints(rows):
    """One hash per planned row, so two plans can be compared by CONTENT.

    The cursor moves which rows are planned, not how many; a count cannot see that.
    Rows are shuffled inside each phase (:1934 randperm), so compare as a set.
    """
    import hashlib

    return {hashlib.blake2b(r.numpy().tobytes(), digest_size=8).digest() for r in rows}


def main():
    import train
    from train import Cfg, build_tokenizer

    tok_path = os.path.join(ROOT, "data", "tokenizer.json")
    dom = os.environ.get("DOMAIN", "cot")
    if not os.path.exists(tok_path):
        print(f"SKIP: no {tok_path}")
        return 0
    if not os.path.isdir(os.path.join(ROOT, "data", "corpus", dom)):
        print(f"SKIP: no data/corpus/{dom}")
        return 0
    # build_tokenizer, not Tokenizer.from_file: it is the only thing that sets the
    # module-global VOCAB_ID (:1471). Loading the tokenizer directly leaves it None, and
    # :1685 writes `VOCAB_ID or ""` -- a 0-byte vocab stamp beside any cache this run
    # rebuilds. :1646 then reads that empty stamp as a mismatch, so the NEXT run
    # retokenizes from scratch: 851,965 docs and five minutes of eight idle cards, which
    # is what this test cost tilerl's gate run at 18:05. /data00 is shared; a CPU-only
    # test is not a side-effect-free test.
    tok = build_tokenizer([])
    assert train.VOCAB_ID, "VOCAB_ID unset after build_tokenizer: a rebuild here would stamp an empty vocab"
    # And refuse rather than rebuild. The assert above proves the STAMP would be right;
    # it does not stop a rebuild, and a rebuild is the expensive thing -- 851,965 docs and
    # five idle cards. This test reads a real /data00 cache for a real domain on purpose
    # (the cursor arithmetic is only meaningful against the pool the run uses), so the
    # guard is what keeps "reads it" from silently becoming "rewrites it" when a shard
    # moves or the seed changes. It raises and names the domain and the stamp.
    from eval.cache_guard import assert_caches_fresh

    assert_caches_fresh([dom])

    Cfg.seq, Cfg.batch, Cfg.accum = 512, 2, 1
    mix = {"total_tokens": 4e6, "domains": {dom: {"weight": 1.0, "epochs": 1}}}
    tmp = tempfile.mkdtemp()
    mix_path = os.path.join(tmp, "mix_cursor_probe.json")
    with open(mix_path, "w", encoding="utf-8") as f:
        json.dump(mix, f)

    bad = []

    rows_base, _, trimmed0 = build(mix_path, tok)
    fp_base = fingerprints(rows_base)
    used_after = dict(getattr(Cfg, "_row_cursor", {}) or {})
    print(f"  baseline, no cursor : {len(rows_base)} rows, _row_cursor={used_after}, "
          f"_plan_trimmed={trimmed0}")
    if not used_after.get(dom):
        print(f"  FAIL: build_mix left no _row_cursor for {dom}; nothing could be saved")
        return 1
    if trimmed0:
        bad.append("_plan_trimmed is true with NO cursor -- the flag cannot mean 'trimmed'")

    # The overlap metric's own broken world, and it needs no mutation: a build that
    # IGNORES the cursor plans exactly what the baseline planned, so a second baseline
    # build is that world. If this does not read ~100%, the metric cannot detect an
    # ignored cursor and check 1 below is decoration.
    rows_again, _, _ = build(mix_path, tok)
    self_overlap = len(fp_base & fingerprints(rows_again)) / max(len(fp_base), 1)
    print(f"  metric self-check   : an ignored cursor would read {self_overlap:.1%} shared")
    if self_overlap < 0.99:
        print(f"  FAIL: two identical builds share only {self_overlap:.1%} of their rows; the "
              f"plan is not reproducible, so overlap cannot measure the cursor")
        return 1

    # 1 + 2 + 4: a real cursor, at half of what the first plan consumed.
    half = {dom: max(1, used_after[dom] // 2)}
    rows_res, log, trimmed1 = build(mix_path, tok, row_cursor=half, cursor_srcfp=None,
                                   cursor_seed=None)
    fp_res = fingerprints(rows_res)
    resumed = [ln.strip() for ln in log.splitlines() if "resuming at row" in ln]
    discarded = [ln.strip() for ln in log.splitlines() if "cursor discarded" in ln]
    shared = len(fp_base & fp_res) / max(len(fp_base), 1)
    print(f"  cursor={half[dom]:>8} : {len(rows_res)} rows, _plan_trimmed={trimmed1}, "
          f"{shared:.1%} of the baseline's rows re-planned")
    for ln in resumed + discarded:
        print("      ", ln)
    if not resumed:
        bad.append("no 'resuming at row' line: the cursor was read but never applied")
    if discarded:
        bad.append(f"cursor discarded on a matching seed/fp: {discarded[0][:90]}")
    # The count cannot move here (rows comes from total_tokens); the CONTENT must.
    # An ignored cursor re-plans pool[0:n] -> ~100% shared. A cursor at n/2 shifts the
    # window by half its length -> about half shared, and never all of it.
    if shared > 0.9:
        bad.append(f"{shared:.1%} of the baseline's rows are planned again: the cursor did "
                   f"not move the window, consumed rows are being re-read")
    if not trimmed1:
        bad.append("_plan_trimmed stayed false, so the LR compensation at 2428 will not fire")

    # 3: the seed guard must FIRE on a mismatch. Without this, check 2 is vacuous --
    # if the guard never looks, "no discard line" holds just as well. (fb: this test IS
    # check 4's broken world, and a blocking gate item in its own right.)
    rows_seed, log_seed, _ = build(mix_path, tok, row_cursor=half, cursor_srcfp=None,
                                  cursor_seed=Cfg.seed + 999)
    fired = [ln.strip() for ln in log_seed.splitlines() if "cursor discarded" in ln]
    same_as_base = len(fingerprints(rows_seed) & fp_base) / max(len(fp_base), 1)
    print(f"  mismatched seed     : {len(rows_seed)} rows, discard lines={len(fired)}, "
          f"{same_as_base:.1%} identical to baseline")
    for ln in fired:
        print("      ", ln)
    if not fired:
        bad.append("a mismatched sample_seed did NOT discard the cursor -- the guard at "
                   "1886 is not firing, so a seed change would index other documents")
    elif same_as_base < 0.99:
        bad.append(f"seed mismatch discarded the cursor but only {same_as_base:.1%} of the "
                   f"plan matches the baseline: the discard did not restore row 0")

    print()
    if bad:
        for b in bad:
            print("  FAIL:", b)
        print("resume cursor: DEFECT PRESENT")
        return 1
    print(f"  ok: cursor moves the planned rows ({shared:.1%} overlap, not ~100%), no silent "
          f"discard, _plan_trimmed true, and the seed guard fires and restores row 0")
    print("resume cursor: OK -- de-13's premise does not hold on this code")
    return 0


if __name__ == "__main__":
    sys.exit(main())

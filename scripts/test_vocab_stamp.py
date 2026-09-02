#!/usr/bin/env python3
"""A token cache's vocabulary stamp must be able to FAIL.

THE INCIDENT (fb, 2026-09-02, killed by exact PID two minutes in). `eval/ppl.py` on card 7
printed `mix: math_owm_stage2 cache was built by another vocabulary, retokenizing` against
the nine caches the live 20B run was training on. train.VOCAB_ID is set only by
train.build_tokenizer; ppl.py comes through scripts.loader, which never touches it, so
VOCAB_ID stayed None. train.py then read

    same_vocab = os.path.exists(stamp) and open(stamp).read().strip() == (VOCAB_ID or "")

and wrote `f.write(VOCAB_ID or "")`. The write side and the read side shared one `or ""`
fallback, so a None VOCAB_ID wrote a 0-byte stamp and the next unstamped reader compared it
equal: the check passed BY CONSTRUCTION and could never fail. Seven 0-byte .vocab files
dated 2026-09-01 sit on the pod, and every one of them read as fresh.

Three properties, and the pair is what makes them real -- a fix that simply declared every
cache stale would retokenize the corpus, so the positive control is not decoration:

    1. build_mix RAISES when VOCAB_ID is None, rather than defaulting it
    2. no empty stamp is ever written -- an unset VOCAB_ID raises at the write site
    3. an empty stamp is STALE, never a match
    4. (control) a correctly stamped cache is still REUSED

Asserted by EXECUTING train.py, not by reading it: two earlier tests in this repo grepped
for a construct and passed a mutation that removed the line feeding it. And the mutant is
built by MUTATING the real train.py -- the three fixed constructs reverted to their pre-fix
form -- so this file proves it goes red against the code that shipped the incident. A test
that only runs against the fixed tree cannot tell a fix from a no-op.

    python3 scripts/test_vocab_stamp.py --selftest
"""

# restartable: every world is a mkdtemp with a copied train.py and a fake cache, removed in a
# finally. Nothing is written outside it and no state carries between runs, so an interrupt
# costs the seconds already spent and leaves at most one temp directory.

import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class _Enc:
    def __init__(self, ids):
        self.ids = ids


class _Tok:
    """Enough tokenizer for encode(): data/tokenizer.json is gitignored and the vocabulary
    is not the variable here -- the stamp's freshness arithmetic is."""

    def token_to_id(self, t):
        return 1

    def encode_batch(self, texts):
        return [_Enc([2, 3, 4, 5, 6, 7, 8, 9]) for _ in texts]


def _load(train_py, tag):
    spec = importlib.util.spec_from_file_location(tag, train_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[tag] = mod
    spec.loader.exec_module(mod)
    return mod


def _mutate(src):
    """The pre-fix train.py, produced from the shipped one. Returns (source, counts).

    Every substitution is counted and the caller refuses a mutation that changed nothing:
    an unmutated copy would pass the judge and read as "the test detects the defect".
    """
    n_guard = 0
    lines, out, i = src.split("\n"), [], 0
    while i < len(lines):
        # Any condition ending in `not VOCAB_ID:`, not the literal `if not VOCAB_ID:`. The
        # shipped guard grew a `tok is not None and` prefix while this file was being written
        # (another session, 2026-09-02) and the literal match silently stopped finding it --
        # which would have reported "the fix was reverted" against a fix that is present.
        if lines[i].strip().startswith("if ") and lines[i].strip().endswith("not VOCAB_ID:"):
            ind = len(lines[i]) - len(lines[i].lstrip())
            i += 1
            while i < len(lines) and (
                not lines[i].strip() or (len(lines[i]) - len(lines[i].lstrip())) > ind
            ):
                i += 1
            n_guard += 1
            continue
        out.append(lines[i])
        i += 1
    src = "\n".join(out)

    read_old = 'same_vocab = os.path.exists(stamp) and open(stamp).read().strip() == (VOCAB_ID or "")'
    read_new = ("same_vocab = bool(VOCAB_ID) and os.path.exists(stamp) and "
                "open(stamp).read().strip() == VOCAB_ID")
    n_read = src.count(read_new)
    src = src.replace(read_new, read_old)
    n_write = src.count("f.write(VOCAB_ID)")
    src = src.replace("f.write(VOCAB_ID)", 'f.write(VOCAB_ID or "")')
    return src, {"guards": n_guard, "read": n_read, "write": n_write}


def _mkcorpus(root, dom, ndocs=400):
    cdir = os.path.join(root, "corpus", dom)
    os.makedirs(cdir, exist_ok=True)
    # "content", not "text": _jsonl_content reads ["content"] and a KeyError here would be
    # the test dying in its own setup rather than measuring anything.
    with open(os.path.join(cdir, f"{dom}_000.jsonl"), "w", encoding="utf-8") as f:
        for i in range(ndocs):
            f.write(json.dumps({"content": f"document {i} alpha beta gamma"}) + "\n")
    return cdir


def _isolate(mod, d):
    """Point the module at a tempdir, and REFUSE if the cache would land outside it.

    scripts/test_domain_loss_val.py set an environment variable train.py never reads and
    wrote a real cache into the pod's shared /data00 beside the live run's, with a 0-byte
    .vocab (2026-09-02). Nothing raised; probe_domain simply happened not to be in the mix.
    """
    mod.DATA = d
    os.makedirs(os.path.join(d, "cache"), exist_ok=True)
    mod.TOKEN_CACHE = os.path.join(d, "cache", "tokens.pt")
    mod.Cfg.seq = 8
    mod.Cfg.fone = False
    mod.Cfg.seed = 0
    mod.Cfg.sample_seed = None
    mod.Cfg.val_frac = 0.05
    mod.Cfg.val_rows_max = 5
    mod.Cfg.anneal_frac = 0.1
    p = mod._domain_cache_path("probe")
    if not os.path.abspath(p).startswith(os.path.abspath(d)):
        raise RuntimeError(f"cache would land at {p}, outside this test's tempdir {d}")


def _call(fn, *a, **kw):
    """(outcome, message, stdout). outcome is 'raised:<type>' or 'returned'."""
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        fn(*a, **kw)
        return "returned", "", buf.getvalue()
    except BaseException as e:  # noqa: BLE001  -- the outcome IS the measurement
        return f"raised:{type(e).__name__}", str(e), buf.getvalue()
    finally:
        sys.stdout = old


def probe(mod):
    """Four known-answer worlds. Returns observations, never verdicts."""
    obs = {}
    d = tempfile.mkdtemp(prefix="vocabstamp_")
    tok = _Tok()
    vid = "a" * 40
    try:
        _isolate(mod, d)

        # 1. build_mix against a cache that is ALREADY FRESH, with VOCAB_ID unset. The cache
        # is built first, under a real fingerprint, so nothing here can fail for want of a
        # file: the only variable left is the guard. Without the guard the mutant does not
        # raise -- it retokenizes the fresh cache and re-stamps it empty, which is the
        # incident (nine domains of the live 20B run, two minutes from being rebuilt).
        mod.VOCAB_ID = vid
        _mkcorpus(d, "probe_bm")
        bm_cache = mod._domain_cache_path("probe_bm")
        bm_stamp = bm_cache + ".vocab"
        mp = os.path.join(d, "mix.json")
        json.dump({"total_tokens": 8 * 100,
                   "domains": {"probe_bm": {"weight": 1.0, "epochs": 1, "anneal": 1.0}}},
                  open(mp, "w"))
        mod.build_mix(mp, tok, True, False, 0, 1)
        obs["setup_fresh"] = os.path.exists(bm_cache) and open(bm_stamp).read().strip() == vid
        before = os.path.getmtime(bm_cache)
        mod.VOCAB_ID = None
        out, msg, _ = _call(mod.build_mix, mp, tok, True, False, 0, 1)
        obs["build_mix"] = out
        obs["build_mix_names_vocab"] = "VOCAB_ID" in msg
        obs["build_mix_kept_cache"] = (
            os.path.getmtime(bm_cache) == before and open(bm_stamp).read().strip() == vid
        )

        # 2. the write site, with no cache to reuse: an unset VOCAB_ID must not leave a stamp.
        mod.VOCAB_ID = None
        _mkcorpus(d, "probe_ws")
        ws_stamp = mod._domain_cache_path("probe_ws") + ".vocab"
        obs["write_outcome"] = _call(mod._domain_seqs, "probe_ws", tok, True, False)[0]
        obs["write_stamp"] = (
            "absent" if not os.path.exists(ws_stamp)
            else ("empty" if not open(ws_stamp).read().strip() else "filled")
        )

        # 3+4. A REAL cache, built by the module itself under a real fingerprint -- then the
        # pod's own state reproduced by truncating its .vocab to 0 bytes.
        mod.VOCAB_ID = vid
        _mkcorpus(d, "probe_st")
        cache = mod._domain_cache_path("probe_st")
        stamp = cache + ".vocab"
        mod._domain_seqs("probe_st", tok, True, False)
        obs["built_stamp"] = open(stamp).read().strip() == vid

        # 4. control: the same cache, same fingerprint. It must be REUSED, or the fix is
        # "retokenize always" -- nine domains and 851,965 documents per eval run.
        _, _, log = _call(mod._domain_seqs, "probe_st", tok, True, False)
        obs["control_reused"] = "retokenizing" not in log and "tokenizing" not in log

        # 3. the 0-byte stamp, against an unset VOCAB_ID: the incident's exact pair.
        open(stamp, "w").close()
        mod.VOCAB_ID = None
        before = os.path.getmtime(cache)
        out, _, log = _call(mod._domain_seqs, "probe_st", tok, True, False)
        obs["empty_outcome"] = out
        obs["empty_silent_reuse"] = (
            out == "returned" and os.path.getmtime(cache) == before
            and "another vocabulary" not in log
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return obs


def judge(obs):
    bad = []
    if not obs["setup_fresh"]:
        bad.append("the fixture is broken: build_mix under a real VOCAB_ID left no stamped "
                   "cache, so world 1 measures a missing file rather than the guard")
    if not obs["build_mix"].startswith("raised"):
        bad.append("build_mix with VOCAB_ID=None did not raise against an ALREADY FRESH "
                   "cache: it retokenized what the run is training on")
    elif not obs["build_mix_names_vocab"]:
        bad.append(f"build_mix raised, but not on the vocabulary -- the message does not name "
                   f"VOCAB_ID, so this is some other failure: {obs['build_mix']}")
    if not obs["build_mix_kept_cache"]:
        bad.append("the fresh cache was rewritten or its stamp changed while VOCAB_ID was "
                   "unset -- the exact write the incident was two minutes from")
    if obs["write_stamp"] == "empty":
        bad.append("a 0-byte .vocab stamp was written -- the read side compares it equal to "
                   "an unset VOCAB_ID, so the cache is permanently unverifiable")
    if not obs["write_outcome"].startswith("raised") and obs["write_stamp"] != "filled":
        bad.append(f"the write site neither raised nor stamped a vocabulary "
                   f"({obs['write_outcome']}, stamp {obs['write_stamp']})")
    if not obs["built_stamp"]:
        bad.append("a build under a real VOCAB_ID did not write it into the stamp")
    if obs["empty_silent_reuse"]:
        bad.append("an EMPTY stamp read as a match: the cache was handed back unchanged with "
                   "no message. This is the seven 0-byte stamps on the pod reading as fresh")
    if not obs["control_reused"]:
        bad.append("a correctly stamped, unchanged cache was NOT reused -- the fix retokenizes "
                   "every domain, which costs more than the defect")
    return bad


def main():
    real = os.path.join(ROOT, "train.py")
    if not os.path.exists(real):
        print("BUG: no train.py")
        return 1

    rc = 0
    fixed = judge(probe(_load(real, "_vs_fixed")))
    if fixed:
        print("BUG: the shipped train.py fails the vocabulary-stamp properties")
        for b in fixed:
            print(f"  {b}")
        rc = 1
    else:
        print("ok: shipped train.py -- build_mix refuses an unset VOCAB_ID, no empty stamp is "
              "written, an empty stamp is stale, a good cache is reused")

    # THE RED HALF. Without it a green above proves only that this file agrees with today's
    # train.py, which a no-op fix also satisfies.
    d = tempfile.mkdtemp(prefix="vocabstamp_mut_")
    try:
        src, n = _mutate(open(real, encoding="utf-8").read())
        if n["guards"] != 2 or n["read"] != 1 or n["write"] != 1:
            print(f"BUG: the pre-fix mutation could not be built from train.py ({n}). Either "
                  f"the fix was reverted or it moved -- re-read train.py; an unmutated copy "
                  f"would pass below and read as a detection.")
            return 1
        mut = os.path.join(d, "train.py")
        open(mut, "w", encoding="utf-8").write(src)
        # fone/ and the rest resolve off ROOT; the copy's own ROOT is this tempdir, which is
        # where its DATA and token cache must land anyway.
        broke = judge(probe(_load(mut, "_vs_prefix")))
        if not broke:
            print("BUG: the pre-fix train.py PASSED. This test cannot see the defect it was "
                  "written for, so its green above means nothing.")
            rc = 1
        else:
            print(f"ok: pre-fix train.py caught, {len(broke)} property(ies) red")
            for b in broke:
                print(f"  red: {b}")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main() if "--selftest" in sys.argv else (print(__doc__) or 0))

#!/usr/bin/env python3
"""Byte-identity check for the parallel exact-dup+holdout global pass.
Builds small w* shards with known exact dups + a holdout-pattern doc, runs the
SERIAL and PARALLEL exact-only global pass in separate out dirs, and asserts the
written {domain}_*.jsonl are byte-identical. Fails loudly on a real mismatch."""
import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import build_corpus as B  # noqa: E402

DUP = "public static void paginate records count cache size large enough text" * 2
UNIQ = "def g(x): return x * 2 + 1 and the rest of this function body is long enough"
HOLD = "REVEAL_ME_SENTINEL_HOLDOUT_0003 is a holdout marker that must be dropped"
# A genuinely-held-out doc (real eval problem in data/eval/holdout_hashes.txt), to
# exercise the holdout-slice gate end-to-end: captured in the pass, through _write_stats,
# into holdout_slice_{phase}.jsonl whose row count must match the stamp's holdout reason.
REAL_HOLD = "小明有10个苹果，他送给小红3个，还剩几个？"


def check_token_count_failure_path():
    """_write_stats must emit the FULL canonical key set even when the token count fails.

    The defect this pins (found 2026-09-18, main was red for this --selftest): both
    failure branches set only `tokens_status`, leaving `tokens` and `tokens_config`
    absent, so _assert_canonical_stats failed on exactly the path it exists to describe.
    CANONICAL_STATS_KEYS' own comment states the contract -- "tokens/tokens_status are
    never absent" -- and the code violated it wherever data/tokenizer.json was missing
    (every laptop and CI) or its count raised.

    WHY None AND NOT 0: an unmeasured count and a measured zero are different facts.
    scripts/count_dir.py:148 reads `st.get("tokens")` and falls back to kept_tokens when
    it is None, then to "carries no integer tokens"; a 0 is an int, so it would pass that
    check and report a delta of the whole corpus against a false zero.

    Exercises the EXCEPT branch specifically -- tokenizer present, count raises -- because
    a machine with no tokenizer takes the `else` branch and would leave this one untested.
    """
    import types

    td = tempfile.mkdtemp()
    out = os.path.join(td, "corpus", "domA")
    os.makedirs(out)
    with open(os.path.join(out, "domA_000.jsonl"), "w") as f:
        f.write('{"content": "x"}\n')
    # plant a tokenizer at the REAL path _write_stats computes, and remove it after
    real_root = os.path.dirname(os.path.dirname(os.path.abspath(B.__file__)))
    tok = os.path.join(real_root, "data", "tokenizer.json")
    planted = not os.path.exists(tok)

    class _Boom:
        @staticmethod
        def from_file(p):
            raise RuntimeError("forced: tokenizer unreadable")

    saved_tok = sys.modules.get("tokenizers")
    # INSTALL THE FAKE tokenizers. Without this the count fails with ModuleNotFoundError
    # instead of the forced RuntimeError -- the test still passes (both land in `except`)
    # but for the wrong reason, and a reader would believe the count path was exercised
    # when only the import was. `types` is imported for this line alone.
    sys.modules["tokenizers"] = types.SimpleNamespace(Tokenizer=_Boom)
    saved_settle, B.SETTLE_S = B.SETTLE_S, 0

    class _A:
        domain = "domA"; filters = "light"; workers = 1
        no_near_dedup = True; phase = None; allow_empty_slice = False

    try:
        # THE PLANT IS INSIDE THE try. It used to sit above it, so anything raising between
        # the plant and the try leaked data/tokenizer.json into the repo tree -- gitignored,
        # so it would persist silently and the NEXT run would see planted=False and clean up
        # nothing. Everything that mutates state this function must restore belongs here.
        if planted:
            os.makedirs(os.path.dirname(tok), exist_ok=True)
            with open(tok, "w") as f:
                f.write("{}")
        B._write_stats(out, "domA", _A(), {}, 1, 5, 1)
        with open(os.path.join(out, "build_corpus_stats.json")) as f:
            st = json.load(f)
        missing = [k for k in B.CANONICAL_STATS_KEYS if k not in st]
        assert not missing, (
            f"the tokenizer-failure path dropped canonical keys {missing}; a stamp that "
            f"cannot carry a count must still carry the same SHAPE")
        assert st["tokens"] is None, (
            f"an unmeasured count must be None, not {st['tokens']!r} -- 0 is an int and "
            f"count_dir would read it as a measured zero")
        assert "unmeasured" in st["tokens_status"], st["tokens_status"]
        # WHICH exception, not just that one happened. Without the fake tokenizers module
        # installed above, _write_stats would fail at `from tokenizers import Tokenizer`
        # with ModuleNotFoundError -- the same `except`, the same green, and a reader
        # believing the count path ran when only the import did. Measured 2026-09-18: a
        # refactor that dropped that install line left this test passing for exactly that
        # wrong reason. The RuntimeError is the one this test forces.
        assert "RuntimeError" in st["tokens_status"], (
            f"the count failed for the wrong reason (expected the forced RuntimeError): "
            f"{st['tokens_status']}")
    finally:
        B.SETTLE_S = saved_settle
        if saved_tok is None:
            sys.modules.pop("tokenizers", None)
        else:
            sys.modules["tokenizers"] = saved_tok
        if planted:
            os.remove(tok)
        shutil.rmtree(td, ignore_errors=True)


def main():
    # three shards; doc 0 x2 (exact dup across shard 0), UNIQ x2 (across shard 0/1),
    # UNIQ x1 (shard 2), one real holdout doc (HOLD sentinel is NOT a real holdout by
    # itself, so REAL_HOLD is what the slice captures under phase="t").
    shards = [
        [{"content": DUP, "url": "u"}, {"content": DUP, "url": "u"},
         {"content": UNIQ, "url": "u"}, {"content": HOLD, "url": "u"}, {"content": REAL_HOLD, "url": "u"}],
        [{"content": UNIQ, "url": "u"}, {"content": "another unique body is here and long", "url": "u"}],
        [{"content": "another unique body is here and long", "url": "u"}],
    ]
    outs = []
    for par in (False, True):
        d = tempfile.mkdtemp()
        w = os.path.join(d, "w")
        os.makedirs(w)
        for i, ss in enumerate(shards):
            with open(os.path.join(w, f"w{i}__000.jsonl"), "w") as f:
                for doc in ss:
                    f.write(json.dumps(doc) + "\n")
        a = argparse.Namespace(
            domain="t", out=w, source=[], filters="light", no_near_dedup=True,
            workers=3 if par else 1, global_only=True, dry=False, exclude=[],
            limit=None, rg_mod=None, rg_idx=None, cache_dir=None, phase="t",
        )
        if par:
            B._parallel_exact_pass(a)
        else:
            B._global_pass(a)
        # H3 (e1): end-to-end --phase coverage -- a real held-out doc captured in the
        # pass, through _write_stats, into the frozen slice, row count == stamp's reason.
        sp = os.path.join(w, "holdout_slice_t.jsonl")
        assert os.path.exists(sp), f"phase='t' did not emit a holdout slice at {sp}"
        with open(sp, encoding="utf-8") as _sf:
            slice_rows = sum(1 for _ in _sf) - 1  # minus the header line
        with open(os.path.join(w, "build_corpus_stats.json"), encoding="utf-8") as _st:
            stats = json.load(_st)
        assert slice_rows == stats["reasons"]["holdout"], (
            f"slice rows {slice_rows} != stamp holdout reason {stats['reasons']['holdout']}"
        )
        merged = os.path.join(w, "t_000.jsonl")
        with open(merged, "rb") as f:
            got = f.read()  # raw bytes: byte-identity is an exact-file comparison
        outs.append((d, w, got))
    (d1, w1, serial) = outs[0]
    (d2, w2, parallel) = outs[1]
    assert serial == parallel, (
        f"BYTE MISMATCH ({len(serial)} vs {len(parallel)} bytes)\n"
        f"serial:   {serial[:200]}\nparallel: {parallel[:200]}"
    )
    # sanity: the cross-shard duplicate (UNIQ) appears exactly once across all
    # output lines, proving global exact-dedup ran (not per-shard only).
    uniq_hits = serial.count(b"def g(x): return x * 2")
    assert uniq_hits == 1, f"expected UNIQ deduped to 1 occurrence, got {uniq_hits}"
    n_lines = serial.count(b"\n")
    check_token_count_failure_path()
    print(f"ok: serial==parallel byte-identical ({len(serial)}B, {n_lines} lines); UNIQ deduped to 1")
    shutil.rmtree(d1, ignore_errors=True)
    shutil.rmtree(d2, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

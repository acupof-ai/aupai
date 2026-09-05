#!/usr/bin/env python3
"""A CONFIGURED cache directory that is missing a cache must REFUSE, never retokenize.

WHY THIS TEST EXISTS. The token caches moved to /mnt/data02/tokens on 2026-09-05, an NVMe
filesystem attached into the container with move_mount, and that mount lives exactly as long as the
container. A restart drops it and leaves the mount POINT behind as an ordinary EMPTY DIRECTORY on
the overlay. Nothing errors. Before this refusal, train.py's freshness branch would find no cache,
retokenize the whole mix onto a rotational disk at 87% full, and the first symptom would be the disk
filling hours later.

THE WORLD IS THE FAILURE, not a proxy for it: a directory that EXISTS and is EMPTY, which is exactly
what a dropped mount leaves. A world with a nonexistent directory would be a different and easier
case -- the operator typo -- and it is tested too, second, because the refusal must not be
accidentally specific to one of them.

THE NEGATIVE CONTROL IS THE POINT. With AUPAI_TOKEN_CACHE_DIR UNSET, the same absent cache must
still tokenize: the default path is this repo's own history, and a first-ever build has to work. A
refusal that fires in both cases would break every fresh checkout, so a test that only asserts the
refusal would pass on a change that bricks the repo.

    python3 scripts/test_cache_absent_refusal.py --selftest

# restartable: a selftest over a tempdir fixture, under a second end to end. An interrupt loses
# nothing -- the tempdir is removed in a finally and there is no artifact to resume from.
"""

import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _selftest():
    sys.path.insert(0, ROOT)
    import train

    fails = []
    d = tempfile.mkdtemp(prefix="cache_absent_")
    old_env = os.environ.pop("AUPAI_TOKEN_CACHE_DIR", None)
    old_data, old_cache = train.DATA, train.TOKEN_CACHE
    old_vid = train.VOCAB_ID
    try:
        # A corpus dir with one shard, so `shards` is non-empty and the branch under test is the
        # cache's absence rather than a missing corpus. Without this the assert on `texts` fires
        # first and the test would pass for the wrong reason.
        data = os.path.join(d, "data")
        dom = "probe_absent"
        ddir = os.path.join(data, "corpus", dom)
        os.makedirs(ddir)
        with open(os.path.join(ddir, "shard_000.jsonl"), "w", encoding="utf-8") as f:
            # BOTH "text" and "content": _jsonl_content reads one of them and which one is not this
            # test's subject. Measured -- with only "text" the rebuild path raised KeyError('content')
            # and the mutation that DELETES the refusal was caught by that crash rather than by the
            # assertion, which is a pass for the wrong reason: the same crash would mask a real
            # regression on a day the refusal was fine.
            f.write('{"text": "def f():\\n    return 1\\n", '
                    '"content": "def f():\\n    return 1\\n"}\n')
        train.DATA = data
        train.VOCAB_ID = "deadbeef"

        empty = os.path.join(d, "mnt_data02_tokens")
        os.makedirs(empty)

        class _Tok:
            """Enough of a tokenizer to get THROUGH the rebuild, not merely into it.

            This matters more than it looks. With a thinner stub, the mutation that DELETES the
            refusal was caught by a crash inside the rebuild -- first KeyError('content'), then
            AttributeError on a missing method -- so the assertion below never ran and the case
            passed for the wrong reason. A crash that happens to be red today masks a real
            regression on a day the refusal is fine, which is the whole failure mode this file
            exists to test for elsewhere. train.encode calls token_to_id("<eos>") and
            encode_batch_fast/encode_batch, so both are here.
            """

            def token_to_id(self, s):
                return 1

            def encode_batch(self, texts):
                class _E:
                    ids = [7, 8, 9]

                return [_E() for _ in texts]

            encode_batch_fast = encode_batch

            def encode(self, s):
                class _E:
                    ids = [7, 8, 9]

                return _E()

        cases = [
            ("the directory EXISTS and is EMPTY (a dropped mount)", empty),
            ("the directory does NOT exist (an operator typo)", os.path.join(d, "nope")),
        ]
        for label, cache_dir in cases:
            os.environ["AUPAI_TOKEN_CACHE_DIR"] = cache_dir
            try:
                train._domain_seqs(dom, _Tok(), is_main=True, ddp=False, workers=1)
                fails.append(f"{label}: NO REFUSAL -- it would retokenize the whole mix into "
                             f"{cache_dir}")
            except RuntimeError as e:
                msg = str(e)
                for want in ("refusing to retokenize", "ABSENT", "attach_nvme_caches",
                             "AUPAI_TOKEN_CACHE_DIR"):
                    if want not in msg:
                        fails.append(f"{label}: the refusal does not mention {want!r}, so a reader "
                                     f"cannot act on it: {msg[:140]}")
            except Exception as e:  # noqa: BLE001 - a wrong-reason failure must be named as one
                fails.append(f"{label}: raised {type(e).__name__} rather than the refusal, so this "
                             f"case proves nothing: {str(e)[:140]}")

        # THE NEGATIVE CONTROL. Unset, the same absent cache must proceed to tokenize. The call is
        # allowed to fail LATER on the stub tokenizer -- what matters is that it does not fail with
        # the refusal, which would mean a fresh checkout can never build its first cache.
        os.environ.pop("AUPAI_TOKEN_CACHE_DIR", None)
        # This test's SUBJECT is the accessor, so it must set the literal the accessor falls back
        # to. Calling _token_cache_dir() here would read the real /data00 or the NVMe mount and
        # make the negative control depend on live disk state.
        train.TOKEN_CACHE = os.path.join(d, "default", "pretrain_1b_tokens.pt")  # cache-path-ok: see above
        os.makedirs(os.path.dirname(train.TOKEN_CACHE), exist_ok=True)
        try:
            train._domain_seqs(dom, _Tok(), is_main=True, ddp=False, workers=1)
        except RuntimeError as e:
            if "refusing to retokenize" in str(e):
                fails.append("NEGATIVE CONTROL FAILED: the refusal fires with "
                             "AUPAI_TOKEN_CACHE_DIR unset, so no fresh checkout could ever build "
                             "its first cache")
        except Exception:
            pass  # any other failure is the stub tokenizer, not the refusal
    finally:
        train.DATA, train.TOKEN_CACHE, train.VOCAB_ID = old_data, old_cache, old_vid
        if old_env is not None:
            os.environ["AUPAI_TOKEN_CACHE_DIR"] = old_env
        else:
            os.environ.pop("AUPAI_TOKEN_CACHE_DIR", None)
        shutil.rmtree(d, ignore_errors=True)

    for f in fails:
        print(f"  FAIL {f}", file=sys.stderr)
    if fails:
        print(f"test_cache_absent_refusal: {len(fails)} failure(s)", file=sys.stderr)
        return 1
    print("test_cache_absent_refusal ok: a configured cache dir missing a cache REFUSES with the "
          "path and the attach command, for both an empty directory (a dropped mount) and a "
          "nonexistent one; with the variable unset the same absent cache still tokenizes, so a "
          "fresh checkout is unaffected")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else 0)

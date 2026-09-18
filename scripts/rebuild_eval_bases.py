#!/usr/bin/env python3
"""Rebuild the two decontam benchmark bases from their upstream sources.

WHY THIS EXISTS. `scripts/filter_gate_domains.py` (stage 2c of the rebuild recipe) exits
immediately unless both files exist, and it does NOT create them in production -- its own
writes are `--selftest` temp fixtures. The 2026-09-16 pod loss destroyed them and they are
`[EXTERNAL-FETCH]`, so a rebuild re-derives them from upstream. Both files also feed
`decontam_fp` (filter_gate_domains.py:166), the fingerprint downstream trust keys on, so
the CONTENT is the point: a hand-edited or wrong-population file is a silently different
filter, and "it parsed" is not evidence.

  data/eval/humaneval/humaneval_164.jsonl  <- openai/human-eval HumanEval.jsonl.gz
  data/eval/mbpp_holdouts.jsonl            <- google-research mbpp.jsonl (974, train)

THE SECOND PATH IS NOT THE SANITIZED SUBSET, and assuming it was cost a full day. The two
MBPP artifacts are different benchmark populations consumed through one path:

  data/eval/mbpp_holdouts.jsonl  974 rows, MBPP-*train*, ids 1..974, key `text`
                                 (REGISTRY `mbpp_holdouts_974`; runs/v2_data_proposal.md:208)
  data/eval/sanitized-mbpp.json  427 rows, sanitized eval set, key `prompt`
                                 (REGISTRY `mbpp_sanitized_427`)

They share ZERO task_ids and only 190/427 prompt prefixes, so writing one where the other
belongs produces a file with the right shape and the wrong questions -- and passes any
check that only counts rows. `--verify` settles it: every `text` in the written file must
hash into the tracked registry, which is a property neither count nor schema can fake.

    python3 scripts/rebuild_eval_bases.py --humaneval-gz /tmp/HumanEval.jsonl.gz \\
        --mbpp-jsonl /tmp/mbpp.jsonl --verify --sha-out /tmp/UPSTREAM_SHA256
    python3 scripts/rebuild_eval_bases.py --selftest
"""
# restartable: both inputs are local files and the outputs are two small jsonl files
# written whole; an interrupt loses nothing (no corpus bytes, no shards, no partial state).
import argparse
import gzip
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HE_OUT = os.path.join(ROOT, "data", "eval", "humaneval", "humaneval_164.jsonl")
MBPP_OUT = os.path.join(ROOT, "data", "eval", "mbpp_holdouts.jsonl")
HOLDOUT = os.path.join(ROOT, "data", "eval", "holdout_hashes.txt")
HE_FIELDS = ("task_id", "prompt", "canonical_solution", "test", "entry_point")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def convert_humaneval(gz_path):
    """canonical HumanEval.jsonl.gz -> the repo's one-row-per-task jsonl.

    The canonical release already carries every field the repo reads
    (`eval/humaneval_gen.py`: prompt/canonical_solution/test/entry_point;
    `filters/decontam_ngram.py`: task_id/prompt/canonical_solution/test), so this is a
    passthrough that ASSERT the contract rather than a mapping that assumes it. The
    assertion is the point: a converter that silently drops `test` yields a filter with
    no test-side grams and a decontamination that reads as working.
    """
    with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()]
    missing = [r.get("task_id") for r in rows if any(f not in r for f in HE_FIELDS)]
    if missing:
        raise SystemExit(f"humanEval rows missing {HE_FIELDS}: {missing[:5]}")
    ids = {r["task_id"] for r in rows}
    want = {f"HumanEval/{i}" for i in range(164)}
    if ids != want:
        raise SystemExit(
            f"humanEval is not the canonical 164: {len(ids)} ids, "
            f"missing {sorted(want - ids)[:5]}, extra {sorted(ids - want)[:5]}"
        )
    rows.sort(key=lambda r: int(r["task_id"].split("/")[1]))
    return rows


def convert_mbpp_holdouts(mbpp_jsonl, want=974):
    """upstream mbpp.jsonl -> data/eval/mbpp_holdouts.jsonl, near-passthrough.

    NOT the sanitized-427 subset, and getting this wrong is silent. `datagen/holdout.py`'s
    REGISTRY entry for this path is `mbpp_holdouts_974`: 974 rows of MBPP-*train*, ids
    `mbpp-train-0..973`, `question_field: ["text"]`, documented at
    `runs/v2_data_proposal.md:208` -- which also records that it has ZERO task_id
    intersection with sanitized-427 and only 190/427 prompt-prefix overlap. They are two
    different benchmark populations that happen to be consumed through one path, and the
    earlier version of this file wrote the 338-row sanitized subset here: right row count
    for the wrong question, feeding `filter_gate_domains.py:166`'s `decontam_fp`.

    The id scheme is bare `1..974` in upstream, while the REGISTRY prose says
    `mbpp-train-N`. Both consumers (`filters/decontam_ngram.py:113`,
    `scripts/audit_gate_contamination.py:64`) only interpolate `r['task_id']` into a key
    string and never parse the prefix, so upstream's own ids are what is written -- the
    prose describes the file, it is not a transform to apply.

    `--verify` is the real judge: every row's `text` must hash into the tracked registry.
    """
    rows = [json.loads(ln) for ln in open(mbpp_jsonl, encoding="utf-8") if ln.strip()]
    if len(rows) != want:
        raise SystemExit(
            f"mbpp.jsonl has {len(rows)} rows, expected {want} -- is this the "
            f"upstream train set, or the sanitized 427?"
        )
    missing = [r.get("task_id") for r in rows if "text" not in r or "task_id" not in r]
    if missing:
        raise SystemExit(f"rows missing task_id/text: {missing[:5]}")
    ids = sorted(int(r["task_id"]) for r in rows)
    if ids != list(range(1, want + 1)):
        raise SystemExit(f"mbpp ids are not 1..{want}: {ids[:5]} ... {ids[-3:]}")
    rows.sort(key=lambda r: int(r["task_id"]))
    return rows


def write_jsonl(path, rows, ensure_ascii=True):
    """Serialise rows, preserving the upstream escaping convention by default.

    `ensure_ascii=True` is the default because it is what BOTH upstream sources emit, and it
    is not cosmetic: the file's sha1 is recorded in `datagen/holdout.py`'s REGISTRY_SHA1, and
    `decontam_fp` hashes its bytes. Measured on mbpp (2026-09-18), re-serialising the same
    974 objects with `ensure_ascii=False` produced 563,678 bytes against upstream's 563,743 --
    a 65-byte difference, exactly the `\\u2019` -> `’` expansion of the apostrophes in the
    corpus. Same objects, same keys, different bytes, different fingerprint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=ensure_ascii) + "\n")
    return sha256_file(path)


def verify():
    """Row counts, then the tracked registry body. Returns (ok, lines).

    The registry check uses `datagen.holdout.qhash` -- the SAME function the guard uses --
    rather than a guess at the hashing scheme. The first version of this function tried
    sha1/sha256/md5 of the raw text and reported "0/164 matched, do NOT proceed" on two
    files that were in fact correct: qhash normalises whitespace and CJK punctuation before
    hashing, so a raw-text guess matches nothing. A verifier that cannot reproduce the
    property it checks reports every correct input as wrong.
    """
    sys.path.insert(0, ROOT)
    from datagen.holdout import qhash

    out = []
    for path, want, field in ((HE_OUT, 164, "prompt"), (MBPP_OUT, 974, "text")):
        if not os.path.exists(path):
            return False, out + [f"MISSING {path}"]
        n = sum(1 for _ in open(path, encoding="utf-8"))
        out.append(f"{os.path.relpath(path, ROOT)}: {n} row(s) (want {want})")
        if n != want:
            return False, out + [f"row count mismatch: {n} != {want}"]
    if not os.path.exists(HOLDOUT):
        return True, out + [
            f"holdout registry absent ({os.path.relpath(HOLDOUT, ROOT)}) "
            f"-- row counts only, content UNVERIFIED"
        ]
    reg = set(l.strip() for l in open(HOLDOUT, encoding="utf-8") if l.strip() and not l.startswith("#"))
    for path, want, field in ((HE_OUT, 164, "prompt"), (MBPP_OUT, 974, "text")):
        hit = 0
        for ln in open(path, encoding="utf-8"):
            if qhash(json.loads(ln).get(field, "")) in reg:
                hit += 1
        out.append(f"  {os.path.basename(path)}: {hit}/{want} {field}(s) in the registry")
        if hit != want:
            return False, out + [
                f"{os.path.basename(path)}: only {hit}/{want} matched -- this is NOT the "
                f"file the registry was built from. Both MBPP populations are consumed "
                f"through one path; check that this holds the 974-row train set (key `text`) "
                f"and not sanitized-427 (key `prompt`). Do not use this file."
            ]
    return True, out


def _selftest():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        gz = os.path.join(d, "HumanEval.jsonl.gz")
        with gzip.open(gz, "wt", encoding="utf-8") as fh:
            for i in range(164):
                fh.write(
                    json.dumps(
                        {
                            "task_id": f"HumanEval/{i}",
                            "prompt": f"def f{i}():\n",
                            "canonical_solution": "    return 1\n",
                            "test": "def check():\n    pass\n",
                            "entry_point": f"f{i}",
                        }
                    )
                    + "\n"
                )
        rows = convert_humaneval(gz)
        assert len(rows) == 164 and rows[0]["task_id"] == "HumanEval/0", rows[0]
        assert rows[-1]["task_id"] == "HumanEval/163"

        # A row missing `test` must REFUSE: that field feeds the decontam test-grams, and
        # a passthrough that drops it produces a filter that still "works".
        bad = os.path.join(d, "bad.jsonl.gz")
        with gzip.open(bad, "wt", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {"task_id": "HumanEval/0", "prompt": "p", "canonical_solution": "c", "entry_point": "f"}
                )
                + "\n"
            )
        try:
            convert_humaneval(bad)
            raise AssertionError("a row missing `test` was accepted")
        except SystemExit as e:
            assert "missing" in str(e), e

        # An incomplete id set must REFUSE, not silently convert a partial download.
        short = os.path.join(d, "short.jsonl.gz")
        with gzip.open(short, "wt", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "task_id": "HumanEval/0",
                        "prompt": "p",
                        "canonical_solution": "c",
                        "test": "t",
                        "entry_point": "f",
                    }
                )
                + "\n"
            )
        try:
            convert_humaneval(short)
            raise AssertionError("a 1-of-164 file was accepted")
        except SystemExit as e:
            assert "not the canonical 164" in str(e), e

        # mbpp: near-passthrough of upstream train, but a WRONG POPULATION must refuse.
        # The world that matters is the one this script got wrong the first time: a
        # sanitized-427 file (key `prompt`, 427 rows) offered at this path. It is the right
        # shape and the wrong questions, so the row-count check alone is what has to catch it.
        ok974 = os.path.join(d, "mbpp974.jsonl")
        with open(ok974, "w", encoding="utf-8") as fh:
            for i in range(1, 975):
                fh.write(json.dumps({"task_id": i, "text": f"t{i}", "code": f"c{i}"}) + "\n")
        rows = convert_mbpp_holdouts(ok974)
        assert len(rows) == 974 and rows[0]["task_id"] == 1, rows[:2]

        san427 = os.path.join(d, "san.jsonl")
        with open(san427, "w", encoding="utf-8") as fh:
            for i in range(427):
                fh.write(json.dumps({"task_id": i, "prompt": f"p{i}"}) + "\n")
        try:
            convert_mbpp_holdouts(san427)
            raise AssertionError("a 427-row sanitized file was accepted as the 974 train set")
        except SystemExit as e:
            assert "expected 974" in str(e), e

        # A train file missing ids must refuse rather than write a hole into decontam_fp.
        gap = os.path.join(d, "gap.jsonl")
        with open(gap, "w", encoding="utf-8") as fh:
            for i in range(1, 975):
                if i == 500:
                    continue
                fh.write(json.dumps({"task_id": i, "text": f"t{i}"}) + "\n")
            fh.write(json.dumps({"task_id": 9999, "text": "x"}) + "\n")
        try:
            convert_mbpp_holdouts(gap)
            raise AssertionError("a non-contiguous id set was accepted")
        except SystemExit as e:
            assert "not 1..974" in str(e), e

        print(
            "rebuild_eval_bases selftest OK: humaneval asserts the 164-id set and every "
            "consumed field; mbpp_holdouts refuses both a 427-row sanitized file and a "
            "non-contiguous id set -- the two ways the wrong population gets in without "
            "failing a schema check"
        )
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--humaneval-gz", help="upstream HumanEval.jsonl.gz")
    ap.add_argument("--mbpp-jsonl", help="upstream mbpp.jsonl (974-row train set)")
    ap.add_argument("--verify", action="store_true", help="check the written files")
    ap.add_argument("--sha-out", help="write a sha256 record (URL + hash per file)")
    a = ap.parse_args()

    wrote = {}
    if a.humaneval_gz:
        rows = convert_humaneval(a.humaneval_gz)
        wrote["humaneval_164.jsonl"] = (HE_OUT, write_jsonl(HE_OUT, rows), len(rows))
    if a.mbpp_jsonl:
        rows = convert_mbpp_holdouts(a.mbpp_jsonl)
        wrote["mbpp_holdouts.jsonl"] = (MBPP_OUT, write_jsonl(MBPP_OUT, rows), len(rows))
    for name, (path, h, n) in wrote.items():
        print(f"wrote {os.path.relpath(path, ROOT)}  {n} rows  sha256 {h}")
    if a.sha_out:
        with open(a.sha_out, "w", encoding="utf-8") as fh:
            for name, (path, h, n) in sorted(wrote.items()):
                fh.write(f"{h}  {n} rows  {name}\n")
        print(f"sha record -> {a.sha_out}")
    if a.verify or (not a.humaneval_gz and not a.mbpp_jsonl):
        ok, lines = verify()
        for ln in lines:
            print("  " + ln)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else main())

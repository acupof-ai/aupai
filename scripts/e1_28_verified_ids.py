#!/usr/bin/env python3
"""e1-28: the VERIFIED-CLEAN population (5,409 ids) -- measurable at n=13 and negative.

# restartable: reads the answers once and writes one 5,409-line file at the end. An
# interrupt costs ~20s of tokenizing and no partial output -- the write is the last
# statement, so there is no half-written population to mistake for a complete one.

6e's ruling 2026-09-04: 5,409 is the floor's population, because it is the only one whose
name is true. 8,307 stays reportable beside it as "not known dirty, 2,898 untestable at
n=13" and never as a floor.

  verified clean = { id : len(words(answer)) >= 13  AND  id not in hit_ids }

MEASURABILITY IS A PROPERTY OF THE HELD-OUT ANSWERS, NOT OF THE CORPUS. An item can only hit
if its answer has at least one 13-gram, i.e. at least 13 word tokens
(e1_28_heldout_contamination.py:329). So this needs no --classify pass and no corpus read --
it reuses that script's own words() and N so the two agree by construction rather than by a
copied regex. The count is asserted against the scan artifact's measurable_denominator; a
mismatch means the tokenizer or N drifted and the ids would be a different population
wearing the same name.

RUNS ON THE POD, because data/sft/control_sft_text_heldout.jsonl lives only there (16,190,349
bytes, mtime 2026-09-02 16:12). The answers are the input; nothing in the repo carries them.

    CUDA_VISIBLE_DEVICES= python3 scripts/e1_28_verified_ids.py --hits <scan.json> --write

Cardless. Writes runs/heldout_v2/ids_clean_v3_verified.txt and prints its digest under BOTH
recipes -- the scanner's sorted-strings sha256 and eval_heldout.ids_sha -- because this
repository has two and "the fingerprint" has no referent (see e1_28_clean_ids.py).
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = os.path.join(ROOT, "runs", "heldout_v2", "ids_shared.txt")
HELDOUT = os.path.join(ROOT, "data", "sft", "control_sft_text_heldout.jsonl")
OUT = os.path.join(ROOT, "runs", "heldout_v2", "ids_clean_v3_verified.txt")

EXPECT_POP = 10421
EXPECT_HITS = 2114
EXPECT_MEASURABLE = 7523
EXPECT_SHORT = 2898
EXPECT_VERIFIED = 5409


def scan_fp(ids):
    """The scanner's recipe: sha256 over the sorted ids AS STRINGS, newline-joined."""
    return hashlib.sha256("\n".join(sorted(str(i) for i in ids)).encode()).hexdigest()[:16]


def ids_sha_of(ids):
    """eval_heldout's recipe -- imported, never reimplemented: it is what the scorers write
    as evaluated_ids_sha256, and guessing it once already produced e64914b26d8562f3 against
    the real cae4daf7ad59388c."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from eval_heldout import ids_sha  # noqa: PLC0415
    return ids_sha(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hits", required=True,
                    help="e1_28_heldout_contamination.json (carries hit_ids and the counts "
                         "this script asserts against)")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from e1_28_heldout_contamination import N, words  # noqa: PLC0415 -- the scan's own tokenizer

    with open(a.hits, encoding="utf-8") as fh:
        d = json.load(fh)
    if d["population"] != EXPECT_POP or d["answer_hits"] != EXPECT_HITS:
        sys.exit(f"REFUSING: {a.hits} is population {d['population']} hits {d['answer_hits']}, "
                 f"expected {EXPECT_POP} {EXPECT_HITS}")
    hits = {int(i) for i in d["hit_ids"]}

    with open(SHARED, encoding="utf-8") as fh:
        shared = [int(x) for x in fh if x.strip()]
    if len(shared) != EXPECT_POP:
        sys.exit(f"REFUSING: {SHARED} holds {len(shared)}, expected {EXPECT_POP}")
    keep = {str(i) for i in shared}

    # STRING COMPARE ON BOTH SIDES: ids_shared.txt holds "0" while the JSONL holds the int 0,
    # and that mismatch once matched 0 of 10,421 rows -- an empty index that would have
    # reported a clean bill of health from comparing nothing.
    rows = []
    with open(HELDOUT, encoding="utf-8") as fh:
        for ln in fh:
            r = json.loads(ln)
            if str(r["id"]) in keep:
                rows.append(r)
    if len(rows) != EXPECT_POP:
        sys.exit(f"REFUSING: matched {len(rows)} rows of {EXPECT_POP} in {HELDOUT} -- the "
                 f"population and the answers are not the same set")

    short = {int(r["id"]) for r in rows if len(words(r["answer"])) < N}
    measurable = {int(r["id"]) for r in rows} - short
    if len(short) != EXPECT_SHORT or len(measurable) != EXPECT_MEASURABLE:
        sys.exit(f"REFUSING: {len(short)} short / {len(measurable)} measurable at n={N}, "
                 f"expected {EXPECT_SHORT} / {EXPECT_MEASURABLE}. The tokenizer or N drifted "
                 f"from the scan, so these ids would be a different population under the "
                 f"same name.")
    if hits - measurable:
        sys.exit(f"REFUSING: {len(hits - measurable)} hit id(s) are not measurable -- a hit "
                 f"requires a 13-gram, so this contradicts the scan")

    verified = sorted(measurable - hits)
    if len(verified) != EXPECT_VERIFIED:
        sys.exit(f"REFUSING: {len(verified)} verified-clean, expected {EXPECT_VERIFIED}")

    notknown = len(shared) - len(hits)
    print(f"population        {len(shared):,}")
    print(f"  measurable      {len(measurable):,}  (answer >= {N} words)")
    print(f"  untestable      {len(short):,}  (answer < {N} words; no {N}-gram exists)")
    print(f"  hits            {len(hits):,}  ({len(hits) / len(measurable):.2%} of measurable)")
    print()
    print(f"VERIFIED CLEAN    {len(verified):,} = {len(measurable):,} - {len(hits):,}   "
          f"<- the floor's population (6e ruling 2026-09-04)")
    print(f"  ids_sha         {ids_sha_of(verified)}   (eval_heldout.ids_sha; scorers write "
          f"this as evaluated_ids_sha256)")
    print(f"  scan fp         {scan_fp(verified)}   (sorted-strings sha256, the scanner's recipe)")
    print()
    print(f"not known dirty   {notknown:,} = {len(shared):,} - {len(hits):,}, of which "
          f"{len(short):,} untestable at n={N}")
    print("  reportable beside the floor, NEVER as a floor")

    if not a.write:
        print("\n(no file written; pass --write)")
        return 0
    with open(OUT, "w", encoding="utf-8") as fh:
        for i in verified:
            fh.write(f"{i}\n")
    print(f"\nwrote {OUT}  n={len(verified):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

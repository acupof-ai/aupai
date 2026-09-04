#!/usr/bin/env python3
"""e1-28: the clean-subset populations for the floor re-score, and why there are two of them.

runs/heldout_v2/ids_clean.txt holds 10,105 ids -- 10,421 minus the 316-item exclusion that
scripts/e1_28_leak_scan.py produced. That scan is SUPERSEDED (facts/contamination.json
#cont.heldout_in_pretrain_corpus): it read 5.0-13.2% of each domain's documents because it
treated token-block cursor rows as documents, and its coverage guard was in the same wrong
unit so it never fired. The whole-corpus scan found 2,114 hits, not 316. So every number
scored against ids_clean.txt answers a question about the wrong population.

TWO populations, and the difference is not cosmetic:

  8,307 = 10,421 - 2,114 hits.  "Not known to be contaminated."
          2,898 of these items have an answer under 13 words, so they have no 13-gram and
          the scan could not test them AT ALL. They are unknown, not clean. This is the
          population that matches how ids_clean.txt was built (drop the hits, keep the rest).

  5,409 = 7,523 measurable - 2,114 hits.  "Verified clean."
          Every item here was actually tested and came back negative. Smaller and it is the
          only set whose name is true.

Which one the re-score should use is a judgement about what claim the floor is meant to
support, so this script reports both and refuses to pick. What it will not do is write one
file called "clean" and leave the reader to assume the other.

THE OLD EXCLUSION IS NOT A SUBSET OF THE NEW ONE. 296 of the 316 are among the 2,114, and
20 are not -- items the superseded scan dropped that the whole-corpus scan says are fine.
So 8,307 is not "10,105 minus more"; the two sets cross. A re-score cannot be compared to
the published 0.457462 as if the population had merely shrunk.

    CUDA_VISIBLE_DEVICES= python3 scripts/e1_28_clean_ids.py --write

Cardless. Writes ONE file, runs/heldout_v2/ids_clean_v2_notknown.txt (8,307 ids), and prints
its eval_heldout.ids_sha digest -- the one the scorers write as evaluated_ids_sha256. The
5,409 verified-clean set is reported as a COUNT and not written: the scan artifact does not
list which items were measurable, so the ids need --classify to recompute. Writing a file
of the 8,307 under a "verified" name would be the mislabelling this script exists to avoid.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN = os.path.join(ROOT, "runs", "e1_28", "e1_28_heldout_contamination.json")
SHARED = os.path.join(ROOT, "runs", "heldout_v2", "ids_shared.txt")
OLD_CLEAN = os.path.join(ROOT, "runs", "heldout_v2", "ids_clean.txt")
OUT_DIR = os.path.join(ROOT, "runs", "heldout_v2")

# From the scan artifact, asserted rather than trusted: a file with this name but different
# contents would define a different population and look exactly like a result.
#
# TWO DIGESTS OVER ONE POPULATION, and they are not interchangeable. The same 10,421 ids hash
# to c3895c21973bf49b under the scanner's recipe (sha256 of the sorted ids as STRINGS, joined
# by newline -- scripts/e1_28_heldout_contamination.py:92) and to cae4daf7ad59388c under
# eval_heldout.ids_sha, which is what every scorer writes as evaluated_ids_sha256 and what the
# audit and the task text quote. My first version of this guard asserted the scanner's
# `fingerprint` field against the audit's value and REFUSED on the correct artifact. Both are
# pinned below, each against the recipe that produced it, because a single "the fingerprint"
# has no referent here.
EXPECT_POP = 10421
EXPECT_SCAN_FP = "c3895c21973bf49b"   # e1_28_heldout_contamination.json's own "fingerprint"
EXPECT_IDS_SHA = "cae4daf7ad59388c"   # eval_heldout.ids_sha, the scorers' evaluated_ids_sha256
EXPECT_HITS = 2114
EXPECT_MEASURABLE = 7523


def load():
    with open(SCAN, encoding="utf-8") as fh:
        d = json.load(fh)
    if d["population"] != EXPECT_POP or d["fingerprint"] != EXPECT_SCAN_FP:
        sys.exit(f"REFUSING: {SCAN} is population {d['population']} fp {d['fingerprint']}, "
                 f"expected {EXPECT_POP} {EXPECT_SCAN_FP}")
    if d["answer_hits"] != EXPECT_HITS or d["measurable_denominator"] != EXPECT_MEASURABLE:
        sys.exit(f"REFUSING: {SCAN} has {d['answer_hits']} hits over "
                 f"{d['measurable_denominator']} measurable, expected {EXPECT_HITS} over "
                 f"{EXPECT_MEASURABLE}")
    with open(SHARED, encoding="utf-8") as fh:
        shared = [int(x) for x in fh if x.strip()]
    if len(shared) != EXPECT_POP:
        sys.exit(f"REFUSING: {SHARED} holds {len(shared)}, expected {EXPECT_POP}")
    got = sha(shared)
    if got != EXPECT_IDS_SHA:
        sys.exit(f"REFUSING: {SHARED} digests to {got} under eval_heldout.ids_sha, expected "
                 f"{EXPECT_IDS_SHA} -- the scorers compare against this one, so a mismatch means "
                 f"the floors and this population are about different item sets")
    hits = set(d["hit_ids"])
    if len(hits) != EXPECT_HITS:
        sys.exit(f"REFUSING: hit_ids holds {len(hits)} distinct, expected {EXPECT_HITS}")
    if hits - set(shared):
        sys.exit(f"REFUSING: {len(hits - set(shared))} hit id(s) are outside the population")
    return d, shared, hits


def sha(ids):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from eval_heldout import ids_sha  # noqa: PLC0415 -- the scorers' own recipe, not a copy
    return ids_sha(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help="write runs/heldout_v2/ids_clean_v2_notknown.txt (the 8,307 set)")
    a = ap.parse_args()

    d, shared, hits = load()
    n_unmeasurable = d["answers_under_n_words"]

    notknown = [i for i in shared if i not in hits]
    # VERIFIED = measurable and negative. The scan does not list the measurable ids, but
    # measurable = population - answers_under_n_words and the hits are all measurable by
    # construction, so |verified| = measurable - hits is arithmetic. The IDS themselves need
    # the per-item measurability, which only --classify recomputes; until then the verified
    # population is a COUNT, not a file, and this script says so rather than writing a file
    # of the wrong ids under the right name.
    n_verified = d["measurable_denominator"] - len(hits)

    print(f"population            {len(shared):,}  fp {d['fingerprint']}")
    print(f"answer hits           {len(hits):,}  ({len(hits) / d['measurable_denominator']:.2%} "
          f"of the {d['measurable_denominator']:,} measurable)")
    print(f"unmeasurable          {n_unmeasurable:,}  (answer under 13 words, no 13-gram exists)")
    print()
    print(f"NOT-KNOWN-CONTAMINATED {len(notknown):,} = {len(shared):,} - {len(hits):,}")
    print(f"  contains {n_unmeasurable:,} items the scan could not test, so this set is "
          f"'not known to be dirty', NOT 'clean'")
    print(f"VERIFIED CLEAN         {n_verified:,} = {d['measurable_denominator']:,} - {len(hits):,}")
    print("  every item tested and negative; ids need --classify per-item measurability, "
          "so this is a count here and not a file")
    print()

    with open(OLD_CLEAN, encoding="utf-8") as fh:
        old = [int(x) for x in fh if x.strip()]
    old_excl = set(shared) - set(old)
    print(f"the superseded exclusion  {len(old_excl)} ids (runs/heldout_v2/ids_clean.txt, "
          f"{len(old):,} kept)")
    print(f"  {len(old_excl & hits)} of them are among the {len(hits):,} hits, "
          f"{len(old_excl - hits)} are NOT")
    print("  so the two exclusion sets CROSS: the new population is not the old one shrunk, "
          "and 0.457462 is not a baseline it can be compared against")
    print()

    if not a.write:
        print("(no files written; pass --write)")
        return 0

    p = os.path.join(OUT_DIR, "ids_clean_v2_notknown.txt")
    with open(p, "w", encoding="utf-8") as fh:
        for i in notknown:
            fh.write(f"{i}\n")
    print(f"wrote {p}  n={len(notknown):,}  sha {sha(notknown)}")
    print("  pass this to eval_heldout.py --ids; it prints evaluated_ids_sha256, which must match")
    return 0


if __name__ == "__main__":
    sys.exit(main())

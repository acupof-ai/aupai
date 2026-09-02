#!/usr/bin/env python3
"""The classification behind N_all is asymmetric, and the artifact it is read from is complete.

# restartable: in-process assertions only, no subprocess, no GPU. Milliseconds.

N_all is the verdict basis, so every way this can be wrong moves a published number. The two that
matter: a doubtful id must land INSIDE N_all (dropping a clean id only makes our own floor more
conservative, keeping a leaked one flatters us), and the per-id gram list must not be truncated --
an id whose only substantive gram fell past a cap would read as universal-only and be excluded.

    python3 scripts/test_e1_28_matched.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import e1_28_matched as M  # noqa: E402


def main():
    fails = []

    # 1. THE UNIVERSAL LIST EXCLUDES ONLY NO-ALTERNATIVE FORMS.
    hello = 'class HelloWorld { public static void main(String[] args) { System.out.println("Hello World"); } }'
    heron = 'area = sqrt(s * (s - a) * (s - b) * (s - c));'
    if not M.universal_only({hello}):
        fails.append("Hello World was not recognised as a universal form")
    if not M.universal_only({heron}):
        fails.append("Heron's formula was not recognised as a universal form")

    # 2. DOUBT COUNTS AS CONTAMINATION. The prime sieve and Kadane are standard algorithms whose
    #    variable names matched -- 1e ruled them into N_all rather than argued about. If they ever
    #    reach the UNIVERSAL list, the verdict silently gets a larger clean subset.
    for g in ('False for i in range(2, int(num**0.5) + 1): if num % i ==',
              'current_sum = max(array[i], current_sum + array[i]) max_sum = max(max_sum, current_sum)',
              '- 55个遗产地 2. 中国 - 55个遗产地 3. 西班牙 - 48个遗产地',
              '一群金色的水仙花； 在湖边，在树下， 在微风中飘动和跳舞。'):
        if M.universal_only({g}):
            fails.append(f"a doubtful/substantive gram was excluded from N_all: {g[:50]!r} -- "
                         f"doubt must count as contamination, not against it")

    # 3. ONE SUBSTANTIVE GRAM IS ENOUGH. An id flagged by Heron's formula AND a verbatim passage
    #    is contaminated; requiring `all` is what makes that true, and `any` would flip it.
    if M.universal_only({heron, '一群金色的水仙花； 在湖边，在树下， 在微风中飘动和跳舞。'}):
        fails.append("an id flagged by BOTH a universal form and real content was excluded -- "
                     "one substantive gram must be enough to keep it in N_all")

    # 4. AN EMPTY GRAM SET IS NOT 'UNIVERSAL'. all() over an empty set is True, which would quietly
    #    exclude any id whose grams went missing -- the empty-population failure again.
    if M.universal_only(set()):
        fails.append("an id with NO grams was classified universal-only: all() over an empty set "
                     "is True, so a missing gram list would exclude the id from N_all")

    # 5. THE ARTIFACT IS COMPLETE. ws_by_id must hold every gram, because the classification reads
    #    it. id 463700 was flagged by 46 grams, so a 40-cap would have silently dropped 6.
    src = open(os.path.join(ROOT, "scripts", "e1_28_matched.py")).read()
    if "sorted(gs)[:" in src:
        fails.append("ws_by_id is truncated with a slice -- an id whose only substantive gram "
                     "falls past the cap reads as universal-only and leaves N_all")

    # 6. N_all IS WRITTEN AS IDS, NOT JUST A COUNT, so the clean subset can actually be built and
    #    its own ids sha computed. A count alone cannot be subtracted from the scored population.
    for key in ("n_all_ids", "universal_only_ids"):
        if f'"{key}"' not in src:
            fails.append(f"the output does not carry {key}; a count cannot be turned into a "
                         f"clean subset with its own evaluated_ids_sha256")

    # 7. GUARD ARTIFACTS ARE NOT SHARDS. data/corpus/<domain>/ carries holdout_slice_<domain>.jsonl,
    #    whose one row is {"phase","rule_fp","n":0}. A bare *.jsonl glob fed it to text_of, which
    #    refused on the missing content field and killed the whole scan mid-run. e1_28_leak_scan.py
    #    has the same glob and survives only because holdout_slice sorts after the numbered shards
    #    and its row cap is reached first -- luck, not design.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "data", "corpus", "cot")
        os.makedirs(base)
        for name in ("cot_000.jsonl", "cot_001.jsonl",
                     "holdout_slice_cot.jsonl", "build_corpus_stats.jsonl"):
            open(os.path.join(base, name), "w").write("{}\n")
        got = [os.path.basename(p) for p in M.shards("cot", root=td)]
        if got != ["cot_000.jsonl", "cot_001.jsonl"]:
            fails.append(f"shards() returned {got}; guard artifacts must be excluded by name, "
                         f"not left to sort order")

    # 8. THE ROW CAP IS THE CURSOR'S, NOT UNBOUNDED. Reading every row on disk scans 232 GB instead
    #    of the 1,189,548 rows the run consumed, which is a different population and would report
    #    overlap from rows the model never saw.
    if "a.rows_per_shard or (cursor or {}).get(dom, 0)" not in src:
        fails.append("the read loop does not fall back to the cursor's per-domain row count, so "
                     "it scans the corpus on disk rather than the consumed population")


    # A FAILING RUN MUST SAY WHY. Inserting cases 7 and 8 above overwrote this loop, so the suite
    # returned 1 while printing nothing at all -- rc=1 with empty stdout AND stderr, which reads
    # exactly like a crash or a hang. I caught it only because a mutation I expected to fail
    # reported "0 FAILs": grep -c over empty output is 0, so the check that was supposed to prove
    # the test works reported the same number as a passing run.
    for f in fails:
        print(f"FAIL: {f}", file=sys.stderr)
    if fails:
        print(f"{len(fails)} check(s) failed", file=sys.stderr)
        return 1
    print("e1_28_matched OK: universal forms excluded; doubt counts as contamination; one "
          "substantive gram keeps an id; an empty gram set is not universal; ws_by_id untruncated; "
          "N_all written as ids; guard artifacts are not shards; the read cap is the cursor's")
    return 0


if __name__ == "__main__":
    sys.exit(main())

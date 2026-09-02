#!/usr/bin/env python3
"""WHICH strings matched? A hit count cannot distinguish leakage from boilerplate.

# restartable: streams the corpus and prints; writes one small JSON. An interrupt costs the scan
# (CPU only, minutes) and nothing else -- no checkpoint, no partial file a later run could mistake
# for complete.

THIS SCRIPT IS WHY THE CHARACTER WINDOW WAS THROWN OUT. e1_28_leak_scan.py reported chat_qa 816
character hits and chatml 816 -- the same count from two different corpora, which is a shared
template, not leakage. Counts cannot tell those apart; the matched STRINGS can:

    '-------------' x538   '_____________' x66   '-----|-------' x30

Markdown rules. After a low-entropy filter (MIN_CHARSET=4) killed those, what survived was still
boilerplate with plenty of distinct characters -- '```pythondefi', '`https://www.', '</tr><tr><td>'
-- and the ids it flagged were flagged for the SFT format's own opening template:

    id 450  '<Thought>\\nAlright, I need to write a Python program that...'
    id 1250 '<Thought>\\nAlright, I need to figure out how to determine...'

412 of 424 character-window ids in one shard were not flagged by the whitespace unit at all.
Thirteen characters cannot separate content from markup at any entropy threshold, so the character
unit is reported as evidence about ITSELF, never folded into a contamination count.

The whitespace unit is the one that holds. Its hits in that same shard were 12 ids, and reading
them settled each one: Wordsworth's daffodils translated (46 distinct grams -- a whole passage),
a cake recipe, the UNESCO world-heritage answer, a sphere surface-area derivation. Two were
boilerplate a human has to judge: a Hello World program, and Heron's formula.

SO THE OUTPUT IS THE MAPPING, NOT A NUMBER. Every flagged id is written with the grams that
flagged it, because "N ids are contaminated" is a claim nobody can check, and the substantive /
skeleton split is a judgement that must be reviewable rather than asserted.
"""
import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import e1_28_leak_scan as S  # noqa: E402

# TWO COUNTS, AND THE ASYMMETRY IS DELIBERATE (1e's ruling).
#
#   N_all          every whitespace-unit hit, minus only the strings that have exactly ONE possible
#                  form in any corpus on earth -- a Hello World program, Heron's formula. This is
#                  the verdict basis.
#   N_substantive  the subset I judge to be real content overlap.
#
# When in doubt an id counts as contaminated, because the two errors are not symmetric: dropping a
# clean id only makes OUR floor more conservative, while keeping a genuinely leaked id biases the
# result in our own favour. So the doubtful cases go into N_all, and the prime-sieve and Kadane
# hits -- standard algorithms whose variable names happened to match -- go there too rather than
# being argued about.
#
# UNIVERSAL is matched against the flagging gram, not the answer, since only the matched span is
# evidence. Anything not listed here is contamination for N_all purposes.
UNIVERSAL = (
    'public static void main(String[] args)',   # Hello World's only possible form in Java
    'System.out.println("Hello World")',
    'sqrt(s * (s - a) * (s - b) * (s - c))',    # Heron's formula, a textbook identity
)


def universal_only(grams):
    """True when EVERY gram that flagged an id is a form with no alternative phrasing.

    One substantive gram is enough to make an id contaminated, so this requires all of them --
    an id flagged by both Heron's formula and a verbatim paragraph is contaminated.
    """
    return bool(grams) and all(any(u in g for u in UNIVERSAL) for g in grams)


# data/corpus/<domain>/ HOLDS NON-CORPUS JSONL. Every chat/code domain carries a
# holdout_slice_<domain>.jsonl whose single row is {"phase", "rule_fp", "n": 0} -- a guard
# artifact, not text. A bare *.jsonl glob feeds it to the scanner, which correctly refuses on the
# missing content field, and reading it as corpus would have been worse than crashing.
#
# e1_28_leak_scan.py has the same glob and got away with it: holdout_slice sorts AFTER the
# numbered shards, so its cursor row cap is always reached first. That is luck, not a design --
# one domain whose cursor exceeds its real shard rows would read the file. Checked, not assumed:
# for all four affected domains the cap lands before the slice.
SHARD_SKIP = ("holdout_slice_", "build_corpus_stats")


def shards(dom, root=None):
    """The domain's text shards, with guard artifacts excluded by NAME rather than by luck."""
    base = os.path.join(root or ROOT, "data", "corpus", dom)
    return [p for p in sorted(glob.glob(os.path.join(base, "*.jsonl")))
            if not any(s in os.path.basename(p) for s in SHARD_SKIP)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default="data/sft/control_sft_text_heldout.jsonl")
    ap.add_argument("--ids", default="runs/heldout_v2/ids_shared.txt")
    ap.add_argument("--out", default="runs/e1_28_matched.json")
    ap.add_argument("--domains", default="", help="comma-separated; default every domain in the "
                    "cursor, which is the population the scan itself used")
    ap.add_argument("--ckpt", default="ckpt_p200m_4b_0902.pt")
    ap.add_argument("--rows_per_shard", type=int, default=0,
                    help="0 = the cursor's row count per domain, i.e. the rows the run actually "
                         "consumed. A NONZERO VALUE IS A SAMPLE, stamped as such in the output: "
                         "the 12-vs-424 reading above came from one 3000-row shard and is not a "
                         "whole-corpus count")
    ap.add_argument("--top", type=int, default=6, help="strings to print per unit per domain")
    a = ap.parse_args()

    keep = {int(x) for x in open(os.path.join(ROOT, a.ids)) if x.strip()}
    tg = {}
    with open(os.path.join(ROOT, a.heldout), errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "id" not in r or "answer" not in r:
                sys.exit(f"REFUSING: heldout record has keys {sorted(r)}; expected id and answer")
            if int(r["id"]) in keep:
                tg[int(r["id"])] = r["answer"]
    if not tg:
        sys.exit("REFUSING: no held-out completions matched the ids file -- an empty target set "
                 "matches nothing and would print a clean-looking report")
    print(f"{len(tg):,} completions, restricted to {len(keep):,} scored ids")

    ws_need, ch_need = {}, {}
    for i, c in tg.items():
        for g in S.ws_grams(c):
            ws_need.setdefault(g, i)
        for g in S.char_grams(c, stride=7):
            ch_need.setdefault(g, i)

    # The domains the CHECKPOINT consumed, so this looks at the same population the scan did.
    cursor = None
    if a.domains:
        doms = [d for d in a.domains.split(",") if d]
    else:
        import torch
        ck = torch.load(os.path.join(ROOT, a.ckpt), map_location="cpu", weights_only=False)
        cursor = ck.get("row_cursor") or {}
        doms = sorted(cursor)
        if not doms:
            sys.exit(f"REFUSING: {a.ckpt} carries no row_cursor")

    out = {"rows_per_shard": a.rows_per_shard, "sample": bool(a.rows_per_shard),
           "min_charset": S.MIN_CHARSET, "per_domain": {}}
    for dom in doms:
        files = shards(dom)
        if not files:
            print(f"=== {dom}: no shards")
            continue
        # THE SAME ROW CAP THE SCAN USED. Without it this reads the whole corpus on disk (232 GB)
        # instead of the 1,189,548 rows the run consumed, which is a different population and would
        # report overlap from rows the model never saw.
        cap = a.rows_per_shard or (cursor or {}).get(dom, 0)
        ws_m, ch_m = collections.Counter(), collections.Counter()
        by_id = collections.defaultdict(set)
        char_ids, n = set(), 0
        for p in files:
            for line in open(p, errors="replace"):
                if cap and n >= cap:
                    break
                n += 1
                t = S.text_of(json.loads(line))
                for g in S.ws_grams(t):
                    if g in ws_need:
                        ws_m[g] += 1
                        by_id[ws_need[g]].add(g)
                if dom in S.CHAR_WINDOW_DOMAINS:
                    for g in S.char_grams(t, stride=1):
                        if g in ch_need and not S.low_entropy(g):
                            ch_m[g] += 1
                            char_ids.add(ch_need[g])
            if cap and n >= cap:
                break
        print(f"=== {dom} ({n:,} rows{' SAMPLE' if a.rows_per_shard else ''}) ===")
        print(f"  ws: {len(by_id)} ids, {len(ws_m)} distinct grams")
        for g, c in ws_m.most_common(a.top):
            print(f"    x{c:<5} {g[:88]!r}")
        if dom in S.CHAR_WINDOW_DOMAINS:
            # Printed to show the unit is unusable, NOT as a contamination count.
            only = char_ids - set(by_id)
            print(f"  char: {len(char_ids)} ids ({len(only)} of them NOT found by ws), "
                  f"{len(ch_m)} distinct grams -- reported as evidence about the unit")
            for g, c in ch_m.most_common(a.top):
                print(f"    x{c:<6} {g!r}")
        out["per_domain"][dom] = {
            "rows": n,
            # id -> the grams that flagged it, so the substantive/skeleton call is reviewable.
            # NOT truncated: the classification reads these, and an id whose only substantive gram
            # fell past a cut would be misclassified as universal-only -- a silent cap that moves
            # the verdict. id 463700 alone has 46 grams.
            "ws_by_id": {str(i): sorted(gs) for i, gs in sorted(by_id.items())},
            "char_ids": sorted(char_ids),
            "char_ids_not_in_ws": sorted(char_ids - set(by_id)),
        }

    with open(os.path.join(ROOT, a.out), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    # THE TWO COUNTS. N_all is the verdict basis; a doubtful id is in it, not out of it.
    grams_by_id = collections.defaultdict(set)
    for r in out["per_domain"].values():
        for i, gs in r["ws_by_id"].items():
            grams_by_id[int(i)] |= set(gs)
    universal = {i for i, gs in grams_by_id.items() if universal_only(gs)}
    n_all = sorted(set(grams_by_id) - universal)
    out["n_all_ids"] = n_all
    out["universal_only_ids"] = sorted(universal)
    with open(os.path.join(ROOT, a.out), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    print(f"\nwhitespace-unit ids across all domains: {len(grams_by_id)}")
    print(f"  N_all = {len(n_all)}  (verdict basis: every ws hit except "
          f"{len(universal)} whose every gram is a form with no alternative)")
    if universal:
        print(f"  excluded as universal: {sorted(universal)}")
    print("  N_substantive needs the per-id read; every id's grams are in "
          f"{a.out} under ws_by_id so the call is reviewable")
    if a.rows_per_shard:
        print(f"*** rows_per_shard={a.rows_per_shard} -- THIS IS A SAMPLE, NOT A COUNT ***")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

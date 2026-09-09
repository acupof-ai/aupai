#!/usr/bin/env python3
# Keep-set join over the persisted near-overlap hit pairs (4c 2026-09-10).
# Read-only. For each hit pair, classify by whether each end is in e1's
# keep set: both in (classifier did NOT dedup this pair), exactly one in
# (classifier split it), neither in. Reported per domain pair.
#
# The scorer writes kept lines to keep_set/<domain>/<shard>.jsonl in source
# order, so a keep file is a subsequence of its source shard: a two-pointer
# walk over (line number, raw bytes) recovers exactly which source rows are
# kept. The hit file's row_a/row_b are 0-based source line numbers
# (near_overlap.py locs), same skipping rules (blank + unparseable), so the
# numbering matches.
#
# GENERATION GUARD (b0 2026-09-10): the hit file's row numbers are pinned to
# the corpus generation that was on disk when near_overlap.py ran
# (2026-09-10, pre-swap). The corpus was swapped to the clean copies afterwards
# (15 dedup08 rp1t shards shrank from ~9k rows to 4-360), so a re-run of this
# script against the swapped-in shards would silently misclassify ~3.5% of
# dedup08 pairs. Every hit row is therefore checked against the shard's
# current line count and a violation hard-fails: the numbers only mean
# something when hit file and shards are the same generation. The landed
# result (/work/aupai/data/decontam/keep_set_join_0910.json) is a frozen
# pre-swap artifact; it was computed before the swap and is not regenerated.
import glob, json, os, sys
from collections import defaultdict

BASE = "/work/aupai/data/corpus"
KEEP = "/work/aupai/data/p1/keep_set"
HITS = "/work/aupai/data/decontam/near_overlap_hits_0909.jsonl"
DOMAINS = ["code_rp1t_dd09", "code_rp1t_b2v2_dd", "code_dedup08"]
SHORT = {"code_rp1t_dd09": "dd09", "code_rp1t_b2v2_dd": "b2v2", "code_dedup08": "dedup08"}


def kept_row_set(src_path, keep_path):
    # (kept 0-based source rows, total source lines) in one pass. A missing or
    # empty keep file means the scorer dropped the whole shard: kept is empty,
    # the line count still feeds the generation guard.
    kept = set()
    try:
        kf = open(keep_path, "rb")
    except FileNotFoundError:
        kf = None
    kline = kf.readline() if kf else None
    n_lines = 0
    with open(src_path, "rb") as sf:
        for ln, sline in enumerate(sf):
            n_lines = ln + 1
            if kline and sline == kline:
                kept.add(ln)
                kline = kf.readline()
    if kf:
        kf.close()
    return kept, n_lines


def main():
    kept, lines = {}, {}  # (short_domain, shard basename) -> set(row) / int
    for dom in DOMAINS:
        for src in sorted(glob.glob(os.path.join(BASE, dom, "*.jsonl"))):
            sf = os.path.basename(src)
            k, n = kept_row_set(src, os.path.join(KEEP, dom, sf))
            kept[(SHORT[dom], sf)] = k
            lines[(SHORT[dom], sf)] = n
        n_shards = sum(1 for k in kept if k[0] == SHORT[dom])
        n_kept = sum(len(v) for k, v in kept.items() if k[0] == SHORT[dom])
        print(f"{SHORT[dom]}: {n_shards} shards, {n_kept} kept rows indexed", flush=True)

    counts = defaultdict(lambda: [0, 0, 0])  # pair -> [both, one, neither]
    total = 0
    with open(HITS) as f:
        for line in f:
            h = json.loads(line)
            ka = (h["domain_a"], os.path.basename(h["shard_a"]))
            kb = (h["domain_b"], os.path.basename(h["shard_b"]))
            for key, row in ((ka, h["row_a"]), (kb, h["row_b"])):
                if key not in lines:
                    raise SystemExit(
                        f"generation guard: shard {key} absent from {BASE} -- the hit file "
                        f"is from a different corpus generation than the shards on disk")
                if row >= lines[key]:
                    raise SystemExit(
                        f"generation guard: {key} row {row} >= {lines[key]} current lines -- "
                        f"the hit file is from a different corpus generation than the shards on "
                        f"disk; the landed JSON is a frozen pre-swap artifact")
            ka_in = h["row_a"] in kept[ka]
            kb_in = h["row_b"] in kept[kb]
            key = " <-> ".join(sorted((h["domain_a"], h["domain_b"])))
            if ka_in and kb_in:
                counts[key][0] += 1
            elif ka_in or kb_in:
                counts[key][1] += 1
            else:
                counts[key][2] += 1
            total += 1

    out = {"total_pairs": total, "pairs": {}}
    print(f"\n{'pair':<24}{'both in':>12}{'one in':>12}{'neither':>12}", flush=True)
    for key in sorted(counts):
        b, o, n = counts[key]
        t = b + o + n
        out["pairs"][key] = {"both_in_keep": b, "exactly_one_in_keep": o, "neither_in_keep": n,
                             "both_frac": round(b / t, 4), "one_frac": round(o / t, 4),
                             "neither_frac": round(n / t, 4)}
        print(f"{key:<24}{b:>12}{o:>12}{n:>12}", flush=True)
    json.dump(out, open("/work/aupai/data/decontam/keep_set_join_0910.json", "w"), indent=1)
    print("\n-> /work/aupai/data/decontam/keep_set_join_0910.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())

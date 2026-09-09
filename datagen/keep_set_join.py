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
import json, os, sys
from collections import defaultdict

BASE = "/work/aupai/data/corpus"
KEEP = "/work/aupai/data/p1/keep_set"
HITS = "/work/aupai/data/decontam/near_overlap_hits_0909.jsonl"
DOMAINS = ["code_rp1t_dd09", "code_rp1t_b2v2_dd", "code_dedup08"]
SHORT = {"code_rp1t_dd09": "dd09", "code_rp1t_b2v2_dd": "b2v2", "code_dedup08": "dedup08"}


def kept_row_set(src_path, keep_path):
    kept = set()
    try:
        kf = open(keep_path, "rb")
    except FileNotFoundError:
        return kept  # scorer never wrote it: treat as nothing kept
    with kf:
        kline = kf.readline()
        if not kline:
            return kept  # empty keep file: whole shard dropped
        with open(src_path, "rb") as sf:
            for ln, sline in enumerate(sf):
                if sline == kline:
                    kept.add(ln)
                    kline = kf.readline()
                    if not kline:
                        break
    return kept


def main():
    kept = {}  # (short_domain, shard basename) -> set(row)
    for dom in DOMAINS:
        odir = os.path.join(KEEP, dom)
        if not os.path.isdir(odir):
            print(f"MISSING keep dir {odir}", flush=True)
            continue
        for sf in sorted(os.listdir(odir)):
            if not sf.endswith(".jsonl"):
                continue
            kept[(SHORT[dom], sf)] = kept_row_set(os.path.join(BASE, dom, sf), os.path.join(odir, sf))
        n_shards = sum(1 for k in kept if k[0] == SHORT[dom])
        n_kept = sum(len(v) for k, v in kept.items() if k[0] == SHORT[dom])
        print(f"{SHORT[dom]}: {n_shards} shards, {n_kept} kept rows indexed", flush=True)

    counts = defaultdict(lambda: [0, 0, 0])  # pair -> [both, one, neither]
    total = 0
    with open(HITS) as f:
        for line in f:
            h = json.loads(line)
            ka = h["row_a"] in kept.get((h["domain_a"], os.path.basename(h["shard_a"])), ())
            kb = h["row_b"] in kept.get((h["domain_b"], os.path.basename(h["shard_b"])), ())
            key = " <-> ".join(sorted((h["domain_a"], h["domain_b"])))
            if ka and kb:
                counts[key][0] += 1
            elif ka or kb:
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

#!/usr/bin/env python3
"""Stratified sample for 27B pattern discovery.

Strata: length bucket (<500 / 500-3000 / >3000 chars) x fingerprint
(has/hasn't the 原文地址 marker) = 6 strata, K per stratum via reservoir
sampling over a random subset of shards.

The plan is written down BEFORE sampling and emitted alongside the sample —
changing stratification after the fact is cherry-picking (fb). Seeded for
reproducibility.

    python datagen/sample_pattern_discovery.py --raw-dir data/raw/cci3_hq \
        --out /tmp/pairs_discovery.txt --plan /tmp/discovery_plan.json
"""
import argparse, json, random, re

FINGERPRINT = re.compile(r"原文地址[:：]")
SEED = 20260830
N_SHARDS = 8
K = 50  # per stratum


def len_bucket(n):
    if n < 500:
        return "short"
    if n <= 3000:
        return "mid"
    return "long"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="/work/aupai/data/raw/cci3_hq")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", required=True)
    args = ap.parse_args()

    rng = random.Random(SEED)
    shards = sorted(rng.sample(range(96), N_SHARDS))
    strata = {(b, fp): [] for b in ("short", "mid", "long") for fp in (0, 1)}
    seen = {(b, fp): 0 for b in ("short", "mid", "long") for fp in (0, 1)}

    for shard_i in shards:
        path = f"{args.raw_dir}/part_{shard_i:06d}.jsonl"
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                key = (len_bucket(len(d["text"])),
                       1 if FINGERPRINT.search(d["text"][:600]) else 0)
                seen[key] += 1
                res = strata[key]
                if len(res) < K:
                    res.append((f"part_{shard_i:06d}.jsonl", d["id"]))
                elif rng.random() < K / seen[key]:
                    res[rng.randrange(K)] = (f"part_{shard_i:06d}.jsonl", d["id"])

    with open(args.out, "w") as f:
        for key in sorted(strata):
            for shard, doc_id in strata[key]:
                f.write(f"{shard}\t{doc_id}\n")

    plan = {
        "seed": SEED, "shards_sampled": [f"part_{i:06d}.jsonl" for i in shards],
        "strata": {"length": ["short<500", "mid500-3000", "long>3000"],
                   "fingerprint": "原文地址[:：] on text[:600]"},
        "K_per_stratum": K,
        "target_total": K * len(strata),
        "actual": {f"{b}/fp{fp}": len(strata[(b, fp)])
                   for b in ("short", "mid", "long") for fp in (0, 1)},
        "population_seen": {f"{b}/fp{fp}": seen[(b, fp)]
                            for b in ("short", "mid", "long") for fp in (0, 1)},
    }
    with open(args.plan, "w") as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)
    print(json.dumps(plan, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

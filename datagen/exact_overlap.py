#!/usr/bin/env python3
# 3b cross-domain exact overlap (2026-09-09): content-hash intersection between
# the three kept code domains. The suspected channel: code_dedup08 contains
# code_py_rp1t, which may be a Python subset of code_rp1t -> exact dups in dd09.
import glob, hashlib, json, sys
from multiprocessing import Pool

sys.path.insert(0, "/work/aupai/datagen")
from build_corpus import _NORM

DOMAINS = {
    "dd09": "/work/aupai/data/corpus/code_rp1t_dd09/*.jsonl",
    "b2v2": "/work/aupai/data/corpus/code_rp1t_b2v2_dd/*.jsonl",
    "dedup08": "/work/aupai/data/corpus/code_dedup08/*.jsonl",
}


def hash_shard(path):
    hs = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            c = json.loads(line).get("content", "")
        except Exception:
            continue
        if not c:
            continue
        hs.append(hashlib.sha1(_NORM.sub("", c).encode("utf-8")).digest())
    return hs


def main():
    sets = {}
    for name, pat in DOMAINS.items():
        shards = sorted(glob.glob(pat))
        with Pool(32) as pool:
            parts = pool.map(hash_shard, shards)
        s = set(h for part in parts for h in part)
        sets[name] = s
        print(f"{name}: {len(s)} unique docs", flush=True)
    names = list(sets)
    print("\n=== exact-overlap (unique hashes) ===")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            inter = len(sets[a] & sets[b])
            print(f"{a} ∩ {b}: {inter} ({inter / len(sets[a]):.4%} of {a}, {inter / len(sets[b]):.4%} of {b})")
    triple = sets[names[0]] & sets[names[1]] & sets[names[2]]
    print(f"all three: {len(triple)}")


if __name__ == "__main__":
    main()

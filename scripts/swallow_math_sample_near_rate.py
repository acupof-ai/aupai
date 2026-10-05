# Read-only: estimate near-dup rate of swallow_math on a 100k-doc random sample
# using the exact production banding (128/64x2), and report bucket-size distribution
# to explain the O(k^2) blow-up. Writes nothing.
import collections
import glob
import json
import sys

sys.path.insert(0, "/work/aupai/datagen")
sys.path.insert(0, "/work/aupai")
import build_corpus as B

paths = sorted(glob.glob("/work/aupai/data/corpus/swallow_math/swallow_math_*.jsonl"))
# sample 100k docs spread across shards (first ~806 docs/shard over 124 shards)
PER = 806
docs, perms, bands, rows = [], 128, 64, 2
ab, mask = B._near_coeffs(perms, 17)
norm = B._norm_skeleton
for p in paths:
    n = 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            if n >= PER:
                break
            line = line.strip()
            if line:
                t = B.SPECIAL_TOKEN.sub("", json.loads(line).get("content") or "").strip()
                sh = B._word_shingle_hashes(norm(t))
                docs.append(B._minhash(sh, ab, mask) if sh else None)
                n += 1
docs = [s for s in docs if s is not None]
print(f"sample signatures: {len(docs)}", flush=True)
# bucket distribution + cluster sizes per band (band 0 representative)
b0 = collections.Counter(s[0 : rows * 8] for s in docs)
sizes = sorted(b0.values(), reverse=True)
buck_multi = sum(1 for v in b0.values() if v > 1)
pairs_b0 = sum(v * (v - 1) // 2 for v in b0.values())
print(
    f"band0 distinct={len(b0)} multi_buckets={buck_multi} largest_buckets={sizes[:10]} O(k2)_pairs_band0={pairs_b0}",
    flush=True,
)
# full candidate pairs across all bands + exact-J cluster (reuse production pieces on sample)
sigs = {o: s for o, s in enumerate(docs)}
rivals = B._lsh_candidates(sigs, bands, rows)
print(f"LSH candidate pairs (all 64 bands): {len(rivals)}", flush=True)
# near rate proxy: union-find on candidates WITHOUT exact-J (upper bound; exact-J filters some)
parent = list(range(len(docs)))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for lo, hi in rivals:
    r1, r2 = find(lo), find(hi)
    if r1 != r2:
        parent[r2] = r1
roots = collections.Counter(find(i) for i in range(len(docs)))
dup_docs = sum(v - 1 for v in roots.values() if v > 1)
print(
    f"upper-bound near-dup docs (candidate clusters, pre-exact-J): {dup_docs} = {dup_docs / len(docs) * 100:.3f}%",
    flush=True,
)
print("DONE")

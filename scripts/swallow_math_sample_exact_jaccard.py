# Read-only: exact-Jaccard >=0.5 decision on the 100k sample's LSH candidates.
# 44 method: MinHash/LSH is candidate generator only; word-3-gram normalized
# Jaccard is the decider. Re-reads source text for the candidate docs only.
import collections
import glob
import json
import sys

sys.path.insert(0, "/work/aupai/datagen")
sys.path.insert(0, "/work/aupai")
import build_corpus as B

paths = sorted(glob.glob("/work/aupai/data/corpus/swallow_math/swallow_math_*.jsonl"))
PER = 806
texts, ab, mask = [], B._near_coeffs(128, 17)[0], B._near_coeffs(128, 17)[1]
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
                texts.append((t, (B._minhash(sh, ab, mask) if sh else None)))
                n += 1
sigs, words = {}, {}
o = 0
for t, s in texts:
    if s is None:
        continue
    sigs[o] = s
    w = norm(t).split()
    words[o] = set(" ".join(w[i : i + 3]) for i in range(len(w) - 2)) if len(w) >= 3 else set()
    o += 1
N = o
print(f"docs with sigs: {N}", flush=True)
rivals = B._lsh_candidates(sigs, 64, 2)
print(f"candidate pairs: {len(rivals)}", flush=True)


def jac(a, b):
    A, C = words[a], words[b]
    if not A or not C:
        return 0.0
    return len(A & C) / len(A | C)



# exact-J only on pairs; union-find survivors
parent = list(range(N))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


kept_pair = 0
tested = 0
by_ord = sorted(rivals)
for lo, hi in by_ord:
    tested += 1
    if jac(lo, hi) >= 0.5:
        kept_pair += 1
        r1, r2 = find(lo), find(hi)
        if r1 != r2:
            parent[r2] = r1
roots = collections.Counter(find(i) for i in range(N))
removed = sum(v - 1 for v in roots.values() if v > 1)
print(
    f"tested={tested} exactJ>=0.5 pairs={kept_pair} ({kept_pair / tested * 100:.2f}% of candidates)",
    flush=True,
)
print(f"EXACT near-dup removed docs={removed} = {removed / N * 100:.3f}% of {N}", flush=True)
# cluster size sanity
big = sorted((v for v in roots.values() if v > 1), reverse=True)[:10]
print(f"largest clusters={big}", flush=True)
print("DONE")

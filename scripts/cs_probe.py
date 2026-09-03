# restartable: ~2 min, read-only until the final json.dump -- an interrupt loses the
# pass, never a partial output.
"""Is code_supply a real code_tests source, or starcoder again under another schema?

Two questions: how pair yield scales with repo coverage, and whether the rows are
already in code_py_starcoder. The second decides it -- a source that overlaps is the
same dead end tilerl-23 just measured.
"""
import collections, glob, hashlib, json, os
import multiprocessing as mp
import pyarrow.parquet as pq

def H(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest(), "big")

def stem(p):
    b = os.path.basename(p)
    if not b.endswith(".py"):
        return None
    b = b[:-3]
    if b.startswith("test_"):
        return b[5:]
    if b.endswith("_test"):
        return b[:-5]
    return None

def load(f):
    t = pq.read_table(f, columns=["repo_name", "path", "size", "content"])
    return (t.column(0).to_pylist(), t.column(1).to_pylist(),
            [int(x) for x in t.column(2).to_pylist()], [H(c or "") for c in t.column(3).to_pylist()])

def match(f):
    hits, n = 0, 0
    with open(f) as fh:
        for line in fh:
            c = json.loads(line).get("content")
            if c:
                n += 1
                if H(c) in CS:
                    hits += 1
    return n, hits

def _init(cs):
    global CS
    CS = cs

if __name__ == "__main__":
    fs = sorted(glob.glob("/work/aupai/data/raw/code_supply/*.parquet"))[:9]
    repo = collections.defaultdict(dict)
    hashes, rows, byts = set(), 0, 0
    for i, f in enumerate(fs, 1):
        rn, pa, sz, hs = load(f)
        for r, p, s, h in zip(rn, pa, sz, hs):
            repo[r][p] = s
            hashes.add(h)
            rows += 1
            byts += s
        pairs = pb = 0
        for r, files in repo.items():
            impl = {os.path.basename(p)[:-3]: (p, s) for p, s in files.items() if p.endswith(".py")}
            for p, s in files.items():
                st = stem(p)
                if st and st in impl:
                    pairs += 1
                    pb += s + impl[st][1]
        print(f"  {i}/9 files: rows {rows} repos {len(repo)} mean {rows/len(repo):.2f} "
              f"pairs {pairs} ({pairs/rows:.3%}) pair_bytes {pb/1e9:.3f}GB", flush=True)
    res = {"files": len(fs), "rows": rows, "repos": len(repo), "bytes": byts,
           "pairs": pairs, "pair_rate": pairs / rows, "pair_bytes": pb,
           "distinct_content": len(hashes)}

    corpus = sorted(glob.glob("/work/aupai/data/corpus/code_py_starcoder/*.jsonl"))
    tot = hit = 0
    with mp.Pool(12, initializer=_init, initargs=(hashes,)) as p:
        for i, (n, h) in enumerate(p.imap_unordered(match, corpus), 1):
            tot += n; hit += h
            if i % 80 == 0:
                print(f"  starcoder {i}/{len(corpus)}: {hit} hits", flush=True)
    res["overlap_vs_starcoder"] = hit / max(len(hashes), 1)
    print(f"\ncode_supply distinct {len(hashes)}, already in code_py_starcoder: {hit} "
          f"= {hit/max(len(hashes),1):.2%}", flush=True)
    json.dump(res, open("/work/aupai/runs/cs_probe.json", "w"), indent=1)
    print("wrote runs/cs_probe.json", flush=True)

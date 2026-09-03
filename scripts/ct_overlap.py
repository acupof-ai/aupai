# restartable: 6 min end to end, read-only until the final json.dump -- an interrupt loses
# the pass, never a partial output. Per-shard checkpointing would cost more than a rerun.
"""code_tests: is the mined set new supply, or the same rows under a new domain name?

Measured, not derived. code_py_starcoder consumed all 59 raw shards under filter
`ast.parse(t)` + non-empty (build_starcoder_py.py:62-67), and code_dedup08's only
inputs are that plus code_py_rp1t -- so set logic already predicts near-total
overlap. This hashes both sides instead of trusting that.

Join key = raw content. Valid because build_corpus.py:332 writes `text` verbatim from
the parquet column; no normalization sits between the two stores.

Memory: the pod has ~890MB free, and a set of 6.2M python ints costs ~430MB -- so the
corpora are NEVER fully materialized. The mined side is built first (small), then each
corpus is streamed and intersected per-file, keeping only the matched hashes.
Key set is hashed to int64 too, so forking 12 workers copies ints, not string tuples.
"""
import glob, hashlib, json, pickle, sys
import multiprocessing as mp
import pyarrow.parquet as pq

sys.path.insert(0, "/work/aupai/datagen")
from build_corpus import reject_light

def H(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest(), "big")

def scan_raw(fp):
    """Rows whose (repo,path) was mined: run the production light filter, hash the keepers."""
    t = pq.read_table(fp, columns=["max_stars_repo_name", "max_stars_repo_path", "content"])
    seen = kept = 0
    reasons, digests = {}, []
    for nm, pa, c in zip(t.column(0).to_pylist(), t.column(1).to_pylist(), t.column(2).to_pylist()):
        if H(f"{nm}\0{pa}") not in KEYS:
            continue
        seen += 1
        why = reject_light(c or "")
        if why is None:
            kept += 1
            digests.append(H(c or ""))
        else:
            reasons[why] = reasons.get(why, 0) + 1
    return seen, kept, reasons, digests

def match_file(fp):
    """Only the hashes that are ALSO in the mined set -- never the corpus set itself."""
    hits, n = [], 0
    with open(fp) as f:
        for line in f:
            c = json.loads(line).get("content")
            if c:
                n += 1
                h = H(c)
                if h in MINED:
                    hits.append(h)
    return n, hits

def _init_keys(k):
    global KEYS
    KEYS = k

def _init_mined(m):
    global MINED
    MINED = m

if __name__ == "__main__":
    D = "/work/aupai/data/raw/code_tests_trial"
    impl = pickle.load(open(f"{D}/kept_impl.pickle", "rb"))
    test = pickle.load(open(f"{D}/kept_test.pickle", "rb"))
    pairs = set(impl) | set(test)
    KEYS = {H(f"{nm}\0{pa}") for nm, pa in pairs}
    print(f"mined keys: impl {len(impl)} test {len(test)} union {len(pairs)} "
          f"(hashed {len(KEYS)})", flush=True)
    del impl, test, pairs

    shards = sorted(glob.glob("/work/aupai/data/raw/ms_starcoder_py/*.parquet"))
    seen = kept = 0
    reasons, mined = {}, set()
    with mp.Pool(8, initializer=_init_keys, initargs=(KEYS,)) as p:
        for i, (s, k, r, d) in enumerate(p.imap_unordered(scan_raw, shards), 1):
            seen += s; kept += k; mined.update(d)
            for w, n in r.items():
                reasons[w] = reasons.get(w, 0) + n
            print(f"  raw {i}/{len(shards)}: seen {seen} kept {kept}", flush=True)

    res = {"mined_keys": len(KEYS), "rows_seen": seen, "rows_kept": kept,
           "kept_rate": kept / max(seen, 1), "reject_reasons": reasons,
           "kept_distinct_hashes": len(mined)}
    print(f"\nFULL (59/59): seen {seen}, kept {kept} ({res['kept_rate']:.1%}), "
          f"{len(mined)} distinct after collapsing duplicate content", flush=True)
    for w, n in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {w:14s} {n:8d}  {n/max(seen,1):.2%}")

    hit_union = set()
    for dom in ("code_py_starcoder", "code_dedup08"):
        files = sorted(glob.glob(f"/work/aupai/data/corpus/{dom}/*.jsonl"))
        rows, hits = 0, set()
        with mp.Pool(12, initializer=_init_mined, initargs=(mined,)) as p:
            for i, (n, hs) in enumerate(p.imap_unordered(match_file, files), 1):
                rows += n; hits.update(hs)
                if i % 50 == 0:
                    print(f"  {dom} {i}/{len(files)}: {rows} rows, {len(hits)} matched", flush=True)
        hit_union |= hits
        res[f"rows_{dom}"] = rows
        res[f"overlap_{dom}"] = len(hits) / max(len(mined), 1)
        print(f"\noverlap vs {dom}: {len(hits)}/{len(mined)} = "
              f"{len(hits)/max(len(mined),1):.2%}  ({rows} corpus rows)", flush=True)

    res["overlap_union"] = len(hit_union) / max(len(mined), 1)
    res["novel_rows"] = len(mined) - len(hit_union)
    print(f"\noverlap vs union: {len(hit_union)}/{len(mined)} = "
          f"{res['overlap_union']:.2%}   NOVEL {res['novel_rows']}", flush=True)
    json.dump(res, open("/work/aupai/runs/ct_overlap.json", "w"), indent=1)
    print("wrote /work/aupai/runs/ct_overlap.json", flush=True)

#!/usr/bin/env python3
# 3b cleanup pass (2026-09-09): one pass over the 3 kept code domains.
#   (1) decontamination: HumanEval+MBPP containment >= 0.5 (pre-registered, 4c (a))
#   (2) exact cross-domain overlap: dedup08 docs whose normalized hash is in dd09|b2v2
# Output:
#   /work/aupai/data/corpus_clean/<domain>/*.jsonl  (clean copies, hit rows dropped)
#   /work/aupai/data/decontam/decontam_hits_0909.jsonl  (full hit list, per 4c)
#   /work/aupai/data/decontam/cleanup_stats_0909.json   (per-domain counts)
# IDF is recomputed over the same 747 paths as the verdict scan, so containment
# values are comparable.
import glob, hashlib, json, os, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/work/aupai/datagen")
from build_corpus import _NORM
import scan_code_contamination as S
from scan_code_contamination import load_holdouts, HoldoutIndex, _df_worker, DEFAULT_THRESHOLD as TH

OUT_ROOT = "/work/aupai/data/corpus_clean"
# data/, not runs/: the manifest is a deletion record for corpus state, not a
# session ledger -- the integration-tree writer guard only covers runs/*.jsonl.
HITLIST = "/work/aupai/data/decontam/decontam_hits_0909.jsonl"
HASHCK = "/work/aupai/runs/overlap_ck/exact_hashes.pkl"
DOMAINS = [
    ("dd09", "/work/aupai/data/corpus/code_rp1t_dd09/*.jsonl"),
    ("b2v2", "/work/aupai/data/corpus/code_rp1t_b2v2_dd/*.jsonl"),
    ("dedup08", "/work/aupai/data/corpus/code_dedup08/*.jsonl"),
]
BASELINE = "/work/aupai/data/corpus/web_hq/*.jsonl"
BATCH = 2048


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


def build_overlap_set():
    """Hashes of dd09+b2v2 docs; dedup08 docs in this set are exact dups."""
    import pickle
    if os.path.exists(HASHCK):
        return pickle.load(open(HASHCK, "rb"))
    s = set()
    for name in ("dd09", "b2v2"):
        pat = dict(DOMAINS)[name]
        with Pool(32) as pool:
            for part in pool.imap_unordered(hash_shard, sorted(glob.glob(pat))):
                s.update(part)
        print(f"overlap set: {name} added, total {len(s)}", flush=True)
    os.makedirs(os.path.dirname(HASHCK), exist_ok=True)
    pickle.dump(s, open(HASHCK, "wb"))
    return s


def main():
    t0 = time.perf_counter()
    holdouts = load_holdouts()
    idx = HoldoutIndex(holdouts)
    print(f"holdouts: {len(holdouts)} ({len(idx.long_cols)} long)", flush=True)

    cand_paths = sorted(p for _, pat in DOMAINS for p in glob.glob(pat))
    base_paths = sorted(glob.glob(BASELINE))
    idf_paths = sorted(set(base_paths) | set(cand_paths))
    idf_ck = "/work/aupai/runs/overlap_ck/idf.npz"
    if os.path.exists(idf_ck):
        z = np.load(idf_ck)
        df, n_idf = z["df"], int(z["n"])
        print(f"IDF: cached {n_idf} docs", flush=True)
    else:
        print(f"IDF pass over {len(idf_paths)} paths...", flush=True)
        S._WORKER_IDX = idx  # fork pool: idx rides COW via the module global
        with Pool(32) as pool:
            parts = pool.map(_df_worker, [(p, None, False) for p in idf_paths])
        df = np.zeros(len(idx.g2i), dtype=np.float64)
        for part, _ in parts:
            df += part
        n_idf = sum(n for _, n in parts)
        os.makedirs(os.path.dirname(idf_ck), exist_ok=True)
        np.savez(idf_ck, df=df, n=n_idf)
    idx.set_idf(df, n_idf)
    print(f"IDF: {n_idf} docs, {int((df > 0).sum())}/{len(df)} bigrams present "
          f"({round(time.perf_counter() - t0)}s)", flush=True)

    overlap = build_overlap_set()
    print(f"overlap set: {len(overlap)} unique hashes", flush=True)

    os.makedirs(os.path.dirname(HITLIST), exist_ok=True)
    hitf = open(HITLIST, "w")
    stats = {}
    for name, pat in DOMAINS:
        out_dir = os.path.join(OUT_ROOT, name)
        os.makedirs(out_dir, exist_ok=True)
        n_in = n_out = n_decont = n_overlap = 0
        limit = int(os.environ.get("CLEANUP_LIMIT", "0"))
        for sp in sorted(glob.glob(pat))[:limit or None]:
            rows = []
            for line in open(sp, encoding="utf-8"):
                if not line.strip():
                    continue
                rows.append(line)
            remove = {}  # row idx -> (kind, detail)
            # decontamination
            for i0 in range(0, len(rows), BATCH):
                texts = []
                for line in rows[i0:i0 + BATCH]:
                    try:
                        texts.append(json.loads(line).get("content", ""))
                    except Exception:
                        texts.append("")
                exact, mc, nf, hr, hidx = idx.scan_chunk(texts, TH)
                for r, t in exact:
                    remove[i0 + r] = ("exact", {"containment": 1.0, "text": t[:120]})
                for r in hidx:
                    if i0 + r in remove:
                        continue
                    R = idx._matrix([texts[r]])
                    if R is None:
                        continue
                    cont = np.asarray((R @ idx.Hw_n).todense()).ravel()
                    # match the verdict scan's counting: long holdouts only. The argmax over
                    # all holdouts would add short-holdout hits the ledger never counted,
                    # making the manifest bigger than the 11,744 ruling it executes.
                    long = cont[idx.long_cols]
                    top_local = int(long.argmax())
                    if long[top_local] >= TH:
                        top = idx.long_cols[top_local]
                        remove[i0 + r] = ("containment",
                                          {"containment": round(float(long[top_local]), 4),
                                           "holdout_id": top,
                                           "doc_len": len(texts[r])})
            # exact overlap (dedup08 only: its docs duplicated in dd09|b2v2)
            if name == "dedup08":
                for i, line in enumerate(rows):
                    if i in remove:
                        continue
                    try:
                        c = json.loads(line).get("content", "")
                    except Exception:
                        continue
                    h = hashlib.sha1(_NORM.sub("", c).encode("utf-8")).digest()
                    if h in overlap:
                        remove[i] = ("overlap", {})
            for i, (kind, detail) in remove.items():
                rec = {"shard": os.path.basename(sp), "row": i, "kind": kind,
                       "domain": name, **detail}
                hitf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if kind in ("exact", "containment"):
                    n_decont += 1
                else:
                    n_overlap += 1
            n_in += len(rows)
            n_out += len(rows) - len(remove)
            if os.environ.get("LIST_ONLY"):
                continue  # 4c 2026-09-09: hit list lands, clean copies wait for identity confirmation
            out_path = os.path.join(out_dir, os.path.basename(sp))
            with open(out_path, "w", encoding="utf-8") as f:
                for i, line in enumerate(rows):
                    if i not in remove:
                        f.write(line)
        stats[name] = {"in": n_in, "out": n_out, "removed": n_in - n_out,
                       "decontamination": n_decont, "exact_overlap": n_overlap}
        print(f"{name}: {n_in} -> {n_out} (decont {n_decont}, overlap {n_overlap}) "
              f"({round(time.perf_counter() - t0)}s)", flush=True)
    hitf.close()
    json.dump(stats, open("/work/aupai/data/decontam/cleanup_stats_0909.json", "w"),
              ensure_ascii=False, indent=2)
    print("DONE", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()

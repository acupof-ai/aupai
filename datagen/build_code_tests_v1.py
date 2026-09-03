#!/usr/bin/env python3
"""Build data/corpus/code_tests_v1/ from MS starcoderdata 'python' parquet.

Domain = 'code whose test actually runs and passes' -- PAIRING SUPPLY ONLY (format-2,
src/+tests/ PAIRS), before the B2 pass-gate (de). This build does NOT run the pass
gate; it reuses the pairing yield trial's filter exactly so the landing reproduces
the trial's 2.82B paired tokens. See data/raw/code_tests_trial/pair_yield.py.

Why NOT build_corpus.main(): the pairing needs the FULL repo's path set + test
imports in scope before any row can be routed, which a parallel per-row worker
cannot see. So the build is two reads of the same 22G parquet: Phase A re-runs the
trial's aggregation to decide, per (repo, path), which paired chapter it belongs to;
Phase B routes content into chapters and emits one JSONL doc per (repo, pkg-prefix)
chapter. The path columns (max_stars_repo_name / max_stars_repo_path) are load-bearing
and are NOT stripped -- every doc carries repo/path/test_paths.

Format-2 shape (bar-join, verified natural in code_py_rp1t): <impl rows, blank
joined> + blank + '#'-hash separator bar + blank + <test>, repeated per test, all in
"content". One chapter groups the tests that share a linked pkg-prefix so impl
content is emitted once per (repo, prefix) -- no double count (trial rule).

    python3 datagen/build_code_tests_v1.py              # full build
    python3 datagen/build_code_tests_v1.py --phase-a    # stop after the link map
    python3 datagen/build_code_tests_v1.py --phase-b    # route+emit from saved map

Build is idempotent per phase: `kept_*.pickle` gate repeats and the output dir is
wiped before Phase B re-emits. Nothing here commits; a new corpus dir is written.
"""

import argparse
import ast
import glob
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time
from collections import defaultdict

import pyarrow.parquet as pq
from tokenizers import Tokenizer

RAW = "/work/aupai/data/raw/ms_starcoder_py"
OUT = "/work/aupai/data/corpus/code_tests_v1"
TK = "/work/aupai/data/tokenizer.json"
TRIAL_DIR = "/work/aupai/data/raw/code_tests_trial"
TRIAL_RESULT = os.path.join(TRIAL_DIR, "result.json")
KEPT_IMPL = os.path.join(TRIAL_DIR, "kept_impl.pickle")
KEPT_TEST = os.path.join(TRIAL_DIR, "kept_test.pickle")
NJOBS = 16
CHARS_PER_TOKEN = 1.5  # build_corpus.py estimate, kept for continuity only
SHARD_BYTES = 100 * 1024 * 1024
# '#'-hash separator bar (format-2). A comment bar reads natural between an impl
# block and its tests and is stable across the code domains.
BAR = "# " + "-" * 78

TEST_SEG = re.compile(r"(?:^|/)(?:tests?|testing)(?:/|$)")
TEST_NAME = re.compile(r"(?:^|/)(?:test_[^/]+|[^/]+_test)\.py$")

SHARDS = sorted(glob.glob(os.path.join(RAW, "train-*.parquet")))


def _is_test(p):
    return bool(TEST_SEG.search(p)) or bool(TEST_NAME.search(p)) or p.endswith("tests.py") or p.endswith("test.py")


def top_imports(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def _load_tk():
    return Tokenizer.from_file(TK)


# --------------------------------------------------------------------------- Phase A
def process_shard(path):
    tk = _load_tk()
    recs = []
    tot_rows = tot_ast = tot_tok = 0
    pf = pq.ParquetFile(path)
    cols = ["max_stars_repo_name", "max_stars_repo_path", "content"]
    for batch in pf.iter_batches(batch_size=2048, columns=cols):
        d = batch.to_pydict()
        repo, pth, content = d["max_stars_repo_name"], d["max_stars_repo_path"], d["content"]
        for r, p, t in zip(repo, pth, content, strict=True):
            tot_rows += 1
            if t is None:
                continue
            t = str(t)
            if not t.strip():
                continue
            try:
                tree = ast.parse(t)
            except Exception:
                continue
            tot_ast += 1
            ntok = len(tk.encode(t).ids)
            tot_tok += ntok
            p = (p or "") if p else ""
            istest = _is_test(p)
            if istest:
                recs.append((r, p, ntok, 1, tuple(sorted(top_imports(tree)))))
            else:
                recs.append((r, p, ntok, 0, ()))
    return recs, tot_rows, tot_ast, tot_tok


def phase_a():
    t0 = time.time()
    with mp.Pool(NJOBS) as pool:
        results = pool.map(process_shard, SHARDS)
    T_ROWS = T_AST = T_TOK = 0
    repos = defaultdict(lambda: {"paths": set(), "ntoks": {}, "tests": []})
    for recs, tr, ta, tt in results:
        T_ROWS += tr
        T_AST += ta
        T_TOK += tt
        for (r, p, ntok, istest, imps) in recs:
            g = repos[r]
            g["paths"].add(p)
            g["ntoks"].setdefault(p, ntok)
            if istest:
                g["tests"].append((p, ntok, imps))
    n_tests = sum(len(g["tests"]) for g in repos.values())
    n_repos = len(repos)

    # Trial accounting (reproduce pair_yield.py EXACTLY) + a link map for Phase B.
    linked_test_rows = linked_test_tokens = linked_impl_rows = impl_tok = pairs = 0
    seen_repo_pref = set()
    kept_impl = {}   # (repo, path) -> prefix   (impl-chapter routing, single assignment)
    kept_test = {}   # (repo, path) -> prefix   (test-chapter routing)
    longest_linked = {}  # repo -> sorted linked prefixes, longest first
    rep_prefix = {}  # repo -> set of linked prefixes
    for r, g in repos.items():
        paths = g["paths"]
        has_src = any(p.startswith("src/") for p in paths)
        linked_prefs = set()
        # Which prefix does each linked test route to (first hit, trial rule)?
        test_route = {}
        for (tp, tn, imps) in g["tests"]:
            hit = None
            for m in imps:
                mdir = m + "/"
                if has_src and any(p.startswith("src/" + mdir) for p in paths):
                    hit = "src/" + mdir
                elif any(p.startswith(mdir) for p in paths):
                    hit = mdir
                if hit:
                    break
            if hit:
                linked_test_rows += 1
                linked_test_tokens += tn
                pairs += 1
                linked_prefs.add(hit)
                test_route[tp] = hit
                kept_test[(r, tp)] = hit
        rep_prefix[r] = linked_prefs
        # impl routing: a NON-test path under the longest linked prefix -> impl chapter.
        if linked_prefs:
            longest_linked[r] = sorted(linked_prefs, key=len, reverse=True)
            for pref in linked_prefs:
                if (r, pref) in seen_repo_pref:
                    continue
                seen_repo_pref.add((r, pref))
                c = 0
                s = 0
                for p in paths:
                    if p.startswith(pref):
                        c += 1
                        s += g["ntoks"][p]
                linked_impl_rows += c
                impl_tok += s
        # single-assignment impl route (longest linked prefix that p starts with,
        # excluding the test rows that route to test chapters).
        if linked_prefs:
            prefs_by_len = longest_linked[r]
            for p in paths:
                if (r, p) in kept_test:
                    continue
                for pref in prefs_by_len:
                    if p.startswith(pref):
                        kept_impl[(r, p)] = pref
                        break

    linked_rows = linked_test_rows + linked_impl_rows
    token_yield = (linked_test_tokens + impl_tok) / T_TOK if T_TOK else 0
    row_yield = linked_rows / T_ROWS if T_ROWS else 0
    paired_tokens = linked_test_tokens + impl_tok

    trial = {}
    if os.path.exists(TRIAL_RESULT):
        trial = json.load(open(TRIAL_RESULT))

    linkmap = {
        "source_parquet": RAW,
        "n_shards": len(SHARDS),
        "total_rows": T_ROWS,
        "ast_valid_rows": T_AST,
        "ast_valid_tokens": T_TOK,
        "repos": n_repos,
        "test_rows": n_tests,
        "test_row_frac": n_tests / T_AST if T_AST else 0,
        "linked_test_rows": linked_test_rows,
        "test_link_rate": linked_test_rows / n_tests if n_tests else 0,
        "linked_impl_row_units": len(seen_repo_pref),
        "linked_impl_rows": linked_impl_rows,
        "linked_rows_total": linked_rows,
        "ROW_YIELD": row_yield,
        "linked_test_tokens": linked_test_tokens,
        "linked_impl_tokens": impl_tok,
        "paired_tokens": paired_tokens,
        "TOKEN_YIELD": token_yield,
        "pairs": pairs,
        "impl_rows_single_assigned": len(kept_impl),
        "trial_paired_tokens": trial.get("paired_tokens"),
        "trial_delta": paired_tokens - trial.get("paired_tokens", 0) if trial else None,
        "elapsed_s": round(time.time() - t0, 1),
        "seed": f"deterministic-full-{len(SHARDS)}shards-sorted",
    }
    import pickle

    with open(KEPT_IMPL, "wb") as f:
        pickle.dump(kept_impl, f)
    with open(KEPT_TEST, "wb") as f:
        pickle.dump(dict(kept_test), f)
    with open(os.path.join(TRIAL_DIR, "linkmap.json"), "w") as f:
        json.dump(linkmap, f, indent=2)
    print(json.dumps(linkmap, indent=2))
    return linkmap


# --------------------------------------------------------------------------- Phase B
def route_shard(path):
    import pickle

    kept_impl = pickle.load(open(KEPT_IMPL, "rb"))
    kept_test = pickle.load(open(KEPT_TEST, "rb"))
    ch = defaultdict(lambda: {"impl": [], "tests": []})
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=4096,
        columns=["max_stars_repo_name", "max_stars_repo_path", "content"],
    ):
        dd = batch.to_pydict()
        for r, p, t in zip(dd["max_stars_repo_name"], dd["max_stars_repo_path"], dd["content"], strict=True):
            if t is None:
                continue
            key_impl = (r, p)
            if key_impl in kept_impl:
                ch[(r, kept_impl[key_impl])]["impl"].append(str(t))
            elif (r, p) in kept_test:
                ch[(r, kept_test[(r, p)])]["tests"].append((p, str(t)))
    return dict(ch)


def phase_b():
    # wipe prior output
    if os.path.isdir(OUT):
        for x in glob.glob(os.path.join(OUT, "code_tests_v1_*.jsonl")):
            os.remove(x)
    else:
        os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    chapters = defaultdict(lambda: {"impl": [], "tests": []})
    with mp.Pool(NJOBS) as pool:
        for partial in pool.imap(route_shard, SHARDS, chunksize=1):
            for k, v in partial.items():
                chapters[k]["impl"].extend(v["impl"])
                chapters[k]["tests"].extend(v["tests"])
    print("routed chapters=%d in %.1fs" % (len(chapters), time.time() - t0), flush=True)

    # deterministic emit order
    keys = sorted(chapters.keys())
    ndocs = 0
    nchars = 0
    out_idx = 0
    out_f = None
    out_size = 0
    out_name = "code_tests_v1_{i:03d}.jsonl"
    for (repo, pref) in keys:
        ch = chapters[(repo, pref)]
        tests = sorted(ch["tests"])  # by (path, content)
        impl = "\n\n".join(x.strip() for x in ch["impl"] if x.strip())
        if not impl and not tests:
            continue
        parts = []
        if impl:
            parts.append(impl)
        for (_tp, tc) in tests:
            parts.append(BAR)
            parts.append(tc.strip())
        content = "\n\n".join(parts) + "\n"
        nchars += len(content)
        ndocs += 1
        row = {
            "content": content,
            "source": "starcoderdata:python",
            "url": "%s::%s" % (repo, pref),
            "repo": repo,
            "path": pref,
            "test_paths": [tp for (tp, _tc) in tests],
        }
        line = json.dumps(row, ensure_ascii=False) + "\n"
        blen = len(line.encode("utf-8"))
        if out_f is None or out_size + blen > SHARD_BYTES:
            if out_f is not None:
                out_f.close()
            out_path = os.path.join(OUT, out_name.format(i=out_idx))
            sys.stderr.write("open %s\n" % out_path)
            out_f = open(out_path, "w", encoding="utf-8")
            out_idx += 1
            out_size = 0
        out_f.write(line)
        out_size += blen
    if out_f is not None:
        out_f.close()

    # content-only token counts (exclude separator bars/blank) per chapter
    tk = _load_tk()
    nimpl_tok = 0
    ntest_tok = 0
    for (repo, pref) in keys:
        ch = chapters[(repo, pref)]
        impl = "\n\n".join(x.strip() for x in ch["impl"] if x.strip())
        nimpl_tok += len(tk.encode(impl).ids) if impl else 0
        for (_tp, tc) in ch["tests"]:
            ntest_tok += len(tk.encode(tc.strip()).ids)
    nshards = len(glob.glob(os.path.join(OUT, "code_tests_v1_*.jsonl")))

    sys.path.insert(0, "/work/aupai/datagen")
    from corpus_fingerprint import fp_dir  # noqa: E402

    fp = fp_dir(OUT)
    script_fp = _script_fp()
    paired_content = nimpl_tok + ntest_tok
    required_pass_rate = (2.0e9 / paired_content) if paired_content else None

    stats = {
        "domain": "code_tests_v1",
        "reasons": {"paired_chapters": ndocs},
        "kept": ndocs,
        "kept_chars": nchars,
        "kept_tokens": int(nchars / CHARS_PER_TOKEN),
        "filters": "code-tests-pairing-v1",
        "workers": NJOBS,
        "n_shards": nshards,
        # filters_fp = content hash of THIS build script, which is what actually
        # produced the bytes (pairing + formatting), not filters/*.py.
        "filters_fp": script_fp,
        "fingerprint": fp,
        "near_dedup": False,
        "near_dedup_note": "impl counted once per (repo,pkg-prefix); the separate calibrated near-dedup post-pass (44) still applies later",
        "tokens": paired_content,
        "tokens_status": "measured",
        "tokens_config": ("data/tokenizer.json; paired content tokens = impl(once per repo,prefix) "
                          "+ test tokens, excluding separator bars and <eos>"),
        "format": {
            "kind": "format-2-src-tests-pair",
            "shape": "<impl rows,blank joined> + blank + '#'-hash separator bar + blank + <test>, "
                     "repeated per test; one chapter per (repo,pkg-prefix) with multiple tests "
                     "sharing the impl block so impl is emitted once (no double count)",
            "separator": BAR,
            "bar_join": "verified natural in code_py_rp1t",
        },
        "pairing": {
            "filter": "trial pair_yield.py: test top-level import matches an impl module "
                      "under the same repo group (src/ or top-level); impl once per (repo,pkg-prefix)",
            "trial_dir": TRIAL_DIR,
            "trial_paired_tokens": None,
            "paired_content_tokens": paired_content,
            "impl_tokens": nimpl_tok,
            "test_tokens": ntest_tok,
            "paired_docs": ndocs,
            "trial_reproduced": None,
        },
        "provenance": {
            "raw_source": "ms_starcoder_py parquet (local, no fetch)",
            "raw_path": RAW,
            "fetch_stats": os.path.join(RAW, "fetch_stats.json"),
            "trial": TRIAL_RESULT,
            "kept_map": [KEPT_IMPL, KEPT_TEST],
            "method": "local parquet -> paired JSONL, no network fetch",
        },
        "pass_gate_caveat": {
            "note": "this domain is PAIRING SUPPLY ONLY, before the B2 pass-gate (de). "
                    "If the gate keeps fraction < required_pass_rate the domain cannot reach 2B post-gate.",
            "target_tokens": 2.0e9,
            "required_pass_rate": required_pass_rate,
        },
    }
    tr = {}
    if os.path.exists(TRIAL_RESULT):
        tr = json.load(open(TRIAL_RESULT))
    stats["pairing"]["trial_paired_tokens"] = tr.get("paired_tokens")
    if tr.get("paired_tokens"):
        stats["pairing"]["trial_reproduced"] = round(paired_content / tr["paired_tokens"], 5)
    with open(os.path.join(OUT, "build_corpus_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    # assert the required filters_fp
    assert stats["filters_fp"], "filters_fp must be non-empty"
    print(json.dumps({"phase": "b", "docs": ndocs, "shards": nshards,
                      "paired_content_tokens": paired_content,
                      "impl_tokens": nimpl_tok, "test_tokens": ntest_tok,
                      "required_pass_rate": required_pass_rate,
                      "fingerprint": fp, "filters_fp": script_fp}, indent=2))
    return stats


def _script_fp():
    p = os.path.realpath(__file__)
    h = hashlib.sha1()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-a", action="store_true")
    ap.add_argument("--phase-b", action="store_true")
    a = ap.parse_args()
    if a.phase_a:
        phase_a()
        return
    if a.phase_b:
        phase_b()
        return
    phase_a()
    phase_b()


if __name__ == "__main__":
    main()

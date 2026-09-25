#!/usr/bin/env python3
# restartable: the only costly stage is 156k bounded 10-s subprocesses over 16 workers;
# an interrupt re-runs them (stage A parse and stage C are minutes). Output is one jsonl
# written whole at the end. Full run is ~1 h on 16 cores, the bounded rerun cost.
"""1e task 2026-09-25: REAL (not extrapolated) SFT-B candidate funnel from the
UNCONSUMED tail of code_ultra_l3_noexec_dc.

Consumption order is reproduced EXACTLY as train.py builds the token cache:
texts = concat(_jsonl_content(shard) for shard in sorted(glob(*.jsonl)));
random.Random(sample_seed=42).shuffle(texts). The checkpoint row_cursor for this
domain is 2,320,255 consumed (full_plan_prefix, seed 42). We reproduce the
PERMUTATION over indices (no 15M-string load) and take the first TAKE=200,000
indices AFTER the consumed prefix, reading those source rows directly.

Funnel (each stage counted):
  n_tail_rows
  -> shape: extract code; needs >=1 top-level/any assert AND a def
  -> reference runs its own asserts green (subprocess, the row must have asserts)
  -> assert-block hygiene: drop rows whose assert lines carry self-correction /
     contradiction markers (a wrong expectation admitting itself)
  -> HE/MBPP 13-gram decontamination (filters/decontam_ngram)
  -> prompt-hash disjoint from SFT-A (question+signature layers) and the RL pool.
Output survivors + a stats json. CPU only, bounded cores by the launcher.
"""
import argparse
import ast
import glob
import hashlib
import json
import os
import random
import re
import sys
import collections

CORPUS = "/work/aupai/data/corpus/code_ultra_l3_noexec_dc"
REPO = os.environ.get("AUPAI_REPO", "/work/aupai")
SEED = 42
CONSUMED = 2_320_255
TAKE = 200_000
TIMEOUT = 10


def _load_decontaminator():
    sys.path.insert(0, os.path.join(REPO, "filters"))
    from decontam_ngram import Decontaminator
    return Decontaminator

_CODE_START = re.compile(r"^(?:def |class |import |from |@|[A-Za-z_][A-Za-z0-9_]*\s*=)", re.M)
_WS = re.compile(r"\s+")
# self-correction / contradiction markers the synthetic author wrote IN the test
# lines admitting the expected value was revised.
MARKERS = ["let's fix", "actually node", "actually correct", "correct is",
           "let me adjust", "should be", "i made a mistake", "on second thought",
           "wait,", "that's wrong", "this is wrong"]
# The barber-shop counter-example ("# ... So N=4 ->1") revises the value but uses
# none of those substrings; only an explicit "So N=k ->v" arrow in an assert
# comment names it. Bare "actually" was rejected: it matches 3,342/50,633
# survivors, almost all benign reasoning that agrees with the assert. A semantic
# contradiction with no marker is uncatchable by keywords (doc §6 residual).
MARKER_RES = [re.compile(r"so\s+n\s*=\s*\d+\s*(?:->|=>)\s*\d+", re.I)]


def norm_text(s):
    return _WS.sub(" ", (s or "").strip()).lower()


def qhash(s):
    return hashlib.sha1(norm_text(s).encode()).hexdigest()


def shard_files():
    return sorted(p for p in glob.glob(os.path.join(CORPUS, "*.jsonl"))
                  if os.path.basename(p) != "build_corpus_stats.json")


def tail_indices(total, consumed, take):
    """The `take` source positions in the post-shuffle order after `consumed`."""
    perm = list(range(total))
    random.Random(SEED).shuffle(perm)
    return perm[consumed:consumed + take]


def extract(content):
    m = _CODE_START.search(content)
    if not m:
        return None, None
    stmt = content[:m.start()]
    code = content[m.start():]
    try:
        ast.parse(code)
    except SyntaxError:
        lines = code.split("\n")
        for cut in range(len(lines), 0, -1):
            cand = "\n".join(lines[:cut])
            try:
                ast.parse(cand)
                code = cand
                break
            except SyntaxError:
                continue
        else:
            return None, None
    return stmt, code


def has_assert(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(n, ast.Assert) for n in ast.walk(tree))


def has_def(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in tree.body)


def marker_in_asserts(code):
    assert_text = "\n".join(l for l in code.split("\n") if "assert " in l).lower()
    hits = [m for m in MARKERS if m in assert_text]
    hits += [p.pattern for p in MARKER_RES if p.search(assert_text)]
    return hits


def _exec_worker(code):
    """Module-level (picklable) reference runner. Returns verdict string."""
    import subprocess as _sp
    import sys as _sys
    import tempfile as _tf
    import os as _os
    d = _tf.mkdtemp(prefix="u3w.")
    p = _os.path.join(d, "u.py")
    open(p, "w").write(code)
    try:
        r = _sp.run([_sys.executable, "-I", p], capture_output=True,
                    text=True, timeout=TIMEOUT)
    except _sp.TimeoutExpired:
        return "timeout"
    return "pass" if r.returncode == 0 else "fail"


def main():
    global CORPUS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--consumed", type=int, default=CONSUMED)
    ap.add_argument("--take", type=int, default=TAKE)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-run", type=int, default=0,
                    help="if >0, only execute the first N shape-ok rows (debug)")
    args = ap.parse_args()

    CORPUS = args.corpus
    files = shard_files()
    # row offset of each shard in the concatenated source order
    bounds, total = [], 0
    for f in files:
        n = sum(1 for ln in open(f) if ln.strip())
        bounds.append((f, total, n))
        total += n
    print(f"shards {len(files)} total_rows {total}", flush=True)

    wanted = tail_indices(total, args.consumed, args.take)
    # group wanted positions by shard
    by_shard = collections.defaultdict(list)
    for pos in wanted:
        for fidx, (f, off, n) in enumerate(bounds):
            if off <= pos < off + n:
                by_shard[fidx].append(pos - off)
                break
    wanted_set = {k: set(v) for k, v in by_shard.items()}

    dcon = _load_decontaminator().load_default()
    # RL-pool prompts + both SFT-A layers precomputed off-pod (the pool/SFT-A
    # live on digest), transferred via host: /tmp/sftb_rl_sfta_hashes.json
    dedup_hashes = set(json.load(open("/tmp/sftb_rl_sfta_hashes.json")))
    sfta_path = "/tmp/sftb_rl_sfta_hashes.json (RL pool + SFT-A sig+q layers, built on digest)"

    funnel = collections.Counter()
    survivors = []
    # Stage A (no subprocess): shape parse + def/assert gates, collect run jobs
    run_jobs = []  # (stmt, code, source_pos)
    rows_read = 0
    for fidx, (f, off, n) in enumerate(bounds):
        want = wanted_set.get(fidx)
        if not want:
            continue
        with open(f) as fh:
            for li, line in enumerate(fh):
                if li not in want:
                    continue
                rows_read += 1
                content = json.loads(line).get("content") or ""
                stmt, code = extract(content)
                if code is None:
                    funnel["no_parseable_code"] += 1
                    continue
                if not has_def(code):
                    funnel["no_def"] += 1
                    continue
                if not has_assert(code):
                    funnel["no_assert"] += 1
                    continue
                funnel["has_def_and_assert"] += 1
                run_jobs.append((stmt, code, off + li))
        print(f"stage A shard {fidx+1}/{len(files)} run_jobs {len(run_jobs)}", flush=True)

    if args.limit_run:
        run_jobs = run_jobs[:args.limit_run]

    # Stage B: parallel reference execution
    from multiprocessing import Pool
    green = []
    with Pool(processes=args.workers) as pool:
        verdicts = pool.imap_unordered(_exec_worker, (c for _s, c, _p in run_jobs),
                                       chunksize=64)
        for i, v in enumerate(verdicts):
            if v == "pass":
                green.append(i)
            else:
                funnel["ref_not_green_" + v] += 1
            if (i + 1) % 5000 == 0:
                print(f"exec {i+1}/{len(run_jobs)} green {len(green)}", flush=True)
    funnel["runs_green"] += len(green)

    # Stage C: hygiene + decontam + dedup, in original run_jobs order
    for i in green:
        stmt, code, srcpos = run_jobs[i]
        if marker_in_asserts(code):
            funnel["assert_self_correction_markers"] += 1
            continue
        funnel["assert_hygiene_ok"] += 1
        if dcon.hit(stmt or "") or dcon.hit(code):
            funnel["ngram_drop"] += 1
            continue
        funnel["decontam_ok"] += 1
        key = qhash(stmt)
        keyc = qhash(code)
        if key in dedup_hashes or keyc in dedup_hashes:
            funnel["sfta_or_rl_dedup"] += 1
            continue
        funnel["kept"] += 1
        survivors.append({"source": "code_ultra_l3_noexec_dc",
                          "source_pos": int(srcpos),
                          "prompt": stmt, "impl": code,
                          "kind": "ultra_l3_function"})

    os.makedirs(args.out, exist_ok=True)
    outp = os.path.join(args.out, "ultra_l3_sftb_candidates.jsonl")
    with open(outp, "w") as fh:
        for r in survivors:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"seed": SEED, "consumed_prefix": CONSUMED, "take": args.take,
               "tail_rows_read": rows_read, "workers": args.workers,
               "dedup_hashes_path": sfta_path,
               "dedup_hash_count": len(dedup_hashes),
               "funnel": dict(funnel), "kept": len(survivors),
               "survivors_path": outp}
    json.dump(summary, open(os.path.join(args.out, "ultra_l3_sftb_funnel.json"), "w"),
              indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)


def _selftest():
    # shuffle permutation reproduces train.py's random.Random(seed).shuffle
    texts = ["id%05d" % i for i in range(2000)]
    ref = texts[:]
    random.Random(SEED).shuffle(ref)
    perm = list(range(2000))
    random.Random(SEED).shuffle(perm)
    assert [texts[i] for i in perm] == ref, "tail permutation must match shuffle"
    # tail slice is contiguous after the consumed prefix
    ti = tail_indices(2000, 255, 10)
    assert len(ti) == 10 and ti == perm[255:265]
    # markers fire on the barber-style contradiction and pass clean asserts
    assert marker_in_asserts(
        "assert barber(4, [2, 5]) == 2  # So N=4 ->1, N=5->2\n")
    # benign "actually ... best" reasoning that agrees with the assert stays clean
    assert not marker_in_asserts(
        "assert solution([[1,2],[3,4]]) == 12  # Actually best: 4+8=12\n")
    assert not marker_in_asserts("x = 1\nassert f(1) == 1\n")
    # shape extractor separates statement from code
    stmt, code = extract('Write `f(x)` that returns x.\n\ndef f(x):\n    return x\n')
    assert stmt.strip().startswith("Write") and code.startswith("def f")
    assert has_def(code) and not has_assert(code)
    assert has_def("def f():\n    assert 1\n") and has_assert("def f():\n    assert 1\n")
    assert not has_def("x = 1\nassert x\n")
    print("sftb_ultra3_tail_funnel selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()

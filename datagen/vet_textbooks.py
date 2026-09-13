#!/usr/bin/env python3
import argparse
import ast
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "filters"))

from decontam_ngram import HUMANEVAL, MBPP, Decontaminator, decontam_fp  # noqa: E402,I001

DATA = os.path.join(ROOT, "data")
SRC_DIR = os.path.join(DATA, "corpus", "textbooks_claude_v41")
OUT_DIR = os.path.join(DATA, "corpus", "textbooks_claude_v41_vetted")
TOK = os.path.join(DATA, "tokenizer.json")
HANDREAD = os.path.join(ROOT, "runs", "textbooks_claude_handread.jsonl")
MIN_TOK, MAX_TOK = 800, 6000
EXEC_TIMEOUT = 10
SEED = 20260913
NEAR_THRESHOLD = 0.8
GROUP_NEAR_THRESHOLD = 0.5
SHINGLE = 5

FENCE = re.compile(r"```([^\n`]*)?\n(.*?)```", re.S)


def code_blocks(text):
    code, skipped = [], []
    for m in FENCE.finditer(text):
        lang = (m.group(1) or "").strip().lower()
        body = m.group(2)
        if lang and lang not in ("python", "py", "python3"):
            skipped.append({"lang": lang, "kind": "non_python"})
            continue
        try:
            ast.parse(body)
        except SyntaxError:
            skipped.append({"lang": lang, "kind": "not_toplevel_python"})
            continue
        code.append(body)
    return code, skipped


def truncated(text):
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    if not lines:
        return True
    if lines[-1].lstrip().startswith('#'):
        return True
    return bool(text.count('```') % 2)


def first_error(err):
    for line in err.strip().splitlines()[::-1]:
        if line.strip():
            return line.strip()[:240]
    return "nonzero exit"


_OS_NET = ("mkfifo", "socket", "websocket", "bind(", "listen(", "connect(", "os.fork",
           "subprocess", "requests.", "http")

# Sandbox refusals, not chapter defects. The chroot/namespaces have no network,
# RLIMIT_NPROC=64 (thread/process chapters: "can't start new thread",
# BlockingIOError on fork), a 2GB AS cap, a tiny /dev, and only stdlib, so a
# chapter that spins threads, forks, imports a third-party module or resolves a
# host fails because of the environment. Classify these from the error text
# directly; requiring an os/network token in the source let threading-only
# chapters be miscounted as code (536/698 "code" fails on 2026-09-14 were this).
_ENV_ERR = (
    "TIMEOUT", "Operation not permitted", "start new thread",
    "unable to start watchdog thread", "Resource temporarily unavailable",
    "Cannot allocate memory", "Memory allocation still failed",
    "Too many open files", "Name or service not known", "gaierror",
    "sigprocmask", "unshare:", "Connection refused", "Network is unreachable",
)
_ENV_PREFIX = ("ModuleNotFoundError:", "ImportError:")  # third-party absent from chroot


def fail_class(code, err):
    joined = "\n".join(code)
    if any(t in err for t in _ENV_ERR):
        return "env_unsandboxable"
    if any(err.lstrip().startswith(p) for p in _ENV_PREFIX):
        return "env_unsandboxable"
    if ("PermissionError" in err) and any(t in joined for t in _OS_NET):
        return "env_unsandboxable"
    if "header.payload" in joined and "is not defined" in err:
        return "prose_in_python_fence"
    return "code"


def add_fail(failures, r, source, gate, extra=None):
    rec = {"topic": r.get("topic"), "source": source, "gate": gate}
    if extra:
        rec.update(extra)
    failures.append(rec)


def run_chapter(code):
    from sandbox_exec import run_sandboxed
    program = "\n\n".join(code)
    rc, _out, err = run_sandboxed(program, timeout=EXEC_TIMEOUT)
    return rc == 0, ("" if rc == 0 else first_error(err))


def prose_norm(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def shingles(text):
    toks = prose_norm(text).split()
    if len(toks) < SHINGLE:
        return set()
    return {" ".join(toks[i:i + SHINGLE]) for i in range(len(toks) - SHINGLE + 1)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--handread", default=HANDREAD)
    ap.add_argument("--no_exec", action="store_true")
    ap.add_argument("--handread_n", type=int, default=20)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK)
    decon = Decontaminator.load_default(ROOT)

    files = sorted(glob.glob(os.path.join(args.src, "*.jsonl")))
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    n_in = len(rows)
    print(f"[load] {n_in} chapters from {len(files)} file(s)", flush=True)

    stats = {"rows_in": n_in, "decontam_fp": decontam_fp(HUMANEVAL, MBPP),
             "exec_pass": 0, "exec_fail": 0, "exec_fail_class": {},
             "exec_skipped_blocks": 0,
             "decon_drop": 0, "exact_dup_drop": 0, "near_dup_drop": 0,
             "group_near_dup_drop": 0,
             "length_drop": 0, "no_code_blocks": 0, "rows_out": 0, "tokens_out": 0}
    stats_by_source = {}

    def _src(source):
        s = stats_by_source.setdefault(source, {
            "rows_in": 0, "exec_pass": 0, "exec_fail": 0, "exec_fail_class": {},
            "decon_drop": 0, "exact_dup_drop": 0, "near_dup_drop": 0,
            "group_near_dup_drop": 0, "length_drop": 0, "no_code_blocks": 0,
            "rows_out": 0, "tokens_out": 0})
        return s

    failures = []

    seen_exact = set()
    kept = []
    kept_shingles = []
    index = defaultdict(list)
    group_index = defaultdict(list)

    for idx, r in enumerate(rows):
        if idx % 250 == 0:
            print(f"[progress] {idx}/{n_in} chapters evaluated", flush=True)
        text = r.get("text") or ""
        source = r.get("source", "unknown")
        ss = _src(source)
        ss["rows_in"] += 1

        ntok = len(tok.encode(text).ids)
        if truncated(text):
            stats["truncated_drop"] = stats.get("truncated_drop", 0) + 1
            add_fail(failures, r, source, "truncated")
            continue
        if not (MIN_TOK <= ntok <= MAX_TOK):
            stats["length_drop"] += 1
            ss["length_drop"] += 1
            add_fail(failures, r, source, "length", {"tokens": ntok})
            continue
        if decon.hit(text) is not None:
            stats["decon_drop"] += 1
            ss["decon_drop"] += 1
            add_fail(failures, r, source, "decontam")
            continue
        exact = hashlib.sha1(prose_norm(text).encode("utf-8")).hexdigest()
        if exact in seen_exact:
            stats["exact_dup_drop"] += 1
            ss["exact_dup_drop"] += 1
            add_fail(failures, r, source, "exact_dup")
            continue

        code, skipped = code_blocks(text)
        stats["exec_skipped_blocks"] += len(skipped)
        if not code:
            stats["no_code_blocks"] += 1
            ss["no_code_blocks"] += 1
            add_fail(failures, r, source, "no_code_blocks")
            continue

        if args.no_exec:
            ok, err = True, ""
        else:
            ok, err = run_chapter(code)
        if not ok:
            stats["exec_fail"] += 1
            ss["exec_fail"] += 1
            fc = fail_class(code, err)
            stats["exec_fail_class"][fc] = stats["exec_fail_class"].get(fc, 0) + 1
            ss["exec_fail_class"][fc] = ss["exec_fail_class"].get(fc, 0) + 1
            add_fail(failures, r, source, "exec", {"error": err, "fail_class": fc})
            continue
        stats["exec_pass"] += 1
        ss["exec_pass"] += 1

        sh = shingles(text)
        cands = {j for s in sh for j in index.get(s, ())}
        near = any(jaccard(sh, kept_shingles[j]) >= NEAR_THRESHOLD for j in cands)
        if near:
            stats["near_dup_drop"] += 1
            ss["near_dup_drop"] += 1
            add_fail(failures, r, source, "near_dup")
            continue

        group = r.get("seed_topic")
        gcands = {j for s in sh for j in group_index.get((group, s), ())} if group else set()
        gnear = any(jaccard(sh, kept_shingles[j]) >= GROUP_NEAR_THRESHOLD for j in gcands)
        if gnear:
            stats["group_near_dup_drop"] += 1
            ss["group_near_dup_drop"] += 1
            add_fail(failures, r, source, "group_near_dup", {"seed_topic": group})
            continue

        j = len(kept)
        seen_exact.add(exact)
        for s in sh:
            index[s].append(j)
            if group:
                group_index[(group, s)].append(j)
        kept_shingles.append(sh)
        kept.append((r, ntok, source))

    os.makedirs(args.out, exist_ok=True)
    by_source = defaultdict(list)
    # Normalise the row's own n field to len(text)//4 on EVERY kept row. Some
    # delivered batches wrote a chapter id / 0 into n; the vetted output is the
    # stage-2 input and must carry the spec value so a downstream sum of n is a
    # character-derived token proxy, not corrupted source data.
    n_recomputed = 0
    n_src_sum = defaultdict(int)
    for r, _ntok, source in kept:
        n_src_sum[source] += r.get("n", 0)
        want = len(r.get("text") or "") // 4
        if r.get("n") != want:
            r["n"] = want
            n_recomputed += 1
        by_source[source].append(r)
    stats["n_recomputed_rows"] = n_recomputed
    stats["n_field_sum_source_corrupted"] = sum(n_src_sum.values())
    for source, recs in by_source.items():
        path = os.path.join(args.out, f"{source}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Stats are recomputed from the files ON DISK after writing, so they describe
    # the vetted directory itself. tokens = gate-tokenizer count (the real number
    # to read); n_field_sum = sum of the normalised n (len(text)//4) on output;
    # n_field_sum_source_corrupted keeps what the delivered source summed to.
    disk = {}
    for source in by_source:
        path = os.path.join(args.out, f"{source}.jsonl")
        dtok = dn = drows = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rr = json.loads(line)
                dtok += len(tok.encode(rr.get("text") or "").ids)
                dn += rr.get("n", 0)
                drows += 1
        disk[source] = {"rows": drows, "tokens": dtok, "n_field_sum": dn}
    for source, d in disk.items():
        ss = stats_by_source[source]
        ss["rows_out"] = d["rows"]
        ss["tokens_out"] = d["tokens"]
        ss["n_field_sum"] = d["n_field_sum"]
        ss["n_field_sum_source_corrupted"] = n_src_sum[source]
    stats["rows_out"] = sum(d["rows"] for d in disk.values())
    stats["tokens_out"] = sum(d["tokens"] for d in disk.values())
    stats["n_field_sum"] = sum(d["n_field_sum"] for d in disk.values())
    for ss in stats_by_source.values():
        ss.setdefault("n_field_sum", 0)

    env_total = stats["exec_fail_class"].get("env_unsandboxable", 0)
    denom_all = stats["exec_pass"] + stats["exec_fail"]
    # Code-only denominator: env_unsandboxable (thread/fork/network/third-party
    # refusals from the chroot) are not chapter defects and must not fail a batch
    # for writing code the sandbox cannot run. Keep the all-failures rate too so
    # the two readings stay comparable and a sandbox regression is visible.
    denom_code = stats["exec_pass"] + stats["exec_fail"] - env_total
    stats["exec_pass_rate_all_failures"] = (
        round(stats["exec_pass"] / denom_all, 4) if denom_all else None)
    stats["exec_pass_rate"] = (
        round(stats["exec_pass"] / denom_code, 4) if denom_code else None)
    stats["length_bounds_tokens"] = [MIN_TOK, MAX_TOK]
    stats["near_dup_jaccard"] = NEAR_THRESHOLD
    stats["group_near_dup_jaccard"] = GROUP_NEAR_THRESHOLD
    for ss in stats_by_source.values():
        env_s = ss["exec_fail_class"].get("env_unsandboxable", 0)
        da = ss["exec_pass"] + ss["exec_fail"]
        dc = da - env_s
        ss["exec_pass_rate_all_failures"] = round(ss["exec_pass"] / da, 4) if da else None
        ss["exec_pass_rate"] = round(ss["exec_pass"] / dc, 4) if dc else None
    stats["by_source"] = stats_by_source

    with open(os.path.join(args.out, "vet_textbooks_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    with open(os.path.join(args.out, "vet_failures.jsonl"), "w", encoding="utf-8") as fh:
        for f in failures:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")

    rng = random.Random(SEED)
    by_source_kept = defaultdict(list)
    for item in kept:
        by_source_kept[item[2]].append(item)

    # If a scored handread file already exists at args.handread, keep it and fold
    # the scores into stats. Otherwise emit a fresh blank sample of handread_n
    # per source.
    handread_rows = []
    if os.path.exists(args.handread):
        with open(args.handread, encoding="utf-8") as fh:
            handread_rows = [json.loads(l) for l in fh if l.strip()]
    scored = [r for r in handread_rows
              if r.get("correctness_1_5") is not None and r.get("pedagogy_1_5") is not None]
    if not scored:
        with open(args.handread, "w", encoding="utf-8") as fh:
            for source in sorted(by_source_kept):
                pool = by_source_kept[source]
                sample = rng.sample(pool, min(args.handread_n, len(pool)))
                for r, ntok, src in sample:
                    fh.write(json.dumps({"topic": r.get("topic"), "source": src,
                                         "tokens": ntok, "text": r.get("text"),
                                         "correctness_1_5": None, "pedagogy_1_5": None,
                                         "reviewer_note": None}, ensure_ascii=False) + "\n")
    else:
        def _mean(xs, k):
            return round(sum(x[k] for x in xs) / len(xs), 3) if xs else None

        hr = {"n": len(scored),
              "correctness_mean": _mean(scored, "correctness_1_5"),
              "pedagogy_mean": _mean(scored, "pedagogy_1_5"),
              "bar": 4.0,
              "by_source": {}}
        for source in sorted({r.get("source") for r in scored}):
            sub = [r for r in scored if r.get("source") == source]
            sc = _mean(sub, "correctness_1_5")
            sp = _mean(sub, "pedagogy_1_5")
            hr["by_source"][source] = {
                "n": len(sub),
                "correctness_mean": sc,
                "pedagogy_mean": sp,
                "passes_bar": sc is not None and sp is not None and sc >= 4 and sp >= 4,
            }
        c = hr["correctness_mean"]
        p = hr["pedagogy_mean"]
        hr["overall_mean"] = round((c + p) / 2, 3)
        # Certification is PER GENERATOR: every source with a sample must clear
        # both axes. A pooled mean can hide one weak source (gen-A 3.9 under a
        # 4.117 pooled correctness on 2026-09-13), so the overall flag is the
        # conjunction of the per-source flags, not a test of the pooled mean.
        hr["passes_bar"] = bool(hr["by_source"]) and all(
            v["passes_bar"] for v in hr["by_source"].values()
        )
        stats["handread"] = hr
        # rewrite stats now that handread is folded in
        with open(os.path.join(args.out, "vet_textbooks_stats.json"), "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)

    print("VET_STATS " + json.dumps(stats), flush=True)
    print(f"[out] {len(kept)} chapters -> {args.out}; {len(failures)} failures logged; "
          f"handread sample -> {args.handread}", flush=True)


if __name__ == "__main__":
    main()

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

from decontam_ngram import Decontaminator  # noqa: E402

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
    if text.count('```') % 2:
        return True
    return False


def first_error(err):
    for line in err.strip().splitlines()[::-1]:
        if line.strip():
            return line.strip()[:240]
    return "nonzero exit"


_OS_NET = ("mkfifo", "socket", "websocket", "bind(", "listen(", "connect(", "os.fork",
           "subprocess", "requests.", "http")


def fail_class(code, err):
    joined = "\n".join(code)
    if ("TIMEOUT" in err or "PermissionError" in err
            or "Operation not permitted" in err) and any(t in joined for t in _OS_NET):
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

    stats = {"rows_in": n_in, "exec_pass": 0, "exec_fail": 0, "exec_fail_class": {},
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

    for r in rows:
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
    for r, ntok, source in kept:
        by_source[source].append(r)
        stats["tokens_out"] += ntok
        ss = stats_by_source[source]
        ss["rows_out"] += 1
        ss["tokens_out"] += ntok
    for source, recs in by_source.items():
        path = os.path.join(args.out, f"{source}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["rows_out"] = len(kept)
    denom = stats["exec_pass"] + stats["exec_fail"]
    stats["exec_pass_rate"] = round(stats["exec_pass"] / denom, 4) if denom else None
    stats["length_bounds_tokens"] = [MIN_TOK, MAX_TOK]
    stats["near_dup_jaccard"] = NEAR_THRESHOLD
    stats["group_near_dup_jaccard"] = GROUP_NEAR_THRESHOLD
    for ss in stats_by_source.values():
        d = ss["exec_pass"] + ss["exec_fail"]
        ss["exec_pass_rate"] = round(ss["exec_pass"] / d, 4) if d else None
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
    with open(args.handread, "w", encoding="utf-8") as fh:
        for source in sorted(by_source_kept):
            pool = by_source_kept[source]
            sample = rng.sample(pool, min(args.handread_n, len(pool)))
            for r, ntok, src in sample:
                fh.write(json.dumps({"topic": r.get("topic"), "source": src,
                                     "tokens": ntok, "text": r.get("text"),
                                     "correctness_1_5": None, "pedagogy_1_5": None,
                                     "reviewer_note": None}, ensure_ascii=False) + "\n")

    print("VET_STATS " + json.dumps(stats), flush=True)
    print(f"[out] {len(kept)} chapters -> {args.out}; {len(failures)} failures logged; "
          f"handread sample -> {args.handread}", flush=True)


if __name__ == "__main__":
    main()

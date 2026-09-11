#!/usr/bin/env python3
"""Write 13-gram-decontaminated copies of the non-Ultra gate domains (ae-7).

For each domain, write data/corpus/<domain>_dc/:
  - shard with ZERO 13-gram hits: hard-linked from the source (identical bytes, no
    copy cost), and
  - shard with ANY hit: rewritten line-by-line dropping contaminated rows.
Source dirs are never touched. A build_corpus_stats.json is written per output
domain carrying rows scanned/dropped, the decontam_fp (content hash of
filters/decontam_ngram.py + the gate files -- vocab-independent), and corpus_fp.

    python3 scripts/filter_gate_domains.py \
        --domains code_py_starcoder,code_py_rp1t,cot,math_owm_stage2,en_c4_stage2
"""
# restartable: hard-link clean shards and rewrite hit shards; an interrupt re-runs
# and converges (a clean shard hard-links idempotently).
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.environ.get("AUPAI_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

HUMANEVAL_REL = os.path.join("data", "eval", "humaneval", "humaneval_164.jsonl")
MBPP_REL = os.path.join("data", "eval", "mbpp_holdouts.jsonl")
N = 13


def _corpus_fp(files):
    import hashlib

    h = hashlib.sha256()
    for p in files:
        with open(p, "rb") as bf:
            h.update(bf.read())
    return h.hexdigest()[:16]


def _process_shard(args):
    """Scan one shard; hard-link if clean, rewrite dropping hits. Returns stats.
    Each pool worker builds the gram set once (lazy default, fork-inherited cwd)."""
    p, dst, root = args
    sys.path.insert(0, root)
    from filters.decontam_ngram import default as _default_decon

    decon = _default_decon()
    name = os.path.basename(p)
    kept_lines, scanned, dropped = [], 0, 0
    problems = {}
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            scanned += 1
            try:
                c = json.loads(s).get("content", "")
            except json.JSONDecodeError:
                c = ""
            hit = decon.hit(c)
            if hit:
                dropped += 1
                problems[hit["problem"]] = problems.get(hit["problem"], 0) + 1
            else:
                kept_lines.append(line if line.endswith("\n") else line + "\n")
    out_p = os.path.join(dst, name)
    if os.path.exists(out_p) or os.path.islink(out_p):
        os.remove(out_p)
    if dropped == 0:
        try:
            os.link(p, out_p)
            action = "linked"
        except OSError:
            import shutil

            shutil.copyfile(p, out_p)
            action = "copied"
    else:
        with open(out_p, "w", encoding="utf-8") as fh:
            fh.writelines(kept_lines)
        action = "rewritten"
    return {"file": name, "scanned": scanned, "dropped": dropped, "action": action,
            "problems": problems}


def filter_domain(domain, root, out_root, workers):
    src = os.path.join(root, "data", "corpus", domain)
    dst = os.path.join(out_root, f"{domain}_dc")
    os.makedirs(dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, "*.jsonl")))
    if not files:
        return {"domain": domain, "error": "no source shards"}

    tasks = [(p, dst, root) for p in files]
    scanned = dropped = rewritten = linked = 0
    per_problem = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for st in ex.map(_process_shard, tasks):
            scanned += st["scanned"]
            dropped += st["dropped"]
            rewritten += st["action"] == "rewritten"
            linked += st["action"] in ("linked", "copied")
            for pid, c in st["problems"].items():
                per_problem[pid] = per_problem.get(pid, 0) + c

    from filters.decontam_ngram import decontam_fp

    stamp = {
        "domain": f"{domain}_dc",
        "source_domain": domain,
        "filter": "filters/decontam_ngram.py 13-word-token containment vs HumanEval+MBPP",
        "n": N,
        "rows_scanned": scanned,
        "rows_dropped": dropped,
        "drop_fraction": (dropped / scanned) if scanned else 0.0,
        "shards_total": len(files), "shards_rewritten": rewritten, "shards_hardlinked": linked,
        "distinct_problems_hit": len(per_problem),
        "problems": per_problem,
        "corpus_fp_source": _corpus_fp(files),
        "workers": workers,
        # module fp is vocab-independent (module + gate files), attached explicitly
        "decontam_fp": decontam_fp(os.path.join(root, HUMANEVAL_REL),
                                   os.path.join(root, MBPP_REL)),
    }
    with open(os.path.join(dst, "build_corpus_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, indent=1)
    return stamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--domains",
                    default="code_py_starcoder,code_py_rp1t,cot,math_owm_stage2,en_c4_stage2")
    ap.add_argument("--out_root", default="")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    root = a.root
    out_root = a.out_root or os.path.join(root, "data", "corpus")
    # benchmark files gate a production run; workers load_default() off this root.
    for rel in (HUMANEVAL_REL, MBPP_REL):
        if not os.path.exists(os.path.join(root, rel)):
            sys.exit(f"decontam benchmark missing: {os.path.join(root, rel)}")
    summary = {}
    for d in [x for x in a.domains.split(",") if x]:
        st = filter_domain(d, root, out_root, a.workers)
        if "error" in st:
            print(f"{d}: {st['error']}", flush=True)
        else:
            print(f"{d}_dc: scanned {st['rows_scanned']} dropped {st['rows_dropped']} "
                  f"({st['drop_fraction']*100:.4f}%) rewritten {st['shards_rewritten']} "
                  f"linked {st['shards_hardlinked']} problems {st['distinct_problems_hit']}",
                  flush=True)
        summary[d] = st
    os.makedirs(os.path.join(root, "runs"), exist_ok=True)
    with open(os.path.join(root, "runs", "gate_decontam_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("wrote runs/gate_decontam_summary.json")


if __name__ == "__main__":
    main()

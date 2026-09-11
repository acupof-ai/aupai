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
_WORKER_DECON = {}


def _worker_decon(root, cls):
    """Per-process cached gate gram set; built once per root."""
    d = _WORKER_DECON.get(root)
    if d is None:
        d = cls.load_default(root)
        _WORKER_DECON[root] = d
    return d


def _corpus_fp(files):
    import hashlib

    h = hashlib.sha256()
    for p in files:
        with open(p, "rb") as bf:
            h.update(bf.read())
    return h.hexdigest()[:16]


def _process_shard(args):
    """Scan one shard; hard-link if clean, rewrite dropping hits. Returns stats.
    Each pool worker builds the gate gram set once per root and caches it."""
    p, dst, root = args
    sys.path.insert(0, root)
    from filters.decontam_ngram import Decontaminator

    decon = _worker_decon(root, Decontaminator)
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

    from datagen.corpus_fingerprint import fp_dir
    from filters.decontam_ngram import decontam_fp

    # "fingerprint" is the field train.py _assert_mix_domains compares to the live dir at
    # launch. fp_dir excludes build_corpus_stats.json, so compute it over the finished
    # shards BEFORE writing the stamp; the stamp must not hash itself.
    fingerprint = fp_dir(dst)
    stamp = {
        "domain": f"{domain}_dc",
        "source_domain": domain,
        "fingerprint": fingerprint,
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


def _selftest():
    """Known-answer build + the launch guard that the first stamps failed.

    Builds a temp root with synthetic gate files and two source shards (one row
    verbatim-holds a canonical solution, two are clean), runs filter_domain, and
    asserts:
      - the contaminated row is dropped, clean rows survive, shard actions right;
      - build_corpus_stats.json "fingerprint" equals fp_dir of the OUTPUT dir and
        passes train.py _assert_mix_domains (the launch-time guard) -- the bug 98
        caught was the missing field, which made the launch refuse every _dc domain;
      - mutating the stamped fingerprint makes the guard REFUSE.
    """
    import tempfile

    sol = ("if n == 0:\n        return 0\n    if n == 1:\n        return 1\n"
           "    return fib(n - 1) + fib(n - 2)\n")
    he = {"task_id": "HumanEval/55", "prompt": "def fibonacci(n):\n",
          "canonical_solution": sol, "test": "def check():\n    assert fib(0) == 0\n",
          "entry_point": "fibonacci"}
    mbpp = {"task_id": "mbpp-1", "text": "write fibonacci", "code": "x = 1\n"}
    clean = {"content": "def load_config(path):\n    with open(path) as fh:\n        return dict(line.strip().split('=') for line in fh if '=' in line)\n"}
    with tempfile.TemporaryDirectory() as root:
        he_dir = os.path.join(root, "data", "eval", "humaneval")
        os.makedirs(he_dir)
        with open(os.path.join(he_dir, "humaneval_164.jsonl"), "w") as fh:
            fh.write(json.dumps(he) + "\n")
        mbpp_dir = os.path.join(root, "data", "eval")
        with open(os.path.join(mbpp_dir, "mbpp_holdouts.jsonl"), "w") as fh:
            fh.write(json.dumps(mbpp) + "\n")
        src = os.path.join(root, "data", "corpus", "dom")
        os.makedirs(src)
        with open(os.path.join(src, "s0.jsonl"), "w") as fh:
            fh.write(json.dumps({"content": "def fibonacci(n):\n" + sol}) + "\n")
            fh.write(json.dumps(clean) + "\n")
        with open(os.path.join(src, "s1.jsonl"), "w") as fh:
            fh.write(json.dumps(clean) + "\n")

        st = filter_domain("dom", root, os.path.join(root, "data", "corpus"), 1)
        assert st["rows_scanned"] == 3, st
        assert st["rows_dropped"] == 1, st
        assert st["shards_rewritten"] == 1 and st["shards_hardlinked"] == 1, st
        dst = os.path.join(root, "data", "corpus", "dom_dc")
        n_kept = sum(1 for _ in open(os.path.join(dst, "s0.jsonl")))
        assert n_kept == 1, n_kept
        stamp = json.load(open(os.path.join(dst, "build_corpus_stats.json")))
        assert stamp["fingerprint"], "stamp lacks the fingerprint train guard reads"
        from datagen.corpus_fingerprint import fp_dir
        assert stamp["fingerprint"] == fp_dir(dst)

        try:
            import train
        except Exception as e:  # torch absent on a dev box: the field equality above is
            print(f"filter_gate_domains selftest OK (train guard SKIPPED, {type(e).__name__}: {e})")
            return
        fps = train._assert_mix_domains(["dom_dc"], os.path.join(root, "data", "corpus"))
        assert fps["dom_dc"] == stamp["fingerprint"]
        # a drifted stamp must be REFUSED by the launch guard
        stamp["fingerprint"] = "deadbeefdeadbeef"
        with open(os.path.join(dst, "build_corpus_stats.json"), "w") as fh:
            json.dump(stamp, fh)
        try:
            train._assert_mix_domains(["dom_dc"], os.path.join(root, "data", "corpus"))
        except AssertionError:
            pass
        else:
            raise AssertionError("drifted _dc stamp was accepted by the launch guard")
    print("filter_gate_domains selftest OK (drop counts + fingerprint passes launch guard)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()

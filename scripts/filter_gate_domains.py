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
GSM8K_REL = os.path.join("data", "eval", "gsm8k_test.jsonl")
MATH500_REL = os.path.join("data", "eval", "math_test_500.jsonl")
N = 13
_WORKER_DECON = {}


def _worker_decon(root, cls, extra_math=False):
    """Per-process cached gate gram set; built once per (root, gate set)."""
    key = (root, bool(extra_math))
    d = _WORKER_DECON.get(key)
    if d is None:
        d = cls.load_default(root, extra_math=extra_math)
        _WORKER_DECON[key] = d
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
    p, dst, root, extra_math = args
    sys.path.insert(0, root)
    from filters.decontam_ngram import Decontaminator

    decon = _worker_decon(root, Decontaminator, extra_math=extra_math)
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


def filter_domain(domain, root, out_root, workers, out_name=None, extra_math=False):
    src = os.path.join(root, "data", "corpus", domain)
    out_name = out_name or f"{domain}_dc"
    dst = os.path.join(out_root, out_name)
    os.makedirs(dst, exist_ok=True)
    # THE HOLDOUT SLICE IS METADATA, NOT CORPUS. A build carrying --phase writes
    # {out}/holdout_slice_{phase}.jsonl beside its shards, and a plain `*.jsonl` glob picks it
    # up: measured on the 2026-09-19 en_c4_stage2 rebuild, `shards_total` read 243 and
    # `rows_scanned` 11,309,623 where the corpus is 242 shards and 11,309,622 documents -- the
    # extra shard and the extra row are both this one file, whose only line is
    # {"phase": ..., "rule_fp": ..., "n": ...}. It is not a document and cannot be
    # decontaminated; counting it makes a reader take 243 for a shard count.
    # `build_corpus.py` already refuses this file in a rewrite set (`_near_write_stats`'s
    # assert); this is the same rule at the other reader.
    files = sorted(
        p for p in glob.glob(os.path.join(src, "*.jsonl"))
        if not os.path.basename(p).startswith("holdout_slice_")
    )
    if not files:
        return {"domain": domain, "error": "no source shards"}

    tasks = [(p, dst, root, extra_math) for p in files]
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

    # INHERIT the garbage-filter provenance from the SOURCE domain. Decontam runs no
    # pass1/2/3_garbage filter itself: a clean shard is a byte-for-byte hardlink and a hit
    # shard is a subset with 13-gram rows dropped, so every surviving byte already passed the
    # source build's garbage filters. The _dc domain's filters_fp is therefore the source's
    # value, copied -- NOT re-derived. A source that never recorded filters_fp means its
    # garbage provenance is unknown; silently writing nothing would make check_corpus_filters_fp
    # read the _dc domain as "built before stamps existed" or, worse, let it inherit a value
    # we guessed, so the decontam run FAILs loud (ae-10, no false inheritance).
    source_stats_p = os.path.join(src, "build_corpus_stats.json")
    source_filters_fp = None
    if os.path.isfile(source_stats_p):
        with open(source_stats_p, encoding="utf-8") as sf:
            source_filters_fp = json.load(sf).get("filters_fp")
    if not source_filters_fp:
        return {
            "domain": domain,
            "error": (f"source {domain}/build_corpus_stats.json has no filters_fp; cannot "
                      "inherit garbage-filter provenance into {domain}_dc -- rebuild the "
                      "source with current filters first (no false inheritance)"),
        }

    # "fingerprint" is the field train.py _assert_mix_domains compares to the live dir at
    # launch. fp_dir excludes build_corpus_stats.json, so compute it over the finished
    # shards BEFORE writing the stamp; the stamp must not hash itself.
    fingerprint = fp_dir(dst)
    gate_rel = [HUMANEVAL_REL, MBPP_REL] + ([GSM8K_REL, MATH500_REL] if extra_math else [])
    gate_desc = "HumanEval+MBPP" + ("+GSM8K+MATH-500" if extra_math else "")
    stamp = {
        "domain": out_name,
        "source_domain": domain,
        "fingerprint": fingerprint,
        "filter": f"filters/decontam_ngram.py 13-word-token containment vs {gate_desc}",
        "extra_math_gates": bool(extra_math),
        "n": N,
        "rows_scanned": scanned,
        "rows_dropped": dropped,
        "drop_fraction": (dropped / scanned) if scanned else 0.0,
        "shards_total": len(files), "shards_rewritten": rewritten, "shards_hardlinked": linked,
        "distinct_problems_hit": len(per_problem),
        "problems": per_problem,
        "corpus_fp_source": _corpus_fp(files),
        "workers": workers,
        # garbage-filter provenance inherited from the source build (decontam adds none);
        # a stale source value mismatches through check_corpus_filters_fp's pair logic.
        "filters_fp": source_filters_fp,
        # module fp is vocab-independent (module + gate files), attached explicitly
        "decontam_fp": decontam_fp(*(os.path.join(root, r) for r in gate_rel)),
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
    ap.add_argument("--extra_math", action="store_true",
                    help="also gate against data/eval/gsm8k_test.jsonl and math_test_500.jsonl")
    ap.add_argument("--out_names", default="",
                    help="optional comma list aligned with --domains; blank entry keeps <dom>_dc")
    a = ap.parse_args()
    root = a.root
    out_root = a.out_root or os.path.join(root, "data", "corpus")
    # benchmark files gate a production run; workers load_default() off this root.
    gate_rels = [HUMANEVAL_REL, MBPP_REL] + ([GSM8K_REL, MATH500_REL] if a.extra_math else [])
    for rel in gate_rels:
        if not os.path.exists(os.path.join(root, rel)):
            sys.exit(f"decontam benchmark missing: {os.path.join(root, rel)}")
    doms = [x for x in a.domains.split(",") if x]
    names = [x or None for x in a.out_names.split(",")] if a.out_names else []
    if names and len(names) != len(doms):
        sys.exit("--out_names must align one-per-domain with --domains (blanks allowed)")
    summary = {}
    for i, d in enumerate(doms):
        out_name = names[i] if names else None
        st = filter_domain(d, root, out_root, a.workers, out_name=out_name,
                           extra_math=a.extra_math)
        label = out_name or f"{d}_dc"
        if "error" in st:
            print(f"{d}: {st['error']}", flush=True)
        else:
            print(f"{label}: scanned {st['rows_scanned']} dropped {st['rows_dropped']} "
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
        # The source build records the garbage-filter provenance the _dc domain must INHERIT.
        source_filters_fp = "srcfp0123456789ab"
        with open(os.path.join(src, "build_corpus_stats.json"), "w") as fh:
            json.dump({"filters_fp": source_filters_fp}, fh)

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
        # ae-10: the _dc stamp carries the SOURCE garbage filters_fp verbatim (inherited, not
        # recomputed) beside its own decontam_fp.
        assert stamp["filters_fp"] == source_filters_fp, stamp
        assert stamp["decontam_fp"], "stamp must keep its own decontam_fp"

        # THE HOLDOUT SLICE IS NOT A SHARD. A --phase build leaves holdout_slice_{phase}.jsonl
        # in the source dir beside its shards, and a bare `*.jsonl` glob counts it. Measured on
        # the 2026-09-19 en_c4_stage2 rebuild: shards_total 243 and rows_scanned 11,309,623
        # against a real 242 shards / 11,309,622 documents -- both off by exactly this file.
        #
        # THE MUTATION IS THE ASSERTION'S OTHER ARM: the same slice added to the SAME fixture
        # must not move either count. Without the exclusion this case goes red by name, which is
        # what makes it a test of the fix rather than of the fixture. A slice whose content is
        # valid-looking JSON is used deliberately: a file the shard parser would happily read is
        # the harder case, and the one a glob actually admits.
        with open(os.path.join(src, "holdout_slice_p.jsonl"), "w") as fh:
            fh.write(json.dumps({"phase": "p", "rule_fp": "f" * 16, "n": 0}) + "\n")
        with_slice = filter_domain("dom", root, os.path.join(root, "data", "corpus"), 1)
        assert with_slice["shards_total"] == st["shards_total"], (
            f"a holdout_slice_ file moved shards_total: {st['shards_total']} -> "
            f"{with_slice['shards_total']} -- the slice is metadata, not a shard")
        assert with_slice["rows_scanned"] == st["rows_scanned"], (
            f"a holdout_slice_ file moved rows_scanned: {st['rows_scanned']} -> "
            f"{with_slice['rows_scanned']} -- its header line is not a document")
        assert with_slice["rows_dropped"] == st["rows_dropped"], with_slice
        # and the slice itself must not be COPIED into the _dc dir as if it were a shard: the
        # counts above would still be right if the file were merely relabelled downstream.
        assert not os.path.exists(
            os.path.join(root, "data", "corpus", "dom_dc", "holdout_slice_p.jsonl")), \
            "the filter carried a holdout_slice file through as corpus"
        # the real shards are still all there -- an exclusion that dropped s0/s1 would pass the
        # three assertions above by emptying the work.
        assert with_slice["shards_rewritten"] == st["shards_rewritten"], with_slice
        assert with_slice["shards_hardlinked"] == st["shards_hardlinked"], with_slice
        os.remove(os.path.join(src, "holdout_slice_p.jsonl"))

        # SOURCE MISSING filters_fp -> the decontam run FAILs loud (no false inheritance);
        # no _dc stamp is written, so the launch guard cannot be faked on unknown provenance.
        src2 = os.path.join(root, "data", "corpus", "dom_nofp")
        os.makedirs(src2)
        with open(os.path.join(src2, "s0.jsonl"), "w") as fh:
            fh.write(json.dumps(clean) + "\n")
        with open(os.path.join(src2, "build_corpus_stats.json"), "w") as fh:
            json.dump({"fingerprint": "x" * 16}, fh)  # deliberately no filters_fp
        bad = filter_domain("dom_nofp", root, os.path.join(root, "data", "corpus"), 1)
        assert "error" in bad and "filters_fp" in bad["error"], bad
        assert not os.path.exists(os.path.join(root, "data", "corpus", "dom_nofp_dc",
                                               "build_corpus_stats.json")), \
            "a source with no filters_fp must not publish a _dc stamp"
        # a source with NO stamp file at all fails the same way (not just an empty field)
        src3 = os.path.join(root, "data", "corpus", "dom_nostamp")
        os.makedirs(src3)
        with open(os.path.join(src3, "s0.jsonl"), "w") as fh:
            fh.write(json.dumps(clean) + "\n")
        bad2 = filter_domain("dom_nostamp", root, os.path.join(root, "data", "corpus"), 1)
        assert "error" in bad2 and "filters_fp" in bad2["error"], bad2

        # EXTRA MATH GATES (zh 2026-09-28): --extra_math shingles GSM8K + MATH-500,
        # --out_name names the output domain. A row verbatim-holding a 13+ word-token
        # span of a math answer must drop; clean rows survive.
        gsm_pad = " ".join(["word"] * 12)
        gsm = {"question": "How many apples remain after giving some away today?",
               "answer": f"{gsm_pad} she makes nine dollars at the market every single "
                         "weekday morning without fail here"}
        m500 = {"instruction": "compute the integral value here", "output": "x = 1\n"}
        with open(os.path.join(mbpp_dir, "gsm8k_test.jsonl"), "w") as fh:
            fh.write(json.dumps(gsm) + "\n")
        with open(os.path.join(mbpp_dir, "math_test_500.jsonl"), "w") as fh:
            fh.write(json.dumps(m500) + "\n")
        srcm = os.path.join(root, "data", "corpus", "domm")
        os.makedirs(srcm)
        with open(os.path.join(srcm, "s0.jsonl"), "w") as fh:
            fh.write(json.dumps({"content": gsm["answer"] + " trailing context"}) + "\n")
            fh.write(json.dumps(clean) + "\n")
        with open(os.path.join(srcm, "build_corpus_stats.json"), "w") as fh:
            json.dump({"filters_fp": source_filters_fp}, fh)
        mst = filter_domain("domm", root, os.path.join(root, "data", "corpus"), 1,
                            out_name="domm_web_dc", extra_math=True)
        assert "error" not in mst, mst
        assert mst["rows_scanned"] == 2 and mst["rows_dropped"] == 1, mst
        assert any(k.startswith("gsm8k:") for k in mst["problems"]), mst["problems"]
        assert mst["domain"] == "domm_web_dc" and mst["extra_math_gates"] is True, mst
        assert os.path.isdir(os.path.join(root, "data", "corpus", "domm_web_dc"))
        # without the extra gate the same row survives
        base = filter_domain("domm", root, os.path.join(root, "data", "corpus"), 1,
                            out_name="domm_base_dc", extra_math=False)
        assert base["rows_dropped"] == 0, base

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

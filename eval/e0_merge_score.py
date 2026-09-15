"""Merge stage-2 shard preds and score FULL + CLEAN for HE and MBPP.

Eight cards each ran --shard_i i --shard_n 8 over the FULL dataset (HumanEval 164,
MBPP sanitized 427), so the union of shards is every task exactly once and no
clean-only data files are needed. CLEAN is recomputed here by excluding the r3
six-domain contamination union recorded in the manifests:
  HumanEval CLEAN = 164 - r3_humaneval_union (8) = 156
  MBPP      CLEAN = 427 - r3_mbpp_union      (89) = 338
The per-shard runs pass --no_clean because a shard is not the full 427 and the
in-script clean-complement invariant only holds for the whole dataset.

n>1 rows carry one (task_id, sample_idx) per sample; this verifies the expected
counts and computes the pass rate over samples (sum c / (tasks*n)), the same
quantity the generators print. paired_bootstrap.py consumes the merged file for
the ET/EC vs E0 one-sided 95% CI.
"""
import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HE_UNION = os.path.join(ROOT, "runs", "contam_r3_he_union.json")
MBPP_UNION = os.path.join(ROOT, "runs", "contam_r3_mbpp_union.json")


def _read_preds(path):
    rows = []
    header = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_header"):
                header = obj
            else:
                rows.append(obj)
    return header, rows


def _exclude_he():
    u = json.load(open(HE_UNION, encoding="utf-8"))
    return set(u["r3_humaneval_union"])


def _exclude_mbpp():
    import re
    u = json.load(open(MBPP_UNION, encoding="utf-8"))
    return {int(re.fullmatch(r"mbpp427:(\d+)", s).group(1)) for s in u["r3_mbpp_union"]}


def merge(paths, total_tasks, n, excluded, bench):
    """(summary, merged_rows). Verifies exact shard coverage and sample count."""
    headers, by_task = [], {}
    shard_ids = set()
    for p in sorted(paths):
        h, rows = _read_preds(p)
        if h:
            headers.append(h)
            if h.get("shard_n"):
                shard_ids.add(h.get("shard_i"))
        for r in rows:
            tid = r["task_id"]
            si = r.get("sample_idx", 0)
            key = (tid, si)
            if key in by_task:
                raise SystemExit(f"duplicate ({tid},{si}) across shards in {p}")
            by_task[key] = r
    if shard_ids and shard_ids != set(range(max(shard_ids) + 1)):
        raise SystemExit(f"shard set {sorted(shard_ids)} is not 0..k; refuse to score a gap")
    tasks = {tid for tid, _ in by_task}
    if len(tasks) != total_tasks:
        raise SystemExit(f"{bench}: merged {len(tasks)} tasks, expected {total_tasks}")
    # every task must carry exactly n samples
    bad_n = sorted(t for t in tasks
                   if sum(1 for (tid, _) in by_task if tid == t) != n)
    if bad_n:
        raise SystemExit(f"{bench}: {len(bad_n)} tasks do not have n={n} samples "
                         f"(e.g. {bad_n[:3]})")
    merged = [by_task[k] for k in sorted(by_task, key=lambda z: (str(z[0]), z[1]))]

    def rate(keep):
        sel = [r for r in merged if keep(r["task_id"])]
        c = sum(int(r["ok"]) for r in sel)
        return c, len(sel), (c / len(sel) if sel else float("nan"))

    fc, fd, fr = rate(lambda _t: True)
    cc, cd, cr = rate(lambda t: t not in excluded)
    empty = sum(1 for r in merged if r.get("empty"))
    return {
        "benchmark": bench, "n": n, "tasks_total": total_tasks,
        "full_pass": fc, "full_denom": fd, "full_rate": round(fr, 6),
        "clean_pass": cc, "clean_denom": cd, "clean_rate": round(cr, 6),
        "empty": empty, "shards": len(headers),
    }, merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["humaneval", "mbpp"])
    ap.add_argument("--glob", required=True, help="glob over the 8 shard preds files")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True, help="merged preds jsonl (paired_bootstrap input)")
    ap.add_argument("--result", required=True, help="JSON summary appended with the other bench")
    args = ap.parse_args()

    paths = glob.glob(args.glob)
    if not paths:
        raise SystemExit(f"no shard files match {args.glob}")
    if args.bench == "humaneval":
        total, excluded = 164, _exclude_he()
    else:
        total, excluded = 427, _exclude_mbpp()
    summary, merged = merge(paths, total, args.n, excluded, args.bench)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_header": 1, **summary}, ensure_ascii=False) + "\n")
        for r in merged:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"{args.bench} FULL {summary['full_pass']}/{summary['full_denom']} "
          f"= {summary['full_rate'] * 100:.2f}%   "
          f"CLEAN {summary['clean_pass']}/{summary['clean_denom']} "
          f"= {summary['clean_rate'] * 100:.2f}%")
    # Merge with the other benchmark's summary into one result JSON.
    existing = {}
    if os.path.exists(args.result):
        existing = json.load(open(args.result, encoding="utf-8"))
    existing[args.bench] = summary
    with open(args.result, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2, ensure_ascii=False)




def _selftest():
    """Synthetic 8-shard preds over the real task-id spaces: exact cover + CLEAN math."""
    import tempfile
    d = tempfile.mkdtemp()

    # Self-contained id spaces: HE string ids; MBPP NON-CONTIGUOUS integer ids mirroring the
    # real sanitized set (427 ids spanning 2..809 with gaps). This is the point -- sharding is
    # by file POSITION and CLEAN exclusion by task_id VALUE, so the test must not assume
    # range(427). excluded sets are synthetic subsets here so the test needs no untracked data.
    he_ids = [f"HumanEval/{i}" for i in range(164)]
    he_excl = {he_ids[20 * i] for i in range(8)}            # exactly 8 -> 156 clean
    mb_ids = [2 + 2 * i for i in range(427)]             # 427 non-contiguous ints: 2..854 step2
    mb_excl = {mb_ids[4 * i] for i in range(89)}           # exactly 89 -> 338 clean
    assert len(he_excl) == 8 and len(mb_excl) == 89

    def write_shards(prefix, tids, n, excluded):
        paths = []
        for si in range(8):
            p = os.path.join(d, f"{prefix}_{si}.jsonl")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"_header": 1, "shard_i": si, "shard_n": 8}) + "\n")
                for idx in range(si, len(tids), 8):       # POSITION modulo, the shard rule
                    tid = tids[idx]
                    for s in range(n):
                        fh.write(json.dumps({"task_id": tid, "sample_idx": s,
                                             "ok": tid not in excluded,   # pass iff CLEAN
                                             "empty": False}) + "\n")
            paths.append(p)
        return paths

    he_paths = write_shards("he", he_ids, 10, he_excl)
    s_he, _ = merge(he_paths, 164, 10, he_excl, "humaneval")
    assert (s_he["full_denom"], s_he["clean_denom"]) == (1640, 1560), s_he
    assert (s_he["full_pass"], s_he["clean_pass"]) == (1560, 1560), s_he

    mb_paths = write_shards("mb", mb_ids, 10, mb_excl)
    s_mb, _ = merge(mb_paths, 427, 10, mb_excl, "mbpp")
    assert (s_mb["full_denom"], s_mb["clean_denom"]) == (4270, 3380), s_mb
    assert (s_mb["full_pass"], s_mb["clean_pass"]) == (3380, 3380), s_mb

    def must_refuse(label, *a):
        try:
            merge(*a)
        except SystemExit:
            return
        raise AssertionError(label)

    # missing shard (gap), a duplicated (task,sample), and a short sample count all refuse
    must_refuse("missing shard gap", he_paths[:7], 164, 10, he_excl, "humaneval")
    dup = os.path.join(d, "he_dup.jsonl")
    with open(dup, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_header": 1, "shard_i": 0, "shard_n": 8}) + "\n")
        fh.write(json.dumps({"task_id": "HumanEval/0", "sample_idx": 0,
                             "ok": True, "empty": False}) + "\n")
    must_refuse("duplicate sample", he_paths + [dup], 164, 10, he_excl, "humaneval")
    short = os.path.join(d, "he_short.jsonl")
    with open(short, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_header": 1}) + "\n")
        for idx in range(164):
            for s in range(9 if idx == 0 else 10):
                fh.write(json.dumps({"task_id": f"HumanEval/{idx}", "sample_idx": s,
                                     "ok": False, "empty": False}) + "\n")
    must_refuse("short sample count", [short], 164, 10, he_excl, "humaneval")
    print("e0_merge_score selftest OK: 164/427 x8 exact cover; CLEAN 156/338; "
          "gap/duplicate/short all refuse")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()

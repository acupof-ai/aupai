"""Read-only replay of the RL pool's global dedup, plus a type-coverage count.

WHY THIS EXISTS: the coverage table (runs/code_type_coverage_0925.jsonl) reports the RL pool
as `rl_pool_deduped`, reproduced from the pool's own dedup rule. That number is only
checkable if the rule and the resulting counts are both in the tree -- genB review c61ac333
found the stats file and this replay were pod-only, so the column's provenance did not
resolve.

WHY IT REPLAYS INSTEAD OF CALLING THE BUILDER: scripts/rl_code_pool.py CONSUMES the files it
reads (pass 2 rewrites each file in place, keeping only the winners). Running it to check a
number would mutate the production pool. This reads the files and filters in memory.

The rule is the builder's own: key = sha1(norm_text(prompt)) with norm_text =
lower(whitespace-collapsed(strip)); survivor = the row with the max test count. Call-style
rows outrank stdin rows on a tie only through the fixture order below, which matches the
builder's own file order.

Usage:
    python3 scripts/rl_pool_dedup_replay.py --pool DIR --stats FILE [--coverage] [--selftest]

Exit code is non-zero if the replay does not reproduce the stats file's survivor count and
per-file counts -- a mismatch means the pool moved, not that the check is advisory.
"""
# restartable: read-only single pass over already-materialized pool files; no cumulative state,
# so an interrupt loses only the rows not yet held. The builder it mirrors is NOT restartable
# (it rewrites its inputs), which is the reason this replay exists.
import argparse
import collections
import hashlib
import json
import os
import re
import sys

WS = re.compile(r"\s+")
# The builder's own file order; a tie on test count keeps the earlier file's row.
POOL_FILES = ["rl_code_taco.jsonl", "rl_code_apps.jsonl",
              "rl_code_apps_stdin.jsonl", "rl_code_taco_stdin.jsonl"]


def norm_text(s):
    return WS.sub(" ", (s or "").strip()).lower()


def qhash(s):
    return hashlib.sha1(norm_text(s).encode("utf-8")).hexdigest()


def n_tests(row):
    t = row.get("tests")
    if isinstance(t, list):
        return len(t)
    if isinstance(t, str):
        return t.count("def test") or sum(1 for l in t.split("\n") if l.strip())
    c = row.get("cases")
    if isinstance(c, list):
        return len(c)
    return 0


def replay(pool_dir, files=POOL_FILES):
    """(survivors, per_file_counts). Survivors are the original dicts, in pool order."""
    winner = {}
    every = []
    for fi, fn in enumerate(files):
        path = os.path.join(pool_dir, fn)
        if not os.path.exists(path):
            continue
        for li, line in enumerate(open(path, encoding="utf-8", errors="replace")):
            if not line.strip():
                continue
            row = json.loads(line)
            key = qhash(row.get("prompt") or "")
            cur = winner.get(key)
            if cur is None or n_tests(row) > cur[2]:
                winner[key] = (fi, li, n_tests(row))
            every.append((fi, li, fn, row))
    keep = {(fi, li) for fi, li, _ in winner.values()}
    survivors = [r for fi, li, fn, r in every if (fi, li) in keep]
    # Count over EVERY input file, not over the survivors: a file with zero survivors is a
    # real result (apps.jsonl is empty) and a Counter omits it, which made a correct replay
    # report MISMATCH against a stats file that names the file with 0.
    per = {fn: 0 for fn in files}
    for fi, li, fn, r in every:
        if (fi, li) in keep:
            per[fn] += 1
    return survivors, per


def _selftest():
    """The rule must be exercised on a fixture whose answer is known by construction, or a
    replay that always returns everything would pass against a pool with no duplicates."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        rows = [
            {"prompt": "A  b", "tests": ["t1"]},                    # normalises to "a b"
            {"prompt": "a\n\nb", "tests": ["t1", "t2", "t3"]},       # same key, more tests -> wins
            {"prompt": "a   B", "tests": []},                        # same key, fewer tests -> loses
            {"prompt": "unique", "tests": ["t1"]},
        ]
        p = os.path.join(d, "rl_code_taco.jsonl")
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        surv, per = replay(d, ["rl_code_taco.jsonl"])
        assert len(surv) == 2, f"expected 2 survivors, got {len(surv)}"
        assert surv[0]["tests"] == ["t1", "t2", "t3"], "survivor must be the max-test row"
        assert per == {"rl_code_taco.jsonl": 2}, per
        # a file with no survivors must still appear, at 0
        surv2, per2 = replay(d, ["rl_code_taco.jsonl", "rl_code_absent.jsonl"])
        assert per2["rl_code_absent.jsonl"] == 0, per2
        # case and whitespace are the normalisation: a case-sensitive key would keep 4.
        assert qhash("A  b") == qhash("a\n\nb") == qhash("a   B")
        print("selftest OK: normalisation collapses whitespace+case; max-test row survives")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="/data00/home/chenkailun.c/sfta/rl_pool")
    ap.add_argument("--stats", default="")
    ap.add_argument("--classifier", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "problem_types.py"))
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    survivors, per = replay(a.pool)
    n = len(survivors)
    if a.stats and os.path.exists(a.stats):
        st = json.load(open(a.stats, encoding="utf-8"))
        ok = (n == st["unique_total"] and per == st["files"])
        print(f"replay survivors {n} vs stats unique_total {st['unique_total']}")
        print(f"replay per-file {per}")
        print(f"stats  per-file {st['files']}")
        if not ok:
            sys.exit("MISMATCH: the pool changed since the stats file was written; "
                     "re-read both before quoting either number")
        print("MATCH")
    else:
        print(f"survivors {n}; per-file {per}; no stats file given to check against")
    if a.coverage:
        import importlib.util
        spec = importlib.util.spec_from_file_location("pt", a.classifier)
        pt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pt)
        c = collections.Counter()
        for r in survivors:
            text = ((r.get("prompt") or "") + "\n" + (r.get("impl") or "") + "\n"
                    + str(r.get("tests") or "")[:2000])
            c[pt.classify(text)[0]] += 1
        print(json.dumps({"name": "rl_pool_deduped", "mode": "pairs", "sampled": n,
                          "stride": 1, "unclassified": c.get("unclassified", 0),
                          "counts": dict(c)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

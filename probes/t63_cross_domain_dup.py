"""Cross-domain exact duplication: the one dedup pass that never ran.

build_corpus._parallel_exact_pass dedupes globally across ONE domain's shards, and each 30B
domain was built into its own directory with a fresh `seen_exact`. dedup_corpus.py is the only
tool that compares two domains, and it has never run (data/dedup/ does not exist) and has no
consumer even if it had. So a document present in two domains is trained on twice with nothing
in the pipeline able to notice.

This measures whether that gap is actually populated. Key is build_corpus.exact_key's normaliser
-- strip [\\s\\W_]+, sha1, first 12 bytes -- so a hit here is a hit that pass would have caught
had it ever spanned domains.

Exact only. A paraphrase or a reformatted copy is counted unique; near-dedup has never run
either, and this probe says nothing about it.

    python3 t63_cross_domain_dup.py <corpus_root> <dom1,dom2,...> [rows_per_domain]
    python3 t63_cross_domain_dup.py --selftest
"""
import glob
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import defaultdict

_NORM = re.compile(r"[\s\W_]+")


def key(text):
    return hashlib.sha1(_NORM.sub("", text).encode()).digest()[:12]


def scan(root, domains, cap=10**9):
    owner, hits, counts = {}, defaultdict(int), {}
    for d in domains:
        shards = sorted(p for p in glob.glob(os.path.join(root, d, "*.jsonl"))
                        if "build_corpus_stats" not in os.path.basename(p))
        n = 0
        for p in shards:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    t = o.get("text") or o.get("content") or ""
                    if not t:
                        continue
                    k = key(t)
                    if k in owner and owner[k] != d:
                        hits[(owner[k], d)] += 1
                    else:
                        owner.setdefault(k, d)
                    n += 1
                    if n >= cap:
                        break
            if n >= cap:
                break
        counts[d] = n
    return {"rows_per_domain": counts, "distinct_keys": len(owner),
            "cross_domain_dups": {f"{a}|{b}": c for (a, b), c in
                                  sorted(hits.items(), key=lambda kv: -kv[1])},
            "total_cross": sum(hits.values())}


def selftest():
    """A zero from this probe means nothing until it is shown to find a planted positive."""
    d = tempfile.mkdtemp()
    shared = {"text": "this exact document lives in two domains"}
    for dom, rows in (("a", [{"text": f"a-only {i}"} for i in range(50)] + [shared]),
                      ("b", [{"text": f"b-only {i}"} for i in range(50)] + [shared, shared])):
        os.makedirs(os.path.join(d, dom))
        with open(os.path.join(d, dom, f"{dom}_000.jsonl"), "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows))
    got = scan(d, ["a", "b"])
    # b's two copies of the shared doc: one collides cross-domain, the second collides with
    # a's key too (owner stays "a"), so both count. Within-domain repeats are NOT counted.
    assert got["total_cross"] == 2, got
    assert got["cross_domain_dups"] == {"a|b": 2}, got
    assert got["rows_per_domain"] == {"a": 51, "b": 52}, got
    print("selftest OK: planted cross-domain duplicate found (a|b = 2)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        rows = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
        print(json.dumps(scan(sys.argv[1], sys.argv[2].split(","), rows)))

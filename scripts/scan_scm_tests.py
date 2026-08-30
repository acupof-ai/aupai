#!/usr/bin/env python3
"""fb's (b): mining feasibility for the 2.0B code-with-tests cell.

The 2.0B cell needs ~4M samples (500 tok/sample); even lambda=1 is 4M generation
calls - outside budget. fb's hypothesis: MINE it from RedPajama github, which
naturally contains repos with paired implementation + test files. This scan
answers how much executable-verification material actually exists.

Measure (meta-only, fast - no tokenization, runs on the sampled file):
  1. per doc: is it a TEST file? (path heuristic)
  2. group by repo_name -> repos with impl-only, test-only, BOTH
  3. headline: docs in a repo that has BOTH impl+test = the minable "code +
     executable verification" candidate supply, token/byte-weighted.
Also reports the gross test-file fraction and per-language split.

Usage (on pod): python3 scripts/scan_scm_tests.py data/raw/code_supply/github_00.sampled.jsonl
"""
import argparse
import json

import re

from collections import defaultdict

_TEST_PATH = re.compile(
    r"(/[Tt]est[s]?/|/(t|T)ests?/|"
    r"(^|[/._-])(test|tests|spec)[/._-]|"
    r"test[a-zA-Z0-9]*\.|\.test\.|_test\.)", re.I
)


def is_test(path):
    return bool(_TEST_PATH.search(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--max_docs", type=int, default=0, help="cap for a quick run (0 = all)")
    a = ap.parse_args()

    repos = defaultdict(lambda: {"test": 0, "impl": 0})  # repo -> {test, impl} doc counts
    test_docs = impl_docs = total = 0
    lang_t = defaultdict(int)
    lang_i = defaultdict(int)
    tbytes = ibytes = 0

    with open(a.path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            m = d.get("meta") or {}
            size = m.get("size") or 0
            try:
                size = int(size)
            except (TypeError, ValueError):
                size = 0
            path = m.get("path") or ""
            repo = m.get("repo_name") or ""
            lang = m.get("language") or ""
            while isinstance(lang, (list, tuple)):  # meta is heterogeneous: str / list / dict
                lang = lang[0] if lang else ""
            if isinstance(lang, dict):
                lang = str(lang.get("name") or lang) or ""
            total += 1
            if is_test(path):
                test_docs += 1
                repos[repo]["test"] += 1
                lang_t[lang] += 1
                tbytes += size
            else:
                impl_docs += 1
                repos[repo]["impl"] += 1
                lang_i[lang] += 1
                ibytes += size
            if a.max_docs and total >= a.max_docs:
                break

    both = {r: c for r, c in repos.items() if c["test"] > 0 and c["impl"] > 0}
    test_only = {r for r, c in repos.items() if c["test"] > 0 and c["impl"] == 0}
    impl_only = sum(1 for r, c in repos.items() if c["impl"] > 0 and c["test"] == 0)

    print(f"docs: {total}  (test-file {test_docs} = {test_docs/max(1,total):.2%}, impl {impl_docs})")
    print(f"bytes: test {tbytes/1e9:.1f}G, impl {ibytes/1e9:.1f}G")
    print(f"repos: {len(repos)} total; {len(both)} with BOTH impl+test "
          f"({len(both)/max(1,len(repos)):.1%}); impl-only {impl_only}; test-only {len(test_only)}")
    # headline: docs that live in a repo that has both impl + test = minable exec-verification carriers
    both_docs = sum(c["test"] + c["impl"] for c in both.values())
    print(f"docs inside both-impl+test repos: {both_docs} = {both_docs/max(1,total):.2%} of all docs")
    both_test_docs = sum(c["test"] for c in both.values())
    print(f"  of which test-file docs: {both_test_docs} = {both_test_docs/max(1,total):.2%} of all docs (the executable check)")
    print("language split (top, test/impl):")
    for lang, n in sorted(lang_t.items(), key=lambda x: -x[1])[:6]:
        print(f"  {lang or '?':12s} test {n:6d} impl {lang_i[lang]:8d}")


if __name__ == "__main__":
    main()

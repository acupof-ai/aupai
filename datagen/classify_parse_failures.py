#!/usr/bin/env python3
"""Clean WHY-histogram of code_rp1t rows failing ast.parse (fb P0 2026-09-01).
First pass EXCLUDES rows that are not actually Python -- code_rp1t is unlabelled
(source = filtered_<hash>, 0 .py paths over 3.7M rows), and a crude import/def
heuristic mislabels Java/Ruby/C. Only the genuinely-Python-ish failures are bucketed:
truncated_mid_construct / not_actually_python / prose_code_mixed / template_boilerplate /
py2_vs_py3 / genuinely_broken_source. The dominant bucket is the repair decision."""
import ast
import glob
import json
import random
from collections import Counter

random.seed(17)


def coerce_likely_nonpython(t):
    """Row is probably NOT python at all -> bucket 'not_actually_python'. Strong signals
    (a real parser for every language would be better; these catch the cheap mislabels)."""
    head = t[:4000]
    n_braces = head.count("{")
    n_semi = head.count(";")
    nl = [ln.strip() for ln in head.split("\n") if ln.strip()]
    first = nl[0].strip() if nl else ""
    # C/C++: preprocessor, or .h-ish
    if "#include" in head or "#ifndef" in head or "#define " in head:
        return True
    # Java: package X;  +  class ... {

    if first.startswith("package ") and ";" in first:
        return True
    # Ruby: require 'x'  /  class X  ...end  (no colon after def)
    if first.startswith("require") and "'" in first:
        return True
    # JS/TS/others: heavy braces+semicolons with C-ish tokens, no python def-colon
    if n_braces >= 3 and n_semi >= 3:
        pyish = (("def " in head and ":" in head) or "print(" in head or "__init__" in head or "self." in head or "import math" in head or "from " in head)
        if not pyish:
            return True
    # config / shell / xml front matter
    if "<?xml" in t.lower() or t.lower().startswith("<html") or "uuid" in t[:100].lower() and "{" in t[:60]:
        return True
    return False


def classify(t):
    if coerce_likely_nonpython(t):
        return "not_actually_python"
    lines = [ln for ln in t.split("\n") if ln.strip()]
    last = lines[-1].strip() if lines else ""
    if last == ":" or any(last.endswith(k.rstrip()) for k in ("def ", "class ", "if ", "elif", "for ", "while ", "else", "try", "except", "with ", "lambda")):
        return "truncated_mid_construct"
    if t.count("{{") + t.count("{%") + t.count("$(") > 0 or t.count("...") > 5:
        return "template_boilerplate"
    # py2-vs-py3: print stmt, leading-zero ints, xrange, no-parenthesized print
    if (">> " in t and "print " in t) or t.count("xrange(") > 0 or "\tprint " in "\n" + t or \
       any(l.strip().startswith("print ") and not l.strip().startswith("print(") for l in lines):
        return "py2_vs_py3"
    prose = sum(1 for l in lines if len(l) > 70 and l.count(" ") > 10 and not any(
        k in l for k in ("def ", "class ", "import ", "return ", "=", "(", ")", "{", "}", ":", ",")))
    code = sum(1 for l in lines if any(k in l for k in ("def ", "import ", "class ", "print(", "=")))
    if prose and code and prose >= len(lines) * 0.15:
        return "prose_code_mixed"
    return "genuinely_broken_source"


def main():
    shards = sorted(glob.glob("/work/aupai/data/corpus/code_rp1t/*.jsonl"))
    fails = []
    # collect rows that (content-likely-python) and fail ast.parse
    for shard in shards:
        for line in open(shard, encoding="utf-8"):
            d = json.loads(line)
            t = d.get("content") or ""
            if not t:
                continue
            h = t[:4000]
            likely_py = h.count("\ndef ") + h.count("    def ") >= 1 or h.count("import ") + h.count("from ") >= 2
            if not likely_py:
                continue
            try:
                ast.parse(t)
            except SyntaxError:
                fails.append(t)
            if len(fails) >= 600:
                break
        if len(fails) >= 600:
            break
    sample = random.sample(fails, min(200, len(fails)))
    hist = Counter(classify(t) for t in sample)
    print(json.dumps({
        "n_failing_py_labelled": len(sample),
        "histogram": dict(hist),
        "config": {"source": "code_rp1t rows, content-likely-python, failing ast.parse",
                   "note": "not_actually_python = excluded-or-ignored in a real filter; the rest are the genuine failures",
                   "n": len(sample), "classifier": "exclude-nonpython first, then truncated/template/py2/prose/broken"},
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

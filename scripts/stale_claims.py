#!/usr/bin/env python3
"""Comments that assert something about code younger than the comment.

Every stale-claim defect found on 2026-09-01 was TRUE WHEN WRITTEN and decayed because the
code moved and the comment did not have to move with it:

  eval/domain_loss.py:47  "the head, matching train.py's val split"  -- a shuffle was added
                          to train.py between the read and the slice, after the comment
  scripts/loader.py:84    docstring asserts `vocab` where the code asserts `vocab_real`
                          -- vocab_real split from vocab after the docstring

So the population worth reading is not "comments that look wrong" -- nobody can grep for
that -- it is **claims older than the code they describe**, which is a git question.

The check: for a comment naming an identifier, compare the comment's author-time against the
author-time of the lines that define or assign that identifier. Comment older => candidate.

FALSE POSITIVES ARE THE POINT, not a defect. This produces a short list to read, which is
what nobody has had; it cannot know whether a claim survived the change. It is a reading
queue, never a verdict -- the same status doc_numbers_check earned after its sweep came back
at ~40% false positives and stayed useful (tilerl/03, 2026-09-01).

    python3 stale_claims.py <file> [<file>...]     # or no args: every tracked .py
    python3 stale_claims.py --selftest
"""
import re
import subprocess
import sys

# A claim, not a note: asserts a relationship that some other code has to keep true.
CLAIM = re.compile(r"\b(match(es|ing)?|same as|equals?|is the|are the|must|always|never|"
                   r"guaranteed|by construction|identical|corresponds)\b", re.I)
IDENT = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(|\.|=[^=])")
NOISE = {"self", "cls", "print", "len", "int", "str", "float", "dict", "list", "set",
         "os", "sys", "re", "json", "torch", "np", "f", "the", "is", "it"}


def blame(path):
    """[(author_time, line_text)] -- porcelain so the timestamp is per line, not per file."""
    out = subprocess.run(["git", "blame", "--line-porcelain", "--", path],
                         capture_output=True, text=True).stdout
    times = [int(t) for t in re.findall(r"^author-time (\d+)", out, re.M)]
    lines = re.findall(r"^\t(.*)$", out, re.M)
    # strict=: a porcelain block with no \t line would silently shorten the pairing
    return list(zip(times, lines, strict=True))


def scan(path):
    rows = blame(path)
    if not rows:
        return []
    # newest author-time per identifier, taken from the lines that DEFINE or ASSIGN it
    defined = {}
    for t, text in rows:
        for m in re.finditer(r"^\s*(?:def|class)\s+(\w+)|^\s*(\w+)\s*=[^=]", text):
            name = m.group(1) or m.group(2)
            if name:
                defined[name] = max(defined.get(name, 0), t)
    hits = []
    for i, (t, text) in enumerate(rows, 1):
        s = text.strip()
        if not (s.startswith("#") or s.startswith('"""') or s.startswith("'''")):
            continue
        if not CLAIM.search(s):
            continue
        for name in {m.group(1) for m in IDENT.finditer(s)} - NOISE:
            ct = defined.get(name)
            if ct and ct > t:
                hits.append((i, name, t, ct, s[:88]))
                break
    return hits


def selftest():
    """Known answer on this repo's own archaeology: domain_loss.py's corrected comment.

    The check must FIND a real case, not merely run. A scanner whose success condition is an
    empty list reports success when its regex is broken -- the highest-risk shape there is,
    and the reason chatml_in_corpus.py asserts it finds ChatML in a row that has it.
    """
    rows = [(100, "# the head, matching build_val's split"),
            (200, "def build_val():"),
            (100, "# unrelated note about performance"),
            (50, "x = 1")]
    defined = {}
    for t, text in rows:
        for m in re.finditer(r"^\s*(?:def|class)\s+(\w+)|^\s*(\w+)\s*=[^=]", text):
            n = m.group(1) or m.group(2)
            if n:
                defined[n] = max(defined.get(n, 0), t)
    assert defined == {"build_val": 200, "x": 50}, defined
    claim = [t for t, s in rows if s.strip().startswith("#") and CLAIM.search(s)]
    assert claim == [100], claim
    assert defined["build_val"] > claim[0], "comment must be older than the code it names"
    # and the negative: a note with no claim verb is not a candidate
    assert not CLAIM.search("# unrelated note about performance")
    print("selftest OK: a claim older than the code it names is found; a plain note is not")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    files = sys.argv[1:] or subprocess.run(
        ["git", "ls-files", "*.py"], capture_output=True, text=True).stdout.split()
    total = 0
    for f in files:
        for ln, name, ct, dt, s in scan(f):
            print(f"{f}:{ln}  claim names `{name}` (code {dt - ct}s newer)\n    {s}")
            total += 1
    print(f"\n{total} candidate(s) across {len(files)} file(s) -- a reading queue, not a verdict")

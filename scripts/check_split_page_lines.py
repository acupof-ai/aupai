#!/usr/bin/env python3
"""Verify every line citation in docs/standards/model_module_split.md against train.py.

Checks CONTENT, not offsets. The page's numbers were correct when written, then
broken by a merge of main that inserted 29 lines into train.py; the repair then
broke a number that was already right, because "every symbol is low by 1" was
induced from ten samples and the eleventh had a different offset (insertions
before `Block` total 0 lines). An offset is not a citation and neither is a
number -- only the line's content is.

Also verifies the page's stamped train.py blob sha. A mismatch means the page's
numbers describe a train.py nobody has any more: re-run this script rather than
trusting them.

    python3 scripts/check_split_page_lines.py          # exit 1 on any mismatch
    python3 scripts/check_split_page_lines.py --stamp  # print the current sha
"""
import re
import subprocess
import sys
from pathlib import Path

PAGE = Path("docs/standards/model_module_split.md")
SRC = Path("train.py")


def blob_sha(path):
    return subprocess.run(["git", "hash-object", str(path)],
                          capture_output=True, text=True, check=True).stdout.strip()


def citations(page):
    """(symbol, line, label) for every citation form the page uses."""
    for sym, a, b in re.findall(r"\| `(\w+)` \| (\d+)-(\d+) \|", page):
        yield sym, int(a), f"{a}-{b}"
    for sym, n in re.findall(r"`(\w+)`（(\d+)）", page):
        yield sym, int(n), n
    for sym, n in re.findall(r"`(\w+)` (\d{3})\b", page):
        yield sym, int(n), n


def quoted(page):
    """(line, snippet) for every `train.py:N` whose content the page quotes.

    The form the page uses in §2 -- `train.py:314` `def forward(self, x, cu=None)` --
    is the strongest citation on the page and the FIRST VERSION OF THIS SCRIPT DID
    NOT CHECK IT. It reported "25 citations verified" while green on the exact error
    fb had caught (313 for 314), because every regex above wants a bare symbol name
    and this form leads with the path. A checker that passes on the defect it was
    written for is worse than no checker: it converts an unverified number into a
    verified-looking one.
    """
    # `train.py:N` followed (same line or next, across the newline) by a code quote
    for n, snip in re.findall(r"`train\.py:(\d+)`[^`]{0,40}`([^`]+)`", page, re.S):
        yield int(n), snip.strip()


def path_citations(page):
    """Every `train.py:N` or `train.py:N-M` on the page, quoted or not.

    quoted() only sees the ones that carry a code quote. `train.py:701-706` carries
    none -- it heads a fenced block instead -- so it was unchecked, and it is the
    number I broke by inducing "every symbol is low by 1" and applying +1 to it.
    The two forms miss opposite things, so both run: this one proves every path
    citation is ACCOUNTED FOR, quoted() proves the quoted ones are RIGHT.
    """
    for n, m in re.findall(r"`train\.py:(\d+)(?:-(\d+))?`", page):
        yield int(n), int(m) if m else None


# Every `train.py:N` citation must be verifiable one of two ways: by an adjacent
# code quote, or by appearing here with the symbol whose definition spans it. A
# range that heads a fenced block cannot be quote-checked, so it is named.
RANGE_OWNERS = {(701, 706): "Block"}

# §2's KDA-state claim is the page's invalidation condition (§7): if state ever
# crosses a block boundary, the draft-head conclusion changes. Its evidence is two
# quoted lines, and a quote can be dropped by an edit without breaking anything
# visible -- rewrite `train.py:314` `def forward...` as prose and coverage falls
# from 2 to 1 while the script still says OK. Counting is not enough; the two
# loads must be named. Raise this floor when §2 gains evidence, never lower it to
# make a run pass.
REQUIRED_QUOTES = 2


def main():
    page = PAGE.read_text()
    if "--stamp" in sys.argv:
        print(blob_sha(SRC))
        return 0

    bad = []

    stamped = re.search(r"train\.py blob `([0-9a-f]{40})`", page)
    actual = blob_sha(SRC)
    if not stamped:
        bad.append("page carries no train.py blob sha -- add one so a reader can tell drift from error")
    elif stamped.group(1) != actual:
        bad.append(f"stamp {stamped.group(1)[:12]} != actual {actual[:12]}: "
                   f"train.py changed since the numbers were verified")

    lines = SRC.read_text().split("\n")
    n_sym = 0
    for sym, line, label in citations(page):
        n_sym += 1
        if not 1 <= line <= len(lines):
            bad.append(f"{sym} {label}: out of range (train.py has {len(lines)} lines)")
            continue
        first = lines[line - 1]
        # a definition, or a module-level assignment for constants like SOFTCAP
        if not re.match(rf"^(class|def) {sym}\b", first) and not first.startswith(sym):
            bad.append(f"{sym} {label}: line {line} is {first[:52]!r}")

    quotes = 0
    for line, snip in quoted(page):
        quotes += 1
        if not 1 <= line <= len(lines):
            bad.append(f"quote at :{line}: out of range")
            continue
        if snip not in lines[line - 1]:
            bad.append(f":{line} quoted as {snip[:40]!r} but the line is "
                       f"{lines[line - 1].strip()[:40]!r}")
    if quotes < REQUIRED_QUOTES:
        bad.append(f"only {quotes} of {REQUIRED_QUOTES} required quoted citations "
                   f"found -- §2's evidence was rewritten into prose, which drops "
                   f"coverage without breaking anything visible")

    # Every path citation must be accounted for: quoted, or a named range.
    quoted_lines = {line for line, _ in quoted(page)}
    ranges = 0
    for n, m in path_citations(page):
        if n in quoted_lines:
            continue
        owner = RANGE_OWNERS.get((n, m))
        if owner is None:
            bad.append(f"`train.py:{n}{'-' + str(m) if m else ''}` is neither quoted "
                       f"nor in RANGE_OWNERS -- unverifiable, add a quote or name it")
            continue
        ranges += 1
        if not re.match(rf"^ *(class|def) {owner}\b", lines[n - 1]):
            # the range names a member; find its owner's span
            start = next((i for i in range(n - 1, 0, -1)
                          if re.match(rf"^(class|def) {owner}\b", lines[i - 1])), None)
            end = next((i for i in range(n, len(lines) + 1)
                        if re.match(r"^(class|def) \w", lines[i - 1])), len(lines))
            if start is None or not start <= n <= end:
                bad.append(f"`train.py:{n}-{m}` claimed inside {owner}, but {owner} "
                           f"spans {start}-{end}")

    if bad:
        print(f"FAIL {len(bad)} problem(s) across {n_sym} symbol + {quotes} quoted "
              f"+ {ranges} range citations (+stamp):", file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        return 1
    print(f"OK {n_sym} symbol + {quotes} quoted + {ranges} range citations verified "
          f"by content; train.py blob {actual[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

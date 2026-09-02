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


def source_lines(page):
    """The train.py the page's numbers describe -- the STAMPED BLOB, not the working tree.

    Once the split lands in the worktree, train.py no longer contains the moved code and every
    citation reads as broken. That is a true statement about the working tree and a useless one
    about the page: the numbers were always about the pre-split file, which is exactly what the
    stamp identifies. So read the blob by sha and keep the working tree out of it.
    """
    m = re.search(r"train\.py blob `([0-9a-f]{40})`", page)
    if not m:
        return None, None
    sha = m.group(1)
    r = subprocess.run(["git", "cat-file", "-p", sha], capture_output=True, text=True)
    if r.returncode != 0:
        return sha, None
    return sha, r.stdout.split("\n")


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

    sha, lines = source_lines(page)
    actual = blob_sha(SRC) if SRC.exists() else None
    if sha is None:
        bad.append("page carries no train.py blob sha -- add one so a reader can tell drift "
                   "from error")
        lines = SRC.read_text().split("\n") if SRC.exists() else []
    elif lines is None:
        bad.append(f"stamped blob {sha[:12]} is not in this repository -- the numbers cite a "
                   f"train.py nobody can read; re-stamp against a blob git holds")
        lines = []
    elif sha != actual:
        # NOT a failure. The split moves the cited code out of train.py by design, so after it
        # lands the working tree and the stamp differ forever. The page documents the pre-split
        # file; the stamp is how it says which one. Report the divergence, verify against the
        # blob (design page §6 step 6 deletes these numbers when the split merges).
        print(f"note: working train.py is {actual[:12]}, page cites {sha[:12]} -- verifying "
              f"against the stamped blob, which is the file the numbers describe")
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
            continue
        if "-" not in label:
            continue
        # THE END OF THE RANGE, which this script did not check and should have. The page gave
        # HybridLM as 732-875 and GatedMLA as 358-414; they end at 862 and 408. Lines 865-873
        # are the Muon section header plus POLAR_EXPRESS and 411-412 the FP8 header plus
        # _FP8_MAX_E4M3 -- all belonging to code that STAYS. Both ranges passed here because
        # 732 and 358 do hold the right definition. A range has two ends; only one was verified,
        # and the split then moved two constants it must not touch (NameError: POLAR_EXPRESS,
        # F821 _FP8_MAX_E4M3). Same family as the start-line defect it was written to prevent.
        end = int(label.split("-")[1])
        if end > len(lines):
            bad.append(f"{sym} {label}: end {end} past EOF ({len(lines)} lines)")
            continue
        # `_?[A-Z_]{2,}` because the constant this missed is _FP8_MAX_E4M3: the first pattern
        # demanded a leading capital, so GatedMLA 358-414 passed while swallowing it. A
        # module-level constant is the exact thing a too-long range takes, and a leading
        # underscore is the normal spelling for a private one.
        nxt = next((k for k in range(line, len(lines))
                    if re.match(r"^(class|def) \w|^_?[A-Z][A-Z0-9_]+ *=", lines[k])),
                   len(lines))
        if end > nxt:
            bad.append(f"{sym} {label}: {sym} ends before line {nxt}, but the range runs to "
                       f"{end} and so swallows {lines[nxt][:44]!r} -- which belongs to the next "
                       f"symbol and must not move")


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
              f"+ {ranges} range citations (blob {(sha or actual or '?')[:12]}):", file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        return 1
    print(f"OK {n_sym} symbol + {quotes} quoted + {ranges} range citations verified "
          f"by content against train.py blob {(sha or actual)[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

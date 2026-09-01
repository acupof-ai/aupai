#!/usr/bin/env python3
"""Recompute every derived number a doc states, against the doc's own base.

Written after seven instances in one day of a stated rule failing to survive
into the document that states it. The failure has a shape: a summary sentence
restates a conclusion WITHOUT restating its inputs, so it stays literally true
while every number under it moves. Two careful re-reads of the same file missed
four of them; this found them in one pass.

Four checks, all mechanical:

  ms=pct    every "N ms = P%" recomputed against the doc's declared span
  sum       every "A + B = C" and "-A + B = -C" re-added
  speedup   every "N ms = P% = X" re-derived as span/(span-N)
  retired   a per-doc blacklist of phrases a correction has retired

The span comes from the doc's own front matter (`span_ms:`), never from this
file, because a checker carrying its own copy of the base is the same
two-bases error it exists to catch.

FALSE POSITIVE, do not chase: a correction that quotes the phrase it retires
is not a violation. `--retired` hits are reported separately from arithmetic
for that reason, and a line whose quote is inside a sentence naming it as
retired is the expected shape, not a finding. Arithmetic hits are never
false positives.

Exit 1 on any arithmetic mismatch. Retired-phrase hits print and do not fail,
since only a reader can tell a quote from a relapse.

WHAT IT DOES NOT COVER, so nobody reads a clean run as more than it is: every
check needs the number and its inputs stated TOGETHER, joined by an operator.
A percentage in prose whose components sit in a table forty lines away has no
operator between them and is invisible here. That shape has already produced a
real error (a head share typed as 92.9% beside a table summing to 92.5%), and
it was caught by a human reading this tool's output against the table, not by
the tool. A clean run means every number stated with its arithmetic is right.
"""
import argparse
import re
import sys

MS_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*ms\s*=\s*(\d+\.\d+)%")
SPEEDUP = re.compile(r"(\d+(?:\.\d+)?)\s*ms\s*=\s*\d+\.\d+%\s*=\s*(\d\.\d+)[x×]")
SUM = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)(?![\d.])")
NEG_SUM = re.compile(r"[-−](\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)\s*=\s*\*{0,2}[-−](\d+(?:\.\d+)?)")
SPAN_DECL = re.compile(r"^span_ms:\s*(\d+(?:\.\d+)?)\s*(?:#.*)?$", re.M)

PCT_TOL = 0.02   # printed to 2dp, so anything past this is a different base
SUM_TOL = 0.05   # printed to 1dp
X_TOL = 0.003    # printed to 3dp


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def check(path, span, retired):
    text = open(path, encoding="utf-8").read()
    hits, notes = [], []

    if span is not None:
      for m in MS_PCT.finditer(text):
        ms, pct = float(m.group(1)), float(m.group(2))
        want = 100 * ms / span
        if abs(want - pct) > PCT_TOL:
            hits.append((line_of(text, m.start()), m.group(0), f"{want:.2f}% against span {span}"))
      for m in SPEEDUP.finditer(text):
        ms, x = float(m.group(1)), float(m.group(2))
        want = span / (span - ms)
        if abs(want - x) > X_TOL:
            hits.append((line_of(text, m.start()), m.group(0), f"{want:.3f}x"))

    for m in SUM.finditer(text):
        a, b, c = (float(g) for g in m.groups())
        if abs(a + b - c) > SUM_TOL:
            hits.append((line_of(text, m.start()), m.group(0), f"{a + b:.1f}"))

    for m in NEG_SUM.finditer(text):
        a, b, c = (float(g) for g in m.groups())
        if abs(-a + b + c) > SUM_TOL:
            hits.append((line_of(text, m.start()), m.group(0), f"{-a + b:.1f}"))

    for i, line in enumerate(text.split("\n"), 1):
        for phrase in retired:
            if phrase in line:
                notes.append((i, phrase, line.strip()[:70]))

    return hits, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("docs", nargs="*")
    ap.add_argument("--span", type=float,
                    help="ms/step base; default reads `span_ms:` from the doc's front matter")
    ap.add_argument("--retired", default="",
                    help="comma-separated phrases a correction has retired")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    retired = [p for p in (x.strip() for x in a.retired.split(",")) if p]
    failed = False
    for path in a.docs:
        text = open(path, encoding="utf-8").read()
        m = SPAN_DECL.search(text)
        span = a.span or (float(m.group(1)) if m else None)
        needs_span = bool(MS_PCT.search(text) or SPEEDUP.search(text))
        if span is None and needs_span:
            # Not a skip. A doc that states percentages whose base cannot be
            # read is a doc whose percentages cannot be checked, and calling
            # that "clean" is the fail-open direction this script exists to
            # close. A doc stating no percentages needs no base -- that is
            # precision, not an exemption.
            print(f"{path}: FAIL -- states `N ms = P%` but declares no `span_ms:`")
            failed = True
            continue
        hits, notes = check(path, span, retired)
        base = f"span {span} ms" if span else "no span needed: states no ms-percentages"
        print(f"\n=== {path} ({base}) ===")
        for ln, got, want in hits:
            print(f"  L{ln}: {got!r} -- recomputes to {want}")
            failed = True
        for ln, phrase, line in notes:
            print(f"  L{ln}: retired phrase {phrase!r} -- {line}")
            print("        (a correction quoting the phrase it retires is NOT a violation)")
        if not hits:
            print("  arithmetic clean")
    return 1 if failed else 0


def selftest():
    """Known-answer, against text carrying the errors this was written for."""
    span = 1676.63
    good = "the seam is 54.9 ms = 3.27% = 1.034x and 39.4 + 54.9 = 94.3\n"
    hits, _ = check(_tmp(good), span, [])
    assert not hits, f"clean text flagged: {hits}"

    # the real one: 35.0 ms = 2.06% is against the 1702 step, not the span
    hits, _ = check(_tmp("swapping the library recovers 35.0 ms = 2.06%\n"), span, [])
    assert len(hits) == 1 and "2.09%" in hits[0][2], hits
    print("  wrong base caught: 35.0 ms is 2.09% of span, not 2.06%")

    hits, _ = check(_tmp("39.4 + 54.9 = 96.3 ms\n"), span, [])
    assert len(hits) == 1, hits
    print("  bad sum caught")

    hits, _ = check(_tmp("the byte cache is -59.6 + 39.4 = **-20.2 ms\n"), span, [])
    assert not hits, f"correct signed sum flagged: {hits}"
    hits, _ = check(_tmp("the byte cache is -59.6 + 39.4 = **-30.2 ms\n"), span, [])
    assert len(hits) == 1, hits
    print("  signed sum: correct passes, wrong caught")

    hits, _ = check(_tmp("54.9 ms = 3.27% = 1.06x\n"), span, [])
    assert len(hits) == 1 and "1.034" in hits[0][2], hits
    print("  stale speedup caught: 54.9 ms is 1.034x, not 1.06x")

    _, notes = check(_tmp("the ceiling is single-digit percent\n"), span, ["single-digit percent"])
    assert len(notes) == 1, notes
    print("  retired phrase reported")

    # the false-positive shape, reported but not failing
    hits, notes = check(
        _tmp("the lede said \"single-digit percent\" and survived a 36 ms correction unchanged\n"),
        span, ["single-digit percent"])
    assert not hits and len(notes) == 1, (hits, notes)
    print("  correction quoting its own retired phrase: noted, does not fail")

    print("selftest: 7/7")
    return 0


def _tmp(text):
    import tempfile
    f = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8")
    f.write(text)
    f.close()
    return f.name


if __name__ == "__main__":
    sys.exit(main())

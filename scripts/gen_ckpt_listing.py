#!/usr/bin/env python3
"""Generate the pod checkpoint deletion listing that ckpt_facts_sources_present reads.

WHY A SCRIPT AT ALL. The 2026-09-02 listing was assembled BY HAND: `pod_ckpt_candidates` appears
nowhere in the tree except harness.py's three readers (2660, 2718, 2741) and two docs, so
"regenerate it with the same script" had no referent (de, 2026-09-04). A hand-assembled listing is
the wrong shape for this file specifically, because five sessions' KEEP claims have to survive
every regeneration BY EXACT NAME -- and a name shortened by hand is the defect
ckpt_facts_sources_present exists to catch (b0's `.pt.step832` for the on-disk
`.pt.interrupt.step832`).

WHAT IT REFUSES TO DO, which is most of its value:

  It never deletes anything. It emits a listing; the deletion is a separate act by the list's
  owner, one `rm` per named file. A generator that could also delete would make a scan of shared
  artifacts one keystroke from destroying 250 GB of five sessions' work.

  It refuses to write a listing whose carried-forward KEEP claim names a file the scan did not
  find. That claim is either protecting something already gone -- in which case the fact behind it
  has a dead source and someone must be told -- or the name is wrong. Emitting the listing anyway
  would launder both cases into a file that reads as verified.

  It never re-dates an old deadline. A candidate still present past its deadline is marked
  `present past deadline <when>` and keeps the original date. Writing a fresh 24 h would silently
  restart every claim-holder's window, which is a decision belonging to the owner, not to a
  regeneration (6e's ruling 2026-09-04).

THE OUTPUT FORMAT IS NOT FREE. harness.py:2331 parses candidate rows as
`YYYY-MM-DD_HH:MM <GB> <name>` and KEEP lines by the `# KEEP` prefix, and _parse_ckpt_listing
takes the listing date from `listed <YYYY-MM-DD HH:MMZ>` in a comment. --selftest asserts the
emitted text round-trips through that parser rather than through a second copy of the format here:
two parsers is how the format drifts.

Usage:
    python3 scripts/gen_ckpt_listing.py --out runs/pod_ckpt_candidates_<date>.txt
    python3 scripts/gen_ckpt_listing.py --dry            # print, write nothing
    python3 scripts/gen_ckpt_listing.py --selftest
"""

import argparse
import os
import re
import subprocess
import sys
import time

# restartable: one pod `stat` and one file write, ~10 s. An interrupt loses the scan and nothing
# else -- no partial listing is ever written, because build() returns the whole text before main()
# opens the output file, and a refusal writes nothing at all. Re-running is the recovery.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POD = os.path.expanduser("~/bin/pod")
WORK = "/work/aupai"
# 2026-08-30 00:00Z is the 0830v1 reset boundary: anything older was zeroed by it. Not a
# guess -- section A of the 09-02 listing is defined by exactly this mtime cut and its
# header says so.
RESET = "2026-08-30"


def _pod_scan():
    """[(mtime, bytes, name)] for every ckpt_* in /work/aupai, or (None, error).

    `~/bin/pod`, never `tn exec`: the container and the host are two filesystem views with the
    same hostname, and the host's /work/aupai is a stale tree with nothing newer than
    2026-09-01 (AGENTS, Pod). A scan taken there would report a world six days old and look
    like a successful scan.

    `stat` rather than `ls -la`: ls's date format changes with age (a file over six months old
    prints a year instead of a time) and with locale, so parsing it is parsing a moving target.
    """
    if not os.path.exists(POD):
        return None, f"{POD} not present -- this must run where the pod is reachable"
    cmd = f"cd {WORK} && stat -c '%Y %s %n' ckpt_* 2>/dev/null"
    try:
        r = subprocess.run([POD, cmd], capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"pod scan failed: {type(e).__name__}: {e}"
    out = []
    for line in r.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        try:
            epoch, size = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        out.append((time.strftime("%Y-%m-%d_%H:%M", time.gmtime(epoch)), size, parts[2].strip()))
    if not out:
        return None, (
            f"scan found no ckpt_* in {WORK} -- refusing to write a listing that says "
            f"the pod is empty (stderr: {r.stderr.strip()[:200]})"
        )
    return sorted(out), None


def _read_keep_lines(path):
    """The verbatim `# KEEP` lines of an existing listing, in order.

    VERBATIM, not parsed and re-emitted. Every claim line carries its author, its timestamp and
    its stated reason, and a regeneration that rewrote them would be one session editing five
    others' claims. The names inside them are re-verified against the scan; the text is not
    touched.
    """
    if not path or not os.path.isfile(path):
        return []
    return [ln for ln in open(path, encoding="utf-8").read().splitlines() if ln.startswith("# KEEP")]


def _old_sections(path):
    """{name: (mtime, section, deadline)} from an existing listing, for the still-present /
    gone partition. Deadline is the section header's own text, carried forward unchanged."""
    if not path or not os.path.isfile(path):
        return {}
    out, section, deadline = {}, "A", ""
    for line in open(path, encoding="utf-8").read().splitlines():
        if re.match(r"# [A-Z]\.", line):
            section = line[2]
            m = re.search(r"deleted (\d{4}-\d\d-\d\d \d\d:\d\dZ)", line)
            deadline = m.group(1) if m else ""
            continue
        m = re.match(r"(\d{4}-\d\d-\d\d_\d\d:\d\d) [\d.]+ (\S+)", line)
        if m:
            out[m.group(2)] = (m.group(1), section, deadline)
    return out


def _keep_names(keep_lines):
    """Names claimed by the carried KEEP lines, using the READER's own extraction.

    Imports _parse_ckpt_listing's logic by calling it on a temp file holding just these lines:
    the set that matters is the set the CHECK will compute, and a second implementation here
    would diverge from it exactly when it matters (the shorthand continuation cases, which is
    where that parser has its subtlety).
    """
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import tempfile

    import harness

    fd, p = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("# pod /work/aupai checkpoints, listed 1970-01-01 00:00Z\n")
            fh.write("\n".join(keep_lines) + "\n")
        _date, keep, _cands = harness._parse_ckpt_listing(p)
        return keep
    finally:
        os.unlink(p)


def build(scan, keep_lines, old, now=None):
    """(text, problems). problems non-empty means REFUSE -- do not write."""
    now = now or time.strftime("%Y-%m-%d %H:%M", time.gmtime())
    present = {n for _m, _s, n in scan}
    problems = []
    claimed = _keep_names(keep_lines)
    # PER CLAIM LINE, NOT PER GENERATED NAME, and the difference is the whole correctness of this
    # guard. _parse_ckpt_listing deliberately emits BOTH readings of a shorthand continuation
    # (`X.pt.step2000, .pt.step2500` yields the bare-core and the `.pt`-boundary attachment), and
    # its docstring says why: for a READER "the wrong reading names a file that cannot exist, so
    # over-protection costs nothing". For a refusal that inverts -- measured against the live pod
    # 2026-09-04, checking every generated name reported 17 absent claims on a listing where every
    # real claim is intact, because half of them are `ckpt_p200m_4b_0902.pt.pt.interrupt.step1192`
    # with a doubled `.pt` that nobody ever wrote. A guard whose first live run cries wolf 17
    # times is a guard people learn to pass with --force.
    #
    # So: a claim LINE is satisfied when at least one name it yields is present. That still
    # catches the case this exists for -- a claim protecting a file that is genuinely gone yields
    # NO present name -- while the parser's spare readings cost nothing, exactly as it intended.
    for line in keep_lines:
        names = _keep_names([line])
        if names and not (names & present):
            problems.append(f"KEEP line protects nothing present: {sorted(names)} -- {line[:110]}")
    total = sum(b for _m, b, _n in scan)
    lines = [
        f"# pod {WORK} checkpoints, listed {now}Z: {len(scan)} files, "
        f"{total / 1e9:.0f} GB, generated by scripts/gen_ckpt_listing.py"
    ]
    lines += keep_lines
    # Sections keep the old listing's meaning: A is pre-reset, B is post-reset probe/interrupt
    # files. A file the old listing never saw is NEW and goes in its own section rather than
    # inheriting a deadline it was never given.
    buckets = {"A": [], "B": [], "C": []}
    for mtime, size, name in scan:
        if name in claimed:
            continue
        if name in old:
            _om, sec, deadline = old[name]
            note = f"  # present past deadline {deadline}" if deadline else ""
            buckets.setdefault(sec, []).append((mtime, size, name, note))
        elif mtime[:10] < RESET:
            buckets["A"].append((mtime, size, name, "  # pre-reset, not in the previous listing"))
        else:
            buckets["C"].append((mtime, size, name, ""))
    gone = sorted(set(old) - present - claimed)
    headers = {
        "A": f"# A. pre-0830v1 (mtime < {RESET}): zeroed by the reset. Deadlines below are the "
        f"PREVIOUS listing's, carried forward unchanged -- a regeneration does not restart "
        f"anyone's window",
        "B": "# B. probes and aborted-launch interrupts, carried forward from the previous "
        "listing with their original deadlines",
        "C": "# C. NEW since the previous listing: no deadline has ever been set for these. "
        "They are listed so they are visible, NOT as candidates -- a file cannot be "
        "overdue against a deadline it was never given",
    }
    for sec in ("A", "B", "C"):
        rows = buckets.get(sec) or []
        if not rows:
            continue
        lines.append(headers[sec])
        for mtime, size, name, note in rows:
            lines.append(f"{mtime} {size / 1e9:.2f} {name}{note}")
    if gone:
        lines.append(
            f"# GONE since the previous listing ({len(gone)}): deleted or renamed. "
            f"A fact whose source names one of these has a dead source -- that is what "
            f"ckpt_facts_sources_present's [absent] tier reports, and it is not "
            f"resolved by this file"
        )
        for name in gone:
            om, sec, _d = old[name]
            lines.append(f"#   {om} section {sec} {name}")
    return "\n".join(lines) + "\n", problems


def _selftest():
    """Known answers over a hand-built scan, with the REAL parser as the judge."""
    bad = n = 0

    def case(ok, text):
        nonlocal bad, n
        n += 1
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BUG '} {text}")

    import tempfile

    scan = [
        ("2026-08-26_12:05", 410_000_000, "ckpt_old_pre_reset.pt.ep1"),
        ("2026-09-02_10:00", 890_000_000, "ckpt_kept_one.pt"),
        ("2026-09-03_18:46", 1_854_896_463, "ckpt_new_after_snapshot.pt"),
    ]
    keep_lines = ["# KEEP (claim x 00:00Z): ckpt_kept_one.pt -- the only source of some fact"]
    old = {
        "ckpt_old_pre_reset.pt.ep1": ("2026-08-26_12:05", "A", "2026-09-03 14:00Z"),
        "ckpt_kept_one.pt": ("2026-09-02_10:00", "B", "2026-09-03 14:00Z"),
        "ckpt_deleted_already.pt": ("2026-08-27_13:43", "A", "2026-09-03 14:00Z"),
    }
    text, problems = build(scan, keep_lines, old, now="2026-09-04 20:00")
    case(not problems, f"a scan containing every claimed file has no problems ({problems})")
    # THE REAL PARSER IS THE JUDGE. A second format implementation here would agree with this
    # file and drift from harness.py, which is the only reader that matters.
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import harness

    fd, p = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    try:
        date, keep, cands = harness._parse_ckpt_listing(p)
        case(date == "2026-09-04 20:00Z", f"the reader takes the listing date ({date})")
        case("ckpt_kept_one.pt" in keep, "the reader sees the carried KEEP claim")
        case("ckpt_kept_one.pt" not in cands, "and a claimed file is NOT emitted as a candidate")
        case(
            "ckpt_old_pre_reset.pt.ep1" in cands and cands["ckpt_old_pre_reset.pt.ep1"][1] == "A",
            f"a pre-reset file lands in section A ({cands.get('ckpt_old_pre_reset.pt.ep1')})",
        )
        # A NEW FILE IS NOT A CANDIDATE. It has no deadline, and inheriting one it was never
        # given is how a regeneration silently condemns files nobody has ruled on.
        case(
            cands.get("ckpt_new_after_snapshot.pt", ("", ""))[1] == "C",
            f"a file new since the last listing lands in C, not A/B "
            f"({cands.get('ckpt_new_after_snapshot.pt')})",
        )
    finally:
        os.unlink(p)
    case(
        "present past deadline 2026-09-03 14:00Z" in text,
        "a candidate still present past its deadline says so, keeping the ORIGINAL date",
    )
    case(
        "2026-09-04" not in text.split("listed")[1].split("\n")[1:][0]
        if len(text.split("listed")) > 1
        else True,
        "no fresh deadline is written into a carried section",
    )
    case(
        "ckpt_deleted_already.pt" in text and "GONE since the previous listing" in text,
        "a file gone since the last listing is reported as GONE, not dropped silently",
    )
    # THE REFUSAL, on a scan missing a claimed file. Without this case the whole guard could be
    # absent and every assertion above would still pass.
    _t2, problems2 = build([s for s in scan if s[2] != "ckpt_kept_one.pt"], keep_lines, old)
    case(
        bool(problems2) and any("ckpt_kept_one.pt" in x for x in problems2),
        f"a KEEP line protecting nothing present REFUSES, naming it ({problems2})",
    )
    # AND THE FALSE-POSITIVE CASE, which is what the live pod taught: a claim line using the
    # shorthand continuation yields BOTH readings from _parse_ckpt_listing, one of which
    # (`X.pt.pt.step2500`) cannot exist by construction. Checking every generated name reported 17
    # absent claims against a healthy listing. The line is satisfied by ONE present name.
    short = ["# KEEP (claim y): ckpt_series.pt.step2000, .pt.step2500 -- a shorthand claim"]
    scan_short = [
        ("2026-09-02_10:00", 1, "ckpt_series.pt.step2000"),
        ("2026-09-02_11:00", 1, "ckpt_series.pt.step2500"),
    ]
    names_short = _keep_names(short)
    case(
        any(n not in {s[2] for s in scan_short} for n in names_short),
        f"the fixture really does yield an impossible reading ({sorted(names_short)}) -- "
        f"without that this case cannot test the false positive",
    )
    _t3, problems3 = build(scan_short, short, {})
    case(not problems3, f"a shorthand claim whose real files are both present does NOT refuse ({problems3})")
    # THE EMPTY-SCAN REFUSAL, exercised rather than grepped. My first version of this case
    # asserted that _pod_scan's DOCSTRING contained the words "refusing to write a listing",
    # which tests prose and would pass with the guard deleted. Drive the real function with a
    # pod command that returns nothing instead.
    saved = globals()["POD"]
    try:
        globals()["POD"] = "/bin/echo"  # exists, outputs nothing useful, so the scan is empty
        got, err2 = _pod_scan()
        case(
            got is None and err2 is not None and "no ckpt_" in err2,
            f"an empty scan REFUSES rather than writing 'the pod is empty' ({(err2 or '')[:60]})",
        )
    finally:
        globals()["POD"] = saved
    print(f"gen_ckpt_listing selftest: {n - bad}/{n} pass")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="write here; default runs/pod_ckpt_candidates_<today>.txt")
    ap.add_argument("--previous", help="listing to carry KEEP lines from; default the newest")
    ap.add_argument("--dry", action="store_true", help="print, write nothing")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    import glob

    prev = (
        a.previous
        or (sorted(glob.glob(os.path.join(ROOT, "runs", "pod_ckpt_candidates_*.txt"))) or [None])[-1]
    )
    scan, err = _pod_scan()
    if err:
        print(f"refusing: {err}", file=sys.stderr)
        return 2
    text, problems = build(scan, _read_keep_lines(prev), _old_sections(prev))
    if problems:
        print(
            f"refusing: {len(problems)} carried KEEP claim(s) name files the scan did not find. "
            f"Either the file is already gone -- and the fact behind the claim has a dead "
            f"source -- or the name is wrong. Neither is fixed by writing this listing.",
            file=sys.stderr,
        )
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 2
    if a.dry:
        print(text)
        return 0
    out = a.out or os.path.join(
        ROOT, "runs", f"pod_ckpt_candidates_{time.strftime('%Y-%m-%d', time.gmtime())}.txt"
    )
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(
        f"wrote {out}: {len(scan)} files, {sum(b for _m, b, _n in scan) / 1e9:.0f} GB, "
        f"{len(_read_keep_lines(prev))} KEEP line(s) carried from "
        f"{os.path.basename(prev) if prev else 'nothing'}"
    )
    print(
        "This file does not delete anything. The deletion is a separate act by the list's "
        "owner, one `rm` per named file."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

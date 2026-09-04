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
    cmd = f"cd {WORK} && stat -c '%Y %s %i %n' ckpt_* 2>/dev/null"
    try:
        r = subprocess.run([POD, cmd], capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"pod scan failed: {type(e).__name__}: {e}"
    out = []
    for line in r.stdout.splitlines():
        # %i, THE INODE, is collected because the roller pins by inode and a deletion plan that
        # cannot see pins would delete pinned files (train.py:2630-2645, §162). Four fields now,
        # and the name is last so a filename containing spaces still parses.
        parts = line.split(None, 3)
        if len(parts) != 4:
            continue
        try:
            epoch, size, ino = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        out.append((time.strftime("%Y-%m-%d_%H:%M", time.gmtime(epoch)), size, parts[3].strip(), ino))
    if not out:
        return None, (
            f"scan found no ckpt_* in {WORK} -- refusing to write a listing that says "
            f"the pod is empty (stderr: {r.stderr.strip()[:200]})"
        )
    return sorted(out), None


def _read_keep_lines(path):
    """The verbatim `# KEEP` and `# RETIRED` lines of an existing listing, in order.

    VERBATIM, not parsed and re-emitted. Every claim line carries its author, its timestamp and
    its stated reason, and a regeneration that rewrote them would be one session editing five
    others' claims. The names inside them are re-verified against the scan; the text is not
    touched.

    RETIRED LINES COME TOO, and they are why this function returns both kinds. A KEEP whose files
    are gone is retired IN PLACE by a following `# RETIRED` line rather than deleted, so the
    listing keeps the record that a claim was made and did not hold -- which is the evidence for
    §162. Carrying the KEEP without its RETIRED would make a dead claim read as live; dropping
    both would erase the incident.
    """
    if not path or not os.path.isfile(path):
        return []
    return [
        ln
        for ln in open(path, encoding="utf-8").read().splitlines()
        if ln.startswith("# KEEP") or ln.startswith("# RETIRED")
    ]


def _retired_names(lines):
    """Names retired by `# RETIRED` lines -- claims already known not to have held.

    A RETIRED line is the author's own record that files a KEEP named are gone (e1, listing line
    8: "THE CLAIM DID NOT HOLD ... verified by stat"). Presence-checking those names would refuse
    every future regeneration over a fact the listing already states, so they are exempted.

    SCOPED TO THE PRECEDING KEEP LINE'S NAMES, intersected with what the RETIRED line mentions --
    and the intersection is the fix, not a nicety. A RETIRED line is PROSE, so parsing it as a
    claim line fails both ways: measured against the real listing 2026-09-04, `_keep_names` on
    line 8 returned `ckpt_p200m_4b_0902.pt.step1000` but MISSED `.pt.step1500` (it follows the
    word "and", not a comma, so the shorthand-continuation rule never fires) and INVENTED two
    unrelated names from later in the same sentence (`ckpt_p500m_20b_0902.pt.step2000` and a
    milestone file, which the sentence mentions as survivors). Missing a name leaves the guard
    refusing a retired claim -- exactly what e1 hit; inventing one silently un-protects a live
    file.
    So: a RETIRED line retires names from the KEEP line above it, and a name is retired when the
    retirement text mentions it at all -- as a whole token or as a `.tail` continuation. Nothing
    outside the preceding claim can be retired by prose, which bounds the damage a loosely-worded
    retirement can do to exactly the claim it is about.
    """
    out, pending = set(), set()
    for ln in lines:
        if ln.startswith("# KEEP"):
            pending = _keep_names([ln])
            continue
        if not ln.startswith("# RETIRED"):
            continue
        for name in pending:
            # The full name, or the tail after the run's `.pt` -- ".pt.step1500", which is how a
            # human writes the second file of a pair.
            #
            # THE TAIL TEST ALSO RETIRES THE PARSER'S IMPOSSIBLE VARIANTS, and that is required,
            # not incidental. _parse_ckpt_listing emits both readings of a shorthand claim, so
            # `.pt.step1500` yields the real `ckpt_X.pt.step1500` AND the doubled
            # `ckpt_X.pt.pt.step1500`, which cannot exist on any disk. Retiring only the exact
            # names left the doubled one in the guard, and the live run still refused -- measured
            # twice, 17 variants first and then this single one. Both variants share the tail, so
            # matching on the tail retires the pair together.
            tail = name[name.index(".pt") :] if ".pt" in name else name
            if name in ln or (len(tail) > 4 and tail in ln):
                out.add(name)
            elif ".pt.pt" in name:
                # The doubled-`.pt` reading of a name whose real form is retired: collapse and
                # re-test, so a retirement of the real file covers its impossible twin.
                real = name.replace(".pt.pt", ".pt", 1)
                rtail = real[real.index(".pt") :] if ".pt" in real else real
                if real in ln or (len(rtail) > 4 and rtail in ln):
                    out.add(name)
    return out


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
    # INDEX, not unpacking: a scan row is (mtime, size, name) from a fixture or
    # (mtime, size, name, inode) from the pod, and unpacking by arity broke every caller the
    # moment the inode column arrived. row[2] is the name in both.
    present = {row[2] for row in scan}
    problems = []
    claimed = _keep_names([ln for ln in keep_lines if not ln.startswith("# RETIRED")])
    # RETIRED LINES ARE EXCLUDED FROM `claimed` as well, not only from the presence guard.
    # `claimed` decides which files are held back from the candidate sections, so leaving a retired
    # name in it would protect a file from listing on the strength of a claim its own author
    # withdrew. The files in question are gone, so today it changes nothing visible -- but a name
    # that reappears (a rerun writing the same .stepN) would silently inherit a dead claim's
    # protection.
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
    #
    # A RETIRED CLAIM IS NOT CHECKED FOR PRESENCE (e1 via 6e, 2026-09-04). `# RETIRED` is the
    # author's own record that the files a KEEP named are gone -- listing line 8 says "THE CLAIM
    # DID NOT HOLD ... verified by stat" about line 7's two files. Presence-checking those names
    # refuses every future regeneration over a fact the listing already states, which is what e1
    # hit: --dry blocked on a claim the file itself marks dead. The retirement is carried forward
    # as history (see _read_keep_lines) and exempted from the guard.
    retired = _retired_names(keep_lines)
    for line in keep_lines:
        if line.startswith("# RETIRED"):
            continue
        names = _keep_names([line])
        # SET DIFFERENCE, so a retirement naming two of a claim's three files retires exactly
        # those two and the third is still guarded. A line-level skip would drop the whole claim.
        names -= retired
        if names and not (names & present):
            problems.append(f"KEEP line protects nothing present: {sorted(names)} -- {line[:110]}")
    total = sum(row[1] for row in scan)
    lines = [
        f"# pod {WORK} checkpoints, listed {now}Z: {len(scan)} files, "
        f"{total / 1e9:.0f} GB, generated by scripts/gen_ckpt_listing.py"
    ]
    lines += keep_lines
    # Sections keep the old listing's meaning: A is pre-reset, B is post-reset probe/interrupt
    # files. A file the old listing never saw is NEW and goes in its own section rather than
    # inheriting a deadline it was never given.
    buckets = {"A": [], "B": [], "C": []}
    for mtime, size, name in ((r[0], r[1], r[2]) for r in scan):
        # `claimed - retired`, not `claimed`. A retired claim protects nothing, so a file that
        # REAPPEARS under a retired name (a rerun writing the same .stepN) must be listed rather
        # than shielded by a claim its own author withdrew. My first version subtracted retired
        # names only in the presence guard and left them in `claimed`, and the selftest case for
        # the reappearing name caught it -- the comment there predicted the defect while the code
        # still had it, which is why the case exists rather than the comment.
        if name in claimed and name not in retired:
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

    import json
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

    # RETIRED. e1 hit this in production: --dry refused on the listing's line 7 KEEP although line
    # 8 RETIRES it with the files verified gone by stat, so a fact the file itself states blocked
    # every regeneration.
    retired_lines = [
        "# KEEP (claim z 15:16Z): ckpt_gone_a.pt, ckpt_gone_b.pt -- only source of some fact",
        "# RETIRED (z 20:13Z): THE CLAIM DID NOT HOLD. ckpt_gone_a.pt and ckpt_gone_b.pt are GONE "
        "from the pod, verified by stat",
    ]
    _t4, problems4 = build([("2026-09-03_21:55", 1, "ckpt_other.pt")], retired_lines, {})
    case(not problems4, f"a RETIRED claim does not block generation ({problems4})")
    _t4b, _p4b = build([("2026-09-03_21:55", 1, "ckpt_other.pt")], retired_lines, {})
    case(
        "# RETIRED" in _t4b and "# KEEP (claim z" in _t4b,
        "and both lines are carried forward -- the record that a claim did not hold is the "
        "evidence for §162, so neither half is dropped",
    )
    # A RETIRED NAME IS NOT PROTECTED FROM LISTING EITHER. Its own author withdrew the claim, so a
    # file that reappears under that name (a rerun writing the same .stepN) must not inherit dead
    # protection.
    _t4c, _p4c = build([("2026-09-03_21:55", 1, "ckpt_gone_a.pt")], retired_lines, {})
    case(
        "ckpt_gone_a.pt" in _t4c.replace(retired_lines[0], "").replace(retired_lines[1], ""),
        "a retired name that REAPPEARS is listed as a candidate, not shielded by the dead claim",
    )
    # PARTIAL RETIREMENT: retiring two of a claim's three files must leave the third guarded.
    # Without this, `if line.startswith("# RETIRED"): continue` at claim level would read as
    # correct while silently exempting files nobody retired.
    partial = [
        "# KEEP (claim z): ckpt_p_a.pt, ckpt_p_b.pt, ckpt_p_c.pt -- three files",
        "# RETIRED (z): ckpt_p_a.pt and ckpt_p_b.pt are GONE",
    ]
    _t5, problems5 = build([("2026-09-03_21:55", 1, "ckpt_unrelated.pt")], partial, {})
    case(
        bool(problems5) and any("ckpt_p_c.pt" in x for x in problems5),
        f"retiring two of three files leaves the THIRD guarded ({problems5})",
    )
    _t6, problems6 = build([("2026-09-03_21:55", 1, "ckpt_p_c.pt")], partial, {})
    case(not problems6, f"and the same claim passes once that third file is present ({problems6})")

    # THE REAL LISTING MUST GENERATE (6e's requirement). Uses the live KEEP/RETIRED lines and a
    # scan synthesised from the files those lines name as PRESENT, so the case tests the generator
    # against the real claim text rather than a fixture's.
    real = os.path.join(ROOT, "runs", "pod_ckpt_candidates_2026-09-02.txt")
    if os.path.isfile(real):
        real_lines = _read_keep_lines(real)
        live_names = _keep_names([x for x in real_lines if not x.startswith("# RETIRED")])
        live_names -= _retired_names(real_lines)
        fake_scan = sorted((("2026-09-03_12:00", 1, nm) for nm in live_names))
        _t7, problems7 = build(fake_scan, real_lines, _old_sections(real))
        case(
            not problems7,
            f"the REAL listing's claim lines generate when their live files are present "
            f"({len(problems7)} problem(s): {problems7[:1]})",
        )
        case(
            any(x.startswith("# RETIRED") for x in real_lines),
            f"...and the real listing really does carry a RETIRED line, without which this case "
            f"tests nothing ({sum(1 for x in real_lines if x.startswith('# RETIRED'))} found)",
        )

    # ---- deletion_plan. Every case here is about what must NOT be deleted.
    plan_listing = os.path.join(tempfile.mkdtemp(prefix="de_plan_"), "listing.txt")
    with open(plan_listing, "w", encoding="utf-8") as fh:
        fh.write(
            "# pod /work/aupai checkpoints, listed 2026-09-02 13:58Z\n"
            "# KEEP (claim q): ckpt_claimed.pt -- a live claim\n"
            "# KEEP (claim r): ckpt_retired.pt -- a claim that did not hold\n"
            "# RETIRED (r): ckpt_retired.pt is GONE from the pod, verified by stat\n"
            "# A. pre-0830v1: zeroed by the reset, deleted 2026-09-03 14:00Z unless claimed\n"
            "2026-08-26_12:05 0.41 ckpt_plain_old.pt\n"
            "2026-08-26_12:06 0.41 ckpt_pinned.pt.step100\n"
            "2026-08-26_12:07 0.41 ckpt_run.milestone_x_step100.pt\n"
            "2026-08-26_12:08 0.41 ckpt_already_gone.pt\n"
            "# C. NEW since the previous listing: no deadline has ever been set for these\n"
            "2026-09-03_20:00 0.41 ckpt_brand_new.pt\n"
        )
    # The pinned file and the milestone share an inode -- that IS the pin, as train.py reads it.
    plan_scan = [
        ("2026-08-26_12:05", 410_000_000, "ckpt_plain_old.pt", 111),
        ("2026-08-26_12:06", 410_000_000, "ckpt_pinned.pt.step100", 222),
        ("2026-08-26_12:07", 410_000_000, "ckpt_run.milestone_x_step100.pt", 222),
        ("2026-09-03_20:00", 410_000_000, "ckpt_brand_new.pt", 333),
        ("2026-09-02_10:00", 410_000_000, "ckpt_claimed.pt", 444),
    ]
    rm, refusals, skipped = deletion_plan(plan_listing, plan_scan)
    names = [n for n, _s in rm]
    case(not refusals, f"a coherent listing plans without refusing ({refusals})")
    case(names == ["ckpt_plain_old.pt"], f"exactly the one deletable candidate is planned ({names})")
    case(
        any("ckpt_pinned.pt.step100" in s and "pinned" in s for s in skipped),
        f"a file sharing a milestone's INODE is skipped as pinned ({skipped})",
    )
    case(
        any("ckpt_run.milestone_x_step100.pt" in s for s in skipped),
        "the milestone hardlink itself is never deleted -- removing it unpins what it protects",
    )
    case(
        any("ckpt_brand_new.pt" in s and "section C" in s for s in skipped),
        "a section-C file is skipped: no deadline was ever set for it",
    )
    case(
        any("ckpt_already_gone.pt" in s and "already gone" in s for s in skipped),
        "a candidate the scan did not find is skipped, not planned",
    )

    # THE SUBJECT PIN, and the broken world is the exact shape that got past the real plan
    # (6e's specification, 2026-09-04): a fact whose `source` names an eval script and whose
    # SUBJECT is a planned candidate. Before _fact_subjects, ckpt_plain_old.pt below was the one
    # file the plan named; with a fact about it, the plan must name nothing.
    fdir = tempfile.mkdtemp()
    json.dump({"facts": [{
        "id": "t.subject_only",
        "value": "known-answer panel 3/4 on ckpt_plain_old.pt",
        # source names the SCRIPT and a /tmp run, never the checkpoint -- so a source scan
        # passes after the file is deleted and the number becomes unrecomputable in silence.
        "source": "eval/math_v2_like.py; pod runs /tmp/ka_math_norm.json",
    }]}, open(os.path.join(fdir, "base_eval.json"), "w"))
    rm4, ref4, sk4 = deletion_plan(plan_listing, plan_scan, facts_dir=fdir)
    case(
        not any(n == "ckpt_plain_old.pt" for n, _s in rm4),
        f"a fact's SUBJECT is not planned for deletion even though no source field names it ({[n for n, _ in rm4]})",
    )
    case(
        any("ckpt_plain_old.pt" in s and "t.subject_only" in s for s in sk4),
        f"and the skip names the fact id, so the reader can judge the claim ({sk4})",
    )
    # THE CONTROL: without that facts dir the same file IS planned. Without this line the case
    # above passes for a plan that names nothing at all, which is the shape of a guard that
    # agrees because it measured nothing.
    empty_fdir = tempfile.mkdtemp()
    rm5, _r5, _s5 = deletion_plan(plan_listing, plan_scan, facts_dir=empty_fdir)
    case(
        any(n == "ckpt_plain_old.pt" for n, _s in rm5),
        f"control: with no fact about it, the same file is planned ({[n for n, _ in rm5]})",
    )
    # A FACTS FILE THAT WILL NOT PARSE MUST REFUSE, not widen the plan by skipping itself.
    bad_fdir = tempfile.mkdtemp()
    open(os.path.join(bad_fdir, "broken.json"), "w").write("{not json")
    try:
        deletion_plan(plan_listing, plan_scan, facts_dir=bad_fdir)
        case(False, "an unparseable facts file must refuse, not silently widen the plan")
    except SystemExit as e:
        case("did not parse" in str(e), f"an unparseable facts file refuses, naming it ({e})")
    # THE WHOLE-RUN REFUSAL. A claimed name reaching a candidate section means the listing
    # contradicts itself, and no part of it can then be trusted to say what is safe.
    #
    # INSERTED UNDER SECTION A, not appended. A line appended at the end of the file falls under
    # the LAST section header, which is C -- verified against harness._parse_ckpt_listing -- and a
    # section-C name is skipped for a different reason entirely, so the case would have passed
    # while proving the weaker thing. The contradiction has to sit in a section whose files are
    # genuinely deletable.
    contradictory = plan_listing + ".contra"
    with open(plan_listing, encoding="utf-8") as fh:
        _src = fh.read().splitlines()
    _out = []
    for _ln in _src:
        _out.append(_ln)
        if _ln.startswith("# A."):
            _out.append("2026-08-26_12:09 0.41 ckpt_claimed.pt")
    with open(contradictory, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_out) + "\n")
    rm2, refusals2, sk2 = deletion_plan(contradictory, plan_scan)
    # A CLAIMED NAME INSIDE A CANDIDATE SECTION IS NORMAL, not a contradiction: the sections are a
    # full pod inventory and the KEEP lines are the exemption laid over them. Measured against the
    # real listing -- ckpt_sft_p324_v3.pt is claimed on line 3 and inventoried on line 145 -- after
    # my abort-the-run version refused 16 correctly-claimed files.
    case(
        not refusals2 and "ckpt_claimed.pt" not in [n for n, _s in rm2],
        f"a claimed name in a candidate section is NOT planned and does not abort ({refusals2})",
    )
    case(
        any("ckpt_claimed.pt" in x and "KEEP" in x for x in sk2),
        f"...and the skip says which claim protected it ({[x for x in sk2 if 'claimed' in x]})",
    )
    # A RETIRED claim must NOT abort it: its author already recorded that the file is gone.
    case(
        not any("ckpt_retired.pt" in r for r in refusals2),
        "a RETIRED claim does not abort the plan the way a live one does",
    )
    # NO INODE COLUMN -> NO PLAN. A 3-tuple scan is what every fixture above uses, so this also
    # proves the refusal is not an artifact of the 4-tuple shape.
    rm3, refusals3, _sk3 = deletion_plan(contradictory, [(m, b, n) for m, b, n, _i in plan_scan])
    case(
        bool(refusals3) and not rm3 and any("inode" in r for r in refusals3),
        f"a scan without inodes REFUSES to plan rather than treating pins as absent ({refusals3})",
    )

    print(f"gen_ckpt_listing selftest: {n - bad}/{n} pass")
    return 1 if bad else 0


def deletion_plan(listing_path, scan, now=None, facts_dir=None):
    """(rm_list, refusals, skipped) for the listing of record. NOTHING IS DELETED HERE.

    rm_list is [(name, bytes)] in listing order. refusals is non-empty when the whole run must be
    abandoned; skipped explains every candidate that was excluded and why, because a plan that
    silently omits a file is indistinguishable from one that forgot it.

    SIX REFUSALS AND EXCLUSIONS, each one bought by an incident:

      A NAME WITH A KEEP LINE ABORTS THE WHOLE RUN, not just its own file. If a claimed name has
      reached a candidate section, the listing's two halves contradict each other and no part of it
      can be trusted to say what is safe -- the one thing this run must never do is what already
      happened to e1's three files (§162).

      NEVER DELETE A NAME THE PARSER INVENTED (6e's ruling 2026-09-04). _parse_ckpt_listing emits
      both readings of a shorthand claim, so `.pt.step1500` also yields `X.pt.pt.step1500`, a name
      that cannot exist on any disk. Those spare readings are harmless in a KEEP set and lethal in
      an rm list: a bug that collapsed one of them to a real name would delete a claimed file. Only
      a name the SCAN found is ever deleted, so every rm target is a file that demonstrably exists
      under exactly that name.

      *.milestone_*.pt IS NEVER TOUCHED. train.py's roller pins by INODE against those files
      (train.py:2630-2645), so a milestone hardlink is what makes a checkpoint survive rotation.
      Deleting the link unpins the file it protects, and the next save rotates the real checkpoint
      away -- the deletion would take effect minutes later, somewhere else.

      A ROTATING .stepN WITHOUT A PIN IS REPORTED, NOT DELETED-BY-DEFAULT. It is already at the
      roller's mercy, so a deletion here is not the decision it appears to be; the plan says so
      rather than taking credit for a removal the roller was going to do anyway.

      ONLY NAMES IN A CANDIDATE SECTION. Section C is "new since the last listing, no deadline has
      ever been set" -- a file cannot be overdue against a deadline it never had.

      data/ IS NOT IN SCOPE AT ALL. This function only ever names files the pod scan returned from
      /work/aupai's ckpt_* glob, so raw corpus cannot enter the list by construction rather than by
      an exclusion someone has to remember.
    """
    # harness's parser, not a local copy: the plan must partition the listing exactly as the
    # CHECK reads it, or a file could be a candidate to one and not the other.
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import harness

    date, keep, cands = harness._parse_ckpt_listing(listing_path)
    lines = _read_keep_lines(listing_path)
    retired = _retired_names(lines)
    live_keep = keep - retired
    present = {row[2]: row[1] for row in scan}
    pinned = _pinned_inodes(scan)
    # NO PIN INFORMATION MEANS NO PLAN. _pinned_inodes returns None when the scan has no inode
    # column, and the roller's pin is an inode match -- so without it every pinned file reads as
    # unpinned and the plan would name files the roller is deliberately protecting. This is the
    # §162 failure with the roles swapped: there a claim was invisible to the deleter; here the
    # deleter would be blind to the pin.
    if pinned is None:
        return (
            [],
            [
                "the scan carries no inode column, so pins cannot be seen -- a plan built "
                "without them would name files the roller is protecting (train.py:2630-2645)"
            ],
            [],
        )
    rm, refusals, skipped = [], [], []
    fact_subjects = _fact_subjects(cands, root=ROOT, facts_dir=facts_dir)
    for name, (mtime, section) in sorted(cands.items()):
        if name in live_keep:
            # SKIPPED, NOT AN ABORT, and my first version had this backwards. I designed a
            # whole-run refusal on the theory that a claimed name inside a candidate section means
            # the listing contradicts itself. It does not: the 09-02 listing's candidate sections
            # are a FULL INVENTORY of the pod, and KEEP lines are the exemption mechanism laid over
            # them -- ckpt_sft_p324_v3.pt is claimed on line 3 and inventoried on line 145, by
            # design. The live plan refused 16 files on that misreading, every one of them a
            # correctly claimed file, which would have read as "the listing is broken" when the
            # listing was working exactly as built.
            #
            # This is the protection doing its job, so it is reported as a skip with the claim
            # named. What §162 was about is the opposite case -- a claim the DELETER never reads --
            # and here the deleter reads it, which is the whole point of planning from the listing.
            skipped.append(f"{name}: a live KEEP line claims it")
            continue
        if section == "C":
            skipped.append(f"{name}: section C, no deadline was ever set for it")
            continue
        if name in fact_subjects:
            # A FACT'S SUBJECT IS NOT IN ITS SOURCE, and that gap is why this condition exists
            # rather than the source scan being enough (b0 2026-09-04, finding
            # b0-plan-deletion-subject-pin). The 09-04 plan named ckpt_0830v1_3.24b.pt.ep1 for
            # deletion while three published facts are ABOUT it -- be.known_answer_panel_3_4,
            # be.lambada_zh_heldout_3p24b, be.math_v2_5k_known_answer -- because all three name
            # eval scripts and /tmp run files in `source` and none names the checkpoint. Every
            # existence check passes after the file is gone: the fact resolves, its artifacts are
            # present, and the number can never be recomputed. That is the failure that reads as
            # healthy, and it is worse than a dangling reference, which at least fails loudly.
            #
            # So the pin is "named anywhere in a facts/*.json fact object", not "named in a source
            # field". Skipped rather than refused, for the same reason as a KEEP: the listing's
            # candidate sections are a full inventory, and a fact naming a file is an exemption
            # laid over it, not a contradiction.
            #
            # DELIBERATELY NOT EXTENDED TO runs/*.jsonl. A score_matrix or experiments row citing a
            # checkpoint RECORDS a measurement that already happened, so deleting the file costs
            # re-scorability, not the record -- 30 of the 33 cited files in the 09-04 plan are that
            # kind, and pinning them would freeze the pod. A fact is a published claim with no
            # other route back; a row is its own evidence.
            skipped.append(f"{name}: named in {fact_subjects[name]} -- a fact's subject, "
                           f"whose deletion makes that number unrecomputable")
            continue
        if name not in present:
            skipped.append(f"{name}: already gone from the pod")
            continue
        if ".milestone_" in name:
            skipped.append(f"{name}: a milestone hardlink -- deleting it unpins what it protects")
            continue
        if name in pinned:
            skipped.append(f"{name}: pinned by a milestone hardlink (same inode)")
            continue
        rm.append((name, present[name]))
    if refusals:
        return [], refusals, skipped
    return rm, [], skipped


def _fact_subjects(names, root=None, facts_dir=None):
    """{checkpoint name -> "fact ids"} for every candidate named anywhere in a facts/*.json fact.

    THE WHOLE POINT IS "ANYWHERE IN THE OBJECT", not "in the source field". A fact's SUBJECT --
    the checkpoint the number is about -- is routinely absent from its source, which names the
    eval script and the run log instead. Measured 2026-09-04: be.math_v2_5k_known_answer's source
    is `eval/math_v2_like.py; mathbank/eval_hard_v2_gen.py; pod runs /tmp/ka5k_{norm,swap}.json`
    and the checkpoint appears only in the value/config text. So this serialises each fact whole
    and does a substring test.

    SUBSTRING, AND THE LONGEST MATCH IS NOT THE ISSUE HERE. `ckpt_x.pt` is a substring of
    `ckpt_x.pt.step500`, so a fact naming the .stepN also flags the base name. That direction is
    safe -- it pins MORE than strictly needed and a pin is recoverable, a deletion is not. The
    reverse would be the dangerous one and does not occur: a fact naming only the base cannot flag
    a .stepN, because the .stepN string is not present.
    """
    import glob
    import json as _json

    fd = facts_dir or os.path.join(root or ROOT, "facts")
    out = {}
    for fp in sorted(glob.glob(os.path.join(fd, "*.json"))):
        try:
            facts = _json.load(open(fp, encoding="utf-8")).get("facts", [])
        except (OSError, ValueError):
            # A FACTS FILE THAT WILL NOT PARSE MUST NOT SILENTLY WIDEN THE PLAN. Skipping it here
            # would mean every checkpoint it pins becomes deletable with no message, so the caller
            # is told through the returned marker rather than by absence.
            out.setdefault("__unreadable__", []).append(os.path.basename(fp))
            continue
        for fact in facts:
            blob = _json.dumps(fact, ensure_ascii=False)
            fid = fact.get("id", "?")
            for n in names:
                if n in blob:
                    out.setdefault(n, [])
                    if fid not in out[n]:
                        out[n].append(fid)
    bad = out.pop("__unreadable__", None)
    if bad:
        raise SystemExit(f"REFUSING to plan a deletion: {bad} did not parse as JSON, so the "
                         f"checkpoints those facts pin would read as deletable. Fix the file.")
    return {k: ", ".join(v) for k, v in out.items()}


def _pinned_inodes(scan):
    """Names in `scan` that share an inode with a *.milestone_*.pt file, or None if unknown.

    The roller's own test (train.py:2630-2645), reproduced rather than approximated: it stats every
    *.milestone_*.pt and skips a stale file whose st_ino matches. Judging a pin by NAME would miss
    exactly the files the pin exists to protect, because a pin IS a hardlink under a different name.

    NONE, NOT AN EMPTY SET, when the scan carries no inode column. My first version returned `set()`
    with a docstring saying the caller must treat it as "no information" -- which is the §162 shape
    in miniature: a stub whose contract lives in prose the caller cannot read. An empty set says
    "nothing is pinned", and a plan built on it would delete pinned files. None forces the caller to
    handle the unknown, and deletion_plan refuses on it.
    """
    inodes = {}
    for row in scan:
        if len(row) < 4:
            return None
        _m, _b, name, ino = row[0], row[1], row[2], row[3]
        inodes.setdefault(ino, []).append(name)
    pinned = set()
    for ino, names in inodes.items():
        if any(".milestone_" in n for n in names):
            pinned |= set(names)
    return pinned


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="write here; default runs/pod_ckpt_candidates_<today>.txt")
    ap.add_argument("--previous", help="listing to carry KEEP lines from; default the newest")
    ap.add_argument("--dry", action="store_true", help="print, write nothing")
    ap.add_argument(
        "--plan-deletion",
        action="store_true",
        help="print the rm plan for the listing of record and exit. Deletes nothing, "
        "and this is the only deletion-related mode this script has: the rm "
        "itself is the list owner's act, one file at a time, with this plan as "
        "its input",
    )
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
    if a.plan_deletion:
        if not prev:
            print("refusing: no listing of record to plan from", file=sys.stderr)
            return 2
        rm, refusals, skipped = deletion_plan(prev, scan)
        if refusals:
            print(
                f"REFUSING the whole run: {len(refusals)} contradiction(s) in {os.path.basename(prev)}",
                file=sys.stderr,
            )
            for r in refusals:
                print(f"  {r}", file=sys.stderr)
            return 2
        print(f"# deletion plan from {os.path.basename(prev)}, scan of {len(scan)} files")
        for name, size in rm:
            print(f"rm {name}    # {size / 1e9:.2f} GB")
        print(f"# {len(rm)} file(s), {sum(s for _n, s in rm) / 1e9:.1f} GB")
        print(f"# {len(skipped)} candidate(s) NOT in the plan:")
        for s in skipped:
            print(f"#   {s}")
        print(
            "# This script does not delete. Run the rm lines yourself, then regenerate the "
            "listing so the after-state is recorded."
        )
        return 0
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
        f"wrote {out}: {len(scan)} files, {sum(row[1] for row in scan) / 1e9:.0f} GB, "
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

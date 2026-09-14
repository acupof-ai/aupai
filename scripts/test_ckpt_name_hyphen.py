#!/usr/bin/env python3
"""b0-30: harness.py's checkpoint-name extraction must admit `-` in a run name.

WHY THIS TEST EXISTS. `ckpt_[\\w.]+?` was the core class at all five extraction sites, and
`\\w` excludes the hyphen, so every hyphenated run name died at its first hyphen and was
NEVER SCANNED. Measured 2026-09-06: ckpt_1.5b-a0.2b-e48_8b.pt and its
milestone_matchedtok_step5000 sibling are cited by
moe.equal_token_gap_vs_dense_b192_per_domain, and reached neither _ckpt_names (so
ckpt_facts_sources_present could not flag them) nor _parse_ckpt_listing's keep set (so no
KEEP line could protect them) -- 8 files on the pod, 2 fact citations.

THE BUG PRODUCED NO FALSE GREEN, WHICH IS WHY IT SURVIVED. Both sides were blind the same
way, so nothing read green on protection it lacked. What it did instead was report a SUBSET:
the check named 2 of the 4 checkpoints that entry cites, and a KEEP line covering the two it
named would have left the other two unprotected with the gate reading WARN. So the assertions
below are not only "the hyphenated name now resolves" -- they include a NEGATIVE CONTROL per
site (the pre-fix pattern must fail the same case), because a test that only exercises the
current source cannot tell a fix from a coincidence.

FIVE SITES, ONE CHANGE, AND THE ORDER IS LOAD-BEARING. Widening _ckpt_names alone leaves a
FAIL no writable claim can clear; widening _parse_ckpt_listing alone lets a claim silently
cover names the scanner still cannot flag, which IS the false-green shape. Worse, the rebase
at harness.py:4045 takes `.group(0)` unguarded: a hyphenated core with no dot before `.pt`
(ckpt_moe-48.pt) matches a widened token regex but no `ckpt_[\\w.]+?(?=\\.)`, so widening the
token site alone converts a silent miss into an AttributeError that aborts every check in the
run. test_hyphen_core_without_dot_does_not_crash pins that.

WHAT MUST NOT CHANGE is asserted too: the docstring's own over-match control
(preds_l1_d3_ckpt_p200m_4b_0902.pt.en.jsonl must not mint a checkpoint), prose dashes must not
be swallowed into a name, and the real repo's fact citations and keep set must not lose a
member -- a widened class that over-matches would be silent, since over-protection never
fails a gate.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

MOE = "ckpt_1.5b-a0.2b-e48_8b.pt"
MOE_PIN = "ckpt_1.5b-a0.2b-e48_8b.milestone_matchedtok_step5000.pt"
DEN = "ckpt_0.2b_8b_b192.pt"
# The core class as it stood before b0-30. Every negative control below asserts that the OLD
# pattern fails the case the new one handles: without this the test passes on the buggy source.
OLD_CORE = r"ckpt_[\w.]+?"

RESULTS = []


def case(ok, label, detail=""):
    RESULTS.append((bool(ok), label, detail))


def _harness():
    import harness
    return harness


def test_names_finds_hyphenated():
    h = _harness()
    got = h._ckpt_names(f"measured on {MOE} and {MOE_PIN}")
    case(MOE in got, "_ckpt_names finds a hyphenated bare .pt", f"got {sorted(got)}")
    case(MOE_PIN in got, "_ckpt_names finds a hyphenated milestone name", f"got {sorted(got)}")
    # NEGATIVE CONTROL: the pre-fix pattern must miss exactly this.
    old = re.findall(rf"(?<![A-Za-z0-9_.]){OLD_CORE}\.pt[\w.]*", MOE)
    case(old == [], "control: the pre-fix core class misses it (so the assertion above has teeth)",
         f"old pattern returned {old}")


def test_names_still_finds_plain():
    h = _harness()
    got = h._ckpt_names(f"measured on {DEN}")
    case(DEN in got, "_ckpt_names still finds a name with no hyphen", f"got {sorted(got)}")


def test_embedded_name_still_not_minted():
    """The docstring's own over-match control, which the widening must not break."""
    h = _harness()
    got = h._ckpt_names("preds_l1_d3_ckpt_p200m_4b_0902.pt.en.jsonl")
    case(got == set(),
         "an embedded checkpoint name in a preds filename still mints nothing",
         f"got {sorted(got)}")


def test_prose_dashes_not_swallowed():
    """`ckpt_x.pt -- the only source of` is the KEEP line's own shape.

    The tail after `.pt` is deliberately NOT widened. If it were, the em-dash prose that every
    KEEP line uses would be absorbed into the checkpoint name and no claim would resolve.
    """
    h = _harness()
    got = h._ckpt_names("ckpt_kept_one.pt -- the only source of some fact")
    case(got == {"ckpt_kept_one.pt"},
         "a trailing ` -- reason` is not absorbed into the name", f"got {sorted(got)}")
    got2 = h._ckpt_names("ckpt_a.pt, ckpt_b.pt -- two files")
    case(got2 == {"ckpt_a.pt", "ckpt_b.pt"},
         "two names plus a dash-prose reason resolve to exactly two", f"got {sorted(got2)}")


def test_path_prefixed_citation_still_resolves():
    """The lookbehind keeps its ORIGINAL class, and this is why.

    Adding `-` to the lookbehind would drop runs/b0-ckpt_x.pt -- a real citation shape, a
    directory whose name ends in a hyphen. That would trade one blind spot for another.
    """
    h = _harness()
    got = h._ckpt_names("see runs/b0-ckpt_x.pt for the arm")
    case("ckpt_x.pt" in got,
         "a citation prefixed by a hyphen-ending path still resolves", f"got {sorted(got)}")


def test_brace_expansion_with_hyphenated_core():
    h = _harness()
    got = h._ckpt_names("ckpt_1.5b-a0.2b-e48_8b.pt.step{5000, 9000}")
    want = {"ckpt_1.5b-a0.2b-e48_8b.pt.step5000", "ckpt_1.5b-a0.2b-e48_8b.pt.step9000"}
    case(want <= got, "brace enumeration expands on a hyphenated core", f"got {sorted(got)}")
    old = re.sub(rf"({OLD_CORE})\.step\{{([\d, ]+)\}}", "EXPANDED",
                 "ckpt_1.5b-a0.2b-e48_8b.pt.step{5000, 9000}")
    case("EXPANDED" not in old,
         "control: the pre-fix brace pattern does not expand a hyphenated core",
         f"old sub gave {old!r}")


def _listing(tmp, keep_lines, rows=()):
    p = os.path.join(tmp, "listing.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("# pod /work/aupai checkpoints, listed 2026-09-06 21:20Z\n")
        for ln in keep_lines:
            fh.write(ln + "\n")
        fh.write("# C. NEW since the previous listing\n")
        for r in rows:
            fh.write(r + "\n")
    return p


def test_keep_claim_on_hyphenated_name_parses():
    """THE POSITIVE WORLD: a hyphenated name is both flagged as a candidate and claimable.

    This is the pair that matters. Before the fix the candidate row parsed (the row regex is
    `(\\S+)`, which has no character class) while the KEEP line naming the same file did not --
    so the file could be listed for deletion and could not be protected.
    """
    import tempfile
    h = _harness()
    with tempfile.TemporaryDirectory() as tmp:
        keep = [f"# KEEP (claim b0 21:2xZ): {MOE}, {MOE_PIN} -- sources of a fact"]
        rows = [f"2026-09-06_13:09 5.99 {MOE}", f"2026-09-06_08:58 6.06 {MOE_PIN}"]
        date, kept, cands = h._parse_ckpt_listing(_listing(tmp, keep, rows))
        case(date == "2026-09-06 21:20Z", "listing date parses", f"got {date!r}")
        case(MOE in cands, "a hyphenated candidate row is seen as a candidate")
        case(MOE in kept, "a KEEP line protects a hyphenated bare name", f"kept {sorted(kept)}")
        case(MOE_PIN in kept, "a KEEP line protects a hyphenated milestone name",
             f"kept {sorted(kept)}")
        # NEGATIVE CONTROL on the parser side, using the pre-fix token pattern.
        old_toks = re.findall(rf"{OLD_CORE}\.pt[\w.]*", keep[0])
        case(MOE not in old_toks,
             "control: the pre-fix parser could not claim it (candidate yes, claim no)",
             f"old tokens {old_toks}")


def test_keep_shorthand_continuation_on_hyphenated_core():
    """`X.pt.step5000, .pt.step9000` shorthand must rebase onto the WHOLE core.

    This is the assertion that found the second defect (b0-30's dotted-base half). The rebase
    used a non-greedy `ckpt_[\\w.]+?(?=\\.)`, which stops at the first dot INSIDE the core, so
    the "bare core" reading of a dotted name was a truncation: ckpt_0.2b_8b_b192 -> 'ckpt_0'.
    Both a hyphenated and a plain-but-dotted name are checked, because the bug needs no hyphen
    and was live on main; ckpt_p200m_4b_0902 is the undotted control that must not change.
    """
    import tempfile
    h = _harness()
    with tempfile.TemporaryDirectory() as tmp:
        keep = ["# KEEP (claim b0): ckpt_1.5b-a0.2b-e48_8b.pt.step5000, .pt.step9000 -- a series"]
        _d, kept, _c = h._parse_ckpt_listing(_listing(tmp, keep))
        case("ckpt_1.5b-a0.2b-e48_8b.pt.step5000" in kept,
             "the first name of a hyphenated shorthand claim is kept", f"kept {sorted(kept)}")
        case("ckpt_1.5b-a0.2b-e48_8b.pt.step9000" in kept,
             "the continuation rebases onto the hyphenated core", f"kept {sorted(kept)}")
        # ONLY A MISSING READING IS A DEFECT; AN EXTRA ONE IS NOT. The keep set is an
        # EXEMPTION set, so a name that cannot exist exempts nothing and costs nothing --
        # _parse_ckpt_listing's docstring makes that its rule. So the assertion is that the
        # correctly-rebased name is PRESENT, not that the truncated or doubled forms are
        # absent. My first two versions asserted absence and were wrong twice: `...pt.pt.step`
        # is the `.pt`-boundary reading, and `ckpt_1.pt.step` is the short reading, both kept
        # on purpose. Demanding their absence would have forced a fix that REPLACES a reading,
        # which is exactly the regression de caught -- for
        # ckpt_n7c_p3.milestone_keep_e1_n8source.pt the short reading is the one that names a
        # real file (test_short_reading_still_protects_the_real_sibling below).
        case("ckpt_1.5b-a0.2b-e48_8b.pt.step9000" in kept,
             "the correctly-rebased hyphenated name is in the keep set", f"kept {sorted(kept)}")

        # NO HYPHEN, still dotted: the half of b0-30 that was live on main independently.
        keep2 = ["# KEEP (claim b0): ckpt_0.2b_8b_b192.pt.step5000, .pt.step9000 -- a series"]
        _d, kept2, _c = h._parse_ckpt_listing(_listing(tmp, keep2))
        case("ckpt_0.2b_8b_b192.pt.step9000" in kept2,
             "a DOTTED but unhyphenated core rebases correctly (needs no hyphen to break)",
             f"kept {sorted(kept2)}")
        case("ckpt_0.pt.step9000" in kept2,
             "the harmless short reading is still kept (an extra reading is not a defect)",
             f"kept {sorted(kept2)}")
        old_base = re.match(rf"{OLD_CORE}(?=\.)", "ckpt_0.2b_8b_b192.pt.step5000")
        case(old_base and old_base.group(0) == "ckpt_0",
             "control: the pre-fix rebase really did truncate to 'ckpt_0'",
             f"old base {old_base.group(0) if old_base else None!r}")

        # UNDOTTED CONTROL: the fix must be byte-identical here, or it changed behaviour
        # for every existing claim in the listing.
        keep3 = ["# KEEP (claim b0): ckpt_p200m_4b_0902.pt.step2500, .pt.step3000 -- a series"]
        _d, kept3, _c = h._parse_ckpt_listing(_listing(tmp, keep3))
        case("ckpt_p200m_4b_0902.pt.step3000" in kept3,
             "an undotted core still rebases exactly as before", f"kept {sorted(kept3)}")


def test_hyphen_core_without_dot_does_not_crash():
    """harness.py:4045 does `.group(0)` unguarded -- widening one site alone arms it.

    ckpt_moe-48.pt has a hyphen and NO dot inside the core. It matches a widened token regex
    but not `ckpt_[\\w.]+?(?=\\.)`, so a half-applied fix raises AttributeError here and aborts
    every check in the run. Both regexes must carry `-` or neither.
    """
    import tempfile
    h = _harness()
    with tempfile.TemporaryDirectory() as tmp:
        keep = ["# KEEP (claim b0): ckpt_moe-48.pt -- a hyphen with no dot before .pt"]
        try:
            _d, kept, _c = h._parse_ckpt_listing(_listing(tmp, keep))
        except AttributeError as e:
            case(False, "a hyphenated core with no inner dot does not crash the parser",
                 f"AttributeError: {e}")
            return
        case("ckpt_moe-48.pt" in kept,
             "a hyphenated core with no inner dot parses and is kept", f"kept {sorted(kept)}")


def test_real_repo_gains_the_moe_names_and_loses_nothing():
    """On the REAL tree: the two MoE citations enter scope, and no existing name leaves.

    Over-protection is silent -- a widened class that swallowed a following word would still
    make every gate green -- so the no-loss half is asserted against counts read from the
    actual listing and facts, not against a fixture.
    """
    import glob
    import json
    h = _harness()
    listings = sorted(glob.glob(os.path.join(ROOT, "runs", "pod_ckpt_candidates_*.txt")))
    if not listings:
        case(False, "the real repo has a candidates listing to read")
        return
    _d, kept, cands = h._parse_ckpt_listing(listings[-1])
    case(MOE in kept, f"real listing: {MOE} is now KEEP-claimed")
    case(MOE_PIN in kept, f"real listing: {MOE_PIN} is now KEEP-claimed")
    case(DEN in kept, f"real listing: the non-hyphenated {DEN} is still KEEP-claimed")

    # Every name the parser returns must be a plausible checkpoint filename: no whitespace and
    # no trailing hyphen. An over-matching class shows up here as a name with prose glued on.
    #
    # `.pt` IS NOT REQUIRED, and my first version of this canary wrongly required it. It flagged
    # ckpt_p500m_20b_0902.interrupt.step83, which is not over-match at all -- it is the BARE-CORE
    # reading of the shorthand claim `X.pt.interrupt.step83`, and _parse_ckpt_listing's docstring
    # says both readings are kept on purpose ("the wrong reading names a file that cannot exist,
    # so over-protection costs nothing"). A canary must be calibrated against what the subject
    # deliberately does, or it reports the design as a defect.
    bad = sorted(n for n in kept if " " in n or n.endswith("-") or "--" in n)
    case(not bad, "no kept name carries glued-on prose (over-match canary)", f"suspect {bad}")
    # And the deliberate bare-core readings must still be there: dropping them would be a
    # silent narrowing, which this canary would otherwise reward.
    bare = sorted(n for n in kept if ".pt" not in n)
    case(bare, "bare-core readings are still produced (they are intentional, not over-match)",
         f"found {len(bare)}")

    cited = set()
    for fp in sorted(glob.glob(os.path.join(ROOT, "facts", "*.json"))):
        try:
            obj = json.load(open(fp, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for e in obj.get("facts", []):
            if not isinstance(e, dict):
                continue
            blob = str(e.get("source", "")) + " " + json.dumps(e.get("config"), ensure_ascii=False)
            cited |= h._ckpt_names(blob)
    case(MOE in cited, "real facts: the MoE endpoint is now within the check's scope")
    case(MOE_PIN in cited, "real facts: the MoE step5000 pin is now within the check's scope")
    bad2 = sorted(n for n in cited if ".pt" not in n or " " in n)
    case(not bad2, "no cited name carries glued-on prose (over-match canary)", f"suspect {bad2}")


def test_short_reading_still_protects_the_real_sibling():
    """THE REGRESSION MY FIRST FIX WOULD HAVE SHIPPED (de caught it, 2026-09-06).

    My first repair REPLACED the short base (first-dot-only) with the core base
    (everything before `.pt`). That is correct for a dotted run name and WRONG here:
    ckpt_n7c_p3.milestone_keep_e1_n8source.pt is a milestone name, and its series siblings
    on the pod are ckpt_n7c_p3.pt.step250 / .step500 -- reachable ONLY from the short
    reading. Replacing it would have removed live protection from e1's KEEP claim in order
    to fix a different name's bug, and no assertion I had written would have noticed.

    So the fix is a UNION of three prefixes, and this is the assertion that pins it. The
    general form: when a function deliberately produces several candidate readings, a repair
    adds one -- swapping one out silently drops whatever depended on it.
    """
    import tempfile
    h = _harness()
    with tempfile.TemporaryDirectory() as tmp:
        keep = ["# KEEP (claim e1): ckpt_n7c_p3.milestone_keep_e1_n8source.pt, .pt.step250, "
                ".pt.step500 -- the milestone plus its two series siblings"]
        _d, kept, _c = h._parse_ckpt_listing(_listing(tmp, keep))
        case("ckpt_n7c_p3.pt.step250" in kept,
             "the SHORT reading still protects the real sibling of a milestone-named claim",
             f"kept {sorted(kept)}")
        case("ckpt_n7c_p3.pt.step500" in kept,
             "...and its second sibling", f"kept {sorted(kept)}")
        case("ckpt_n7c_p3.milestone_keep_e1_n8source.pt" in kept,
             "the claimed milestone itself is kept", f"kept {sorted(kept)}")


def test_first_dot_after_a_hyphen_does_not_crash():
    """de's fifth name: `ckpt_k3-mla_2b_step2000.pt` matched NOTHING before the fix.

    A second failure mode in the same expression, distinct from the truncation: when the
    first `.` follows a hyphen there is no `[\\w.]+?` match before it at all, so the old
    unguarded `.group(0)` raised AttributeError rather than returning a wrong base. Whether
    that surfaced as a crash or as silence depended on the caller, and a swallowed one means
    a KEEP claim on such a series protects zero files with nothing in the output to notice.
    """
    import tempfile
    h = _harness()
    with tempfile.TemporaryDirectory() as tmp:
        keep = ["# KEEP (claim x): ckpt_k3-mla_2b_step2000.pt, .pt.step100 -- series"]
        try:
            _d, kept, _c = h._parse_ckpt_listing(_listing(tmp, keep))
        except AttributeError as e:
            case(False, "a name whose first dot follows a hyphen does not crash the parser",
                 f"AttributeError: {e}")
            return
        case("ckpt_k3-mla_2b_step2000.pt" in kept,
             "a name whose first dot follows a hyphen is kept", f"kept {sorted(kept)}")
        case("ckpt_k3-mla_2b_step2000.pt.step100" in kept,
             "...and its continuation rebases", f"kept {sorted(kept)}")
        # NEGATIVE CONTROL: the pre-fix expression really did return None here.
        old = re.match(rf"{OLD_CORE}(?=\.)", "ckpt_k3-mla_2b_step2000.pt")
        case(old is None,
             "control: the pre-fix rebase matched nothing on this name (hence .group(0) raised)",
             f"old match {old.group(0) if old else None!r}")


def test_gate_rolling_save_is_seen_and_claimable():
    """de-111: the FIRST GATE CHECKPOINT is a rolling save, ckpt_v41_gate_0911.pt.step2000.

    train.py saves numbered checkpoints as `<core>.pt.step<N>`; the gate's first checkpoint
    (~step 2000) has that shape, and the HumanEval facts 66 writes cite it. The provenance
    chain must extract the FULL rolling name from the fact text, see it as a listing
    candidate, let a `core, .pt.step2000` KEEP shorthand protect it, and make
    ckpt_facts_sources_present resolve it. This name has an UNDOTTED core
    (`v41_gate_0911`) plus a `.pt.step<N>` tail, so it is orthogonal to the b0-30 hyphen
    cases and earns its own pin.
    """
    import json
    import tempfile
    h = _harness()
    GATE = "ckpt_v41_gate_0911.pt.step2000"

    got = h._ckpt_names(f"eval/humaneval_sample.py --ckpt {GATE}")
    case(GATE in got, "the gate rolling-save name extracts with its full .step2000 tail",
         f"got {sorted(got)}")

    with tempfile.TemporaryDirectory() as tmp:
        keep = ["# KEEP (gate 2026-09-12): ckpt_v41_gate_0911.pt, .pt.step2000 -- HumanEval fact"]
        rows = ["2026-09-12_17:29 2080.0 " + GATE]
        _d, kept, cands = h._parse_ckpt_listing(_listing(tmp, keep, rows))
        case(GATE in cands, "the gate rolling save is parsed as a deletion candidate")
        case(GATE in kept, "a `core, .pt.step2000` KEEP shorthand protects the gate save",
             f"kept {sorted(kept)}")

    # END TO END through the actual check: a fact citing the gate save resolves KEEP-green
    # when the listing claims it, and goes [absent] (still naming the FULL name) when it does
    # not -- proving the extraction feeds the check and a regression cannot mute it.
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "runs"))
        os.makedirs(os.path.join(root, "facts"))
        fact = {"facts": [{
            "id": "x.gate_humaneval_0912", "value": "n/a", "measured": "2026-09-12",
            "source": f"eval/humaneval_sample.py on {GATE}",
            "config": {"ckpt": GATE}, "uncertainty": "u", "status": "measured"}]}

        def listing_text(claim):
            head = "# checkpoint listing listed 2026-09-12 17:30Z\n"
            row = f"2026-09-12_17:29 2080.0 {GATE}\n"
            return head + row + (claim + "\n" if claim else "")

        lp = os.path.join(root, "runs", "pod_ckpt_candidates_20260912.txt")
        with open(os.path.join(root, "facts", "g.json"), "w") as fh:
            json.dump(fact, fh)
        with open(lp, "w") as fh:
            fh.write(listing_text(
                "# KEEP (gate): ckpt_v41_gate_0911.pt, .pt.step2000 -- HumanEval fact"))
        st_kept, _ = h.check_ckpt_facts_sources_present(root)
        case(st_kept == h.PASS, "the fact resolves KEEP-green for the claimed gate save",
             f"state {st_kept}")
        with open(lp, "w") as fh:
            fh.write(listing_text(""))
        st_absent, msg_absent = h.check_ckpt_facts_sources_present(root)
        case(st_absent == h.FAIL and GATE in msg_absent,
             "without a claim the check FAILs [absent] naming the full gate save",
             f"state {st_absent} msg {msg_absent[:120]}")


def test_gate_milestone_pin_is_run_anchored_and_resolves():
    """de-111: the gate HumanEval retention pin uses the EXISTING milestone convention.

    66 hardlinks ckpt_v41_gate_0911.pt.step2000 to
    ckpt_v41_gate_0911.milestone_he2k_step2000.pt (same dir/inode) and writes a
    runs/milestones.jsonl row {ckpt: rolling name, milestone: he2k_step2000}. The fact
    cites the ROLLING name; check_milestone_ckpt_pinned must treat it as present when the
    rolling file has rotated out but the run-anchored milestone pin exists, and FAIL only
    when both are gone. The run anchor is ckpt_v41_gate_0911 (strip .pt[.stepN]); a
    SIBLING run's pin (different anchor) must not vouch for it.
    """
    import json
    import tempfile
    h = _harness()
    RUN = "ckpt_v41_gate_0911"
    ROLLING = RUN + ".pt.step2000"
    PIN = f"{RUN}.milestone_he2k_step2000.pt"
    row = {"ckpt": ROLLING, "milestone": "he2k_step2000"}

    def world(pin_name=None, with_ckpt_glob=True):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "runs"))
        with open(os.path.join(d, "runs", "milestones.jsonl"), "w") as fh:
            fh.write(json.dumps(row) + "\n")
        if with_ckpt_glob:
            # any ckpt_*.pt satisfies the pod-only gate; create the pin or a decoy.
            open(os.path.join(d, pin_name or "ckpt_marker.pt"), "w").close()
        return d

    import shutil
    # 1) rolling file gone, run-anchored pin present -> PASS (the retention contract).
    d = world()
    open(os.path.join(d, PIN), "w").close()
    st, _ = h.check_milestone_ckpt_pinned(d)
    case(st == h.PASS, "the gate rolling save resolves via its run-anchored milestone pin",
         f"state {st}")
    shutil.rmtree(d)

    # 2) only a SIBLING run's pin exists -> FAIL (the pin is anchored to the run).
    d = world()
    open(os.path.join(d, "ckpt_other_run.milestone_he2k_step2000.pt"), "w").close()
    st, msg = h.check_milestone_ckpt_pinned(d)
    case(st == h.FAIL and RUN in msg,
         "a sibling run's milestone pin does not vouch for the gate save",
         f"state {st} msg {msg[:120]}")
    shutil.rmtree(d)

    # 3) no pin at all -> FAIL, naming the rolling ckpt and milestone.
    d = world()
    st, msg = h.check_milestone_ckpt_pinned(d)
    case(st == h.FAIL and ROLLING in msg, "with no pin the gate milestone is reported lost",
         f"state {st} msg {msg[:120]}")
    shutil.rmtree(d)

    # 4) the rolling file itself present -> PASS even with no pin.
    d = world()
    open(os.path.join(d, ROLLING), "w").close()
    st, _ = h.check_milestone_ckpt_pinned(d)
    case(st == h.PASS, "the rolling file present satisfies the row directly", f"state {st}")
    shutil.rmtree(d)


def main():
    for fn in (test_names_finds_hyphenated, test_names_still_finds_plain,
               test_embedded_name_still_not_minted, test_prose_dashes_not_swallowed,
               test_path_prefixed_citation_still_resolves,
               test_brace_expansion_with_hyphenated_core,
               test_keep_claim_on_hyphenated_name_parses,
               test_keep_shorthand_continuation_on_hyphenated_core,
               test_short_reading_still_protects_the_real_sibling,
               test_first_dot_after_a_hyphen_does_not_crash,
               test_hyphen_core_without_dot_does_not_crash,
               test_gate_rolling_save_is_seen_and_claimable,
               test_gate_milestone_pin_is_run_anchored_and_resolves,
               test_real_repo_gains_the_moe_names_and_loses_nothing):
        fn()
    bad = [r for r in RESULTS if not r[0]]
    for ok, label, detail in RESULTS:
        print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f" ({detail})" if detail and not ok else ""))
    print(f"test_ckpt_name_hyphen: {len(RESULTS) - len(bad)}/{len(RESULTS)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

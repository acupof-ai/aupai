#!/usr/bin/env python3
"""launch-gate: compute, from artifacts, whether the 500M run may start.

WHY THIS EXISTS. "All eleven gates are green" was a claim that lived only in the
controller's head. Nothing computed it. That is the same defect class this repo
spent 2026-09-01 cataloguing -- `echo DONE` printed on reaching the last line, a
coverage check that saw 27 of 36 files, a comment asserting an invariant nothing
verified -- and the launch decision was its largest instance.

THE DESIGN RULE, and it is the one that makes this different from a checklist:
every condition must be COMPUTED FROM AN ARTIFACT, never satisfied by a field
existing. `epochs_pool_source` being present is not the question; its value not
being ESTIMATED is. A recipe value being set is not the question; a provenance
row naming it is.

WHAT A GO MEANS, AND WHAT IT DOES NOT. Nine conditions computed from artifacts.
This is not a proof that the run is safe to start; it is a proof that these nine
specific failure modes are not present. Nothing checks that nine is the right
nine -- that limit, and why three ways of closing it were rejected, are in
docs/lessons/who_checks_the_gate.md. Read it before treating a GO as authority.

Usage: python scripts/launch_gate.py [--mix data/mix_500m.json] [--world 7]
       Run it WHERE THE DATA IS (the pod): a dev worktree has no data/corpus and
       no tokenizer, so gates 3 and 8 report NO-GO on absence rather than on a
       real defect.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from harness import EVIDENCE  # noqa: E402 -- per-check evidence location (fb, 2026-09-01)


def _launch_root(root):
    """The tree launch state is read from: MAIN's, not whichever worktree ran this.

    Controller rule 2026-09-01: gate state is read off main. A worktree can be ahead
    of main (a fix committed but unmerged) or behind it (a peer's fix not pulled), and
    both directions give an answer about a tree nothing will launch from. The launch
    happens from main, so main is the only tree whose state is the launch's state.

    Returns (root_to_read, note). A worktree whose HEAD differs from main is REPORTED
    rather than silently redirected -- a gate that quietly reads somewhere other than
    where it was pointed is its own defect.
    """
    try:
        import subprocess as _sp
        here = _sp.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                       text=True, timeout=20).stdout.strip()
        main = _sp.run(["git", "rev-parse", "main"], cwd=root, capture_output=True,
                       text=True, timeout=20).stdout.strip()
    except (OSError, ValueError):
        return root, "could not read git HEAD/main; gate state is this tree's"
    if not here or not main:
        return root, "not a git tree; gate state is this tree's"
    if here == main:
        return root, f"tree is at main ({here[:8]})"
    try:
        behind = _sp.run(["git", "merge-base", "--is-ancestor", "HEAD", "main"],
                         cwd=root, capture_output=True, timeout=20).returncode == 0
    except (OSError, ValueError):
        behind = False
    where = "behind" if behind else "ahead of or diverged from"
    return root, (f"WARNING: this tree ({here[:8]}) is {where} main ({main[:8]}). "
                  f"Launch state is main's. Merge main and re-run before trusting a GO")

GO, NOGO, UNKNOWN = "GO", "NO-GO", "UNKNOWN"

# The recipe values that must each trace to an artifact. Derived from run_ddp.sh's
# own flag surface rather than hand-listed, so a flag added there cannot silently
# escape provenance -- see gate 5.
RECIPE_FLAGS = ("dim", "layers", "heads", "ffn_hidden", "batch", "accum",
                "lr_scale", "grad_ckpt",
                # Added by e1-9 (2026-09-02). These four were PASSED on every launch
                # command and listed in NO gate, so the four omissions of 2026-09-02 --
                # --warmdown, --anneal_frac, --warmup, --save_every -- fell outside what
                # any check looked at. The gate written to reconcile the recipe could not
                # see the keys the incident was about.
                "warmdown", "anneal_frac", "warmup", "save_every")


def _mix(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gate_mix_file(root, mix_path, world):
    """1. The mix exists, is not launch-blocked, and agrees with its own inputs."""
    if not os.path.exists(mix_path):
        return NOGO, f"{os.path.relpath(mix_path, root)} does not exist"
    try:
        m = _mix(mix_path)
    except (OSError, ValueError) as e:
        return NOGO, f"{os.path.basename(mix_path)} is unreadable: {e}"
    blocked = m.get("_blocked")
    if blocked:
        return NOGO, (f"_blocked is non-empty ({len(blocked)} entr(ies): "
                      f"{', '.join(list(blocked)[:4])}) -- b0's UNTRUSTED_SUPPLY gate")
    if "domains" not in m or not m["domains"]:
        return NOGO, "the mix names no domains"
    return GO, f"{os.path.basename(mix_path)}: {len(m['domains'])} domains, _blocked empty"


def gate_epochs_measured(root, mix_path, world):
    """2. Every domain's epochs came from a real token cache, not a stamp estimate.

    The value, not the field. math's cap died last round on the gap between the
    stamp and the real packed row count."""
    m = _mix(mix_path)
    est = []
    missing = []
    for name, spec in m["domains"].items():
        src = spec.get("epochs_pool_source")
        flag = spec.get("epochs_pool_measured")
        if src is None and flag is None:
            missing.append(name)
        elif flag is not None:
            # A structured field, checked first. Grepping the prose for "ESTIMATED" tested a
            # WORD, not the property: writing "DERIVED from the stamp" for four domains with
            # no cache flipped this gate from NO-GO to GO with nothing measured (b0,
            # 2026-09-01, caught in my own change). A mix that carries the boolean is judged
            # on the boolean; the string stays for the reader, not the gate.
            if not flag:
                est.append(name)
        elif "ESTIMATED" in str(src).upper():
            est.append(name)
        else:
            # Neither a boolean nor the legacy marker: unreadable provenance is not a pass.
            missing.append(name)
    if missing:
        return NOGO, (f"{len(missing)} domain(s) carry no epochs_pool_source at all: "
                      f"{', '.join(missing[:4])} -- provenance absent, not merely estimated")
    if est:
        return NOGO, (f"{len(est)} domain(s) still ESTIMATED: {', '.join(est[:4])}. "
                      "epochs must be derived from the token cache before launch")
    return GO, f"all {len(m['domains'])} domains' epochs measured from a token cache"


def gate_corpora(root, mix_path, world):
    """3. Each domain's corpus dir exists with shards, filters_fp matches, and no
    domain points at a frozen mix_scale_* pool."""
    m = _mix(mix_path)
    bad = []
    for name, spec in m["domains"].items():
        d = os.path.join(root, "data", "corpus", name)
        if not os.path.isdir(d):
            bad.append(f"{name}: no data/corpus/{name}")
            continue
        if not glob.glob(os.path.join(d, "*.jsonl")):
            bad.append(f"{name}: dir exists but holds no shards")
            continue
        stats = os.path.join(d, "build_corpus_stats.json")
        if not os.path.exists(stats):
            bad.append(f"{name}: no build_corpus_stats.json, so filters_fp cannot be read")
            continue
        want = spec.get("fingerprint")
        try:
            got = json.load(open(stats, encoding="utf-8")).get("fingerprint")
        except (OSError, ValueError) as e:
            bad.append(f"{name}: stats unreadable ({e})")
            continue
        # A COMPARISON THAT DID NOT RUN IS NOT A COMPARISON THAT PASSED. `if want and
        # got != want` skips the check when the mix carries no fingerprint, and the
        # gate then prints GO having compared nothing. b0 measured the blast radius:
        # 12 of 13 mixes have at least one such domain, and mix_500m -- the launch mix
        # -- has NINE OF NINE. The gate would have certified "fingerprints match" for a
        # mix in which no fingerprint exists. Same sentence as "0 files, all
        # compliant": a universal claim over an empty set.
        #
        # Restored from da99cda for the SECOND time (de, 2026-09-01). It was lost once
        # to a merge that took one side whole, and again to a later merge that raised
        # no conflict at all -- which is why the count below says how many were
        # compared rather than how many domains exist: a number that has to move is
        # harder to lose quietly than a branch that has to run.
        if not want:
            bad.append(f"{name}: mix carries no fingerprint, so nothing was compared")
        elif not got:
            bad.append(f"{name}: build_corpus_stats.json carries no fingerprint")
        elif got != want:
            bad.append(f"{name}: fingerprint {got} != mix's {want}")
        if re.search(r"mix_scale_[\d.]+b", str(spec.get("role", "")) + name):
            bad.append(f"{name}: points at a frozen mix_scale_* pool")
    if bad:
        return NOGO, f"{len(bad)} domain(s) failed: {'; '.join(bad[:3])}"
    n = len(m["domains"])
    return GO, f"all {n} corpora present, sharded, {n} fingerprints compared and match"


ARCH_TESTS = ("scripts/test_arch_L32.py", "scripts/test_e2e.py")
# Which of them actually READ the launch data. The mix requirement below applies to these
# and only these (6e's ruling, 2026-09-03): test_arch_L32 builds models from random ids and
# opens no corpus -- every "mix" in that file is a `mixer`, the model component -- so its
# row records mix=None truthfully, and demanding it prove which data it ran on is demanding
# proof of something it does not do. A gate that asks an impossible question of an honest
# row cannot be satisfied except by a dishonest one. test_e2e is the test that tokenizes,
# so the mix half stays there, where the incident it was written for happened (de-10: e2e
# green on the sample mix while the 20B launch died at step 0 on KeyError('content')).
READS_LAUNCH_DATA = ("scripts/test_e2e.py",)


def _launch_shape_from_env():
    """The shape being launched: LAUNCH_SHAPE_JSON if set, else the 493.6M default.

    WHY AN ENV VAR AND NOT A PARAMETER (de's challenge, 2026-09-03). A `shape=None`
    parameter on gate_arch_tests would need six call sites updated in step -- the GATES
    dispatch at run(), four direct calls in this file's selftest, and two in
    launch_tests.py -- and a caller that forgets it silently gets the 493.6M shape. That
    is LAUNCH_SHAPE's present defect moved one level out, not fixed: the gate would still
    answer about a model nobody is training, and nothing would say so.

    This follows LAUNCH_MIX's route instead (de-10), which already solved the identical
    problem for the DATA half: one module-level source that the launch side sets, read by
    the gate and by the tests that write rows. test_e2e.py:48 reads E2E_MIX exactly this
    way. Same shape of fix, one level over.

    A malformed value RAISES rather than falling back. A silent fallback here would
    produce the very thing this exists to prevent -- a gate confidently comparing against
    a shape the operator did not ask for.
    """
    raw = os.environ.get("LAUNCH_SHAPE_JSON", "").strip()
    default = {"d": 1024, "layers": 32, "heads": 8, "ffn_hidden": 3072}
    if not raw:
        return default
    try:
        got = json.loads(raw)
    except ValueError as e:
        raise SystemExit(f"LAUNCH_SHAPE_JSON is not JSON: {e}. A gate that fell back to "
                         f"the default here would compare against a shape nobody asked "
                         f"for, which is the defect this variable exists to fix.")
    if not isinstance(got, dict):
        raise SystemExit(f"LAUNCH_SHAPE_JSON must be a JSON object, got {type(got).__name__}")
    missing = [k for k in default if k not in got]
    if missing:
        raise SystemExit(f"LAUNCH_SHAPE_JSON is missing {missing}: a shape that does not "
                         f"state all four keys cannot be compared against a recorded row")
    return {k: got[k] for k in default}


LAUNCH_SHAPE = _launch_shape_from_env()
# The mix being launched, beside the shape and for the same reason: launch_tests needs to
# say whether a recorded arch-test pass touched the launch DATA, and it may not hold a
# second copy of this path (_launch_shape's docstring: a second copy drifts invisibly in
# exactly the case the warning exists for). A module constant rather than the --mix default
# it used to be, so both readers and the parser take it from one place (de-10).
LAUNCH_MIX = "data/mix_500m.json"


def _cards_set(raw):
    """"5" / "0-3" / "0,2,5" -> {"5"} / {"0","1","2","3"} / {"0","2","5"}. ValueError otherwise.

    ONE parser for both sides. The request and the grant are expanded by the same code on
    purpose: two expanders is how "0-3" comes to mean four cards on one side and one key
    named "0-3" on the other, which is the divergence this whole item is about.
    """
    out = set()
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b or b - a > 63:
                raise ValueError(f"{part!r} is not a card range")
            out |= {str(i) for i in range(a, b + 1)}
        elif part.isdigit():
            out.add(part)
        else:
            raise ValueError(f"{part!r} is neither a card index nor a range")
    return out


def _launch_cards_from_env():
    """The cards this launch wants, as a sorted list of index strings, or None if unstated.

    Same route as LAUNCH_SHAPE and LAUNCH_MIX: one module-level source the launch side sets and
    the gate reads, rather than a parameter every call site must remember. None means "the
    launcher did not say", and gate_cards then answers about the BLOCK, which is the launch it
    was written for. A lane launch says LAUNCH_CARDS=5 and gets an answer about card 5.

    A malformed value RAISES for the same reason LAUNCH_SHAPE_JSON does: a gate that fell back
    here would confirm a grant for cards nobody asked for.
    """
    raw = os.environ.get("LAUNCH_CARDS", "").strip()
    if not raw:
        return None
    try:
        out = _cards_set(raw)
    except ValueError as e:
        raise SystemExit(f"LAUNCH_CARDS={raw!r}: {e}") from None
    return sorted(out, key=int) or None


LAUNCH_CARDS = _launch_cards_from_env()


def _launch_owner(root):
    """Who is launching, as the name the grant would use. LAUNCH_OWNER wins; else the
    worktree's own name, since one session per worktree is the standing rule (`aupai-de` -> de).
    """
    o = os.environ.get("LAUNCH_OWNER", "").strip()
    if o:
        return o
    base = os.path.basename(os.path.abspath(root))
    return base[len("aupai-"):] if base.startswith("aupai-") else base


def _sha256(p):
    """Chunked sha256. RAISES on a missing file, matching launch_tests._sha256 exactly.

    It used to return None first, and that branch was UNREACHABLE from both call sites:
    gate_arch_tests:236 returns NO-GO on any absent ARCH_TESTS file before the digest is
    taken, and the selftest hashes files it has just copied. Worse than dead -- if it ever
    became reachable, the `here and` in the comparison below would have SKIPPED the check
    for a file that is gone, so a deleted test would read as one whose sha still matches;
    that guard is deleted with it. Deleted
    rather than merged into launch_tests': that module lazily imports THIS one and
    documents why (a tree holding one file and not the other is one named single-file push
    away), so a module-level import here would invert the protection and stop the gate
    itself from loading. Two six-line twins with identical behaviour cost less than that.
    """
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def gate_arch_tests(root, mix_path, world):
    """4. The shape-specific arch test and the e2e test have PASSED on this shape.

    Passing is read from a recorded result, not from the file existing: a test
    that exists and was never run is the coverage-shaped nothing this gate is for.

    Three things the first version could not see, each of which produces a GO that
    means nothing (de, 2026-09-01):

    1. It accepted ANY key valued "pass". `{"ok": "pass"}` cleared it without either
       named test having run. The record must carry a row per test in ARCH_TESTS.
    2. It said "on this shape" and read no shape. Both tests run at whatever their
       config says, and test_e2e ran at Cfg.layers=12 until today -- a genuine pass of
       a different model. Each row states the shape it ran, and it must be the shape
       being launched.
    3. It could not tell a real kernel from a stand-in. test_arch_L32 prints 10/10 on a
       machine without fla, having replaced chunk_kda with a lambda; that green is the
       most misleading artifact in this directory, because it is all-clear on exactly
       the question the gate is asking. A row must say a real kernel ran.

    And a fourth, which is the same shape one field over (de-10). The row carries `mix`
    and this gate read only `shape`, so a pass on data/mix_sample.json cleared a launch
    on data/mix_500m.json. That is not hypothetical: the 20B launch died at step 0 on
    KeyError('content') from a holdout slice, and e2e had gone green because the sample
    mix's corpus dir holds zero holdout slices. record_launch_test does print
    "[NOT THE LAUNCH MIX]", but a printed warning is not a gate -- nothing reads stdout at
    launch time, and the row it wrote was accepted here unread. `mix_path` was already a
    parameter of this function and was never referenced in its body.
    """
    # The mix asked about, relative to the tree, because rows record a repo-relative path
    # while run() passes os.path.join(root, mix_rel). Falling back to LAUNCH_MIX rather
    # than to "" keeps the comparison meaningful when a caller passes no mix.
    want_mix = os.path.relpath(mix_path, root) if mix_path else LAUNCH_MIX
    missing = [n for n in ARCH_TESTS if not os.path.exists(os.path.join(root, n))]
    if missing:
        return NOGO, f"absent: {', '.join(missing)}"
    results = os.path.join(root, "runs", "launch_tests.json")
    if not os.path.exists(results):
        return UNKNOWN, ("both test files exist but runs/launch_tests.json records no "
                         "run of them on this shape -- existence is not a pass")
    try:
        r = json.load(open(results, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return NOGO, f"runs/launch_tests.json unreadable: {e}"
    # KEYED BY (test path, SHAPE) since 2026-09-04, looked up through launch_tests.rows_for
    # so the two Stage E arms coexist. Before this the key was the test path alone -- correct
    # for one launch shape at a time, and N7 Stage E runs L12 and L16 concurrently, so b0's
    # L16 certification overwrote the L12 rows and this gate then reported "ran at L16,
    # launch is L12" for an arm that HAD been certified. Absence and failure are distinct
    # outcomes (main, taken in the merge): absent means nobody ran it, a failing row means it
    # ran and failed. The first is UNKNOWN, the second NO-GO, and collapsing them loses the
    # only fact that says what to do.
    sys.path.insert(0, os.path.join(root, "scripts"))
    from launch_tests import rows_for

    found = {n: rows_for(r, n, LAUNCH_SHAPE) for n in ARCH_TESTS}
    unrecorded = [n for n in ARCH_TESTS if found[n] is None]
    if unrecorded:
        return UNKNOWN, (f"launch_tests.json records no result for {', '.join(unrecorded)} "
                         f"at {LAUNCH_SHAPE} (it has: {', '.join(sorted(r)[:4]) or 'nothing'})"
                         f" -- a record that does not name the required test at the launch "
                         f"shape is not evidence it ran")
    problems = []
    for name in ARCH_TESTS:
        row = found[name]
        if not isinstance(row, dict):
            problems.append(f"{name}: bare {row!r}, which names no shape and no kernel")
            continue
        if row.get("result") != "pass":
            problems.append(f"{name}: {row.get('result')!r}")
            continue
        shape = row.get("shape") or {}
        differs = {k: (v, shape.get(k)) for k, v in LAUNCH_SHAPE.items() if shape.get(k) != v}
        if differs:
            problems.append(f"{name}: ran at {shape or 'an unstated shape'}, "
                            f"launch is {LAUNCH_SHAPE} (differs: {sorted(differs)})")
        elif not row.get("real_kernel"):
            problems.append(f"{name}: real_kernel is not true -- a stand-in chunk_kda "
                            f"passes every case without touching a KDA kernel")
        elif name in READS_LAUNCH_DATA and row.get("mix") is None:
            # UNRECORDED and WRONG are separate problems for the same reason absence and
            # failure are separate outcomes above: a row predating the mix field cannot be
            # shown to have touched the launch data either way, and saying "ran on None"
            # would read as a mix named None.
            problems.append(f"{name}: the row records no mix, so it cannot be shown to "
                            f"have run on {want_mix} -- a pass on the sample mix is what "
                            f"this field exists to distinguish")
        elif name in READS_LAUNCH_DATA and row.get("mix") != want_mix:
            problems.append(f"{name}: ran on {row['mix']}, launch is {want_mix} -- "
                            f"the sample mix's corpus dir holds no holdout slices, so a "
                            f"pass there cannot see the launch mix's step-0 failures")
        else:
            # The row must describe the test that is here now. Without this the record
            # survives an edit to the test, which is the failure this repo has bought
            # three times over (vocab_id, .srcfp, filters_fp) -- and both of these
            # files changed three times on the day this format was written.
            want = row.get("test_sha256")
            here = _sha256(os.path.join(root, name))
            if want is None:
                problems.append(f"{name}: the row carries no test_sha256, so it cannot "
                                f"be shown to describe the file that is here now")
            elif want != here:
                problems.append(f"{name}: recorded against {want[:12]}, the file here "
                                f"is {here[:12]} -- the test changed after it passed")
    if problems:
        return NOGO, "; ".join(problems[:3])
    return GO, (f"{len(ARCH_TESTS)} shape test(s) passed at "
                f"d{LAUNCH_SHAPE['d']} L{LAUNCH_SHAPE['layers']} on a real kernel, "
                f"on {want_mix}")


CITATION = re.compile(
    # jsonl BEFORE json: regex alternation takes the first branch that matches, so
    # ".../experiments.jsonl" matched ".json" and the trailing "l" was dropped -- the gate
    # then reported "runs/experiments.json does not exist", a dead citation for a file that
    # is tracked. Every .jsonl citation in recipe_provenance was unciteable, and the first
    # two ever written (e1-9, 2026-09-02) tripped it immediately. The trailing (?![\w])
    # stops a prefix match from standing in for the whole extension.
    #
    # The optional @<rev> is the repo's OWN retirement form, not a new proposal: 30b9010
    # (44-13) deleted 22 probes and rewrote 39 refs in facts/*.json plus
    # scripts/harness.py:386,647,693 to probes/<name>.py@<sha>. All 26 distinct ones
    # resolve. This gate could not read one of them -- the pattern stopped at the
    # extension, the @sha was discarded, and the bare path then failed ls-files -- so every
    # retired citation read as dead, and telling the citer to "use an accepted spelling"
    # would have been advice with no accepted spelling behind it. Second instance of one
    # shape: the gate not recognising a reference format the repo is already using (the
    # first was jsonl eaten as json).
    #
    # The rev is [\w~^-]+ joined by single ./ -- NOT [\w./~^-]+. The greedy version
    # captured the sentence-ending period ("...py@4016bdc." -> rev "4016bdc."), which git
    # rejects, so a correctly-retired citation would have gone NO-GO on the punctuation
    # after it. Found by running the pattern on real prose, not by reading it: both
    # spellings look right in the source. Still admits 4016bdc~1, 4016bdc^2, v1.2,
    # refs/tags/v1.
    r"(?<![\w./-])((?:runs|facts|data|docs|scripts|eval|probes|datagen|bench_eff)/"
    r"[\w./-]+\.(?:jsonl|json|log|md|py|sh|txt))(?![\w])"
    r"(?:@([\w~^-]+(?:[./][\w~^-]+)*))?")


def dead_citations(root, text, tracked):
    """Which file references in one prose source cannot be followed.

    Only paths under the repo, and only ones that look like a file: a source is prose
    that may mention a run name, a person or a date, and demanding every token resolve
    would make this fire on sentences. Dead here means "named a file that is not there",
    not "is unverified".

    Two forms, one property -- can a reader reach the bytes:
      path        must be TRACKED, not merely present on disk (b0's pair review):
                  os.path.exists would pass on the pod and fail on every laptop for a
                  pod-only artifact like runs/w7_b16a2.log, and a gate whose answer
                  depends on which machine ran it is the defect this repo spent a night
                  on. Tracked is also stronger: an untracked file can vanish with no
                  commit.
      path@rev    the blob must be reachable at that rev. `git cat-file -e <rev>:<path>`
                  is the whole judgement; its three failures all return 128 -- no such
                  rev, that rev never held the path, the short sha is ambiguous. The rev
                  need NOT be where the path last existed (fb's ruling, 2026-09-02): a
                  source cites the content of one version, not the newest one. So a
                  deletion no longer kills a citation, which is what the retirement
                  convention is for.
    """
    dead = []
    for ref, rev in CITATION.findall(str(text)):
        if rev:
            if subprocess.run(["git", "cat-file", "-e", f"{rev}:{ref}"], cwd=root,
                              capture_output=True, text=True).returncode != 0:
                dead.append(f"{ref}@{rev} (no such blob in git)")
        elif ref not in tracked:
            dead.append(f"{ref} (" + ("present but untracked"
                                      if os.path.exists(os.path.join(root, ref))
                                      else "does not exist") + ")")
    return dead


def _recipe_for_shape(prov, shape=None):
    """The recipe entries for the shape being launched. Returns (mapping, why-if-None).

    runs/recipe_provenance.json used to be FLAT -- twelve flags at the top level, all
    describing the 493.6M L32 run. gate_recipe_provenance then returned GO for a 206M L12
    launch, and it was not wrong to: it asks whether each value HAS a source, not whether
    that source is about the model being launched. Twelve entries arguing for another
    shape read as twelve justified values, and scripts/test_e2e.py:59 had already written
    the hazard down without the file being able to express the distinction.

    So the file now carries prov["shapes"][<label>], and this picks the label whose
    d/layers/heads/ffn_hidden equal the shape being launched. Matching on the SHAPE, not
    on the label text: a label is a name someone typed and "206M-L12" would keep matching
    after the entries under it changed. Each group states its own dim/layers/heads/
    ffn_hidden as recipe values, so the group can be checked against LAUNCH_SHAPE using
    the group's own content.

    Flat files still work, unchanged -- one recipe, no shapes key, and the caller gets
    exactly what it got before. That is not politeness to old data: the params leg comes
    next and will add a third group, and a reader who has to migrate the file to answer a
    gate will migrate it wrongly at 3am.

    Returns None with a reason rather than falling back to the flat top level when a
    shapes key exists but no group matches. A fallback there would answer the launch with
    whatever recipe happened to be lying around at the top of the file -- the exact defect
    the partition was made to remove, restored as an error path (b0, 2026-09-03).
    """
    shape = LAUNCH_SHAPE if shape is None else shape
    groups = prov.get("shapes")
    if not isinstance(groups, dict):
        return prov, ""
    matched = []
    for label, entries in groups.items():
        if not isinstance(entries, dict):
            continue
        got = {"d": entries.get("dim"), "layers": entries.get("layers"),
               "heads": entries.get("heads"), "ffn_hidden": entries.get("ffn_hidden")}
        # The values are PROSE ("12 -- the anchor depth, NOT a choice..."), so the shape
        # is read as the leading integer of each entry. A group whose dim entry does not
        # start with a number cannot be shape-matched and is skipped rather than guessed.
        try:
            nums = {k: int(re.match(r"\s*(\d+)", str(v)).group(1)) for k, v in got.items()}
        except (AttributeError, TypeError):
            continue
        if all(nums[k] == shape[k] for k in ("d", "layers", "heads", "ffn_hidden")):
            matched.append((label, entries))
    if not matched:
        have = ", ".join(groups) or "none"
        return None, (f"runs/recipe_provenance.json has no recipe group for the shape being "
                      f"launched (d{shape['d']} L{shape['layers']} h{shape['heads']} "
                      f"ffn{shape['ffn_hidden']}); groups present: {have}. A recipe for "
                      f"another shape is not a source for this one")
    if len(matched) > 1:
        return None, (f"{len(matched)} recipe groups claim the launched shape "
                      f"({', '.join(lbl for lbl, _ in matched)}) -- which one argues for "
                      f"this run is undecidable, and picking either would be a coin flip "
                      f"dressed as provenance")
    return matched[0][1], ""


def gate_recipe_provenance(root, mix_path, world):
    """5. Every recipe value traces to an experiment row or a committed probe.

    A value being set is not provenance. This looks each flag up in
    runs/recipe_provenance.json, which must name a source per flag."""
    p = os.path.join(root, "runs", "recipe_provenance.json")
    if not os.path.exists(p):
        return NOGO, (f"no runs/recipe_provenance.json: {len(RECIPE_FLAGS)} recipe values "
                      "have no recorded source, so none of them can be justified")
    try:
        prov = json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return NOGO, f"recipe_provenance.json unreadable: {e}"
    prov, why = _recipe_for_shape(prov)
    if prov is None:
        return NOGO, why
    # A PLACEHOLDER IS NOT A SOURCE. My first version tested only that the string was
    # non-empty, and it returned GO on a schema file whose eight values were all the
    # literal "UNSOURCED" -- a file I had just written to keep this gate RED. The gate
    # written to refuse unjustified values accepted the word "unjustified" as a value.
    # Emptiness and placeholder-ness are the same fact and both must fail.
    placeholders = {"", "unsourced", "tbd", "todo", "unknown", "n/a", "none", "-"}
    unsourced = [f for f in RECIPE_FLAGS
                 if str(prov.get(f, "")).strip().lower() in placeholders]
    if unsourced:
        return NOGO, (f"{len(unsourced)} recipe value(s) with no source: "
                      f"{', '.join(unsourced)}")
    # A SOURCE THAT DOES NOT RESOLVE IS A PLACEHOLDER THAT READS AS EVIDENCE. accum cited
    # runs/w7_peak.log, which exists on neither machine (fb, 2026-09-02). The VALUE was
    # never in doubt -- 7 x 32 x 4096 = 917,504 is forced by arithmetic -- but the gate
    # passed a citation nobody could follow, which is the whole thing it checks. Same job
    # fact_refs_resolve already does for facts/<f>.json#<id>; these sources are free text,
    # so nothing was verifying them. dead_citations holds the judgement and its reasons.
    tracked = set(subprocess.run(["git", "ls-files"], cwd=root, capture_output=True,
                                 text=True).stdout.split())
    dead = [f"{f} -> {d}" for f in RECIPE_FLAGS
            for d in dead_citations(root, prov.get(f, ""), tracked)]
    if dead:
        return NOGO, (f"{len(dead)} recipe source(s) name a file git does not track: "
                      f"{'; '.join(dead[:4])} -- a citation that cannot be followed is "
                      f"not a source, however right the value is")
    return GO, f"all {len(RECIPE_FLAGS)} recipe values name a source that resolves"


def gate_memory_measured(root, mix_path, world):
    """6. A measured peak at THIS world size, not extrapolated from another.

    Two points and a straight line was nearly drawn four times on 2026-09-01."""
    p = os.path.join(root, "runs", "memory_peaks.json")
    if not os.path.exists(p):
        return NOGO, f"no runs/memory_peaks.json: no measured peak for world={world}"
    try:
        peaks = json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return NOGO, f"memory_peaks.json unreadable: {e}"
    key = str(world)
    if key not in peaks:
        # skip _schema/_filled_by metadata when reporting what IS measured, or the
        # message tells a reader we have peaks at world "_schema"
        have = ", ".join(sorted(k for k in peaks if not k.startswith("_"))) or "none"
        return NOGO, (f"no measured peak at world={world} (have: {have}). "
                      "Extrapolating from another world size is what this gate refuses")
    return GO, f"measured peak at world={world}: {peaks[key]}"


def gate_cards(root, mix_path, world):
    """7. The cards this launch wants are granted to it, by name.

    Two launches ask two different questions and the first version of this gate only ever
    answered one. A BLOCK launch asks "is the block mine": `launch_block_granted` plus
    `block_cards`. A LANE launch asks "is card N mine for this job", and the old gate returned
    GO to it off `launch_block_granted` alone -- a lane job on a card held by someone else got
    the block's grant read back as its own. Measured on the live grant 2026-09-04: cards 1 and 2
    are HELD by tileRL jobs and card 6 by a foreign occupant, and every one of those returned GO.

    So: LAUNCH_CARDS states what is wanted. Cards outside the grant refuse, naming the card and
    its recorded owner. A lane card must ALSO be granted to this launch by name, and the name is
    read from `lane_to`, a FIELD -- not from the prose. MEASURED on the live grant 2026-09-04:
    matching a session name inside `cards[N]`/`lane_note` text says the lane is de's, because
    "de" occurs inside "excluded"; adding word boundaries then says it is b0's and fb's too,
    out of "ckpt_b0_sd_equalcompute" and the granter's own name. A two-character session name
    cannot be recovered from prose at all, so the gate refuses to try. `card_claim.grant_lane`
    is the writer that sets `lane_to` with the prose, and a lane grant without it is UNKNOWN
    naming that command -- not a guess, because a guessed owner is what makes a card gate
    believable and wrong.
    """
    p = os.path.join(root, "runs", "card_assignment.json")
    if not os.path.exists(p):
        return UNKNOWN, ("no runs/card_assignment.json -- card ownership cannot be read from "
                         "an artifact and needs the controller")
    try:
        a = json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return NOGO, f"card_assignment.json unreadable: {e}"
    cards = a.get("cards") or {}
    if not isinstance(cards, dict):
        cards = {}

    def owner(idx):
        """cards[N] as prose, from an exact key or from a range key that covers N."""
        if idx in cards:
            return str(cards[idx])
        for k, v in cards.items():
            try:
                if idx in _cards_set(k) and len(_cards_set(k)) > 1:
                    return str(v)
            except ValueError:
                continue
        return ""

    try:
        block = _cards_set(a.get("block_cards") or "")
        lane = _cards_set(a.get("lane_card") or "")
    except ValueError as e:
        return NOGO, f"card_assignment.json: block_cards/lane_card unparseable: {e}"

    if LAUNCH_CARDS is None:
        if a.get("launch_block_granted"):
            return GO, (f"controller granted the block {a.get('block_cards')!r}: "
                        f"{a.get('note', '')[:60]}")
        return UNKNOWN, ("no launch_block_granted -- and LAUNCH_CARDS is unset, so this gate "
                         "cannot tell which cards to ask about. Set LAUNCH_CARDS for a lane job")

    want = set(LAUNCH_CARDS)
    shown = ",".join(sorted(want, key=int))
    granted = block | lane
    ungranted = sorted(want - granted, key=int)
    if ungranted:
        detail = "; ".join(f"card {c}: {owner(c)[:70] or 'no cards[] entry'}" for c in ungranted)
        return NOGO, (f"LAUNCH_CARDS={shown} wants card(s) {','.join(ungranted)} that "
                      f"block_cards={a.get('block_cards')!r} and lane_card="
                      f"{a.get('lane_card')!r} do not grant -- {detail}")

    in_block = sorted(want & block, key=int)
    if in_block and not a.get("launch_block_granted"):
        return NOGO, (f"LAUNCH_CARDS={shown} wants block card(s) {','.join(in_block)} but "
                      f"launch_block_granted is false")

    who = _launch_owner(root)
    in_lane = sorted(want & lane, key=int)
    lane_to = str(a.get("lane_to") or "").strip()
    if in_lane and not lane_to:
        return UNKNOWN, (f"lane card(s) {','.join(in_lane)} are granted, but card_assignment.json "
                         f"carries no lane_to, so this gate cannot say WHOSE the lane is. The "
                         f"prose does not answer it: matching a name in cards[N]/lane_note reads "
                         f"'de' out of 'excluded'. Re-grant with card_claim.py grant-lane")
    if in_lane and lane_to != who:
        detail = "; ".join(f"card {c}: {owner(c)[:60] or 'no cards[] entry'}" for c in in_lane)
        return NOGO, (f"lane card(s) {','.join(in_lane)} are granted to {lane_to!r}, not to "
                      f"{who!r} -- {detail}. A lane grant names one job; ask the controller")
    parts = []
    if in_block:
        parts.append(f"block {','.join(in_block)} granted ({a.get('note', '')[:40]})")
    if in_lane:
        parts.append(f"lane {','.join(in_lane)} granted to {lane_to} "
                     f"({str(a.get('lane_note') or owner(in_lane[0]))[:60]})")
    return GO, f"LAUNCH_CARDS={shown}: " + "; ".join(parts)


def gate_vocab_id(root, mix_path, world):
    """8. The tokenizer's identity matches what the caches were built against.

    OURS ONLY. The question is whether the checkpoints THIS REPO TRAINED share one
    vocabulary; a foreign control legitimately carries another one, and counting it made
    the gate NO-GO for the control's existence. Measured 2026-09-04: 54 checkpoints at
    0bce3584bc24f255 and one row `hf` -- pythia-160m-step2000, the Pythia-160M control
    (e1-25), which has a different vocabulary by definition and cannot be made to agree.

    The partition is the row's checkpoint name, not its vocab_id: a `ckpt_*.pt` this repo
    wrote is ours, anything else is not. Keying on the vocab_id VALUE would be circular --
    `hf` is exactly the string being excluded, so the gate would pass by declaring the
    disagreeing value uninteresting. The excluded rows are NAMED in the evidence either
    way, because a gate that silently drops rows to go green is the shape this file exists
    to avoid: an excluded row must stay readable, so a real foreign checkpoint cannot hide
    behind a control's exemption."""
    tokp = os.path.join(root, "data", "tokenizer.json")
    if not os.path.exists(tokp):
        return NOGO, "data/tokenizer.json is absent"
    ids, foreign = {}, {}
    for r in glob.glob(os.path.join(root, "runs", "score_matrix.jsonl")):
        for line in open(r, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            v = row.get("vocab_id")
            if not v:
                continue
            ck = str(row.get("ckpt") or row.get("checkpoint") or "?")
            bucket = ids if os.path.basename(ck).startswith("ckpt_") else foreign
            bucket.setdefault(v, set()).add(ck)
    excl = ""
    if foreign:
        named = sorted(c for cks in foreign.values() for c in cks)
        excl = (f"; excluded {len(named)} non-ckpt_* row(s) as foreign models, named so they "
                f"cannot hide: {', '.join(named[:4])}")
    if not ids:
        return UNKNOWN, "no vocab_id recorded on any ckpt_* row to compare the tokenizer against" + excl
    if len(ids) > 1:
        detail = "; ".join(f"{v}: {', '.join(sorted(cks)[:2])}" for v, cks in sorted(ids.items()))
        return NOGO, f"{len(ids)} distinct vocab_id among our checkpoints: {detail}{excl}"
    v, cks = next(iter(ids.items()))
    return GO, f"one vocab_id across {len(cks)} of our checkpoints: {v}{excl}"


def gate_checks_and_drift(root, mix_path, world):
    """9. harness check has 0 FAIL and the pod does not diverge from main."""
    try:
        out = subprocess.run([sys.executable, os.path.join(root, "scripts", "harness.py"),
                              "check"], capture_output=True, text=True, timeout=600, cwd=root)
    except (OSError, subprocess.TimeoutExpired) as e:
        return UNKNOWN, f"could not run harness check: {e}"
    # ASSERT PRESENCE BEFORE ASSERTING THE PROPERTY. My first version returned GO
    # whenever no [FAIL] line appeared -- and the selftest caught it reporting
    # "0 FAIL" on a world where harness.py had been DELETED. No output contains no
    # failures, so absence of bad news read as good news. Same shape as the monitor
    # that closed a row ok on log silence: silence is not evidence.
    ran = [ln for ln in out.stdout.splitlines() if re.search(r"\[\s*(PASS|FAIL|WARN|SKIP)\s*\]", ln)]
    if not ran:
        return NOGO, (f"harness check produced no check lines at all (rc={out.returncode}) -- "
                      f"nothing ran, which is not the same as nothing failed"
                      + (f": {out.stderr.strip().splitlines()[-1][:70]}" if out.stderr.strip() else ""))
    fails = [ln.strip() for ln in ran if "[FAIL]" in ln]
    if fails:
        state, why = _partition_fails(fails, _here(), len(ran))
        if state is not None:
            return state, why
    return GO, f"harness check: {len(ran)} checks ran, 0 FAIL"


# The nine, in the order they are reported. A gate added here is automatically
# covered by --selftest's broken-world requirement (see selftest below).
# WHERE EACH GATE'S TRUTH LIVES (fb 2026-09-01; ruling and measured cases in
# docs/lessons/gate_authority.md 判决4):
#   MAIN  code/config -- the launch is cut from main
#   POD   data/machine -- corpora and token caches exist nowhere else
#   BOTH  different things in each place, needs both readings
AUTHORITY = {
    "mix_file": "main", "recipe_provenance": "main", "vocab_id": "main",
    # main: the register and the recipe are both tracked files, so the reconciliation
    # answers the same on either machine. It is the pod that cannot answer -- the pod
    # has no git and no runs/experiments.jsonl of its own.
    "launch_command": "main",
    "arch_tests": "main", "cards": "main", "memory_measured": "main",
    "corpora": "pod", "epochs_measured": "pod",
    "checks_and_drift": "both",
    # main: the probe file and the mix are both tracked, and the command comes from the
    # register -- the pod holds no experiments.jsonl to read a launch line from.
    "cloze_regions": "main",
}


def _fail_name(line):
    m = re.search(r"\[\s*FAIL\s*\]\s+(\S+)", line)
    return m.group(1) if m else None


def _partition_fails(fails, here, n_ran):
    """Env-state FAILs gate on every machine; repo-scan FAILs gate only on main,
    their authority. On the pod they are UNKNOWN -- not GO (the evidence is not
    here) and not NO-GO (the tree they scanned is not main's)."""
    env = [f for f in fails if EVIDENCE.get(_fail_name(f)) == "pod"]
    scan = [f for f in fails if EVIDENCE.get(_fail_name(f)) != "pod"]
    if env:
        return NOGO, (f"{len(env)} env-state FAIL of {n_ran} checks "
                      f"(authority=pod): {env[0][:80]}")
    if scan and here == "pod":
        return UNKNOWN, (f"env-state clean; {len(scan)} repo-scan FAIL on pod-only files "
                         f"is not attributable here -- authority=main, run gate 9 there: "
                         f"{scan[0][:60]}")
    if scan:
        return NOGO, f"{len(scan)} FAIL of {n_ran} checks: {scan[0][:80]}"
    return None, ""


def _here():
    """pod or main-side. The pod is the box that holds the corpus; a dev worktree
    is not, and neither is the integration tree."""
    return "pod" if os.path.isdir("/work/aupai") and os.path.abspath(ROOT).startswith("/work/") else "main"


def reconcile_command(cmd, prov, flags=RECIPE_FLAGS):
    """Which recipe flags the launch command does not carry. Pure: no I/O, no git.

    Returns (missing, unjustified). `missing` is a justified flag absent from the
    command; `unjustified` is a flag present in the command with no entry in
    recipe_provenance -- the other direction, a knob turned without a recorded reason.

    PRESENCE, NOT VALUE. A gate that reads the resolved config instead of the command
    cannot see five of these twelve, because omitting them falls back to a Cfg default
    that HAPPENS TO EQUAL the recipe:

        key          recipe   fallback   omission visible in resolved config?
        dim          1024     1024       NO
        heads        8        8          NO
        ffn_hidden   3072     3072       NO
        batch        32       32         NO
        accum        1        1          NO
        layers       32       12         yes
        lr_scale     0.85     1.0        yes
        grad_ckpt    True     False      yes
        warmdown     0.1      0.65       yes
        anneal_frac  0.0      0.1        yes
        warmup       300      20         yes
        save_every   500      1000       yes

    So the 00:03 launch that "omitted eight values" omitted eight and only two could
    ever have been noticed -- layers (12, wrong depth) and grad_ckpt (False, OOM at
    94.87 GiB). The other six were invisible by construction. The Kubernetes docs state
    the general form: a validating webhook "cannot distinguish user-supplied from
    defaulted values", and "a dropped field is indistinguishable from success".

    Absence of a flag is the property; whether it MATTERS is a separate question this
    function does not answer. mix_500m.json has |weight - anneal| == 0 in all nine
    domains, so a missing --anneal_frac changes row order and nothing else, while a
    missing --warmdown puts 13.00B tokens in the cosine tail instead of 2.00B, a 6.5x
    error. Both are reported; severity is the caller's, and shape 28's rule holds in
    both directions -- "a parameter was omitted" does not set severity, and "the
    effective value is correct" does not prove the parameter was passed.
    """
    # `--no-X` and `--no_X` COUNT AS PRESENT (b0, 2026-09-03). train.py:1992 declares
    # grad_ckpt, attn_res, attn_res_dyn_q and fone as BooleanOptionalAction, whose whole
    # point is that absent and False are different values: absent leaves the Cfg default,
    # `--no-grad_ckpt` writes False. The old pattern's `(?<![\w-])` lookbehind rejected
    # the `--no-` form, so a command that turned a switch off EXPLICITLY was reported
    # identically to one that never mentioned it -- "justified but NOT in the command,
    # falls back to a Cfg default silently". It does not fall back; it was passed.
    #
    # Found by running the data leg's real command through this function before launch:
    # the 206M leg runs uncheckpointed, so its command carries --no-grad_ckpt, and the
    # gate called it missing. That is shape 140 exactly -- one signal for two worlds, and
    # the harmless one (explicitly off) is the one that gets the alarm, so the alarm has
    # to be waved past, which is how the real omission gets waved past with it.
    optional_prefix = r"(?:no[-_])?"
    missing, unjustified = [], []
    for f in flags:
        present = re.search(rf"(?<![\w-])--{optional_prefix}{re.escape(f)}(?![\w-])",
                            cmd) is not None
        justified = str(prov.get(f, "")).strip() != ""
        if justified and not present:
            missing.append(f)
        elif present and not justified:
            unjustified.append(f)
    return missing, unjustified


def gate_launch_command(root, mix_path, world, cmd=None):
    """10. The launch command carries every recipe value that has a recorded source.

    recipe_provenance certifies that each value HAS a source. It does not certify that
    the command passed it, and on 2026-09-02 one command missed three times running
    while all nine gates stayed green (shape 28). The flags fall back silently and the
    log prints the fallback, so nothing anywhere says a value was dropped.

    UNKNOWN, never GO, when there is no command to read: this gate's evidence is the
    argv of a launch that has not happened yet. It is written as a pure function
    (reconcile_command) so run_ddp.sh / supervise_run.sh can call it at launch -- the
    Kubernetes admission shape, refusing before the run is persisted rather than
    auditing afterwards. The audit twin stays here, reading the recorded command of a
    run already started.
    """
    p = os.path.join(root, "runs", "recipe_provenance.json")
    if not os.path.exists(p):
        return UNKNOWN, "no runs/recipe_provenance.json, so there is nothing to reconcile"
    try:
        with open(p, encoding="utf-8") as fh:
            prov = json.load(fh)
    except (OSError, ValueError) as e:
        return NOGO, f"recipe_provenance.json unreadable: {e}"
    # The SAME group gate_recipe_provenance certified, for the same reason it exists: if
    # this gate read the flat top level while that one read the launched shape's group, the
    # two would reconcile the command against different recipes and each would look
    # internally consistent (b0, 2026-09-03). One selection, one place.
    prov, why = _recipe_for_shape(prov)
    if prov is None:
        return NOGO, why
    if cmd is None:
        cmd, src = _recorded_cmd(root)
        if cmd is None:
            return UNKNOWN, (f"no launch command recorded for a running run ({src}); "
                             "this gate answers on argv, which exists at launch time")
    else:
        src = "the command given to this gate"
    missing, unjustified = reconcile_command(cmd, prov)
    if missing:
        return NOGO, (f"{len(missing)} recipe value(s) justified but NOT in the command "
                      f"({src}): {', '.join(missing)} -- each falls back to a Cfg default "
                      "silently, and the log prints the fallback")
    if unjustified:
        return GO, (f"all {len(RECIPE_FLAGS)} recipe values are in the command ({src}); "
                    f"{len(unjustified)} flag(s) passed with no recorded source: "
                    f"{', '.join(unjustified)}")
    return GO, f"all {len(RECIPE_FLAGS)} recipe values appear in the command ({src})"


def _recorded_cmd(root):
    """(cmd, source) for the run that is running, from runs/experiments.jsonl.

    FOLDED TERMINAL-WINS, through exp.fold -- the register's one reduction. This function used
    to fold by POSITION (`latest[name] = r` over the file), which is the rule exp.py:40-56 and
    harness.py:2016 both document as wrong and had already been converted away from: a union
    merge concatenates two branches' rows in whatever order it likes, and a pod pull re-appends
    a run's `running` row, so a start event landing AFTER a close reopens a finished run.

    MEASURED 2026-09-04, and it was blocking a launch rather than allowing one:
    e1_c11_doccu_rescore closed `ok` at 05:41 and a pod pull appended its `running` row after
    that close. This function then saw TWO rows claiming running (b0_se_16lnew_1b and the
    finished one) and returned None with "2 runs claim status running" -- so the gate had no
    command to check and could not clear b0's launch. Verified on one event list: terminal-wins
    gives [b0_se_16lnew_1b], position gives both.

    Also folding on (name, started) rather than name alone, which is what makes an earlier
    FAILED attempt of the same name a different run instead of a shadow of the live one.

    The except-fallback repeats terminal-wins inline rather than reverting to position, for
    harness.py:2025's reason: a missing exp.py must degrade to the correct answer, not to the
    one this change exists to delete.
    """
    p = os.path.join(root, "runs", "experiments.jsonl")
    if not os.path.exists(p):
        return None, "runs/experiments.jsonl does not exist"
    evs = []
    try:
        with open(p, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                if r.get("name"):
                    evs.append(r)
    except (OSError, ValueError) as e:
        return None, f"runs/experiments.jsonl unreadable: {e}"
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
        from exp import fold
        folded = list(fold(evs))
    except Exception:
        out = {}
        for r in evs:
            key = (r.get("name"), r.get("started"))
            prev = out.get(key)
            if (prev is not None and prev.get("status") != "running"
                    and r.get("status") == "running"):
                continue
            if (prev is not None and prev.get("status") == "retracted"
                    and r.get("status") != "retracted"):
                continue
            out[key] = r
        folded = list(out.values())
    running = [r for r in folded if r.get("status") == "running" and r.get("cmd")]
    if not running:
        return None, "no row with status running and a cmd field"
    if len(running) > 1:
        names = ", ".join(sorted(r["name"] for r in running))
        return None, f"{len(running)} runs claim status running ({names})"
    return running[0]["cmd"], f"runs/experiments.jsonl, {running[0]['name']}"


def gate_cloze_regions(root, mix_path, world, cmd=None):
    """11. A memory arm may not read into the cloze probe's UNSEEN region.

    Readout 2 compares what the control read against what it never read, and both regions are
    row indices into ONE cache ordering. UNSEEN = pool [alloc, len(pool)) where alloc is the
    phase allocation int(total_rows x weight) -- unread by CONSTRUCTION, because the plan never
    allocates past it. An arm that allocates more, or runs longer, starts reading into the
    region the probe calls never-seen, and the readout silently becomes seen-vs-seen.

    Three conditions, and only the first is load-bearing for UNSEEN (e1, measured):

      mix name   the header records the mix the regions were computed against. A later stage
                 pointed at the same cache with a different mix is the real residual risk:
                 nothing else in the file notices, because the cache and its fingerprint are
                 unchanged -- only the ALLOCATION moves.
      alloc      the boundary itself. int(total_tokens/seq x weight) must equal the probe's
                 unseen_lo. This is a statement about the MIX, not about the run: the plan is
                 built for the WHOLE budget (train.py:1692 rows = total_tokens/seq, :1798
                 want = int(rows x frac x weight) -- no step term anywhere), so stopping early
                 consumes a PREFIX of that plan. On mix_200m_8b starcoder gets 643,969 rows =
                 0.301 epochs of its 2,139,719-row pool, and pool [643969, 2139719) is never
                 allocated to ANY run on this mix, wherever it stops.
      seed       does not protect UNSEEN at all -- UNSEEN sits past the allocation under every
                 seed. It protects SEEN: the plan's shuffle is seeded, so an arm at another
                 seed reads different rows INSIDE the same allocation and SEEN stops describing
                 what it trained on. That breaks delta_seen, not delta_unseen.

    NO max_steps TERM. The first version had one, on the reasoning that a longer run reads
    further; it does not read PAST the allocation, because the allocation never depended on
    the step count. A longer arm reads more of SEEN, which is a different property -- worth
    recording, not refusing, since the arms are meant to run the control's line unchanged.

    The boundary is READ FROM THE ITEM FILE's header, never hardcoded. A constant here and a
    number in the file are two copies of one quantity, and the failure this gate exists to
    prevent is precisely the two drifting apart (the first item file was built on a wrong
    boundary and looked internally consistent).

    UNKNOWN, not GO, when there is no probe file or no command: this gate answers about a
    launch's argv against a registered probe, and a missing probe means the question has no
    subject -- a GO there would certify a comparison nobody has defined yet.
    """
    # BOTH PATHS, newest location first. The file moved data/eval/ -> data/probes/ on
    # 2026-09-05 (32b4ed22) because everything under data/eval/ must be in holdout.py's
    # REGISTRY, and REGISTRY drives EXCLUSION -- registering a probe drawn FROM the corpus
    # on purpose would have deleted what it measures at the next corpus build. This gate
    # hardcoded the old path and went UNKNOWN on every launch the moment it moved: UNKNOWN
    # does not refuse, so the gate was off and said so in a line nobody reads at launch.
    # A path is a join between two files that no check owns; carrying both is the cheap fix.
    probe = None
    for _rel in (os.path.join("data", "probes", "api_cloze.jsonl"),
                 os.path.join("data", "eval", "api_cloze.jsonl")):
        if os.path.exists(os.path.join(root, _rel)):
            probe = os.path.join(root, _rel)
            break
    if probe is None:
        return UNKNOWN, ("no api_cloze.jsonl under data/probes/ or data/eval/: no registered "
                         "cloze probe, so no region for a launch to violate")
    try:
        with open(probe, encoding="utf-8") as fh:
            head = json.loads(fh.readline())
    except (OSError, ValueError) as e:
        return NOGO, f"{os.path.relpath(probe, root)} header unreadable: {e}"
    # READ FROM `bounds`, WHICH IS WHERE THE FILE PUTS THEM. The first version of this gate
    # read unseen_lo/seed/mix off the header's top level -- a shape I assumed rather than
    # opened, and the real file nests them under `bounds` (top level carries domain, chance,
    # regions, n_seen_rows, stats). It went NO-GO on every launch with "header lacks
    # unseen_lo, seed", which is the safe direction but for the wrong reason: a gate refusing
    # because it cannot read its own subject would have been switched off by the first person
    # to hit it. Top level is still consulted as a fallback so a future flattening does not
    # break it again.
    b = head.get("bounds") or {}
    def _f(k):
        return b.get(k, head.get(k))
    need = {"domain": head.get("domain"), "unseen_lo": _f("unseen_lo"), "seed": _f("seed")}
    missing_hdr = [k for k, v in need.items() if v is None]
    if missing_hdr:
        return NOGO, (f"cloze header lacks {', '.join(missing_hdr)}: the boundary this gate "
                      f"enforces is not stated in the file it protects")
    dom, unseen_lo, want_seed = need["domain"], int(need["unseen_lo"]), int(need["seed"])
    if cmd is None:
        cmd, src = _recorded_cmd(root)
        if cmd is None:
            return UNKNOWN, (f"no launch command recorded ({src}); this gate answers on argv")
    else:
        src = "the command given to this gate"
    # A LINE THAT IS NOT A TRAINING LAUNCH HAS NONE OF THIS GATE'S TERMS, and answering it
    # is a category error in both directions. GO would certify a line the gate never read
    # (not-red is not green); NO-GO refuses legitimate work -- measured on e1's scoring run
    # `python3 eval/api_cloze.py --ckpt ... --device cuda:0 --json`, which this gate refused
    # for lacking --seed, a flag a scoring run has no business carrying. A gate that blocks
    # a correct command is one that gets switched off, so it must say "not my subject"
    # instead. The tell is the mix: a training launch names one, on the line or as the
    # gate's own mix_path argument, and a scorer names a checkpoint and a probe.
    if not re.search(r"--mix\s+\S+", cmd) and re.search(r"--ckpt\s+\S+", cmd):
        return UNKNOWN, (f"not a training launch ({src}): no --mix, and --ckpt present -- this "
                         f"reads as a scoring/eval command, which carries none of this gate's "
                         f"terms (mix, allocation, seed). The line this gate must see is the "
                         f"ARM's training launch, not a run that scores its checkpoint")
    m = re.search(r"--mix\s+(\S+)", cmd)
    launch_mix = m.group(1) if m else mix_path
    try:
        mix = json.load(open(os.path.join(root, launch_mix), encoding="utf-8"))
    except (OSError, ValueError) as e:
        return NOGO, f"launch mix {launch_mix} unreadable: {e}"
    d = mix.get("domains", {}).get(dom)
    if d is None:
        return NOGO, (f"{launch_mix} does not name {dom}, the domain the cloze regions index; "
                      f"the probe cannot describe this run's data")
    seq = int(_f("seq") or 4096)
    alloc = int((mix["total_tokens"] / seq) * d["weight"])
    bad = []
    if alloc != unseen_lo:
        bad.append(f"allocation int(rows x weight)={alloc:,} != the probe's unseen_lo="
                   f"{unseen_lo:,}, so this launch reads {alloc - unseen_lo:,} rows into UNSEEN"
                   if alloc > unseen_lo else
                   f"allocation int(rows x weight)={alloc:,} != the probe's unseen_lo="
                   f"{unseen_lo:,}: the boundary moved, so the item file's row indices no "
                   f"longer mean what its header says")
    # The header stores a BASENAME ("mix_200m_8b.json"); the launch line carries a path
    # ("data/mix_200m_8b.json"). Compare basenames or every launch fails a name check on a
    # difference that is not one.
    want_mix = _f("mix")
    if want_mix and os.path.basename(want_mix) != os.path.basename(launch_mix):
        bad.append(f"--mix {launch_mix} != the probe's {want_mix}: the regions were computed "
                   f"against that mix, and another one moves the allocation while the cache "
                   f"and its fingerprint stay identical -- nothing else would notice")
    sd = re.search(r"--seed\s+(\d+)", cmd)
    seed = int(sd.group(1)) if sd else None
    if seed != want_seed:
        bad.append(f"--seed {seed} != the probe's {want_seed}: the plan shuffle is seeded, so "
                   f"this arm reads DIFFERENT rows inside the same allocation and SEEN stops "
                   f"describing what it trained on (delta_seen, not delta_unseen)")
    if bad:
        return NOGO, f"cloze regions ({src}): " + "; ".join(bad)
    return GO, (f"cloze regions hold ({src}): {dom} alloc {alloc:,} == unseen_lo on "
                f"{launch_mix}, --seed {seed}")


GATES = [
    ("mix_file", gate_mix_file),
    ("epochs_measured", gate_epochs_measured),
    ("corpora", gate_corpora),
    ("arch_tests", gate_arch_tests),
    ("recipe_provenance", gate_recipe_provenance),
    ("memory_measured", gate_memory_measured),
    ("cards", gate_cards),
    ("vocab_id", gate_vocab_id),
    ("checks_and_drift", gate_checks_and_drift),
    ("launch_command", gate_launch_command),
    ("cloze_regions", gate_cloze_regions),
]


def run(root, mix_path, world, here=None):
    """Each gate runs only where its answer means something.

    A gate asked in the wrong place returns UNKNOWN naming the right place --
    NOT a NO-GO and not a GO. Both of those get believed, and a believable answer
    from a filesystem that cannot hold the evidence is worse than no answer.
    """
    here = here or _here()
    rows = []
    for name, fn in GATES:
        auth = AUTHORITY.get(name, "main")
        if auth not in (here, "both"):
            rows.append((name, UNKNOWN,
                         f"not readable here ({here}); this gate's evidence lives on "
                         f"{auth} -- run it there"))
            continue
        try:
            state, why = fn(root, mix_path, world)
        except Exception as e:  # a gate that crashes is NOT a pass
            state, why = NOGO, f"the gate itself raised: {type(e).__name__}: {e}"
        if auth == "both":
            why = f"[{here}] {why}"
        rows.append((name, state, why))
    return rows


def pod_attribution(root):
    """A pod verdict is attributable to main only when the pod's manifested tree
    IS main's tree: check_pod clean (drift=0). Drift alone is the right quantity
    -- it compares exactly the files the launch is cut from.

    Unregistered .py are deliberately NOT a refusal condition (fb, 2026-09-01,
    correcting this function's first version). The 229 files the manifest does
    not name split into 51 main holds unpushed, 178 pod-only (10 ever in git),
    and 168 that never entered git at all -- one-off scripts written directly on
    the pod over months. A one-off in the pod root says nothing about whether the
    training code is main's code, and refusing on their count would hang the
    launch on the pod's housekeeping. The deeper fix is the AUTHORITY cut in
    _partition_fails: repo-scan checks answer on main, where those files do not
    exist. A verdict from a DRIFTED tree is still noise, so drift != 0 refuses."""
    from pod_drift import check_pod
    ok, msg = check_pod(root)
    if not ok:
        return False, f"pod drifted ({msg})"
    return True, msg


def main():
    for _k in [k for k in os.environ if k.startswith("GIT_")]:
        os.environ.pop(_k)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default=os.path.join(ROOT, LAUNCH_MIX))
    ap.add_argument("--world", type=int, default=7)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())

    here = _here()
    root, note = (ROOT, f"running on the pod ({ROOT})") if here == "pod" else _launch_root(ROOT)
    if here == "pod":
        ok, why = pod_attribution(root)
        if not ok:
            print(f"REFUSING to print a verdict: {why}")
            print("          A verdict from an unattributable tree is noise, not a gate.")
            return 1
    rows = run(root, a.mix, a.world, here)
    elsewhere = sorted(n for n, _ in GATES if AUTHORITY.get(n, "main") not in (here, "both"))
    print(f"launch-gate  mix={os.path.relpath(a.mix, ROOT)}  world={a.world}  here={here}")
    print(f"             {note}")
    if here == "main":
        print("             repo-scan-only: env-state gates answer on the pod, not here")
    if elsewhere:
        print(f"             {len(elsewhere)} gate(s) answerable only elsewhere: "
              f"{', '.join(elsewhere)}")
    print()
    for name, state, why in rows:
        print(f"  [{state:^7}] {name:<20} {why}")
    blocking = [r for r in rows if r[1] != GO]
    print()
    if blocking:
        print(f"NO-GO: {len(blocking)} of {len(rows)} gate(s) not GO")
        for name, state, why in blocking:
            print(f"  {state}: {name} -- {why}")
        return 1
    if note.startswith("WARNING"):
        print("REFUSING to print GO: " + note)
        return 1
    if elsewhere:
        # A GO computed where half the gates could not run is the exact failure fb
        # caught: two locations each reporting a believable half of the world.
        print(f"REFUSING to print GO: {len(elsewhere)} gate(s) could not be read here "
              f"({', '.join(elsewhere)}). A full GO requires a main run AND a pod run.")
        return 1
    print(f"GO: all {len(rows)} gates computed GO from artifacts.")
    print("     This is not a proof the run is safe -- it is a proof that these nine")
    print("     failure modes are absent. See docs/lessons/who_checks_the_gate.md.")
    return 0


def selftest():
    """Every gate must FAIL on a broken world built by damaging a REAL artifact.

    Hand-written worlds share the check's assumptions -- 2026-09-01 produced two
    cases where a fixture and the code it tested believed the same fiction. So
    each world here copies the real tree and breaks one thing in it.

    The loop is over GATES, not a hand-listed set: a gate added without a broken
    world fails this selftest rather than passing silently. That is the one
    structural defence against this file becoming the thing it was written to
    replace -- see WHO_CHECKS_THE_GATE in docs.
    """
    import atexit
    import shutil
    import subprocess
    import tempfile

    tracked = subprocess.run(["git", "-C", ROOT, "ls-files", "data", "runs", "scripts"],
                             capture_output=True, text=True).stdout.split("\n")
    tracked = [p for p in tracked if p.strip()]
    if not tracked:
        print("SKIP: git ls-files returned nothing, so a world would be empty")
        return 0

    worlds = []
    atexit.register(lambda: [shutil.rmtree(w, ignore_errors=True) for w in worlds])

    def world(mutate):
        # Tracked files only: 12 MB against 1.5 GB for the whole of data/, and every
        # artifact a gate reads is committed. The pattern-based copytree that stood here
        # excluded corpus/ and _* but not data/*.jsonl, so each world carried 1.5 GB of
        # SFT corpora no gate opens; 7886 undeleted worlds filled the disk to 2 GB free
        # and broke two sessions' selftests (2026-09-01).
        d = tempfile.mkdtemp(prefix="gate_world_")
        worlds.append(d)
        for rel in tracked:
            dst = os.path.join(d, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # copy2, not copy: copy() drops the mode, and pod_drift's gate compares mode
            # as of b0-19 -- so a world built with copy() showed 15 files whose content
            # matched and whose exec bit did not, and pod_attribution read that as "the
            # pod drifted" in a world built to be clean. A fixture must carry the property
            # the gate under test reads, or it tests the fixture's own defect.
            shutil.copy2(os.path.join(ROOT, rel), dst)
        os.makedirs(os.path.join(d, "data", "corpus"), exist_ok=True)
        mutate(d)
        return d

    mix_rel = os.path.join("data", "mix_30b_stage2.json")
    real_mix = os.path.join(ROOT, mix_rel)
    if not os.path.exists(real_mix):
        print("SKIP: no real mix to damage")
        return 0

    def write_mix(d, fn):
        m = json.load(open(real_mix, encoding="utf-8"))
        fn(m)
        p = os.path.join(d, mix_rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(m, open(p, "w", encoding="utf-8"), ensure_ascii=False)
        return p

    broken = {}
    # 1: the mix is launch-blocked
    d = world(lambda d: write_mix(d, lambda m: m.__setitem__("_blocked", {"cot": "no supply"})))
    broken["mix_file"] = (d, os.path.join(d, mix_rel))
    # 2: one domain reverts to ESTIMATED
    def _est(m):
        k = next(iter(m["domains"]))
        m["domains"][k]["epochs_pool_source"] = "stamp (ESTIMATED)"
    d = world(lambda d: write_mix(d, _est))
    broken["epochs_measured"] = (d, os.path.join(d, mix_rel))
    # 3: corpora absent (the copy excludes data/corpus by design)
    d = world(lambda d: write_mix(d, lambda m: None))
    broken["corpora"] = (d, os.path.join(d, mix_rel))
    # 4/5/6/7: the recording artifacts removed from a real copy
    # recipe_provenance gets TWO worlds: the file missing, and the file present with
    # placeholder values. The second is the one my first version passed -- it returned
    # GO on eight literal "UNSOURCED" strings, because it tested non-emptiness rather
    # than sourced-ness. A gate that accepts the word "unjustified" as a justification
    # needs the world where that is the input, not only the world where the file is gone.
    def _placeholders(d):
        write_mix(d, lambda m: None)
        os.makedirs(os.path.join(d, "runs"), exist_ok=True)
        json.dump({f: "UNSOURCED" for f in RECIPE_FLAGS},
                  open(os.path.join(d, "runs", "recipe_provenance.json"), "w",
                       encoding="utf-8"))
    dph = world(_placeholders)
    st, why = gate_recipe_provenance(dph, os.path.join(dph, mix_rel), 7)
    assert st != GO, f"placeholder provenance must not pass: {why}"

    for gate, fname in (("arch_tests", "launch_tests.json"),
                        ("recipe_provenance", "recipe_provenance.json"),
                        ("memory_measured", "memory_peaks.json"),
                        ("cards", "card_assignment.json")):
        def _rm(d, fname=fname):
            write_mix(d, lambda m: None)
            p = os.path.join(d, "runs", fname)
            if os.path.exists(p):
                os.remove(p)
        d = world(_rm)
        broken[gate] = (d, os.path.join(d, mix_rel))
    # 8: two vocab_ids in the ledger
    def _two(d):
        write_mix(d, lambda m: None)
        # The tokenizer first, or the gate returns "tokenizer.json is absent" at its
        # first line and never reaches the two ids. data/tokenizer.json is gitignored,
        # so it is missing from every copied world -- this world was failing on absence
        # and certifying nothing, which the reason check caught the moment it came back
        # (de, 2026-09-01). Contents are irrelevant: the gate only tests existence.
        with open(os.path.join(d, "data", "tokenizer.json"), "w", encoding="utf-8") as f:
            f.write('{"model":{"vocab":{}}}')
        p = os.path.join(d, "runs", "score_matrix.jsonl")
        with open(p, "a", encoding="utf-8") as f:
            # ckpt_-PREFIXED, both of them. The gate partitions by whether the row's
            # checkpoint is ours (2026-09-04), so the old names `x.pt` and `y.pt` were both
            # foreign and this world went GREEN with two vocab_ids in it -- a world whose
            # mutation the gate is now right to ignore, which certifies nothing. The
            # property is "OUR checkpoints disagree", so the fixture has to be ours.
            f.write(json.dumps({"ckpt": "ckpt_x.pt", "vocab_id": "0" * 16}) + "\n")
            f.write(json.dumps({"ckpt": "ckpt_y.pt", "vocab_id": "1" * 16}) + "\n")
    d = world(_two)
    broken["vocab_id"] = (d, os.path.join(d, mix_rel))
    # 9: harness.py removed, so the check cannot report 0 FAIL
    def _noharness(d):
        write_mix(d, lambda m: None)
        p = os.path.join(d, "scripts", "harness.py")
        if os.path.exists(p):
            os.remove(p)
    d = world(_noharness)
    broken["checks_and_drift"] = (d, os.path.join(d, mix_rel))

    # launch_command: the register carries the run's own 00:03 attempt, which omitted
    # --warmdown --anneal_frac --warmup --save_every and was NOT recoverable from any
    # reconstruction -- it is the verbatim cmd field of a real row. Written as the only
    # running row so _recorded_cmd finds it.
    def _badcmd(d):
        write_mix(d, lambda m: None)
        os.makedirs(os.path.join(d, "runs"), exist_ok=True)
        with open(os.path.join(d, "runs", "experiments.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "name": "p500m_20b_0902", "status": "running",
                "cmd": ("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NGPU=8 ./run_ddp.sh "
                        "--mix data/mix_500m.json --name p500m_20b_0902 --dim 1024 "
                        "--layers 32 --heads 8 --ffn_hidden 3072 --batch 32 --accum 1 "
                        "--grad_ckpt --lr_scale 0.85"),
            }) + "\n")
    d = world(_badcmd)
    broken["launch_command"] = (d, os.path.join(d, mix_rel))

    # cloze_regions: the probe file states the boundary, and the world nudges the MIX's
    # starcoder weight so the launch allocates past it. Mutating the mix rather than the
    # header is the direction that matters -- a real arm violates this by changing its own
    # composition, not by editing the probe. The running row carries the control's line so
    # _recorded_cmd has a command to read.
    #
    # This world carries mix_200m_8b.json, NOT the shared mix_30b_stage2 the other worlds
    # use: the probe indexes code_py_starcoder, which mix_30b_stage2 does not name, and a
    # world whose mix lacks the domain would exercise the "domain absent" branch instead of
    # the boundary one -- passing for the wrong reason.
    _CLOZE_MIX = os.path.join("data", "mix_200m_8b.json")

    def _badcloze(dd):
        _src = os.path.join(ROOT, _CLOZE_MIX)
        _dst = os.path.join(dd, _CLOZE_MIX)
        os.makedirs(os.path.dirname(_dst), exist_ok=True)
        shutil.copy2(_src, _dst)
        _w0 = json.load(open(_src, encoding="utf-8"))["domains"]["code_py_starcoder"]["weight"]
        _m = json.load(open(_dst, encoding="utf-8"))
        _unseen_lo = int((_m["total_tokens"] / 4096) * _w0)   # from the UNMUTATED weight
        _m["domains"]["code_py_starcoder"]["weight"] = _w0 * 1.01
        json.dump(_m, open(_dst, "w"))
        os.makedirs(os.path.join(dd, "data", "probes"), exist_ok=True)
        with open(os.path.join(dd, "data", "probes", "api_cloze.jsonl"), "w",
                  encoding="utf-8") as f:
            # nested under `bounds`, the shape the real file ships
            f.write(json.dumps({"_header": True, "domain": "code_py_starcoder",
                                "n_seen_rows": 80280,
                                "bounds": {"seq": 4096, "seed": 42, "unseen_lo": _unseen_lo,
                                           "mix": os.path.basename(_CLOZE_MIX)}}) + "\n")
        os.makedirs(os.path.join(dd, "runs"), exist_ok=True)
        with open(os.path.join(dd, "runs", "experiments.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "name": "memory_arm_M1", "status": "running",
                "cmd": (f"NGPU=2 ./run_ddp.sh --name memory_arm_M1 --mix {_CLOZE_MIX} "
                        "--batch 16 --accum 2 --max_steps 3815 --seed 42"),
            }) + "\n")
    d = world(_badcloze)
    broken["cloze_regions"] = (d, os.path.join(d, _CLOZE_MIX))

    ungated = [n for n, _ in GATES if n not in broken]
    assert not ungated, (
        "gate(s) with no broken world: " + ", ".join(ungated) +
        " -- a gate nobody can make fail is the shape this file exists to retire")

    # arch_tests: three worlds the missing-file world cannot reach. Each one CLEARED the
    # gate before 2026-09-01, and each is a GO that means nothing rather than a crash --
    # which is why the file-absent world was not enough to find them. Written as records
    # because a record is the artifact the gate reads; there is nothing else to damage.
    for label, record in (
        ("a key that is neither named test",
         {"ok": "pass", "shape": LAUNCH_SHAPE}),
        # DERIVED from the launch shape, not the literal 12 it used to be (de's
        # challenge, 2026-09-03). Once the shape is settable, `layers=12` is a CORRECT
        # record for an L12 launch -- the gate must then say GO, this world's
        # `assert st != GO` goes red, and the red means THE WORLD DIED rather than the
        # gate stopped comparing. One signal for two opposite facts is the family this
        # repo keeps paying for. layers+1 is wrong at every launch shape by construction.
        ("both tests passing at the WRONG shape",
         {n: {"result": "pass",
              "shape": dict(LAUNCH_SHAPE, layers=LAUNCH_SHAPE["layers"] + 1),
              "real_kernel": True}
          for n in ARCH_TESTS}),
        ("both tests passing against a STAND-IN kernel",
         {n: {"result": "pass", "shape": dict(LAUNCH_SHAPE), "real_kernel": False}
          for n in ARCH_TESTS}),
        # The record the old gate actually accepted. The two dict worlds above fail the
        # old gate too, but for the wrong reason -- it could not parse a dict row at all,
        # so its NO-GO said nothing about shape or kernel. This one is the honest
        # discriminator: {name: "pass"} cleared the old gate (GO, "2 shape test(s)
        # recorded pass") while naming neither the shape nor whether a kernel ran.
        ("a flat record naming no shape and no kernel",
         {n: "pass" for n in ARCH_TESTS}),
    ):
        def _rec(d, record=record):
            write_mix(d, lambda m: None)
            os.makedirs(os.path.join(d, "runs"), exist_ok=True)
            with open(os.path.join(d, "runs", "launch_tests.json"), "w",
                      encoding="utf-8") as f:
                json.dump(record, f)
        dw = world(_rec)
        st, why = gate_arch_tests(dw, os.path.join(dw, mix_rel), 7)
        assert st != GO, f"arch_tests passed on {label}: {why}"
    # and the record it is meant to accept must actually pass, or the gate is just
    # a refusal wearing three reasons
    def _good(d):
        write_mix(d, lambda m: None)
        os.makedirs(os.path.join(d, "runs"), exist_ok=True)
        with open(os.path.join(d, "runs", "launch_tests.json"), "w", encoding="utf-8") as f:
            json.dump({n: {"result": "pass", "shape": dict(LAUNCH_SHAPE),
                           "real_kernel": True, "mix": mix_rel,
                           "test_sha256": _sha256(os.path.join(d, n))}
                       for n in ARCH_TESTS}, f)
    dg = world(_good)
    st, why = gate_arch_tests(dg, os.path.join(dg, mix_rel), 7)
    assert st == GO, f"arch_tests refuses the record it is written to accept: {why}"
    assert mix_rel in why, (f"the GO line does not name the mix it cleared: {why!r}. A gate "
                            f"that reads a field must say what it read, or a reader cannot "
                            f"tell this version from the one that ignored it")

    # THE SHAPE IS ACTUALLY READ FROM LAUNCH_SHAPE -- positive evidence, not the absence
    # of a failure (de's second criterion, 2026-09-03). ONE record, TWO launch shapes:
    # it must clear the shape it names and be refused by the other. The negative world
    # above cannot show this: it stays red whether the gate compares shapes or has
    # stopped reading the field, so it certifies nothing about WHERE the expected shape
    # comes from. Both calls run against the same tree and the same file, so the only
    # thing that differs is LAUNCH_SHAPE.
    #
    # WHAT THIS CATCHES THAT NOTHING ELSE DOES, measured on two mutants: a gate that
    # compares shapes but reads a HARD-CODED expected shape stays green under every
    # negative world above, and that is exactly what the first version of this fix would
    # have shipped -- a `shape=None` parameter threaded through nine call sites, where any
    # caller omitting it silently gets L32. The negative worlds establish THAT a
    # comparison happens; only this one establishes WHERE its right-hand side comes from.
    # That gap is why 1e withdrew their first criterion and why de's replacement is
    # stronger: the withdrawn one would have passed the broken version.
    _saved = dict(LAUNCH_SHAPE)
    try:
        other = dict(_saved, layers=_saved["layers"] + 1)
        LAUNCH_SHAPE.clear(), LAUNCH_SHAPE.update(other)
        st_other, why_other = gate_arch_tests(dg, os.path.join(dg, mix_rel), 7)
        assert st_other != GO, (
            f"the record cleared a DIFFERENT launch shape too (recorded L{_saved['layers']}, "
            f"launch L{other['layers']}): the gate is not reading LAUNCH_SHAPE at all, so a "
            f"pass here says nothing about the model being trained. why={why_other!r}")
        assert f"L{other['layers']}" in why_other or str(other["layers"]) in why_other, (
            f"the refusal does not name the shape it wanted, so an operator cannot see "
            f"which shape was compared: {why_other!r}")
    finally:
        LAUNCH_SHAPE.clear(), LAUNCH_SHAPE.update(_saved)
    st_back, _ = gate_arch_tests(dg, os.path.join(dg, mix_rel), 7)
    assert st_back == GO, (
        "restoring LAUNCH_SHAPE did not restore the GO -- the two calls above were not "
        "measuring the shape, and every assertion in this block is about something else")

    # THE MIX EXEMPTION IS NARROW (6e's ruling, 2026-09-03). A test that reads no corpus
    # records mix=None truthfully and must not be asked to prove which data it ran on --
    # but exempting it from the MIX must not exempt it from anything else. My first version
    # wrote `elif name not in READS_LAUNCH_DATA: pass`, which fell past the else branch and
    # silently dropped the test_sha256 check for that test: a data-free test would no longer
    # be pinned to its own content, and nothing would have said so. Both halves asserted
    # here, because the hole was invisible in the passing case.
    def _mixworld(d, over):
        write_mix(d, lambda m: None)
        os.makedirs(os.path.join(d, "runs"), exist_ok=True)
        rows = {}
        for n in ARCH_TESTS:
            r = {"result": "pass", "real_kernel": True, "shape": dict(LAUNCH_SHAPE),
                 "test_sha256": _sha256(os.path.join(d, n)),
                 "mix": mix_rel if n in READS_LAUNCH_DATA else None}
            r.update(over.get(n, {}))
            rows[n] = r
        with open(os.path.join(d, "runs", "launch_tests.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f)

    dx = world(lambda d: _mixworld(d, {}))
    st, why = gate_arch_tests(dx, os.path.join(dx, mix_rel), 7)
    assert st == GO, f"a data-free test's mix=None must not refuse the gate: {why}"
    # ...and the exemption reaches ONLY the mix.
    arch = [n for n in ARCH_TESTS if n not in READS_LAUNCH_DATA]
    assert arch, "no data-free arch test to check the exemption's width with"
    dy = world(lambda d: _mixworld(d, {arch[0]: {"test_sha256": "0" * 64}}))
    st, why = gate_arch_tests(dy, os.path.join(dy, mix_rel), 7)
    assert st != GO, (
        f"{arch[0]} is exempt from the mix requirement, and its test_sha256 went unchecked "
        f"too -- the exemption widened past what it was written for: {why}")
    assert "test changed after it passed" in why, (
        f"refused, but not for the tampered sha, so this world proves nothing about the "
        f"exemption's width: {why}")

    # THE MIX HALF (de-10), and the first world is the incident: e2e passed on the sample
    # mix while the launch mix died at step 0 on KeyError('content'), because the sample
    # mix's corpus dir holds zero holdout slices. Both worlds mutate the record the gate
    # is written to ACCEPT, so the only thing that differs is the field under test -- an
    # empty tree would fail for the absence of everything and prove nothing (de's rule).
    # MUTATE THE TEST THAT READS DATA, not ARCH_TESTS[0]. It was [0] -- test_arch_L32 --
    # and once that test became exempt from the mix requirement (it reads no corpus), both
    # worlds went GREEN: they were mutating a field the gate no longer reads for that test,
    # so they asserted nothing while looking untouched. READS_LAUNCH_DATA[0] is the test
    # whose mix the gate does read, which is what these two worlds were always about.
    _mixed_test = READS_LAUNCH_DATA[0]
    for label, mutate, want_phrase in (
        ("a pass recorded on the sample mix", lambda rows: rows[_mixed_test].update(
            {"mix": "data/mix_sample.json"}), "data/mix_sample.json"),
        ("a row predating the mix field", lambda rows: rows[_mixed_test].pop("mix"),
         "records no mix"),
    ):
        def _mixed(d, mutate=mutate):
            _good(d)
            p = os.path.join(d, "runs", "launch_tests.json")
            rows = json.load(open(p, encoding="utf-8"))
            mutate(rows)
            json.dump(rows, open(p, "w", encoding="utf-8"))
        dm = world(_mixed)
        st, why = gate_arch_tests(dm, os.path.join(dm, mix_rel), 7)
        assert st != GO, f"arch_tests passed on {label}: {why}"
        assert want_phrase in why, (f"{label} failed for the wrong reason: {why!r} does not "
                                    f"name {want_phrase!r}, so this world cannot show the "
                                    f"mix check is what refused it")

    # and a row recorded against a different version of the test must not pass
    def _stale(d):
        _good(d)
        p = os.path.join(d, "runs", "launch_tests.json")
        rows = json.load(open(p, encoding="utf-8"))
        rows[ARCH_TESTS[0]]["test_sha256"] = "0" * 64
        json.dump(rows, open(p, "w", encoding="utf-8"))
    ds = world(_stale)
    st, why = gate_arch_tests(ds, os.path.join(ds, mix_rel), 7)
    assert st != GO, f"a row recorded against a different test version passed: {why}"


    # WHY THE REASON IS CHECKED AND NOT ONLY THE STATE (de's rule, 2026-09-01):
    # four broken worlds elsewhere were empty trees, so the gate failed for the
    # ABSENCE of everything rather than for the planted defect -- undo the mutation
    # and they still FAIL, which means the selftest proved nothing. A world that
    # fails for the wrong reason is a world that cannot detect a regression in the
    # thing it claims to test. So each world declares the phrase its intended defect
    # must produce.
    expect = {
        "mix_file": "_blocked",
        "epochs_measured": "ESTIMATED",
        "corpora": "data/corpus",          # dirs excluded from the copy on purpose
        "arch_tests": "launch_tests.json",
        "recipe_provenance": "recipe_provenance.json",
        "memory_measured": "world=7",
        "cards": "card_assignment.json",
        "vocab_id": "distinct vocab_id",
        "checks_and_drift": "no check lines",
        "launch_command": "warmdown",
        "cloze_regions": "unseen_lo",
    }
    bad = []
    for name, fn in GATES:
        d, mixp = broken[name]
        try:
            state, why = fn(d, mixp, 7)
        except Exception as e:
            state, why = NOGO, f"raised {type(e).__name__}"
        if state == GO:
            bad.append(f"{name} reported GO on its broken world ({why[:60]})")
            continue
        want = expect.get(name)
        if want and want.lower() not in why.lower():
            bad.append(f"{name} failed for the WRONG REASON: expected a message naming "
                       f"{want!r}, got {why[:70]!r} -- the world may be failing on "
                       f"absence rather than on the planted defect")
    # DE'S TEST, and it is stronger than the reason check above: undo the mutation
    # and the world must go GREEN. A world that still fails with the defect removed
    # was failing on something else all along, and no wording assertion catches that.
    # Only the gates whose defect is reversible in-place are checked here; the ones
    # whose world is "the artifact is absent" are reversed by writing it back.
    def _undo_check(name, fn, d, mixp, restore):
        restore(d)
        st, why = fn(d, mixp, 7)
        return st, why

    reversible = {}
    # recipe_provenance: write a real source back
    dr, mr = broken["recipe_provenance"]
    reversible["recipe_provenance"] = (dr, mr, lambda d: json.dump(
        {f: "experiments.jsonl:pretrain_30b_s2" for f in RECIPE_FLAGS},
        open(os.path.join(d, "runs", "recipe_provenance.json"), "w", encoding="utf-8")))
    # memory_measured: write a peak at world 7
    dm, mm = broken["memory_measured"]
    reversible["memory_measured"] = (dm, mm, lambda d: json.dump(
        {"7": {"peak_GiB": 1.0}},
        open(os.path.join(d, "runs", "memory_peaks.json"), "w", encoding="utf-8")))
    # cards: grant the block
    dc, mc = broken["cards"]
    reversible["cards"] = (dc, mc, lambda d: json.dump(
        {"launch_block_granted": True, "note": "granted for the selftest"},
        open(os.path.join(d, "runs", "card_assignment.json"), "w", encoding="utf-8")))

    # launch_command: add the four missing flags to the command. Reversibility matters
    # more here than for the others -- the gate could refuse because the register is
    # unreadable, or because no row is running, both of which look identical in the
    # summary. Undoing ONLY the omission must clear it.
    dl, ml = broken["launch_command"]

    def _fixcmd(d):
        path = os.path.join(d, "runs", "experiments.jsonl")
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(ln) for ln in fh if ln.strip()]
        for r in rows:
            if r.get("status") == "running":
                r["cmd"] += " --warmdown 0.1 --anneal_frac 0 --warmup 300 --save_every 500"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    reversible["launch_command"] = (dl, ml, _fixcmd)

    # Shape-keyed recipe selection. The defect this replaced was a GO, not a red, so the
    # decisive case is the NEGATIVE one: a file holding a recipe for ANOTHER shape only
    # must refuse. Called directly -- _recipe_for_shape is pure, like reconcile_command.
    _l12 = {"d": 1024, "layers": 12, "heads": 8, "ffn_hidden": 3072}

    def _grp(d, L, h, F):
        return {"dim": f"{d} -- w", "layers": f"{L} -- w",
                "heads": f"{h} -- w", "ffn_hidden": f"{F} -- w"}

    _other = {"shapes": {"493.6M-L32": _grp(1024, 32, 8, 3072)}}
    if _recipe_for_shape(_other, _l12)[0] is not None:
        bad.append("a recipe file holding ONLY a d1024-L32 group answered a d1024-L12 "
                   "launch -- twelve values arguing for another shape read as twelve "
                   "justified values, which is the GO this partition exists to remove")
    # ...and the matching group must still be found, or the fix trades a false GO for a
    # false NO-GO and every launch is blocked instead.
    _both = {"shapes": {"493.6M-L32": _grp(1024, 32, 8, 3072), "206M-L12": _grp(1024, 12, 8, 3072)}}
    if _recipe_for_shape(_both, _l12)[0] is None:
        bad.append("_recipe_for_shape refused a file that DOES carry the launched shape's "
                   "group: the partition blocks every launch instead of the wrong one")
    # A flat file predates the partition and must keep working: the params leg adds a third
    # group later, and a reader forced to migrate the file to answer a gate migrates it wrong.
    if _recipe_for_shape({f: "runs/experiments.jsonl:r" for f in RECIPE_FLAGS}, _l12)[0] is None:
        bad.append("a FLAT recipe_provenance.json (no shapes key) is now refused -- the "
                   "partition broke the format every earlier run recorded its recipe in")
    # Two groups claiming one shape is undecidable, not a coin flip.
    _dup = {"shapes": {"a": _grp(1024, 12, 8, 3072), "b": _grp(1024, 12, 8, 3072)}}
    if _recipe_for_shape(_dup, _l12)[0] is not None:
        bad.append("two recipe groups both claiming the launched shape were resolved by "
                   "picking one -- provenance decided by dict order")
    # Depth alone must separate them: an L12 launch must not match the L32 group when every
    # OTHER field agrees, which is the near-miss the params leg will actually produce.
    if _recipe_for_shape({"shapes": {"x": _grp(1024, 32, 8, 3072)}}, _l12)[0] is not None:
        bad.append("a group differing from the launch ONLY in layers was accepted")

    # BooleanOptionalAction: `--no-X` is PRESENT, and a real omission is still caught.
    # Called directly rather than through a world, for reconcile_command's own reason --
    # it is pure, so a world would add a filesystem that has nothing to do with the
    # property. Both directions asserted, because the fix is a widened regex and a regex
    # widened too far reports every flag present and every gate green (b0, 2026-09-03).
    _sw = "grad_ckpt"
    if _sw in RECIPE_FLAGS:
        _prov = {f: "runs/experiments.jsonl:some_run" for f in RECIPE_FLAGS}
        _full = " ".join(f"--{f}" for f in RECIPE_FLAGS)
        for _form in (f"--no-{_sw}", f"--no_{_sw}"):
            _cmd = _full.replace(f"--{_sw}", _form)
            _miss, _ = reconcile_command(_cmd, _prov)
            if _sw in _miss:
                bad.append(
                    f"reconcile_command calls {_sw} missing from a command carrying "
                    f"{_form} -- BooleanOptionalAction's explicit OFF reads as an "
                    f"omission, so 'turned it off on purpose' and 'never passed it, "
                    f"took the Cfg default' produce one signal (shape 140)")
        # ...and the omission it exists to catch must still fire. A regex that matched
        # anything containing the flag name would pass the two cases above by accident.
        _miss_real, _ = reconcile_command(_full.replace(f"--{_sw}", ""), _prov)
        if _sw not in _miss_real:
            bad.append(f"reconcile_command does NOT report {_sw} missing when the "
                       f"command omits it entirely -- the --no- widening swallowed the "
                       f"omission this gate exists for")
        # A flag whose NAME CONTAINS another flag's name must not answer for it: `--no_`
        # plus a substring is the way a widened alternation starts matching neighbours.
        _miss_sub, _ = reconcile_command(_full.replace(f"--{_sw}", f"--outer_{_sw}"), _prov)
        if _sw not in _miss_sub:
            bad.append(f"--outer_{_sw} satisfied the check for --{_sw}: the pattern "
                       f"matches inside a longer flag name")

    # Citation forms, exercised on the REAL tree rather than in a temp world, because both
    # properties are about git and a world has no .git: `git cat-file -e` returns 128 there
    # for EVERY rev, so an @sha world would go red without the judgement being reached --
    # the shape fb ruled on tonight (a world that fails before the stage under test proves
    # nothing). dead_citations is pure enough to call directly with constructed prose, which
    # is the same reason reconcile_command is pure.
    _tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                                  text=True).stdout.split())
    _probe = [p for p in ("runs/experiments.jsonl", "runs/board.jsonl", "runs/tasks.jsonl")
              if p in _tracked]
    if not _probe:
        bad.append("no tracked .jsonl in the repo to probe the citation regex with")
    elif dead_citations(ROOT, f"read from {_probe[0]}, some run name", _tracked):
        # Before e1-9 the alternation put json ahead of jsonl, so every .jsonl citation was
        # truncated to .json and reported as a file that does not exist -- a dead citation
        # for a tracked file, failing CLOSED and so invisible until the first was written.
        bad.append(f"a mention of {_probe[0]!r} reads as a dead citation -- the extension "
                   "was truncated, so a tracked file looks missing (jsonl before json)")

    # @rev: one live commit, four judgements, on MANUFACTURED history. The history this
    # test needs is built in a temp repo -- commit a probe, delete it -- not read from
    # this tree: a branch behind at the moment this gate landed had no deleted probe at
    # its HEAD, and the merge that brought the deletions was the very commit the gate
    # then blocked (b0, 2026-09-02, 76 commits behind, deadlock). A selftest whose
    # premise is the repo's branch state fails on exactly the trees that need it.
    with tempfile.TemporaryDirectory(prefix="rev_selftest_") as _tmp:
        def _git(*args):
            r = subprocess.run(["git", *args], cwd=_tmp, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[:160]}")
            return r.stdout.strip()

        try:
            _git("init", "-q")
            _gone = "probes/selftest_retired_probe.py"
            os.makedirs(os.path.join(_tmp, "probes"), exist_ok=True)
            with open(os.path.join(_tmp, _gone), "w") as f:
                f.write("# manufactured by launch_gate --selftest; never tracked in the real repo\n")
            _git("add", _gone)
            _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "add probe")
            _git("rm", "-q", _gone)
            _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "retire probe")
            _sha = _git("rev-parse", "HEAD")
            _tmp_tracked = set(_git("ls-files").split())
        except (OSError, RuntimeError) as e:
            bad.append(f"@rev world build failed ({e}); a world that fails before the "
                       "stage under test proves nothing")
        else:
            for text, want_dead, what in (
                (f"measured in {_gone}@{_sha}~1.", False,
                 "a retired path at the commit before its deletion must RESOLVE"),
                (f"measured in {_gone}.", True,
                 "a deleted path with no @rev must stay DEAD, or @rev support has "
                 "silently retired the tracked test for every bare path"),
                (f"measured in {_gone}@0000000000000000000000000000000000000000.", True,
                 "@ a rev that does not exist must be DEAD (fb's world)"),
                (f"measured in probes/never_existed_xyz.py@{_sha}.", True,
                 "@ a real rev that never held the path must be DEAD"),
            ):
                if bool(dead_citations(_tmp, text, _tmp_tracked)) != want_dead:
                    bad.append(f"citation {text!r}: {what}")

    for name, (d, mixp, restore) in reversible.items():
        fn = dict(GATES)[name]
        st, why = _undo_check(name, fn, d, mixp, restore)
        if st != GO:
            bad.append(f"{name} still {st} after the defect was UNDONE ({why[:70]}) -- "
                       f"the world was failing on something other than its planted defect")

    # LAUNCH_CARDS, the lane path (de, 2026-09-04). The nine worlds above all run with
    # LAUNCH_CARDS unset, i.e. the BLOCK question, so they certify nothing about a lane launch --
    # this selftest was green on the old ownership-blind gate and would stay green on it. The
    # world is the REAL grant file, unmutated: cards 1, 2 and 6 in it are held by jobs that are
    # not ours, and the old gate returned GO for a lane launch on every one of them.
    _real_grant = os.path.join(ROOT, "runs", "card_assignment.json")
    if os.path.exists(_real_grant):
        _g = json.load(open(_real_grant, encoding="utf-8"))
        _lane = sorted(_cards_set(_g.get("lane_card") or ""), key=int)
        _block = _cards_set(_g.get("block_cards") or "")
        _held = sorted({c for c in _g.get("cards") or {}
                        if c.isdigit() and c not in _block and c not in _lane}, key=int)
        _dcards = os.path.dirname(_real_grant)
        _dcards = os.path.dirname(_dcards)
        _saved_cards = globals()["LAUNCH_CARDS"]
        _saved_owner = os.environ.get("LAUNCH_OWNER")
        try:
            for _c in _held:
                globals()["LAUNCH_CARDS"] = [_c]
                st, why = gate_cards(_dcards, None, 1)
                if st == GO:
                    bad.append(f"gate_cards reported GO for a lane launch on card {_c}, which "
                               f"the real grant gives to neither the block nor the lane: "
                               f"{why[:70]}")
                elif _c not in why:
                    bad.append(f"gate_cards refused card {_c} without naming it: {why[:70]}")
            # The positive, in the same world: the lane card IS granted, to the name `lane_to`
            # gives it. Without a positive the three cases above are satisfied by a gate that
            # refuses everything. The owner is read from the FIELD -- see gate_cards' docstring
            # for why the prose cannot answer it -- so a grant written before `lane_to` existed
            # must show as UNKNOWN naming the writer, never as a guess and never as silence.
            if _lane:
                _c = _lane[0]
                _to = str(_g.get("lane_to") or "").strip()
                globals()["LAUNCH_CARDS"] = [_c]
                if _to:
                    os.environ["LAUNCH_OWNER"] = _to
                    st, why = gate_cards(_dcards, None, 1)
                    if st != GO:
                        bad.append(f"gate_cards refused lane card {_c} to {_to!r}, whom the real "
                                   f"grant's lane_to names: {st} {why[:70]} -- a gate that "
                                   f"refuses every lane launch passes the negatives for free")
                    os.environ["LAUNCH_OWNER"] = "nobody_in_this_grant"
                    st, why = gate_cards(_dcards, None, 1)
                    if st == GO:
                        bad.append(f"gate_cards granted lane card {_c} to a session lane_to does "
                                   f"not name: {why[:70]}")
                else:
                    os.environ["LAUNCH_OWNER"] = "anyone"
                    st, why = gate_cards(_dcards, None, 1)
                    if st != UNKNOWN or "grant-lane" not in why:
                        bad.append(f"the real grant carries no lane_to, so a lane launch on card "
                                   f"{_c} must be UNKNOWN naming card_claim grant-lane; got "
                                   f"{st} {why[:70]}")
        finally:
            globals()["LAUNCH_CARDS"] = _saved_cards
            if _saved_owner is None:
                os.environ.pop("LAUNCH_OWNER", None)
            else:
                os.environ["LAUNCH_OWNER"] = _saved_owner

    # WHERE a gate refuses, not only whether. All nine worlds above were green on a
    # tree that had lost AUTHORITY: a gate answering from the wrong filesystem still
    # refuses. It is a false-GO path -- main() suppresses GO only for gates AUTHORITY
    # excludes here, so an empty mapping prints a full GO computed from main alone,
    # where no corpus exists. This world is deliberately undamaged, so the assertion
    # cannot be met by the absence the other worlds rely on (e1's case).
    pod_gates = sorted(n for n, a in AUTHORITY.items() if a == "pod")
    assert pod_gates, "no gate claims pod authority -- AUTHORITY is gone or empty"
    d_ok, m_ok = broken["mix_file"]  # any shaped world; the mix is whole in it
    for name, state, why in run(d_ok, m_ok, 7, here="main"):
        if name in pod_gates:
            if state != UNKNOWN:
                bad.append(f"{name} answered {state} on main, where its evidence cannot "
                           f"exist ({why[:60]}) -- a believable answer from the wrong "
                           f"filesystem is worse than no answer")
            elif "run it there" not in why:
                bad.append(f"{name} returned UNKNOWN without naming where to run it: "
                           f"{why[:70]}")
    # and the converse, or the check above passes on a gate that says UNKNOWN always
    for name, state, why in run(d_ok, m_ok, 7, here="pod"):
        if name in pod_gates and state == UNKNOWN and "run it there" in why:
            bad.append(f"{name} declined to answer on pod too -- it is not location-aware, "
                       f"it is just always UNKNOWN")

    # POD ATTRIBUTION (44-8): a verdict from a DRIFTED tree is refused. Drift is
    # the only refusal quantity (fb, 2026-09-01): it compares exactly the files
    # the launch is cut from. Unregistered files are NOT refused -- 168 of them
    # are one-off scripts that only ever existed on the pod, and a one-off says
    # nothing about whether the training code is main's code.
    from pod_drift import mode_disk, sha_disk

    def attributable_world(mutate):
        # A world whose manifest names exactly what it holds, so check_pod's
        # missing-file branch cannot fire on the subset copy world() makes.
        d = world(lambda d: None)
        mp = os.path.join(d, "data", "pod_head_manifest.txt")
        lines = []
        for dirpath, _, filenames in os.walk(d):
            if os.path.relpath(dirpath, d).split(os.sep)[0] == "data":
                continue  # manifest territory is code/config; the manifest lists none of data/ here
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                # The mode column too (b0-19). Written as three columns, every 755 file
                # read back as 644 -- read_manifest's default for an old row -- and the
                # gate reported 15 mode drifts in a world built to be CLEAN. The default
                # is right for a manifest committed before the column existed and wrong
                # for one generated now: a fixture that hand-writes the artifact must
                # write every field the reader reads.
                lines.append(f"{sha_disk(fp)}  {os.path.relpath(fp, d)}  code  "
                             f"{mode_disk(fp)}")
        with open(mp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        mutate(d)
        return d

    d = attributable_world(lambda d: None)
    ok, msg = pod_attribution(d)
    if not ok:
        bad.append(f"pod_attribution refused a clean attributable world ({msg[:60]})")
    d = attributable_world(lambda d: open(
        os.path.join(d, "scripts", "probe_throwaway.py"), "w", encoding="utf-8").write("# not in any manifest\n"))
    ok, msg = pod_attribution(d)
    if not ok:
        bad.append(f"pod_attribution refused an unregistered one-off (fb 2026-09-01: drift "
                   f"is the quantity, not the file count): {msg[:60]}")
    d = attributable_world(lambda d: open(
        os.path.join(d, "scripts", "harness.py"), "a", encoding="utf-8").write("\n# drifted after manifest\n"))
    ok, msg = pod_attribution(d)
    if ok or "drifted" not in msg:
        bad.append(f"pod_attribution did not refuse a drifted file ({msg[:60]})")

    # FAIL PARTITION (fb, 2026-09-01): env-state FAILs gate on every machine;
    # repo-scan FAILs gate only on main -- on the pod they are UNKNOWN, not NO-GO.
    def fl(name):
        return f"  [FAIL] {name:<22} synthetic (0.0s)"
    s, _ = _partition_fails([fl("mix_supply")], "pod", 40)
    if s != NOGO:
        bad.append(f"env-state FAIL did not NO-GO on the pod ({s})")
    s, _ = _partition_fails([fl("entrypoint_help")], "main", 40)
    if s != NOGO:
        bad.append(f"repo-scan FAIL did not NO-GO on main ({s})")
    s, why = _partition_fails([fl("entrypoint_help")], "pod", 40)
    if s != UNKNOWN or "authority=main" not in why:
        bad.append(f"repo-scan FAIL on pod was {s}, not UNKNOWN-with-authority ({why[:50]})")
    s, _ = _partition_fails([fl("mix_supply"), fl("entrypoint_help")], "pod", 40)
    if s != NOGO:
        bad.append(f"mixed FAILs on pod did not NO-GO on the env-state one ({s})")
    # Declaration completeness lives next to the declarations: harness's own
    # selftest asserts every CHECKS entry is in EVIDENCE and vice versa.

    # gate_vocab_id partitions by whether the row's checkpoint is OURS, and the exclusion
    # must not become a way to go green. Built on the REAL ledger with one row appended,
    # because the population is what matters: 54 of our checkpoints at one vocab_id and one
    # `hf` row (pythia-160m-step2000, the e1-25 control) is the state that made this NO-GO.
    #
    # The third case is the load-bearing one. Keying the exclusion on the vocab_id VALUE
    # would be circular -- `hf` is exactly the string being excused -- so a ckpt_* claiming
    # vocab_id `hf` must still NO-GO. The second asserts an excluded row stays NAMED: a gate
    # that silently drops rows to pass is how a real foreign checkpoint hides behind a
    # control's exemption.
    _real = os.path.join(ROOT, "runs", "score_matrix.jsonl")
    _tok = os.path.join(ROOT, "data", "tokenizer.json")
    if os.path.exists(_real) and os.path.exists(_tok):
        import shutil
        import tempfile
        _src = open(_real, encoding="utf-8").read()
        for _label, _row, _want, _named in (
            ("a ckpt_* with a second vocab_id", {"ckpt": "ckpt_rebuilt_vocab.pt",
                                                 "vocab_id": "deadbeefdeadbeef"}, NOGO, None),
            ("a second foreign model", {"ckpt": "qwen3-1.7b", "vocab_id": "hf2"},
             GO, "qwen3-1.7b"),
            ("a ckpt_* claiming vocab_id 'hf'", {"ckpt": "ckpt_sneaky.pt", "vocab_id": "hf"},
             NOGO, None),
        ):
            with tempfile.TemporaryDirectory(prefix="vocab_selftest_") as _d:
                os.makedirs(os.path.join(_d, "runs"))
                os.makedirs(os.path.join(_d, "data"))
                shutil.copy(_tok, os.path.join(_d, "data", "tokenizer.json"))
                with open(os.path.join(_d, "runs", "score_matrix.jsonl"), "w",
                          encoding="utf-8") as _f:
                    _f.write(_src)
                    _f.write(json.dumps(_row) + "\n")
                _s, _why = gate_vocab_id(_d, "data/mix_scale_3.24b.json", 7)
                if _s != _want:
                    bad.append(f"gate_vocab_id on {_label}: {_s}, wanted {_want} ({_why[:70]})")
                elif _named and _named not in _why:
                    bad.append(f"gate_vocab_id excluded {_named} without naming it: {_why[:90]}")

    # gate_cloze_regions: four worlds off the REAL mix and a real-shaped header. Each of the
    # three conditions must refuse on its own, and a fifth asserts the gate does not
    # manufacture a verdict when the probe is absent -- UNKNOWN, because a GO there would
    # certify a comparison nobody has defined. The mix is copied and MUTATED (the weight
    # nudged), never hand-written: a hand-made mix shares the gate's own assumption about
    # which fields exist and would pass a gate that reads none of them.
    _mix_src = os.path.join(ROOT, "data", "mix_200m_8b.json")
    if os.path.exists(_mix_src):
        import shutil
        import tempfile
        _CTRL = ("env CUDA_VISIBLE_DEVICES=4,2 NGPU=2 ./run_ddp.sh --name arm "
                 "--mix data/mix_200m_8b.json --batch 16 --accum 2 --max_steps 3815 --seed 42")
        _mix0 = json.load(open(_mix_src, encoding="utf-8"))
        _alloc = int((_mix0["total_tokens"] / 4096)
                     * _mix0["domains"]["code_py_starcoder"]["weight"])
        _hdr = {"_header": True, "domain": "code_py_starcoder", "n_seen_rows": 80280,
                "bounds": {"unseen_lo": _alloc, "seed": 42, "seq": 4096,
                           "mix": "mix_200m_8b.json"}}
        for _label, _mut, _cmd, _want in (
            ("the control's own line", None, _CTRL, GO),
            ("a mix whose starcoder weight is nudged up",
             ("weight", _mix0["domains"]["code_py_starcoder"]["weight"] * 1.01), _CTRL, NOGO),
            ("a launch on another mix", None,
             _CTRL.replace("--mix data/mix_200m_8b.json", "--mix data/mix_scale_3.24b.json"),
             NOGO),
            ("an arm at another seed", None, _CTRL.replace("--seed 42", "--seed 43"), NOGO),
        ):
            with tempfile.TemporaryDirectory(prefix="cloze_selftest_") as _d:
                os.makedirs(os.path.join(_d, "data", "probes"))
                shutil.copy(_mix_src, os.path.join(_d, "data", "mix_200m_8b.json"))
                if _mut:
                    _mp = os.path.join(_d, "data", "mix_200m_8b.json")
                    _m = json.load(open(_mp, encoding="utf-8"))
                    _m["domains"]["code_py_starcoder"][_mut[0]] = _mut[1]
                    json.dump(_m, open(_mp, "w"))
                with open(os.path.join(_d, "data", "probes", "api_cloze.jsonl"), "w",
                          encoding="utf-8") as _f:
                    _f.write(json.dumps(_hdr) + "\n")
                _s, _why = gate_cloze_regions(_d, "data/mix_200m_8b.json", 2, cmd=_cmd)
                if _s != _want:
                    bad.append(f"gate_cloze_regions on {_label}: {_s}, wanted {_want} "
                               f"({_why[:80]})")
        with tempfile.TemporaryDirectory(prefix="cloze_selftest_") as _d:
            os.makedirs(os.path.join(_d, "data"))
            shutil.copy(_mix_src, os.path.join(_d, "data", "mix_200m_8b.json"))
            _s, _why = gate_cloze_regions(_d, "data/mix_200m_8b.json", 2, cmd=_CTRL)
            if _s != UNKNOWN:
                bad.append(f"gate_cloze_regions with no probe file: {_s}, wanted UNKNOWN")

    if bad:
        raise AssertionError("gates that cannot fail:\n  " + "\n  ".join(bad))
    print(f"launch_gate selftest OK: {len(GATES)} gates, each FAILs on a damaged real artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())

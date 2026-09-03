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
    # Absence and failure are distinct outcomes (main, taken in the merge): absent
    # means nobody ran it, a failing row means it ran and failed. The first is UNKNOWN,
    # the second NO-GO, and collapsing them loses the only fact that says what to do.
    unrecorded = [n for n in ARCH_TESTS if n not in r]
    if unrecorded:
        return UNKNOWN, (f"launch_tests.json records no result for {', '.join(unrecorded)} "
                         f"(it has: {', '.join(sorted(r)[:4]) or 'nothing'}) -- a record "
                         f"that does not name the required test is not evidence it ran")
    problems = []
    for name in ARCH_TESTS:
        row = r[name]
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
    """7. The block is free, or a controller assignment says otherwise."""
    p = os.path.join(root, "runs", "card_assignment.json")
    if os.path.exists(p):
        try:
            a = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError) as e:
            return NOGO, f"card_assignment.json unreadable: {e}"
        if a.get("launch_block_granted"):
            return GO, f"controller granted the block: {a.get('note', '')[:60]}"
    return UNKNOWN, ("no runs/card_assignment.json with launch_block_granted -- card "
                     "ownership cannot be read from an artifact and needs the controller")


def gate_vocab_id(root, mix_path, world):
    """8. The tokenizer's identity matches what the caches were built against."""
    tokp = os.path.join(root, "data", "tokenizer.json")
    if not os.path.exists(tokp):
        return NOGO, "data/tokenizer.json is absent"
    ids = set()
    for r in glob.glob(os.path.join(root, "runs", "score_matrix.jsonl")):
        for line in open(r, encoding="utf-8"):
            if line.strip():
                try:
                    v = json.loads(line).get("vocab_id")
                except ValueError:
                    continue
                if v:
                    ids.add(v)
    if not ids:
        return UNKNOWN, "no vocab_id recorded anywhere to compare the tokenizer against"
    if len(ids) > 1:
        return NOGO, f"{len(ids)} distinct vocab_id in the ledger: {sorted(ids)[:3]}"
    return GO, f"one vocab_id across the ledger: {ids.pop()}"


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

    The register is append-only and folded by name, so the LAST row for a name wins --
    an earlier failed attempt of the same run must not be read as the live command.
    """
    p = os.path.join(root, "runs", "experiments.jsonl")
    if not os.path.exists(p):
        return None, "runs/experiments.jsonl does not exist"
    latest = {}
    try:
        with open(p, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                if r.get("name"):
                    latest[r["name"]] = r
    except (OSError, ValueError) as e:
        return None, f"runs/experiments.jsonl unreadable: {e}"
    running = [r for r in latest.values() if r.get("status") == "running" and r.get("cmd")]
    if not running:
        return None, "no row with status running and a cmd field"
    if len(running) > 1:
        names = ", ".join(sorted(r["name"] for r in running))
        return None, f"{len(running)} runs claim status running ({names})"
    return running[0]["cmd"], f"runs/experiments.jsonl, {running[0]['name']}"


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
            f.write(json.dumps({"ckpt": "x.pt", "vocab_id": "0" * 16}) + "\n")
            f.write(json.dumps({"ckpt": "y.pt", "vocab_id": "1" * 16}) + "\n")
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

    if bad:
        raise AssertionError("gates that cannot fail:\n  " + "\n  ".join(bad))
    print(f"launch_gate selftest OK: {len(GATES)} gates, each FAILs on a damaged real artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())

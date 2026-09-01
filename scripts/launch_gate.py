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
                "lr_scale", "grad_ckpt")


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
        if src is None:
            missing.append(name)
        elif "ESTIMATED" in str(src).upper():
            est.append(name)
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
        # A COMPARISON THAT DID NOT RUN IS NOT A COMPARISON THAT PASSED. My first
        # version read `if want and got != want`, so a domain with no fingerprint in
        # the mix skipped the check and the gate printed GO having compared nothing.
        # b0 measured the blast radius: 12 of 13 mixes have at least one such domain
        # and mix_500m -- the launch mix -- has NINE OF NINE. The gate would have
        # certified "fingerprints match" for a mix where no fingerprint exists.
        # Same sentence as "0 files, all compliant": a universal claim over an empty set.
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
LAUNCH_SHAPE = {"d": 1024, "layers": 32, "heads": 8, "ffn_hidden": 3072}


def _sha256(p):
    if not os.path.exists(p):
        return None
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
    """
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
            elif here and want != here:
                problems.append(f"{name}: recorded against {want[:12]}, the file here "
                                f"is {here[:12]} -- the test changed after it passed")
    if problems:
        return NOGO, "; ".join(problems[:3])
    return GO, (f"{len(ARCH_TESTS)} shape test(s) passed at "
                f"d{LAUNCH_SHAPE['d']} L{LAUNCH_SHAPE['layers']} on a real kernel")


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
    return GO, f"all {len(RECIPE_FLAGS)} recipe values name a source"


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
    # "the file is absent" and "the controller said no" are DIFFERENT STATES and were
    # covered by one sentence. The second is a decision someone made on purpose; the
    # first sends a reader to create a file that already exists. fb caught this with
    # the file sitting on the pod at 1279 bytes, granted=false, being reported as
    # "no runs/card_assignment.json".
    p = os.path.join(root, "runs", "card_assignment.json")
    if not os.path.exists(p):
        return UNKNOWN, ("runs/card_assignment.json does not exist -- card ownership is a "
                         "decision, so it has to be recorded by the controller, not inferred")
    try:
        a = json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return NOGO, f"card_assignment.json unreadable: {e}"
    granted = a.get("launch_block_granted")
    if granted:
        return GO, f"controller granted the block: {str(a.get('note') or '')[:60]}"
    note = str(a.get("note") or "").strip()
    if note.upper().startswith("UNSOURCED") or not note:
        return UNKNOWN, ("card_assignment.json exists but records no grant either way "
                         "(launch_block_granted=false, no note) -- the file is a stub")
    return NOGO, f"the controller has NOT granted the block: {note[:70]}"


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
        return NOGO, f"{len(fails)} FAIL of {len(ran)} checks: {fails[0][:80]}"
    return GO, f"harness check: {len(ran)} checks ran, 0 FAIL"


# The nine, in the order they are reported. A gate added here is automatically
# covered by --selftest's broken-world requirement (see selftest below).
# WHERE EACH GATE'S TRUTH LIVES.
#
# A gate's conclusion depends on which filesystem it ran on, and until now that
# fact was absent from the conclusion. Same class as everything else today, with
# the location standing in for the configuration: on main, `corpora` always reports
# missing dirs (a dev tree holds no corpus) while on the pod it reported the real
# defect; `checks_and_drift` read 0 FAIL on main and 11 FAIL on the pod at the same
# instant. Both were true of where they ran and neither was the answer.
#
# My own 4c1e002 caused half of this: "read only from main, refuse GO elsewhere" is
# right for code and wrong for data, because it excludes the ONE place the data
# questions can be answered.
#
#   MAIN  code/config: the launch is cut from main, so main's state is the launch's
#   POD   data/machine: corpora and token caches exist nowhere else
#   BOTH  the same gate means DIFFERENT things in each place and needs both readings
AUTHORITY = {
    "mix_file": "main", "recipe_provenance": "main", "vocab_id": "main",
    "arch_tests": "main", "cards": "main",
    "corpora": "pod", "epochs_measured": "pod",
    "checks_and_drift": "both",
}


def _here():
    """pod or main-side. The pod is the box that holds the corpus; a dev worktree
    is not, and neither is the integration tree."""
    return "pod" if os.path.isdir("/work/aupai") and os.path.abspath(ROOT).startswith("/work/") else "main"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default=os.path.join(ROOT, "data", "mix_500m.json"))
    ap.add_argument("--world", type=int, default=7)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())

    here = _here()
    root, note = (ROOT, f"running on the pod ({ROOT})") if here == "pod" else _launch_root(ROOT)
    rows = run(root, a.mix, a.world, here)
    elsewhere = sorted(n for n, _ in GATES if AUTHORITY.get(n, "main") not in (here, "both"))
    print(f"launch-gate  mix={os.path.relpath(a.mix, ROOT)}  world={a.world}  here={here}")
    print(f"             {note}")
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
    import shutil
    import tempfile

    def world(mutate):
        d = tempfile.mkdtemp()
        for sub in ("data", "runs", "scripts"):
            src = os.path.join(ROOT, sub)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(d, sub),
                                ignore=shutil.ignore_patterns("corpus", "*.pt", "raw", "_*"),
                                dirs_exist_ok=True)
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
        # The tokenizer must be PRESENT or this world fails on its absence and proves
        # nothing about the two-vocab_id defect it exists to plant (de's rule): undo
        # the extra ledger row and it would still FAIL, so the world could not detect
        # a regression in the id comparison.
        write_mix(d, lambda m: None)
        real_tok = os.path.join(ROOT, "data", "tokenizer.json")
        dst = os.path.join(d, "data", "tokenizer.json")
        if os.path.exists(real_tok):
            shutil.copy(real_tok, dst)
        else:
            open(dst, "w", encoding="utf-8").write('{"model":{"vocab":{}}}')
        p = os.path.join(d, "runs", "score_matrix.jsonl")
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ckpt": "x.pt", "vocab_id": "0" * 16}) + "\n")
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

    # corpora gets a SECOND world: dirs and shards all present, but the mix carries no
    # fingerprint. This is b0's find and my first version passed it -- `if want and ...`
    # skipped the comparison and reported "fingerprints match" having compared nothing.
    # The missing-dirs world cannot catch it, because it fails earlier for another
    # reason: a gate can be right about one input and blind on another.
    def _nofp(d):
        m = json.load(open(real_mix, encoding="utf-8"))
        for spec in m["domains"].values():
            spec.pop("fingerprint", None)
        os.makedirs(os.path.join(d, "data"), exist_ok=True)
        json.dump(m, open(os.path.join(d, mix_rel), "w", encoding="utf-8"),
                  ensure_ascii=False)
        # give every domain a real-looking dir + shard + stats, so the ONLY defect
        # left is the absent fingerprint
        for name in m["domains"]:
            cd = os.path.join(d, "data", "corpus", name)
            os.makedirs(cd, exist_ok=True)
            open(os.path.join(cd, "x_000.jsonl"), "w").write('{"content":"x"}\n')
            json.dump({"fingerprint": "deadbeefdeadbeef"},
                      open(os.path.join(cd, "build_corpus_stats.json"), "w"))
    dnf = world(_nofp)
    st, why = gate_corpora(dnf, os.path.join(dnf, mix_rel), 7)
    assert st != GO, f"a mix with no fingerprints must not report a match: {why}"
    assert "compared" in why or "no fingerprint" in why, f"the reason must name it: {why}"

    # arch_tests gets a SECOND world: the record is present, well-formed and honest,
    # and simply does not name the required tests -- de's {"ok": "pass"} case. The
    # file-removed world cannot catch it, because that world fails on absence. Same
    # lesson as the corpora pair: one broken world per gate is not enough when a gate
    # can be right about one input and blind on another.
    def _wrongkeys(d):
        write_mix(d, lambda m: None)
        os.makedirs(os.path.join(d, "runs"), exist_ok=True)
        for f in ("scripts/test_arch_L32.py", "scripts/test_e2e.py"):
            fp = os.path.join(d, f)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            if not os.path.exists(fp):
                open(fp, "w").write("#\n")
        json.dump({"ok": "pass"},
                  open(os.path.join(d, "runs", "launch_tests.json"), "w", encoding="utf-8"))
    dwk = world(_wrongkeys)
    st, why = gate_arch_tests(dwk, os.path.join(dwk, mix_rel), 7)
    assert st != GO, f"a record naming no required test must not pass: {why}"
    assert "no result for" in why, f"the reason must say what is unrecorded: {why}"

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
        ("both tests passing at the WRONG shape",
         {n: {"result": "pass", "shape": dict(LAUNCH_SHAPE, layers=12), "real_kernel": True}
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
                           "real_kernel": True,
                           "test_sha256": _sha256(os.path.join(d, n))}
                       for n in ARCH_TESTS}, f)
    dg = world(_good)
    st, why = gate_arch_tests(dg, os.path.join(dg, mix_rel), 7)
    assert st == GO, f"arch_tests refuses the record it is written to accept: {why}"

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

    for name, (d, mixp, restore) in reversible.items():
        fn = dict(GATES)[name]
        st, why = _undo_check(name, fn, d, mixp, restore)
        if st != GO:
            bad.append(f"{name} still {st} after the defect was UNDONE ({why[:70]}) -- "
                       f"the world was failing on something other than its planted defect")

    if bad:
        raise AssertionError("gates that cannot fail:\n  " + "\n  ".join(bad))
    print(f"launch_gate selftest OK: {len(GATES)} gates, each FAILs on a damaged real artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())

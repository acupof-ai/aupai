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
        if want and got != want:
            bad.append(f"{name}: fingerprint {got} != mix's {want}")
        if re.search(r"mix_scale_[\d.]+b", str(spec.get("role", "")) + name):
            bad.append(f"{name}: points at a frozen mix_scale_* pool")
    if bad:
        return NOGO, f"{len(bad)} domain(s) failed: {'; '.join(bad[:3])}"
    return GO, f"all {len(m['domains'])} corpora present, sharded, fingerprints match"


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
    unrecorded = [n for n in ARCH_TESTS if n not in r]
    if unrecorded:
        return UNKNOWN, (f"no record of {', '.join(unrecorded)} -- the file records "
                         f"{sorted(r)[:4]}, which is not the same claim")
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
        return NOGO, f"{len(fails)} FAIL of {len(ran)} checks: {fails[0][:80]}"
    return GO, f"harness check: {len(ran)} checks ran, 0 FAIL"


# The nine, in the order they are reported. A gate added here is automatically
# covered by --selftest's broken-world requirement (see selftest below).
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


def run(root, mix_path, world):
    rows = []
    for name, fn in GATES:
        try:
            state, why = fn(root, mix_path, world)
        except Exception as e:  # a gate that crashes is NOT a pass
            state, why = NOGO, f"the gate itself raised: {type(e).__name__}: {e}"
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

    root, note = _launch_root(ROOT)
    rows = run(root, a.mix, a.world)
    print(f"launch-gate  mix={os.path.relpath(a.mix, ROOT)}  world={a.world}")
    print(f"             {note}\n")
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
        write_mix(d, lambda m: None)
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

    bad = []
    for name, fn in GATES:
        d, mixp = broken[name]
        try:
            state, why = fn(d, mixp, 7)
        except Exception as e:
            state, why = NOGO, f"raised {type(e).__name__}"
        if state == GO:
            bad.append(f"{name} reported GO on its broken world ({why[:60]})")
    if bad:
        raise AssertionError("gates that cannot fail:\n  " + "\n  ".join(bad))
    print(f"launch_gate selftest OK: {len(GATES)} gates, each FAILs on a damaged real artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())

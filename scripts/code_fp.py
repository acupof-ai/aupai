#!/usr/bin/env python3
# restartable: two `git show` reads per sha and an AST parse; writes nothing. An interrupt
# costs a second, rerun the same command.
"""Which training-math units differ between two commits.

    python3 scripts/code_fp.py <sha>                # the ten fingerprints at that commit
    python3 scripts/code_fp.py <sha_a> <sha_b>      # the units that DIFFER between them
    python3 scripts/code_fp.py --selftest

The question this answers is "which diffs do I have to read", not "may I proceed". b0-20
asked for a gate that refuses a reused baseline whose code moved; the measurement said no
usable gate exists here. Over the 57 commits touching train.py/model.py in three days, a
whole-file content hash gives 57 distinct values, a whole-file AST hash 52, and even this
ten-unit whitelist jumps 11 times -- against baselines that are typically days old, the
exemption rate approaches 100%, and an exemption nobody reads is worse than no gate
(facts/efficiency.json#eff.code_fingerprint_gate_unusable). So this reports.

WHY THESE TEN UNITS: they are pure functions of tensors and config -- no logging, no flag
plumbing, no startup validation. That property is the definition; the list is derived from
it, and a new unit is classified by asking it. Derived BEFORE it was tested against the one
range b0-20 claimed hand-verified, which is why the zero false alarms there mean something.

WHAT IS DELIBERATELY OUT: Cfg, build_optimizers, AttnRes, Block, HybridLM. Each carries
real arithmetic, and each is also where the benign changes live -- an optimizer group label,
a default-off flag pair, a startup refusal branch. Including them produced three false
alarms on the hand-verified range. THE KNOWN MISS, written down rather than left implicit: a
change to the residual order inside Block.forward is arithmetic and this tool will not name
it. That is the price of a report a reader still trusts.

SHAS ONLY, NEVER THE WORKING TREE. A fingerprint taken from a dirty tree corresponds to no
commit while looking exactly like one that does -- the failure mode this repo hit five times
on 2026-09-03, where a well-formed value carried a wrong answer and nothing flagged it.
There is deliberately no "default to HEAD": that convenience is the door the working tree
comes in through.
"""
import ast
import hashlib
import subprocess
import sys

# unit name -> the file it must live in. Sorted output, so a diff of two runs is stable.
UNITS = {
    "train.py": ("Muon", "set_schedule", "MasterWeights", "_fp8_mm", "FP8LinearFunction"),
    "model.py": ("RMSNorm", "rms_scale", "DeltaRecurrence", "GatedMLA", "SwiGLU"),
}


def _blob(sha, path):
    """One file at one commit. Raises on an unresolvable sha rather than falling back."""
    out = subprocess.run(["git", "show", f"{sha}:{path}"], capture_output=True)
    if out.returncode != 0:
        raise SystemExit(
            f"cannot read {path} at '{sha}': {out.stderr.decode(errors='replace').strip()}\n"
            "This tool takes commit shas only. It never reads the working tree: a fingerprint "
            "taken from a dirty tree corresponds to no commit while looking exactly like one "
            "that does."
        )
    return out.stdout.decode("utf-8", "replace")


def fingerprints(sha):
    """{'train:Muon': '<10 hex>', ...} for the ten units at this commit.

    Keyed on ast.dump, so a comment, a docstring or reformatting does not move a value while
    any change to the code does. A unit absent at that commit is simply missing from the
    result, and a caller comparing two commits sees that as a difference -- which it is.
    """
    out = {}
    for path, keep in UNITS.items():
        tree = ast.parse(_blob(sha, path))
        for node in tree.body:
            name = getattr(node, "name", None)
            if name in keep:
                key = f"{path.split('.')[0]}:{name}"
                out[key] = hashlib.sha256(ast.dump(node).encode()).hexdigest()[:10]
    return out


def differing(sha_a, sha_b):
    a, b = fingerprints(sha_a), fingerprints(sha_b)
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def selftest():
    """Known answers in BOTH directions, from measurements taken 2026-09-03.

    A test that only checks "reports NONE when nothing changed" is passed by an
    implementation that always reports NONE, so the second case is the one that matters.
    """
    fails = []
    cases = [
        # The two legs of the params-vs-data experiment. 6925ce02 is the sha
        # runs/data_leg_206m_8b.log records as the pod's code at launch; the one commit
        # between it and e28ddc06 that touched these files is 169da865
        # (--attn_res_fp32_logits, default off), which lands entirely in the wiring layer.
        ("6925ce02", "e28ddc06", [], "the two legs: default-off flag, no math"),
        # A real arithmetic change that MUST be named: Muon's shape-based lr, A/B (2a).
        ("2a0096c7^", "2a0096c7", ["train:Muon"], "A/B (2a) Muon shape-based lr"),
        # A wiring-only commit adjacent to the range above: an optimizer group label.
        ("4c86dc71^", "4c86dc71", [], "b0-14 per-group lr on the step line"),
    ]
    for a, b, want, why in cases:
        try:
            got = differing(a, b)
        except SystemExit as e:
            print(f"SKIP {a}..{b} ({why}): {str(e).splitlines()[0]}")
            continue
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {a}..{b}  {why}\n"
              f"         want {want or 'NONE'}, got {got or 'NONE'}")
        if not ok:
            fails.append(f"{a}..{b}: want {want}, got {got}")

    # The working tree must be refused, not silently used.
    try:
        fingerprints("definitely-not-a-sha-91ab")
        fails.append("an unresolvable sha did not raise: the tool would fall back to something")
    except SystemExit:
        print("  ok   an unresolvable sha is refused rather than falling back")

    # Every whitelisted unit must actually exist, or the whitelist has drifted from the code
    # and a renamed unit silently stops being watched -- the fingerprint would still print.
    try:
        have = fingerprints("HEAD~1")
        want_n = sum(len(v) for v in UNITS.values())
        if len(have) != want_n:
            missing = [f"{p.split('.')[0]}:{n}" for p, ks in UNITS.items() for n in ks
                       if f"{p.split('.')[0]}:{n}" not in have]
            fails.append(f"whitelist names {want_n} units, found {len(have)}; missing {missing} "
                         "-- a renamed unit stops being watched while the output still looks fine")
        else:
            print(f"  ok   all {want_n} whitelisted units exist in the tree")
    except SystemExit as e:
        print(f"  SKIP unit-existence check: {str(e).splitlines()[0]}")

    print(f"code_fp selftest: {len(fails)} FAIL")
    return 1 if fails else 0


def main(argv):
    if "--selftest" in argv:
        return selftest()
    if len(argv) == 1:
        for k, v in sorted(fingerprints(argv[0]).items()):
            print(f"{v}  {k}")
        return 0
    if len(argv) == 2:
        diff = differing(argv[0], argv[1])
        print(f"pure-compute units differing {argv[0]}..{argv[1]}: "
              + (", ".join(diff) if diff else "NONE"))
        if diff:
            print("Read those diffs before reusing a baseline across this range. Reported, not "
                  "refused: see the module docstring for why no gate is viable here.")
        return 0
    print(__doc__.strip().splitlines()[0])
    print("usage: code_fp.py <sha> | <sha_a> <sha_b> | --selftest")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

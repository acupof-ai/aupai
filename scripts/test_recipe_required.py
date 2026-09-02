#!/usr/bin/env python3
"""e1-16's acceptance: omitting a justified recipe knob must be UNSATISFIABLE.

RED TODAY, ON PURPOSE. train.py is frozen while p500m_20b_0902 runs, so every
assertion here fails right now -- and it must fail on "the omission was ACCEPTED",
never on an import error or a missing attribute. A test that goes red because it
cannot reach the parser proves nothing about the parser; that is the shape fb ruled
on 2026-09-02 (a broken world failing before the stage under test). Each check
below therefore reports WHY it is red, and `--strict` turns "red for the wrong
reason" into a distinct exit code.

The defect: four launch commands on 2026-09-02 omitted recipe values and all nine
gates stayed green, because an omitted flag lands on a default. Five of the twelve
defaults EQUAL the declared recipe value (dim, heads, ffn_hidden, batch, accum), so
no check that reads the effective config can see those five missing at all. e1-9
detects the omission from the recorded command; e1-16 removes the failure class by
making the launch refuse.

THREE MECHANISMS, one per knob class, measured on main 2026-09-02 (e1-16 prep):

  ten generated knobs   train.py:2031/2036/2043 build them with default=None and
                        write Cfg only when passed, so the Cfg field is the silent
                        fallback -- delete the Cfg default
  lr_scale, save_every  argparse-only (:2095, :2066), no Cfg field at all --
                        required=True
  grad_ckpt             store_true has NO None state: absent and False are one
                        value, so a deleted default changes nothing. The fix is
                        argparse.BooleanOptionalAction with required=True, which
                        gives the --grad_ckpt/--no-grad_ckpt pair for free (fb's
                        ruling; better than the three-valued flag I proposed).
                        train.py:2112-2119 records the same trap for attn_res,
                        where a blanket `is not None` sweep would have silently
                        disabled Attention Residuals on every run.

    python scripts/test_recipe_required.py            # report, exit 0 while red
    python scripts/test_recipe_required.py --strict    # exit 1 unless all green

DELIBERATELY NOT IN THE HOOK'S SELFTEST_FILES MAP, and this needs saying because
check_selftests_are_gated's docstring is right that an unrun test reads as coverage.
That map runs each file and requires exit 0, and this file must EXIT 0 WHILE RED
today -- registering it would either break every commit or force the red to be
silenced, and a silenced red is the failure mode the map exists to prevent.

Which the gate then caught me on, correctly. Its predicate is the string "--self"
"test" ANYWHERE in a tracked .py, deliberately broad: a narrow version keyed to
add_argument missed real carriers, so it was widened (de, 2026-09-02). My first
draft explained the exemption using that exact token in this docstring, and the gate
read the prose as a carrier and FAILed -- a check whose subject is text cannot tell
an explanation from a declaration. The token is therefore split above. The gate is
not wrong here; my prose was.

The step that makes this file real is in e1-16's own acceptance: when train.py
unfreezes and the twelve become required, this goes green and is added to
SELFTEST_FILES in the SAME commit, with --strict, so it can never silently stop
holding.
"""
import argparse
import ast
import contextlib
import io
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# The twelve knobs runs/recipe_provenance.json argues for, taken from launch_gate so
# this file cannot fall out of step with the gate. A hardcoded list here would be a
# second copy of RECIPE_FLAGS, and e1-9's whole finding was that the gate went blind
# to four keys precisely because two lists disagreed.
from launch_gate import RECIPE_FLAGS  # noqa: E402

# A minimal argv that is otherwise complete: only the knob under test is missing, so a
# SystemExit can only be about that knob. --name is required by train.py today.
BASE = {
    "dim": "1024", "layers": "32", "heads": "8", "ffn_hidden": "3072",
    "batch": "32", "accum": "1", "lr_scale": "0.85", "warmdown": "0.1",
    "anneal_frac": "0", "warmup": "300", "save_every": "500",
}
SWITCHES = ("grad_ckpt",)

# What the recipe argues for, and the ONE knob whose Cfg field is spelled differently.
# --dim writes Cfg.d through an explicit special case at train.py:2108, not the generic
# loop, because "d" has no flag of its own.
RECIPE = dict(BASE, grad_ckpt="True")
CFG_NAME = {"dim": "d"}


def cfg_defaults():
    """train.py's Cfg class attributes, read from source without importing torch.

    The effective value of an omitted generated knob is Cfg's, not argparse's None
    (train.py:2110-2122 writes Cfg only when the flag was passed), so a test that
    reports the parser's None alone describes the mechanism and hides the consequence.
    Parsed rather than imported: importing train.py pulls in torch and the model.
    """
    with open(os.path.join(ROOT, "train.py"), encoding="utf-8") as fh:
        body = ast.parse(fh.read()).body
    cls = next((n for n in body
                if isinstance(n, ast.ClassDef) and n.name == "Cfg"), None)
    out = {}
    for stmt in getattr(cls, "body", []):
        target = None
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            target, value = stmt.target.id, stmt.value
        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name):
            target, value = stmt.targets[0].id, stmt.value
        if target is not None and value is not None:
            with contextlib.suppress(ValueError):
                out[target] = ast.literal_eval(value)
    return out


def build_parser():
    """train.py's REAL parser, executed from its own source, never a copy.

    main() builds the parser inline and calls parse_args() 25 statements later, so
    there is no factory to import and importing main() would start training. This
    execs exactly those statements in a namespace holding argparse -- so a flag added
    to train.py appears here with no edit, and a copy cannot drift. Returns None with
    a reason when the region cannot be located, which is a DIFFERENT outcome from a
    parser that accepted an omission.
    """
    with open(os.path.join(ROOT, "train.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    main = next((n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    if main is None:
        return None, "train.py has no top-level main()"
    start = next((i for i, s in enumerate(main.body)
                  if "ArgumentParser" in ast.unparse(s)), None)
    stop = next((i for i, s in enumerate(main.body)
                 if "parse_args" in ast.unparse(s)), None)
    if start is None or stop is None or stop <= start:
        return None, "could not delimit the parser region in main()"
    region = ast.Module(body=main.body[start:stop], type_ignores=[])
    ns = {"argparse": argparse}
    try:
        exec(compile(region, "<train.py parser>", "exec"), ns)  # noqa: S102
    except Exception as e:  # a statement in the region needed more than argparse
        return None, f"{type(e).__name__} while building the parser: {e}"
    p = ns.get("parser")
    return (p, "") if p is not None else (None, "the region defined no `parser`")


def _argv(omit=None):
    out = ["--name", "recipe_required_probe"]
    for k, v in BASE.items():
        if k != omit:
            out += [f"--{k}", v]
    for s in SWITCHES:
        if s != omit:
            out.append(f"--{s}")
    return out


def _parse(parser, argv):
    """(args, None) or (None, SystemExit code). argparse writes usage to stderr."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            return parser.parse_args(argv), None
    except SystemExit as e:
        return None, (e.code if e.code is not None else 0)


def local_deps():
    """train.py and every repo-local module it imports at module level.

    NOT train.py alone, and this correction is b0's own finding turned back on this
    file. b0's check ran `ruff --select F821 model.py` on the cut where the six
    _FP8_MAX_E4M3 uses had been left behind in train.py; ruff said "All checks passed"
    because the file it was pointed at was the clean half of the break. A dangling
    reference lives on ONE SIDE of a split, and which side is not knowable in advance,
    so naming a file is guessing. Measured on main 2026-09-02: train.py imports fone,
    and `ruff F821 train.py` cannot see fone.py at all.

    Derived from train.py's own import statements rather than listed, so b0's
    model.py enters this set the moment the split lands and no one has to remember.
    """
    with open(os.path.join(ROOT, "train.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    mods = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    out = [os.path.join(ROOT, "train.py")]
    out += [p for m in sorted(mods)
            if os.path.exists(p := os.path.join(ROOT, f"{m}.py"))]
    return out


def undefined_names(paths):
    """Names these modules use and never bind, via ruff F821. Static, and that is the point.

    b0's shape, 2026-09-02, and it applies to this file: its model.py/train.py split
    moved _FP8_MAX_E4M3 into model.py while six uses stayed in train.py, and all FOUR of
    its dynamic acceptance checks went green because the paths they exercise never touch
    FP8. ruff F821 named it in one second. A dynamic assertion speaks about the code it
    RUNS; a name that is gone from code the assertion does not reach is invisible to it.

    Measured on this very file before adding this: with `_GONE = _NEVER_DEFINED_XYZ`
    injected into a real copy of train.py, the fourteen assertions below report the
    identical "14 red, 0 wrong" -- because build_parser() execs only the parser REGION,
    so a name broken anywhere else in the module cannot reach them. e1-16 edits argparse
    defaults and Cfg fields, exactly the kind of edit that leaves a dangling reference
    behind, so the acceptance would have gone green on a train.py that cannot import.

    Returns None with a reason when ruff did not deliver a verdict -- absence of the
    tool is not evidence of absence of the defect, so main() reports that as a separate,
    third outcome rather than as a pass. THE VERDICT IS THE EXIT CODE, not the presence
    of matching text: ruff exits 0 clean, 1 with diagnostics, 2 when it could not run.
    Measured 2026-09-02 with a stub `ruff` that exits 127 -- stdout is empty, so
    filtering stdout for "F821" found nothing and this reported GREEN on a check that
    never ran. My earlier missing-tool test stubbed the binary AWAY, which raises OSError
    and took a different branch, so it did not cover a tool that runs and fails.

    Every reported line is kept, not only F821-tagged ones, for the same reason: a file
    ruff cannot PARSE exits 1 with "invalid-syntax", which the F821 filter dropped -- a
    module too broken to tokenize read as a module with no undefined names.
    """
    try:
        r = subprocess.run(["ruff", "check", "--select", "F821", "--output-format",
                            "concise", *paths], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"could not run ruff: {e}"
    if r.returncode not in (0, 1):
        return None, (f"ruff exited {r.returncode} without a verdict: "
                      f"{(r.stderr or r.stdout).strip().splitlines()[:1]}")
    return [ln for ln in r.stdout.splitlines() if ": F821 " in ln or ": invalid-syntax" in ln], ""


def main():
    strict = "--strict" in sys.argv
    parser, why = build_parser()
    if parser is None:
        # RED FOR THE WRONG REASON, and said so rather than counted as a finding.
        print(f"CANNOT REACH THE PARSER: {why}")
        print("This is not evidence about e1-16 either way -- fix the harness first.")
        return 2

    cfg = cfg_defaults()
    red, green, wrong = [], [], []

    # STATIC FIRST, because a dangling name is invisible to every assertion below and
    # e1-16's edit is exactly the kind that leaves one (see undefined_names).
    deps = local_deps()
    dangling, ruff_why = undefined_names(deps)
    scope = ", ".join(os.path.basename(p) for p in deps)
    if dangling is None:
        wrong.append(f"{ruff_why} -- the static half of this acceptance did not run, and "
                     "a tool that could not run is not a clean result")
    elif dangling:
        wrong.append(f"{len(dangling)} unresolved name/parse error(s) across {scope}, so "
                     f"train.py cannot import regardless of what the parser accepts: "
                     f"{dangling[0]}")
    else:
        green.append(f"every name bound across {scope} (ruff F821)")

    # 1. Omitting any justified knob must be refused.
    for knob in RECIPE_FLAGS:
        if knob not in BASE and knob not in SWITCHES:
            wrong.append(f"{knob}: not in this test's BASE argv -- the test is incomplete")
            continue
        args, code = _parse(parser, _argv(omit=knob))
        if code is not None:
            green.append(f"{knob}: omission refused (exit {code})")
            continue
        # Accepted. Report what the RUN would actually use, which is not what the parser
        # returns: for the ten generated knobs argparse yields None and train.py:2110-2122
        # then leaves Cfg's own value standing, so `None` is the mechanism and the Cfg
        # field is the effective value. Reporting the parser's None alone would understate
        # this -- it reads like "no value" when the run proceeds with a real one.
        got = getattr(args, knob, "<no attribute>")
        eff = cfg.get(CFG_NAME.get(knob, knob), got) if got is None else got
        note = f"argparse None -> Cfg.{CFG_NAME.get(knob, knob)} = {eff!r}" if got is None \
            else f"argparse default {eff!r}"
        invisible = str(eff) == RECIPE.get(knob)
        red.append(f"{knob}: ACCEPTED, run would use {eff!r} ({note})"
                   + ("  -- EQUALS the recipe, so the omission is invisible downstream"
                      if invisible else f"  -- recipe wants {RECIPE.get(knob)}"))

    # 2. grad_ckpt specifically: store_true cannot express "not given", so absent and
    #    --no_grad_ckpt must be distinguishable. BooleanOptionalAction gives both.
    args, code = _parse(parser, _argv(omit="grad_ckpt"))
    if code is None:
        got = getattr(args, "grad_ckpt", "<none>")
        red.append(f"grad_ckpt: absent parsed as {got!r} -- store_true has no None "
                   "state, so this is indistinguishable from --no_grad_ckpt")
    # BooleanOptionalAction spells the negation with a HYPHEN -- "--no-grad_ckpt", not
    # "--no_grad_ckpt" -- because it prepends "--no-" to the option string verbatim. I
    # asserted the underscore form first and it stayed red under a correct fix, which is a
    # test wrong about its own subject. Both are accepted here: whichever e1-16 lands, the
    # property is that a negation EXISTS and is distinguishable from absence. run_ddp.sh
    # and any caller that turns it off must use the spelling argparse actually generates.
    negations = [f for f in ("--no-grad_ckpt", "--no_grad_ckpt")
                 if _parse(parser, _argv() + [f])[1] is None]
    if not negations:
        red.append("no --no-grad_ckpt / --no_grad_ckpt flag yet: with store_true, "
                   "'off' and 'not given' are the same argv, so an omission cannot be "
                   "refused without also losing the ability to disable it "
                   "(argparse.BooleanOptionalAction supplies the pair)")
    else:
        green.append(f"a negation exists and parses: {', '.join(negations)}")

    # 3. Passing a value explicitly must still work and must equal the recipe. This is
    #    the half that keeps the fix honest: making omission fatal is worthless if the
    #    explicit form stops meaning what the recipe says.
    args, code = _parse(parser, _argv())
    if code is not None:
        wrong.append(f"the COMPLETE command was refused (exit {code}) -- the fix would "
                     "break every real launch, not just the incomplete ones")
    else:
        for knob, want in BASE.items():
            got = getattr(args, knob, None)
            if got is None:
                wrong.append(f"{knob}: passed explicitly but parsed as None")
            # Compare as the flag's own TYPE, not as strings: --anneal_frac 0 parses to
            # 0.0 and "0" != "0.0" made this report a failure on a value that is exactly
            # right. A string comparison is not a value comparison.
            elif type(got)(want) != got:
                wrong.append(f"{knob}: passed {want} but parsed {got!r}")
        if not wrong:
            green.append(f"the complete command parses, all {len(BASE)} values as passed")

    for line in red:
        print(f"  RED    {line}")
    for line in wrong:
        print(f"  WRONG  {line}")
    for line in green:
        print(f"  green  {line}")
    print(f"\n{len(red)} red (the e1-16 work), {len(wrong)} red for the WRONG reason, "
          f"{len(green)} green")
    if wrong:
        print("A red-for-the-wrong-reason is not progress: fix those before reading the rest.")
        return 2
    if red:
        print("Expected while train.py is frozen. These turn green when e1-16 lands.")
        return 1 if strict else 0
    print("All green: every justified knob's omission is now unsatisfiable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

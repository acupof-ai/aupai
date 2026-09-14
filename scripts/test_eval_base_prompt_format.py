#!/usr/bin/env python3
"""No eval script may hand a BASE checkpoint a ChatML prompt (e1-22).

The property, in one line: for kind == "base", the prompt a generative eval builds
contains no ChatML marker. Not "the code looks right" -- the prompt string itself.

WHY THIS IS A DEFECT AND NOT A STYLE PREFERENCE, measured in this repo before this file
existed. scripts/loader.format_prompt emits

    <|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n

and `<|im_start|>` occurs 0 times in 168,000 corpus rows sampled across all 42 domains
(AGENTS.md:200 -- stated as a bound, since 0 of 4000 only puts a domain's rate under
0.075%). So a base checkpoint handed that prefix is not answering a question, it is
continuing a token sequence absent from its training data: it repeats the input or drifts
into web boilerplate. eval/score_code_exec.py:9-31 measured the size of it -- 41 of 2586
generations carried a code fence under ChatML (1.6%) against 469 of 497 under 1-shot plain
continuation (94.4%), same checkpoint family. Every base generative zero taken before
2026-09-02 measures response to an unseen prefix rather than capability.

Five scripts fed format_prompt to whatever checkpoint they were pointed at: eval/gsm8k.py,
eval/run_eval.py (GSM8K path), eval/math_zh.py, eval/math_hard.py, eval/code_zh.py. They
now take the format from eval/score_matrix.classify(cfg, name) via loader.prompt_fn.

TWO HALVES, because either alone can pass while the bug is live:

  the chooser   prompt_fn("base") is format_continuation and its output holds no marker
  the callers   no eval/*.py reaches format_prompt directly any more, and every one of
                the five threads a format from the load site

The second half is static (ast, not grep: a name in a docstring is not a call), and it has
to be, because the first half cannot see a script that imports format_prompt and calls it
regardless of what prompt_fn returns.

    python scripts/test_eval_base_prompt_format.py
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

CONVERTED = [
    "eval/gsm8k.py",
    "eval/run_eval.py",
    "eval/math_zh.py",
    "eval/math_hard.py",
    "eval/code_zh.py",
]
MARKERS = ("<|im_start|>", "<|im_end|>", "im_start", "im_end")

#: Modules OUTSIDE eval/ that build a ChatML prompt for a checkpoint they loaded. The
#: converted list above was scoped to eval/ and enumerated by hand, so a caller anywhere
#: else was invisible to it -- and one was live: algorithms/rlvr_trainer.py takes --resume,
#: calls format_prompt on every problem in the batch, and contained no reference to
#: classify or kind anywhere in the file (found 2026-09-02, e1-22's second half). Handed a
#: base checkpoint it would have run to completion: rewards near zero for a formatting
#: reason, advantages near zero with them, and the result reading as "RL did not help".
#: Discovered by walking the repo rather than by reading this list, which is why the walk
#: below now backs the list up.
TRAINERS = ["algorithms/rlvr_trainer.py"]

#: DUAL-ARM SCORERS that legitimately keep a format_prompt call for an EXPLICIT
#: post-SFT arm, while their base path is pure continuation. These are neither
#: CONVERTED (which asserts the script never calls format_prompt at all) nor
#: EXEMPT (their output IS a recorded eval): the safety property is that the
#: ChatML call is lexically unreachable for a base run -- it must sit inside an
#: `if <chatml flag>` branch, with argparse refusing to combine that flag with
#: the base continuation arm. _chatml_arm_gated() enforces that on the AST, with
#: a known-answer gated-vs-unconditional world in --selftest; a call that drifts
#: out from under the flag (which would silently hand every base checkpoint a
#: ChatML prompt) FAILs. eval/humaneval_gen.py: rstrip/standard base columns are
#: continuation; --chatml is the SFT arm and is refused with --rstrip_nl and n>1.
CHATML_ARM = [
    "eval/humaneval_gen.py",
]

#: chat.py is exempt, and not because it is small: it loads a fixed ckpt.pt and answers a
#: person typing at a prompt, so a nonsense generation is visible to whoever caused it and
#: nothing it produces enters a fact, a ledger row, or a decision.
EXEMPT = {"chat.py": "interactive, fixed ckpt.pt, output goes to a human not a record"}


def _ast_parents(tree):
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    return parent


def _is_positive_chatml_test(node):
    """True only for a direct POSITIVE reference to the chatml flag.

    Accepts exactly a bare Name `chatml` or an Attribute whose last segment is
    `chatml` (`args.chatml`) used as a truthiness test. Everything else is
    rejected, because each is a way to gate a call that a base run (chatml
    False) still takes:
      - UnaryOp(Not): `if not args.chatml:` -- the False/base branch enters.
      - Compare: `if args.chatml == False:` / `is False:` -- reads inverted.
      - another name containing the token: `if no_chatml:` -- substring match
        would call that a gate for the opposite flag.
      - BoolOp/binop: ambiguous polarity; not used by the real subject, so not
        accepted rather than guessed (extend explicitly if ever needed).
    """
    if isinstance(node, ast.Name):
        return node.id == "chatml"
    if isinstance(node, ast.Attribute):
        return node.attr == "chatml"
    return False


def chatml_call_arm_gated(src):
    """(gated, why): every format_prompt call must be under an `if <chatml flag>:`.

    The known-answer guard for CHATML_ARM scripts. A call nested in a branch
    whose test is a positive direct reference to the chatml flag is opt-in (a
    base run never takes it); a call with no such enclosing branch, or one
    gated by a negation/inverted-comparison/oppositely-named flag, reaches base
    runs and must fail. Returns (False, reason) rather than raising so the
    caller names the defect. Assumes the source parses (the caller skips
    SyntaxError files).
    """
    tree = ast.parse(src)
    parent = _ast_parents(tree)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "format_prompt"]
    if not calls:
        return False, "no format_prompt call present"
    for call in calls:
        node = call
        gated = False
        while node in parent:
            node = parent[node]
            if isinstance(node, ast.If) and _is_positive_chatml_test(node.test):
                gated = True
                break
        if not gated:
            return False, ("a format_prompt call is not gated by a positive "
                           "`if <chatml flag>:` branch, so a base run (chatml "
                           "False) can take it -- reject `if not chatml`, "
                           "`== False`, and no_chatml-style names")
    return True, ""


def chatml_callers_repo_wide():
    """[(relpath, consults_kind)] for every .py that CALLS format_prompt and loads a
    checkpoint, found by walking -- so a new caller cannot hide from a hand-kept list.

    The two hardcoded lists above are what this file checks in detail; this walk is the
    backstop that says the lists are complete. A hand-kept subject list going stale is
    the defect one layer up from the defect being checked.
    """
    out = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs
                   if d not in {".git", "__pycache__", "data", "runs", "node_modules"}]
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(base, f), ROOT)
            try:
                calls = called_names(rel)
            except (OSError, SyntaxError):
                continue
            if "format_prompt" not in calls:
                continue
            if not ({"load_checkpoint"} & calls):
                continue  # builds the string but holds no checkpoint (packers, tests)
            out.append((rel, bool({"classify", "prompt_fn"} & calls)))
    return out


def called_names(path):
    """Every function name CALLED in this module, from the ast.

    Not a grep for the string: this file's own prose names format_prompt repeatedly, and
    so do the converted scripts' comments explaining why they no longer call it. A
    substring search cannot tell an explanation from a call -- the same shape that made
    check_selftests_are_gated read my docstring as a carrier (e1-16, 2026-09-02).
    """
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def main():
    fails = []

    # 1. THE CHOOSER. prompt_fn is the single owner of "which format for which type";
    #    a copy in each of the five scripts would be five things to keep in step, which
    #    is the shape of the defect this replaces (two lists disagreeing).
    from loader import format_continuation, format_prompt, prompt_fn

    if prompt_fn("base") is not format_continuation:
        fails.append(f"prompt_fn('base') is {prompt_fn('base').__name__}, must be "
                     "format_continuation -- a base checkpoint cannot answer ChatML")
    for kind in ("sft", "rl"):
        if prompt_fn(kind) is not format_prompt:
            fails.append(f"prompt_fn({kind!r}) is {prompt_fn(kind).__name__}, must be "
                         "format_prompt -- ChatML IS the format these were tuned on, and "
                         "moving them to continuation would break the working path")

    # 2. THE PROMPT ITSELF, zero-shot and few-shot, since demos are joined into the string
    #    and a marker could enter through a demo as easily as through the question.
    probes = {
        "zero-shot": format_continuation("12+30=?"),
        "few-shot": format_continuation("2+2=?", [("1+1=?", "2"), ("3+3=?", "6")]),
    }
    for label, text in probes.items():
        for marker in MARKERS:
            if marker in text:
                fails.append(f"base {label} prompt contains {marker!r}: {text!r}")
    # A few-shot prompt that dropped its demos would pass the marker check and measure
    # something else entirely, so assert the demo actually landed.
    if "问：1+1=?\n答：2" not in probes["few-shot"]:
        fails.append(f"the demo pair is missing from the few-shot prompt: {probes['few-shot']!r}")

    # 3. THE CALLERS. A converted script must not call format_prompt at all -- with the
    #    format threaded from the load site there is no legitimate direct call left, and
    #    one would silently reintroduce ChatML for every checkpoint type.
    for path in CONVERTED:
        if not os.path.exists(os.path.join(ROOT, path)):
            fails.append(f"{path}: gone -- this test's subject list is stale")
            continue
        calls = called_names(path)
        if "format_prompt" in calls:
            fails.append(f"{path} still CALLS format_prompt; the format must come from "
                         "prompt_fn(classify(cfg, name)) at the load site")
        if "prompt_fn" not in calls:
            fails.append(f"{path} never calls prompt_fn, so nothing makes its format "
                         "follow the checkpoint type")
        if "classify" not in calls:
            fails.append(f"{path} never calls classify, so its prompt_fn argument is not "
                         "derived from the checkpoint -- a hardcoded kind is a guess")

    # 4. THE TRAINERS. A ChatML caller outside eval/ must consult the checkpoint's kind
    #    too, and for RLVR the correct behaviour is to REFUSE a base checkpoint rather than
    #    switch format: continuation-format RLVR is a different method, not a fallback.
    for path in TRAINERS:
        if not os.path.exists(os.path.join(ROOT, path)):
            fails.append(f"{path}: gone -- this test's subject list is stale")
            continue
        calls = called_names(path)
        if "classify" not in calls:
            fails.append(f"{path} calls format_prompt on a loaded checkpoint without "
                         f"calling classify -- a base checkpoint would be trained against "
                         f"rewards that measure format, not reasoning")
        src = open(os.path.join(ROOT, path), encoding="utf-8").read()
        if "classify" in calls and 'kind == "base"' not in src and "kind=='base'" not in src:
            fails.append(f"{path} calls classify but never branches on base -- reading the "
                         f"kind and not acting on it is not a guard")

    # 4b. THE DUAL-ARM SCORERS. format_prompt may stay, but only inside an explicit
    #     chatml-flag branch that a base run cannot take.
    for path in CHATML_ARM:
        if not os.path.exists(os.path.join(ROOT, path)):
            fails.append(f"{path}: gone -- this test's subject list is stale")
            continue
        src = open(os.path.join(ROOT, path), encoding="utf-8").read()
        ok, why = chatml_call_arm_gated(src)
        if not ok:
            fails.append(f"{path}: {why} -- a dual-arm scorer may call format_prompt "
                         "only inside an `if ...chatml...` opt-in branch")

    # 5. THE LISTS ARE COMPLETE. Both subject lists are hand-kept, and a hand-kept list
    #    going stale is the defect one layer above the one being checked: rlvr_trainer was
    #    missed for exactly that reason. The walk finds every ChatML caller holding a
    #    checkpoint and requires each to be covered here or explicitly exempt.
    known = set(CONVERTED) | set(TRAINERS) | set(CHATML_ARM)
    for rel, consults in chatml_callers_repo_wide():
        if os.path.basename(rel) in EXEMPT or rel in known:
            continue
        fails.append(f"{rel} calls format_prompt on a loaded checkpoint and is in no list "
                     f"here (consults kind: {consults}) -- add it to CONVERTED/TRAINERS/"
                     f"CHATML_ARM, or to EXEMPT with the reason it cannot mislead a record")

    for f in fails:
        print(f"  FAIL {f}")
    print(f"\n{len(fails)} failure(s); "
          f"{len(CONVERTED)} converted script(s), {len(TRAINERS)} trainer(s), "
          f"{len(CHATML_ARM)} chatml-arm scorer(s) and the "
          f"chooser checked; {len(chatml_callers_repo_wide())} ChatML caller(s) found "
          f"repo-wide")
    if fails:
        return 1
    print("OK: base checkpoints cannot be handed a ChatML prompt by any eval or trainer.")
    return 0


GATED_SRC = """
def _prompt(p):
    if args.chatml:
        from scripts.loader import format_prompt
        return format_prompt(p["prompt"])
    return p["prompt"].rstrip("\\n")
"""
# Each NEGATED form is a branch a base run (chatml False) TAKES, so the call
# inside must read as ungated. The first predicate ("chatml" substring) accepted
# all four -- de, 2026-09-14.
NEGATED_SRCS = {
    "not": """
def _prompt(p):
    if not args.chatml:
        from scripts.loader import format_prompt
        return format_prompt(p["prompt"])
""",
    "eq_false": """
def _prompt(p):
    if args.chatml == False:
        from scripts.loader import format_prompt
        return format_prompt(p["prompt"])
""",
    "is_false": """
def _prompt(p):
    if args.chatml is False:
        from scripts.loader import format_prompt
        return format_prompt(p["prompt"])
""",
    "opposite_name": """
def _prompt(p):
    if args.no_chatml:
        from scripts.loader import format_prompt
        return format_prompt(p["prompt"])
""",
}
UNGATED_SRC = """
def _prompt(p):
    from scripts.loader import format_prompt
    return format_prompt(p["prompt"])
"""
NO_CALL_SRC = "def _prompt(p):\n    return p['prompt'].rstrip('\\n')\n"


def selftest():
    """Known-answer worlds for the CHATML_ARM gate.

    The defect the category exists to catch: a format_prompt call drifting out
    from under the explicit chatml flag, after which every base run is handed
    ChatML. A hand-edit to the list would make main green with no such guard, so
    the discriminator must itself be tested on a gated and an unconditional call.
    """
    ok, _ = chatml_call_arm_gated(GATED_SRC)
    assert ok, "a format_prompt call inside `if args.chatml` must read as gated"
    # Negated / inverted / opposite-named gates must all FAIL: each is a branch
    # the base run takes.
    for label, src in NEGATED_SRCS.items():
        ok, why = chatml_call_arm_gated(src)
        assert not ok and "not gated" in why, (
            f"negated gate {label!r} was accepted as a positive chatml gate: {why}")
    ok, why = chatml_call_arm_gated(UNGATED_SRC)
    assert not ok and "not gated" in why, (
        f"an unconditional format_prompt call must FAIL, got ok={ok} why={why}")
    ok, _ = chatml_call_arm_gated(NO_CALL_SRC)
    assert not ok, "a script with no format_prompt call cannot satisfy the CHATML_ARM contract"
    # And the real subject must currently be gated, or this test proves nothing about it.
    real = open(os.path.join(ROOT, "eval/humaneval_gen.py"), encoding="utf-8").read()
    ok, why = chatml_call_arm_gated(real)
    assert ok, f"eval/humaneval_gen.py no longer satisfies CHATML_ARM: {why}"
    print("eval_base_prompt_format selftest OK: positive gate accepted; negated "
          "(not/==False/isFalse/no_chatml), unconditional, and no-call refused; "
          "humaneval_gen.py verified gated")
    return 0


if __name__ == "__main__":
    import sys as _sys
    if "--selftest" in _sys.argv:
        # The commit hook always passes --selftest; this file predates argparse and that
        # flag historically ran main() itself, so --selftest runs the full walk AND the
        # known-answer worlds -- the worlds alone would not gate the live repo.
        rc = main()
        if rc:
            _sys.exit(rc)
        _sys.exit(selftest())
    _sys.exit(main())

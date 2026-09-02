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

    for f in fails:
        print(f"  FAIL {f}")
    print(f"\n{len(fails)} failure(s); "
          f"{len(CONVERTED)} converted script(s) and the chooser checked")
    if fails:
        return 1
    print("OK: base checkpoints cannot be handed a ChatML prompt by any eval script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

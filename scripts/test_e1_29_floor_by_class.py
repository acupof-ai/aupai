#!/usr/bin/env python3
"""The class split is a real split, and it is not the data's own broken label.

# restartable: in-process assertions only. Milliseconds, no GPU, no files written.

The floor-gap decomposition rests entirely on classify(): if it mislabels, the English-only gap is
a number about the wrong items. And the split must ADD BACK UP to the pass it splits -- a per-class
table that does not reconstruct the total is a second measurement, not a decomposition.

    python3 scripts/test_e1_29_floor_by_class.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import e1_29_floor_by_class as M  # noqa: E402


def main():
    fails = []

    # 1. THE FOUR CLASSES, on the real shapes seen in the held-out set (quoted from items I read
    #    while probing it).
    cases = [
        ("古文阅读 阅读下文，完成下面小题。【甲】潭中鱼可百许头，皆若空游无所依，日光下澈",
         "善：善于，擅长 句意：听说有个麻桥人庞安常擅长医术但耳朵聋。", "zh-prose"),
        ("小明的书包里有3本书和1个铅笔盒，小红的书包里有2本书和2个铅笔盒，谁的书包更重？",
         "首先，我们需要知道每本书和每个铅笔盒的重量。假设每本书重500克。", "zh-prose"),
        ("Can you create an intelligent optimised caching system in C#",
         "<Thought> Alright, I need to create a cache. </Thought>\n```csharp\npublic class Cache",
         "en-code"),
        ("What is the capital of France?", "The capital of France is Paris.", "en-prose"),
        ("用 Python 实现快排", "```python\ndef quicksort(a):\n    return a\n```", "zh-code"),
        # A Chinese item that is MOSTLY code and latin identifiers must still clear the CJK share.
        # This is the case an absolute CJK count gets wrong in the other direction.
        ("请写一个函数计算斐波那契数列并解释它的时间复杂度是多少",
         "```python\ndef fib(n):\n    return n if n < 2 else fib(n-1)+fib(n-2)\n```\n"
         "时间复杂度是指数级的。", "zh-code"),
        # And a long English answer with one Chinese clause must stay English.
        ("Explain the GIL in CPython and why it matters for threading performance in practice",
         "The Global Interpreter Lock serializes bytecode execution, so CPU-bound threads do not "
         "run in parallel; use multiprocessing instead. 中文", "en-prose"),
    ]
    for q, ans, want in cases:
        got = M.classify(q, ans)
        if got != want:
            fails.append(f"classify({q[:24]!r}...) = {got}, expected {want}")

    # 2. A SINGLE CJK CHARACTER IS NOT CHINESE TEXT. An English answer mentioning 中文 must stay
    #    en-*, or the Chinese class absorbs English items and the English-only gap is computed on
    #    a polluted subset -- in the direction that would hide the confound being tested.
    if M.classify("Translate 中 to English", "The character 中 means middle.") != "en-prose":
        fails.append("two CJK characters in an English item made it Chinese; ZH_MIN is not "
                     "protecting the English subset")
    if M.ZH_MIN_FRAC <= 0.0 or M.ZH_MIN_FRAC > 0.5:
        fails.append(f"ZH_MIN_FRAC={M.ZH_MIN_FRAC} is outside a sane band; a share this "
                     f"extreme either absorbs English items or rejects Chinese ones")

    # 3. CODE IS DETECTED IN THE ANSWER, NOT THE QUESTION. "write a function that..." is a prose
    #    answer to a code question; scoring is over the ANSWER (prompts are masked), so the class
    #    must follow what was actually scored.
    if M.classify("write a function that adds", "Sure, add the two numbers together.") != "en-prose":
        fails.append("a code-flavoured QUESTION with a prose answer was called code -- the class "
                     "must follow the scored text, which is the answer")
    if M.classify("explain recursion", "```python\ndef f(): pass\n```") != "en-code":
        fails.append("a code answer to a prose question was not called code")

    # 4. THE MARKERS EXCLUDE PUNCTUATION THAT APPEARS IN CHINESE PROSE. A brace or semicolon is
    #    not a marker, deliberately: over-calling code would manufacture a prose-vs-code contrast.
    for m in M.CODE_MARKS:
        if m.strip() in ("{", "}", ";", ":", "(", ")"):
            fails.append(f"{m!r} is a bare punctuation marker -- it will fire on prose")
    if M.classify("论述", "关于函数的定义{我们需要}考虑；以下几点：") != "zh-prose":
        fails.append("Chinese prose containing braces and semicolons was called code")

    # 5. THE ENGLISH-ONLY AGGREGATION IS BYTE-WEIGHTED, NOT A MEAN OF RATIOS. Averaging two
    #    classes' nll/byte would weigh a 100-byte class like a 100,000-byte one. Tested as the
    #    ARITHMETIC PROPERTY on a fixture, not by grepping for an expression: the first version of
    #    this check grepped `ours_nll_per_byte * bytes_ours` and went red when the script started
    #    summing the stored per-class nll directly -- which is the same quantity without the
    #    round-trip through a rate. A test that pins the spelling fails on an improvement.
    tiny = {"en-code": {"ours_nll": 1.0, "ctrl_nll": 1.0, "bytes": 10, "n": 1},
            "en-prose": {"ours_nll": 100.0, "ctrl_nll": 900.0, "bytes": 10000, "n": 99}}
    weighted = sum(d["ctrl_nll"] for d in tiny.values()) / sum(d["ours_nll"] for d in tiny.values())
    mean_of_ratios = sum(d["ctrl_nll"] / d["ours_nll"] for d in tiny.values()) / len(tiny)
    if abs(weighted - 8.9218) > 0.01 or abs(mean_of_ratios - 5.0) > 0.01:
        fails.append(f"fixture is wrong: weighted={weighted:.4f} mean_of_ratios={mean_of_ratios}")
    src = open(os.path.join(ROOT, "scripts", "e1_29_floor_by_class.py")).read()
    import re as _re
    eo = _re.search(r'out\["english_only"\] = \{(.*?)\}\n', src, _re.S)
    if not eo:
        fails.append("english_only is not assigned as a dict literal; the aggregation cannot be "
                     "checked, and a mean of per-class ratios would let a 10-byte class swing it")
    elif "/ len(" in eo.group(1) or "statistics" in eo.group(1) or "mean(" in eo.group(1):
        fails.append(f"english_only averages per-class values: {eo.group(1)[:120]!r}")
    elif 'sum(gaps[c]["ctrl_nll"] for c in en)' not in src or \
         'sum(gaps[c]["bytes"] for c in en)' not in src:
        fails.append("english_only does not sum per-class nll and bytes before dividing -- that "
                     "is the only form that is byte-weighted")

    # 6a. BOTH ARMS MUST SCORE ONE POPULATION. This is the defect in the published pair, not a
    #     hypothetical: floor_ours.json and floor_control.json both carry supervised_bytes
    #     10554038 while dropping 28 and 220 overlong items, so the control's 0.903758 divides a
    #     10201-item loss by a 10421-item byte count. Inherited per class it would be worse than
    #     a small bias -- the dropped items are the longest, and length tracks code and English,
    #     the two axes being split on.
    #
    #     Checked by PARSING, not by grepping for a substring. Two earlier versions of this check
    #     grepped `tok["control"][0]` and `gap_shared_population`; both stayed green when the
    #     computation was deleted, because the same strings appear later in a print() and in a
    #     dict key. A mention is not a computation, and a `not in src` grep cannot tell them
    #     apart. Executing it instead needs two GPU models, so the middle path is the AST.
    import ast
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    if fn is None:
        fails.append("main() not found; the checks below cannot be made")
    else:
        assigns = {}
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        assigns[t.id] = n.value
                    elif isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant):
                        assigns[f"[{t.slice.value}]"] = n.value
        both = assigns.get("both")
        if both is None:
            fails.append("main() never assigns `both`: the two arms are not intersected, so each "
                         "scores its own kept-set and every per-class ratio compares two "
                         "populations")
        elif not (isinstance(both, ast.BinOp) and isinstance(both.op, ast.BitAnd)):
            got = (type(both.op).__name__ if isinstance(both, ast.BinOp)
                   else type(both).__name__)
            fails.append(f"`both` is built with {got}, not a set intersection (BitAnd) -- it must "
                         f"be the AND of both arms' kept ids; a union would ADD each arm's "
                         f"unscorable items instead of removing them")
        else:
            arms = {c.value for c in ast.walk(both) if isinstance(c, ast.Constant)
                    and c.value in ("ours", "control")}
            if arms != {"ours", "control"}:
                fails.append(f"`both` intersects {sorted(arms)}, not both arms")
        if "[gap_shared_population]" not in assigns:
            fails.append("no whole-population gap is COMPUTED on the shared population (a "
                         "print() mentioning it does not count) -- the per-class split then has "
                         "no headline it is a split OF")
    if "REFUSING: class" not in src:
        fails.append("a per-class n/bytes mismatch between arms does not refuse; after the "
                     "intersection it is unreachable, so it must assert rather than warn")

    # 6. THE PASS MUST BE RECONSTRUCTED, i.e. the script refuses when per-item does not sum to the
    #    total it splits. Grepped rather than executed because the alternative needs two GPU
    #    models; a refusal that exists only in intent is not a refusal.
    if "the split is not a split of this number" not in src:
        fails.append("nothing checks that the per-class split reconstructs the scored pass")
    if "REFUSING" not in src or a_ctrl_guard(src) is False:
        fails.append("the control floor's model dir is not guarded -- eval_heldout's --model_dir "
                     "default is NOT the model behind the published 0.903758")

    # 7. main() MUST BE COMPILABLE-CLEAN, because nothing here executes it. Every check above
    #    calls classify() or greps the source, so a name that exists only inside main() is never
    #    evaluated -- and `ZH_MIN` survived there after the constant was renamed to ZH_MIN_FRAC,
    #    all five mutations green. It would have raised NameError while building the output dict,
    #    i.e. AFTER loading both models onto a card and scoring both passes: ~4 minutes of GPU
    #    thrown away at the last statement. A static pass over the whole module costs 30ms and
    #    catches the entire class, not just this instance.
    import subprocess
    pf = subprocess.run([sys.executable, "-m", "pyflakes",
                         os.path.join(ROOT, "scripts", "e1_29_floor_by_class.py")],
                        capture_output=True, text=True)
    if pf.returncode == 2 and "No module named" in pf.stderr:
        fails.append("pyflakes is not installed, so undefined names in main() go unchecked")
    elif pf.stdout.strip() or pf.returncode not in (0, 1):
        for line in pf.stdout.strip().splitlines():
            fails.append(f"pyflakes: {line}")
    elif pf.returncode == 1:
        fails.append(f"pyflakes failed: {pf.stderr.strip()[:200]}")

    for f in fails:
        print(f"FAIL: {f}", file=sys.stderr)
    if fails:
        print(f"{len(fails)} check(s) failed", file=sys.stderr)
        return 1
    print("e1_29_floor_by_class OK: four classes on real held-out shapes; incidental CJK stays "
          "English; code follows the answer not the question; no bare-punctuation markers; "
          "english_only is byte-weighted; the split must reconstruct the pass; main() has no "
          "undefined names (nothing here runs it)")
    return 0


def a_ctrl_guard(src):
    return "runs/e1_control_model_fp.json records its weight hash" in src


if __name__ == "__main__":
    sys.exit(main())

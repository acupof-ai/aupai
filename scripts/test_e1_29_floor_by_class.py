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

    # 6a. BOTH ARMS MUST SCORE ONE POPULATION, checked here rather than assumed from whatever
    #     --ids the caller passed. On the published floors this is a no-op -- both restrict to
    #     ids_shared.txt and report evaluated_ids_sha256 cae4daf7ad59388c over 10,421 items, so
    #     2.004x is sound. (I reported the opposite first, reading floor_*.json's dropped_overlong
    #     28 vs 220 as two populations; that field is counted at eval_heldout.py:515, BEFORE the
    #     --ids restriction at 531-547, and describes the 10,641-row file rather than the scored
    #     set. A field name is not the field's definition.)
    #
    #     The check still earns its place: if a class ever holds different items in the two arms,
    #     every per-class ratio silently becomes two measurements side by side, and per class that
    #     is worse than a small bias because the longest items are also the likeliest to be code
    #     and English -- the two axes this splits on.
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

    # 6b. THE SCORER'S ARGUMENTS MUST BE THE RIGHT KINDS, checked without a GPU. `pad_id` goes
    #     straight into `torch.tensor(xs, dtype=torch.long)`, so a non-int kills the run AFTER the
    #     model is loaded and 10,421 items are tokenized -- which is exactly what happened:
    #     load_ours returns (model, ck.get("vocab_id")), a vocab identifier STRING, and
    #     load_control returns (model, None). Destructuring either as `model, pad` and passing it
    #     on died with "'str' object cannot be interpreted as an integer". A two-tuple's second
    #     slot is not labelled by what the caller needs it to be.
    if fn is not None:
        # Written as a plain loop. The comprehension version of this -- `[... for n in walk if
        # cond for t in n.targets for e in t.elts if ...]` -- read as if the guard applied to the
        # whole chain and reported 0 FAIL on the very bug it was written for; the same walk in a
        # loop finds `pad_id` immediately. A check whose own logic is subtle enough to be wrong
        # silently is worth less than a longer one that is obviously right.
        loader_pad = []
        for n in ast.walk(fn):
            if not (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)):
                continue
            f = n.value.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name not in ("load_ours", "load_control"):
                continue
            for t in n.targets:
                if not isinstance(t, ast.Tuple):
                    continue
                for e in t.elts:
                    if isinstance(e, ast.Name) and "pad" in e.id.lower():
                        loader_pad.append((e.id, n.lineno))
        if loader_pad:
            nm, ln = loader_pad[0]
            fails.append(f"line {ln} takes pad from a loader's return tuple (`{nm}`) -- load_ours "
                         f"returns a vocab-id string there and load_control returns None; neither "
                         f"is a pad id, and score() puts it into a long tensor")

        if "not isinstance(pad_id, int)" not in src:
            fails.append("nothing asserts pad_id is an int before score() -- a string there "
                         "crashes only after the model is loaded and every item tokenized")
        for want in ("token_to_id", "eos_token_id"):
            if want not in src:
                fails.append(f"pad_id is not derived via {want}, which is how eval_heldout's own "
                             f"main() gets it (lines 563-569)")

    # 6c. THE RECONSTRUCTION TOLERANCE IS RELATIVE. An absolute one is a different test at every
    #     magnitude, and this cost a completed scoring pass: at 1e-3 absolute the guard refused
    #     4759488.235578 against 4759488.226891 -- off by 0.0087, which is 1.8e-9 RELATIVE. The
    #     aggregate is one fused float32 reduction per batch; per-item is reduction="none", then
    #     .sum(dim=1), then a Python sum over 10,421 floats. Same values, different summation
    #     trees, so ~1e-9 drift is non-associativity, not a defect. 1e-3 was calibrated on a 2-row
    #     fixture (~1e-4 relative there); at 4.76M it means 2e-10, i.e. bit-exactness. A guard
    #     that tightens as the data grows fires first on the largest and most real run.
    if src.count("max(1e-6 * abs(") < 2:
        fails.append("a reconstruction tolerance is still absolute; at 4.76M nll an absolute 1e-3 "
                     "demands bit-exactness between two different summation orders")
    for drift, total, want_ok in ((0.0087, 4759488.226891, True),   # the real drift: must pass
                                  (0.0087, 2.0, False),            # same drift, tiny total: fail
                                  (18.0, 4759488.226891, False),   # one lost 40-byte item: fail
                                  (60.0, 4759488.226891, False)):  # 1.3e-5 relative: a real defect
        ok = drift <= max(1e-6 * abs(total), 1e-3)
        if ok != want_ok:
            fails.append(f"tolerance shape wrong: drift {drift} on total {total} -> "
                         f"{'pass' if ok else 'fail'}, expected {'pass' if want_ok else 'fail'}")

    # 6d. DISPERSION ACROSS CLASSES IS THE DISCRIMINANT, so it must be COMPUTED, not just
    #     discussed. English carries 52.5% of the bytes, so en-only landing near the
    #     whole-population gap is compositionally forced and settles nothing; what separates a real
    #     language advantage from a small confound is whether zh-prose sits well above the en-*
    #     classes or all four cluster near the mean.
    if fn is not None and "[gap_dispersion]" not in assigns:
        fails.append("gap_dispersion is never computed -- en-only alone cannot separate 'the "
                     "confound is small' from 'English dominates the denominator'")

    # 6e. THE PER-CLASS SPLIT MUST RECONSTRUCT THE SCORED PASS, and the control's model dir must be
    #     the one behind the published number.
    #
    #     Checked structurally. The grep version looked for "the split is not a split of this
    #     number" and went red while the guard was fully present -- rewrapping the f-string put
    #     "the split is not a " and "split of this number" in adjacent literals, so the sentence
    #     exists in the output and not in the source. Third time a substring check has been wrong
    #     about this file (a print() mention, an improvement, now a line wrap): the source text is
    #     not the behaviour, and only the syntax tree is.
    if fn is not None:
        exits = [n for n in ast.walk(fn)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "exit"]
        recon = [n for n in exits
                 if any("per-item NLLs sum to" in c.value for c in ast.walk(n)
                        if isinstance(c, ast.Constant) and isinstance(c.value, str))]
        if not recon:
            fails.append("nothing refuses when the per-item NLLs fail to reconstruct the scored "
                         "pass -- the per-class table would then be a split of a different number")
        if len(exits) < 4:
            fails.append(f"only {len(exits)} sys.exit guards in main(); the ctrl-dir, empty-"
                         f"intersection, reconstruction and per-class-population refusals are all "
                         f"load-bearing")
    if a_ctrl_guard(src) is False:
        fails.append("the control floor's model dir is not guarded -- eval_heldout's --model_dir "
                     "default is NOT the model behind the published 0.903758")



    # 7. main() MUST BE COMPILABLE-CLEAN, because nothing here executes it. Every check above
    #    calls classify() or greps the source, so a name that exists only inside main() is never
    #    evaluated -- and `ZH_MIN` survived there after the constant was renamed to ZH_MIN_FRAC,
    #    all five mutations green. It would have raised NameError while building the output dict,
    #    i.e. AFTER loading both models onto a card and scoring both passes: ~4 minutes of GPU
    #    thrown away at the last statement. A static pass over the whole module costs 30ms and
    #    catches the entire class, not just this instance.
    #     A missing pyflakes must be LOUD, not a silent skip -- a static check that quietly
    #     disappears on the machine that actually runs the job is the same "guard nobody calls"
    #     shape this file keeps finding. It is detected by the message, not by an exit code: I
    #     guessed `python3 -m` returns 2 for a missing module and it returns 1, so the pod's real
    #     failure surfaced as a generic "pyflakes failed" instead of the sentence written for it.
    import subprocess
    pf = subprocess.run([sys.executable, "-m", "pyflakes",
                         os.path.join(ROOT, "scripts", "e1_29_floor_by_class.py")],
                        capture_output=True, text=True)
    if "No module named pyflakes" in pf.stderr:
        alt = subprocess.run(["ruff", "check", "--select", "F821,F811,F841",
                              os.path.join(ROOT, "scripts", "e1_29_floor_by_class.py")],
                             capture_output=True, text=True)
        if "No such file" in alt.stderr or alt.returncode not in (0, 1):
            fails.append("neither pyflakes nor ruff is available, so an undefined name in main() "
                         "would only surface after both models are loaded -- install one, or run "
                         "this check on a machine that has it before spending card time")
        elif alt.returncode == 1:
            for line in alt.stdout.strip().splitlines():
                if line.strip():
                    fails.append(f"ruff: {line}")
    elif pf.stdout.strip():
        for line in pf.stdout.strip().splitlines():
            fails.append(f"pyflakes: {line}")
    elif pf.returncode not in (0, 1):
        fails.append(f"pyflakes exited {pf.returncode}: {pf.stderr.strip()[:200]}")

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

#!/usr/bin/env python3
"""The 2x2 scorer is language-symmetric, and the truncation still fires in both languages.

# restartable: in-process assertions only. Milliseconds, no GPU, no model, no files written.

    python3 eval/test_l1_fewshot_2x2.py

WHY THIS EXISTS. eval/l1_fewshot.py grew a --demo_lang axis so that "does the demo language
drive the two arms' answer-present gap" is measured rather than assumed. That axis touches the
three things the number is made of: the prompt scaffold, the stop sequence model_turn cuts on,
and the answer markers the present-rate counts. Every one of them was Chinese-only, and a
Chinese-only scorer applied to an English-demo arm does not measure a weaker model -- it
measures its own vocabulary, and reports the difference as capability.

The failure would be silent and would land exactly on the comparison the 2x2 was built for.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("FLA_FLASH_KDA", "0")

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "l1f", os.path.join(ROOT, "eval", "l1_fewshot.py"))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)


def main():
    fails = []
    demos = [("小明有3个苹果", "3个。答案是：3"), ("2+2 等于几", "\\boxed{4}")]

    # 1. THE SCAFFOLD SWITCHES, AND THE QUESTION NEVER DOES. The questions are the Chinese
    #    math-500 items in both cells: that is what makes demo language a single variable.
    for lang, want_in, want_out in (("zh", "题目：", "Problem: "),
                                    ("en", "Problem: ", "题目：")):
        p = L.build_prompt(demos, "小红有5支笔", lang)
        if want_in not in p:
            fails.append(f"--demo_lang {lang} prompt lacks its own opener {want_in!r}")
        if want_out in p:
            fails.append(f"--demo_lang {lang} prompt carries the OTHER language's opener "
                         f"{want_out!r} -- the scaffold did not switch")
        if "小红有5支笔" not in p:
            fails.append(f"--demo_lang {lang} dropped the target question; the questions must "
                         f"be identical across cells or the axis is not a single variable")

    # 2. TRUNCATION FIRES IN BOTH LANGUAGES. model_turn is the guard against the model
    #    inventing its own next problem and the last-box rule grading an answer to a question
    #    nobody asked (43.5% of 3-demo generations open a new problem). A literal zh opener
    #    here would stop cutting under --demo_lang en, i.e. the guard would vanish on the arm
    #    it was extended for, and the resulting wrong answers would read as a weaker model.
    zh_gen = "答案是：7\n\n题目：另一个问题\n解答：答案是：99"
    en_gen = "The answer is: 7\n\nProblem: another one\nSolution: The answer is: 99"
    if L.model_turn(zh_gen, "zh") != "答案是：7\n\n":
        fails.append(f"zh truncation wrong: {L.model_turn(zh_gen, 'zh')!r}")
    if L.model_turn(en_gen, "en") != "The answer is: 7\n\n":
        fails.append(f"en truncation wrong: {L.model_turn(en_gen, 'en')!r}")
    if "99" in L.model_turn(en_gen, "en"):
        fails.append("en truncation let the fabricated second problem through -- the last-box "
                     "rule would grade the answer to a question nobody asked")

    # 3. THE ANSWER MARKERS ARE SYMMETRIC. This is the present-rate's definition, and the
    #    present rate is the ONE layer where the two arms might genuinely differ.
    for s, why in (("答案是：42", "Chinese marker"),
                   ("The answer is: 42", "English marker"),
                   ("the answer is 42", "lowercase English, no colon")):
        if not L.ANS_RE.search(s):
            fails.append(f"ANS_RE misses the {why}: {s!r} -- an English-demo generation that "
                         f"answers would be counted as producing no answer")
    if L.ANS_RE.search("there is no answer here"):
        fails.append("ANS_RE fires on prose containing no answer marker")

    # 4. SCORING IS LANGUAGE-SYMMETRIC ON THE SAME GOLD. Same numeric answer, same verdict,
    #    whichever language the model wrapped it in -- otherwise the en cell is penalised by
    #    the scorer rather than by the model.
    gold = "\\boxed{7}"
    for gen, lang in (("答案是：7", "zh"), ("The answer is: 7", "en"),
                      ("\\boxed{7}", "zh"), ("\\boxed{7}", "en")):
        if L.score(gen, gold, lang) != 1.0:
            fails.append(f"score({gen!r}, lang={lang}) != 1.0 -- a correct answer marked wrong "
                         f"because of the language it was written in")
    for gen, lang in (("答案是：8", "zh"), ("The answer is: 8", "en")):
        if L.score(gen, gold, lang) != 0.0:
            fails.append(f"score({gen!r}, lang={lang}) != 0.0 -- a wrong answer marked correct")

    # 5. THE PRESENT-RATE TEST IS DERIVED FROM ANS_RE, NOT RESTATED. The two were separately
    #    hardcoded to 答案是 and drifted apart the moment English was added -- two lines apart
    #    in the same file. Grepped because the counter lives inside main()'s batch loop.
    src = open(os.path.join(ROOT, "eval", "l1_fewshot.py"), encoding="utf-8").read()
    if 'n_box += int("\\\\boxed" in turn or "答案是" in turn)' in src:
        fails.append("the present-rate counter still hardcodes 答案是 while ANS_RE accepts both "
                     "languages -- the en arm's answers would not be counted")
    if "ANS_RE.search(turn)" not in src:
        fails.append("the present-rate counter does not use ANS_RE, so the markers it accepts "
                     "and the markers the scorer accepts can drift apart")

    # 6. EVERY VARIABLE THAT CHANGES THE NUMBER IS IN THE PREDICTIONS PATH. Four cells writing
    #    two filenames is how be.l1_3shot_retracted happened: preds_l1_d3.jsonl was overwritten
    #    by an unlogged run and three published numbers lost their artifact.
    for tok in ('f".{args.demo_lang}"', '(".hf" if args.hf else "")'):
        if tok not in src:
            fails.append(f"the preds path omits {tok} -- two cells of the 2x2 would collide on "
                         f"one file, and --force would silently overwrite the first")

    for f in fails:
        print(f"FAIL: {f}", file=sys.stderr)
    if fails:
        print(f"{len(fails)} check(s) failed", file=sys.stderr)
        return 1
    print("l1_fewshot 2x2 OK: scaffold switches while questions do not; truncation fires in "
          "both languages; answer markers and scoring are language-symmetric; the present-rate "
          "derives from ANS_RE; all four cells write distinct predictions files")
    return 0


if __name__ == "__main__":
    sys.exit(main())

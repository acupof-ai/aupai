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

    # 7. THE RESTARTABILITY MARKER MUST BE TRUE OF THE FILE, NOT JUST THE LOOP. open_artifact's
    #    default mode is "w", so a plain rerun TRUNCATES: "rows are written as they are scored"
    #    was true of the loop and false of the file. A 497-problem cell is ~15 minutes of
    #    generation, so the difference is the whole run.
    if "# restartable:" not in src.split('"""')[0]:
        fails.append("no restartability marker above the docstring, and this script generates for "
                     "minutes per cell")
    if 'mode="a" if args.resume else "w"' not in src:
        fails.append("--resume does not switch open_artifact to append mode, so a resumed run "
                     "truncates the rows it claims to continue")
    if "fout.flush()" not in src:
        fails.append("rows are not flushed per write; Python buffers ~8 KB against ~1 KB rows, so "
                     "an interrupt drops several completed rows and the marker overstates what "
                     "survives")
    if "seen.add(" not in src or "not in seen" not in src:
        fails.append("--resume does not skip questions already in the file")
    for tok in ("correct += int(d[\"ok\"])", "n_box += int"):
        if tok not in src:
            fails.append(f"resume does not rebuild {tok!r} from the existing rows -- acc and "
                         f"answer-present would cover only the tail while carrying the whole "
                         f"population's label")
    if "n_target = total + len(evals)" not in src:
        fails.append("the progress denominator is not fixed before the loop; computed from "
                     "`total` it grows with the numerator and always prints n/n")
    # THE RESUME MUST READ THE FILE THE RUN WRITES. With --run, open_artifact writes a versioned
    # path; reading the unversioned one rebuilds counts from a different run's rows and appends
    # to a file whose contents were never examined. The harness caught this as "reports
    # preds_path, not out_path" -- a second reader finding the same class of bug the ledger
    # already knows about (l1_15b_final attested the file it did not touch).
    if "out_path = versioned_path(out_path, args.run)" not in src:
        fails.append("out_path is not versioned before it is used, so --resume would read the "
                     "unversioned file while open_artifact wrote the versioned one")
    if "run=args.run" in src:
        fails.append("open_artifact still gets run=, but out_path is already versioned -- the "
                     "path would be versioned twice (verified: preds_x.r1.r1.jsonl)")

    # 8. THE DECODER IS THE SAME ON BOTH ARMS, AND THE RUN RECORDS WHICH DECODER IT WAS.
    #    train.generate_batch gates rep_stop on `tokenizer is not None` (train.py:944), so the
    #    original call here -- which omitted the argument -- ran OUR arm with NO repetition stop
    #    while the control arm, whose tokenizer was passed explicitly, ran with one. The 2x2's
    #    length gap (794-851 characters against 84-86) and our 98.8% loop rate are partly that:
    #    a decoder difference read as a model difference, from an argument whose absence means a
    #    silent False rather than an error.
    if "tokenizer=tok" not in src:
        fails.append("generate_batch is called without tokenizer=, so rep_stop is silently False "
                     "for our arm while the control arm has it -- the arms would run different "
                     "decoders and the length gap would be an artefact of the call site")
    if src.count("rep_stop=not args.no_rep_stop") < 2:
        fails.append("--no_rep_stop does not reach both decode paths; one arm would keep the stop "
                     "while the other loses it")
    if '"rep_stop": not args.no_rep_stop' not in src:
        fails.append("the summary JSON does not record whether the stop was on, so two runs "
                     "differing only in the decoder are indistinguishable afterwards")
    if '".norepstop" if args.no_rep_stop' not in src:
        fails.append("the preds path omits the rep_stop state, so a stop-off rerun overwrites the "
                     "stop-on rows")

    # 9. answer_marker IS THE WHOLE PREDICATE, AND EVERY READER CALLS IT. answer-present is a
    #    DISJUNCTION -- \boxed OR ANS_RE -- and l1_2x2_diagnose imported ANS_RE alone as "the
    #    marker". That agreed with the scorer on one BRANCH and read as full agreement: it reported
    #    0/497 markers for cells published at 37.0%, because the boxed branch is the one our arm
    #    uses. A shared constant is not enough when the predicate spans two operators; the predicate
    #    itself has to be the shared thing.
    for text, want, why in (("答案是：42", True, "ANS_RE branch alone"),
                            ("所以 \\boxed{42}", True, "boxed branch alone"),
                            ("The answer is: 42", True, "English ANS_RE branch"),
                            ("这里没有答案", False, "neither branch")):
        got = L.answer_marker(text) is not None
        if got != want:
            fails.append(f"answer_marker({text!r}) -> {got}, want {want} ({why} dropped) -- a "
                         f"reader taking only the other branch would report zero markers on rows "
                         f"that have one")
    # THE POSITION IS THE MIN OF THE BRANCHES, so "did the decoder reach a marker" is answered by
    # the FIRST marker, not by whichever branch is checked first in the source.
    if L.answer_marker("答案是：1 然后 \\boxed{2}") != 0:
        fails.append("answer_marker does not return the EARLIEST marker; the position would depend "
                     "on branch order and would overstate how far the decoder had to run")
    dsrc = open(os.path.join(ROOT, "eval", "l1_2x2_diagnose.py"), encoding="utf-8").read()
    if "_L.answer_marker" not in dsrc:
        fails.append("l1_2x2_diagnose does not import answer_marker, so its marker-position column "
                     "and the published answer-present rate can count different things")
    # AST, NOT grep. The first version of this check read the source text and went RED on the
    # COMMENT that explains the bug -- the fixed file failed its own guard for describing what it
    # fixed. Same shape as the two greps that stayed GREEN by matching a print(): a name in prose
    # and a name in an expression are different facts, and only the parser separates them.
    import ast
    dtree = ast.parse(dsrc)
    ans_re_uses = [n for n in ast.walk(dtree)
                   if (isinstance(n, ast.Name) and n.id == "ANS_RE")
                   or (isinstance(n, ast.Attribute) and n.attr == "ANS_RE")]
    if ans_re_uses:
        fails.append(f"l1_2x2_diagnose USES ANS_RE at line(s) "
                     f"{sorted({n.lineno for n in ans_re_uses})} -- that is one branch of the "
                     f"predicate, and taking it for the whole is the 0/497-vs-37.0% bug")

    if src.count("answer_marker(turn) is not None") < 2:
        fails.append("a present-rate counter still spells out the disjunction instead of calling "
                     "answer_marker; the resume path and the live path would be able to drift")

    # 10. THE PREDICATE IS EXERCISED WHERE ITS OUTPUT IS LARGE, not only where it is near zero.
    #     This is the check that would have caught the 0/497 bug, and the reason it is separate from
    #     group 9: the four mutations in group 9 prove the GUARDS fail on a defect, which is a
    #     different claim from "the predicate was ever asked a question whose answer is big". The
    #     broken version ran over real data, wrote a real JSON, and raised nothing -- because the
    #     control arm is SUPPOSED to be near zero, so a predicate that always returns None is
    #     indistinguishable from a correct one on those cells. A published 37.0% cell existed, and
    #     nothing compared against it. Both magnitudes, from a source outside the predicate.
    HIGH = ("解答：3 个苹果。\\boxed{3}", 267, "our arm: markers come from the boxed branch")
    LOW = ("题目：小明有", 0, "control arm: a copied demo opener with no marker")
    if L.answer_marker(HIGH[0]) is None:
        fails.append(f"answer_marker finds nothing in {HIGH[0]!r} ({HIGH[2]}) -- a predicate "
                     f"verified only on near-zero cells cannot be distinguished from one that "
                     f"always returns None; this is the shape that reported 0/497 against a "
                     f"published 37.0%")
    if L.answer_marker(LOW[0]) is not None:
        fails.append(f"answer_marker fires on {LOW[0]!r} ({LOW[2]}) -- the near-zero side must "
                     f"stay near zero, or the large side proves nothing either")
    # AND THE PUBLISHED LARGE NUMBER IS THE ANCHOR. runs/l1_2x2_diagnose.json is the artifact whose
    # marker_rows must agree with the audit's 267/497; a rerun that silently returns to 0 would
    # otherwise look like a clean run.
    dj = os.path.join(ROOT, "runs", "l1_2x2_diagnose.json")
    if os.path.exists(dj):
        import json
        cells = json.load(open(dj, encoding="utf-8")).get("cells", {})
        oz = cells.get("ours-zh", {})
        if oz.get("marker_rows") in (0, None):
            fails.append(f"runs/l1_2x2_diagnose.json has ours-zh marker_rows="
                         f"{oz.get('marker_rows')!r}; the audit publishes 267/497, so the "
                         f"predicate lost its large side again")

    # 11. THE PINNED READING OF answer-present, frozen 2026-09-03Z before the shared-decoder rerun
    #     produced any number. Turning rep_stop off makes a looping generation run to max_new, so
    #     the answer can sit BEFORE the loop. Reading (a) -- a marker ANYWHERE in the model's turn --
    #     is pinned; reading (b) -- terminates normally AND ends with an answer -- is rejected,
    #     because "answered but could not stop" is our arm's KNOWN behaviour and (b) would score it
    #     as failure, writing the conclusion into the definition.
    answered_then_looped = "答案是：7\n" + "再来一次 " * 40
    if L.answer_marker(L.model_turn(answered_then_looped, "zh")) is None:
        fails.append("a generation that answers and THEN loops counts as no answer -- that is "
                     "reading (b), and it scores our arm's known behaviour (answers, cannot stop) "
                     "as failure; reading (a) was pinned before the rerun")
    if L.answer_marker("\\boxed{7} 后面还有很多字") is None:
        fails.append("answer-present requires the marker at the END; the pinned reading is "
                     "position-independent within the turn")
    # AND "ANYWHERE" IS BOUNDED BY THE TURN. A marker only inside a problem the model fabricated
    # for itself must NOT count -- that is the defect model_turn exists to fix (43.5% of 3-demo
    # generations open a new 题目), and crediting it would answer a question nobody asked.
    fabricated_only = "先想一想。\n\n题目：另一个问题\n解答：答案是：99"
    if L.answer_marker(L.model_turn(fabricated_only, "zh")) is not None:
        fails.append("a marker that appears ONLY inside the model's fabricated next problem counts "
                     "as answer-present -- 'anywhere' must be bounded by model_turn, or the metric "
                     "credits an answer to a question the model invented for itself")

    for f in fails:
        print(f"FAIL: {f}", file=sys.stderr)
    if fails:
        print(f"{len(fails)} check(s) failed", file=sys.stderr)
        return 1
    print("l1_fewshot 2x2 OK: scaffold switches while questions do not; truncation fires in "
          "both languages; answer markers and scoring are language-symmetric; the present-rate "
          "calls answer_marker(), the WHOLE disjunction rather than one branch of it; the arms "
          "share one decoder and the run records which; all four cells write distinct preds files")
    return 0


if __name__ == "__main__":
    sys.exit(main())

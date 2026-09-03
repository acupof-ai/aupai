#!/usr/bin/env python3
"""L1: few-shot continuation math on a base checkpoint (reasoning_panel.md S2).

Pre-registered 2026-08-30 (before p324 landed): N>=500, exact-match final answer,
non-zero > 2*delta (delta=1.4/sqrt(N)) => the production instrument exists.

Format (pinned before running): plain-text continuation -- 3 solved demos (problems
0-2 of math_test_500, full gold solutions) then the target problem; the base model
continues the solution. Demos excluded from eval (N=497). ChatML is NOT used: the
base saw 1.18% chat-domain data and the zero-shot ChatML 0/500 confounds format
with capability; plain continuation is the clean bridge.

Usage: CUDA_VISIBLE_DEVICES=7 python3 eval/l1_fewshot.py --ckpt ckpt_p324.pt
"""
import argparse
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "scripts"))
from eval_artifacts import attest, open_artifact  # noqa: E402
import json
import os
import re
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# FLA_FLASH_KDA must stay unset: the new-arch ladder checkpoints (attn_every 4)
# route 9/12 layers through chunk_kda, and the eval runners' "0" default makes
# that import fail (train.py:107 -> chunk_kda=None -> forward crash). score_matrix
# leaves it unset and scores the same checkpoints fine.
import fone  # noqa: E402
from eval.gsm8k import generate_batch  # noqa: E402
from algorithms.rlvr_reward import reward_fn, extract_boxed  # noqa: E402
from scripts.loader import load_checkpoint, load_tokenizer  # noqa: E402

TEST_PATH = os.path.join(ROOT, "data", "eval", "math_test_500.jsonl")
TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
N_DEMOS = 3  # the pinned default only; --demos sizes the pool (see split_rows)

# Inlined from eval/math_zh.py: importing that module sets FLA_FLASH_KDA=0,
# which kills chunk_kda for the new-arch ladder checkpoints.
#
# BOTH LANGUAGES' ANSWER MARKERS, because this regex is the fallback the answer-present rate
# depends on when the model does not emit \boxed. Chinese-only, it would count an English-demo
# generation that answers "The answer is: 42" as having produced NO answer -- scoring the
# --demo_lang en arm lower for a reason that is purely the scorer's vocabulary. The measured
# quantity would then be "did the model answer in Chinese", reported as "did the model answer".
ANS_RE = re.compile(r"(?:答案是|[Tt]he answer is)[:：]?\s*(.+?)(?:[。.\n]|$)")


EX_OPEN = "题目："  # kept: the zh opener, still referenced by the model_turn docstring's
#                    account of the truncation bug. scaffold() is the live source -- see
#                    model_turn, which derives the cut string rather than restating it.


def model_turn(gen, demo_lang="zh"):
    """The model's answer to the question it was ASKED, i.e. everything before it
    opens a new few-shot example of its own.

    Continuation prompting has no turn terminator, so generation runs to max_new and
    a 512-token budget buys three or four more problems after the answer. The model
    invents them: 43.5% of the 3-demo generations open a new 题目 and solve it. Nothing
    in the harness passed EX_OPEN as a stop sequence, so `score` read the whole buffer
    and `extract_boxed`'s last-box rule -- correct for a single-answer SFT output --
    graded the answer to a question nobody asked. The measured case: gold 8, the model
    answers \\boxed{8}, then writes "小明有10个苹果，他给了小李3个" and answers \\boxed{7};
    last-box scored it wrong. 45 first-box vs 25 last-box on the same file was the
    disagreement that surfaced this (de, 2026-09-01).

    Truncating is not the same as taking the first box. Within a turn the last box is
    still right -- a solution that boxes an intermediate result and then the answer is
    graded on the answer -- and 62 of 497 turns hold more than one box. First-box beats
    last-box only by accident, because it happens to stop before the fabrications.
    """
    # DERIVED FROM THE SAME PLACE build_prompt GETS IT, never restated. When --demo_lang en
    # was added, a literal EX_OPEN here would have kept cutting on 题目： while the prompt
    # opened every example with "Problem: " -- truncation would silently stop firing on
    # exactly the arm it was added for, and the failure looks like a capability difference.
    i = gen.find(scaffold(demo_lang)[0])
    return gen if i < 0 else gen[:i]


def score(gen, gold, demo_lang="zh"):
    gold_ans = extract_boxed(gold)
    if gold_ans is None:
        return 0.0
    gen = model_turn(gen, demo_lang)
    if extract_boxed(gen) is not None:
        return reward_fn(gen, gold_ans)
    m = ANS_RE.search(gen)
    if m:
        return reward_fn(f"\\boxed{{{m.group(1).strip()}}}", gold_ans)
    return 0.0


def scaffold(demo_lang):
    """(example opener, solution lead-in) for the demo language.

    THE OPENER IS ALSO THE STOP SEQUENCE. model_turn() truncates at the opener because
    continuation prompting has no turn terminator and the model invents further problems
    (43.5% of 3-demo generations open a new one). So an English scaffold cannot just change
    the printed words: the opener it uses must be the same string model_turn cuts on, or
    truncation silently stops firing and the last-box rule grades an answer to a question
    nobody asked -- the exact bug this file's model_turn docstring records.
    """
    return ("题目：", "\n解答：") if demo_lang == "zh" else ("Problem: ", "\nSolution: ")


def build_prompt(demos, target_q, demo_lang="zh"):
    op, lead = scaffold(demo_lang)
    parts = [f"{op}{q}{lead}{a}" for q, a in demos]
    parts.append(f"{op}{target_q}{lead}")
    return "\n\n".join(parts)


def split_rows(rows, n_demos, eval_from=None):
    """(demos, evals, err). The demo pool is SIZED by n_demos -- it used to be built
    from the hardcoded N_DEMOS = 3 and then sliced by args.demos, so --demos 8 ran 3
    demos while the console printed "8 demos" and the preds landed at preds_l1_d8:
    both the log and the artifact asserted a configuration that never ran. A 3-vs-8
    comparison came back byte-identical (md5 e2639d8b) which is the only reason it was
    caught (de, 2026-09-01).

    It is a function, not four lines in main(), for one reason: main() needs a
    checkpoint and a GPU, so nothing living there can be tested. The first version of
    scripts/test_fewshot_demos.py rebuilt the pool itself and asserted against
    build_prompt -- it passed on the defective code, because the test and the code
    agreed on everything except the line that was wrong.

    The eval set excludes the demos, so it SHRINKS as n_demos grows -- 497 at 3 demos,
    492 at 8. Two arms with different demo counts therefore score different problem
    sets, and comparing their rates compares populations as well as prompts. eval_from
    pins the first eval index so a sweep holds the set fixed. Stated rather than
    silently equalised: dropping problems to match would change the denominator of the
    arm that did not need it.
    """
    n_demos = max(0, n_demos)
    if n_demos >= len(rows):
        return None, None, (f"--demos {n_demos} but the set holds {len(rows)} problems; "
                            f"there would be nothing left to evaluate")
    start = n_demos if eval_from is None else eval_from
    if start < n_demos:
        return None, None, (f"--eval-from {start} overlaps the {n_demos} demo problems: "
                            f"the model would be scored on problems it was shown the "
                            f"answers to")
    demos = [(r["instruction"], r["output"]) for r in rows[:n_demos]]
    assert len(demos) == n_demos, f"asked for {n_demos} demos, built {len(demos)}"
    return demos, rows[start:], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--demos", type=int, default=3,
                    help="number of solved demonstrations (fb's demo-count sweep: "
                         "0-shot continuation vs 3-shot; pre-registered reading: "
                         "0-shot > 3-shot by >2delta=12.6pt => demos interfere; "
                         "both zero => the zero is not a prompt-format artefact)")
    ap.add_argument("--eval-from", type=int, default=None,
                    help="first problem index to evaluate (default: right after the "
                         "demos). Pin it when sweeping --demos, or each arm scores a "
                         "different set and the comparison carries a population change "
                         "as well as a prompt change.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokenizer", default=TOK_PATH)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature; 0 = greedy")
    ap.add_argument("--out", default=None, help="write JSON summary to this path")
    ap.add_argument("--force", action="store_true",
        help="overwrite an existing predictions file (default: refuse; the rows are the only copy)")
    ap.add_argument("--run", default=None,
        help="name this run so predictions version instead of colliding: preds_x.<run>.jsonl")
    ap.add_argument("--hf", action="store_true",
        help="--ckpt is a transformers directory (the control arm: Pythia-160M and its own "
             "50,304-entry NeoX BPE). The PROMPT TEXT is byte-identical to our arm's -- both "
             "models are base models on plain-text continuation, so continuation format is "
             "in-distribution for both and there is nothing to vary. Only the tokenizer differs, "
             "which is the same split eval/humaneval_bpb.py and the byte-denominator ruling use: "
             "share the text, let each side tokenize it.")
    ap.add_argument("--demo_lang", choices=("zh", "en"), default="zh",
        help="language of the DEMO scaffolding (the questions themselves are always the "
             "Chinese math-500 items and never change, so this is a clean single variable).\n"
             "I first hardcoded Chinese with the reasoning that switching would 'reintroduce "
             "the language confound e1-29 measured'. That reasoning is a TESTABLE HYPOTHESIS "
             "wearing the clothes of a premise, and it cuts both ways: Chinese demos also make "
             "the whole prompt maximally out-of-distribution for a Pile-trained model, so a gap "
             "measured under them is equally a language effect, just signed the other way. "
             "Either choice would let someone say the choice decided the result. Decoding is "
             "cheap, so both are measured: if the two arms' answer-present gap has the same "
             "magnitude under zh and en demos, language is not the driver; if it appears only "
             "under zh, it is. (6e's ruling -- and the second time tonight that inferring from "
             "a compositional relation instead of measuring it was the error.)")
    args = ap.parse_args()

    if args.hf:
        # Loaded here rather than through load_checkpoint: that path builds a HybridLM from a
        # Cfg and would have nothing to do with a transformers model.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(args.ckpt)
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt, torch_dtype=torch.bfloat16).to(args.device).eval()
        cfg, tok = None, None
        fone_on, num_id = False, None
    else:
        # dtype goes through load_checkpoint (a3a0de0 upcasts KDA A_log/dt_bias to
        # fp32 after the cast); a separate .to(bf16) here would undo the upcast.
        model, cfg = load_checkpoint(args.ckpt, device=args.device, dtype=torch.bfloat16)
        tok = load_tokenizer(args.tokenizer, cfg)
        hf_tok = None
        fone_on = getattr(cfg, "fone", False)
        num_id = getattr(cfg, "num_id", None)

    rows = [json.loads(l) for l in open(TEST_PATH, encoding="utf-8")]
    demos, evals, err = split_rows(rows, args.demos, args.eval_from)
    if err:
        sys.exit(err)
    print(f"L1 few-shot: {len(demos)} demos, {len(evals)} eval problems", flush=True)

    # The checkpoint's name is IN the path. Without it every checkpoint writes the same
    # file, so open_artifact refuses the second one -- the guard firing correctly on a
    # path that was wrong: score_matrix scored two checkpoints in one session and the
    # second got ArtifactExists rather than a number (fb, 2026-09-02). --run also
    # versions, but --run is a name a human remembers to pass and the checkpoint is not.
    # Same shape as eval/math_zh.py:103 and eval/code_zh.py:160, which already do this.
    # THE PATH CARRIES EVERY VARIABLE THAT CHANGES THE NUMBER. Adding --demo_lang without
    # adding it here would make the zh and en runs collide on one filename: open_artifact
    # refuses the second, or --force silently overwrites the first -- and the 2x2 would end up
    # with two of its four cells pointing at the same rows. This is the retraction in
    # be.l1_3shot_retracted (preds_l1_d3.jsonl overwritten by an unlogged run) in advance.
    preds_path = os.path.join(ROOT, "data", "eval",
                              f"preds_l1_d{args.demos}_{os.path.basename(args.ckpt)}"
                              + f".{args.demo_lang}"
                              + (".hf" if args.hf else "")
                              + (f".t{args.temperature}" if args.temperature else "")
                              + ".jsonl")
    correct = total = 0
    n_box = 0
    with open_artifact(preds_path, force=args.force, run=args.run) as fout:
        # --run versions the path, so the handle's name is the file that exists.
        out_path = fout.name
        for s in range(0, len(evals), args.batch):
            batch = evals[s : s + args.batch]
            texts_in = [build_prompt(demos, r["instruction"], args.demo_lang) for r in batch]
            if args.hf:
                prompts, pvals = [hf_tok(t)["input_ids"] for t in texts_in], None
            elif fone_on:
                prompts, pvals = fone.encode_prompts(texts_in, tok, num_id)
            else:
                prompts, pvals = [tok.encode(t).ids for t in texts_in], None
            with torch.no_grad():
                out = generate_batch(model, prompts, args.max_new, args.device, args.temperature, pvals)
            out_ids, out_vals = out if fone_on else (out, [None] * len(batch))
            for r, ids, vs in zip(batch, out_ids, out_vals):
                if args.hf:
                    gen = hf_tok.decode(ids, skip_special_tokens=True)
                else:
                    gen = fone.decode_text(ids, vs, tok, num_id) if fone_on else tok.decode(ids)
                ok = score(gen, r["output"], args.demo_lang)
                correct += int(ok)
                total += 1
                turn = model_turn(gen, args.demo_lang)
                # THE ANSWER-PRESENT TEST USES THE SAME MARKERS THE SCORER DOES, derived from
                # ANS_RE rather than restated. Hardcoded `"答案是" in turn`, this metric would
                # count an English-demo generation answering "The answer is: 42" as producing NO
                # answer -- and answer-present rate is the ONE layer where the two arms might
                # genuinely differ, so a scorer-vocabulary artefact would land precisely on the
                # comparison the 2x2 exists to make. Two lines apart from the identical bug in
                # ANS_RE, which is what a restated constant does.
                n_box += int("\\boxed" in turn or ANS_RE.search(turn) is not None)
                fout.write(json.dumps({"q": r["instruction"], "gen": gen, "ok": ok},
                                      ensure_ascii=False) + "\n")
            if total % 64 < args.batch or total == len(evals):
                print(f"  {total}/{len(evals)} acc={correct / total:.1%}", flush=True)

    # attest what was WRITTEN, not what was requested: --run versions the path, and
    # attesting preds_path recorded a hash for a file this run never touched.
    attest(out_path)  # the citation contract: the writer proves these bytes existed
    delta = 1.4 / (total ** 0.5)
    acc = correct / total
    print(f"L1 math-500 few-shot: {correct}/{total} = {acc:.1%}")
    print(f"binomial delta={delta:.1%} -> 2*delta={2 * delta:.1%}; "
          f"instrument exists iff acc > {2 * delta:.1%}")
    print(f"answer-present rate {n_box / total:.1%}")
    print(f"preds saved: {out_path}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"correct": correct, "total": total, "acc": acc,
                        "binomial_delta": delta, "answer_present_rate": n_box / total,
                        "demos": args.demos, "temperature": args.temperature,
                        "demo_lang": args.demo_lang, "arm": "control" if args.hf else "ours",
                        "ckpt": os.path.basename(args.ckpt.rstrip("/")),
                        "preds_path": out_path}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# restartable: one JSON line per problem is written as it is scored, and a rerun with the same
# --resume skips the questions already in the file. An interrupt costs the batch in flight, not
# the run. Without this a 497-problem cell is ~15 minutes of generation that a Ctrl-C throws
# away entirely -- open_artifact's default mode is "w", so the rerun truncates rather than
# continues, and "the rows are appended" was true of the loop while being false of the file.
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
from eval_artifacts import attest, open_artifact, versioned_path  # noqa: E402
import hashlib
import json
import os
import re
import sys
import time

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
from scripts.loader import load_checkpoint, load_tokenizer, vocab_fingerprint  # noqa: E402

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


def answer_marker(text):
    """Where an extractable answer starts in `text`, or None. THE definition of answer-present.

    A FUNCTION, not a regex, because the predicate is a DISJUNCTION: `\\boxed` OR ANS_RE. Every
    reader that took ANS_RE alone as "the marker" silently dropped the boxed branch -- which is the
    branch our arm actually uses. l1_2x2_diagnose imported ANS_RE, believing that was the shared
    definition, and measured 0/497 markers in cells whose answer-present rate is 37.0%. Importing the
    regex made the two agree on a SUBSTRING of the predicate and read as full agreement.

    Returns a character offset into `text` as given, so a caller can ask "did the decoder run far
    enough to reach a marker" -- pass the RAW generation for that, model_turn's output for scoring.
    """
    cands = []
    i = text.find("\\boxed")
    if i >= 0:
        cands.append(i)
    m = ANS_RE.search(text)
    if m:
        cands.append(m.start())
    return min(cands) if cands else None


def _hf_tok_fp(hf_tok):
    """Fingerprint a transformers tokenizer's vocabulary, mirroring loader.vocab_fingerprint.

    Its own function rather than a call to loader.vocab_fingerprint, which takes a `tokenizers`
    object: get_vocab() is the one method both expose with the same meaning, and reaching for the
    shared helper would work by coincidence and break on the first interface difference.
    """
    try:
        h = hashlib.sha256()
        for t, _i in sorted(hf_tok.get_vocab().items(), key=lambda kv: kv[1]):
            h.update(str(t).encode())
        return h.hexdigest()[:16]
    except Exception:
        return None


def ckpt_sha256(path):
    """sha256 of the checkpoint FILE, or None for an --hf directory.

    THE FILE, not the cfg's vocab_id, because vocab_id is the field the old checkpoints do not
    have: ckpt_b0_sd_equalcompute.pt reads `vocab_id: None`, and an identity built on it would be
    None-vs-None for every pre-fingerprint checkpoint -- equal, and proving nothing (6e's ruling).
    A directory returns None rather than a hash of one file inside it; the control arm's identity
    is its path plus the header's other fields.
    """
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_path(ckpt, demos, demo_lang="zh", hf=False, no_rep_stop=False, temperature=0.0,
                  run=None, root=None):
    """The predictions path for one cell. THE ONLY PLACE THIS NAME IS BUILT.

    It used to be an expression inside main(), which meant a reader that wanted the path had to
    restate it -- and metric_l1_fewshot, which now verifies the artifact before transcribing it,
    would have been the second copy. Two copies of a filename rule is how the 2x2 nearly collided
    two cells onto one file; a rule with one implementation cannot disagree with itself.

    Every variable that changes the number is in the name: test_l1_fewshot_2x2 group 6 CALLS this
    function with each axis both ways and requires four distinct paths, rather than searching for
    the expression that builds them.

    NOT named `preds_path`: that identifier is harness.check_reported_path_is_written's marker for
    the stale-unversioned-variable defect, and a call to a function of that name reads to it as the
    defect itself (the check looks for the Name, not for how it is used).
    """
    root = root or ROOT
    p = os.path.join(root, "data", "eval",
                     f"preds_l1_d{demos}_{os.path.basename(str(ckpt).rstrip('/'))}"
                     + f".{demo_lang}"
                     + (".hf" if hf else "")
                     + (".norepstop" if no_rep_stop else "")
                     + (f".t{temperature}" if temperature else "")
                     + ".jsonl")
    return versioned_path(p, run) if run else p


# The header is row 0 of the predictions file and every reader must skip it. It exists because a
# 497-row artifact with acc 0.0181 sat on disk while its score_matrix row said ERROR, and nothing
# in the file said WHICH checkpoint produced it: the keys were q/gen/ok only, so mtime and filename
# were the whole identity -- and this repo has rewritten same-named checkpoints (6e's ruling,
# audit_0904 E22/E23). A reader distinguishes it by the "_header" key, which no scored row has.
HEADER_KEY = "_header"


def build_header(args, ckpt_sha, vocab_id, tok_fp, n_evals):
    """The identity row. `ckpt_sha` is the checkpoint file's, not the cfg's vocab_id."""
    return {
        HEADER_KEY: 1,
        "ckpt": os.path.basename(str(args.ckpt).rstrip("/")),
        "ckpt_sha256": ckpt_sha,
        "vocab_id": vocab_id or "none",
        "tokenizer_fp": tok_fp or "none",
        "demos": args.demos,
        "demo_lang": args.demo_lang,
        "arm": "control" if args.hf else "ours",
        "rep_stop": not args.no_rep_stop,
        "temperature": args.temperature,
        "n_evals": n_evals,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def read_header(path):
    """The header dict, or None if the file has none (every artifact written before this change).

    None is NOT an error here -- it is the honest answer for the seven pre-header artifacts, and
    the caller's job is to refuse rather than to guess. A header-less file still parses as rows.
    """
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                return d if HEADER_KEY in d else None
    except (OSError, json.JSONDecodeError):
        return None
    return None


def verify_artifact(path, ckpt):
    """(header, reason). A header whose ckpt_sha256 matches `ckpt` on disk, or None + why not.

    THE COMPARISON IS AGAINST THE BYTES ON DISK, so a checkpoint rewritten under the same name
    fails it -- which is the whole point: filename-and-mtime is not identity. Returns the header
    only when the artifact provably came from THIS checkpoint's bytes.
    """
    if not os.path.exists(path):
        return None, f"{os.path.basename(path)} absent"
    h = read_header(path)
    if h is None:
        return None, (f"{os.path.basename(path)} has no header row -- written before headers "
                      f"existed, so the checkpoint that produced it is unrecorded")
    want = ckpt_sha256(ckpt)
    if want is None:
        # STATES THE FACT, NOT A CAUSE. This said "--ckpt is a directory", which is only one of
        # the two ways ckpt_sha256 returns None (the other is an absent path) -- a message that
        # names a cause it did not check misdiagnoses the other one.
        return None, (f"{os.path.basename(path)}: {ckpt} is not a file, so there is no "
                      f"checkpoint hash to compare")
    got = h.get("ckpt_sha256")
    if got != want:
        return None, (f"{os.path.basename(path)} was written from checkpoint sha "
                      f"{str(got)[:16]}..., the file on disk is {want[:16]}... -- same name, "
                      f"different bytes")
    return h, None


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


def hf_generate_batch(model, prompts, max_new, device, temperature, tokenizer, pad_id,
                      rep_stop=True):
    """Greedy/sampled decode for a transformers model, matching train.generate_batch's CONTRACT.

    WHY NOT train.generate_batch. It calls `model(x, num_vals=..., no_head=True)` and reads
    `model.cfg` / `model.lm_logits` -- our HybridLM's interface. A transformers model has none of
    those, so the --hf path would die on the first batch. My tokenizer smoke test roundtripped both
    languages and told me nothing about this, because it never reached generation: a check that
    exercises the cheap half and reports "the path works" is the same defect as a guard nobody calls.

    WHAT MUST MATCH, or the two arms differ in more than the model:
      - left-padded batch, greedy at temperature 0, same max_new;
      - the SAME repetition stop (whitespace 8-gram or CJK 12-gram repeated 3x, checked every 32
        tokens). facts/base_eval.json #be.degenerate_repetition records that greedy decoding on
        this family loops at 25-56% and that format-class metrics are meaningless without the
        decoder pinned -- so an arm without rep_stop would run every degenerate row to max_new and
        its answer-present rate would be measured under a different decoder.
    Returns generated ids per row (prompt stripped), like the function it stands in for.
    """
    import torch
    from collections import Counter

    B = len(prompts)
    lengths = [len(p) for p in prompts]
    width = max(lengths)
    # LEFT pad: a decoder-only model's next token comes from the last position, so right padding
    # would read a pad token as the context to continue from.
    x = torch.full((B, width), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, width), dtype=torch.long, device=device)
    for i, p in enumerate(prompts):
        x[i, width - lengths[i]:] = torch.tensor(p, device=device)
        attn[i, width - lengths[i]:] = 1
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
    done = torch.zeros(B, dtype=torch.bool, device=device)
    gen = [[] for _ in range(B)]
    past = None
    cur, cur_attn = x, attn
    with torch.no_grad():
        for step in range(max_new):
            out = model(input_ids=cur, attention_mask=cur_attn, past_key_values=past,
                        use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            if temperature > 0:
                nxt = torch.multinomial(
                    torch.softmax(logits.float() / temperature, dim=-1), 1).squeeze(1)
            else:
                nxt = logits.argmax(dim=-1)
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            for i in range(B):
                if not done[i]:
                    gen[i].append(int(nxt[i]))
            done |= nxt == eos
            cur = nxt.unsqueeze(1)
            cur_attn = torch.cat([cur_attn, torch.ones((B, 1), dtype=torch.long, device=device)],
                                 dim=1)
            if rep_stop and step > 0 and step % 32 == 31:
                for i in range(B):
                    if done[i] or len(gen[i]) < 64:
                        continue
                    text = tokenizer.decode(gen[i], skip_special_tokens=True)
                    hit = False
                    words = text.split()
                    if len(words) >= 24:
                        grams = [tuple(words[j:j + 8]) for j in range(len(words) - 7)]
                        hit = any(c >= 3 for c in Counter(grams).values())
                    if not hit:
                        cjk = sum(1 for c in text if '一' <= c <= '鿿')
                        if cjk > len(text) * 0.3 and len(text) >= 36:
                            chars = list(text)
                            cg = [tuple(chars[j:j + 12]) for j in range(len(chars) - 11)]
                            hit = any(c >= 3 for c in Counter(cg).values())
                    if hit:
                        done[i] = True
            if bool(done.all()):
                break
    return gen


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
    ap.add_argument("--no_rep_stop", action="store_true",
        help="disable the repetition stop for this run. THE POINT IS TO SEPARATE TWO THINGS the "
             "2x2 leaves fused: the control's answer-present is 0.6-1.0%% and its generations stop "
             "at 84-86 characters against our 794-851, both under the SAME stop rule. So 'the "
             "control has no instruction-following behaviour' and 'the stop rule truncates the "
             "control before an answer marker appears' predict the same 0.6%%. With the stop off: "
             "if the control's length rises to our magnitude AND answer-present rises with it, "
             "part of the 0.6%% was the stop rule; if length rises and answer-present stays down, "
             "the gap is behavioural. One cell, one switch (6e's design).")
    ap.add_argument("--resume", action="store_true",
        help="append to an existing predictions file and skip the questions already in it. The "
             "marker at the top of this file promises an interrupt costs one batch; without this "
             "flag open_artifact's mode=\"w\" truncates on rerun and the promise is false. Counts "
             "are rebuilt from the existing rows, so acc and answer-present cover the whole set "
             "and not just the tail.")
    args = ap.parse_args()

    if args.hf:
        # Loaded here rather than through load_checkpoint: that path builds a HybridLM from a
        # Cfg and would have nothing to do with a transformers model.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(args.ckpt)
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt, torch_dtype=torch.bfloat16).to(args.device).eval()
        # eos as pad: this tokenizer has no pad token, and `or 0` would silently make token 0 the
        # pad for a model where 0 is a real token. Same trap as eval_heldout's pad_id.
        hf_pad = (hf_tok.pad_token_id if hf_tok.pad_token_id is not None
                  else hf_tok.eos_token_id)
        if hf_pad is None:
            sys.exit("REFUSING: the control tokenizer has neither a pad nor an eos token, so a "
                     "padded batch would be indistinguishable from real content")
        cfg, tok = None, None
        fone_on, num_id = False, None
        # THE CONTROL ARM'S TOKENIZER FINGERPRINT comes from the transformers tokenizer, so the
        # header can tell two control runs apart if the control model is ever swapped. vocab_id is
        # ours-only, hence "none" here rather than a borrowed value.
        vocab_id = None
        tok_fp = _hf_tok_fp(hf_tok)
    else:
        # dtype goes through load_checkpoint (a3a0de0 upcasts KDA A_log/dt_bias to
        # fp32 after the cast); a separate .to(bf16) here would undo the upcast.
        model, cfg = load_checkpoint(args.ckpt, device=args.device, dtype=torch.bfloat16)
        tok = load_tokenizer(args.tokenizer, cfg)
        hf_tok = None
        fone_on = getattr(cfg, "fone", False)
        num_id = getattr(cfg, "num_id", None)
        # BOTH, because they answer different questions and one of them is missing on the
        # checkpoints this header exists for. vocab_id is the checkpoint's recorded fingerprint and
        # reads None on every pre-fingerprint checkpoint (ckpt_b0_sd_equalcompute.pt is one);
        # tokenizer_fp is computed here from the tokenizer actually loaded, so it is never absent.
        vocab_id = getattr(cfg, "vocab_id", None)
        tok_fp = vocab_fingerprint(tok)

    rows = [json.loads(l) for l in open(TEST_PATH, encoding="utf-8")]
    demos, evals, err = split_rows(rows, args.demos, args.eval_from)
    if err:
        sys.exit(err)
    print(f"L1 few-shot: {len(demos)} demos, {len(evals)} eval problems", flush=True)

    # THE CHECKPOINT'S NAME IS IN THE PATH, and the path is built by artifact_path() -- one
    # implementation, because metric_l1_fewshot now needs the same name to verify the artifact
    # before transcribing it, and a second copy of the rule is a second thing to get wrong.
    # Without the checkpoint in the name every checkpoint writes the same file, so open_artifact
    # refuses the second one -- the guard firing correctly on a path that was wrong: score_matrix
    # scored two checkpoints in one session and the second got ArtifactExists rather than a number
    # (fb, 2026-09-02). Same shape as eval/math_zh.py:103 and eval/code_zh.py:160.
    # THE PATH CARRIES EVERY VARIABLE THAT CHANGES THE NUMBER. Adding --demo_lang without adding it
    # there would make the zh and en runs collide on one filename: open_artifact refuses the second,
    # or --force silently overwrites the first -- and the 2x2 would end up with two of its four
    # cells pointing at the same rows. This is the retraction in be.l1_3shot_retracted
    # (preds_l1_d3.jsonl overwritten by an unlogged run) in advance.
    # BUILT AND VERSIONED IN ONE CALL, under the name that is then used everywhere. There is no
    # unversioned second path left in this function to read, print, or record by mistake -- which
    # is the defect harness.check_reported_path_is_written guards: four runners attested the
    # versioned path and then printed the unversioned one, and an hour went on 2026-09-01 to a log
    # naming a file that did not exist. run= is NOT passed to open_artifact below, because that
    # would version an already-versioned name (verified: preds_x.r1.r1.jsonl).
    out_path = artifact_path(args.ckpt, args.demos, args.demo_lang, args.hf,
                             args.no_rep_stop, args.temperature, args.run)
    correct = total = 0
    n_box = 0
    # RESUME REBUILDS THE COUNTS, not just the skip-set. Counting only the newly generated rows
    # would report acc over the tail while the file holds the whole set -- a partial-population
    # number wearing the whole population's label.
    seen = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                # THE HEADER IS NOT A SCORED ROW. Counting it would put 1 in the denominator with
                # no `ok` to add to the numerator -- a KeyError if it raised, and an off-by-one
                # accuracy if it did not. Every reader of this file needs this skip; that is why
                # the key is a name no scored row carries rather than a position.
                if HEADER_KEY in d:
                    continue
                seen.add(d["q"])
                correct += int(d["ok"])
                total += 1
                turn = model_turn(d["gen"], args.demo_lang)
                n_box += int(answer_marker(turn) is not None)
        print(f"  resuming: {total} already scored, {correct} correct", flush=True)
        evals = [r for r in evals if r["instruction"] not in seen]
        if not evals:
            print("nothing left to score; the file already covers the eval set")
    n_target = total + len(evals)  # resumed rows + rows this process will generate
    # run= is NOT passed: artifact_path already versioned it above, and open_artifact would version
    # it again -- verified, it yields preds_x.r1.r1.jsonl. The handle's .name is still what gets
    # attested, so the contract that check exists for is unchanged.
    with open_artifact(out_path, force=args.force,
                       mode="a" if args.resume else "w") as fout:
        # THE HEADER GOES FIRST, before any generation, so an interrupted run still carries its
        # identity: the 497-row artifact this exists for was complete and unattributable, and one
        # that dies at row 12 must not be worse off. Written only on a fresh run -- a resume is
        # appending to a file that already has one, and a second header row would make
        # read_header's answer depend on which line it reached first.
        if not args.resume or total == 0:
            fout.write(json.dumps(build_header(
                args, ckpt_sha256(args.ckpt), vocab_id, tok_fp, len(evals) + total),
                ensure_ascii=False) + "\n")
            fout.flush()
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
                # THE TOKENIZER IS PASSED, WHICH IS WHAT TURNS rep_stop ON. train.generate_batch
                # gates it on `tokenizer is not None` (train.py:944), and the original call here
                # omitted the argument -- so OUR arm ran with NO repetition stop while the control
                # arm, whose tokenizer I passed explicitly, ran with it. The 2x2's length gap
                # (794-851 characters against 84-86) and our 98.8% loop rate are partly that:
                # one arm running to max_new, the other truncated. A decoder difference read as a
                # model difference, from an argument that defaults to a silent False.
                if args.hf:
                    out = hf_generate_batch(model, prompts, args.max_new, args.device,
                                            args.temperature, hf_tok, hf_pad,
                                            rep_stop=not args.no_rep_stop)
                else:
                    out = generate_batch(model, prompts, args.max_new, args.device,
                                         args.temperature, pvals, tokenizer=tok,
                                         rep_stop=not args.no_rep_stop)
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
                # THE ANSWER-PRESENT TEST CALLS answer_marker(), the same predicate the scorer
                # uses -- ONE function, not a re-spelled disjunction. Hardcoded `"答案是" in turn`,
                # this metric would count an English-demo generation answering "The answer is: 42"
                # as producing NO answer, and answer-present is the ONE layer where the arms might
                # genuinely differ. Spelling out `"\\boxed" in turn or ANS_RE.search(turn)` here was
                # already better than a constant and STILL not enough: l1_2x2_diagnose imported
                # ANS_RE as "the marker", lost the boxed branch, and reported 0/497 for a cell whose
                # rate is 37.0%. A predicate spread over two operators can be half-copied.
                #
                # PINNED 2026-09-03Z, BEFORE the shared-decoder rerun produced any number (6e's
                # ruling, my scope refinement). Turning rep_stop OFF on both arms makes
                # answer-present ambiguous, because a looping generation now runs to max_new and
                # the answer can sit BEFORE the loop starts. Two readings were available:
                #   (a) a marker ANYWHERE in the model's turn  <- PINNED
                #   (b) the generation terminates normally AND ends with an answer
                # (a), because the question is whether the model PRODUCES an answer, not whether it
                # stops. Stopping is what rep_stop measured, and we just removed it from the
                # variables. (b) would score "answered correctly but could not stop" as a failure --
                # which is our arm's known behaviour, so (b) writes the conclusion into the
                # definition. Position-independent, hence `answer_marker(turn) is not None` and
                # never a check on where the marker sits.
                #
                # SCOPE: the model's TURN, not the raw buffer. model_turn cuts at the point the
                # model opens a fabricated next problem, and 43.5% of 3-demo generations do that.
                # Counting markers in the raw buffer would credit an answer to a question the model
                # invented for itself -- the same defect the last-box rule had before model_turn
                # existed. So "anywhere" is bounded by the turn, and that bound is the reason the
                # metric is about the question that was ASKED.
                #
                # Pinned before the run because a definition chosen after the numbers exist is
                # indistinguishable from one chosen to make them look good.
                n_box += int(answer_marker(turn) is not None)
                fout.write(json.dumps({"q": r["instruction"], "gen": gen, "ok": ok},
                                      ensure_ascii=False) + "\n")
                # FLUSHED PER ROW, because the restartability marker promises an interrupt costs
                # one batch. Python buffers ~8 KB and a row here is ~1 KB, so without this a
                # Ctrl-C drops the last several rows and the resume regenerates them -- the
                # promise would be true of the loop and false of the file, which is exactly the
                # distinction that made open_artifact's mode="w" a silent truncation.
                fout.flush()
            # THE DENOMINATOR IS THE WHOLE EVAL SET, not the slice this process generated. With
            # --resume, `evals` is filtered to the unscored remainder while `total` counts the
            # resumed rows too, so `total/len(evals)` would print 400/97 and the accuracy would
            # look like it was measured on 97 problems. n_target is fixed BEFORE the loop: my
            # first version recomputed it from `total` each iteration, which made it grow with
            # the numerator and always read n/n.
            if total % 64 < args.batch or total == n_target:
                print(f"  {total}/{n_target} acc={correct / total:.1%}", flush=True)

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
                        "rep_stop": not args.no_rep_stop,
                        "ckpt": os.path.basename(args.ckpt.rstrip("/")),
                        # THE SAME sha THE HEADER CARRIES, so a summary and its artifact can be
                        # matched to each other and to the checkpoint. Without it the summary is
                        # the third thing whose only link to a checkpoint is a filename.
                        "ckpt_sha256": ckpt_sha256(args.ckpt),
                        "preds_path": out_path}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

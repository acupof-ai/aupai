#!/usr/bin/env python3
"""3b 2026-09-24: SFT mix A external-source rendering (sc2-exec + APPS).

Runs on the digest Linux CPU box, not the pod (the pod stays reserved for the
training job). Reads the HF-mirror downloads under raw/, emits one jsonl of
RAW continuation pairs per source plus a stats file:

  raw/sc2_train.parquet   (bigcode/self-oss-instruct-sc2-exec-filter-50k, ODC-By)
  raw/apps_train.jsonl    (codeparrot/apps train, MIT)

Pair shape (NO ChatML -- the gate scoring arm is --rstrip_nl continuation):
  sc2:   prompt = instruction text + "\n", output = the exec-validated response
  APPS:  only call-style problems (input_output.fn_name present); starter_code
         (the class/def signature) is the prompt, the shortest parseable
         solution defining that fn/class is the output. stdin/stdout problems
         are skipped, not force-rendered.

Gates, identical for both:
  - 13-gram DROP against HumanEval+MBPP (filters/decontam_ngram) on prompt and
    output independently;
  - output 8..256 gate tokens, prompt <= 512 tokens (gate tokenizer);
  - holdout hash gate (datagen/holdout.is_holdout) on the prompt.
Sampling is seeded; sc2 and APPS rows are shuffled with seed 42 then taken in
order until the token quota is filled. Quotas are LOSS tokens (the output side;
the prompt is masked), matching the code_if 8.43M body-token figure that sets
55% of the pack: sc2 4.60M (30%), APPS 0.77M (5%).
# restartable: pure CPU over one jsonl (APPS, 5k rows) and one parquet (sc2,
# 50k rows) already downloaded locally; a full rerun is ~1-2 minutes on 8
# threads, seeded and deterministic, so an interrupt just reruns.
"""
import argparse
import ast
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
# APPS input_output embeds big-int test literals (>4300 digits); 3.11 refuses
# them under the default conversion cap. The field is parsed, never evaluated.
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "filters"))
sys.path.insert(0, os.path.join(ROOT, "datagen"))

from tokenizers import Tokenizer
from decontam_ngram import Decontaminator, decontam_fp
from holdout import is_holdout

RAW = os.path.join(ROOT, "data", "sft", "sfta", "raw")
OUT_DIR = os.path.join(ROOT, "data", "sft", "sfta")
TOK = os.path.join(ROOT, "data", "tokenizer.json")
HE = os.path.join(ROOT, "data", "eval", "humaneval", "humaneval_164.jsonl")
MBPP = os.path.join(ROOT, "data", "eval", "mbpp_holdouts.jsonl")
SEED = 42
MAX_OUT, MIN_OUT, MAX_PROMPT = 256, 8, 512


def gates(tok, dcon):
    def accept(prompt, output):
        if not prompt.strip() or not output.strip():
            return "empty"
        if is_holdout(prompt):
            return "holdout"
        if dcon.hit(prompt):
            return "ngram_prompt"
        if dcon.hit(output):
            return "ngram_output"
        lp, lo = len(tok.encode(prompt).ids), len(tok.encode(output).ids)
        if lp > MAX_PROMPT:
            return "prompt_long"
        if lo > MAX_OUT or lo < MIN_OUT:
            return "output_len"
        return None
    return accept


def write_pairs(name, pairs, quota, drops, extra):
    random.Random(SEED).shuffle(pairs)
    kept, loss_tokens = [], 0
    for prompt, output in pairs:
        if loss_tokens >= quota:
            break
        # caller already gated; re-tokenise for the running token total
        kept.append((prompt, output))
        loss_tokens += extra["tok"].encode(output).ids.__len__()
    out = os.path.join(OUT_DIR, name + ".jsonl")
    with open(out, "w") as f:
        for prompt, output in kept:
            f.write(json.dumps({"prompt": prompt, "output": output,
                                "source": name}, ensure_ascii=False) + "\n")
    stats = {"source": name, "kept_pairs": len(kept), "loss_tokens": loss_tokens,
             "quota": quota, "drops": drops}
    with open(os.path.join(OUT_DIR, name + "_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats), flush=True)
    return kept


def render_apps(tok, accept, quota):
    """Call-style APPS only. prompt=starter signature, output=parseable solutions.

    Up to 3 distinct parseable in-length solutions per problem (length order):
    one problem offers a median 9 compliant solutions, so taking only the
    shortest throws away supply. Repeated prompts are accepted -- the answers
    differ -- but capped so one problem cannot dominate the slice.
    """
    pairs, drops = [], {}
    PER_PROBLEM = 4

    def drop(k):
        drops[k] = drops.get(k, 0) + 1

    path = os.path.join(RAW, "apps_train.jsonl")
    for line in open(path):
        try:
            r = json.loads(line)
            io = json.loads(r.get("input_output") or "{}")
            fn = io.get("fn_name")
            sols = json.loads(r.get("solutions") or "[]")
            starter = (r.get("starter_code") or "").strip()
        except (json.JSONDecodeError, TypeError, ValueError):
            drop("bad_json")
            continue
        if not fn or not starter or not sols:
            drop("not_call_style")
            continue
        seen = set()
        cand = []
        for s in sols:
            s = s.strip()
            if s in seen or (("def " + fn) not in s and "class " not in s):
                continue
            seen.add(s)
            try:
                ast.parse(s)
            except SyntaxError:
                continue
            cand.append(s)
        if not cand:
            drop("no_parseable_solution")
            continue
        cand.sort(key=len)
        taken = 0
        for best in cand:
            why = accept(starter, best)
            if why:
                drop(why)
                continue
            pairs.append((starter, best))
            taken += 1
            if taken >= PER_PROBLEM:
                break
    return write_pairs("apps_call", pairs, quota, drops, {"tok": tok})


def render_sc2(tok, accept, quota):
    """sc2-exec: prompt=instruction, output=the first parseable solution block.

    Every response carries one or more ```python fences; the first fence is the
    exec-validated solution in 98% of rows (probe, n=10000: 9828 def/class-only,
    0 pure-test, 172 solution+asserts). Later fences are tests. Taking the fence
    instead of the whole response drops the prose explanation and gives a
    continuation-shaped code target (block median 66 tokens vs 166 full).
    """
    import ast
    import re
    import pyarrow.parquet as pq

    fence = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)
    pairs, drops = [], {}

    def drop(k):
        drops[k] = drops.get(k, 0) + 1

    tbl = pq.read_table(os.path.join(RAW, "sc2_train.parquet"))
    cols = {n: tbl.column(n).to_pylist() for n in tbl.column_names}
    for i in range(tbl.num_rows):
        instruction = (cols["instruction"][i] or "").strip()
        solution = None
        for block in fence.findall(cols["response"][i] or ""):
            block = block.strip()
            if "def " not in block and "class " not in block:
                continue  # skip assert-only / test fences
            try:
                ast.parse(block)
            except SyntaxError:
                continue
            solution = block
            break
        if not instruction:
            drop("empty_instruction")
            continue
        if solution is None:
            drop("no_solution_fence")
            continue
        why = accept(instruction + "\n", solution)
        if why:
            drop(why)
            continue
        pairs.append((instruction + "\n", solution))
    return write_pairs("sc2_exec", pairs, quota, drops, {"tok": tok})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["apps", "sc2", "both"], default="both")
    ap.add_argument("--sc2_quota", type=int, default=4_600_000)
    ap.add_argument("--apps_quota", type=int, default=770_000)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    tok = Tokenizer.from_file(TOK)
    dcon = Decontaminator.load_default(ROOT)
    accept = gates(tok, dcon)
    print("decontam_fp_inputs", decontam_fp(HE, MBPP), flush=True)
    if args.source in ("apps", "both"):
        render_apps(tok, accept, args.apps_quota)
    if args.source in ("sc2", "both"):
        render_sc2(tok, accept, args.sc2_quota)


if __name__ == "__main__":
    main()

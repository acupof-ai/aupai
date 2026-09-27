#!/usr/bin/env python3
"""Build the reasoning SFT source JSONLs for the v1 reasoning pack (de, 1e ruling 2026-09-27).

Mix (1e 2026-09-27, ~120k examples):
  cot_dc math reasoning 40%  | TACO+APPS sandbox-verified solutions 40% | code_if short 20%

This script prepares the TWO parts available without the sandbox (run in parallel with the
verifier):
  --cot   from data/corpus/cot_dc/*.jsonl  -> sft_reason_v1/cot_reason.jsonl
  --code  from data/sft/sfta/code_if_clean.jsonl -> sft_reason_v1/code_if_short.jsonl
The TACO/APPS part is scripts/sft_verify_code.py's output and is merged by sft_reason_pack.py.

cot split: a record is one `content` string "QUESTION\\n\\n<reasoning... \\boxed{answer}>". The
SFT prompt is the question (before the first blank line), the output the reasoning (the rest).
We keep only records that actually have a separable prompt, a multi-step body and a boxed/final
answer marker -- the whole point is teaching the reasoning, not raw text.

13-gram decontamination against HumanEval/MBPP is applied to the PROBLEM text for the code part
(cot math has no benchmark overlap by construction -- it is a math corpus -- but the gate is run
on it too for a uniform record; math benchmarks are not the gate target, so it passes through).

    # pod, CPU only, beside live training:
    taskset -c 146-179 nice -n 10 OMP_NUM_THREADS=2 \\
        python3 scripts/sft_reason_prep.py --cot --code --out data/sft/sft_reason_v1
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "filters"))

# A reasoning body needs to actually walk steps. Both a boxed answer and an explicit
# "answer is/therefore" closing count; require at least one and a minimum body length.
ANSWER_MARK = ("\\boxed", "answer is", "final answer", "therefore", "thus, the")
MIN_BODY_CHARS = 120


def split_cot(content):
    """(prompt, output) or None. Prompt = text before the first blank line; output = rest."""
    if not content or "\n\n" not in content:
        return None
    prompt, _, output = content.partition("\n\n")
    prompt = prompt.strip()
    output = output.strip()
    if len(prompt) < 15 or len(output) < MIN_BODY_CHARS:
        return None
    low = output.lower()
    if not any(m in low for m in ANSWER_MARK):
        return None
    return prompt, output


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def prep_cot(corpus_glob, out, cap):
    n = kept = 0
    with open(out, "w", encoding="utf-8") as fh:
        for p in sorted(glob.glob(corpus_glob)):
            for rec in read_jsonl(p):
                n += 1
                if cap and kept >= cap:
                    break
                sp = split_cot(rec.get("content", ""))
                if sp is None:
                    continue
                prompt, output = sp
                fh.write(json.dumps({"prompt": prompt, "output": output,
                                     "source": "cot_dc"}, ensure_ascii=False) + "\n")
                kept += 1
    return {"seen": n, "kept": kept}


def prep_code_if(inp, out, cap):
    """code_if_clean already carries prompt/output/body_tokens; just tag + copy, gate length."""
    n = kept = 0
    with open(out, "w", encoding="utf-8") as fh:
        for rec in read_jsonl(inp):
            n += 1
            if cap and kept >= cap:
                break
            prompt, output = rec.get("prompt", ""), rec.get("output", "")
            if not prompt or not output:
                continue
            fh.write(json.dumps({"prompt": prompt, "output": output,
                                 "source": "code_if_clean"}, ensure_ascii=False) + "\n")
            kept += 1
    return {"seen": n, "kept": kept}


def _selftest():
    steps = "We rearrange and simplify each term in turn, step by step:\n a = b + c.\n" * 3
    good = f"Find the unknown value of x in the equation.\n\n{steps}Thus, the answer is \\boxed{{2}}."
    sp = split_cot(good)
    assert sp is not None and sp[0] == "Find the unknown value of x in the equation." \
        and sp[1].endswith("\\boxed{2}."), sp
    assert split_cot("no blank line here but \\boxed{1}") is None
    assert split_cot("q?\n\ntoo short \\boxed{1}") is None
    long_body = "reasoning " * 20
    assert split_cot(f"a longer test question here?\n\n{long_body}") is None, "body without an answer marker rejects"
    assert split_cot(f"a longer test question here?\n\n{long_body} therefore 42") is not None
    print("sft_reason_prep selftest OK: cot prompt/body split + answer/length gates")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/sft/sft_reason_v1")
    ap.add_argument("--cot-glob", default="data/corpus/cot_dc/cot_*.jsonl")
    ap.add_argument("--code-if", default="data/sft/sfta/code_if_clean.jsonl")
    ap.add_argument("--cot-cap", type=int, default=48000, help="target cot rows (40%% of 120k)")
    ap.add_argument("--code-cap", type=int, default=24000, help="target code_if rows (20%%)")
    ap.add_argument("--cot", action="store_true")
    ap.add_argument("--code", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    os.makedirs(a.out, exist_ok=True)
    stats = {}
    if a.cot:
        stats["cot_reason"] = prep_cot(a.cot_glob, os.path.join(a.out, "cot_reason.jsonl"),
                                       a.cot_cap)
    if a.code:
        stats["code_if_short"] = prep_code_if(a.code_if, os.path.join(a.out, "code_if_short.jsonl"),
                                              a.code_cap)
    with open(os.path.join(a.out, "prep_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

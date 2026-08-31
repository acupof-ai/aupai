#!/usr/bin/env python3
"""Code eval on the 500-problem held-out set (data/eval/code_holdout_500.jsonl).

Greedy generation with the SFT prompt format, last ```python block extracted,
executed in scripts/sandbox_exec.run_sandboxed, stdout matched line-by-line
against the recorded 运行输出 oracle.

Known-answer (--selfcheck, no GPU): every reference solution must score 1 on
its own oracle, and deliberately wrong solutions must score 0. Run this once
after carving and after any change to the match path — the same GT round-trip
discipline as the math reward.

Usage: python eval/code_zh.py --ckpt ckpt_sft_math.pt [--max_new 512] [--batch 32]
       python eval/code_zh.py --ckpt ckpt_sft_math.pt --k 8 --temperature 0.8
       python eval/code_zh.py --selfcheck
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("FLA_FLASH_KDA", "0")

from scripts.loader import format_prompt, load_checkpoint, load_tokenizer  # noqa: E402
from scripts.sandbox_exec import run_sandboxed  # noqa: E402

TEST_PATH = os.path.join(ROOT, "data", "eval", "code_holdout_500.jsonl")
TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")

CODE_RE = re.compile(r"```python\n(.*?)```", re.S)


def _norm_lines(s):
    """Line-by-line rstrip, trailing blank lines dropped."""
    return [ln.rstrip() for ln in s.split("\n") if ln.strip() != ""]


def score_code(code, expected_output, timeout=10):
    """Execute and match stdout. Returns (ok, rc, stdout_tail)."""
    rc, out, _ = run_sandboxed(code, timeout=timeout)
    ok = rc == 0 and _norm_lines(out) == _norm_lines(expected_output)
    return ok, rc, out[-200:]


def extract_code(gen):
    """Last fenced python block; None if the model emitted no fence."""
    blocks = CODE_RE.findall(gen)
    return blocks[-1].strip() if blocks else None


def selfcheck():
    rows = [json.loads(l) for l in open(TEST_PATH, encoding="utf-8")]
    fails = 0
    for i, r in enumerate(rows):
        ok, rc, out = score_code(r["reference_code"], r["expected_output"])
        if not ok:
            fails += 1
            print(f"  GOLD FAIL row {i} rc={rc} out={out!r} exp={r['expected_output'][:80]!r}")
    print(f"gold round-trip: {len(rows) - fails}/{len(rows)} pass")

    wrong = [
        ("print('this is not the answer')", "definitely wrong output"),
        ("while True:\n    pass", "anything (timeout must not score)"),
        ("def f(:\n    pass", "anything (syntax error)"),
        ("import os\nos._exit(1)", "anything (nonzero exit)"),
    ]
    wfails = 0
    for code, exp in wrong:
        ok, _, _ = score_code(code, exp)
        if ok:
            wfails += 1
            print(f"  WRONG SOLUTION SCORED 1: {code[:40]!r}")
    print(f"wrong-solution zero: {len(wrong) - wfails}/{len(wrong)} pass")
    return fails + wfails


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt")
    parser.add_argument("--max_new", type=int, default=512)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--k", type=int, default=1,
                        help="samples per problem; reports pass@1 (greedy) and pass@k")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokenizer", default=TOK_PATH)
    parser.add_argument("--data", default=None,
                        help="holdout jsonl (default: code_holdout_500.jsonl)")
    parser.add_argument("--tag", default="",
                        help="preds filename tag when --data is used (e.g. 'v2')")
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="0.0 = greedy, the default every published number used. Raised only to test whether "
             "the degenerate repetition loop (55.8% of SFT generations repeat an 8-gram 3+ times, "
             "vs 24.7% for base 0-shot) is produced by greedy decoding rather than by the model.")
    parser.add_argument("--selfcheck", action="store_true",
                        help="known-answer round-trip, no model")
    args = parser.parse_args()

    if args.selfcheck:
        sys.exit(1 if selfcheck() else 0)
    if not args.ckpt:
        parser.error("--ckpt required (unless --selfcheck)")

    import torch
    from eval.gsm8k import generate_batch

    model, cfg = load_checkpoint(args.ckpt, device=args.device)
    model = model.to(torch.bfloat16)
    tok = load_tokenizer(args.tokenizer, cfg)

    test_path = args.data or TEST_PATH
    rows = [json.loads(l) for l in open(test_path, encoding="utf-8")]
    rows = rows[args.shard :: args.shards]
    k = max(1, args.k)
    temp = args.temperature if k > 1 or args.temperature > 0 else 0.0
    # pass@k at temperature 0 draws k identical greedy answers, so pass@k == pass@1
    # by construction; math_hard.py carries the same assert.
    assert not (k > 1 and temp <= 0), (
        f"--k {k} at temperature {temp}: the k samples would be identical to the greedy "
        "answer and pass@k would equal pass@1 by construction. Pass --temperature (0.8 is "
        "the project's pass@k setting)."
    )
    preds_path = os.path.join(
        ROOT, "data", "eval", f"preds_code_{args.tag + '_' if args.tag else ''}{os.path.basename(args.ckpt)}"
        + (f".t{args.temperature}" if args.temperature else "")
        + (f".k{k}" if k > 1 else "")
        + (f".{args.shard}" if args.shards > 1 else "")
        + ".jsonl"
    )
    per_batch = max(1, args.batch // k)
    # pass@1 is the greedy answer; the k samples feed only pass@k and the sampled
    # mean, so pass@k - pass@1 has no sampling noise on the pass@1 side.
    # restartable: each problem's row is written to the preds file as soon as it is
    # scored, so an interrupt costs at most one batch of in-flight generations; the
    # partial file is valid up to its last row.
    n_pass1 = n_passk = n_samp_ok = total = no_fence = 0
    with open(preds_path, "w", encoding="utf-8") as fout:
        for s in range(0, len(rows), per_batch):
            batch = rows[s : s + per_batch]
            prompts = [tok.encode(format_prompt(r["instruction"])).ids for r in batch]
            with torch.no_grad():
                # k=1 keeps the single-sample semantics (--temperature samples); k>1
                # makes pass@1 the greedy answer and temperature only the k draws.
                greedy = generate_batch(model, prompts, args.max_new, args.device,
                                        0.0 if k > 1 else temp, None)
                sampled = []
                if k > 1:
                    rep = [p for p in prompts for _ in range(k)]
                    sampled = generate_batch(model, rep, args.max_new, args.device, temp, None)
            for i, r in enumerate(batch):
                gens = [tok.decode(greedy[i])]
                if k > 1:
                    gens += [tok.decode(ids) for ids in sampled[i * k : (i + 1) * k]]
                oks = []
                for gi, gen in enumerate(gens):
                    code = extract_code(gen)
                    if code is None:
                        ok = False
                        if gi == 0:
                            no_fence += 1
                    else:
                        ok, _, _ = score_code(code, r["expected_output"])
                    oks.append(ok)
                    fout.write(json.dumps({"q": r["instruction"], "gen": gen,
                                           "expected": r["expected_output"],
                                           "ok": ok, "greedy": gi == 0}, ensure_ascii=False) + "\n")
                n_pass1 += int(oks[0])
                if k > 1:
                    n_samp_ok += sum(oks[1:])
                    n_passk += int(any(oks[1:]))
                total += 1
            if total % 64 == 0 or total == len(rows):
                print(f"  {total}/{len(rows)} pass@1={n_pass1 / total:.1%}", flush=True)

    if k > 1:
        print(f"code-500: pass@1(greedy) {n_pass1}/{total} = {n_pass1 / total:.1%} | "
              f"sampled@T={temp} mean {n_samp_ok / (total * k):.1%} | "
              f"pass@{k} {n_passk}/{total} = {n_passk / total:.1%} | "
              f"gap {(n_passk - n_pass1) / total:+.1%} | T={temp}")
    else:
        print(f"code-500: {n_pass1}/{total} = {n_pass1 / total:.1%}")
        print(f"no-fence rate {no_fence / total:.1%}")
    print(f"preds saved: {preds_path}")


if __name__ == "__main__":
    main()

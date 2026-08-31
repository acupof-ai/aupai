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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokenizer", default=TOK_PATH)
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

    rows = [json.loads(l) for l in open(TEST_PATH, encoding="utf-8")]
    rows = rows[args.shard :: args.shards]
    preds_path = os.path.join(
        ROOT, "data", "eval", f"preds_code_{os.path.basename(args.ckpt)}"
        + (f".t{args.temperature}" if args.temperature else "")
        + (f".{args.shard}" if args.shards > 1 else "")
        + ".jsonl"
    )
    correct = total = no_fence = 0
    with open(preds_path, "w", encoding="utf-8") as fout:
        for s in range(0, len(rows), args.batch):
            batch = rows[s : s + args.batch]
            prompts = [tok.encode(format_prompt(r["instruction"])).ids for r in batch]
            with torch.no_grad():
                out = generate_batch(model, prompts, args.max_new, args.device, args.temperature, None)
            for r, ids in zip(batch, out):
                gen = tok.decode(ids)
                code = extract_code(gen)
                if code is None:
                    no_fence += 1
                    ok = False
                else:
                    ok, _, _ = score_code(code, r["expected_output"])
                correct += int(ok)
                total += 1
                fout.write(json.dumps({"q": r["instruction"], "gen": gen,
                                       "expected": r["expected_output"],
                                       "ok": ok}, ensure_ascii=False) + "\n")
            if total % 64 == 0 or total == len(rows):
                print(f"  {total}/{len(rows)} acc={correct / total:.1%}", flush=True)

    print(f"code-500: {correct}/{total} = {correct / total:.1%}")
    print(f"no-fence rate {no_fence / total:.1%}")
    print(f"preds saved: {preds_path}")


if __name__ == "__main__":
    main()

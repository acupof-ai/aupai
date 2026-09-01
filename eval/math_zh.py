#!/usr/bin/env python3
"""Math eval on the 500-problem held-out set (data/eval/math_test_500.jsonl).

Greedy generation with the SFT prompt format, \\boxed{} extraction via
algorithms.rlvr_reward (falls back to 答案是：...), exact/numeric match.

Usage: python eval/math_zh.py --ckpt ckpt_sft_math.pt [--max_new 512] [--batch 16]
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
os.environ.setdefault("FLA_FLASH_KDA", "0")

import fone  # noqa: E402
from eval.gsm8k import generate_batch  # noqa: E402
from algorithms.rlvr_reward import reward_fn, extract_boxed  # noqa: E402
from scripts.loader import format_prompt, load_checkpoint, load_tokenizer  # noqa: E402

TEST_PATH = os.path.join(ROOT, "data", "eval", "math_test_500.jsonl")
TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
from scripts.eqcheck import check_steps  # noqa: E402

ANS_RE = re.compile(r"答案是[:：]\s*(.+?)(?:[。\n]|$)")


def score(gen, gold):
    """gold is the full solution text; extract its boxed answer first."""
    gold_ans = extract_boxed(gold)
    if gold_ans is None:
        return 0.0
    if extract_boxed(gen) is not None:
        return reward_fn(gen, gold_ans)
    m = ANS_RE.search(gen)
    if m:
        return reward_fn(f"\\boxed{{{m.group(1).strip()}}}", gold_ans)
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--max_new", type=int, default=512)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--shards", type=int, default=1, help="split the test set across N processes")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokenizer", default=TOK_PATH, help="vocabulary the checkpoint was trained on")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="sampling temperature; 0 = greedy")
    parser.add_argument("--k", type=int, default=1,
        help="samples per problem; reports pass@1 (greedy) and pass@k. Mirrors code_zh's "
             "semantics exactly: at k>1 pass@1 is the GREEDY answer and --temperature "
             "applies only to the k draws, so pass@k - pass@1 carries no sampling noise on "
             "the pass@1 side. Added 2026-09-01 for the sampled arm -- math-500 had no k at "
             "all, so the pre-registered pass@8 was not runnable on it "
             "(docs/lessons/sampled_arm_prereg.md).")
    parser.add_argument("--no_rep_stop", action="store_true",
        help="disable the repetition stop. It halts a generation when a whitespace 8-gram "
             "repeats 3x -- which IS the degenerate case -- so under greedy it fires on most "
             "generations and the recorded text is a TRUNCATED PREFIX. Every degeneration rate "
             "on record was measured over that truncation (e1, 2026-09-01). Required for the "
             "sampled arm: left on, it fires at different rates in the greedy and sampled arms "
             "and confounds the treatment with a decode-time intervention "
             "(docs/lessons/sampled_arm_prereg.md).")
    parser.add_argument("--force", action="store_true",
        help="overwrite an existing predictions file (default: refuse; the rows are the only copy)")
    parser.add_argument("--run", default=None,
        help="name this run so predictions version instead of colliding: preds_x.<run>.jsonl")
    args = parser.parse_args()
    # Before the model load, not after: this is an argument error, and paying a 959MB
    # checkpoint load to discover one wastes minutes and a card. code_zh asserts after
    # its load for the same reason it should not.
    assert not (args.k > 1 and args.temperature == 0.0), (
        f"--k {args.k} at temperature 0: the k samples would be identical to the greedy "
        "answer and pass@k would equal pass@1 by construction. Pass --temperature (0.8 is "
        "the project's pass@k setting)."
    )

    # dtype through load_checkpoint (a3a0de0 upcasts KDA A_log/dt_bias to fp32
    # after the cast); a separate .to(bf16) here would undo the upcast.
    model, cfg = load_checkpoint(args.ckpt, device=args.device, dtype=torch.bfloat16)
    tok = load_tokenizer(args.tokenizer, cfg)
    # A FoNE checkpoint must decode through fone: tok.decode emits the [NUM] token
    # itself, so no answer parses and the score collapses for a non-model reason.
    fone_on = getattr(cfg, "fone", False)
    num_id = getattr(cfg, "num_id", None)

    rows = [json.loads(l) for l in open(TEST_PATH, encoding="utf-8")]
    rows = rows[args.shard :: args.shards]
    preds_path = os.path.join(
        ROOT, "data", "eval", f"preds_{os.path.basename(args.ckpt)}"
        + (f".t{args.temperature}" if args.temperature else "")
        + (f".{args.shard}" if args.shards > 1 else "")
        + ".jsonl"
    )
    k = max(1, args.k)
    temp = args.temperature if k > 1 or args.temperature > 0 else 0.0
    correct = n_passk = n_samp_ok = total = 0
    n_box = n_eq = n_bad = n_rewrite = tot_len = 0
    by_steps = {}
    with open_artifact(preds_path, force=args.force, run=args.run) as fout:
        # --run versions the path, so the handle's name is the file that exists.
        out_path = fout.name
        per_batch = max(1, args.batch // k)
        for s in range(0, len(rows), per_batch):
            batch = rows[s : s + per_batch]
            texts_in = [format_prompt(r["instruction"]) for r in batch]
            if fone_on:
                prompts, pvals = fone.encode_prompts(texts_in, tok, num_id)
            else:
                prompts, pvals = [tok.encode(t).ids for t in texts_in], None
            with torch.no_grad():
                # k=1 keeps the single-sample semantics (--temperature samples); k>1
                # makes pass@1 the greedy answer and temperature only the k draws.
                out = generate_batch(model, prompts, args.max_new, args.device,
                                     0.0 if k > 1 else temp, pvals,
                                     tokenizer=tok, rep_stop=not args.no_rep_stop)
                sampled = []
                if k > 1:
                    # FoNE carries a per-position value list per prompt, so the repeat has
                    # to cover both or the values no longer align with their prompts.
                    rep = [p for p in prompts for _ in range(k)]
                    rep_v = [v for v in pvals for _ in range(k)] if fone_on else None
                    sampled = generate_batch(model, rep, args.max_new, args.device, temp, rep_v,
                                             tokenizer=tok, rep_stop=not args.no_rep_stop)
            out_ids, out_vals = out if fone_on else (out, [None] * len(batch))
            s_ids, s_vals = (sampled if fone_on else (sampled, [None] * len(sampled))) if k > 1 else ([], [])
            for bi, (r, ids, vs) in enumerate(zip(batch, out_ids, out_vals)):
                gen = fone.decode_text(ids, vs, tok, num_id) if fone_on else tok.decode(ids)
                ok = score(gen, r["output"])
                correct += int(ok)
                total += 1
                if k > 1:
                    oks = []
                    for j in range(bi * k, (bi + 1) * k):
                        g = (fone.decode_text(s_ids[j], s_vals[j], tok, num_id)
                             if fone_on else tok.decode(s_ids[j]))
                        o = score(g, r["output"])
                        oks.append(o)
                        fout.write(json.dumps({"q": r["instruction"], "gold": r["output"][-80:],
                                               "gen": g, "ok": o, "greedy": False},
                                              ensure_ascii=False) + "\n")
                    n_samp_ok += sum(oks)
                    n_passk += int(any(oks))
                n_box += extract_boxed(gen) is not None
                n_rewrite += "解答" in gen
                tot_len += len(ids)
                e, b = check_steps(gen)
                n_eq += e; n_bad += b
                # NOT `k`: that names the sample count now, and rebinding it here made the
                # pass@k line below divide by the last difficulty bucket (max 3) and print
                # "pass@3" for a --k 8 run. Silent, plausible, wrong.
                bucket = min(check_steps(r["output"])[0], 3)  # difficulty by gold step count
                by_steps.setdefault(bucket, [0, 0])
                by_steps[bucket][0] += int(ok); by_steps[bucket][1] += 1
                fout.write(json.dumps({"q": r["instruction"], "gold": r["output"][-80:],
                                       "gen": gen, "ok": ok, "greedy": True},
                                      ensure_ascii=False) + "\n")
            if total % 64 == 0 or total == len(rows):
                print(f"  {total}/{len(rows)} acc={correct / total:.1%}", flush=True)

    # attest what was WRITTEN, not what was requested: --run versions the path, and
    # attesting preds_path recorded a hash for a file this run never touched.
    attest(out_path)  # the citation contract: the writer proves these bytes existed
    print(f"math-500: {correct}/{total} = {correct / total:.1%} (t={temp if k == 1 else 0.0})")
    if k > 1:
        print(f"TOTAL math-500: pass@1(greedy) {correct / total:.1%} ({correct}/{total}) | "
              f"sampled mean {n_samp_ok / (total * k):.1%} | pass@{k} {n_passk / total:.1%} "
              f"({n_passk}/{total}) | gap {(n_passk - correct) / total:+.1%} | t={temp}")
    print(f"boxed rate {n_box / total:.1%} | rewrite('解答') rate {n_rewrite / total:.1%} | "
          f"avg gen tokens {tot_len / total:.0f} | step-eq wrong {n_bad}/{n_eq} = {n_bad / max(n_eq, 1):.1%}")
    print("acc by gold steps: " + ", ".join(
        f"{b}{'+' if b == 3 else ''}: {c}/{n}={c / n:.0%}" for b, (c, n) in sorted(by_steps.items())))
    print(f"preds saved: {preds_path}")


if __name__ == "__main__":
    main()

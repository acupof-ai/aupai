#!/usr/bin/env python3
"""Sampled pass@k HumanEval, sharing the greedy path's judge and controls.

The p1 acceptance spec (docs/standards/p1_data_recipe.md): temp 0.2 / top-p 0.95
/ 20-sample pass@1, sharing ONE judge with the greedy path, both reported. This
file imports judge / truncate / run_control / STOPS from eval.humaneval_gen --
the baseline 0/164 and the SFT 3/164 were scored with that judge, and a sampled
number is only comparable on the same scorer. The generation loop is the greedy
loop with argmax replaced by nucleus sampling: same cfg.seq window, eos tid 1,
stop check every 16 tokens, max_new 280.

pass@1 estimator: c/n per problem -- for k=1 the unbiased estimator
1 - C(n-c,1)/C(n,1) = c/n. Greedy is n=1.

    CUDA_VISIBLE_DEVICES="" python3 eval/humaneval_sample.py --control   # scorer self-check, CPU
    python3 eval/humaneval_sample.py --ckpt <ckpt>                       # greedy + 20-sample arms
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from eval_artifacts import attest, open_artifact  # noqa: E402

import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# One judge, one stop list, one control set: imported, not copied.
from eval.humaneval_gen import (  # noqa: E402
    DATA_PATH,
    STOPS,
    TOK_PATH,
    judge,
    run_control,
    truncate,
)


def nucleus(logits, temperature, top_p):
    """logits: [1, V] -> [1, 1] sampled id. Standard nucleus: sort, keep the
    smallest set whose cumulative mass reaches top_p (always keeping rank 0)."""
    probs = torch.softmax(logits / temperature, dim=-1)
    sp, si = torch.sort(probs, descending=True, dim=-1)
    cum = torch.cumsum(sp, dim=-1)
    keep = cum <= top_p
    keep[..., 1:] = keep[..., :-1].clone()
    keep[..., 0] = True
    sp = sp * keep
    sp = sp / sp.sum(dim=-1, keepdim=True)
    return si.gather(-1, torch.multinomial(sp, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--max_new", type=int, default=280)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=20, help="samples per problem (SmolLM published 24 at this setting)")
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--no-greedy", action="store_true", help="skip the greedy arm (reproduced separately)")
    ap.add_argument("--control", action="store_true", help="run the known-answer controls only, no model (CPU)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--run", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    probs = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    print(f"HumanEval sampled: {len(probs)} problems, n={args.n}, temp={args.temp}, top_p={args.top_p}", flush=True)
    run_control(probs)
    if args.control:
        return
    if not args.ckpt:
        ap.error("--ckpt required (unless --control)")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        sys.exit("REFUSING: CUDA_VISIBLE_DEVICES is unset, so cuda:0 is physical "
                 "GPU 0 -- tileRL's card. Set it to your granted card.")

    from scripts.loader import load_checkpoint
    from tokenizers import Tokenizer
    model, cfg = load_checkpoint(args.ckpt, device=args.device)
    model.eval()
    tok = Tokenizer.from_file(TOK_PATH)
    torch.manual_seed(args.seed)

    def gen(prompt, temperature):
        """The greedy loop with argmax replaced by nucleus sampling at
        temperature > 0. Everything else verbatim from humaneval_gen.gen."""
        ids = tok.encode(prompt).ids
        x = torch.tensor([ids], device=args.device)
        new = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for step in range(args.max_new):
                lg = model(x[:, -cfg.seq:])[0][:, -1]
                nxt = lg.argmax(-1, keepdim=True) if temperature <= 0 else nucleus(lg, temperature, args.top_p)
                tid = nxt.item()
                if tid == 1:
                    break
                new.append(tid)
                x = torch.cat([x, nxt], 1)
                if step % 16 == 15:
                    if any(st in tok.decode(new) for st in STOPS):
                        break
        return truncate(tok.decode(new))

    preds_path = os.path.join(
        ROOT, "data", "eval",
        f"preds_humaneval_sample_{os.path.basename(str(args.ckpt).rstrip('/'))}.jsonl")
    t0 = time.time()
    # The format_sft result this run must reproduce (docs/standards/p1_data_recipe.md:
    # "moved it to 3/164 with the empty rate at 72/164"). Greedy is deterministic, so
    # the match must be exact; a miss says the shared judge/STOPS/truncate diverged
    # from humaneval_gen and the sampled arm would be meaningless (4c, 2026-09-09).
    GREEDY_EXPECTED_PASS, GREEDY_EXPECTED_EMPTY = 3, 72
    greedy_pass = greedy_empty = 0
    sample_pass = 0.0
    ctrl_pass = 0
    sample_empty = 0
    with open_artifact(preds_path, force=args.force, run=args.run) as fout:
        out_path = fout.name
        fout.write(json.dumps({
            "_header": 1,
            "ckpt": os.path.basename(str(args.ckpt).rstrip("/")),
            "data": os.path.basename(args.data),
            "n": args.n, "temp": args.temp, "top_p": args.top_p, "seed": args.seed,
            "max_new": args.max_new, "stops": STOPS, "n_problems": len(probs),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, ensure_ascii=False) + "\n")

        # Phase 1: greedy, all 164 problems. Gate the sampled arm on reproduction.
        if not args.no_greedy:
            for i, p in enumerate(probs, 1):
                g = gen(p["prompt"], 0.0)
                ok = judge(p, g)
                empty = not g.strip()
                greedy_pass += int(ok)
                greedy_empty += int(empty)
                fout.write(json.dumps(
                    {"phase": "greedy", "task_id": p["task_id"], "gen": g, "ok": ok, "empty": empty},
                    ensure_ascii=False) + "\n")
                fout.flush()
                if i % 20 == 0 or i == len(probs):
                    print(f"  greedy {i}/{len(probs)}  pass@1 = {greedy_pass}/{i}  "
                          f"empty {greedy_empty}/{i}  ({time.time() - t0:.0f}s)", flush=True)
            print(f"\nHUMANEVAL pass@1 (greedy) = {greedy_pass}/{len(probs)} = "
                  f"{100 * greedy_pass / len(probs):.2f}%  empty {greedy_empty}/{len(probs)}", flush=True)
            if (greedy_pass, greedy_empty) != (GREEDY_EXPECTED_PASS, GREEDY_EXPECTED_EMPTY):
                sys.exit(
                    f"GREEDY REPRODUCTION FAILED: {greedy_pass}/{len(probs)} pass, "
                    f"{greedy_empty}/{len(probs)} empty "
                    f"(expected {GREEDY_EXPECTED_PASS}/{GREEDY_EXPECTED_EMPTY}). The shared "
                    "judge/STOPS/truncate diverged from humaneval_gen; NOT running the sampled arm.")
            print("greedy reproduction OK -- running sampled arm", flush=True)

        # Phase 2: n samples per problem.
        for i, p in enumerate(probs, 1):
            samples = []
            for _ in range(args.n):
                c = gen(p["prompt"], args.temp)
                samples.append({"gen": c, "ok": judge(p, c)})
            c = sum(s["ok"] for s in samples)
            sample_pass += c / args.n
            ctrl_pass += int(judge(p, p["canonical_solution"]))
            sample_empty += sum(not s["gen"].strip() for s in samples)
            fout.write(json.dumps(
                {"phase": "sample", "task_id": p["task_id"], "c": c, "samples": samples},
                ensure_ascii=False) + "\n")
            fout.flush()
            if i % 20 == 0 or i == len(probs):
                print(f"  sample {i}/{len(probs)}  pass@1(n={args.n}) = "
                      f"{100 * sample_pass / i:.2f}%  ({time.time() - t0:.0f}s)", flush=True)

        # The known-answer control through the SAMPLED phase's own judge call, plus
        # the sample empty rate: a pass@1 that quietly counts empty completions as
        # failures is a format number, and a sampled figure without its control is
        # not a figure (e1-58).
        n_samples = len(probs) * args.n
        fout.write(json.dumps(
            {"phase": "sample_summary", "control_canonical_pass": ctrl_pass,
             "control_n": len(probs), "sample_empty": sample_empty,
             "sample_empty_n": n_samples}, ensure_ascii=False) + "\n")
        print(f"\nCONTROL canonical_solution through sampled path = "
              f"{ctrl_pass}/{len(probs)} (must be {len(probs)})", flush=True)
        print(f"sampled empty completions = {sample_empty}/{n_samples} = "
              f"{100 * sample_empty / n_samples:.1f}%", flush=True)
        if ctrl_pass != len(probs):
            sys.exit(
                f"SAMPLED CONTROL FAILED: canonical_solution scored {ctrl_pass}/{len(probs)} "
                "through the sampled phase's judge -- the pass@1 above is not a figure.")

    attest(out_path)
    print(f"\nHUMANEVAL pass@1 (n={args.n}, temp={args.temp}, top_p={args.top_p}) = "
          f"{sample_pass}/{len(probs)} = {100 * sample_pass / len(probs):.2f}%", flush=True)
    print(f"preds saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""What the model does at the token position HumanEval hands it control.

# restartable: one checkpoint load, 164 forward passes plus 164 short greedy rollouts,
# ~3 min. An interrupt loses the pass and nothing else; the JSON is written at the end.

ckpt_1.5b-a0.2b-e48_30b.pt.step34000 scores 0/164 on HumanEval with a 164/164
known-answer control, and 160 of those completions are EMPTY. This probe measures why,
and it is the before/after instrument for runs/prereg.jsonl#format_sft_humaneval_0909.

IT REPORTS TWO THINGS BECAUSE ONE OF THEM ALONE MISLEADS, and that is not a design
preference -- it is what the first version of this measurement got wrong. Reading only
P(<eos>) at the prompt end gave 0.3131 on a hand-typed prompt and the conclusion "the
model wants to stop". On all 164 the same quantity is 0.2821 and is the argmax on 34, so
it explains 33 of the 160 empty completions. The other 127 are the model writing '\\ndef '
-- the NEXT function's signature -- which the standard stop set truncates to nothing. A
before/after that watched only P(<eos>) would call a 77%-moving intervention a null.

So: section 1 is the distribution at the boundary (P(<eos>), and whether it is the
argmax); section 2 is what greedy actually produces there, classified. Same run, same
checkpoint, same prompts.

THE PROMPTS ARE THE EVAL SET, not hand-written. A hand-written prompt measures the
author's idea of the distribution, and a post-SFT rerun against a different idea is not a
comparison.

Usage:
    CUDA_VISIBLE_DEVICES=4 python3 probes/humaneval_first_move.py --ckpt <ckpt> [--json out.json]
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "eval", "humaneval", "humaneval_164.jsonl")
TOK = os.path.join(ROOT, "data", "tokenizer.json")
EOS = 1  # pinned by harness check `pinned_ids`; a rebuild that moved it turns that check red.

# Verbatim from eval/humaneval_gen.py. The classification below is only meaningful against
# the stop set the scored run actually uses -- a probe with its own list would report a
# cause the eval never acts on.
STOPS = [
    "\ndef ",
    "\nclass ",
    "\nif __name__",
    "\nprint(",
    "\n#",
    "\n@",
    "\nassert ",
    '\n"""',
    "\nimport ",
    "\nfrom ",
]

# One plausible first body line, appended to make the contrast position. A CONSTANT and not
# the model's own greedy continuation on purpose: the greedy token differs per problem and
# per checkpoint, so a post-SFT rerun would measure a different position and the two numbers
# would not subtract.
BODY_PREFIX = "    if not "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--tokenizer", default=TOK)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--raw_tokens", type=int, default=40, help="untruncated greedy tokens per problem for the cause split"
    )
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.device.startswith("cuda") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        sys.exit(
            "REFUSING: CUDA_VISIBLE_DEVICES is unset, so cuda:0 is physical GPU 0 -- not "
            "necessarily a card this repo was granted. Set it to your granted card."
        )

    import torch
    from tokenizers import Tokenizer

    from scripts.loader import load_checkpoint

    with open(a.data, encoding="utf-8") as fh:
        probs = [json.loads(x) for x in fh if x.strip()]
    model, cfg = load_checkpoint(a.ckpt, device=a.device)
    model.eval()
    tok = Tokenizer.from_file(a.tokenizer)
    out = {"ckpt": os.path.basename(a.ckpt), "n": len(probs)}

    def dist(text):
        x = torch.tensor([tok.encode(text).ids], device=a.device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pr = torch.softmax(model(x)[0][:, -1].float(), -1)[0]
        v, i = torch.topk(pr, a.topk)
        top = [(tok.decode([int(t)]) or f"<id{int(t)}>", float(q)) for q, t in zip(v, i, strict=True)]
        return float(pr[EOS]), top, int(pr.argmax())

    print(f"### 1 -- THE DISTRIBUTION AT THE BOUNDARY, {len(probs)} HumanEval prompts\n", flush=True)
    out["positions"] = {}
    for name, suffix in (("prompt_end", ""), ("body_started", BODY_PREFIX)):
        vals, argmax_eos, first_top = [], 0, None
        for p in probs:
            v, top, am = dist(p["prompt"] + suffix)
            vals.append(v)
            argmax_eos += int(am == EOS)
            first_top = first_top or top
        rec = {
            "mean_p_eos": statistics.fmean(vals),
            "median_p_eos": statistics.median(vals),
            "min_p_eos": min(vals),
            "max_p_eos": max(vals),
            # The count, not the mean, is what greedy decoding acts on.
            "n_eos_is_argmax": argmax_eos,
            "top_tokens_first_problem": first_top,
        }
        out["positions"][name] = rec
        print(
            f"[{name}]  mean P(<eos>) {rec['mean_p_eos']:.4f}  median {rec['median_p_eos']:.4f}  "
            f"range {rec['min_p_eos']:.4f}..{rec['max_p_eos']:.4f}"
        )
        print(f"   <eos> is argmax on {argmax_eos}/{len(probs)} = {100 * argmax_eos / len(probs):.1f}%")
        print(
            f"   top-{a.topk} at {probs[0]['task_id']}: " + ", ".join(f"{t!r} {q:.4f}" for t, q in first_top),
            flush=True,
        )
    d = out["positions"]["prompt_end"]["mean_p_eos"] - out["positions"]["body_started"]["mean_p_eos"]
    out["prompt_end_minus_body_started"] = d
    print(f"\n   CONTRAST prompt_end - body_started = {d:+.4f}", flush=True)

    def greedy_raw(prompt):
        ids = tok.encode(prompt).ids
        x = torch.tensor([ids], device=a.device)
        new, eos_first = [], False
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for _ in range(a.raw_tokens):
                nxt = model(x[:, -cfg.seq :])[0][:, -1].argmax(-1, keepdim=True)
                if int(nxt.item()) == EOS:
                    eos_first = not new
                    break
                new.append(int(nxt.item()))
                x = torch.cat([x, nxt], 1)
        return tok.decode(new), eos_first

    print(f"\n### 2 -- WHAT GREEDY PRODUCES, {a.raw_tokens} untruncated tokens\n", flush=True)
    cause = {"eos_first": 0, "stop_at_0": 0, "whitespace_only": 0, "nonempty": 0}
    by_stop, rows = {}, []
    # This section runs 164 rollouts with no KV cache and prints only at the end unless it
    # says otherwise -- measured at ~6 min on one H20, long enough that a silent run is
    # indistinguishable from a hung one.
    t0 = time.time()
    for i, p in enumerate(probs, 1):
        raw, eos_first = greedy_raw(p["prompt"])
        # The eval truncates at the first stop string; a stop string at or near position 0
        # is a completion that scores as empty despite the model having written something.
        first_cut = min([raw.find(s) for s in STOPS if raw.find(s) != -1], default=len(raw))
        if eos_first:
            c = "eos_first"
        elif not raw[:first_cut].strip() and raw.strip():
            c = "stop_at_0"
            hit = next(s for s in STOPS if raw.find(s) == first_cut)
            by_stop[hit] = by_stop.get(hit, 0) + 1
        elif not raw[:first_cut].strip():
            c = "whitespace_only"
        else:
            c = "nonempty"
        cause[c] += 1
        rows.append({"task_id": p["task_id"], "cause": c, "raw_head": raw[:80]})
        if i % 20 == 0 or i == len(probs):
            print(f"   {i}/{len(probs)}  ({time.time() - t0:.0f}s)", flush=True)
    out["cause_split"] = cause
    out["stop_string_at_cut"] = by_stop
    out["rows"] = rows
    for k, v in cause.items():
        print(f"   {k:16} {v:4d}/{len(probs)} = {100 * v / len(probs):5.1f}%")
    if by_stop:
        print(
            "   which stop string cut it: "
            + ", ".join(f"{k!r} {v}" for k, v in sorted(by_stop.items(), key=lambda x: -x[1]))
        )
    print(f"   example head: {rows[0]['raw_head']!r}", flush=True)

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)
        print(f"\n   json: {a.json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# restartable: read-only; loads a checkpoint and prints continuations, writes nothing.
"""Local continuation REPL for a BASE checkpoint (no ChatML: the pretraining corpus has none).

    python3 scripts/base_generate.py --ckpt ~/models/aupai_v41_ced_0926/v41_ced_0926_final_weights_bf16.pt \
        --tokenizer ~/models/aupai_v41_ced_0926/tokenizer.json
    python3 scripts/base_generate.py --ckpt ... --tokenizer ... --prompt "def fib(n):\\n    '''Return the n-th Fibonacci number.'''\\n"

Prompts that match pretraining: a Python signature + docstring, "Question: ...\\nAnswer:",
"问：...\\n答：". A literal \\n in the typed prompt becomes a newline. Empty line quits.
Each step re-runs the whole sequence (no KV cache), so speed falls as the text grows.
"""
import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.loader import load_checkpoint, load_tokenizer  # noqa: E402

HOLD = 16
STOPS = ["\n\n\n", "\nQuestion:", "\n问：", "\ndef ", "\nclass ", "\nif __name__"]


def pick_device(want):
    if want != "auto":
        return want
    return "cpu"  # measured 2026-09-28 on M4 Pro: cpu 7.8 tok/s vs mps 4.5 (small per-token kernels)


def generate(model, tok, prompt, device, max_new, temp, code_stops):
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], device=device)
    new, printed, text, cut = [], "", "", []
    t0 = time.time()
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(x[:, -model.cfg.seq:])[0][:, -1].float()
            if temp <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial(torch.softmax(logits / temp, -1), 1)
            tid = nxt.item()
            if tid == 1:
                break
            new.append(tid)
            x = torch.cat([x, nxt.view(1, 1)], 1)
            text = tok.decode(new)
            cut = [text.find(s) for s in (STOPS if code_stops else STOPS[:3]) if s in text]
            if cut:
                text = text[:min(cut)]
            if cut:
                sys.stdout.write(text[len(printed):])
                printed = text
                break
            safe = text[:max(len(printed), len(text) - HOLD)]  # hold back a possible stop prefix
            sys.stdout.write(safe[len(printed):])
            sys.stdout.flush()
            printed = safe
    sys.stdout.write(text[len(printed):] if not cut else "")
    dt = time.time() - t0
    print(f"\n[{len(new)} tokens, {dt:.1f}s, {len(new) / max(dt, 1e-9):.1f} tok/s]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--device", default="auto", help="auto (= cpu) | mps | cpu")
    ap.add_argument("--prompt", default=None, help="one prompt, then exit")
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy")
    args = ap.parse_args()
    device = pick_device(args.device)
    model, cfg = load_checkpoint(os.path.expanduser(args.ckpt), device="cpu", dtype=torch.bfloat16, low_mem=True)
    held = []
    if device == "mps":
        # MPS has no float64. MoEFFN pins its load counters (h_load, h_sums) to float64 in its own
        # _apply, so they are detached for the move and put back as float32 on the device; they are
        # training-side statistics and their precision does not touch the logits.
        for m in model.modules():
            for k, b in list(m._buffers.items()):
                if b is not None and b.dtype == torch.float64:
                    held.append((m, k, b.float()))
                    m._buffers[k] = None
    model = model.to(device).eval()
    for m, k, b in held:
        m._buffers[k] = b.to(device)
    model.cfg = cfg
    tok = load_tokenizer(os.path.expanduser(args.tokenizer), cfg)
    print(f"loaded on {device}; greedy" if args.temp <= 0 else f"loaded on {device}; temp {args.temp}")
    prompts = [args.prompt] if args.prompt is not None else None
    while True:
        p = prompts.pop(0) if prompts else (None if prompts is not None else input("\nprompt> "))
        if not p:
            return
        p = p.replace("\\n", "\n")
        if p.lstrip().startswith("def "):
            p = p.rstrip("\n")  # HumanEval scores the rstrip-nl arm; a trailing blank line reads as end-of-function
        print(p, end="")
        generate(model, tok, p, device, args.max_new, args.temp, code_stops=p.lstrip().startswith("def "))


if __name__ == "__main__":
    main()

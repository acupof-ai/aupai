#!/usr/bin/env python3
"""Chat with the trained model. Usage: python chat.py [prompt]"""

import os
import sys
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer

from sampling import top_p_sample
from train import TOK_PATH, HybridLM

device = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt.pt")


def generate(model, tok, prompt, max_new=512, temp=0.8, top_p=0.95):
    eos = tok.token_to_id("<eos>")
    x = torch.tensor([tok.encode(prompt).ids], device=device)
    model.eval()
    for _ in range(max_new):
        with torch.no_grad():
            logits = model(x[:, -model.cfg.seq :])[0][:, -1] / temp
        nxt = top_p_sample(logits, top_p)
        x = torch.cat([x, nxt], dim=1)
        if nxt.item() == eos:
            break
    return tok.decode(x[0].tolist(), skip_special_tokens=True)


def main():
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])  # use saved config, not current Cfg
    model = HybridLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    tok = Tokenizer.from_file(TOK_PATH)
    if len(sys.argv) > 1:
        print(generate(model, tok, sys.argv[1]))
        return
    print("(empty line to quit)")
    while True:
        try:
            q = input("问 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        print(generate(model, tok, f"问：{q}\n答："))


if __name__ == "__main__":
    main()

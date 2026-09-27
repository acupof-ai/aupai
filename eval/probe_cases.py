#!/usr/bin/env python3
# restartable: read-only; rewrites --out from scratch on every run, so an interrupt leaves nothing to resume.
"""Greedy continuations on a small graded prompt set (basic -> hard code, math, Chinese), for reading
what a base checkpoint actually writes. Code prompts carry tests and are judged with HumanEval's judge;
text prompts are only printed.

    CUDA_VISIBLE_DEVICES= python3 eval/probe_cases.py --ckpt X --data eval/probe_cases.jsonl --out runs/probe_X.jsonl
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))
import humaneval_gen as hg  # noqa: E402

TEXT_STOPS = ["\nQuestion:", "\n问：", "\n\n\n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "eval", "probe_cases.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from tokenizers import Tokenizer

    from scripts.loader import load_checkpoint
    model, cfg = load_checkpoint(args.ckpt, device="cpu")
    model.eval()
    tok = Tokenizer.from_file(hg.TOK_PATH)
    rows = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
    out = []
    for p in rows:
        is_code = "test" in p
        ids = tok.encode(p["prompt"].rstrip("\n") if is_code else p["prompt"]).ids
        x = torch.tensor([ids])
        new = []
        with torch.no_grad():
            for step in range(args.max_new):
                tid = model(x[:, -cfg.seq:])[0][:, -1].argmax(-1).item()
                if tid == 1:
                    break
                new.append(tid)
                x = torch.cat([x, torch.tensor([[tid]])], 1)
                if step % 16 == 15:
                    s = tok.decode(new)
                    if (is_code and hg.hits_stop(s, p["entry_point"])) or (
                            not is_code and any(t in s for t in TEXT_STOPS)):
                        break
        s = tok.decode(new)
        if is_code:
            s = hg.truncate(s, p["entry_point"])
            ok = bool(hg.judge(p, s, prompt_text=p["prompt"].rstrip("\n")))
        else:
            cut = min([s.find(t) for t in TEXT_STOPS if t in s] or [len(s)])
            s, ok = s[:cut], None
        out.append({"task_id": p["task_id"], "ok": ok, "gen": s})
        print(f"===== {p['task_id']} {'' if ok is None else ('PASS' if ok else 'FAIL')}\n{s.rstrip()}\n", flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_header": 1, "ckpt": os.path.basename(args.ckpt)}) + "\n")
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

"""MBPP sanitized (427) base-model continuation scorer.

Prompt shape (the in-distribution base continuation, same finding that made
HumanEval use --rstrip_nl): the sanitized "prompt" is a one-line task
description with NO signature; the required name lives only in test_list and
the canonical code. The prompt is therefore the canonical def signature line
plus the task text as a closed, indented docstring, trailing newline stripped
so the model emits its own newline+indent tokens.

Pass = prompt + completion + test_imports + test_list execs cleanly (in-process,
6s SIGALRM ceiling). Canonical control: every problem's canonical body must pass
its own tests under this exact prompt construction (run CONTROL=ALL).

Sampling: --n/--temperature mirror eval/humaneval_gen.py and share
eval/sampling.py's task-seeded RNG, so T and C checkpoints draw identical
choices per task_id (paired stage-2 protocol). CPU or CUDA.

  python3 eval/mbpp_gen.py --data data/eval/sanitized-mbpp.json \
      --ckpt <ckpt> --device cpu --rstrip_nl --n 10 --temperature 0.2 \
      --run eT_mbpp_n10temp02
"""
import argparse
import contextlib
import io
import json
import os
import re
import signal
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

DATA_PATH = os.path.join(ROOT, "data", "eval", "sanitized-mbpp.json")
TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
# Other column-0 top-level constructs end the completion; a self re-declaration of
# the entry def is kept (later def wins), matching humaneval_gen's truncate.
OTHER_STOPS = ["\nclass ", "\nif __name__", "\nprint(", "\n#", "\n@", '\nassert ']


class TO(Exception):
    pass


def _h(*_a):
    raise TO()


signal.signal(signal.SIGALRM, _h)


def signature(rec):
    line = next(l for l in rec["code"].splitlines() if re.match(r"def\s+\w", l))
    assert line.rstrip().endswith(":"), (rec.get("task_id"), line)
    return line.rstrip(), re.match(r"def\s+(\w+)", line).group(1)


def preamble(rec):
    lines = rec["code"].splitlines()
    i = next(k for k, l in enumerate(lines) if re.match(r"def\s+\w", l))
    return "\n".join(lines[:i])


def body_indent(rec):
    for l in rec["code"].splitlines():
        m = re.match(r"(\s+)\S", l)
        if m:
            return m.group(1)
    return "    "


def model_prompt(rec):
    pre = preamble(rec)
    sig, _ = signature(rec)
    ind = body_indent(rec)
    head = sig + "\n" + ind + '"""' + rec["prompt"] + "\n" + ind + '"""'
    return (pre + ("\n" if pre else "") + head).rstrip("\n")


def canonical_body(rec):
    lines = rec["code"].splitlines()
    i = next(k for k, l in enumerate(lines) if re.match(r"def\s+\w", l))
    return "\n".join(lines[i + 1:])


def judge(rec, completion):
    src = (model_prompt(rec) + completion + "\n"
           + "\n".join(rec.get("test_imports", [])) + "\n"
           + "\n".join(rec["test_list"]) + "\n")
    g = {"__name__": "__main__"}
    signal.alarm(6)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(src, g)
        return True
    except BaseException:
        return False
    finally:
        signal.alarm(0)


def truncate(raw, entry):
    cut = len(raw)
    for st in OTHER_STOPS + ["\ndef "]:
        start = 0
        while True:
            i = raw.find(st, start)
            if i == -1:
                break
            if st == "\ndef " and raw[i:i + 12 + len(entry)].startswith(f"\ndef {entry}("):
                start = i + 1
                continue
            cut = min(cut, i)
            break
    return raw[:cut]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--max_new", type=int, default=280)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--control", choices=["20", "ALL"], default=None,
                    help="judge canonical solutions, no model; ALL must pass")
    args = ap.parse_args()

    recs = json.load(open(args.data, encoding="utf-8"))
    if args.data.endswith("sanitized-mbpp.json"):
        assert len(recs) == 427, len(recs)
    if args.control:
        subset = recs if args.control == "ALL" else recs[:20]
        bad = [r["task_id"] for r in subset if not judge(r, "\n" + canonical_body(r))]
        print(f"canonical-sig control ({len(subset)}): failed", len(bad), bad[:10])
        sys.exit(1 if bad else 0)
    if args.first:
        recs = recs[: args.first]
    if args.n > 1 and args.temperature <= 0:
        ap.error(f"--n {args.n} at temperature 0 draws identical greedy; pass --temperature 0.2")

    if str(args.device).startswith("cpu"):
        if args.threads:
            torch.set_num_threads(args.threads)
        if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            sys.exit("REFUSING: --device cpu but CUDA_VISIBLE_DEVICES is unset -- set it empty.")

    from eval_artifacts import attest, open_artifact
    from tokenizers import Tokenizer

    from scripts.loader import load_checkpoint
    model, cfg = load_checkpoint(args.ckpt, device=args.device)
    model.eval()
    tok = Tokenizer.from_file(TOK_PATH)

    suffix = f".n{args.n}temp{args.temperature:g}" if args.n > 1 else ""
    preds_path = os.path.join(
        ROOT, "data", "eval",
        f"preds_mbpp_{os.path.basename(str(args.ckpt).rstrip('/'))}.{args.run}{suffix}.jsonl")
    t0 = time.time()
    npass = nempty = 0
    with open_artifact(preds_path, force=args.force, run=args.run) as fout:
        out_path = fout.name
        fout.write(json.dumps({
            "_header": 1, "variant": "sig-docstring-rstrip", "benchmark": "mbpp-sanitized",
            "n_problems": len(recs), "n": args.n, "temperature": args.temperature,
            "max_new": args.max_new, "ckpt": os.path.basename(str(args.ckpt).rstrip("/")),
        }, ensure_ascii=False) + "\n")
        for i, rec in enumerate(recs, 1):
            _sig, entry = signature(rec)
            prompt = model_prompt(rec)
            if args.n > 1:
                from eval.sampling import sample_completions
                raws = sample_completions(model, tok, tok.encode(prompt).ids, rec["task_id"],
                                          args.n, args.temperature, args.max_new, args.device,
                                          cfg.seq)
            else:
                raws = [_greedy(model, tok, tok.encode(prompt).ids, args.max_new, args.device,
                                cfg.seq)]
            for si, raw in enumerate(raws):
                c = truncate(raw, entry)
                ok = judge(rec, c)
                npass += int(ok)
                nempty += int(not c.strip())
                row = {"task_id": rec["task_id"], "gen": c, "ok": ok, "empty": not c.strip()}
                if args.n > 1:
                    row["sample_idx"] = si
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 25 == 0 or i == len(recs):
                denom = i * args.n
                print(f"  {i}/{len(recs)} tasks  c={npass}/{denom} "
                      f"({100 * npass / denom:.2f}%)  ({__import__('time').time() - t0:.0f}s)",
                      flush=True)
    attest(out_path)
    denom = len(recs) * args.n
    label = f"n={args.n} T={args.temperature:g}" if args.n > 1 else "greedy"
    print(f"MBPP sig-rstrip ({label}) pass = {npass}/{denom} = {100 * npass / denom:.2f}%")
    print(f"empty {nempty}/{denom}")
    print("preds:", out_path, flush=True)


@torch.no_grad()
def _greedy(model, tok, prompt_ids, max_new, device, seq_window):
    x = torch.tensor([prompt_ids], device=torch.device(device))
    for _ in range(max_new):
        lg = model(x[:, -seq_window:])[0][:, -1]
        nxt = lg.argmax(-1, keepdim=True)
        if nxt.item() == 1:
            break
        x = torch.cat([x, nxt], 1)
    return tok.decode(x[0, len(prompt_ids):].tolist())


if __name__ == "__main__":
    main()

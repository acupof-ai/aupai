#!/usr/bin/env python3
"""Generative HumanEval pass@1, normalized from 4c's pod _humaneval_run.py.

The baseline (0/164 on ckpt_1.5b-a0.2b-e48_30b.pt.step34000, exp
humaneval_pass1_step34000) was measured with the pod-only runner this file
normalizes. The model path is preserved verbatim -- same STOPS list, same
exec+SIGALRM judge, same greedy loop (cfg.seq window, eos tid 1, stop check
every 16 tokens) -- so the docstring arm on the same checkpoint must reproduce
0/164. fone is False on that checkpoint's cfg (probed 2026-09-09), so plain
tokenizer encoding is the correct path.

Added over the pod runner, none of it touching the model path:
- --strip-docstrings: the sig-only arm (3b's negative control). The docstring
  is the first triple-quoted block in the prompt; stripping it leaves the def
  header, so the model must start the body cold.
- known-answer control pair: canonical solutions must pass (all 164, on the
  prompt variant the run uses) and a constant-False body on HumanEval/0 must
  FAIL. The pod runner had only the positive half; without the negative half a
  model zero is indistinguishable from a scorer that passes everything.
- predictions artifact: the pod runner saved no completions, so the baseline
  empty-completion rate was unrecoverable. This runner saves every completion.

Prereg: runs/prereg.jsonl#format_sft_humaneval_0909 (threshold >=5/164,
Fisher one-sided p=0.030 vs the 0/164 baseline; failure discriminators
empty-completion rate and P(<eos>) after docstring).

Usage:
    CUDA_VISIBLE_DEVICES="" python3 eval/humaneval_gen.py --control   # scorer self-check, CPU
    python3 eval/humaneval_gen.py --ckpt <ckpt>                       # docstring arm
    python3 eval/humaneval_gen.py --ckpt <ckpt> --strip-docstrings    # sig-only arm
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from eval_artifacts import attest, open_artifact  # noqa: E402

import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "eval", "humaneval", "humaneval_164.jsonl")
TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")

# Verbatim from _humaneval_run.py: the baseline 0/164 was scored with this list.
STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n@", "\nassert ",
         '\n"""', "\nimport ", "\nfrom "]


class TO(Exception):
    pass


def _h(*a):
    raise TO()


signal.signal(signal.SIGALRM, _h)


def judge(prob, completion):
    """prompt + completion + test + check(entry_point); pass iff clean exit.

    Verbatim from _humaneval_run.py: in-process exec with a 6s SIGALRM ceiling.
    NOT the chroot sandbox -- the baseline was scored this way, and the SFT
    comparison is only valid on the same scorer.
    """
    src = prob["prompt"] + completion + "\n" + prob["test"] + f"\ncheck({prob['entry_point']})\n"
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


def truncate(s):
    """Verbatim from _humaneval_run.py."""
    cut = len(s)
    for st in STOPS:
        i = s.find(st)
        if i != -1:
            cut = min(cut, i)
    return s[:cut]


def strip_docstring(prompt):
    """The sig-only arm: drop the first triple-quoted block, keep the def header.

    HumanEval prompts are signature + docstring, so the first triple-quoted
    block IS the docstring. Prompts without one (comment-only or bare) are
    returned unchanged -- those problems are identical across arms.
    """
    m = re.search(r'("""|\'\'\')', prompt)
    if not m:
        return prompt
    end = prompt.find(m.group(1), m.end())
    if end == -1:
        return prompt  # unterminated -- leave unchanged rather than guess
    return prompt[:m.start()] + prompt[end + 3:]


def run_control(probs):
    """The known-answer controls, all on the standard prompt composition.

    Positive: every canonical solution must pass (exit if under 90%, same
    threshold as the pod runner). Negative: a constant-False body on
    HumanEval/0 must FAIL -- its tests include a True case, so this is a wrong
    answer that runs clean, not an exception. Empty-completion control: "",
    "\\n" and "    pass\\n" must score wrong on every problem -- the baseline
    arm is 97.6% empty completions, so a judge that credits them would read
    the SFT gain off the scorer, not the model. Plus the strip self-check.
    """
    ok = sum(judge(p, p["canonical_solution"]) for p in probs)
    print(f"CONTROL canonical_solution pass = {ok}/{len(probs)} = "
          f"{100 * ok / len(probs):.1f}%  (must be ~100)", flush=True)
    if ok < len(probs) * 0.9:
        print("JUDGE IS BROKEN -- a model score from this harness would be "
              "meaningless. Stopping.", flush=True)
        sys.exit(1)
    by_id = {p["task_id"]: p for p in probs}
    p0 = by_id["HumanEval/0"]
    if judge(p0, "    return False\n"):
        sys.exit("CONTROL FAILED: constant-False body on HumanEval/0 scored "
                 "correct -- the scorer passes everything, a model zero would "
                 "be indistinguishable from a harness zero")
    print("CONTROL wrong-answer: constant-False on HumanEval/0 -> FAIL "
          "(must FAIL)", flush=True)
    for label, body in (("empty", ""), ("newline", "\n"), ("pass", "    pass\n")):
        n = sum(judge(p, body) for p in probs)
        if n:
            sys.exit(f"CONTROL FAILED: the {label} completion scores correct on "
                     f"{n}/{len(probs)} problems. The baseline arm is 97.6% empty "
                     f"completions, so a scorer that credits them would read the SFT "
                     f"gain off the judge, not the model.")
    print("CONTROL empty-completion: empty/newline/pass score 0/164 each "
          "(must be 0)", flush=True)
    s0 = strip_docstring(p0["prompt"])
    assert s0 != p0["prompt"], "HumanEval/0 has a docstring; strip must change it"
    assert s0.rstrip().endswith(":"), "stripped prompt must end at the def header"
    compile(s0 + "    pass\n", "<sig-only>", "exec")
    n_stripped = sum(strip_docstring(p["prompt"]) != p["prompt"] for p in probs)
    print(f"CONTROL strip: {n_stripped}/{len(probs)} prompts carry a docstring "
          f"the sig-only arm removes", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--max_new", type=int, default=280)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--strip-docstrings", action="store_true",
                    help="sig-only arm: remove the docstring from each prompt")
    ap.add_argument("--control", action="store_true",
                    help="run the known-answer controls only, no model (CPU)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing predictions file (default: refuse)")
    ap.add_argument("--run", default=None,
                    help="name this run so predictions version instead of colliding")
    args = ap.parse_args()

    probs = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    print(f"HumanEval: {len(probs)} problems"
          f"{' (sig-only arm)' if args.strip_docstrings else ''}", flush=True)
    prompts = [strip_docstring(p["prompt"]) if args.strip_docstrings else p["prompt"]
               for p in probs]
    run_control(probs)
    if args.control:
        return
    if not args.ckpt:
        ap.error("--ckpt required (unless --control)")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        sys.exit("REFUSING: CUDA_VISIBLE_DEVICES is unset, so cuda:0 is physical "
                 "GPU 0 -- tileRL's card. Set it to your granted card.")

    # load_checkpoint claims the card (device names cuda) under this file's stem.
    from scripts.loader import load_checkpoint
    from tokenizers import Tokenizer
    model, cfg = load_checkpoint(args.ckpt, device=args.device)
    model.eval()
    tok = Tokenizer.from_file(TOK_PATH)

    def gen(prompt):
        """Verbatim from _humaneval_run.py: greedy, cfg.seq window, eos tid 1,
        stop check every 16 tokens, final truncate in the caller. Returns the
        decoded completion and the stop reason -- the reason is the
        eos_first/stop_at_0 split (prereg format_sft_humaneval_0909 amendment 1:
        77.4% of baseline empties are a STOPS string at position 0, the model
        writing the next top-level def, not eos)."""
        ids = tok.encode(prompt).ids
        x = torch.tensor([ids], device=args.device)
        new = []
        stop_reason = "max_new"
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for step in range(args.max_new):
                lg = model(x[:, -cfg.seq:])[0][:, -1]
                nxt = lg.argmax(-1, keepdim=True)
                tid = nxt.item()
                if tid == 1:
                    stop_reason = "eos"
                    break
                new.append(tid)
                x = torch.cat([x, nxt], 1)
                if step % 16 == 15:
                    s = tok.decode(new)
                    if any(st in s for st in STOPS):
                        stop_reason = "stop"
                        break
        return tok.decode(new), stop_reason

    preds_path = os.path.join(
        ROOT, "data", "eval",
        f"preds_humaneval_{os.path.basename(str(args.ckpt).rstrip('/'))}"
        + (".nodoc" if args.strip_docstrings else "")
        + ".jsonl")
    t0 = time.time()
    npass = nempty = neos = nstop = 0
    with open_artifact(preds_path, force=args.force, run=args.run) as fout:
        out_path = fout.name
        fout.write(json.dumps({
            "_header": 1,
            "ckpt": os.path.basename(str(args.ckpt).rstrip("/")),
            "data": os.path.basename(args.data),
            "strip_docstrings": args.strip_docstrings,
            "max_new": args.max_new,
            "stops": STOPS,
            "n_problems": len(probs),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, ensure_ascii=False) + "\n")
        for i, (p, prompt) in enumerate(zip(probs, prompts), 1):
            raw, stop_reason = gen(prompt)
            c = truncate(raw)
            ok = judge(p, c)
            empty = not c.strip()
            # The two-column split (prereg amendment 1): an empty completion is
            # eos_first (model ended the turn) or stop_at_0 (a STOPS string at
            # position 0 -- the model wrote the next top-level def). Reported as
            # two columns, never the aggregate: the total dropping could be the
            # two swapping. empty_max_new (280 tokens decoding to blank) has
            # never been seen and gets its own bucket rather than a wrong label.
            if not empty:
                empty_kind = "nonempty"
            elif stop_reason == "eos":
                empty_kind = "eos_first"
                neos += 1
            elif stop_reason == "stop":
                empty_kind = "stop_at_0"
                nstop += 1
            else:
                empty_kind = "empty_max_new"
            npass += int(ok)
            nempty += int(empty)
            fout.write(json.dumps(
                {"task_id": p["task_id"], "gen": c, "ok": ok, "empty": empty,
                 "empty_kind": empty_kind},
                ensure_ascii=False) + "\n")
            fout.flush()
            if i % 20 == 0 or i == len(probs):
                print(f"  {i}/{len(probs)}  pass@1 = {npass}/{i} = "
                      f"{100 * npass / i:.2f}%   ({time.time() - t0:.0f}s)", flush=True)

    attest(out_path)
    print(f"\nHUMANEVAL pass@1 (greedy) = {npass}/{len(probs)} = "
          f"{100 * npass / len(probs):.2f}%", flush=True)
    print(f"empty-completion split: eos_first {neos}/{len(probs)}, "
          f"stop_at_0 {nstop}/{len(probs)} (total empty {nempty}/{len(probs)} = "
          f"{100 * nempty / len(probs):.1f}%)", flush=True)
    print(f"preds saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()

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
from contextlib import nullcontext as _nullctx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def truncate(s, entry_point=None):
    """Verbatim from _humaneval_run.py, with one 66-14 exception: a stop that is the
    model RE-DECLARING THE FUNCTION IT WAS ASKED TO COMPLETE ("\\ndef <entry_point>(")
    is not a stop. The prompt ends at the closing docstring, so a completion-style model
    naturally begins "\\ndef <itself>("; cutting there empties a real answer. A second
    def of the same name is valid Python and the later definition wins in prompt+completion.
    Every other stop (a genuinely different top-level def/class/etc.) still cuts at the
    earliest position. entry_point=None preserves the legacy verbatim behaviour."""
    cut = len(s)
    for st in STOPS:
        start = 0
        while True:
            i = s.find(st, start)
            if i == -1:
                break
            if (entry_point is not None and st == "\ndef "
                    and s[i:i + 12 + len(entry_point)].startswith(f"\ndef {entry_point}(")):
                start = i + 1  # self re-declaration -- skip this occurrence, keep scanning
                continue
            cut = min(cut, i)
            break
    return s[:cut]


def hits_stop(s, entry_point):
    """Whether the live (untruncated) decoded continuation contains any STOP that is not
    the function's own re-declaration. Mirrors truncate's matcher so the gen loop and the
    final cut decide identically (66-14)."""
    return truncate(s, entry_point) != s


def repetitive(s, tail_chars=200):
    """True if the last `tail_chars` chars contain a line or word token repeated at least
    3 times consecutively. The degeneration shape seen at step6000: doctest lines and
    '10 10 10 ...'. Lines are compared stripped and blank lines ignored; tokens are
    [A-Za-z_0-9]+ runs so commas/brackets in '[1, 1, 1]' do not create their own run of
    punctuation."""
    tail = s[-tail_chars:]
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    run = 0
    prev = None
    for ln in lines:
        run = run + 1 if ln == prev else 1
        if run >= 3:
            return True
        prev = ln
    toks = re.findall(r"[A-Za-z_0-9]+", tail)
    run = 0
    prev = None
    for t in toks:
        run = run + 1 if t == prev else 1
        if run >= 3:
            return True
        prev = t
    return False


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


def _docstring_span(prompt):
    """(content_start, quote_end_index) of the first triple-quoted block, or None."""
    m = re.search(r'("""|\'\'\')', prompt)
    if not m:
        return None
    end = prompt.find(m.group(1), m.end())
    if end == -1:
        return None
    return m.end(), end


def strip_doctests(prompt):
    """Diagnostic arm (fb 2026-09-12): inside the first docstring remove doctest examples
    and their expected output, keep the prose. An example is a '>>>' line, its '...'
    continuation lines, and the following non-blank lines (the expected output) up to the
    next blank line. Text outside the docstring and prose lines inside it are untouched.
    """
    span = _docstring_span(prompt)
    if span is None:
        return prompt
    cstart, cend = span
    lines = prompt[cstart:cend].splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        body = lines[i].lstrip()
        if body.startswith(">>>"):
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith("..."):
                i += 1
            # Expected output: non-blank lines until the next blank OR the next prompt.
            while (i < len(lines) and lines[i].strip()
                   and not lines[i].lstrip().startswith(">>>")):
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return prompt[:cstart] + "".join(out) + prompt[cend:]


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
    # 66-14: a completion that RE-DECLARES the requested function (the model continuing from
    # the closing docstring by rewriting "\ndef <itself>(") must be judged on its body, not
    # truncated to empty by the "\ndef " stop. Same-name redefinition is valid Python; the later
    # def wins. The control wraps each BODY-form canonical under a copy of its own def header
    # (the exact thing the truncator would otherwise empty). Canonicals that are themselves
    # nested-helper bodies (indented 'def' inside the entry fn) contain no top-level self-def and
    # are excluded -- they exercise the normal path, already covered above.
    def _def_header(prob):
        for line in prob["prompt"].splitlines():
            if re.match(r"def\s+" + re.escape(prob["entry_point"]) + r"\b", line) \
                    and line.rstrip().endswith(":"):
                return line
        return None

    def _body_prologue(prob):
        """Indented import lines the prompt places in the function body before the docstring
        (e.g. HumanEval/115's '    import math'). A self-redeclaration that rewrites the def
        must carry these or the synthetic canonical body is missing a name the prompt bound."""
        out = []
        seen_def = False
        for line in prob["prompt"].splitlines():
            if not seen_def and re.match(r"def\s+" + re.escape(prob["entry_point"]) + r"\b", line):
                seen_def = True
                continue
            if seen_def:
                if re.match(r"\s+(import|from)\s", line):
                    out.append(line)
                elif line.strip().startswith('"""') or line.strip().startswith("'''") or line.strip():
                    break
        return ("\n".join(out) + "\n") if out else ""

    def _redeclares(prob):
        body = prob["canonical_solution"]
        if body.lstrip().startswith("def "):
            return None  # full/nested form -- not a same-name top-level self-redeclaration
        hdr = _def_header(prob)
        return None if hdr is None else "\n" + hdr + "\n" + _body_prologue(prob) + body + "\n"

    redecl_probs = [(p, c) for p in probs if (c := _redeclares(p)) is not None]
    rc_redecl = sum(judge(p, c) for p, c in redecl_probs)
    if redecl_probs and rc_redecl < len(redecl_probs) * 0.95:
        sys.exit(f"CONTROL FAILED: self-redeclared canonical body passes only "
                 f"{rc_redecl}/{len(redecl_probs)} -- the 66-14 re-declaration path is broken.")
    print(f"CONTROL self-redeclare canonical: {rc_redecl}/{len(redecl_probs)} PASS "
          f"(same-name def, body scored not truncated)", flush=True)
    wrong_redecl = judge(p0, "\n" + _def_header(p0) + "\n    return False\n")
    if wrong_redecl:
        sys.exit("CONTROL FAILED: self-redeclared wrong body (constant-False on "
                 "HumanEval/0) scored PASS.")
    print("CONTROL self-redeclare wrong body -> FAIL (must FAIL)", flush=True)
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
    ap.add_argument("--device", default="cuda:0",
                    help="cuda:0 (default) or cpu -- the CPU path runs fp32 with no autocast")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU threads (CPU path only; e.g. 32)")
    ap.add_argument("--strip-docstrings", action="store_true",
                    help="sig-only arm: remove the docstring from each prompt")
    ap.add_argument("--strip-doctests", action="store_true",
                    help="diagnostic arm: remove >>> examples + expected output from each "
                         "docstring, keep the prose")
    ap.add_argument("--first", type=int, default=None,
                    help="score only the first N task_ids (diagnostic arm)")
    ap.add_argument("--control", action="store_true",
                    help="run the known-answer controls only, no model (CPU)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing predictions file (default: refuse)")
    ap.add_argument("--run", default=None,
                    help="name this run so predictions version instead of colliding")
    ap.add_argument("--preds", default=None,
                    help="score an existing preds jsonl (pass/empty/repetition) and exit; "
                         "no model, cardless")
    args = ap.parse_args()

    if args.preds:
        with open(args.preds, encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh if l.strip() and "_header" not in l]
        nonempty = [r for r in rows if not r.get("empty")]
        nrep = sum(bool(repetitive(r["gen"])) for r in nonempty)
        npass = sum(bool(r.get("ok")) for r in rows)
        nempty = len(rows) - len(nonempty)
        rep_frac = f"{100 * nrep / len(nonempty):.1f}%" if nonempty else "n/a (0 non-empty)"
        pass_frac = f"{100 * npass / len(rows):.2f}%" if rows else "n/a (empty preds file)"
        empty_frac = f"{100 * nempty / len(rows):.1f}%" if rows else "n/a"
        print(f"{os.path.basename(args.preds)}: n={len(rows)} pass@1={npass}/{len(rows)} "
              f"= {pass_frac}  nonempty={len(nonempty)} "
              f"repetitive={nrep}/{len(nonempty)} = {rep_frac} of non-empty "
              f"(empty {nempty}/{len(rows)} = {empty_frac})", flush=True)
        return

    probs = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    arm = ("sig-only" if args.strip_docstrings else
           "no-doctest" if args.strip_doctests else "standard")
    print(f"HumanEval: {len(probs)} problems ({arm} arm)"
          f"{f' scoring first {args.first}' if args.first else ''}", flush=True)
    def _prompt(p):
        if args.strip_docstrings:
            return strip_docstring(p["prompt"])
        if args.strip_doctests:
            return strip_doctests(p["prompt"])
        return p["prompt"]
    prompts = [_prompt(p) for p in probs]
    run_control(probs)
    if args.control:
        return
    if args.first:
        probs = probs[:args.first]
        prompts = prompts[:args.first]
    if not args.ckpt:
        ap.error("--ckpt required (unless --control)")
    is_cpu = str(args.device).startswith("cpu")
    if is_cpu:
        if args.threads:
            torch.set_num_threads(args.threads)
        if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            # Cardless is the point of --device cpu; the env must be set EMPTY explicitly, so an
            # unset env can never silently land a "cpu" run on physical GPU 0 on a default change.
            sys.exit("REFUSING: --device cpu but CUDA_VISIBLE_DEVICES is unset -- set it empty "
                     "(CUDA_VISIBLE_DEVICES=) to run cardless.")
    elif not os.environ.get("CUDA_VISIBLE_DEVICES"):
        sys.exit("REFUSING: CUDA_VISIBLE_DEVICES is unset, so cuda:0 is physical "
                 "GPU 0 -- tileRL's card. Set it to your granted card, or pass --device cpu "
                 "with CUDA_VISIBLE_DEVICES= to run cardless.")

    # load_checkpoint claims the card only when device names cuda; a CPU load claims nothing.
    from scripts.loader import load_checkpoint
    from tokenizers import Tokenizer
    model, cfg = load_checkpoint(args.ckpt, device=args.device)
    model.eval()
    tok = Tokenizer.from_file(TOK_PATH)

    def gen(prompt, entry_point):
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
        # GPU autocasts bf16; the CPU path runs fp32 with no autocast (autocast device_type
        # "cuda" raises on a CPU-only host). Same argmax loop either way.
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if not is_cpu else _nullctx()
        with torch.no_grad(), ctx:
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
                    if hits_stop(s, p["entry_point"]):
                        stop_reason = "stop"
                        break
        return tok.decode(new), stop_reason

    preds_path = os.path.join(
        ROOT, "data", "eval",
        f"preds_humaneval_{os.path.basename(str(args.ckpt).rstrip('/'))}"
        + (".nodoc" if args.strip_docstrings else "")
        + ".jsonl")
    t0 = time.time()
    npass = nempty = neos = nstop = nrep = 0
    with open_artifact(preds_path, force=args.force, run=args.run) as fout:
        out_path = fout.name
        fout.write(json.dumps({
            "_header": 1,
            "ckpt": os.path.basename(str(args.ckpt).rstrip("/")),
            "data": os.path.basename(args.data),
            "strip_docstrings": args.strip_docstrings,
            "strip_doctests": args.strip_doctests,
            "first_n": args.first,
            "device": str(args.device),
            "cpu_threads": (args.threads if is_cpu else None),
            "max_new": args.max_new,
            "stops": STOPS,
            "n_problems": len(probs),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, ensure_ascii=False) + "\n")
        for i, (p, prompt) in enumerate(zip(probs, prompts), 1):
            raw, stop_reason = gen(prompt, p["entry_point"])
            c = truncate(raw, p["entry_point"])
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
            nrep += int(not empty and repetitive(c))
            fout.write(json.dumps(
                {"task_id": p["task_id"], "gen": c, "ok": ok, "empty": empty,
                 "empty_kind": empty_kind},
                ensure_ascii=False) + "\n")
            fout.flush()
            if i % 20 == 0 or i == len(probs):
                print(f"  {i}/{len(probs)}  pass@1 = {npass}/{i} = "
                      f"{100 * npass / i:.2f}%   ({time.time() - t0:.0f}s)", flush=True)

    attest(out_path)
    n_nonempty = len(probs) - nempty
    print(f"\nHUMANEVAL pass@1 (greedy) = {npass}/{len(probs)} = "
          f"{100 * npass / len(probs):.2f}%", flush=True)
    print(f"empty-completion split: eos_first {neos}/{len(probs)}, "
          f"stop_at_0 {nstop}/{len(probs)} (total empty {nempty}/{len(probs)} = "
          f"{100 * nempty / len(probs):.1f}%)", flush=True)
    rep_frac = f"{100 * nrep / n_nonempty:.1f}%" if n_nonempty else "n/a (0 non-empty)"
    print(f"repetitive non-empty (last 200 chars, >=3 consecutive equal lines or tokens): "
          f"{nrep}/{n_nonempty} = {rep_frac} of non-empty", flush=True)
    print(f"preds saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()

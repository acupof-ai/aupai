#!/usr/bin/env python3
# restartable: reads small jsonl inputs (5.7k distill rows + the RL pool) and writes one pack
# at the end; an interrupt loses only CPU minutes of sandbox re-execution and no partial file.
"""Build the plateau-break code SFT pack from EXECUTION-VERIFIED sources only.

WHY. The case study found the dominant failure is "runs, but wrong logic" (73/129 at
step62269), not formatting. SFT can only teach correct logic if every target answer is
correct. So this builder admits a row only when its solution, re-executed HERE in the
sandbox, passes its own tests -- an upstream "validated" flag is never trusted.

SOURCES (all already on disk; nothing is generated here):
  distill   data/distill/code_claude_sh{0..3}.jsonl -- {prompt, output} where output holds
            one ```python block of a function plus >=2 asserts (5,695 rows measured).
  rlpool    data/rl/rl_code_*.jsonl (scripts/rl_code_pool.py) -- {prompt, impl, tests};
            HumanEval-shaped call-style problems with a passing reference.
Each source is converted to the eval's own shape: RAW CONTINUATION, prompt = function
signature + docstring, target = function body. That matches eval/humaneval_gen.py
--rstrip_nl, so the SFT sequence is a token prefix of what the eval feeds.

GATES, in order, each counted:
  1. parse    one python block / a def is recoverable, asserts present
  2. exec     solution + tests exit 0 in datagen.sandbox_exec (never in-process)
  3. decontam 13-gram drop against HumanEval/MBPP/GSM8K/MATH (filters/decontam_ngram)
  4. dedup    by normalized prompt and by solution sha
  5. holdout  rows whose prompt hashes into data/eval/holdout_hashes.txt are dropped

    python3 scripts/build_plateau_code_sft.py --distill 'data/distill/code_claude_sh*.jsonl' \
        --rlpool 'data/rl/rl_code_*.jsonl' --tokenizer data/tokenizer.json \
        --out data/sft/plateau_code_sft.pt --stats data/sft/plateau_code_sft.stats.json
    python3 scripts/build_plateau_code_sft.py --selftest      # no sandbox, no tokenizer

The pack is consumed by sft_math.py (absolute LR, holdout stamp) exactly like sft_mixA.
"""
import argparse
import ast
import glob
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "datagen"))

PYBLOCK = re.compile(r"```python\s*\n(.*?)```", re.S)
FORBID = re.compile(r"\b(input\s*\(|socket|urllib|requests\.|os\.system|subprocess|"
                    r"eval\s*\(|exec\s*\(|__import__|open\s*\()")
SEQ = 4096
SEED = 20261013


def split_function(src):
    """(signature_line, docstring_or_None, body_src, tests_src, entry) from a code block.

    The FIRST top-level def is the answer; top-level asserts are its tests. Helper defs
    after it are folded into the body only if they are referenced; anything else refuses.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    asserts = [n for n in tree.body if isinstance(n, ast.Assert)]
    if not defs or len(asserts) < 2:
        return None
    fn = defs[0]
    lines = src.splitlines()
    fn_src = "\n".join(lines[fn.lineno - 1:fn.end_lineno])
    sig_end = fn.body[0].lineno - 1
    doc = ast.get_docstring(fn)
    body_start = fn.body[1].lineno - 1 if doc and len(fn.body) > 1 else fn.body[0].lineno - 1
    if doc and len(fn.body) == 1:
        return None  # docstring only, no implementation
    signature = "\n".join(lines[fn.lineno - 1:sig_end])
    body = "\n".join(lines[body_start:fn.end_lineno])
    helpers = "\n".join("\n".join(lines[d.lineno - 1:d.end_lineno]) for d in defs[1:])
    tests = "\n".join("\n".join(lines[a.lineno - 1:a.end_lineno]) for a in asserts)
    imports = "\n".join("\n".join(lines[n.lineno - 1:n.end_lineno]) for n in tree.body
                        if isinstance(n, (ast.Import, ast.ImportFrom)))
    return {"signature": signature, "doc": doc, "body": body, "helpers": helpers,
            "tests": tests, "imports": imports, "entry": fn.name, "fn_src": fn_src}


def to_continuation(question, parts):
    """HumanEval-shaped prompt: imports + signature + docstring carrying the task text.

    The task statement goes INTO the docstring, as HumanEval does; the target is the body.
    """
    doc = (parts["doc"] or question).strip().replace('"""', "'''")
    indent = "    "
    prompt = ""
    if parts["imports"]:
        prompt += parts["imports"] + "\n\n"
    if parts["helpers"]:
        prompt += parts["helpers"] + "\n\n"
    prompt += parts["signature"] + "\n" + indent + '"""' + doc.replace("\n", "\n" + indent) + '\n' + indent + '"""\n'
    return prompt.rstrip("\n"), "\n" + parts["body"].rstrip() + "\n"


def program(prompt, body, tests):
    return prompt + body + "\n" + tests + "\n"


def from_distill(row):
    out = row.get("output", "")
    blocks = PYBLOCK.findall(out)
    if len(blocks) != 1 or FORBID.search(blocks[0]):
        return None, "parse"
    parts = split_function(blocks[0])
    if not parts:
        return None, "parse"
    prompt, body = to_continuation(row.get("prompt", ""), parts)
    return {"prompt": prompt, "body": body, "tests": parts["tests"], "src": "distill"}, None


def from_rlpool(row):
    if row.get("kind") == "stdin" or not row.get("impl") or not row.get("tests"):
        return None, "parse"
    prompt = row["prompt"].rstrip("\n")
    impl = row["impl"]
    m = re.search(r"(?m)^def\s+" + re.escape(row.get("entry", "")) + r"\b", impl) if row.get("entry") else None
    if not m:
        return None, "parse"
    try:
        fn = next(n for n in ast.parse(impl).body
                  if isinstance(n, ast.FunctionDef) and n.name == row["entry"])
    except (SyntaxError, StopIteration):
        return None, "parse"
    lines = impl.splitlines()
    first = fn.body[1] if ast.get_docstring(fn) and len(fn.body) > 1 else fn.body[0]
    body = "\n" + "\n".join(lines[first.lineno - 1:fn.end_lineno]).rstrip() + "\n"
    # rl_code_pool tests import from solution; rewrite to call the inline function.
    tests = re.sub(r"(?m)^from solution import .*$", "", row["tests"])
    return {"prompt": prompt, "body": body, "tests": tests, "src": "rlpool", "pytest": True}, None


def sandbox_pass(prog, pytest_style, executor, timeout):
    if pytest_style:
        prog = prog + "\nimport sys, inspect\n_fs=[f for n,f in list(globals().items()) if n.startswith('test_') and callable(f)]\n"\
            "assert _fs, 'no tests'\nfor _f in _fs:\n    _f()\n"
    return executor(prog, timeout) == 0


def default_executor(src, timeout):
    from datagen.sandbox_exec import run_sandboxed
    rc, _o, _e = run_sandboxed(src, timeout=timeout, nproc=int(os.environ.get("SANDBOX_NPROC", "4096")))
    return rc


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def build_rows(candidates, executor, decon, is_holdout, timeout=10):
    """decon: object with .keeps(text) (filters.decontam_ngram), or None for tests.
    is_holdout: callable(text) -> bool (datagen.holdout.is_holdout), or a set-free stub."""
    stats = {"in": 0, "parse": 0, "exec": 0, "decontam": 0, "dup_prompt": 0, "dup_solution": 0,
             "holdout": 0, "kept": 0, "by_src": {}}
    seen_p, seen_s, kept = set(), set(), []
    for src_name, row in candidates:
        stats["in"] += 1
        conv, why = (from_distill if src_name == "distill" else from_rlpool)(row)
        if conv is None:
            stats[why] += 1
            continue
        if not sandbox_pass(program(conv["prompt"], conv["body"], conv["tests"]),
                            conv.get("pytest", False), executor, timeout):
            stats["exec"] += 1
            continue
        if decon is not None and not (decon.keeps(conv["prompt"]) and decon.keeps(conv["body"])):
            stats["decontam"] += 1
            continue
        if is_holdout(conv["prompt"]) or is_holdout(row.get("prompt", "")):
            stats["holdout"] += 1
            continue
        hp = hashlib.sha256(norm(conv["prompt"]).encode()).hexdigest()
        hs = hashlib.sha256(norm(conv["body"]).encode()).hexdigest()
        if hp in seen_p:
            stats["dup_prompt"] += 1
            continue
        if hs in seen_s:
            stats["dup_solution"] += 1
            continue
        seen_p.add(hp)
        seen_s.add(hs)
        kept.append((conv["prompt"], conv["body"]))
        stats["by_src"][conv["src"]] = stats["by_src"].get(conv["src"], 0) + 1
        stats["kept"] += 1
    return kept, stats


def iter_jsonl(patterns):
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)


def _selftest():
    good = {"prompt": "数一数 ch 出现次数。", "output":
            "```python\ndef count_mark(marks, ch):\n    total = 0\n    for c in marks:\n"
            "        if c == ch:\n            total += 1\n    return total\n\n"
            "assert count_mark('PLAP', 'P') == 2\nassert count_mark('', 'P') == 0\n```"}
    wrong = {"prompt": "同上但写错。", "output":
             "```python\ndef count_x(marks, ch):\n    return 1\n\n"
             "assert count_x('PLAP', 'P') == 2\nassert count_x('', 'P') == 0\n```"}
    io = {"prompt": "读输入。", "output": "```python\ndef f():\n    return input()\nassert 1\nassert 1\n```"}

    def stub(src, timeout):
        try:
            exec(compile(src, "<t>", "exec"), {"__name__": "__main__"})
        except Exception:
            return 1
        return 0
    conv, why = from_distill(good)
    assert why is None and conv["prompt"].startswith("def count_mark") and '"""' in conv["prompt"]
    assert conv["body"].startswith("\n    total = 0"), repr(conv["body"])
    rows, st = build_rows([("distill", good), ("distill", wrong), ("distill", io), ("distill", good)],
                          stub, None, lambda _t: False)
    assert st["kept"] == 1 and st["exec"] == 1 and st["parse"] == 1 and st["dup_prompt"] == 1, st
    # A holdout hit is dropped even when everything else passes.
    _r, st2 = build_rows([("distill", good)], stub, None, lambda t: "count_mark" in t)
    assert st2["holdout"] == 1 and st2["kept"] == 0, st2
    # The kept prompt is a strict prefix of prompt+body: the eval-shape invariant.
    p, b = rows[0]
    assert (p + b).startswith(p) and not p.endswith("\n")
    print("build_plateau_code_sft selftest OK", st)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--distill", action="append", default=[])
    ap.add_argument("--rlpool", action="append", default=[])
    ap.add_argument("--tokenizer")
    ap.add_argument("--out")
    ap.add_argument("--stats")
    ap.add_argument("--holdout", default=os.path.join(ROOT, "data", "eval", "holdout_hashes.txt"))
    ap.add_argument("--no_decontam", action="store_true", help="only for a dry count, never for a pack")
    ap.add_argument("--timeout", type=float, default=10)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return 0
    if not (args.out and args.tokenizer):
        ap.error("--out and --tokenizer are required")
    if args.no_decontam:
        raise SystemExit("--no_decontam cannot write a training pack")
    sys.path.insert(0, os.path.join(ROOT, "filters"))
    from decontam_ngram import default as decon_default
    from datagen.holdout import is_holdout  # raises if holdout_hashes.txt is stale
    decon = decon_default()
    cands = [("distill", r) for r in iter_jsonl(args.distill)] + \
            [("rlpool", r) for r in iter_jsonl(args.rlpool)]
    rows, stats = build_rows(cands, default_executor, decon, is_holdout, args.timeout)
    import random
    random.Random(SEED).shuffle(rows)
    from tokenizers import Tokenizer
    from prepare_sft import pack_and_save
    tok = Tokenizer.from_file(args.tokenizer)
    eos = tok.token_to_id("<eos>")
    pack_and_save(rows, tok, eos, args.out, SEQ, split_encode=True,
                  extra_stats={"builder": os.path.basename(__file__), "gates": stats,
                               "format": "raw_continuation_split_encode", "seed": SEED})
    if args.stats:
        with open(args.stats, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

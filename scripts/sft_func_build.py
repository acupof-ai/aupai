#!/usr/bin/env python3
# restartable: verification results append to <out>/verified.jsonl keyed by qid and a rerun skips
# qids already there; the pack step rebuilds from that file deterministically.
"""SFT v2: function-completion examples in the HumanEval shape, sandbox-verified (1e 2026-09-28).

Why: SFT v1 (stdin/stdout contest solutions + math CoT) moved HumanEval 36 -> 35/33 on the 0926
base; its code rows never showed the model a "signature + docstring -> body" task. TACO/APPS carry
call-based problems (input_output.fn_name, a top-level `def` starter) that are exactly that shape.

Per problem (TACO + APPS, deduplicated by question text):
  * keep it only when the starter is one top-level `def <fn_name>(...)`;
  * keep a reference solution only when its top level is imports + that one function, and it
    returns the bundled outputs for every bundled case (up to --max-cases) inside algorithms.isolate;
  * emit prompt = "def <sig>:\\n    \\"\\"\\"<question>\\"\\"\\"\\n", output = the function body
    (imports moved to the top of the body), which is what humaneval_gen feeds and scores.
Then decontaminate prompts at 13-gram against HumanEval/MBPP, carve a problem-disjoint RL held-out,
and pack (prompt, output) with prepare_sft.pack_and_save.

    python3 scripts/sft_func_build.py --selftest
    taskset -c 146-179 nice -n 10 python3 scripts/sft_func_build.py verify --workers 32
    python3 scripts/sft_func_build.py pack --heldout 400 --repeat 3
"""
import argparse
import ast
import glob
import hashlib
import json
import math
import os
import random
import re
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "data", "sft", "sft_func_v2")
DOC_MAX = 1500

HARNESS = r'''
import json as _j
_cases = _j.loads(%r)
def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
        except Exception:
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b
for _args, _want in _cases:
    _got = %s(*_args)
    if not (_same(_got, _want) or (isinstance(_want, list) and len(_want) == 1 and _same(_got, _want[0]))):
        print("MISMATCH"); raise SystemExit(1)
print("ALLPASS")
'''


def qid_of(question):
    return hashlib.sha256(" ".join(question.split()).lower().encode()).hexdigest()[:16]


def starter_sig(starter, fn):
    m = re.match(r"\s*def\s+(" + re.escape(fn) + r")\s*\((.*?)\)\s*(->[^:]*)?:", starter, re.S)
    if not m:
        return None
    return f"{m.group(1)}({' '.join(m.group(2).split())}){(m.group(3) or '').rstrip()}"


def split_solution(code, fn):
    """(imports, body) when the top level is imports + exactly the one target function, else None."""
    code = code.replace("\t", "    ")
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    imports, target = [], None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.get_source_segment(code, node))
        elif isinstance(node, ast.FunctionDef) and node.name == fn and target is None:
            target = node
        else:
            return None
    if target is None or not target.body:
        return None
    lines = code.splitlines()
    first = target.body[0]
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
            and isinstance(first.value.value, str) and len(target.body) > 1:
        first = target.body[1]  # drop the solution's own docstring; the prompt carries the spec
    body = textwrap.dedent("\n".join(lines[first.lineno - 1:target.end_lineno]))
    return imports, body


def render(sig, question, imports, body):
    doc = question.strip().replace('"""', "'''").replace("\\", "\\\\")
    if len(doc) > DOC_MAX:
        return None
    doc = textwrap.indent(doc, "    ").lstrip()
    prompt = f'def {sig}:\n    """{doc}\n    """\n'
    out_lines = [f"import {i[len('import '):]}" if i.startswith("import ") else i for i in imports] + [body]
    output = textwrap.indent("\n".join(out_lines), "    ") + "\n"
    return prompt, output


def candidates():
    """Yield dicts {qid, question, sig, fn, cases, sols, source, difficulty} from TACO then APPS."""
    import pyarrow.parquet as pq
    seen = set()

    def one(question, io_raw, starter, sols_raw, source, difficulty):
        try:
            io = json.loads(io_raw or "{}")
            sols = json.loads(sols_raw or "[]")
        except (ValueError, TypeError):
            return None
        fn = io.get("fn_name")
        ins, outs = io.get("inputs"), io.get("outputs")
        if not fn or not isinstance(ins, list) or not isinstance(outs, list) or not ins or len(ins) != len(outs):
            return None
        sig = starter_sig(starter or "", fn)
        q = qid_of(question)
        if sig is None or q in seen:
            return None
        seen.add(q)
        return {"qid": q, "question": question, "sig": sig, "fn": fn, "cases": list(zip(ins, outs)),
                "sols": [s for s in sols if isinstance(s, str)], "source": source, "difficulty": difficulty}

    for f in sorted(glob.glob(os.path.join(ROOT, "data/sft_raw/taco/train-*.parquet"))):
        cols = ["question", "solutions", "input_output", "starter_code", "source", "difficulty"]
        for r in pq.read_table(f, columns=cols).to_pylist():
            c = one(r["question"], r["input_output"], r["starter_code"], r["solutions"], r["source"], r["difficulty"])
            if c:
                yield c
    apps = os.path.join(ROOT, "data/sft_raw/apps/train.jsonl")
    if os.path.exists(apps):
        for line in open(apps, encoding="utf-8"):
            d = json.loads(line)
            c = one(d["question"], d.get("input_output"), d.get("starter_code"), d.get("solutions"),
                    "apps", d.get("difficulty"))
            if c:
                yield c


def verify_one(c, max_cases, max_solutions, keep):
    from scripts.sft_verify_code import _run_one
    kept, seen = [], set()
    cases = json.dumps(c["cases"][:max_cases])
    for sol in c["sols"][:max_solutions]:
        parts = split_solution(sol, c["fn"])
        if parts is None:
            continue
        imports, body = parts
        norm = " ".join(body.split())
        if norm in seen:
            continue
        r = _run_one(sol.replace("\t", "    ") + "\n" + HARNESS % (cases, c["fn"]), "")
        if r.get("rc") == 0 and not r.get("timed_out") and "ALLPASS" in (r.get("stdout") or ""):
            rendered = render(c["sig"], c["question"], imports, body)
            if rendered:
                seen.add(norm)
                kept.append(rendered)
        if len(kept) >= keep:
            break
    return [{"qid": c["qid"], "prompt": p, "output": o, "source": c["source"], "difficulty": c["difficulty"]}
            for p, o in kept]


def cmd_verify(a):
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "verified.jsonl")
    done = {json.loads(line)["qid"] for line in open(path)} if os.path.exists(path) else set()
    tried_path = os.path.join(a.out, "tried.txt")
    tried = set(open(tried_path).read().split()) if os.path.exists(tried_path) else set()
    todo = [c for c in candidates() if c["qid"] not in done and c["qid"] not in tried]
    print(f"candidates to verify: {len(todo)} (already verified {len(done)}, tried {len(tried)})", flush=True)
    n_ok = 0
    with open(path, "a") as out, open(tried_path, "a") as tr, ThreadPoolExecutor(a.workers) as ex:
        for i, (c, rows) in enumerate(zip(todo, ex.map(lambda c: verify_one(c, a.max_cases, a.max_solutions, a.keep), todo))):
            for r in rows:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_ok += bool(rows)
            tr.write(c["qid"] + "\n")
            if i % 100 == 99:
                out.flush(); tr.flush()
                print(f"{i + 1}/{len(todo)} problems, {n_ok} with a verified solution", flush=True)
    print(f"done: {n_ok}/{len(todo)} problems verified", flush=True)


def cmd_pack(a):
    from scripts.sft_reason_pack import decontaminate, load_decontaminator
    rows = [json.loads(line) for line in open(os.path.join(a.out, "verified.jsonl"))]
    dec = load_decontaminator()
    for r in rows:  # the problem text AND the solution: a canonical HumanEval body is a hit too
        r["_dc"] = r["prompt"] + "\n" + r["output"]
    kept, dropped, hits = decontaminate(rows, "_dc", dec)
    for r in rows:
        r.pop("_dc", None)
    qids = sorted({r["qid"] for r in kept})
    random.Random(a.seed).shuffle(qids)
    held = set(qids[:a.heldout])
    train = [r for r in kept if r["qid"] not in held]
    with open(os.path.join(a.out, "heldout_func.jsonl"), "w") as fh:
        for r in kept:
            if r["qid"] in held:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    examples = [(r["prompt"], r["output"]) for r in train] * a.repeat
    codeif = os.path.join(ROOT, "data/sft/sft_reason_v1/code_if_short.jsonl")
    n_if = 0
    if a.codeif_cap and os.path.exists(codeif):
        for line in open(codeif, encoding="utf-8"):
            d = json.loads(line)
            examples.append((d["prompt"], d["output"]))
            n_if += 1
            if n_if >= a.codeif_cap:
                break
    # humaneval_gen scores the rstrip-nl arm: the prompt it feeds ends at the closing quotes and the
    # model generates the newline itself. Train on that same boundary.
    examples = [(p[:-1], "\n" + o) if p.endswith("\n") else (p, o) for p, o in examples]
    random.Random(a.seed).shuffle(examples)
    stats = {"verified_rows": len(rows), "decontam_dropped": len(dropped), "hits": hits,
             "problems": len(qids), "heldout_problems": len(held), "train_rows_unique": len(train),
             "repeat": a.repeat, "code_if_rows": n_if, "examples": len(examples)}
    print(json.dumps(stats), flush=True)
    json.dump(stats, open(os.path.join(a.out, "pack_stats.json"), "w"), indent=1)
    if a.dry:
        return
    from tokenizers import Tokenizer

    from datagen.prepare_sft import pack_and_save
    tok = Tokenizer.from_file(a.tokenizer)
    eos = tok.token_to_id("<eos>")
    assert eos is not None, "tokenizer has no <eos>"
    pack_and_save(examples, tok, eos, a.pack, a.seq, split_encode=True, extra_stats=stats)


def _selftest():
    sol = "import math\ndef area(r):\n    '''own doc'''\n    return math.pi * r * r\n"
    imports, body = split_solution(sol, "area")
    assert imports == ["import math"] and body.strip() == "return math.pi * r * r", (imports, body)
    assert split_solution("def area(r):\n    return 1\nprint(area(2))\n", "area") is None
    assert split_solution("def helper():\n    pass\ndef area(r):\n    return 1\n", "area") is None
    assert starter_sig("def is_anagram(test, original):\n\t", "is_anagram") == "is_anagram(test, original)"
    p, o = render("area(r)", 'Area of a "circle".', imports, body)
    src = p + o + "\nassert abs(area(1) - 3.14159) < 1e-4\n"
    exec(compile(src, "<selftest>", "exec"), {})
    ns = {}
    cases = json.dumps([[[2], [4]], [[3], 9]])
    exec("def sq(x):\n    return x * x\n" + HARNESS % (cases, "sq"), ns)
    bad = "def sq(x):\n    return x + x\n" + HARNESS % (json.dumps([[[3], [9]]]), "sq")
    try:
        exec(bad, {})
        raise AssertionError("harness accepted a wrong answer")
    except SystemExit as e:
        assert e.code == 1
    assert math.isfinite(1.0)
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["verify", "pack"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-cases", type=int, default=8)
    ap.add_argument("--max-solutions", type=int, default=8)
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--heldout", type=int, default=400)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--codeif-cap", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--pack", default=os.path.join(OUT, "sft_func_v2.pt"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    {"verify": cmd_verify, "pack": cmd_pack}[a.cmd](a)


if __name__ == "__main__":
    main()

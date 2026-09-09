#!/usr/bin/env python3
"""Extract function-level (signature+docstring -> body) pairs from code_py_starcoder.

Format SFT for the function-interface defect measured 2026-09-09: on the 30B
checkpoint P(<eos>) right after a docstring is 0.3131 while in-body it is 0.0000,
HumanEval 0-shot is 0.00% with the scorer verified on 164/164 canonical solutions,
and Code-500 3-shot is 14.0%. The model writes correct bodies when the body is
started for it; the interface is what is broken.

Each train pair's output is the COMPLETE body: ast's end_lineno sets the cut, so
the sample teaches both ends -- the body starts after the docstring, and the turn
terminator (packed as <|im_end|> by prepare_sft) comes after the body's last line.

Contamination: every sampled pair is scanned against HumanEval-164
(data/eval/humaneval/humaneval_164.jsonl) by whitespace-normalized containment in
both directions on prompt and canonical solution; hits are dropped and the rate is
printed for the exp row.

Negative control: signature-only pairs (no docstring) go to a separate file and
never enter the pack. If SFT fixes docstring prompts but not signature-only
prompts, the lesson was the docstring signal; both fixing means the model learned
where the body starts.

Shard shape: one json per line, {"content": <whole python file>, ...}.
"""

# restartable: a full run is ~2-5 minutes on 48 workers over immutable shards, seeded
# (seed 42, per-shard rng), so an interrupt costs one deterministic re-run; the packed
# .pt is downstream and re-derived by prepare_sft_code_if.py.

import argparse
import ast
import glob
import json
import multiprocessing as mp
import os
import random
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
DATA = os.path.join(ROOT, "data")
TOK_PATH = os.path.join(DATA, "tokenizer.json")
HUMANEVAL = os.path.join(DATA, "eval", "humaneval", "humaneval_164.jsonl")

MIN_BODY_LINES = 3
MIN_DOCSTRING_CHARS = 10
MAX_TOKENS = 1024
TRAIN_TARGET = 100_000
CONTROL_TARGET = 10_000
PER_SHARD_CAP = 400  # 283 shards x 400 = 113k >= TRAIN_TARGET, uniform over shards
CONTROL_SHARD_CAP = 60  # 283 x 60 = 17k >= CONTROL_TARGET

_WS = re.compile(r"\s+")


def norm(s):
    return _WS.sub(" ", s).strip()


def _is_docstring(stmt):
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def extract_pairs(content, tok, max_tokens):
    """Yield (kind, prompt, output) for top-level functions in one file.

    kind is 'train' (docstring) or 'control' (signature only). The cut is
    ast's end_lineno, so output always runs to the function's last source line.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return
    lines = content.splitlines(keepends=True)
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.decorator_list or not node.body:
            continue
        first = node.body[0]
        has_ds = _is_docstring(first)
        body_start = first.end_lineno if has_ds else first.lineno - 1
        prompt = "".join(lines[node.lineno - 1 : body_start])
        output = "".join(lines[body_start : node.end_lineno])
        body_n = sum(1 for l in output.splitlines() if l.strip())
        if body_n < MIN_BODY_LINES:
            continue
        if has_ds and len(first.value.value.strip()) < MIN_DOCSTRING_CHARS:
            continue
        # the pair is a source slice of a file that parsed, but re-parse guards the cut
        try:
            ast.parse(prompt + output)
        except SyntaxError:
            continue
        if len(tok.encode(prompt + output, add_special_tokens=False).ids) > max_tokens:
            continue
        yield ("train" if has_ds else "control", prompt, output)


def _shard_worker(args):
    shard, cap, ctrl_cap, seed, max_tokens = args
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOK_PATH)
    rng = random.Random(seed)
    train, control = [], []
    with open(shard, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                content = json.loads(line)["content"]
            except (json.JSONDecodeError, KeyError):
                continue
            for kind, prompt, output in extract_pairs(content, tok, max_tokens):
                (train if kind == "train" else control).append((prompt, output))
    rng.shuffle(train)
    rng.shuffle(control)
    return train[:cap], control[:ctrl_cap], len(train), len(control)


def load_humaneval(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append((norm(d["prompt"]), norm(d["canonical_solution"])))
    return out


def contaminated(prompt, output, he):
    """Whitespace-normalized containment both ways, prompt and solution.

    Exact match is the known HumanEval contamination shape (a vendored solution);
    containment both ways with a length floor catches prompt fragments either side.
    """
    np_, no = norm(prompt), norm(output)
    for hp, hs in he:
        if len(np_) >= 20 and (hp in np_ or np_ in hp):
            return "prompt"
        if len(no) >= 20 and (hs in no or no in hs):
            return "solution"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(DATA, "corpus", "code_py_starcoder"))
    ap.add_argument("--humaneval", default=HUMANEVAL)
    ap.add_argument("--out-train", default=os.path.join(DATA, "sft", "code_if_pairs_train.jsonl"))
    ap.add_argument("--out-control", default=os.path.join(DATA, "sft", "code_if_pairs_control.jsonl"))
    ap.add_argument("--stats", default=os.path.join(ROOT, "runs", "code_if_pairs_stats.json"))
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    shards = sorted(glob.glob(os.path.join(args.root, "*.jsonl")))
    assert shards, f"no shards under {args.root}"
    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    he = load_humaneval(args.humaneval)
    print(f"{len(shards)} shards, {len(he)} HumanEval problems", flush=True)

    work = [(s, PER_SHARD_CAP, CONTROL_SHARD_CAP, args.seed + i, MAX_TOKENS) for i, s in enumerate(shards)]
    train, control = [], []
    n_cand = n_ctrl_cand = 0
    with mp.Pool(args.workers) as pool:
        for i, (t, c, tc, cc) in enumerate(pool.imap_unordered(_shard_worker, work)):
            train.extend(t)
            control.extend(c)
            n_cand += tc
            n_ctrl_cand += cc
            if (i + 1) % 32 == 0:
                print(f"  {i + 1}/{len(shards)} shards, sampled {len(train)}+{len(control)}", flush=True)
    print(
        f"candidates: {n_cand} train, {n_ctrl_cand} control; sampled {len(train)}+{len(control)}", flush=True
    )

    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(control)
    train = train[:TRAIN_TARGET]
    control = control[:CONTROL_TARGET]

    hits = {"prompt": 0, "solution": 0}
    kept = []
    for prompt, output in train:
        kind = contaminated(prompt, output, he)
        if kind:
            hits[kind] += 1
        else:
            kept.append((prompt, output))
    ctrl_hits = 0
    kept_ctrl = []
    for prompt, output in control:
        if contaminated(prompt, output, he):
            ctrl_hits += 1
        else:
            kept_ctrl.append((prompt, output))

    with open(args.out_train, "w", encoding="utf-8") as f:
        for prompt, output in kept:
            f.write(json.dumps({"prompt": prompt, "output": output}, ensure_ascii=False) + "\n")
    with open(args.out_control, "w", encoding="utf-8") as f:
        for prompt, output in kept_ctrl:
            f.write(json.dumps({"prompt": prompt, "output": output}, ensure_ascii=False) + "\n")

    stats = {
        "shards": len(shards),
        "candidates_train": n_cand,
        "candidates_control": n_ctrl_cand,
        "sampled_train": len(train),
        "sampled_control": len(control),
        "contamination_hits_train": hits,
        "contamination_rate_train": sum(hits.values()) / max(1, len(train)),
        "contamination_hits_control": ctrl_hits,
        "contamination_rate_control": ctrl_hits / max(1, len(control)),
        "written_train": len(kept),
        "written_control": len(kept_ctrl),
        "humaneval_problems": len(he),
        "config": {
            "min_body_lines": MIN_BODY_LINES,
            "min_docstring_chars": MIN_DOCSTRING_CHARS,
            "max_tokens": MAX_TOKENS,
            "per_shard_cap": PER_SHARD_CAP,
            "seed": args.seed,
        },
    }
    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1), flush=True)


def _selftest():
    """Known-answer world: one good pair, one HumanEval copy, one control, two rejects."""
    he = [(norm("def f(x):\n    '''identity function'''\n"), norm("    return x\n"))]
    good = "def add(a, b):\n    '''add two ints\n    >>> add(1, 2)\n    3\n    '''\n    s = a + b\n    t = s * 2\n    return t\n"
    vendored = "def f(x):\n    '''identity function'''\n    y = x\n    z = y\n    return z\n"
    no_ds = "def sig_only(x):\n    y = x + 1\n    z = y * 2\n    return z\n"
    decorated = "@cache\ndef g(x):\n    '''decorated'''\n    return x\n"
    short = "def short(x):\n    '''short body'''\n    return x\n"
    content = "\n".join([good, vendored, no_ds, decorated, short]) + "\n"

    class FakeTok:
        def encode(self, text, add_special_tokens=False):
            class Ids:
                ids = list(range(len(text) // 4 + 1))

            return Ids()

    pairs = list(extract_pairs(content, FakeTok(), 10**9))
    kinds = [(k, p.splitlines()[0]) for k, p, _ in pairs]
    assert ("train", "def add(a, b):") in kinds, kinds
    assert ("control", "def sig_only(x):") in kinds, kinds
    assert not any("g(x)" in line for _, line in kinds), "decorated function leaked"
    assert not any("short(x)" in line for _, line in kinds), "2-line body leaked"
    # the good pair's output runs to the function's last line (acceptance 1)
    out = next(o for k, p, o in pairs if k == "train" and "def add" in p)
    assert out.rstrip().endswith("return t"), repr(out)
    # the vendored HumanEval copy is caught by the contamination scan
    vp, vo = next((p, o) for k, p, o in pairs if k == "train" and "def f(x)" in p)
    assert contaminated(vp, vo, he) == "prompt", "HumanEval prompt copy not caught"
    assert (
        contaminated("def unrelated(z):\n    '''doc'''\n    return z + 1\n", "    return z + 1\n", he) is None
    )
    print("extract_code_if_pairs selftest OK (3 kept, 2 rejected, 1 contamination hit)")


if __name__ == "__main__":
    main()

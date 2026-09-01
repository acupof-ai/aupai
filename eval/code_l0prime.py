#!/usr/bin/env python3
# restartable: single-shot eval, ~minutes on pod (500 problems x executed
# mutants); an interrupt costs one rerun, no shard state to lose.
"""L0' code discriminator (reasoning_panel.md §6): per-problem win rate of the
reference solution's per-token NLL against an executed-failing single-operator
mutant. The math_v2_like of code: labels come from execution, not a judge.

Easy perturbation layer only. The hard layer (model-generated failing solutions,
frozen from one checkpoint) is deferred — the easy layer first answers whether
the instrument reads at 200M at all; if it saturates there, the hard layer is
what buys headroom.

Prompt (pinned, same target shape as code_fewshot): 题目：{q}\n```python\n
Both solutions scored as the continuation span, per-token MEAN NLL (not sum:
sum is length-biased and a one-char operator swap can change token count).
Win = ref mean NLL < mutant mean NLL. Floor 50%.

Mutants: tokenize-level OP swaps, parse-checked, sandbox-executed; kept only if
they FAIL the problem's own test (wrong output / crash / timeout). First failing
mutant in (position, operator) order — deterministic, no model anywhere in the
pipeline, so the mutant set is frozen by construction and reusable across
checkpoints (§6 freeze discipline).

Known-answer (instrument validation): a working instrument prefers the
reference on >90% of problems — same code, one operator apart. Below that the
instrument is broken, not the model weak.

Usage:
    python eval/code_l0prime.py --selftest
    python eval/code_l0prime.py --ckpt <ckpt> --out runs/l0prime_<tag>.json
"""

import argparse
import io
import json
import os
import sys
import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from datagen.sandbox_exec import run_sandboxed  # noqa: E402
from scripts.loader import load_checkpoint, load_tokenizer  # noqa: E402

TEST_PATH = os.path.join(ROOT, "data", "eval", "code_holdout_500.jsonl")

# Operator swaps. Token-level (tokenize.OP), so strings/comments are immune by
# construction. '<' and '>' are their own tokens only outside '<='/'>='.
MUTATIONS = {
    "+": "-", "-": "+",
    "*": "/", "/": "*",
    "<": ">", ">": "<",
    "<=": ">=", ">=": "<=",
    "==": "!=", "!=": "==",
}
MAX_MUTANTS_PER_PROBLEM = 12
EXEC_TIMEOUT = 5


def _norm_lines(s):
    return [ln.rstrip() for ln in s.split("\n") if ln.strip() != ""]


def passes(code, expected_output):
    rc, out, _ = run_sandboxed(code, timeout=EXEC_TIMEOUT)
    return rc == 0 and _norm_lines(out) == _norm_lines(expected_output)


def mutated_sources(src):
    """Yield mutated sources in (position, operator) order, parse-checked."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:
        return
    for i, tok in enumerate(toks):
        new_op = MUTATIONS.get(tok.string)
        if new_op is None or tok.type != tokenize.OP:
            continue
        lines = src.splitlines(keepends=True)
        row, col = tok.start
        line = lines[row - 1]
        lines[row - 1] = line[:col] + new_op + line[col + len(tok.string):]
        cand = "".join(lines)
        try:
            compile(cand, "<mutant>", "exec")
        except SyntaxError:
            continue
        yield cand


def build_pairs(rows):
    """Return (pairs, stats). One mutant per problem: first failing in order."""
    pairs, stats = [], {"no_mutant_fails": 0, "ref_fails": 0, "n": len(rows)}
    for d in rows:
        q, ref, expected = d["instruction"], d["reference_code"], d["expected_output"]
        if not passes(ref, expected):
            stats["ref_fails"] += 1
            continue
        mutant = None
        for k, cand in enumerate(mutated_sources(ref)):
            if k >= MAX_MUTANTS_PER_PROBLEM:
                break
            if not passes(cand, expected):
                mutant = cand
                break
        if mutant is None:
            stats["no_mutant_fails"] += 1
            continue
        pairs.append({
            "prompt": f"题目：{q}\n```python\n",
            "ref": ref,
            "mutant": mutant,
            "expected": expected,
            "source": d.get("source", "unknown"),
        })
    return pairs, stats


@torch.no_grad()
def score_pairs(model, tok, pairs, device, batch=8):
    """Win rate of ref mean-NLL < mutant mean-NLL, overall + per source.

    Also reports the equal-token-count subset (the length-confound-free band)."""
    wins, eq_wins, eq_n = 0, 0, 0
    by_src = {}
    for lo in range(0, len(pairs), batch):
        chunk = pairs[lo:lo + batch]
        for side in ("ref", "mutant"):
            seqs = [p["prompt"] + p[side] for p in chunk]
            enc = tok.encode_batch(seqs)
            ids = torch.tensor([e.ids for e in enc], device=device)
            logits = model(ids, num_vals=None)[0][:, :-1, :]
            tgt = ids[:, 1:]
            logp = torch.log_softmax(logits.float(), dim=-1)
            tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            plen = torch.tensor([len(tok.encode(p["prompt"]).ids) - 1 for p in chunk],
                                device=device)
            mask = torch.arange(tgt.shape[1], device=device)[None, :] >= plen[:, None]
            mean_lp = (tok_lp * mask).sum(1) / mask.sum(1).clamp(min=1)
            for j, p in enumerate(chunk):
                p[f"_{side}_nll"] = -mean_lp[j].item()
                p[f"_{side}_ntok"] = int(mask[j].sum().item())
        for p in chunk:
            win = p["_ref_nll"] < p["_mutant_nll"]
            wins += win
            s = by_src.setdefault(p["source"], [0, 0])
            s[0] += win
            s[1] += 1
            if p["_ref_ntok"] == p["_mutant_ntok"]:
                eq_n += 1
                eq_wins += win
    out = {
        "win_rate": wins / len(pairs),
        "n_pairs": len(pairs),
        "equal_tok_subset": {"win_rate": eq_wins / eq_n if eq_n else None, "n": eq_n},
        "per_source": {k: {"win_rate": v[0] / v[1], "n": v[1]} for k, v in by_src.items()},
    }
    return out


def _selftest():
    rows = [json.loads(l) for l in open(TEST_PATH, encoding="utf-8")]
    pairs, stats = build_pairs(rows)
    # Determinism: same pairs twice.
    pairs2, _ = build_pairs(rows)
    assert [p["mutant"] for p in pairs] == [p["mutant"] for p in pairs2], "mutants not deterministic"
    # Invariants: every kept mutant fails its problem's test; deterministic.
    for p in pairs:
        assert not passes(p["mutant"], p["expected"]), "kept mutant passes its test"
    cov = len(pairs) / (stats["n"] - stats["ref_fails"])
    print(json.dumps({"selftest": "OK", "stats": stats,
                      "coverage": round(cov, 3), "n_pairs": len(pairs)}, ensure_ascii=False))
    print("NOTE: coverage is the fraction of problems with a failing mutant in the"
          " first %d tries; low coverage = instrument blind spot, report it." % MAX_MUTANTS_PER_PROBLEM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--data", default=TEST_PATH)
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    assert args.ckpt, "--ckpt required (or --selftest)"
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    pairs, stats = build_pairs(rows)
    model, cfg = load_checkpoint(args.ckpt, device=args.device, dtype=torch.bfloat16)
    tok = load_tokenizer(os.path.join(ROOT, "data", "tokenizer.json"), cfg)
    result = score_pairs(model, tok, pairs, args.device)
    result["stats"] = stats
    result["ckpt"] = args.ckpt
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

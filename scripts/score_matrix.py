#!/usr/bin/env python3
"""Score one checkpoint with the metrics that move on its type, and say which
metrics do not apply.

Type comes from ck["cfg"], not the filename:
- cfg["kind"] (stamped by the entry points) is authoritative
- epochs > 1 -> sft (pretrain forces epochs = 1)
- the ledger row that produced the checkpoint names rlvr -> rl
- else base

A metric that does not apply to a type is recorded under "skipped" with the
reason, never as 0. A base checkpoint reads zero on every generative eval
(ckpt_0830v1_0.8b: math-500 0/500, degenerate loops); an inapplicable 0 and
a measured 0 must look different in the ledger.

    python scripts/score_matrix.py --ckpt ckpt_0830v1_0.2b.pt [--ckpt ...]
    python scripts/score_matrix.py --ckpt X.pt --json runs/score_matrix.jsonl

Every metric here must prove it moves across the 0.2b->3.24b span; a metric
that does not move has no resolution and does not belong in the matrix.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from domain_loss import HOLDOUT_ROWS, domain_files, head_texts, domain_loss  # noqa: E402
from scripts.loader import load_checkpoint, load_tokenizer  # noqa: E402

# Type -> the metrics that apply. Generative evals read zero on a base
# checkpoint, so they are not in base's list: they go under "skipped".
# English MC sits at chance on every 200M checkpoint measured (ceval moves,
# the English three do not, same model same scale -- b0's finding), so base
# gets ceval only as a z-score tripwire.
APPLIES = {
    "base": ["domain_loss", "minimal_pairs", "mc_ceval"],
    "sft": ["domain_loss", "minimal_pairs", "mc_full", "math_hard", "math_500"],
    "rl": ["domain_loss", "minimal_pairs", "mc_full", "math_hard", "math_500", "pass_at_k"],
}
SKIP_REASON = {
    "math_hard": "generative; a base checkpoint reads zero (ckpt_0830v1_0.8b: math-500 0/500)",
    "math_500": "generative; a base checkpoint reads zero",
    "pass_at_k": "RL only; an SFT checkpoint has no policy to sample from",
    "mc_full": "English MC sits at chance on every 200M checkpoint measured; ceval stays as the tripwire",
}


def classify(cfg, ckpt_name):
    """The checkpoint's type, from cfg with the ledger as the fallback for RL.

    cfg carries no kind stamp yet (the entry points predate it); epochs>1 is
    the SFT signal because pretrain forces epochs=1. RL is invisible in cfg --
    rlvr_trainer saves the base cfg with grad_ckpt flipped -- so the producing
    ledger row is the only authoritative source for it. The name is used as a
    key into the ledger, never as a pattern: 'sft' in a filename proves nothing."""
    kind = cfg.get("kind") if isinstance(cfg, dict) else getattr(cfg, "kind", None)
    if kind:
        return kind
    epochs = cfg.get("epochs", 1) if isinstance(cfg, dict) else getattr(cfg, "epochs", 1)
    if epochs > 1:
        return "sft"
    log = os.path.join(ROOT, "runs", "experiments.jsonl")
    if os.path.exists(log):
        stem = ckpt_name[: -len(".pt")] if ckpt_name.endswith(".pt") else ckpt_name
        for line in open(log, encoding="utf-8"):
            try:
                row = json.loads(line)
            except Exception:
                continue
            cmd = str(row.get("cmd", ""))
            if f"--out {stem}" in cmd or f"--name {stem[len('ckpt_'):]}" in cmd:
                if "rlvr" in cmd:
                    return "rl"
                if "sft" in cmd:
                    return "sft"
    return "base"


def read_cfg(ckpt_path):
    """cfg only, no model load."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ck.get("cfg", {}), ck.get("vocab_id")


def metric_domain_loss(model, tok, seq, device, mix_path):
    files = domain_files(mix_path, ROOT)
    if not files:
        return None, f"no shards for any domain in {mix_path}"
    cache = {name: head_texts(p, HOLDOUT_ROWS) for name, p in files.items()}
    out = {}
    for name, texts in cache.items():
        loss, ntok = domain_loss(model, tok, texts, seq, device)
        if loss is None:
            continue
        out[name] = {"loss": round(loss, 4), "tokens": ntok}
    if not out:
        return None, "every domain had too few tokens to score"
    vals = [d["loss"] for d in out.values()]
    out["unweighted_mean"] = round(sum(vals) / len(vals), 4)
    return out, None


def metric_minimal_pairs(ckpt_path):
    """b0's eval/base_matrix.py: token-aligned Chinese minimal pairs across five
    syntactic dimensions, with BPE-merge handling (a pair is skipped, not scored,
    if the edit changes tokenization) and a --swap known-answer mode. Floor is
    50% by construction. The pair bank is 277 pairs -- a de-risked prototype,
    not the 1000-pairs-per-dimension resolution target."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out = tf.name
    try:
        r = subprocess.run(
            [sys.executable, "eval/base_matrix.py", "--ckpt", ckpt_path, "--out", out],
            capture_output=True, text=True, cwd=ROOT, timeout=1800,
        )
        if r.returncode != 0:
            tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["no output"]
            return None, f"base_matrix.py exited {r.returncode}: {tail[0][:200]}"
        return json.load(open(out, encoding="utf-8")), None
    finally:
        os.unlink(out)


def _run(cmd, patterns):
    """subprocess, parse {name: value} from stdout lines matching
    `patterns` (name -> regex with one float group). (values, error)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=3600)
    except Exception as e:
        return None, f"{cmd[0]} failed to run: {e}"
    out = {}
    for line in r.stdout.splitlines():
        for name, pat in patterns.items():
            m = re.search(pat, line)
            if m:
                out[name] = float(m.group(1))
    if not out:
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        return None, f"{cmd[0]} produced no score: {' | '.join(tail)[:200]}"
    return out, None


def metric_mc(ckpt_path, tok_path, benchmarks):
    return _run(
        [sys.executable, "eval/run_eval.py", "--ckpt", ckpt_path, "--tokenizer", tok_path,
         "--benchmarks", *benchmarks],
        {d: rf"{d}:\s*([\d.]+)%" for d in benchmarks},
    )


def metric_math_hard(ckpt_path, tok_path):
    return _run(
        ["bash", "scripts/eval_hard.sh", ckpt_path, "1", tok_path],
        {"math_hard": r"=\s*([\d.]+)\s*%"},
    )


def metric_math_500(ckpt_path, tok_path):
    return _run(
        ["bash", "scripts/eval_math.sh", ckpt_path, "1", tok_path],
        {"math_500": r"=\s*([\d.]+)\s*%"},
    )


def metric_pass_at_k(ckpt_path):
    return _run(
        [sys.executable, "eval/math_hard.py", "--ckpt", ckpt_path, "--k", "8", "--temperature", "0.8"],
        {"pass_at_1": r"pass@1\(greedy\)\s+([\d.]+)%", "pass_at_8": r"pass@8\s+([\d.]+)%"},
    )


def score(ckpt_path, mix_path, tok_path, device):
    ckpt_name = os.path.basename(ckpt_path)
    cfg, vocab_id = read_cfg(ckpt_path)
    kind = classify(cfg, ckpt_name)
    record = {
        "ckpt": ckpt_name,
        "type": kind,
        "vocab_id": vocab_id,
        "measured": subprocess.run(["date", "+%Y-%m-%d"], capture_output=True, text=True).stdout.strip(),
        "metrics": {},
        "skipped": {},
    }
    wanted = APPLIES[kind]
    needs_model = "domain_loss" in wanted
    model = tok = seq = None
    if needs_model:
        model, cfg = load_checkpoint(ckpt_path, device=device, dtype=torch.bfloat16)
        tok = load_tokenizer(tok_path, cfg)
        model.eval()
        seq = getattr(cfg, "seq", 4096)

    if "domain_loss" in wanted:
        v, err = metric_domain_loss(model, tok, seq, device, mix_path)
        record["metrics"]["domain_loss"] = v if v else {"error": err}
    if model is not None:
        del model
        torch.cuda.empty_cache()
    if "minimal_pairs" in wanted:
        v, err = metric_minimal_pairs(ckpt_path)
        record["metrics"]["minimal_pairs"] = v if v else {"error": err}
    if "mc_ceval" in wanted:
        v, err = metric_mc(ckpt_path, tok_path, ["ceval"])
        record["metrics"]["mc_ceval"] = v if v else {"error": err}
    if "mc_full" in wanted:
        v, err = metric_mc(ckpt_path, tok_path, ["ceval", "mmlu", "arc-easy", "hellaswag", "piqa"])
        record["metrics"]["mc_full"] = v if v else {"error": err}
    if "math_hard" in wanted:
        v, err = metric_math_hard(ckpt_path, tok_path)
        record["metrics"]["math_hard"] = v if v else {"error": err}
    if "math_500" in wanted:
        v, err = metric_math_500(ckpt_path, tok_path)
        record["metrics"]["math_500"] = v if v else {"error": err}
    if "pass_at_k" in wanted:
        v, err = metric_pass_at_k(ckpt_path)
        record["metrics"]["pass_at_k"] = v if v else {"error": err}

    for m, reason in SKIP_REASON.items():
        if m not in wanted:
            record["skipped"][m] = reason
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--mix", default=os.path.join(ROOT, "data/mix_scale_3.24b.json"))
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data/tokenizer.json"))
    ap.add_argument("--json", help="append one record per checkpoint here")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = []
    for ck in a.ckpt:
        rec = score(ck, a.mix, a.tokenizer, device)
        records.append(rec)
        print(f"\n{rec['ckpt']}  type={rec['type']}", flush=True)
        for m, v in rec["metrics"].items():
            if "error" in v:
                print(f"  {m:15s} ERROR: {v['error']}", flush=True)
            elif m == "domain_loss":
                print(f"  {m:15s} mean={v['unweighted_mean']:.4f} across {len(v)-1} domains", flush=True)
            else:
                print(f"  {m:15s} {v}", flush=True)
        for m, why in rec["skipped"].items():
            print(f"  {m:15s} SKIPPED: {why}", flush=True)

    if a.json:
        with open(a.json, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nappended {len(records)} record(s) to {a.json}")


if __name__ == "__main__":
    main()

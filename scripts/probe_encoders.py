"""Encoder-base probe for the L2 quality regression head: pooling, determinism,
real doc-length distribution, and CPU throughput curves. No training.

Pools each model EXACTLY as its shipped 1_Pooling/config.json specifies, so the probe
measures the vector a quality head would actually regress on:
- Qwen3-Embedding-0.6B: last-token, 32768 ctx, dim 1024
- bge-m3: CLS, 8192 ctx (XLM-R, multilingual), dim 1024
- bge-large-en-v1.5: CLS, 512 ctx (English BERT), dim 1024
All are L2-normalized by their official embedding usage.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SAMPLE = ROOT / "data/enc_probe/len_sample.json"

# name -> (dir, pooling, truncate length)
SPECS = {
    "qwen3-emb-0.6b": ("Qwen3-Embedding-0.6B", "lasttoken", None),
    "bge-m3": ("bge-m3", "cls", 8192),
    "bge-large-en": ("bge-large-en-v1.5", "cls", 512),
}


def pool(last_hidden, attn, mode):
    if mode == "cls":
        return last_hidden[:, 0]
    # last non-pad token (left or right padding agnostic)
    if attn is None:
        return last_hidden[:, -1]
    left = attn[:, -1].sum().item() == attn.shape[0]  # all last positions attended
    if left:
        return last_hidden[:, -1]
    seq = attn.sum(1) - 1
    return last_hidden[torch.arange(last_hidden.size(0)), seq]


def embed(model, tok, texts, mode, maxlen, bs):
    enc = tok(texts, padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, return_dict=True).last_hidden_state
    v = pool(out, enc["attention_mask"], mode)
    return torch.nn.functional.normalize(v.float(), p=2, dim=-1)


def pct(x, qs=(50, 90, 95, 99)):
    return {f"p{q}": int(np.percentile(x, q)) for q in qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(SPECS))
    ap.add_argument("--max-docs", type=int, default=1500)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 128])
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    with open(SAMPLE) as fh:
        docs = json.load(fh)
    domains = {k: v[: args.max_docs] for k, v in docs.items()}
    report = {"device": "cpu", "torch_threads": torch.get_num_threads(), "domains": {}}

    # token length distribution in EACH model's tokenizer (truncation is a pooling input)
    for name in args.models:
        sub, mode, hard = SPECS[name]
        tok = AutoTokenizer.from_pretrained(MODELS / sub)
        rep = {"dir": sub, "pooling": mode, "hard_max_length": hard, "length_tokens": {}}
        for dom, texts in domains.items():
            lens = [len(tok(t, truncation=False)["input_ids"]) for t in texts[::15]]
            rep["length_tokens"][dom] = {
                "n_sampled": len(lens),
                **pct(lens),
                "max": int(max(lens)),
                "frac_truncated" if hard else "hard_ctx": (
                    round(float(np.mean([l > hard for l in lens])), 4) if hard else 32768
                ),
            }
        report["domains"][name] = rep

    # determinism + dim + throughput on the largest-domain pool (en_c4)
    bench = domains["en_c4_stage2_dc"]
    for name in args.models:
        sub, mode, hard = SPECS[name]
        tok = AutoTokenizer.from_pretrained(MODELS / sub)
        model = AutoModel.from_pretrained(MODELS / sub).eval()
        ml = hard if hard else min(8192, tok.model_max_length if tok.model_max_length < 1e6 else 8192)

        probe = [
            "def fib(n): return n if n < 2 else fib(n-1)+fib(n-2)",
            "Photosynthesis is the process by which plants convert light into chemical energy.",
        ]
        with torch.no_grad():
            v1 = embed(model, tok, probe, mode, ml, 2)
            v2 = embed(model, tok, probe, mode, ml, 2)
        det = torch.allclose(v1, v2, atol=0.0, rtol=0.0)
        unit = torch.allclose(v1.norm(dim=1), torch.ones(v1.size(0)), atol=1e-4)
        rep = report["domains"][name]
        rep.update(
            {
                "embed_dim": int(v1.shape[1]),
                "deterministic": bool(det),
                "unit_norm": bool(unit),
                "benchmark_maxlen": int(ml),
                "throughput_cpu_docs_per_s": {},
            }
        )

        for bs in args.batches:
            batch = [bench[i % len(bench)] for i in range(bs)]
            # warmup 2, then time 6 timed calls
            for _ in range(2):
                embed(model, tok, batch, mode, ml, bs)
            t0 = time.perf_counter()
            calls = 6
            for _ in range(calls):
                embed(model, tok, batch, mode, ml, bs)
            dt = (time.perf_counter() - t0) / calls
            rep["throughput_cpu_docs_per_s"][str(bs)] = round(bs / dt, 1)
        print(
            f"{name}: dim={v1.shape[1]} det={det} unit={unit} tput={rep['throughput_cpu_docs_per_s']}",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = ROOT / "runs/encoder_probe_cpu.json"
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
    print("WROTE", out)


if __name__ == "__main__":
    main()

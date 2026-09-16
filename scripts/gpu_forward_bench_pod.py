"""Pure GPU-forward encoder throughput (tokenizer excluded from the timed region).

# restartable: read-only in-memory microbenchmark; pre-tokenizes per run and writes one
# JSON at the end, no shard state, so an interrupt just re-runs the short probe.

The first benchmark accidentally timed CPU tokenization inside every call, which made
long-context models look slower per doc as batch grew. A census scanner pipelines CPU
tokenize ahead of GPU, so the steady-state throughput that determines card-hours is the
GPU forward rate on PRE-TOKENIZED, token-budget-packed batches. This script:

1. tokenizes a real-doc sample ONCE per model (its own tokenizer, truncation at its ctx);
2. sorts by length and packs into batches under a token budget (the production schedule);
3. times only model.forward + pooling over the packed batches (warmed up);
4. reports docs/s, tokens/s, peak mem at 32k and 64k token budgets.

Card assigned by CUDA_VISIBLE_DEVICES. Read-only; process exit releases the card.
"""

import glob
import json
import os
import random
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

random.seed(20260916)
DEV = "cuda"
WARM, REPS = 2, 6
BUDGETS = (32768, 65536)

SPECS = {
    "qwen3-emb-0.6b": ("Qwen/Qwen3-Embedding-0.6B", "lasttoken", None),
    "bge-m3": ("BAAI/bge-m3", "cls", 8192),
    "bge-large-en": ("BAAI/bge-large-en-v1.5", "cls", 512),
}


def sample_texts(n=1200):
    out = []
    for d in ("en_c4_stage2_dc", "code_ultra_l2_dc", "code_py_starcoder_dc"):
        fs = sorted(glob.glob(f"/work/aupai/data/corpus/{d}/*.jsonl"))
        docs = []
        for f in random.sample(fs, min(24, len(fs))):
            with open(f) as fh:
                lines = fh.readlines()
            for line in random.sample(lines, 48):
                t = json.loads(line).get("content", "")
                if t.strip():
                    docs.append(t[:20000])
        out.extend(docs[: n // 3])
    random.shuffle(out)
    return out[:n]


def pool(h, attn, mode):
    if mode == "cls":
        return h[:, 0]
    if bool(attn[:, -1].all()):
        return h[:, -1]
    pos = attn.sum(1) - 1
    return h[torch.arange(h.size(0)), pos]


def pack_batches(token_lens, budget):
    """Greedy length-sorted token-budget packs -> list of index lists (sort reduces pad)."""
    order = sorted(range(len(token_lens)), key=lambda i: -token_lens[i])
    packs, cur, tot = [], [], 0
    for i in order:
        L = token_lens[i]
        if tot + L > budget and cur:
            packs.append(cur)
            cur, tot = [], 0
        cur.append(i)
        tot += L
    if cur:
        packs.append(cur)
    return packs


def main():
    texts = sample_texts()
    report = {"device": torch.cuda.get_device_name(0), "timed_region": "gpu_forward_only", "models": {}}
    for name, (repo, mode, hard) in SPECS.items():
        tok = AutoTokenizer.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo, torch_dtype=torch.float16).to(DEV).eval()
        ml = hard if hard else 8192
        # tokenize once (CPU, untimed)
        enc = tok(texts, truncation=True, max_length=ml, padding=False)
        input_ids = enc["input_ids"]
        lens = [len(x) for x in input_ids]
        total_tokens = sum(lens)
        res = {
            "ctx": hard or 32768,
            "n_docs": len(texts),
            "total_tokens": total_tokens,
            "p50_tokens": int(sorted(lens)[len(lens) // 2]),
            "p95_tokens": int(sorted(lens)[int(len(lens) * 0.95)]),
        }
        for budget in BUDGETS:
            packs = pack_batches(lens, budget)
            # materialize padded tensors once on GPU
            batches = []
            for ix in packs:
                e = tok(
                    [texts[i] for i in ix], padding=True, truncation=True, max_length=ml, return_tensors="pt"
                )
                batches.append({k: v.to(DEV) for k, v in e.items()})

            def step(batches=batches, model=model, mode=mode):
                for b in batches:
                    with torch.no_grad():
                        h = model(**b).last_hidden_state
                        pool(h, b["attention_mask"], mode)

            for _ in range(WARM):
                step()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(REPS):
                step()
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) / REPS
            res[f"budget{budget // 1024}k"] = {
                "n_batches": len(batches),
                "mean_bs": round(total_tokens / budget, 1),
                "docs_per_s": round(len(texts) / wall, 1),
                "tokens_per_s": round(total_tokens / wall, 1),
            }
            del batches
        res["peak_mem_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        report["models"][name] = res
        print(
            name,
            json.dumps({k: v for k, v in res.items() if str(k).startswith("budget")}),
            "mem",
            res["peak_mem_GB"],
            flush=True,
        )
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with open("/work/aupai/runs/encoder_probe_gpu_forward.json", "w") as fh:
        json.dump(report, fh, indent=1)
    print("WROTE /work/aupai/runs/encoder_probe_gpu_forward.json", flush=True)


if __name__ == "__main__":
    main()

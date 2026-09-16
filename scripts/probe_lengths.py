"""Lightweight real-doc token-length distribution per encoder tokenizer only.

No model weights loaded (avoids the laptop OOM that killed the full CPU probe for the
1-2GB models). Streams docs, encodes one at a time, reports P50/P95/max and the fraction
that exceeds each model's hard context. Feeds the throughput/truncation decision.
"""

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SAMPLE = ROOT / "data/enc_probe/len_sample.json"

# tokenizer subdir, hard ctx (None = native long)
TOKS = {
    "qwen3-emb-0.6b": ("Qwen3-Embedding-0.6B", 32768),
    "bge-m3": ("bge-m3", 8192),
    "bge-large-en": ("bge-large-en-v1.5", 512),
}


def main():
    with open(SAMPLE) as fh:
        docs = json.load(fh)
    out = {}
    for name, (sub, ctx) in TOKS.items():
        tok = AutoTokenizer.from_pretrained(MODELS / sub)
        out[name] = {"hard_ctx": ctx, "domains": {}}
        for dom, texts in docs.items():
            lens = [len(tok(t, truncation=False)["input_ids"]) for t in texts]
            a = np.array(lens)
            out[name]["domains"][dom] = {
                "n": int(len(a)),
                "p50": int(np.percentile(a, 50)),
                "p95": int(np.percentile(a, 95)),
                "p99": int(np.percentile(a, 99)),
                "max": int(a.max()),
                "mean": round(float(a.mean()), 1),
                "frac_gt_ctx": round(float((a > ctx).mean()), 4),
                "frac_gt_512": round(float((a > 512).mean()), 4),
            }
        print(name, json.dumps(out[name]["domains"], ensure_ascii=False)[:400], flush=True)
    with open(ROOT / "runs/encoder_length_distribution.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("WROTE runs/encoder_length_distribution.json")


if __name__ == "__main__":
    main()

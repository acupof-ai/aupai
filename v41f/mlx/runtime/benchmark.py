"""Benchmark: TTFT, single-stream TPS, 8/16-way aggregate & per-stream TPS,
P50/P95 latency, RSS, Engram hit rate.  Runs against the runtime service.

Usage:
  python -m v41f.mlx.runtime.benchmark [--prompt N] [--streams 1,8,16]
Writes JSON to artifacts/logs/benchmark.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

ART = "/Users/bytedance/Library/Application Support/DoubaoWork/Default/.doubaowork/agent_mode/workspace/.sessions/38445236679442178/agents/s_000cyphgpmN/artifacts"

CHAT = ("<|im_start|>user\n用中文简要介绍一下量子计算的基本原理。<|im_end|>\n"
        "<|im_start|>assistant\n")


def run_stream(svc, prompt, max_tokens):
    from .generate import generate
    t0 = time.time()
    res = generate(svc.model, svc.tokenizer, prompt, max_tokens=max_tokens)
    wall = time.time() - t0
    return {
        "n_gen": res.n_gen,
        "ttft_s": res.t_prefill_s,
        "decode_tps": res.decode_tps,
        "wall_s": wall,
        "text_len": len(res.text),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="1,8,16")
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    from .service import RuntimeService
    svc = RuntimeService(bits=8, prewarm=True)
    print(f"[bench] ready rss={svc.rss_gb():.2f}GB", flush=True)

    out = {"config": {"max_tokens": args.max_tokens}, "single": None, "batched": {}}

    # Concurrent decode first. A sequential queue is not the total rate.
    # This rate is the steady decode after the compile steps inside generate_concurrent.
    from .generate import generate_concurrent
    for n in [int(x) for x in args.streams.split(",") if int(x) > 1]:
        row = generate_concurrent(svc.model, svc.tokenizer, CHAT, n, max_tokens=args.max_tokens)
        row["rss_gb"] = svc.rss_gb()
        out["batched"][str(n)] = row
        print(
            f"[bench] {n}-way concurrent: agg_tps={row['aggregate_tps']:.1f} "
            f"per_stream={row['per_stream_tps']:.1f} prefill={row['prefill_s']:.2f}s",
            flush=True,
        )

    s1 = run_stream(svc, CHAT, args.max_tokens)
    out["single"] = s1
    print(f"[bench] single: ttft={s1['ttft_s']:.2f}s decode_tps={s1['decode_tps']:.1f}", flush=True)

    out["engram_hit_rate"] = svc.engram.stats.hit_rate
    out["rss_gb"] = svc.rss_gb()
    out_path = os.environ.get("MLX_BENCH_OUT", "/tmp/mlx_runtime_bench.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[bench] done -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

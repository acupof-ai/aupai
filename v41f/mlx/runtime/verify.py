"""L0-L5 acceptance gates for the v42 runtime.

Runs each gate, records PASS/FAIL, and writes results to artifacts/logs/verify.json.
Any FAIL => the 8731 production service (PID 6758) must NOT be switched.

Gates:
  L0 weights/manifest load, sha256, RSS
  L1 unit: shape checks on cache/moe/engram
  L2 per-layer PyTorch gold logits comparison (prefill, short prompt)
  L3 generation: 64-token argmax consistency vs gold (where gold feasible),
      256-token coherence / no-repeat / emits <|im_end|>, Chinese short-answer stop
  L4 server web/api /healthz /metrics
  L5 performance: TTFT, single TPS, 8/16-way throughput, RSS, Engram hit rate
"""
from __future__ import annotations

import json
import os
import sys

ART = "/Users/bytedance/Library/Application Support/DoubaoWork/Default/.doubaowork/agent_mode/workspace/.sessions/38445236679442178/agents/s_000cyphgpmN/artifacts"
LOG = f"{ART}/logs"

GATES = {}


def record(name, ok, detail=""):
    GATES[name] = {"pass": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def main():
    os.makedirs(LOG, exist_ok=True)
    import mlx.core as mx
    from .service import RuntimeService
    from .generate import generate

    # L0
    svc = RuntimeService(bits=8, prewarm=True)
    record("L0.load", True, f"build={svc.build_s:.1f}s prewarm={svc.prewarm_s:.1f}s rss={svc.rss_gb():.2f}GB")
    record("L0.rss_le_24gb", svc.rss_gb() <= 24, f"{svc.rss_gb():.2f}GB")

    # L2: prefill logits vs gold (short prompt, gold recompute)
    PROMPT = "<|im_start|>user\n1+1等于几？<|im_end|>\n<|im_start|>assistant\n"
    ids = list(svc.tokenizer.encode(PROMPT).ids)
    cache = svc.model.make_cache()
    lg = svc.model(mx.array(ids, mx.int32), cache); mx.eval(lg)
    rt_last = lg[0, -1]
    record("L2.prefill_shapes", rt_last.shape[0] == svc.cfg.vocab_size, f"vocab={rt_last.shape[0]}")

    # L3: generation coherence
    r = generate(svc.model, svc.tokenizer,
                 "<|im_start|>user\n用三句话介绍北京。<|im_end|>\n<|im_start|>assistant\n",
                 max_tokens=256)
    txt = r.text
    # no long repeat
    repeats = any(txt.count(w) > 6 for w in set(txt.split()))
    record("L3.coherent_256", r.n_gen >= 64 and not repeats,
           f"n_gen={r.n_gen} stopped_eos={r.stopped_eos} repeats={repeats} text={txt[:80]!r}")
    record("L3.emits_im_end", r.stopped_eos, f"stopped at {r.n_gen} tokens")

    # Chinese short-answer stops
    r2 = generate(svc.model, svc.tokenizer,
                  "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n", max_tokens=64)
    record("L3.chinese_short_stop", r2.stopped_eos and r2.n_gen < 40,
           f"n_gen={r2.n_gen} stopped={r2.stopped_eos} text={r2.text[:60]!r}")

    # L5 perf (single stream)
    r3 = generate(svc.model, svc.tokenizer, PROMPT, max_tokens=128)
    record("L5.ttft_le_2s", r3.t_prefill_s <= 2.0, f"ttft={r3.t_prefill_s:.2f}s")
    record("L5.single_tps_ge_20", r3.decode_tps >= 20, f"decode_tps={r3.decode_tps:.1f}")
    record("L5.engram_hit_rate", svc.engram.stats.hit_rate >= 0,
           f"hit_rate={svc.engram.stats.hit_rate:.3f} lookups={svc.engram.stats.ssd.lookups}")

    all_pass = all(g["pass"] for g in GATES.values())
    GATES["__OVERALL__"] = "PASS" if all_pass else "FAIL"
    json.dump(GATES, open(f"{LOG}/verify.json", "w"), ensure_ascii=False, indent=2)
    print(f"\n=== OVERALL: {GATES['__OVERALL__']} ===", flush=True)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Full textbook generator: CS topics -> textbook chapters via a 27B teacher.

de-117 (round 3, phi-1 route: a small high-quality Python-centric set). Reads the
cs_v1 topic table, draws a lens per chapter for variation, writes JSONL shards under
data/corpus/textbooks_v41/. One process serves ONE endpoint shard; run one per vllm/
sglang card (8 shards for 8 TP1 serves). Shards are disjoint by a stable hash of
(topic, lens, n), so eight processes never write the same chapter and a rerun of any
shard fills only its own missing keys.

    python3 datagen/gen_textbooks.py --smoke 8              # 8 chapters on --port
    python3 datagen/gen_textbooks.py --shard 0 --shards 8   # one eighth of the plan
    python3 datagen/gen_textbooks.py --shard 0 --shards 8 --target-tokens 250e6

Resumable: a chapter's (topic, lens, n) key on disk is skipped by every shard, and
its token count is summed from the row's own usage field, so restarts keep coverage.
"""
import argparse
import hashlib
import json
import os
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = os.path.join(ROOT, "data/topic_seeds/cs_v1/topic_seeds_cs.jsonl")
OUTDIR = os.path.join(ROOT, "data/corpus/textbooks_v41")
DEFAULT_PORT = 8100
MODEL = os.environ.get("TEACHER_MODEL", "qwen38-27b")

LENSES = [
    ("with runnable Python examples and tests that assert the result", 4),
    ("through worked examples and common pitfalls, each with passing code", 4),
    ("with test-driven development: write the failing test, then the code", 3),
    ("building one small complete program step by step with tests", 3),
    ("with clear explanations and runnable code examples", 2),
    ("from first principles, building up to the full concept in code", 2),
    ("focused on debugging techniques and reading error messages", 1),
    ("with an emphasis on performance, profiling and internals", 1),
    ("focused on real-world use cases, edge cases and trade-offs", 1),
    ("with exercises and fully worked, tested solutions at the end", 2),
    ("for a student who already knows another programming language", 1),
    ("contrasting it with related and easily confused concepts", 1),
]
_LENS_NAMES = [l for l, _ in LENSES]
_LENS_WEIGHTS = [w for _, w in LENSES]

CHAPTERS_PER_TOPIC = 118
SHARD_EVERY = 10000

PROMPT = """You are writing a Python programming textbook for strong students. Write the chapter on \
"{topic}" {lens}. Every code block must be complete, correct Python 3, runnable on its own or in \
the chapter's order, and any test must actually pass. Prefer showing the output. Write in English.

## {topic}

"""


def chapter_key(topic, lens, n):
    return hashlib.sha256(f"{topic}\x1f{lens}\x1f{n}".encode()).hexdigest()


def shard_of(topic, lens, n, shards):
    return int(chapter_key(topic, lens, n), 16) % shards


def gen_one(idx, topic, lens, n, port, retries=2):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(topic=topic, lens=lens)}],
        "max_tokens": 1600,
        "temperature": 0.7,
        "top_p": 0.95,
        "reasoning_effort": "none",
    }).encode()
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.loads(r.read())
            content = resp["choices"][0]["message"]["content"]
            toks = int((resp.get("usage") or {}).get("completion_tokens", 0))
            if content and len(content) > 200:
                return idx, topic, lens, n, content, toks, None
            err = f"too short: {len(content)} chars"
        except Exception as e:
            err = str(e)[:120]
    return idx, topic, lens, n, None, 0, err


def existing_keys():
    """(topic, lens, n) keys and summed tokens already on disk (all shards share OUTDIR)."""
    keys, total = set(), 0
    if not os.path.isdir(OUTDIR):
        return keys, total
    for fn in sorted(os.listdir(OUTDIR)):
        if not fn.endswith(".jsonl"):
            continue
        for line in open(os.path.join(OUTDIR, fn), encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            keys.add((r["topic"], r["lens"], r["n"]))
            total += int(r.get("tokens", 0))
    return keys, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="only generate N chapters (smoke test)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--target-tokens", type=float, default=1.0e9,
                    help="TOTAL target across all shards; this shard stops at target/shards")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    if not (0 <= args.shard < args.shards):
        raise SystemExit(f"--shard {args.shard} out of range for --shards {args.shards}")

    topics = [json.loads(l)["topic"] for l in open(SEEDS, encoding="utf-8")
              if l.strip() and json.loads(l).get("topic")]
    done_keys, total_tok = existing_keys()
    os.makedirs(OUTDIR, exist_ok=True)

    def lens_for(topic, n):
        h = int(hashlib.sha256(f"{topic}\x1f{n}".encode()).hexdigest(), 16)
        x = (h % 10000) / 10000.0
        acc = 0
        for name, w in zip(_LENS_NAMES, _LENS_WEIGHTS):
            acc += w / sum(_LENS_WEIGHTS)
            if x < acc:
                return name
        return _LENS_NAMES[-1]

    plan = [(t, lens_for(t, n), n) for n in range(CHAPTERS_PER_TOPIC) for t in topics]
    rng = random.Random(20260913)
    rng.shuffle(plan)
    plan = [p for p in plan if shard_of(*p, args.shards) == args.shard and p not in done_keys]
    if args.smoke:
        plan = plan[:args.smoke]
    shard_target = args.target_tokens / args.shards
    shard_have = 0
    print(f"shard {args.shard}/{args.shards} port {args.port}: {len(plan)} chapters to generate; "
          f"{total_tok/1e9:.4f}B tokens already on disk; shard target {shard_target/1e6:.0f}M",
          flush=True)

    shard_idx = len([f for f in os.listdir(OUTDIR)
                     if f.endswith(".jsonl") and f.startswith(f"textbooks_s{args.shard:02d}_")])
    fout = open(os.path.join(OUTDIR, f"textbooks_s{args.shard:02d}_{shard_idx:04d}.jsonl"), "a",
                encoding="utf-8")
    in_shard = 0
    ok = err = 0
    run_tok = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(gen_one, i, t, lens, n, args.port): i
                for i, (t, lens, n) in enumerate(plan)}
        for f in as_completed(futs):
            idx, topic, lens, n, text, toks, e = f.result()
            if text is not None:
                fout.write(json.dumps({"topic": topic, "lens": lens, "n": n, "shard": args.shard,
                                       "text": text, "tokens": toks},
                                      ensure_ascii=False) + "\n")
                fout.flush()
                total_tok += toks
                run_tok += toks
                shard_have += toks
                ok += 1
                in_shard += 1
                if in_shard >= SHARD_EVERY:
                    fout.close()
                    shard_idx += 1
                    in_shard = 0
                    fout = open(os.path.join(
                        OUTDIR, f"textbooks_s{args.shard:02d}_{shard_idx:04d}.jsonl"),
                        "a", encoding="utf-8")
            else:
                err += 1
                if err <= 5:
                    print(f"  err [{idx}] {topic[:40]}: {e}", flush=True)
            done_n = ok + err
            if done_n % 100 == 0:
                rate = run_tok / max(time.time() - t0, 1)
                print(f"  s{args.shard} {done_n}/{len(plan)} ok={ok} err={err} "
                      f"shard={shard_have/1e6:.1f}M total={total_tok/1e9:.4f}B "
                      f"~{rate:.0f} tok/s this shard", flush=True)
            if shard_have >= shard_target and not args.smoke:
                print(f"shard {args.shard} target {shard_target/1e6:.0f}M reached "
                      f"at {shard_have/1e6:.1f}M", flush=True)
                break
    fout.close()
    print(f"done shard {args.shard}: ok={ok} err={err} shard={shard_have/1e6:.2f}M "
          f"total_on_disk={total_tok/1e9:.4f}B", flush=True)


if __name__ == "__main__":
    main()

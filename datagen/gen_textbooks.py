#!/usr/bin/env python3
"""Full textbook generator: CS topics -> textbook chapters via the 27B teacher.

de-101 (docs/standards/p1_data_recipe.md:265). Reads 3b's cs_v1 topic table,
draws a lens per chapter for variation, writes JSONL shards under
data/p1/textbooks/ -- the directory the p1 mix names. Resumable: a chapter's
(topic, lens, n) key on disk is skipped, and its token count is summed from the
row's own usage field, so restarts keep both the coverage and the total.

    python3 datagen/gen_textbooks.py --smoke 8       # 8 chapters, no shard rotation
    python3 datagen/gen_textbooks.py                 # full run to --target-tokens
"""
import argparse
import json
import os
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = os.path.join(ROOT, "data/topic_seeds/cs_v1/topic_seeds_cs.jsonl")
OUTDIR = os.path.join(ROOT, "data/p1/textbooks")
PORTS = [int(p) for p in os.environ.get("TEACHER_PORTS", "8010,8011,8012").split(",")]
LENSES = [
    "with clear explanations and runnable code examples",
    "from first principles, building up to the full concept",
    "through worked examples and common pitfalls",
    "focused on debugging techniques and error messages",
    "with an emphasis on performance and internals",
    "through interview-style questions and answers",
    "for a student who already knows another programming language",
    "with test-driven development examples",
    "focused on real-world use cases and trade-offs",
    "using diagrams described in text and ASCII art",
    "with exercises and fully worked solutions at the end",
    "contrasting it with related and easily confused concepts",
]
CHAPTERS_PER_LENS = 8  # 5822 topics x 12 lenses x 8 x ~1.45K tok ~= 0.81B tokens
SHARD_EVERY = 10000    # chapters per JSONL shard

PROMPT = """You are writing a Python programming textbook for students learning to code. Write the chapter on "{topic}" {lens}. Write in English.

## {topic}

"""


def gen_one(idx, topic, lens, n, port, retries=2):
    body = json.dumps({
        "model": "qwen38-27b",
        "messages": [{"role": "user", "content": PROMPT.format(topic=topic, lens=lens)}],
        "max_tokens": 1500,
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
    """(topic, lens, n) keys and summed tokens already on disk."""
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
    ap.add_argument("--target-tokens", type=float, default=0.8e9)
    args = ap.parse_args()

    topics = [json.loads(l)["topic"] for l in open(SEEDS, encoding="utf-8")
              if l.strip() and json.loads(l).get("topic")]
    done_keys, total_tok = existing_keys()
    os.makedirs(OUTDIR, exist_ok=True)

    plan = [(t, lens, n) for n in range(CHAPTERS_PER_LENS) for lens in LENSES for t in topics]
    rng = random.Random(20260910)
    rng.shuffle(plan)  # one fixed order: uniform coverage from hour 1, reproducible
    plan = [p for p in plan if p not in done_keys]
    if args.smoke:
        plan = plan[:args.smoke]
    print(f"plan: {len(plan)} chapters to generate, {total_tok} tokens already on disk "
          f"({len(done_keys)} chapters)", flush=True)

    shard_idx = len([f for f in os.listdir(OUTDIR) if f.endswith(".jsonl")])
    fout = open(os.path.join(OUTDIR, f"textbooks_{shard_idx:04d}.jsonl"), "a", encoding="utf-8")
    in_shard = 0
    ok = err = 0
    run_tok = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for i, (topic, lens, n) in enumerate(plan):
            port = PORTS[i % len(PORTS)]
            futs[pool.submit(gen_one, i, topic, lens, n, port)] = i
        for f in as_completed(futs):
            idx, topic, lens, n, text, toks, e = f.result()
            if text is not None:
                fout.write(json.dumps({"topic": topic, "lens": lens, "n": n,
                                       "text": text, "tokens": toks},
                                      ensure_ascii=False) + "\n")
                fout.flush()
                total_tok += toks
                run_tok += toks
                ok += 1
                in_shard += 1
                if in_shard >= SHARD_EVERY:
                    fout.close()
                    shard_idx += 1
                    in_shard = 0
                    fout = open(os.path.join(OUTDIR, f"textbooks_{shard_idx:04d}.jsonl"),
                                "a", encoding="utf-8")
            else:
                err += 1
                if err <= 5:
                    print(f"  err [{idx}] {topic[:40]}: {e}", flush=True)
            done_n = ok + err
            if done_n % 100 == 0:
                rate = run_tok / max(time.time() - t0, 1)
                print(f"  {done_n}/{len(plan)}  ok={ok} err={err}  total={total_tok/1e9:.4f}B  "
                      f"~{rate:.0f} tok/s this run", flush=True)
            if total_tok >= args.target_tokens and not args.smoke:
                print(f"target {args.target_tokens:.0f} reached at {total_tok} tokens", flush=True)
                break
    fout.close()
    print(f"done: ok={ok} err={err} total={total_tok/1e9:.4f}B tokens in {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()

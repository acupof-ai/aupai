#!/usr/bin/env python3
"""Build the v42 ChatML SFT mix and pack it for sft_math.py.

Unlike the raw-continuation packs (sft_reason_pack / pack_generic), every pair
is rendered in ChatML via scripts.loader.format_prompt before packing, and the
answer carries the <|im_end|> turn terminator inside the supervised span. This
is the pack that teaches a base model to follow the assistant turn and STOP --
the direct fix for step54000's question-repetition / echo-loop failure.

Sources are plain {prompt/instruction/query, output/answer} jsonl (or parquet
for orca_math); this script does NOT read pretraining {content} corpora.

Pipeline:
  1. load each source through a field adapter -> (prompt, answer) text pairs
  2. length/quality gate (answer target 300-1500 tokens, hard 2500; prompt max
     2200 chars; nonempty; for code, ast.parse must succeed)
  3. per-source cap and deterministic shuffle (seed), then a global shuffle
  4. 13-gram decontam against HumanEval/MBPP (+math when --extra_math)
  5. render ChatML prompt, keep answer text + <|im_end|>, hand to
     prepare_sft.pack_and_save (prompt-masked, split_encode, SEQ 4096)

The realized mix (per-source kept counts, reject histogram, supervised tokens)
is written beside the pack. Usage (pod, CPU):

  python3 scripts/sft_chatml_pack.py \\
      --spec scripts/sft_chatml_mix.json --out data/sft/v42_sft_v1.pt \\
      --tokenizer data/tokenizer.json --max_answer_tokens 2500
"""
import argparse
import glob
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "filters"))

# char proxy before tokenization: ~3.5 chars/token upper bound incl. code/zh
CHAR_PER_TOK = 3.8
SYSTEM_ZH = "你是一个严谨的解题助手，请逐步推理，并在最后给出明确的最终答案。"
SYSTEM_EN = "You are a careful problem solver. Reason step by step and end with the final answer."


def iter_pairs(path, prompt_field, answer_field, lang):
    """Yield (prompt, answer, lang) from jsonl or parquet."""
    if path.endswith(".parquet"):
        yield from _iter_parquet(path, prompt_field, answer_field, lang)
        return
    op = __import__("gzip").open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = r.get(prompt_field) or r.get("prompt") or r.get("question")
            a = r.get(answer_field) or r.get("output") or r.get("answer")
            if p and a:
                yield str(p), str(a), lang


def _iter_parquet(pattern, pf, af, lang):
    import pyarrow.parquet as pq

    for path in sorted(glob.glob(pattern)):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            d = batch.to_pydict()
            ps = d.get(pf) or d.get("prompt") or d.get("question")
            ans = d.get(af) or d.get("output") or d.get("answer")
            for p, a in zip(ps or [], ans or []):
                if p and a:
                    yield str(p), str(a), lang


def code_ok(answer, lang):
    """For code-bucket sources, require a parseable python block unless lang is free text."""
    if lang != "code":
        return True
    import ast
    import re

    blocks = re.findall(r"```(?:python)?\n(.*?)```", answer, re.S)
    code = blocks[-1].strip() if blocks else answer.strip()
    if len(code) < 40:
        return False
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="json mix spec (see sft_chatml_mix.example.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer.json"))
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--min_answer_tokens", type=int, default=60)
    ap.add_argument("--target_answer_tokens", type=int, default=1500)
    ap.add_argument("--max_answer_tokens", type=int, default=2500)
    ap.add_argument("--max_prompt_chars", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--smoke", type=int, default=0, help="if >0, cap total pairs for a smoke pack")
    ap.add_argument("--no_decontam", action="store_true")
    ap.add_argument("--extra_math", action="store_true")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    from datagen.prepare_sft import pack_and_save
    from loader import format_prompt

    spec = json.load(open(args.spec, encoding="utf-8"))
    tok = Tokenizer.from_file(args.tokenizer)
    eos = tok.token_to_id("<eos>")
    im_end = tok.token_to_id("<|im_end|>")
    assert eos and im_end, f"special tokens missing eos={eos} im_end={im_end}"

    rng = random.Random(args.seed)
    pairs = []  # (prompt, answer_text_with_im_end, bucket, source)
    reject = {}
    per_source = {}
    used_files = []  # concrete source files that yielded pairs, for sources_fp

    def note(src, k):
        reject[f"{src}:{k}"] = reject.get(f"{src}:{k}", 0) + 1

    for s in spec["sources"]:
        src = s["name"]
        bucket = s["bucket"]  # code | math | general
        lang = s.get("lang", "text")  # code | zh | en | text
        cap = int(s.get("cap", 10 ** 12))
        sys_prompt = s.get("system") or (SYSTEM_ZH if lang == "zh" else SYSTEM_EN if lang == "en" else None)
        loaded = []
        hit_files = sorted(glob.glob(os.path.join(ROOT, s["glob"])))
        for path in hit_files:
            for p, a, _l in iter_pairs(path, s.get("prompt", "prompt"), s.get("answer", "output"), lang):
                loaded.append((p, a))
        rng.shuffle(loaded)
        kept_here = 0
        for p, a in loaded:
            if kept_here >= cap:
                note(src, "cap")
                continue
            if len(p) > args.max_prompt_chars or len(p) < 10 or len(a) < 10:
                note(src, "length_prompt")
                continue
            amin = int(args.min_answer_tokens * CHAR_PER_TOK)
            atgt = int(args.target_answer_tokens * CHAR_PER_TOK)
            amax = int(args.max_answer_tokens * CHAR_PER_TOK)
            if not (amin <= len(a) <= amax):
                note(src, "answer_len")
                continue
            if len(a) > atgt:
                note(src, "answer_over_target_kept")  # kept up to hard max
            if bucket == "code" and not code_ok(a, lang):
                note(src, "code_not_parseable")
                continue
            loaded_prompt = format_prompt(p, sys_prompt)
            pairs.append((loaded_prompt, a.rstrip() + "<|im_end|>", bucket, src))
            kept_here += 1
        per_source[src] = per_source.get(src, 0) + kept_here
        if kept_here:
            used_files.extend((f, s.get("prompt", "prompt"), s.get("answer", "output")) for f in hit_files)
        print(f"{src}: loaded={len(loaded)} kept={kept_here}", flush=True)

    # bucket balance: spec states desired weights; truncate over-represented buckets
    weights = spec.get("weights", {})
    if weights:
        by_b = {}
        for item in pairs:
            by_b.setdefault(item[2], []).append(item)
        if args.smoke:
            rng.shuffle(pairs)
            pairs = pairs[: args.smoke]
        else:
            balanced = []
            for b, items in by_b.items():
                w = weights.get(b, 0)
                if not w:
                    continue
                n = int(round(len(pairs) * w / sum(weights.values())))
                rng.shuffle(items)
                balanced.extend(items[:n])
            rng.shuffle(balanced)
            pairs = balanced
    elif args.smoke:
        rng.shuffle(pairs)
        pairs = pairs[: args.smoke]

    # eval-holdout gate (the 305k-hash set sft_math.py cross-checks via holdout_fp)
    from holdout import is_holdout

    n_holdout = 0
    unheld = []
    for p, a, b, s in pairs:
        q = p.split("<|im_start|>user\n", 1)[-1].rsplit("<|im_end|>", 1)[0]
        if is_holdout(q):
            n_holdout += 1
        else:
            unheld.append((p, a, b, s))
    if n_holdout:
        print(f"holdout: excluded {n_holdout} eval-holdout questions", flush=True)
    pairs = unheld

    # 13-gram decontam on the user-visible prompts (ChatML wrapper is constant length)
    dropped = set()
    if not args.no_decontam:
        from decontam_ngram import Decontaminator

        dec = Decontaminator.load_default(root=ROOT, extra_math=args.extra_math)
        kept = []
        for i, (p, a, b, s) in enumerate(pairs):
            # strip the constant ChatML wrapper prefix to scan the real question
            q = p.split("<|im_start|>user\n", 1)[-1].rsplit("<|im_end|>", 1)[0]
            if dec.hit(q) or dec.hit(a):
                dropped.add(i)
            else:
                kept.append((p, a, b, s))
        print(f"decontam: scanned={len(pairs)} dropped={len(dropped)}", flush=True)
        pairs = kept

    examples = [(p, a) for p, a, _b, _s in pairs]
    sources = used_files or None
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pack_and_save(
        examples, tok, eos, args.out, args.seq,
        sources=sources, split_encode=True,
        extra_stats={
            "plan": spec.get("plan", "v42 chatml sft"),
            "format": "chatml_prompt_masked_split_encode",
            "weights": spec.get("weights", {}),
            "per_source_kept": per_source,
            "reject": reject,
            "decontam_dropped": len(dropped),
            "pairs": len(examples),
            "seed": args.seed,
            "smoke": args.smoke,
        },
    )
    print(f"saved {args.out} pairs={len(examples)}")


if __name__ == "__main__":
    main()

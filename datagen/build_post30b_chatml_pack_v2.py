#!/usr/bin/env python3
"""3b-25: SFT pack v2 = the v1 code/en/zh slices plus a ~10% chain-of-thought slice.
# restartable: one-shot CPU pack from on-disk shards; deterministic, idempotent.

Reuses every v1 slice (datagen/build_post30b_chatml_pack.py) unchanged and adds:

  cot  (target 10% of rows): data/corpus/cot_dc/*.jsonl (851k decontaminated math rows,
         question paragraph + solution ending in \\boxed{...}). Selected rows already carry
         explicit numbered steps (`1.` / `Step 1`) with >= 2 steps, so the rendered answer is
         literally "numbered reasoning steps -> final answer" -- no renumbering of prose. The
         final answer is whatever follows the last step (kept verbatim); rows without a
         boxed answer or without >= 2 explicit steps are skipped. Single-turn ChatML via
         scripts/loader.format_example, same masking as every other slice.

v1's three slices are scaled together to 90% of the pack (their internal ratio fixed), so
v2 lands at code/en/zh/cot = 0.9*72 / 0.9*18 / 0.9*10 / 10. v1 .pt is never touched; the
output is a new file. Same mask invariants and a hand-read that shows 10 CoT rows.
"""

import argparse
import glob
import json
import os
import random
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from loader import format_example  # noqa: E402
from prepare_sft import pack_and_save  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import build_post30b_chatml_pack as v1  # noqa: E402
from holdout import is_holdout  # noqa: E402

DATA = os.path.join(ROOT, "data")
COT_GLOB = os.path.join(DATA, "corpus", "cot_dc", "cot_*.jsonl")
SEQ = v1.SEQ
SEED = v1.SEED
COT_SHARE = 0.10
V1_SCALE = 1.0 - COT_SHARE

_STEP = re.compile(r"(?m)^\s*(?:step\s*)?(\d+)[.)]\s+\S")
_BOXED = re.compile(r"\\boxed\s*\{")


def split_qa(content):
    q, sep, ans = content.strip().partition("\n\n")
    if not sep or not q or not ans:
        return None
    return q.strip(), ans.strip()


def numbered_steps(ans):
    nums = [int(m.group(1)) for m in _STEP.finditer(ans)]
    if len(nums) < 2:
        return None
    if not _BOXED.search(ans):
        return None
    return ans


def build_cot(decon):
    pairs, st = [], {"in": 0, "bad_qa": 0, "no_numbered": 0, "no_boxed": 0,
                     "holdout": 0, "decontam": 0, "empty": 0}
    seen = set()
    files = sorted(glob.glob(COT_GLOB))
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                st["in"] += 1
                qa = split_qa(d.get("content") or "")
                if qa is None:
                    st["bad_qa"] += 1
                    continue
                q, ans = qa
                if not q or not ans:
                    st["empty"] += 1
                    continue
                if is_holdout(q):
                    st["holdout"] += 1
                    continue
                if not decon.keeps(ans):
                    st["decontam"] += 1
                    continue
                if not _BOXED.search(ans):
                    st["no_boxed"] += 1
                    continue
                if numbered_steps(ans) is None:
                    st["no_numbered"] += 1
                    continue
                key = hash(q)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(format_example(q, ans))
    return pairs, st


def solve_v2(v1_counts, v1_means, cot_pairs, cot_mean, target_tokens):
    v1_w = sum(V1_SCALE * v1.RATIO[s] * v1_means[s] for s in v1.RATIO)
    cot_w = COT_SHARE * cot_mean
    n_total = int(target_tokens / (v1_w + cot_w))
    counts = {}
    capped = {}
    for s in v1.RATIO:
        want = int(n_total * V1_SCALE * v1.RATIO[s])
        have = v1_counts[s]
        counts[s] = min(want, have)
        capped[s] = want > have
    cot_want = int(n_total * COT_SHARE)
    counts["cot"] = min(cot_want, len(cot_pairs))
    capped["cot"] = cot_want > len(cot_pairs)
    return counts, capped, n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=v1.TOK)
    ap.add_argument("--out", default=os.path.join(DATA, "sft", "sft_v41_chatml_post30b_0912_v2.pt"))
    ap.add_argument("--handread",
                    default=os.path.join(ROOT, "runs", "sft_v41_chatml_handread_40_v2.jsonl"))
    ap.add_argument("--target-tokens", type=float, default=40_000_000)
    ap.add_argument("--stats",
                    default=os.path.join(DATA, "sft", "sft_v41_chatml_post30b_0912_v2.stats.json"))
    args = ap.parse_args()

    random.seed(SEED)
    tok = Tokenizer.from_file(args.tokenizer)
    eos = tok.token_to_id("<eos>")
    print(f"tokenizer vocab {tok.get_vocab_size()} eos {eos}", flush=True)

    sys.path.insert(0, os.path.join(ROOT, "filters"))
    from decontam_ngram import Decontaminator  # noqa: E402

    decon = Decontaminator.load_default(ROOT)

    code, cst = v1.build_code(decon)
    print("code", len(code), cst, flush=True)
    en, est = v1.build_en(decon)
    print("en", len(en), est, flush=True)
    zh, zst = v1.build_zh(decon)
    print("zh", len(zh), zst, flush=True)
    cot, cotst = build_cot(decon)
    print("cot", len(cot), cotst, flush=True)

    pools = {"code": code, "en": en, "zh": zh, "cot": cot}
    means = {s: v1.mean_pair_tokens(tok, pools[s]) for s in pools}
    print("mean pair tokens", {k: round(v, 1) for k, v in means.items()}, flush=True)

    v1_avail = {s: len(pools[s]) for s in v1.RATIO}
    counts, capped, n_total = solve_v2(v1_avail, means, cot, means["cot"], args.target_tokens)
    print("slice counts", counts, "capped", capped, "n_total_est", n_total, flush=True)
    if any(capped.values()):
        print("WARNING one or more slices capped by source supply; ratio is approximate", flush=True)

    tagged = []
    for s in pools:
        rnd = random.Random(SEED + hash(s) % 100000)
        rnd.shuffle(pools[s])
        for pair in pools[s][:counts[s]]:
            tagged.append((s, pair))
    random.shuffle(tagged)
    examples = [pair for _, pair in tagged]

    src_files = [(v1.CODE_SRC, "prompt", "output"), (v1.EN_PARQUET, "instruction/output", "parquet")]
    for zp in sorted(glob.glob(v1.ZH_GLOB)):
        src_files.append((zp, "content", "rendered ChatML"))
    for cp in sorted(glob.glob(COT_GLOB)):
        src_files.append((cp, "content", "cot_dc math numbered-steps"))
    pack_and_save(
        examples, tok, eos, args.out, SEQ,
        sources=src_files,
        extra_stats={
            "seed": SEED, "seq": SEQ, "version": 2,
            "target_row_ratio": {**{s: V1_SCALE * v1.RATIO[s] for s in v1.RATIO}, "cot": COT_SHARE},
            "rows_per_slice": counts, "capped": capped, "mean_pair_tokens": means,
            "code_filter": cst, "en_filter": est, "zh_filter": zst, "cot_filter": cotst,
        },
    )

    rows = v1.mask_invariants(args.out, tok, eos)
    print(f"packed {len(examples)} examples -> {rows} rows at {args.out}", flush=True)

    hr = []
    for s in ("code", "en", "zh", "cot"):
        n_sel = 10 if s == "cot" else 10
        sel = [p for tag, p in tagged if tag == s][:n_sel]
        for prompt, answer in sel:
            hr.append({"slice": s,
                       "prompt": tok.decode(tok.encode(prompt).ids, skip_special_tokens=False),
                       "answer": tok.decode(tok.encode(answer).ids, skip_special_tokens=False)})
    with open(args.handread, "w", encoding="utf-8") as fh:
        for r in hr:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"hand-read {len(hr)} rows (10/slice incl 10 cot) -> {args.handread}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# restartable: assembles finished jsonl parts (cot/code_if/verified) in memory and writes ONE
# pack at the end; an interrupt loses only the CPU tokenize/pack (minutes on the granted cores),
# never fetched bytes -- the expensive steps (fetch + sandbox verification) are separate scripts
# that persist their own per-file/per-solution outputs. Re-running rebuilds the pack deterministically.
"""Assemble + pack the v1 reasoning SFT pack (de, 1e ruling 2026-09-27).

Inputs (prepared by sft_reason_prep.py and sft_verify_code.py):
  data/sft/sft_reason_v1/cot_reason.jsonl        math reasoning (question -> steps+answer)
  data/sft/sft_reason_v1/code_if_short.jsonl     signature -> body short code answers
  data/sft_raw/verified/{taco,apps}_verified.jsonl  sandbox-verified question -> solution

Steps:
  1. Decontaminate the CODE problem text (TACO/APPS questions + code_if prompts) at 13-gram
     against data/eval/humaneval/humaneval_164.jsonl and data/eval/mbpp_holdouts.jsonl, using
     the repo's own filters.decontam_ngram.Decontaminator. cot_dc is math (no code-benchmark
     overlap) and is passed through, still recorded.
  2. Carve a PROBLEM-DISJOINT held-out (~2000 verified TACO/APPS problems, seed 42) into
     data/sft/sft_reason_v1/heldout_code.jsonl -- SFT never trains on it; it is reserved for RL.
  3. Merge the in-SFT parts and hand (prompt, output) pairs to prepare_sft.pack_and_save
     (split_encode, the same raw-continuation packing the existing sft_mixA pack uses).

Mix (1e): cot 40% / verified code 40% / code_if 20%. The verifier yields fewer unique code rows
than the code target, so (1e 2026-09-27) the in-SFT verified code rows repeat --code-repeat 2
times rather than adding a new source; realized unique vs repeated rows and the effective mix
are printed and written to pack_stats.json / the pack manifest, so the reader sees the realized
mix and the duplication rather than the assumed one.

    # pod, CPU only:
    taskset -c 146-179 nice -n 10 OMP_NUM_THREADS=2 python3 scripts/sft_reason_pack.py \\
        --in data/sft/sft_reason_v1 --verified data/sft_raw/verified \\
        --out data/sft/sft_reason_v1/sft_reason_v1.pt --heldout 2000
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "filters"))


def read_jsonl(path):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_decontaminator():
    from decontam_ngram import Decontaminator
    return Decontaminator.load_default(ROOT)


def decontaminate(rows, text_key, dec):
    """Split rows into (clean, dropped, hit_counts). hit_counts maps benchmark -> rows dropped."""
    clean, dropped, hits = [], [], {}
    for r in rows:
        hit = dec.hit(r.get(text_key, "")) if dec is not None else None
        if hit:
            dropped.append(r)
            prob = hit["problem"].split(":", 1)[0]
            hits[prob] = hits.get(prob, 0) + 1
        else:
            clean.append(r)
    return clean, dropped, hits


def build(in_dir, verified_dir, heldout_n, seed, cot_cap, codeif_cap, code_repeat):
    cot = list(read_jsonl(os.path.join(in_dir, "cot_reason.jsonl")))[:cot_cap or None]
    code_if = list(read_jsonl(os.path.join(in_dir, "code_if_short.jsonl")))[:codeif_cap or None]
    verified = []
    for name in ("taco_verified.jsonl", "apps_verified.jsonl"):
        gen = read_jsonl(os.path.join(verified_dir, name))
        if gen is None:
            continue
        for r in gen:
            # verified record carries question + solution (one record per passing solution).
            # problem_qid is shared by that problem's solutions; qid is per-solution.
            verified.append({"prompt": r["question"], "output": r["solution"],
                             "source": f"verified_{r['source']}", "qid": r["qid"],
                             "problem_qid": r.get("problem_qid", r["qid"].split("#")[0])})

    dec = load_decontaminator()
    stats = {"raw": {"cot": len(cot), "code_if": len(code_if), "verified_code": len(verified)}}

    # Decontaminate every CODE problem text. cot passes through (math corpus).
    code_if_c, code_if_d, hit_if = decontaminate(code_if, "prompt", dec)
    ver_c, ver_d, hit_ver = decontaminate(verified, "prompt", dec)
    stats["decontam_dropped"] = {"code_if": len(code_if_d), "verified_code": len(ver_d)}
    stats["decontam_hits_by_benchmark"] = {
        "code_if": hit_if, "verified_code": hit_ver}

    # Problem-disjoint held-out for RL: split by PROBLEM id (all of a problem's solutions move
    # together -- never train one solution and hold out another for the same problem), only from
    # the verified code pool.
    rng = random.Random(seed)
    by_problem = {}
    for r in ver_c:
        by_problem.setdefault(r["problem_qid"], []).append(r)
    pids = sorted(by_problem)
    rng.shuffle(pids)
    hold = set(pids[: min(heldout_n, len(pids))])
    heldout, ver_train = [], []
    for pid in pids:
        (heldout if pid in hold else ver_train).extend(by_problem[pid])
    write_jsonl(os.path.join(in_dir, "heldout_code.jsonl"), heldout)
    stats["heldout_code_problems"] = len(hold)
    stats["heldout_code_solutions"] = len(heldout)

    # 1e ruling 2026-09-27: no new source. The verified code arm is the only execution-checked
    # data and the target is HumanEval, so after the held-out carve its TRAIN rows are repeated
    # code_repeat times in the pack to lift the code share. The held-out is carved first, so no
    # held-out solution is ever repeated into SFT.
    ver_train_rep = ver_train * code_repeat
    parts = {"cot_reason": cot, "verified_code": ver_train_rep, "code_if_short": code_if_c}
    stats["in_sft"] = {k: len(v) for k, v in parts.items()}
    stats["verified_code_unique_rows"] = len(ver_train)
    stats["verified_code_repeat"] = code_repeat
    examples = []
    for rows in parts.values():
        examples.extend((r["prompt"], r["output"]) for r in rows)
    rng.shuffle(examples)
    return examples, parts, stats


def _selftest():
    # held-out split is problem-disjoint: a qid is either train or heldout, never both.
    rows = [{"qid": f"q{i}", "prompt": "x" * 20, "output": "y"} for i in range(50)]
    rng = random.Random(42)
    qids = sorted({r["qid"] for r in rows})
    rng.shuffle(qids)
    hold = set(qids[:10])
    tr = [r for r in rows if r["qid"] not in hold]
    ho = [r for r in rows if r["qid"] in hold]
    assert len(hold) == 10 and not {r["qid"] for r in tr} & {r["qid"] for r in ho}
    assert len(tr) + len(ho) == 50
    print("sft_reason_pack selftest OK: problem-disjoint held-out split")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="in_dir", default="data/sft/sft_reason_v1")
    ap.add_argument("--verified", default="data/sft_raw/verified")
    ap.add_argument("--out", default="data/sft/sft_reason_v1/sft_reason_v1.pt")
    ap.add_argument("--heldout", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--tokenizer", default="data/tokenizer.json",
                    help="tokenizer.json; loaded directly so packing does not import train/torch.compile")
    ap.add_argument("--body-gate-tokens", type=int, default=256)
    ap.add_argument("--cot-cap", type=int, default=48000)
    ap.add_argument("--codeif-cap", type=int, default=24000)
    ap.add_argument("--code-repeat", type=int, default=2,
                    help="repeat the in-SFT verified code rows N times (1e 2026-09-27: no new "
                         "source; code is the only execution-checked arm, repeat 2x to hit mix)")
    ap.add_argument("--dry", action="store_true", help="assemble + stats only, no token pack")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()

    examples, parts, stats = build(a.in_dir, a.verified, a.heldout, a.seed,
                                   a.cot_cap, a.codeif_cap, a.code_repeat)
    total = len(examples)
    stats["total_examples"] = total
    stats["realized_mix_pct"] = {k: round(100 * len(v) / max(total, 1), 1)
                                 for k, v in parts.items()}
    print(json.dumps(stats, indent=2))
    with open(os.path.join(a.in_dir, "pack_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    if a.dry:
        return 0

    # Load the tokenizer directly (like sfta_build_pack.py): importing train pulls torch.compile /
    # dynamo, which is unused for packing and errors on a bare CPU invocation.
    from datagen.prepare_sft import pack_and_save  # noqa: I001 (1st-party vs tokenizers import split)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    eos = tok.token_to_id("<eos>")
    assert eos is not None, "tokenizer has no <eos>"
    sources = [(os.path.join(a.in_dir, n), "prompt", "output")
               for n in ("cot_reason.jsonl", "code_if_short.jsonl")]
    pack_and_save(examples, tok, eos, a.out, a.seq, split_encode=True,
                  sources=sources, extra_stats={"plan": "sft reasoning v1 (1e 2026-09-27)",
                                                "mix": stats["realized_mix_pct"],
                                                "verified_code_repeat": a.code_repeat,
                                                "verified_code_unique_rows":
                                                    stats["verified_code_unique_rows"]})
    print("packed ->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

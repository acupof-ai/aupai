#!/usr/bin/env python3
"""3b-24 (round 3, phi-1 route): CodeExercises analogue from the L3 stub domain.
# restartable: one-shot CPU build from the frozen stub shards; the 2% hash split and rank
# order are deterministic, so an interrupt only loses in-memory tokenization and a rerun
# reproduces the same manifest, columns and pack bytes.

phi-1's CodeExercises are "a docstring of a function that needs to be completed";
the model is taught to continue a `def signature` + docstring with the body. That
is the HumanEval continuation prompt exactly, so the base eval reads the SFT
checkpoint with no format adapter (unlike the post-30B ChatML pack).

Holdout FIRST. 2% of code_ultra_l3_stub_dc is held by document hash,
deterministically, BEFORE any SFT use, and the held urls+hashes are written to a
manifest the round-3 pretrain cache build excludes (coordinated with 98/0e). The
split key is sha1(_norm(content)) mod 100 < 2: _norm is whitespace-insensitive,
so it does not move on a byte-only reflow.

Two disjoint columns, both RAW continuation (no ChatML), prompt-masked, EOS
supervised, packed with split_encode=True so the masked prompt is a token-by-token
prefix of what the eval runner encodes:
  primary  -- held docs whose docstring carries no >>> doctest; deterministic
              rank order, taken until ~40M supervised body tokens;
  doctest  -- every held doc whose docstring contains '>>>' (~2.0M), a second
              column so the doctest answer format is measured on its own.

prompt = `def sig(...):\n    \"\"\"docstring\"\"\"\n` (ast.unparse of def+docstring),
target = the 4-space-indented body, EOS appended by the packer. 13-gram
decontamination against HumanEval/MBPP runs on the rebuilt function (the source
domain is already _dc, so drops are expected ~0 but the gate is run on the pack
as built, not assumed).
"""

import argparse
import ast
import copy
import glob
import hashlib
import json
import os
import random
import sys
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from tokenizers import Tokenizer  # noqa: E402

from gen_exercises import _norm  # noqa: E402
from holdout import is_holdout  # noqa: E402
from prepare_sft import pack_and_save  # noqa: E402

DATA = os.path.join(ROOT, "data")
DOMAIN = os.path.join(DATA, "corpus", "code_ultra_l3_stub_dc")
SHARD_GLOB = os.path.join(DOMAIN, "code_ultra_l3_stub_dc_[0-9]*.jsonl")
TOK = os.path.join(DATA, "tokenizer.json")
SFT_DIR = os.path.join(DATA, "sft")
MANIFEST = os.path.join(SFT_DIR, "phi_l3_stub_holdout_manifest.jsonl")
OUT_PRIMARY = os.path.join(SFT_DIR, "sft_phi_codeexercises_0913.pt")
OUT_DOCTEST = os.path.join(SFT_DIR, "sft_phi_codeexercises_doctest_0913.pt")
HANDREAD = os.path.join(ROOT, "runs", "sft_phi_codeexercises_handread.jsonl")
SEQ = 4096  # logical context; pack_and_save emits seq+1-token rows, like prepare_format_sft
SEED = 20260913
HOLD_PCT = 2
PRIMARY_TARGET_TOK = 40_000_000


def hold_pick(normed):
    return int(hashlib.sha1(normed.encode("utf-8")).hexdigest()[:12], 16) % 100 < HOLD_PCT


def rank(normed, col):
    h = hashlib.sha1((col + "|" + normed).encode("utf-8")).hexdigest()
    return int(h[:14], 16) / float(16 ** 14)


def split_function(src):
    """-> (prompt, body, docstring) in HumanEval/phi shape, or None.

    prompt is ast.unparse(def + signature + docstring-only body) + newline; body
    is the remaining statements, each indented four spaces. Decorators are
    dropped (HumanEval entry points carry none)."""
    tree = ast.parse(src)
    fn = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if fn is None or ast.get_docstring(fn, clean=False) is None:
        return None
    real = fn.body[1:]
    if not real:
        return None
    head = copy.deepcopy(fn)
    head.body = [fn.body[0]]
    head.decorator_list = []
    prompt = ast.unparse(head) + "\n"
    body = "\n".join("    " + ln for stmt in real for ln in ast.unparse(stmt).splitlines()) + "\n"
    return prompt, body, ast.get_docstring(fn, clean=False)


def _shard(path):
    """Parse+classify one shard. Holdout decision and splitting only; no tokenizer."""
    out = {"held": 0, "no_func_or_doc": 0, "rows": []}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            content = rec["content"]
            normed = _norm(content)
            if not hold_pick(normed):
                continue
            out["held"] += 1
            sp = split_function(content)
            if sp is None:
                out["no_func_or_doc"] += 1
                out["rows"].append({"url": rec.get("url"), "h": _doc_hash(content),
                                    "col": "unusable"})
                continue
            prompt, body, doc = sp
            col = "doctest" if ">>>" in doc else "primary"
            out["rows"].append({"url": rec.get("url"), "h": _doc_hash(content), "col": col,
                                "p": prompt, "b": body})
    return out


def _doc_hash(content):
    return hashlib.sha1(_norm(content).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--primary_target", type=int, default=PRIMARY_TARGET_TOK)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--out_primary", default=OUT_PRIMARY)
    ap.add_argument("--out_doctest", default=OUT_DOCTEST)
    ap.add_argument("--handread", default=HANDREAD)
    args = ap.parse_args()

    tok = Tokenizer.from_file(TOK)
    eos = tok.token_to_id("<eos>")
    print(f"tokenizer vocab {tok.get_vocab_size()} eos {eos}", flush=True)

    sys.path.insert(0, os.path.join(ROOT, "filters"))
    from decontam_ngram import Decontaminator  # noqa: E402
    decon = Decontaminator.load_default(ROOT)

    files = sorted(glob.glob(SHARD_GLOB))
    print(f"[scan] {len(files)} stub shards, holding {HOLD_PCT}% by hash ...", flush=True)
    rows, held, unusable = [], 0, 0
    with Pool(args.workers) as pool:
        for i, st in enumerate(pool.imap_unordered(_shard, files), 1):
            held += st["held"]
            unusable += st["no_func_or_doc"]
            rows.extend(st["rows"])
            if i % 40 == 0:
                print(f"  {i}/{len(files)} held={held}", flush=True)

    usable = [r for r in rows if "p" in r]
    print(f"[split] held={held} usable={len(usable)} unusable={unusable}", flush=True)

    # Tokenize the held usable set once: length filter (pack drops >seq), body
    # supervised tokens, and deterministic rank-based primary selection.
    def encode(r):
        ep = tok.encode(r["p"]).ids
        eb = tok.encode(r["b"]).ids
        r["plen"], r["btok"] = len(ep), len(eb)
        return r

    doctest, primary_pool = [], []
    for r in usable:
        (doctest if r["col"] == "doctest" else primary_pool).append(encode(r))
    for r in primary_pool:
        r["rank"] = rank(r["h"], "primary")

    def fits(r):
        return r["plen"] + r["btok"] + 1 <= SEQ + 1

    doctest = [r for r in doctest if fits(r)]
    primary_pool = sorted((r for r in primary_pool if fits(r)), key=lambda r: r["rank"])

    # Holdout + 13-gram gate, applied to the rebuilt function (prompt+body).
    def gate(rows, label):
        kept, decon_drop, hold_drop = [], 0, 0
        for r in rows:
            full = r["p"] + r["b"]
            if is_holdout(r["p"]):
                hold_drop += 1
                continue
            if decon.hit(full) is not None:
                decon_drop += 1
                continue
            kept.append(r)
        print(f"[gate] {label}: kept={len(kept)} decon_drop={decon_drop} holdout_drop={hold_drop}",
              flush=True)
        return kept, {"kept": len(kept), "decon_drop": decon_drop, "holdout_drop": hold_drop}

    doctest, dg = gate(doctest, "doctest")

    primary, sup, pg = [], 0, None
    for r in primary_pool:
        if sup >= args.primary_target:
            break
        primary.append(r)
        sup += r["btok"]
    primary, pg = gate(primary, "primary")

    os.makedirs(SFT_DIR, exist_ok=True)
    # Full held manifest (incl unusable) is the pretrain exclusion contract.
    chosen_h = {r["h"] for r in primary} | {r["h"] for r in doctest}
    doctest_h = {r["h"] for r in doctest}
    primary_h = {r["h"] for r in primary}
    with open(args.manifest, "w", encoding="utf-8") as fh:
        for r in rows:
            if r["h"] in primary_h:
                col = "primary"
            elif r["h"] in doctest_h:
                col = "doctest"
            elif "p" in r:
                col = "dropped"
            else:
                col = "unusable"
            fh.write(json.dumps({"url": r.get("url"), "content_sha1_norm": r["h"],
                                 "split": "holdout", "column": col,
                                 "in_sft_pack": r["h"] in chosen_h}, ensure_ascii=False) + "\n")
    print(f"[manifest] {len(rows)} held docs -> {args.manifest}", flush=True)

    sources = [(p, "content", "l3_stub holdout 2%") for p in files]

    def emit(col_rows, path, tag):
        pairs = [(r["p"], r["b"]) for r in col_rows]
        sup_tok = sum(r["btok"] for r in col_rows)
        random.Random(SEED).shuffle(pairs)
        pack_and_save(pairs, tok, eos, path, SEQ, sources=sources, split_encode=True,
                      extra_stats={"seed": SEED, "seq": SEQ, "column": tag,
                                   "examples": len(pairs), "supervised_body_tokens": sup_tok,
                                   "split_rule": f"sha1(_norm(content)) mod100 < {HOLD_PCT}"})
        print(f"[pack] {tag}: {len(pairs)} examples, {sup_tok} supervised body tokens -> {path}",
              flush=True)
        return len(pairs), sup_tok

    n_prim, sup_prim = emit(primary, args.out_primary, "primary")
    n_doc, sup_doc = emit(doctest, args.out_doctest, "doctest")

    hr = []
    for tag, col_rows in (("primary", primary), ("doctest", doctest)):
        for r in col_rows[:10]:
            hr.append({"column": tag,
                       "prompt": tok.decode(tok.encode(r["p"]).ids, skip_special_tokens=False),
                       "answer": tok.decode(tok.encode(r["b"]).ids, skip_special_tokens=False)})
    with open(args.handread, "w", encoding="utf-8") as fh:
        for rec in hr:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[handread] {len(hr)} rows -> {args.handread}", flush=True)

    print("PHI_PACK_STATS " + json.dumps({
        "held": held, "usable": len(usable), "unusable": unusable,
        "primary_examples": n_prim, "primary_supervised_tokens": sup_prim,
        "doctest_examples": n_doc, "doctest_supervised_tokens": sup_doc,
        "total_supervised_tokens": sup_prim + sup_doc,
        "gates": {"primary": pg, "doctest": dg},
    }), flush=True)


if __name__ == "__main__":
    main()

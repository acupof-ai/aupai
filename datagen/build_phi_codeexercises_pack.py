#!/usr/bin/env python3
"""3b-24 (round 3, phi-1 route): CodeExercises analogue from the L3 stub domain.

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

The scan streams one part file per source shard (parts/<shard>.jsonl, one write
per held doc); a rerun skips a shard whose part exists, so an interrupt loses at
most one shard and never re-buffers the full held set in RAM.
"""

import argparse
import ast
import copy
import glob
import hashlib
import json
import os
import random
import shutil
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
PARTS_DIR = os.path.join(SFT_DIR, "phi_l3_stub_holdout_parts")
MANIFEST = os.path.join(SFT_DIR, "phi_l3_stub_holdout_manifest.jsonl")
OUT_PRIMARY = os.path.join(SFT_DIR, "sft_phi_codeexercises_0913.pt")
OUT_DOCTEST = os.path.join(SFT_DIR, "sft_phi_codeexercises_doctest_0913.pt")
HANDREAD = os.path.join(ROOT, "runs", "sft_phi_codeexercises_handread.jsonl")
SEQ = 4096
SEED = 20260913
HOLD_PCT = 2
PRIMARY_TARGET_TOK = 40_000_000


def hold_pick(normed):
    return int(hashlib.sha1(normed.encode("utf-8")).hexdigest()[:12], 16) % 100 < HOLD_PCT


def rank(normed, col):
    h = hashlib.sha1((col + "|" + normed).encode("utf-8")).hexdigest()
    return int(h[:14], 16) / float(16 ** 14)


def doc_hash(content):
    return hashlib.sha1(_norm(content).encode("utf-8")).hexdigest()


def split_function(src):
    """-> (prompt, body, docstring) in HumanEval/phi shape, or None.

    prompt is ast.unparse(def + signature + docstring-only body) + newline; body
    is the remaining statements, each indented four spaces. Decorators are
    dropped (HumanEval entry points carry none)."""
    tree = ast.parse(src)
    fn = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if fn is None or ast.get_docstring(fn, clean=False) is None:
        return None
    if not fn.body[1:]:
        return None
    head = copy.deepcopy(fn)
    head.body = [fn.body[0]]
    head.decorator_list = []
    prompt = ast.unparse(head) + "\n"
    body = "\n".join("    " + ln for stmt in fn.body[1:] for ln in ast.unparse(stmt).splitlines()) + "\n"
    return prompt, body, ast.get_docstring(fn, clean=False)


def _shard(args):
    """Stream one shard's held records to parts/<shard>.jsonl (one write per doc).

    Returns counts only. An existing part means the shard is done, so a rerun
    skips it (its counts are re-derived by reading the part)."""
    path, parts_dir, force = args
    name = os.path.basename(path).replace(".jsonl", ".held.jsonl")
    part = os.path.join(parts_dir, name)
    if os.path.exists(part) and not force:
        return _count_part(part) | {"rescanned": False}
    held = unusable = 0
    with open(path, encoding="utf-8") as src, open(part, "w", encoding="utf-8") as out:
        for line in src:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            content = rec["content"]
            if not hold_pick(_norm(content)):
                continue
            held += 1
            sp = split_function(content)
            if sp is None:
                unusable += 1
                rec_out = {"url": rec.get("url"), "h": doc_hash(content), "col": "unusable"}
            else:
                prompt, body, doc = sp
                rec_out = {"url": rec.get("url"), "h": doc_hash(content),
                           "col": "doctest" if ">>>" in doc else "primary",
                           "p": prompt, "b": body}
            out.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
    return {"held": held, "unusable": unusable, "rescanned": True}


def _count_part(part):
    held = unusable = 0
    for line in open(part, encoding="utf-8"):
        held += 1
        if json.loads(line)["col"] == "unusable":
            unusable += 1
    return {"held": held, "unusable": unusable}


def _iter_parts():
    for part in sorted(glob.glob(os.path.join(PARTS_DIR, "*.held.jsonl"))):
        for line in open(part, encoding="utf-8"):
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--primary_target", type=int, default=PRIMARY_TARGET_TOK)
    ap.add_argument("--force_rescan", action="store_true")
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
    os.makedirs(PARTS_DIR, exist_ok=True)
    print(f"[scan] {len(files)} stub shards, holding {HOLD_PCT}% by hash, streaming parts ...",
          flush=True)
    held = unusable = 0
    with Pool(args.workers) as pool:
        tasks = [(p, PARTS_DIR, args.force_rescan) for p in files]
        for i, st in enumerate(pool.imap_unordered(_shard, tasks), 1):
            held += st["held"]
            unusable += st["unusable"]
            if i % 40 == 0:
                print(f"  {i}/{len(files)} held={held}", flush=True)

    def gate(prompt, body):
        full = prompt + body
        if is_holdout(prompt):
            return "holdout"
        if decon.hit(full) is not None:
            return "decon"
        return None

    doctest_tmp = os.path.join(PARTS_DIR, "column_doctest.jsonl")
    primary_tmp = os.path.join(PARTS_DIR, "column_primary.jsonl")
    drops = {"primary": {"decon": 0, "holdout": 0, "toolong": 0},
             "doctest": {"decon": 0, "holdout": 0, "toolong": 0}}
    with open(primary_tmp, "w", encoding="utf-8") as pf, \
            open(doctest_tmp, "w", encoding="utf-8") as dfh:
        for r in _iter_parts():
            if r["col"] == "unusable":
                continue
            ep = tok.encode(r["p"]).ids
            eb = tok.encode(r["b"]).ids
            if len(ep) + len(eb) + 1 > SEQ + 1:
                drops[r["col"]]["toolong"] += 1
                continue
            why = gate(r["p"], r["b"])
            if why:
                drops[r["col"]][why] += 1
                continue
            rec = {"h": r["h"], "p": r["p"], "b": r["b"], "plen": len(ep), "btok": len(eb)}
            if r["col"] == "doctest":
                json.dump(rec, dfh, ensure_ascii=False)
                dfh.write("\n")
            else:
                rec["rank"] = rank(r["h"], "primary")
                json.dump(rec, pf, ensure_ascii=False)
                pf.write("\n")

    doctest_rows = sorted((json.loads(l) for l in open(doctest_tmp, encoding="utf-8")),
                          key=lambda r: r["h"])
    primary_rows = sorted((json.loads(l) for l in open(primary_tmp, encoding="utf-8")),
                          key=lambda r: r["rank"])
    chosen, sup = [], 0
    for r in primary_rows:
        if sup >= args.primary_target:
            break
        chosen.append(r)
        sup += r["btok"]
    primary_rows = chosen
    print(f"[select] primary={len(primary_rows)} ({sum(r['btok'] for r in primary_rows)} tok); "
          f"doctest={len(doctest_rows)} ({sum(r['btok'] for r in doctest_rows)} tok); drops={drops}",
          flush=True)

    primary_h = {r["h"] for r in primary_rows}
    doctest_h = {r["h"] for r in doctest_rows}

    os.makedirs(SFT_DIR, exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as fh:
        for r in _iter_parts():
            h = r["h"]
            if r["col"] == "unusable":
                col, used = "unusable", False
            elif h in primary_h:
                col, used = "primary", True
            elif h in doctest_h:
                col, used = "doctest", True
            else:
                col, used = "dropped", False
            fh.write(json.dumps({"url": r.get("url"), "content_sha1_norm": h, "split": "holdout",
                                 "column": col, "in_sft_pack": used}, ensure_ascii=False) + "\n")
    print(f"[manifest] {held} held docs -> {args.manifest}", flush=True)

    sources = [(p, "content", "l3_stub holdout 2%") for p in files]

    def emit(col_rows, path, tag):
        pairs = [(r["p"], r["b"]) for r in col_rows]
        sup_tok = sum(r["btok"] for r in col_rows)
        random.Random(SEED).shuffle(pairs)
        pack_and_save(pairs, tok, eos, path, SEQ, sources=sources, split_encode=True,
                      extra_stats={"seed": SEED, "seq": SEQ, "column": tag,
                                   "examples": len(pairs), "supervised_body_tokens": sup_tok,
                                   # The PACK membership gate is the manifest COLUMN, not the
                                   # source-holdout hash split. in_sft_pack is True exactly for
                                   # column in {primary, doctest} (0 exceptions over the manifest);
                                   # column 'dropped' (decon/holdout/toolong fail or primary target
                                   # cap) and 'unusable' are excluded. Verified 2026-09-14 against
                                   # phi_l3_stub_holdout_manifest.jsonl. The sha mod100 rule below
                                   # is a DIFFERENT split: which stub-domain docs formed the 2%
                                   # holdout SOURCE before any pack gate. Do not read it as the
                                   # pack gate -- applying it to the manifest misclassifies rows.
                                   "pack_gate": "manifest column in {primary, doctest}; "
                                                "in_sft_pack == (column not in {dropped, unusable})",
                                   "source_holdout_rule": f"sha1(_norm(content)) mod100 < {HOLD_PCT} "
                                                          "(selects the 2% held SOURCE slice, "
                                                          "not pack membership)"})
        print(f"[pack] {tag}: {len(pairs)} examples, {sup_tok} supervised body tokens -> {path}",
              flush=True)
        return len(pairs), sup_tok

    n_prim, sup_prim = emit(primary_rows, args.out_primary, "primary")
    n_doc, sup_doc = emit(doctest_rows, args.out_doctest, "doctest")

    hr = []
    for tag, col_rows in (("primary", primary_rows), ("doctest", doctest_rows)):
        for r in col_rows[:10]:
            hr.append({"column": tag,
                       "prompt": tok.decode(tok.encode(r["p"]).ids, skip_special_tokens=False),
                       "answer": tok.decode(tok.encode(r["b"]).ids, skip_special_tokens=False)})
    with open(args.handread, "w", encoding="utf-8") as fh:
        for rec in hr:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[handread] {len(hr)} rows -> {args.handread}", flush=True)

    shutil.rmtree(PARTS_DIR, ignore_errors=True)
    print("PHI_PACK_STATS " + json.dumps({
        "held": held, "usable": held - unusable, "unusable": unusable,
        "primary_examples": n_prim, "primary_supervised_tokens": sup_prim,
        "doctest_examples": n_doc, "doctest_supervised_tokens": sup_doc,
        "total_supervised_tokens": sup_prim + sup_doc, "drops": drops,
    }), flush=True)


if __name__ == "__main__":
    main()

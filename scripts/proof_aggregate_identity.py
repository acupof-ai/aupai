#!/usr/bin/env python3
"""Gate for the sidecar aggregate phase 2 (fb ruling 2026-09-11).

On two real converted shards, run the OLD serial dedup (json parse + _norm +
sha1 per line) and the NEW sidecar path (precomputed sha1 .sigs + raw line
copy) through aggregate(), then assert the emitted final shard files are
byte-identical and every stats-relevant number matches. The sidecar driver is
accepted for the production aggregate only when this prints PROOF_BYTE_IDENTICAL.

Paths are env-overridable so the same script runs against the checkout
(AUPAI_ROOT, default /work/aupai) with either the live module or a staging copy
(AUPAI_NEW, AUPAI_OLD). The OLD driver is the pre-sidecar aggregate checked in
at git history; AUPAI_OLD points at a copy placed one package-deep so its
root computation (dirname(dirname(__file__))) resolves data/eval correctly.
"""
import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.environ.get("AUPAI_ROOT", "/work/aupai")
SRC = os.environ.get("AUPAI_PROOF_SRC", f"{ROOT}/data/corpus/code_ultra_l3_noexec")
NEW = os.environ.get("AUPAI_NEW", f"{ROOT}/datagen/ultradata_shards.py")
OLD = os.environ.get("AUPAI_OLD", f"{ROOT}/datagen/ultradata_shards_pre_sidecar.py")
TOK = f"{ROOT}/data/tokenizer.json"
PREFIX = "code_ultra_l3_noexec"
SHARDS = os.environ.get("AUPAI_PROOF_SHARDS", "s001,s002").split(",")


def build_case(td):
    d = os.path.join(td, PREFIX)
    os.makedirs(d, exist_ok=True)
    k = kt = kc = i = 0
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK)
    for s in SHARDS:
        for f in sorted(glob.glob(f"{SRC}/{PREFIX}_{s}_*.jsonl")):
            if f.endswith((".clean", ".ngdrop", ".sigs")):
                continue
            dst = os.path.join(d, f"{PREFIX}_g00_{i:03d}.jsonl")
            lines = open(f, encoding="utf-8").readlines()
            open(dst, "w", encoding="utf-8").writelines(lines)
            for line in lines:
                c = json.loads(line)["content"]
                k += 1
                kt += len(tok.encode(c).ids) + 1
                kc += len(c)
            i += 1
    stats = json.load(open(f"{SRC}/stats_s001.json"))
    stats.update(kept=k, kept_tokens=kt, kept_chars=kc, total_rows=k, n_shards=0)
    stats["reasons"] = {key: 0 for key in stats["reasons"]}
    stats["reasons"]["kept"] = k
    json.dump(stats, open(os.path.join(d, "stats_g00.json"), "w"))
    return k


def run(driver, td):
    out = os.path.join(td, PREFIX)
    fin = os.path.join(td, "dc")
    pkg = os.path.dirname(os.path.dirname(driver))
    env = dict(os.environ, PYTHONPATH=pkg)
    r = subprocess.run([sys.executable, driver, "--level", "L3",
                        "--aggregate", "stats_g*.json", "--out", out,
                        "--final-out", fin, "--tokenizer", TOK,
                        "--agg-workers", "4"],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit(f"{driver} failed rc={r.returncode}")
    return fin


def main():
    base = "/tmp/proof_identity"
    shutil.rmtree(base, ignore_errors=True)
    ot, nt = os.path.join(base, "O"), os.path.join(base, "N")
    os.makedirs(ot)
    os.makedirs(nt)
    rows = build_case(ot)
    shutil.copytree(os.path.join(ot, PREFIX), os.path.join(nt, PREFIX), dirs_exist_ok=True)
    print("proof shards", SHARDS, "rows", rows, flush=True)
    fo, fn = run(OLD, ot), run(NEW, nt)
    so = json.load(open(os.path.join(fo, "build_corpus_stats.json")))
    sn = json.load(open(os.path.join(fn, "build_corpus_stats.json")))
    os.remove(os.path.join(fo, "build_corpus_stats.json"))
    os.remove(os.path.join(fn, "build_corpus_stats.json"))
    o_files, n_files = sorted(os.listdir(fo)), sorted(os.listdir(fn))
    assert o_files == n_files, (o_files, n_files)
    for name in o_files:
        a = open(os.path.join(fo, name), "rb").read()
        b = open(os.path.join(fn, name), "rb").read()
        assert a == b, f"final file differs {name} {len(a)} {len(b)}"
    for key in ("kept", "kept_chars", "kept_tokens", "n_shards", "fingerprint"):
        assert so[key] == sn[key], (key, so[key], sn[key])
    assert so["decontam_ngram"]["rows_dropped"] == sn["decontam_ngram"]["rows_dropped"]
    assert so["reasons"]["cross_group_dup"] == sn["reasons"]["cross_group_dup"]
    print("PROOF_BYTE_IDENTICAL kept", sn["kept"], "chars", sn["kept_chars"],
          "tok", sn["kept_tokens"], "shards", sn["n_shards"],
          "ng", sn["decontam_ngram"]["rows_dropped"],
          "xdup", sn["reasons"]["cross_group_dup"], "fp", sn["fingerprint"])


if __name__ == "__main__":
    main()

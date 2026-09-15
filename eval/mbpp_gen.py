"""MBPP sanitized (427) base-model continuation scorer.

Prompt shape (the in-distribution base continuation, same finding that made
HumanEval use --rstrip_nl): the sanitized "prompt" is a one-line task
description with NO signature; the required name lives only in test_list and
the canonical code. The prompt is therefore the canonical def signature line
plus the task text as a closed, indented docstring, trailing newline stripped
so the model emits its own newline+indent tokens.

Pass = prompt + completion + test_imports + test_list execs cleanly (in-process,
6s SIGALRM ceiling). Canonical control: every problem's canonical body must pass
its own tests under this exact prompt construction (run CONTROL=ALL).

Sampling: --n/--temperature mirror eval/humaneval_gen.py and share
eval/sampling.py's task-seeded RNG, so T and C checkpoints draw identical
choices per task_id (paired stage-2 protocol). CPU or CUDA.

CLEAN column: by default the r3 six-domain contamination union
(runs/contam_r3_mbpp_union.json#r3_mbpp_clean, 338 of 427, PR #344) is loaded
and every run prints BOTH FULL/427 and CLEAN/338; with n>1 the CLEAN denominator
is clean tasks times n, keeping the paired per-task unit. --no_clean disables.

  python3 eval/mbpp_gen.py --data data/eval/sanitized-mbpp.json \
      --ckpt <ckpt> --device cpu --n 10 --temperature 0.2 \
      --run eT_mbpp_n10temp02
"""
import argparse
import contextlib
import io
import json
import os
import re
import signal
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from eval.shard import label as shard_label  # noqa: E402
from eval.shard import select as shard_select
from eval.shard import validate as shard_validate

DATA_PATH = os.path.join(ROOT, "data", "eval", "sanitized-mbpp.json")
TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
# r3 six-domain 13-gram contamination union; its r3_mbpp_clean list (338/427,
# PR #344) is the CLEAN denominator reported alongside FULL.
CLEAN_PATH = os.path.join(ROOT, "runs", "contam_r3_mbpp_union.json")
# Other column-0 top-level constructs end the completion; a self re-declaration of
# the entry def is kept (later def wins), matching humaneval_gen's truncate.
OTHER_STOPS = ["\nclass ", "\nif __name__", "\nprint(", "\n#", "\n@", '\nassert ']


def _clean_id(s):
    """'mbpp427:101' -> 101 (sanitized task_id is the bare int); None if not that shape."""
    m = re.fullmatch(r"mbpp427:(\d+)", str(s))
    return int(m.group(1)) if m else None


def _id_list(union, key):
    v = union.get(key)
    if not isinstance(v, list) or not v:
        raise RuntimeError(f"contam manifest key {key!r} missing or not a non-empty list")
    out = set()
    for s in v:
        i = _clean_id(s)
        if i is None:
            raise RuntimeError(f"manifest id {s!r} in {key!r} is not mbpp427:<num>")
        out.add(i)
    return out


def clean_consistency(union, data_ids):
    """clean == scored dataset - union as int sets; returns the clean set.

    The manifest carries only the union (89) and clean (338) id LISTS; the full
    427 set is the scored dataset itself. Recorded counts (_n ints) are checked
    too. Raises on absent keys, a malformed id, or a clean set that is not the
    exact complement, so a stale manifest cannot silently rename the denominator.
    """
    data_ids = set(data_ids)
    union_ids = _id_list(union, "r3_mbpp_union")
    clean_ids = _id_list(union, "r3_mbpp_clean")
    expect = data_ids - union_ids
    if clean_ids != expect:
        raise RuntimeError(
            f"r3_mbpp_clean ({len(clean_ids)}) != dataset {len(data_ids)} - union "
            f"{len(union_ids)} = {len(expect)} -- recompute the manifest")
    recorded = union.get("r3_mbpp_clean_n")
    if isinstance(recorded, int) and recorded != len(clean_ids):
        raise RuntimeError(f"r3_mbpp_clean_n={recorded} != {len(clean_ids)} listed ids")
    recorded_all = union.get("mbpp427_all_n")
    if isinstance(recorded_all, int) and recorded_all != len(data_ids):
        raise RuntimeError(f"mbpp427_all_n={recorded_all} != {len(data_ids)} scored rows")
    return clean_ids


def load_clean_ids(data_ids, path=CLEAN_PATH):
    """The CLEAN int task_id set from a contamination manifest, checked against data.

    Refuses a missing manifest/key and any clean id outside the scored dataset,
    so a wrong manifest cannot shrink or inflate the CLEAN denominator quietly.
    """
    if not os.path.exists(path):
        raise RuntimeError(
            f"CLEAN manifest not found: {path}. Pass --no_clean for FULL-only or "
            "supply --clean <runs/contam_r3_mbpp_union.json>")
    with open(path, encoding="utf-8") as fh:
        return clean_consistency(json.load(fh), data_ids)


class TO(Exception):
    pass


def _h(*_a):
    raise TO()


signal.signal(signal.SIGALRM, _h)


def signature(rec):
    line = next(l for l in rec["code"].splitlines() if re.match(r"def\s+\w", l))
    assert line.rstrip().endswith(":"), (rec.get("task_id"), line)
    return line.rstrip(), re.match(r"def\s+(\w+)", line).group(1)


def preamble(rec):
    lines = rec["code"].splitlines()
    i = next(k for k, l in enumerate(lines) if re.match(r"def\s+\w", l))
    return "\n".join(lines[:i])


def body_indent(rec):
    for l in rec["code"].splitlines():
        m = re.match(r"(\s+)\S", l)
        if m:
            return m.group(1)
    return "    "


def model_prompt(rec):
    pre = preamble(rec)
    sig, _ = signature(rec)
    ind = body_indent(rec)
    head = sig + "\n" + ind + '"""' + rec["prompt"] + "\n" + ind + '"""'
    return (pre + ("\n" if pre else "") + head).rstrip("\n")


def canonical_body(rec):
    lines = rec["code"].splitlines()
    i = next(k for k, l in enumerate(lines) if re.match(r"def\s+\w", l))
    return "\n".join(lines[i + 1:])


def judge(rec, completion):
    src = (model_prompt(rec) + completion + "\n"
           + "\n".join(rec.get("test_imports", [])) + "\n"
           + "\n".join(rec["test_list"]) + "\n")
    g = {"__name__": "__main__"}
    signal.alarm(6)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(src, g)
        return True
    except BaseException:
        return False
    finally:
        signal.alarm(0)


def truncate(raw, entry):
    cut = len(raw)
    for st in OTHER_STOPS + ["\ndef "]:
        start = 0
        while True:
            i = raw.find(st, start)
            if i == -1:
                break
            if st == "\ndef " and raw[i:i + 12 + len(entry)].startswith(f"\ndef {entry}("):
                start = i + 1
                continue
            cut = min(cut, i)
            break
    return raw[:cut]


def main():
    if "--selftest" in sys.argv:
        return _selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--max_new", type=int, default=280)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--clean", default=CLEAN_PATH,
                    help="contam manifest carrying r3_mbpp_union/r3_mbpp_clean")
    ap.add_argument("--no_clean", action="store_true", help="FULL/427 only, no CLEAN column")
    ap.add_argument("--shard_i", type=int, default=None,
                    help="multi-card shard: score only fixed-order indices i with "
                         "i %% --shard_n == shard_i. Use --shard_n; with sharding pass "
                         "--no_clean (a shard is not the full 427, so the clean-complement "
                         "check cannot run); the merger recomputes FULL/427 and CLEAN/338.")
    ap.add_argument("--shard_n", type=int, default=None)
    ap.add_argument("--control", choices=["20", "ALL"], default=None,
                    help="judge canonical solutions, no model; ALL must pass")
    args = ap.parse_args()

    recs_all = json.load(open(args.data, encoding="utf-8"))
    shard_validate(args.shard_i, args.shard_n)
    if args.shard_n is not None and not args.no_clean:
        ap.error("sharded MBPP runs must pass --no_clean: a shard is not the full 427, so "
                 "the clean-complement invariant cannot be checked per shard. The merger "
                 "recomputes FULL/427 and CLEAN/338 from the union manifest.")
    if args.data.endswith("sanitized-mbpp.json"):
        assert len(recs_all) == 427, len(recs_all)
    recs = [r for _, r in shard_select(recs_all, args.shard_i, args.shard_n)]
    clean_ids = set()
    if not args.no_clean:
        clean_ids = load_clean_ids([r["task_id"] for r in recs], args.clean)
        print(f"CLEAN denominator: {len(clean_ids)} tasks (manifest "
              f"{os.path.relpath(args.clean, ROOT)})")
    if args.control:
        if args.shard_n is not None:
            ap.error("--control checks canonical answers over the FULL dataset, not a shard")
        subset = recs if args.control == "ALL" else recs[:20]
        bad = [r["task_id"] for r in subset if not judge(r, "\n" + canonical_body(r))]
        print(f"canonical-sig control ({len(subset)}): failed", len(bad), bad[:10])
        sys.exit(1 if bad else 0)
    if args.first:
        recs = recs[: args.first]
    if args.n > 1 and args.temperature <= 0:
        ap.error(f"--n {args.n} at temperature 0 draws identical greedy; pass --temperature 0.2")

    if str(args.device).startswith("cpu"):
        if args.threads:
            torch.set_num_threads(args.threads)
        if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            sys.exit("REFUSING: --device cpu but CUDA_VISIBLE_DEVICES is unset -- set it empty.")

    from eval_artifacts import attest, open_artifact
    from tokenizers import Tokenizer

    from scripts.loader import load_checkpoint
    model, cfg = load_checkpoint(args.ckpt, device=args.device)
    model.eval()
    tok = Tokenizer.from_file(TOK_PATH)

    suffix = f".n{args.n}temp{args.temperature:g}" if args.n > 1 else ""
    preds_path = os.path.join(
        ROOT, "data", "eval",
        f"preds_mbpp_{os.path.basename(str(args.ckpt).rstrip('/'))}.{args.run}{suffix}"
        f"{shard_label(args.shard_i, args.shard_n)}.jsonl")
    t0 = time.time()
    npass = nempty = nclean_pass = 0
    clean_tasks_seen = 0
    with open_artifact(preds_path, force=args.force, run=args.run) as fout:
        out_path = fout.name
        fout.write(json.dumps({
            "_header": 1, "variant": "sig-docstring-rstrip", "benchmark": "mbpp-sanitized",
            "n_problems": len(recs), "n_total_problems": len(recs_all),
            "shard_i": args.shard_i, "shard_n": args.shard_n,
            "n": args.n, "temperature": args.temperature,
            "max_new": args.max_new, "ckpt": os.path.basename(str(args.ckpt).rstrip("/")),
            "clean_denominator": (len(clean_ids) if clean_ids else None),
        }, ensure_ascii=False) + "\n")
        for i, rec in enumerate(recs, 1):
            _sig, entry = signature(rec)
            prompt = model_prompt(rec)
            is_clean = rec["task_id"] in clean_ids if clean_ids else False
            if args.n > 1:
                from eval.sampling import sample_completions
                raws = sample_completions(model, tok, tok.encode(prompt).ids, rec["task_id"],
                                          args.n, args.temperature, args.max_new, args.device,
                                          cfg.seq)
            else:
                raws = [_greedy(model, tok, tok.encode(prompt).ids, args.max_new, args.device,
                                cfg.seq)]
            for si, raw in enumerate(raws):
                c = truncate(raw, entry)
                ok = judge(rec, c)
                npass += int(ok)
                nempty += int(not c.strip())
                if is_clean:
                    nclean_pass += int(ok)
                row = {"task_id": rec["task_id"], "gen": c, "ok": ok, "empty": not c.strip()}
                if clean_ids:
                    row["clean"] = is_clean
                if args.n > 1:
                    row["sample_idx"] = si
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            if is_clean:
                clean_tasks_seen += 1
            fout.flush()
            if i % 25 == 0 or i == len(recs):
                denom = i * args.n
                print(f"  {i}/{len(recs)} tasks  c={npass}/{denom} "
                      f"({100 * npass / denom:.2f}%)  ({__import__('time').time() - t0:.0f}s)",
                      flush=True)
    attest(out_path)
    denom = len(recs) * args.n
    label = f"n={args.n} T={args.temperature:g}" if args.n > 1 else "greedy"
    print(f"MBPP sig-rstrip FULL ({label}) = {npass}/{denom} = {100 * npass / denom:.2f}%")
    print(f"empty {nempty}/{denom}")
    if clean_ids:
        cden = clean_tasks_seen * args.n
        print(f"MBPP sig-rstrip CLEAN ({label}) = {nclean_pass}/{cden} = "
              f"{100 * nclean_pass / cden:.2f}% (r3 six-domain union excluded)")
    print("preds:", out_path, flush=True)


@torch.no_grad()
def _greedy(model, tok, prompt_ids, max_new, device, seq_window):
    x = torch.tensor([prompt_ids], device=torch.device(device))
    is_cuda = str(device).startswith("cuda")
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if is_cuda else contextlib.nullcontext()
    with ctx:
        for _ in range(max_new):
            lg = model(x[:, -seq_window:])[0][:, -1]
            nxt = lg.argmax(-1, keepdim=True)
            if nxt.item() == 1:
                break
            x = torch.cat([x, nxt], 1)
    return tok.decode(x[0, len(prompt_ids):].tolist())


def _selftest():
    """CLEAN manifest contract: clean == dataset - union; bad manifests refused.

    Builds a synthetic dataset {1..6} and union {1,2,3}; clean must be {4,5,6}.
    Counterexamples: a clean list that is not the complement, a malformed id,
    a missing key, a missing file, and a clean id outside the dataset.
    """
    import tempfile

    data_ids = [1, 2, 3, 4, 5, 6]

    def manifest(clean, union, mids):
        return {"r3_mbpp_union": [f"mbpp427:{i}" for i in union],
                "r3_mbpp_clean": [f"mbpp427:{i}" for i in clean],
                "r3_union_n": len(union), "r3_mbpp_clean_n": len(clean),
                "mbpp427_all_n": mids}

    with tempfile.TemporaryDirectory() as td:
        good = os.path.join(td, "good.json")
        with open(good, "w", encoding="utf-8") as fh:
            json.dump(manifest([4, 5, 6], [1, 2, 3], 6), fh)
        got = load_clean_ids(data_ids, good)
        assert got == {4, 5, 6}, got

        # clean is not the complement
        bad = os.path.join(td, "bad.json")
        with open(bad, "w", encoding="utf-8") as fh:
            json.dump(manifest([5, 6], [1, 2, 3], 6), fh)
        for path in (bad,):
            try:
                load_clean_ids(data_ids, path)
                raise AssertionError("a non-complement clean set was accepted")
            except RuntimeError as e:
                assert "!=" in str(e), str(e)

        # malformed id
        m = manifest([4, 5, 6], [1, 2, 3], 6)
        m["r3_mbpp_clean"][0] = "4"
        w = os.path.join(td, "w.json")
        with open(w, "w", encoding="utf-8") as fh:
            json.dump(m, fh)
        try:
            load_clean_ids(data_ids, w)
            raise AssertionError("a bare-int clean id was accepted")
        except RuntimeError as e:
            assert "mbpp427:" in str(e), str(e)

        # missing key
        m2 = manifest([4, 5, 6], [1, 2, 3], 6)
        del m2["r3_mbpp_clean"]
        k = os.path.join(td, "k.json")
        with open(k, "w", encoding="utf-8") as fh:
            json.dump(m2, fh)
        try:
            load_clean_ids(data_ids, k)
            raise AssertionError("a manifest missing r3_mbpp_clean was accepted")
        except RuntimeError as e:
            assert "r3_mbpp_clean" in str(e), str(e)

        # missing file
        try:
            load_clean_ids(data_ids, os.path.join(td, "nope.json"))
            raise AssertionError("a missing manifest was accepted")
        except RuntimeError as e:
            assert "not found" in str(e), str(e)

        # recorded count disagrees with the list
        m3 = manifest([4, 5, 6], [1, 2, 3], 6)
        m3["r3_mbpp_clean_n"] = 99
        n = os.path.join(td, "n.json")
        with open(n, "w", encoding="utf-8") as fh:
            json.dump(m3, fh)
        try:
            load_clean_ids(data_ids, n)
            raise AssertionError("a stale r3_mbpp_clean_n was accepted")
        except RuntimeError as e:
            assert "r3_mbpp_clean_n" in str(e), str(e)

    print("mbpp_gen selftest OK: clean=dataset-union; non-complement/malformed/"
          "missing-key/missing-file/stale-count all refused")
    return 0


if __name__ == "__main__":
    main()

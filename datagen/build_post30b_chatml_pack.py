#!/usr/bin/env python3
"""3b-22: build the post-30B V4.1 ChatML SFT pack.
# restartable: one-shot CPU pack from on-disk shards; deterministic, idempotent, ~3 min rerun.

Three slices, single-turn ChatML rendered with scripts/loader.format_example and
packed by datagen/prepare_sft.pack_and_save (default concat-then-encode path --
the ChatML boundary is <|im_start|>assistant\\n, so split_encode stays off):

  code  (target 72% of rows): data/sft/code_if_pairs_train.jsonl prompt=def
         signature+docstring, answer = the COMPLETE function (signature repeated,
         then body) -- the re-declaration shape the base already emits at the
         HumanEval prompt. 13-gram HE/MBPP decontaminated; eval-holdout excluded;
         a pair whose docstring carries `>>>` doctests is kept ONLY if the complete
         function passes its own doctests.
  en    (target 18%): tatsu-lab/alpaca parquet (data/sft/en_chat), single-turn.
  zh    (target 10%): the rendered ChatML domain data/corpus/chatml (alpaca_gpt4_zh
         + coig), holdout_slice excluded.

Counts are solved from a per-slice mean-token estimate so the packed pack lands at
--target-tokens with the 72/18/10 ROW ratio (rebalanced only if a source caps). The
combined pairs are shuffled (seed) before greedy packing so rows mix.

Out: a pack_and_save .pt {input_ids, labels, vocab_id, ...} plus a 30-row hand-read
jsonl (10 code / 10 en / 10 zh, decoded question/answer). CPU only; set
CUDA_VISIBLE_DEVICES="" so the GPU block is never touched.
"""

import argparse
import contextlib
import doctest
import glob
import io
import json
import os
import random
import re
import signal
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from holdout import is_holdout  # noqa: E402
from loader import IM_END, IM_START, format_example  # noqa: E402
from prepare_sft import pack_and_save  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

DATA = os.path.join(ROOT, "data")
TOK = os.path.join(DATA, "tokenizer.json")
CODE_SRC = os.path.join(DATA, "sft", "code_if_pairs_train.jsonl")
EN_PARQUET = os.path.join(DATA, "sft", "en_chat", "alpaca_train.parquet")
ZH_GLOB = os.path.join(DATA, "corpus", "chatml", "chatml_*.jsonl")
SEQ = 4096
SEED = 42
RATIO = {"code": 0.72, "en": 0.18, "zh": 0.10}

_DEF_NAME = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.M)
_DOCSTRING = re.compile(r"(\"\"\"|''')")


def _have_doctest(sig_docstr):
    return ">>>" in sig_docstr


def doctest_passes(complete, name, timeout_s=10):
    """Classify the completed function against executable doctests in its docstring.

    fb 3b-22: drop only a pair whose own docstring makes a concrete, runnable
    claim the completed function does not satisfy -- the shape the model loops
    on. Self-contained means: a single `>>>` source line (no `...` continuation)
    whose expected output is literal and which runs in the function's own
    namespace (no helper name the pair does not define). Multi-line examples,
    numpy pretty-print, and external-helper references are docstring-parser
    artifacts, not false targets, and count as untestable (KEPT).

    Returns "pass" (>=1 executable example, none failed), "fail" (one failed ->
    DROP), or "untest" (no executable self-contained example -> keep).
    """
    def _handler(signum, frame):
        raise TimeoutError("doctest timeout")

    ns = {}
    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout_s)
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                exec(compile(complete, "<code_if>", "exec"), ns)  # noqa: S102
        finally:
            signal.alarm(0)
        fn = ns.get(name)
        if fn is None:
            return "untest"
        doc = getattr(fn, "__doc__", "") or ""
        if ">>>" not in doc:
            return "untest"
        examples = doctest.DocTestParser().get_examples(doc, name)
    except (SystemExit, KeyboardInterrupt):
        return "untest"
    except Exception:  # noqa: BLE001
        return "untest"
    finally:
        signal.alarm(0)

    globs = dict(ns)
    ran = failed = 0

    for ex in examples:
        src = ex.source.strip()
        if src.startswith("...") or "\n" in ex.source.strip():
            continue
        want = ex.want
        sink = io.StringIO()
        is_assert = src.startswith("assert ")
        try:
            signal.alarm(timeout_s)
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                exec(compile(src, "<doctest>", "single"), globs)  # noqa: S102
            signal.alarm(0)
            got = sink.getvalue()
        except AssertionError:
            signal.alarm(0)
            if is_assert:
                failed += 1
                ran += 1
            continue
        except (NameError, AttributeError):
            signal.alarm(0)
            continue
        except BaseException:  # noqa: BLE001 -- SystemExit/parse artefacts -> untestable
            signal.alarm(0)
            continue
        ran += 1
        if want.strip() and not doctest.OutputChecker().check_output(ex.want, got, 0):
            failed += 1
    if ran == 0:
        return "untest"
    return "fail" if failed else "pass"


def build_code(decon, limit=0):
    pairs, st = [], {"in": 0, "empty": 0, "holdout": 0, "decontam": 0,
                     "no_def_name": 0, "doctest_testable": 0, "doctest_pass": 0,
                     "doctest_fail_dropped": 0, "doctest_untest_kept": 0}
    with open(CODE_SRC, encoding="utf-8") as fh:
        for line in fh:
            if limit and st["in"] >= limit:
                break
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            stub = (d.get("prompt") or "").rstrip("\n")
            body = d.get("output") or ""
            st["in"] += 1
            if not stub or not body.strip():
                st["empty"] += 1
                continue
            if is_holdout(stub):
                st["holdout"] += 1
                continue
            complete = stub + "\n" + body if not stub.endswith("\n") else stub + body
            if not decon.keeps(complete):
                st["decontam"] += 1
                continue
            m = _DEF_NAME.search(stub)
            if not m:
                st["no_def_name"] += 1
                continue
            if _have_doctest(stub):
                st["doctest_testable"] += 1
                verdict = doctest_passes(complete, m.group(1))
                if verdict == "fail":
                    st["doctest_fail_dropped"] += 1
                    continue
                if verdict == "pass":
                    st["doctest_pass"] += 1
                else:
                    st["doctest_untest_kept"] += 1
            pairs.append(format_example(stub, complete))
    return pairs, st


def build_en(decon):
    import pyarrow.parquet as pq

    t = pq.read_table(EN_PARQUET)
    instr = t.column("instruction").to_pylist()
    inp = t.column("input").to_pylist()
    out = t.column("output").to_pylist()
    pairs, st = [], {"in": len(instr), "empty": 0, "holdout": 0, "decontam": 0}
    for i, o, x in zip(instr, out, inp, strict=True):
        i = (i or "").strip()
        o = (o or "").strip()
        x = (x or "").strip()
        if not i or not o:
            st["empty"] += 1
            continue
        q = f"{i}\n\n{x}" if x else i
        if is_holdout(q):
            st["holdout"] += 1
            continue
        if not decon.keeps(o):
            st["decontam"] += 1
            continue
        pairs.append(format_example(q, o))
    return pairs, st


def build_zh(decon):
    head = IM_START + "user\n"
    asst = IM_START + "assistant\n"

    def split_render(content):
        if content.count(asst) != 1 or not content.startswith(head):
            return None
        ublock, ablock = content.split(asst, 1)
        if not ublock.startswith(head) or not ublock.endswith(IM_END + "\n"):
            return None
        q = ublock[len(head):-len(IM_END + "\n")]
        a = ablock
        if a.endswith(IM_END + "\n"):
            a = a[:-len(IM_END + "\n")]
        elif a.endswith(IM_END):
            a = a[:-len(IM_END)]
        return q, a

    pairs, st = [], {"in": 0, "malformed": 0, "empty": 0, "holdout": 0, "decontam": 0}
    seen = set()
    for p in sorted(glob.glob(ZH_GLOB)):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                content = d.get("content") or ""
                st["in"] += 1
                qa = split_render(content)
                if qa is None:
                    st["malformed"] += 1
                    continue
                q, a = qa[0].strip(), qa[1].strip()
                if not q or not a:
                    st["empty"] += 1
                    continue
                if is_holdout(q):
                    st["holdout"] += 1
                    continue
                if not decon.keeps(a):
                    st["decontam"] += 1
                    continue
                key = hash(a)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(format_example(q, a))
    return pairs, st


def mean_pair_tokens(tok, pairs, sample=4000):
    rnd = random.Random(SEED)
    sub = pairs if len(pairs) <= sample else rnd.sample(pairs, sample)
    tot = 0
    B = 1024
    for i in range(0, len(sub), B):
        chunk = sub[i:i + B]
        fs = tok.encode_batch([p + a for p, a in chunk])
        tot += sum(len(f.ids) + 1 for f in fs)
    return tot / len(sub)


def solve_counts(means, caps, target_tokens):
    w = sum(RATIO[s] * means[s] for s in RATIO)
    n_total = int(target_tokens / w)
    raw = {s: int(n_total * RATIO[s]) for s in RATIO}
    counts, st = {}, {}
    for s in RATIO:
        counts[s] = min(raw[s], caps[s])
        st[s] = {"wanted": raw[s], "capped": raw[s] > caps[s]}
    return counts, st


def mask_invariants(path, tok, eos, check_rows=200):
    """Mirror scripts/test_sft_pack.py invariants on the REAL produced pack."""
    import torch

    d = torch.load(path, weights_only=True)
    from loader import vocab_fingerprint

    assert d["vocab_id"] == vocab_fingerprint(tok), "pack vocab_id != tokenizer fingerprint"
    ids, lab = d["input_ids"], d["labels"]
    n = min(check_rows, ids.shape[0])
    im_end = tok.token_to_id(IM_END)
    for r in range(n):
        row, la = ids[r].tolist(), lab[r].tolist()
        cur, start = (la[0] == -100), 0
        spans = []
        for i, y in enumerate(list(la) + [None]):
            m = (y == -100) if y is not None else not cur
            if m != cur:
                spans.append((cur, start, i))
                cur, start = m, i
        assert spans[0][0], f"row {r}: does not open masked"
        for masked, a, b in spans:
            text = tok.decode(row[a:b], skip_special_tokens=False)
            if masked and b - a > 2 and set(row[a:b]) != {eos}:
                assert text.endswith("assistant\n"), f"row {r}: masked span {text[-30:]!r}"
                assert text.count("<|im_start|>user") == 1
            elif not masked:
                assert "<|im_start|>" not in text, f"row {r}: role marker supervised"
        sup = [t for t, y in zip(row, la, strict=True) if y != -100]
        assert im_end in sup, f"row {r}: im_end never supervised"
    return ids.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=TOK)
    ap.add_argument("--out", default=os.path.join(DATA, "sft", "sft_v41_chatml_post30b_0912.pt"))
    ap.add_argument("--handread", default=os.path.join(ROOT, "runs", "sft_v41_chatml_handread_30.jsonl"))
    ap.add_argument("--target-tokens", type=float, default=40_000_000)
    ap.add_argument("--stats", default=os.path.join(DATA, "sft", "sft_v41_chatml_post30b_0912.stats.json"))
    args = ap.parse_args()

    random.seed(SEED)
    tok = Tokenizer.from_file(args.tokenizer)
    eos = tok.token_to_id("<eos>")
    print(f"tokenizer vocab {tok.get_vocab_size()} eos {eos}", flush=True)

    sys.path.insert(0, os.path.join(ROOT, "filters"))
    from decontam_ngram import Decontaminator  # noqa: E402

    decon = Decontaminator.load_default(ROOT)

    code, cst = build_code(decon)
    print("code", len(code), cst, flush=True)
    en, est = build_en(decon)
    print("en", len(en), est, flush=True)
    zh, zst = build_zh(decon)
    print("zh", len(zh), zst, flush=True)

    pools = {"code": code, "en": en, "zh": zh}
    means = {s: mean_pair_tokens(tok, pools[s]) for s in pools}
    print("mean pair tokens", {k: round(v, 1) for k, v in means.items()}, flush=True)
    counts, solve = solve_counts(means, {s: len(pools[s]) for s in pools}, args.target_tokens)
    print("slice counts", counts, solve, flush=True)

    tagged = []
    for s in pools:
        rnd = random.Random(SEED + hash(s) % 100000)
        rnd.shuffle(pools[s])
        for pair in pools[s][:counts[s]]:
            tagged.append((s, pair))
    random.shuffle(tagged)
    examples = [pair for _, pair in tagged]

    src_files = [(CODE_SRC, "prompt", "output"), (EN_PARQUET, "instruction/output", "parquet")]
    for zp in sorted(glob.glob(ZH_GLOB)):
        src_files.append((zp, "content", "rendered ChatML"))
    pack_and_save(
        examples, tok, eos, args.out, SEQ,
        sources=src_files,
        extra_stats={
            "seed": SEED, "seq": SEQ, "target_row_ratio": RATIO,
            "rows_per_slice": counts, "mean_pair_tokens": means,
            "code_filter": cst, "en_filter": est, "zh_filter": zst,
        },
    )

    rows = mask_invariants(args.out, tok, eos)
    print(f"packed {len(examples)} examples -> {rows} rows at {args.out}", flush=True)

    hr = []
    for s in ("code", "en", "zh"):
        sel = [p for tag, p in tagged if tag == s][:10]
        for prompt, answer in sel:
            hr.append({"slice": s,
                       "prompt": tok.decode(tok.encode(prompt).ids, skip_special_tokens=False),
                       "answer": tok.decode(tok.encode(answer).ids, skip_special_tokens=False)})
    with open(args.handread, "w", encoding="utf-8") as fh:
        for r in hr:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"hand-read {len(hr)} rows -> {args.handread}", flush=True)


if __name__ == "__main__":
    main()

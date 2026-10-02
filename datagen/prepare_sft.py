#!/usr/bin/env python3
"""Prepare SFT data: ChatML (scripts/loader.format_example), prompt-masked, packed.

Reads all SFT sources, tokenizes with data/tokenizer.json, masks instruction
tokens (labels=-100), greedily packs whole examples (never split across a row,
over-length dropped) into (seq+1)-token rows right-padded with <eos>, and saves
data/sft/sft_all.pt as {"input_ids": int32 (N, seq+1), "labels": int32 (N, seq+1)}.
labels[t] = input_ids[t] for output/eos tokens, -100 for prompt/pad tokens.
sft.py resets KDA state + SWA attention at every <eos> (Cfg.doc_mask), so the
clean per-document <eos> boundary this produces is what the doc mask keys on.
Training slices x=[:, :-1], y=labels[:, 1:].
"""

import hashlib
import json
import os
import random
import sys
from collections import deque

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# loader.py lives in scripts/; this file moved out of scripts/ on 2026-08-31 (c3a47e8f) and the
# move left `from loader import ...` unresolvable, so this script has died at import ever since.
# datagen/build_corpus.py:27 and pack_control_sft_ours.py:42 got this line then; this one did not.
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from holdout import is_holdout  # noqa: E402
from loader import format_example  # noqa: E402

import fone  # noqa: E402

DATA = os.path.join(ROOT, "data")
TOK_PATH = os.path.join(DATA, "tokenizer.json")
OUT_PATH = os.path.join(DATA, "sft", "sft_all.pt")

SEQ = 4096  # model context; rows are SEQ+1 (input + 1 to predict)
MAX_EXAMPLES = 500_000
#: examples the packer may look past to fill a row's tail. Large enough that some
#: example fits almost any remaining room, small enough that order stays locally
#: shuffled -- length-sorted packing would bias what the model sees first.
LOOKAHEAD = 512
ENC_BATCH = 8192

SOURCES = [
    (os.path.join(DATA, "alpaca_gpt4_zh.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "coig.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "openo1_sft.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "gsm8k_zh.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "school_math_r1_zh.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "s1k.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "sft", "fable5_cot.jsonl"), "prompt", "response"),
    # t29 (2026-08-31): dropped -- this IS the code-500 carve source. Its 2413
    # same-template sibling rows made SFT code-500 measure in-distribution
    # template recall, not capability (be.sft_v3_code500, dose-acc r=0.69).
    # The family-clean pack builds without it; the file stays on disk for the
    # eval's provenance (cont.code_holdout_carved).
    # t43 (2026-08-31): v5 addon -- English Evol-Instruct Python tasks, a
    # different generator and language family from the dropped Chinese carve
    # source. >12.6pt on code-500 = cross-generator transfer (capability);
    # ~0 = strong template recall.
    (os.path.join(DATA, "sft", "v5_evol_code_2300.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "synthetic", "knowledge_qa_zh.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "synthetic", "math_gsm8k_zh.jsonl"), "instruction", "output"),
]


def _fp_file(path):
    """Content hash of a single file. Content-based, not git sha: uncommitted edits
    change what a pack contains, and a sha would not see them."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _fp_sources(sources=None):
    """Content hash of the source files a pack was ACTUALLY built from.

    `sources` is the caller's own list. Defaulting to this module's SOURCES was wrong in
    two ways at once, and the second is the expensive one:

      it crashed every caller that supplies examples directly. pack_and_save takes
      (prompt, output) pairs, so test_sft_pack, test_arch_compat and prepare_sft.selftest
      never touch SOURCES -- but the stamp opened all ten anyway, and all ten are
      gitignored pod data. Both tests died with FileNotFoundError on
      data/alpaca_gpt4_zh.jsonl in their own bookkeeping, after their real assertions had
      passed. test_arch_compat is the gate fb named for the step-832 interrupt checkpoint's
      loadability, and its legacy round-trip prints OK two lines before the crash.

      IT WROTE A FALSE PROVENANCE. prepare_sft_math.py has its OWN four-file SOURCES
      (school_math_train, gsm8k_zh_train, alpaca_gpt4_zh, coig_50k) and calls this same
      packer, so every ckpt_sft_p324_v* pack carries a sources_fp computed over
      prepare_sft's TEN files -- a fingerprint naming sources that pack never read. A
      provenance field that describes the wrong inputs is worse than an absent one: it
      answers "which sources built this?" with a confident wrong answer, and nothing
      reads it today, so nothing would have caught it.

    None means the caller supplied examples directly and no source list applies; the stamp
    then says so rather than inventing one.
    """
    if sources is None:
        return "caller-supplied examples, no source list"
    h = hashlib.sha256()
    missing = []
    for path, _, _ in sources:
        if not os.path.exists(path):
            missing.append(os.path.basename(path))
            continue
        with open(path, "rb") as f:
            h.update(os.path.basename(path).encode() + b"\0" + hashlib.sha256(f.read()).digest())
    if missing:
        # Absent inputs are not a hash over what remains: that would give a stable-looking
        # fingerprint for a pack built from a different set of files.
        raise FileNotFoundError(
            f"{len(missing)} source file(s) absent, so this pack's provenance cannot be "
            f"computed: {missing[:4]}"
        )
    return h.hexdigest()[:16]


def read_examples():
    """Yield (prompt, output) text pairs from all sources, excluding eval-holdout questions."""
    n_holdout = 0
    for path, qk, ak in SOURCES:
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                q = (d.get(qk) or "").strip()
                a = (d.get(ak) or "").strip()
                inp = (d.get("input") or "").strip()
                if inp:
                    q = f"{q}\n{inp}"
                if not q or not a:
                    continue
                if is_holdout(q):  # never train on a question the eval holds out
                    n_holdout += 1
                    continue
                yield format_example(q, a)
                n += 1
        print(f"  {os.path.basename(path)}: {n}", flush=True)
    if n_holdout:
        print(f"  excluded {n_holdout} eval-holdout questions", flush=True)


def _encode_pairs(batch, tok, num_id, split=False):
    """(prompt, answer) pairs -> [(prompt_ids, full_ids, full_values)], one per pair.

    num_id None is plain BPE and the values come back empty. Otherwise numbers
    collapse to one [NUM] each and carry a value per position, exactly as train.py
    encodes the pretraining corpus -- a FoNE model has never seen a number written
    any other way, so packing SFT data the old way would fine-tune it out of its
    own input distribution.

    split=True encodes prompt and answer SEPARATELY and concatenates the token
    streams, so inference (which encodes the prompt alone) sees a token-by-token
    prefix of the training sequence. Concatenating the strings first lets
    byte-level BPE merge the prompt's trailing token with the answer's leading
    one (~98% of code pairs: "\\n" + indentation), and no masking rule can then
    make the first supervised position match what inference feeds. Plain BPE
    only: this path exists for the raw format-SFT pack, whose model has
    fone=False; pack_and_save refuses the FoNE combination.
    """
    prompts = [p for p, _ in batch]
    if split:
        answers = [a for _, a in batch]
        return [(ep.ids, ep.ids + ea.ids, ())
                for ep, ea in zip(tok.encode_batch(prompts), tok.encode_batch(answers))]
    fulls = [p + a for p, a in batch]
    if num_id is None:
        return [(ep.ids, ef.ids, ()) for ep, ef in zip(tok.encode_batch(prompts), tok.encode_batch(fulls))]
    # Only the full text's values are kept. A prompt ending mid-number reads a
    # different value there than the full text does, and the full text is the one
    # the row actually contains.
    pp, _ = fone.encode_text(prompts, tok, num_id)
    fp, fv = fone.encode_text(fulls, tok, num_id)
    out, fi = [], 0
    for p_ids, f_ids in zip(pp, fp):
        k = int((f_ids == num_id).sum())
        out.append((p_ids.tolist(), f_ids.tolist(), fv[fi : fi + k].tolist()))
        fi += k
    return out


# Imported, not re-implemented: a pack's fingerprint must equal the vocab_id train.py
# stamps into every checkpoint, or sft_math.py's equality check can never fire.
from loader import vocab_fingerprint as _vocab_fingerprint  # noqa: E402


def pack_and_save(examples, tok, eos, out_path, seq, num_id=None, sources=None,
                  split_encode=False, extra_stats=None):
    """Greedily pack (prompt, output) text pairs into (seq+1)-token rows and save.

    One example never split across rows; over-length examples dropped; rows are
    prompt-masked (labels=-100) and right-padded with <eos>. Saves out_path as
    {"input_ids": int32 (N, seq+1), "labels": int32 (N, seq+1)}.

    num_id set adds "values": float32 (N, seq+1), the number at every [NUM]
    position and 0 elsewhere, which sft_math.py feeds to the FoNE embedding.

    split_encode (default OFF): encode prompt and answer separately and
    concatenate the streams, so the inference sequence is a token-by-token
    prefix of the training sequence. Only the raw format-SFT pack opts in;
    every existing pack keeps its exact bytes. See _encode_pairs. The default
    path's byte-invariance under this flag is a code-reading judgment (the
    default branch is untouched; 4c review 2026-09-09), not a golden-pack
    test. Reversal condition: a future edit that changes the default branch
    ITSELF -- not adding a branch before it -- needs a golden-pack comparison.

    extra_stats: merged into build_stats, for counts only the caller knows
    (e.g. how many pairs its own filter dropped).

    `sources` is the caller's own source list, stamped into sources_fp. Pass it when the
    examples came from files; leave it None when they were built in-process. It is a
    parameter rather than this module's SOURCES because prepare_sft_math.py has a different
    four-file list and calls this same function -- see _fp_sources.
    """
    if split_encode and num_id is not None:
        raise ValueError("split_encode is plain-BPE only; the FoNE path encodes prompt+answer as one string")
    # Never split an example across rows; drop over-length ones. sft.py doc-masks by
    # <eos>, so within-row cross-example attention is already blocked, but a truncated
    # example has no prompt for the mask to supply.
    rows_ids, rows_lab, rows_val = [], [], []
    cur_ids, cur_lab, cur_val = [], [], []
    n_drop = 0
    n_mismatch = 0
    n_pad = 0
    row_len = seq + 1

    def flush():
        """Emit the current row, right-padded with <eos> that carry no loss."""
        nonlocal cur_ids, cur_lab, cur_val, n_pad
        if not cur_ids:
            return
        pad = row_len - len(cur_ids)
        n_pad += pad
        rows_ids.append(cur_ids + [eos] * pad)
        rows_lab.append(cur_lab + [-100] * pad)
        rows_val.append(cur_val + [0.0] * pad)
        cur_ids, cur_lab, cur_val = [], [], []

    def place(item):
        ids_f, plen, dense = item
        cur_ids.extend(ids_f)
        cur_lab.extend([-100] * plen + ids_f[plen:])
        if num_id is not None:
            cur_val.extend(dense)

    pending = deque()
    for i in range(0, len(examples), ENC_BATCH):
        batch = examples[i : i + ENC_BATCH]
        for _ex, (ids_p, ids_f, vals_f) in enumerate(_encode_pairs(batch, tok, num_id, split=split_encode)):
            ids_f = ids_f + [eos]
            # split_encode: ids_f IS ep+eb, so the prefix holds by construction. A
            # mismatch there means the pack silently fell back to the common-prefix
            # mask this flag exists to replace, whose only trace would be a stats
            # count nobody reads before training -- raise instead (4c review,
            # 2026-09-09). Default path: concat-then-encode can merge across the
            # boundary, and the common-prefix fallback masks the merged token.
            plen = len(ids_p)
            if ids_f[:plen] != ids_p:
                if split_encode:
                    raise RuntimeError(
                        f"split_encode: prompt is not a token prefix of the packed "
                        f"sequence at example {i + _ex} -- the boundary invariants broke")
                n_mismatch += 1
                plen = 0
                for a, b in zip(ids_p, ids_f):
                    if a != b:
                        break
                    plen += 1
            if len(ids_f) > row_len:
                n_drop += 1  # drop rather than truncate: a truncated head has no question
                continue
            dense = None
            if num_id is not None:
                # values back onto their own positions: the k-th [NUM] takes the k-th value
                dense, k = [], 0
                for t in ids_f:
                    dense.append(vals_f[k] if t == num_id else 0.0)
                    k += t == num_id
                assert k == len(vals_f), f"{k} [NUM] but {len(vals_f)} values"
            pending.append((ids_f, plen, dense))
        # Pack with a bounded lookahead. Plain first-fit closes a row as soon as the NEXT
        # example does not fit, so the leftover is whatever that example's length happened
        # to be -- 11.8% of the 3.24b pack was tail padding, all of it forward and backward
        # on nothing. Scanning ahead for one that fits recovers most of it. Sorting by
        # length (classic FFD) would pack tighter still, but it puts every long example at
        # the front of training, so the shuffle has to survive: a window keeps the order
        # locally random. Drain only what a full window can see, so the tail of one encode
        # batch still packs against the head of the next.
        while len(pending) > LOOKAHEAD:
            room = row_len - len(cur_ids)
            j = next((k for k in range(min(LOOKAHEAD, len(pending))) if len(pending[k][0]) <= room), None)
            if j is None:
                flush()
                continue
            place(pending[j])
            del pending[j]
        if (i // ENC_BATCH) % 10 == 0:
            print(f"  tokenized {min(i + ENC_BATCH, len(examples))}/{len(examples)}", flush=True)
    while pending:
        room = row_len - len(cur_ids)
        j = next((k for k in range(len(pending)) if len(pending[k][0]) <= room), None)
        if j is None:
            flush()
            continue
        place(pending[j])
        del pending[j]
    flush()

    n_rows = len(rows_ids)
    input_ids = torch.tensor(rows_ids, dtype=torch.int32)
    labels = torch.tensor(rows_lab, dtype=torch.int32)

    # "vocab_id": the same key checkpoints carry, or the fingerprint check compares
    # two differently-named fields. The other three fingerprints make the pack's
    # provenance self-describing: which packer built it, which sources it read, and
    # which holdout set it was checked against. A stale holdout_fp is the
    # contamination that nothing currently catches.
    blob = {
        "input_ids": input_ids,
        "labels": labels,
        "vocab_id": _vocab_fingerprint(tok),
        "packer_fp": _fp_file(os.path.abspath(__file__)),
        "sources_fp": _fp_sources(sources),
        "holdout_fp": _fp_file(os.path.join(ROOT, "data", "eval", "holdout_hashes.txt")),
        # Printed since the beginning and never persisted, which made a question like "how
        # many examples did seq 4096 drop on this pack?" answerable only from the build's
        # scrollback. A pack outlives its log: the control comparison's report header needs
        # both arms' drop counts side by side, and one of them was unrecoverable.
        "build_stats": {"examples": len(examples), "dropped_overlong": n_drop,
                        "rows": n_rows, "row_len": row_len, "pad_tokens": n_pad,
                        "prefix_mismatches": n_mismatch, "seq": seq,
                        "split_encode": split_encode, **(extra_stats or {})},
    }
    if num_id is not None:
        blob["values"] = torch.tensor(rows_val, dtype=torch.float32)
        n_num = int((input_ids == num_id).sum())
        print(f"[NUM] tokens: {n_num / 1e6:.2f}M ({100 * n_num / input_ids.numel():.2f}% of the pack)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, out_path)

    total_tokens = input_ids.numel()
    loss_tokens = (labels != -100).sum().item()
    print(f"kept examples: {len(examples)}", flush=True)
    print(f"dropped (> {row_len} tokens): {n_drop}", flush=True)
    print(
        f"padding tokens: {n_pad / 1e6:.2f}M ({100 * n_pad / max(input_ids.numel(), 1):.1f}% of the pack)",
        flush=True,
    )
    print(f"prefix mismatches: {n_mismatch}", flush=True)
    print(f"packed rows: {n_rows} ({row_len} tokens each)", flush=True)
    print(f"total tokens: {total_tokens / 1e6:.2f}M", flush=True)
    print(f"loss tokens: {loss_tokens / 1e6:.2f}M ({100 * loss_tokens / total_tokens:.1f}%)", flush=True)
    print(f"saved {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB)", flush=True)



def _selftest():
    """The packer reorders examples to fill row tails; the failure it can hide is losing or
    duplicating one, so assert conservation on real pack_and_save output, not on a model of it."""
    import tempfile

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    class _FakeTok:  # ids are lengths made unique per example, so conservation is checkable
        def encode_batch(self, texts):
            return [_Enc([hash(t) % 900 + 100] * len(t)) for t in texts]

    global _vocab_fingerprint
    real_fp, _vocab_fingerprint = _vocab_fingerprint, lambda _t: "selftest"
    random.seed(0)
    pairs = [("q" * random.randint(5, 60), "a" * random.randint(5, 400)) for _ in range(4000)]
    try:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "p.pt")
            pack_and_save(pairs, _FakeTok(), 0, out, 511)
            blob = torch.load(out, weights_only=True)
    finally:
        _vocab_fingerprint = real_fp

    ids, lab = blob["input_ids"], blob["labels"]
    assert ids.shape == lab.shape and ids.shape[1] == 512, ids.shape
    # Every example is one contiguous run of a single id; count runs per id and compare to
    # how many examples carry that id. A dropped or duplicated example moves a count.
    from collections import Counter

    want = Counter()
    for q, a in pairs:
        want[hash(q + a) % 900 + 100] += 1
    got = Counter()
    for row in ids.tolist():
        prev = None
        for t in row:
            if t != prev and t != 0:
                got[t] += 1
            prev = t
    assert got == want, f"packing lost or duplicated examples: {len(want - got)} missing"
    pad = int((lab == -100).sum()) - int(((lab == -100) & (ids != 0)).sum())
    assert pad >= 0
    return f"{len(pairs)} examples conserved across {ids.shape[0]} rows"


def _peel_chat_wrapper(text):
    """A packed prompt span carries its chat wrapper ('user\\n<q>\\nassistant\\n'); the holdout
    set hashes the bare question. Measured 2026-09-28 packs: without this, zh_think_v1's prompts
    hashed to nothing at all and the check returned a zero that meant 'blind', not 'clean'."""
    lines = text.split("\n")
    body = lines[1:] if lines and lines[0].strip() in ("user", "system") else lines[:]
    while body and body[-1].strip() in ("", "assistant"):
        body.pop()
    return "\n".join(body).strip()


def pack_holdout_hits(path, tok=None):
    """(n_prompts, n_hits, hit_examples) for the questions actually inside a packed .pt.

    Reads the PACK, not its source files, for two reasons. A source file can be deleted --
    sft_mixA_0924's sc2_exec.jsonl and apps_call.jsonl are gone from the pod, 60,375 of its
    143,687 pairs -- and a source file can drift after the pack was built, so it answers a
    question about the source rather than about the bytes that would be trained on.

    A prompt span is a run of labels == -100 that is FOLLOWED by a supervised token; the
    trailing pad run has no successor and is skipped. Both the raw span and the wrapper-peeled
    span are hashed, because the packs disagree on the wrapper and checking both can only make
    the test more sensitive, never less.
    """
    if tok is None:
        tok = Tokenizer.from_file(TOK_PATH)
    d = torch.load(path, map_location="cpu", weights_only=False)
    ids, lab = d["input_ids"], d["labels"]
    n = hits = 0
    examples = []
    for r in range(ids.shape[0]):
        L, I = lab[r].tolist(), ids[r].tolist()
        start = None
        for i, x in enumerate(L):
            if x == -100:
                if start is None:
                    start = i
            else:
                if start is not None:
                    n += 1
                    text = tok.decode(I[start:i])
                    if any(is_holdout(c) for c in {text, text.strip(), _peel_chat_wrapper(text)} if c):
                        hits += 1
                        if len(examples) < 5:
                            examples.append(_peel_chat_wrapper(text)[:160])
                    start = None
    return n, hits, examples


def restamp_holdout_fp(path, tok=None):
    """Re-verify a pack against the CURRENT holdout set and, only then, update its holdout_fp.

    sft_math.py refuses a pack whose holdout_fp is stale, and every pack on the pod went stale
    when 15836799 added 1,821 lines to data/eval/holdout_hashes.txt (e252f0f6d82c0237 ->
    b382de09159701c0). The refusal is right and must not gain an escape hatch: a flag that let a
    stale pack through would let a genuinely contaminated one through on the same argument.

    So the VERIFICATION is the gate. This re-runs is_holdout over the pack's own prompts against
    the live set and refuses to touch a pack with a single hit. A contaminated pack cannot be
    re-stamped by this path at all; it has to be repacked.

    Written temp+rename, never in place: a pack may be hardlinked (the same inode), and writing
    a "copy" in place truncates whatever else points at it.
    """
    if tok is None:
        tok = Tokenizer.from_file(TOK_PATH)
    live = _fp_file(os.path.join(ROOT, "data", "eval", "holdout_hashes.txt"))
    d = torch.load(path, map_location="cpu", weights_only=False)
    old = d.get("holdout_fp")
    n, hits, examples = pack_holdout_hits(path, tok)
    # FAIL CLOSED ON A BLIND SCAN, before reading `hits` at all. A scan that extracted nothing
    # returns hits == 0, and hits == 0 is the certify condition -- so without this, a pack whose
    # wrapper or label convention this scan cannot read certifies itself. That is not
    # hypothetical: the first version of pack_holdout_hits returned 0 for zh_think_v1, whose
    # sources hold 1,285 held-out questions, because it did not peel the chat wrapper. The peel
    # fixed today's packs and does nothing for tomorrow's.
    #
    # The floor is derived from the pack's own shape, not a tuned constant: the packer places
    # WHOLE examples and never emits an empty row, so every row carries at least one example and
    # therefore at least one prompt span. n < rows means spans are being missed. Measured on the
    # six 2026-09-28 packs the ratio is 13x to 22x, so the floor is nowhere near the live values.
    rows = int(d["input_ids"].shape[0])
    if n < rows:
        raise SystemExit(
            f"REFUSING to re-stamp {path}: the scan extracted {n} prompt span(s) from {rows} "
            f"packed row(s). Every row holds at least one whole example, so this is BLIND, NOT "
            f"CLEAN -- the pack's wrapper or label convention is one this scan cannot read, and "
            f"its zero hit count means nothing. Fix the extraction before certifying the pack.")
    if hits:
        raise SystemExit(
            f"REFUSING to re-stamp {path}: {hits} of {n} packed prompts are held-out questions "
            f"under the current holdout set {live}. Examples: {examples}. This pack is "
            f"contaminated, not merely stale -- repack it with the holdout filter on.")
    if old == live:
        print(f"{path}: already stamped {live}, {n} prompts verified clean")
        return live
    # BOTH fingerprints, and the scope that produced the 0. Overwriting holdout_fp alone would
    # make a re-verified 2026-09-28 build indistinguishable from a fresh repack -- the
    # derived-artifact-provenance failure this repo has already paid for three times. A reader
    # asking "was this rebuilt or re-certified?" gets an answer from the pack itself.
    d["holdout_fp_built"] = d.get("holdout_fp_built", old)
    d["holdout_fp"] = live
    d.setdefault("build_stats", {})["restamped"] = {
        "built_against": old,
        "verified_against": live,
        "scope": "every labels==-100 prompt span followed by a supervised token, decoded with "
                 "data/tokenizer.json and hashed raw AND wrapper-peeled",
        "prompts_verified": n,
        "hits": hits,
        "method": "datagen/prepare_sft.pack_holdout_hits (the pack's own bytes, not its sources)",
        "by": "datagen/prepare_sft.py --restamp (de)",
        "date": "2026-09-30",
    }
    tmp = path + ".restamp.tmp"
    torch.save(d, tmp)
    os.replace(tmp, path)
    print(f"{path}: {n} prompts verified clean, holdout_fp {old} -> {live}")
    return live


def main():
    global SOURCES
    if "--restamp" in sys.argv:
        # `--restamp <pack.pt> [more.pt ...]`: re-verify and re-certify, never repack. Every
        # following argument up to the next flag is a pack path.
        i = sys.argv.index("--restamp") + 1
        packs = []
        while i < len(sys.argv) and not sys.argv[i].startswith("--"):
            packs.append(sys.argv[i])
            i += 1
        if not packs:
            raise SystemExit("--restamp takes one or more pack paths")
        tok = Tokenizer.from_file(TOK_PATH)
        for p in packs:
            restamp_holdout_fp(p, tok)
        return
    out_path = OUT_PATH
    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]
    if "--only" in sys.argv:
        # comma-separated basenames; an unknown name refuses rather than packing less than asked
        want = sys.argv[sys.argv.index("--only") + 1].split(",")
        known = {os.path.basename(p) for p, _, _ in SOURCES}
        assert set(want) <= known, f"--only names unknown sources: {sorted(set(want) - known)}"
        SOURCES = [s for s in SOURCES if os.path.basename(s[0]) in want]
    if "--src" in sys.argv:
        # explicit instruction/output jsonl paths (repeatable), replacing SOURCES
        SOURCES = [(os.path.abspath(sys.argv[i + 1]), "instruction", "output")
                   for i, a in enumerate(sys.argv) if a == "--src"]
    random.seed(42)
    tok = Tokenizer.from_file(TOK_PATH)
    eos = tok.token_to_id("<eos>")
    assert eos is not None, "tokenizer has no <eos>"

    examples = list(read_examples())
    random.shuffle(examples)
    if len(examples) > MAX_EXAMPLES:
        examples = examples[:MAX_EXAMPLES]
    print(f"total examples: {len(examples)}", flush=True)

    pack_and_save(examples, tok, eos, out_path, SEQ, sources=SOURCES)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        print("prepare_sft selftest OK:", _selftest())
    else:
        main()

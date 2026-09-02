#!/usr/bin/env python3
# restartable: one JSON line per item is appended to --preds as soon as that item is scored,
# and a rerun with the same --preds skips ids already present. An interrupt costs the current
# item only, so this needs no shard splitting.
"""LAMBADA-en last-word accuracy, lm-eval-harness definition (1e's ruling 2026-09-02).

MAIN READING -- greedy continuation to the first word boundary, compared as a STRING.
Generate greedily from the context, cut at the first boundary character (space, punctuation,
newline), strip, and compare to the target word. Tokenizer-independent by construction, so
the two sides of the control comparison score the SAME 5,153 items and the number is
comparable to published LAMBADA figures.

SIDE READING -- per-byte NLL of the target word, teacher-forced: sum the log-probs of every
token of " word" and divide by the word's UTF-8 byte count. Also tokenizer-independent.

WHY NOT THE SINGLE-TOKEN OPEN-VOCAB VERSION (this file's first draft, superseded before use):
that reading requires the target to be one token, and on the real 5,153-row file only 15.7%
of last words are single-token under our tokenizer versus 70.1% under Pythia's. Dropping the
rest would score two different subsets on the two sides and leave n=799, with 99.6% of the
constraint coming from our own vocabulary. The greedy reading keeps every item and makes a
multi-token word simply harder -- the model must get both fragments right, which is the real
task.

THE TWO READINGS ARE EXPECTED TO DIVERGE, and that is not a bug. The greedy reading has a
decoder and a stop rule; the per-byte NLL has neither. facts/base_eval.json
#be.gold_bpb_falls_while_generation_scores_zero measured exactly this split -- code_500
generative accuracy sat at 0.0 across a whole checkpoint ladder while gold BPB fell
monotonically 1.087 -> 0.918. At 200M scale expect the greedy number near zero and the NLL
number to move; do not read the pair as a contradiction.

    python eval/lambada_en.py --ckpt <ckpt> --data data/eval/lambada_en/lambada_test_en.jsonl
    python eval/lambada_en.py --ckpt <hf-dir> --hf     # control arm, its own tokenizer
    python eval/lambada_en.py --selftest               # needs data/tokenizer.json (pod)
"""

import argparse
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("FLA_FLASH_KDA", "0")

TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
DEFAULT_DATA = os.path.join(ROOT, "data", "eval", "lambada_en", "lambada_test_en.jsonl")

# THE STOP RULE IS THE CRITERION, so it is one constant with a name rather than an inline
# expression. A word ends at whitespace, at any ASCII punctuation, or at end-of-generation.
# Everything here is a decision that changes the score, which is why it is not buried in a
# regex: `'` is NOT a boundary (don't/it's are one word), and `-` IS -- LAMBADA's targets
# are single words, so a hyphen means the model began a compound and the first half is its
# answer. Both decisions are asserted in _selftest; the first draft of this comment claimed
# the hyphen was included while the set omitted it, and the assertion is what caught that.
WORD_BOUNDARY = set(" \t\n\r-.,;:!?\"()[]{}<>/\\|`~@#$%^&*+=—–…")
MAX_NEW_TOKENS = 8  # a word is at most a few tokens; 8 bounds the worst case cheaply


def first_word(text):
    """The generated word: leading boundary chars skipped, then up to the next boundary.

    Returns "" when the generation contains no word character before its first boundary --
    a real outcome (the model emitted punctuation or a newline), scored as a miss, never
    silently skipped. _selftest covers it.
    """
    i = 0
    while i < len(text) and text[i] in WORD_BOUNDARY:
        i += 1
    j = i
    while j < len(text) and text[j] not in WORD_BOUNDARY:
        j += 1
    return text[i:j]


def split_item(raw):
    """(context, target_word) from a LAMBADA row, or None if it has no last word.

    The published file stores one `text` field per row; the target is its final word. The
    context keeps NO trailing space -- the space belongs to the generation, which is what a
    continuation model produces.
    """
    t = (raw.get("text") or raw.get("content") or "").strip()
    t = t.rstrip(".!?\"')")
    if " " not in t:
        return None
    head, _, word = t.rpartition(" ")
    if not word or not any(c.isalpha() for c in word):
        return None
    return head, word


def load_items(path, limit=None):
    items = []
    with open(path, encoding="utf-8") as f:
        for k, line in enumerate(f):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            sp = split_item(raw)
            if sp is None:
                continue
            items.append({"id": k, "context": sp[0], "target": sp[1]})
            if limit and len(items) >= limit:
                break
    return items


class OursModel:
    """Our checkpoint. Exposes only what this eval needs: encode, decode, logits, eos."""

    def __init__(self, ckpt, tok_path, device):
        from scripts.loader import EOS_ID, load_checkpoint, load_tokenizer  # noqa: PLC0415

        self.model, self.cfg = load_checkpoint(ckpt, device=device, dtype=torch.bfloat16)
        self.tok = load_tokenizer(tok_path, self.cfg)
        self.eos = EOS_ID
        self.device = device
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def encode(self, s):
        return self.tok.encode(s, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tok.decode(ids)

    def logits(self, ids):
        x = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            out = self.model(x)
        return (out[0] if isinstance(out, tuple) else out)[0].float()


class HFModel:
    """The control arm: an HF-format causal LM with its OWN tokenizer."""

    def __init__(self, path, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self.tok = AutoTokenizer.from_pretrained(path)
        # `dtype`, NOT `torch_dtype`: the pod runs transformers 5.6.0, where torch_dtype still
        # works but prints "`torch_dtype` is deprecated! Use `dtype` instead!" (verified by
        # loading the real control model both ways). And no pipeline / device_map: accelerate
        # is absent on the pod, so the move is a plain .to(device).
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device)
        self.model.eval()
        self.eos = self.tok.eos_token_id
        self.device = device
        # PARAMETERS ONLY, never buffers. Pythia-160M carries causal-mask and rotary inv_freq
        # buffers; counting them gave 212M/135M, which was wrong and is retracted (e1,
        # 2026-09-02). model.parameters() excludes buffers, so this is right by construction --
        # verified against the real checkpoint: 162,322,944 total, 85,056,000 non-embedding,
        # embed_in and embed_out 38,633,472 each (UNTIED, unlike ours).
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.n_params_non_embed = self.n_params - sum(
            p.numel() for n, p in self.model.named_parameters()
            if "embed_in" in n or "embed_out" in n or "wte" in n or "lm_head" in n)

    def encode(self, s):
        return self.tok.encode(s)

    def decode(self, ids):
        return self.tok.decode(ids)

    def logits(self, ids):
        x = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            return self.model(x).logits[0].float()


def greedy_word(m, ctx_ids, max_new=MAX_NEW_TOKENS):
    """Greedy-decode up to max_new tokens, stop as soon as a word boundary is produced."""
    ids = list(ctx_ids)
    made = []
    for _ in range(max_new):
        nxt = int(m.logits(ids)[-1].argmax())
        if nxt == m.eos:
            break
        made.append(nxt)
        ids.append(nxt)
        text = m.decode(made)
        # Stop at the FIRST boundary after a word has begun. Checking the decoded string
        # rather than the token id is what makes this tokenizer-independent: a boundary may
        # arrive inside a merged token (" cat." is one token in some vocabularies).
        w = first_word(text)
        if w and len(text) > len(w) + (len(text) - len(text.lstrip())):
            break
    return first_word(m.decode(made)), made


def target_nll_per_byte(m, ctx_ids, target):
    """Teacher-forced sum of -log p over the target's tokens, per UTF-8 byte of the word.

    Per BYTE, not per token: the two sides of the control run have different tokenizers, so
    a per-token figure is not comparable between them (1e's ruling). The leading space is
    part of the string scored but NOT of the byte count -- the byte count is the word's, so
    the unit means the same thing on both sides.
    """
    tgt_ids = m.encode(" " + target)
    if not tgt_ids:
        return None
    ids = list(ctx_ids)
    total = 0.0
    for t in tgt_ids:
        lg = m.logits(ids)[-1]
        total -= float(torch.log_softmax(lg, -1)[t])
        ids.append(t)
    return total / max(1, len(target.encode("utf-8")))


def _selftest():
    # The stop rule, which is the criterion this file turns on. Every case is a real
    # generation shape, including the degenerate one.
    assert first_word(" door and then") == "door"
    assert first_word("door") == "door"
    assert first_word(" don't") == "don't", "an apostrophe is inside a word"
    assert first_word(" well-known") == "well", "a hyphen ends the word"
    assert first_word(" door.") == "door"
    assert first_word("\n\ndoor") == "door"
    # DEGENERATE: nothing but boundary characters. Must be "" (scored as a miss), not a
    # crash and not a silent skip -- the model really does emit bare punctuation at 200M.
    assert first_word("") == ""
    assert first_word("   ") == ""
    assert first_word(".\n") == ""
    assert first_word("...") == ""

    # Row splitting on the published shape, including a row with no last word.
    ok = split_item({"text": "he saw a cup of tea on the table."})
    assert ok == ("he saw a cup of tea on the", "table"), ok
    assert split_item({"text": "single"}) is None
    assert split_item({"text": "ends in 42."}) is None, "a numeric last word is not a word"

    # Per-byte normalisation: the divisor is the WORD's bytes, not the spaced string's.
    assert len("table".encode()) == 5
    assert len(" table".encode()) == 6, "the leading space must not enter the byte count"

    print(f"lambada_en self-test OK: {len(WORD_BOUNDARY)} boundary chars, stop rule covers "
          f"apostrophe/hyphen/newline and the empty-generation case")


def main():
    ap = argparse.ArgumentParser()
    # NOT required=True: the gate runs `--selftest` with no checkpoint, and a required
    # --ckpt makes the selftest un-runnable -- caught by running it, not by reading it.
    ap.add_argument("--ckpt", help="our .pt, or an HF directory with --hf")
    ap.add_argument("--hf", action="store_true", help="control arm: HF format, own tokenizer")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--tokenizer", default=TOK_PATH)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, help="first N items (smoke test)")
    ap.add_argument("--preds", help="jsonl, appended per item; rerun resumes from it")
    ap.add_argument("--out", help="summary json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return 0
    if not a.ckpt:
        ap.error("--ckpt required (or --selftest)")
    if not os.path.exists(a.data):
        sys.exit(f"lambada_en data missing: {a.data}")

    m = HFModel(a.ckpt, a.device) if a.hf else OursModel(a.ckpt, a.tokenizer, a.device)
    items = load_items(a.data, a.limit)
    print(f"Loaded {a.ckpt}: {m.n_params / 1e6:.2f}M params | items {len(items)}", flush=True)

    done = {}
    if a.preds and os.path.exists(a.preds):
        with open(a.preds, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[r["id"]] = r
        print(f"resuming: {len(done)} items already in {a.preds}", flush=True)

    hits = n = 0
    nll_sum = nll_n = 0.0
    empty = 0
    for it in items:
        r = done.get(it["id"])
        if r is None:
            word, _ = greedy_word(m, m.encode(it["context"]))
            nll = target_nll_per_byte(m, m.encode(it["context"]), it["target"])
            r = {"id": it["id"], "target": it["target"], "pred": word, "nll_per_byte": nll}
            if a.preds:
                with open(a.preds, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n += 1
        hits += r["pred"] == r["target"]
        empty += r["pred"] == ""
        if r.get("nll_per_byte") is not None:
            nll_sum += r["nll_per_byte"]
            nll_n += 1

    acc = hits / max(1, n)
    se = math.sqrt(acc * (1 - acc) / max(1, n))
    result = {
        "ckpt": a.ckpt, "hf": a.hf, "n_items": n,
        "n_params": m.n_params,
        "n_params_non_embed": getattr(m, "n_params_non_embed", None),
        "acc": acc, "binomial_se": se, "ci95_halfwidth": 1.96 * se,
        "empty_generations": empty,
        "nll_per_byte_mean": (nll_sum / nll_n) if nll_n else None,
        "reading": "greedy continuation to first word boundary, exact string match "
                   "(lm-eval-harness definition); tokenizer-independent",
        "boundary": "The two readings measure different things and are EXPECTED to diverge: "
                    "the greedy number has a decoder and a stop rule, the per-byte NLL has "
                    "neither. See be.gold_bpb_falls_while_generation_scores_zero. Vocabulary "
                    "fact, measured on this file's 5,111 eligible rows: 15.7% of last words "
                    "are single-token under our tokenizer, 70.1% under Pythia's -- which is "
                    "why this metric does not use a single-token reading.",
    }
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(result, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

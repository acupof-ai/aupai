#!/usr/bin/env python3
"""E1: base continuation generation check.

Contract: docs/standards/eval_ladder_v1_1009.md section 5 (from the v42 Engram
handover doc section 6, E1). The subject is a BASE checkpoint, so the protocol is
raw continuation: the prompt is fed as it appears in pretraining and the model is
never handed a ChatML prefix (the pretraining corpus effectively contains no ChatML,
AGENTS.md "Chat Format"). E0 established that this checkpoint loads and returns
finite logits; E1 asks whether the generator is stable enough for the later stages
to be scored on it.

Checks. The gate is R8's set; `stopped` is recorded and is not a gate.

  protocol    no ChatML special in the prompt (text) and no special token id in the
              prompt's token ids. A ChatML prefix is the wrong protocol for a base
              checkpoint and is the negative control below.
  nonempty    n_new > 0 and the decoded text is not blank. An empty generation is
              identifiable rather than silently scored as a short one.
  no_echo     the generation does not give the whole input back: neither as a
              contiguous run of prompt token ids nor as the whitespace-normalised
              prompt text inside the output.
  no_repeat   token-level, never the decoded string. Two rules and either one fails
              the item (R9): the worst token 8-gram occurs more than
              --repeat-max-count (=2) times, or the repeated material covers more than
              --repeat-span-frac (=0.5) of the generated tokens, where a position is
              covered when it sits inside an 8-gram that occurs more than once. The
              first run's evidence is why the old "count > 4" rule was too loose:
              zh-qa-01 looped the question to 100% coverage and an 8-gram count of 3,
              zh-qa-04 to 100%, en-cont-01 to 70%, and all three were scored pass.
              R9 rules on the coverage ratio, which is the stricter of the two readings
              of "the longest repeated span"; the literal reading -- the longest block
              that occurs twice, over the generated length -- is measured and reported
              beside it as longest_repeat_span_frac, so an auditor can tell them apart.
              The tail loop is also reported as a diagnostic.
  lang        the output's script is the prompt's language (R8's companion check): a
              zh prompt's continuation must be CJK-majority, an en prompt's
              Latin-majority over the letters it contains. Digits, punctuation and
              whitespace vote for neither.
  stopped     RECORDED, NOT GATING (R8). stop_reason comes from ladder_common by token
              id and stopped_rate is in summary.json, but a base checkpoint has no EOS
              cue on a prose prefix -- all 16 items of the first run ended max_new --
              so a terminator requirement measures the incomplete prompt, not the
              model. The per-item stop_reason is in items.jsonl either way.

Generation is train.generate_batch -- the one decoding implementation in this tree --
with greedy argmax (temperature 0) so the gate is reproducible. rep_stop is OFF by
default: that switch stops on a decoded-text 8-gram heuristic, and a stage that
measures repetition at the token level must see the whole generation rather than a
text heuristic's truncation of it. --rep-stop turns it back on for comparison, and
the setting is recorded either way.

Output. items.jsonl carries R3/R10's per-item fields -- id, prompt (verbatim),
response_raw, parsed, passed, stop_reason, n_tokens, fail_class, meta -- with
prompt_sha256 kept as an extra, and every aggregate in summary.json is recomputable
from those rows. `parsed` is this stage's parsed view of the same bytes: a base
continuation has no answer-extraction step, so parsed.text == response_raw, and the
token ids the judge saw are in parsed.gen_ids for an independent recount.

Controls (contract section 6). One frozen positive world, six one-field mutations of
it, and one evidence world, all judged by the same judge() the real run uses:

  positive_known_good        RECORDED_POSITIVE, a generation recorded from a real E1
                             run on this checkpoint (see the constant), or before that
                             a continuation tokenised from a literal. Must PASS every
                             gating check.
  negative_echo_forced       gen_ids := prompt_ids -- the input forced back out.
  negative_chatml_prefix     the prompt carries "<|im_start|>user\n": wrong protocol.
  negative_repeat_loop       a 4-token block appended 6 times: the count rule.
  negative_repeat_span       a 20-token block said twice, so every 8-gram occurs
                             exactly twice and only the span rule can catch it.
  negative_lang_mismatch     a Latin continuation for a Chinese prompt.
  negative_empty_output      gen_ids := []: nothing came back.
  evidence_truncated_max_new one token with cap 1: stop_reason must read max_new and
                             no gating check may fail -- R8's seam, asserted.

control_selftest() also runs skip_forward_control(), which calls this stage's own main()
with --skip-forward and asserts that it REFUSES. A run that validates inputs only is
not a result, and the ladder releases the next stage on this summary's exit_code.

Each negative must fail on its own named check, not merely fail. `--control` runs
these plus a live positive (one real generation on the designated prompt, asserting
the pipeline produced a non-empty judged record -- the frozen world asserts full PASS,
the live one asserts liveness) and writes them into summary.json controls{}; the stage
exits 3 if the controls do not come out as required, whatever the real items did.

Prompt sets. `--prompts` takes either this stage's operational file
(data/eval/ladder/e1_prompts.jsonl, one {"id","lang","kind","prompt"} row per line) or
a cross-check set written as one JSON object {"prompts": [...], "negative_controls":
[...]}. Both go through the same protocol rule and the same judge(); the ids in the
report are the input file's ids. The >= 6 per language composition is enforced on this
stage's own operational file; any other --prompts path is held to the same per-item
rules (lang and kind are inferred from the text when absent) and its composition is
reported in inputs{composition_ok} instead of refusing a set that never claimed it.
The cross-check file's own negative_controls are recorded, not used: E1's controls are
control_worlds() below.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval import ladder_common as lc  # noqa: E402

PROMPTS = ROOT / "data" / "eval" / "ladder" / "e1_prompts.jsonl"
TOKENIZER = ROOT / "data" / "tokenizer.json"
PROTOCOL = "base-continuation"
REQUIRED_KEYS = ("id", "lang", "kind", "prompt")
LANGS = ("zh", "en")
KINDS = ("continuation", "qa")
MIN_PER_LANG = 6
MIN_PROMPT_TOKENS = 4
# ChatML and the document terminator as literal text. A continuation prefix must not
# carry one: the first is another protocol's prefix, the second is a document boundary.
FORBIDDEN_TEXT = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
DEFAULT_MAX_NEW = 64
CONTROL_MAX_NEW = 32
# R9's two thresholds. One source of truth: the CLI defaults and control_selftest read
# these, so the gate the stage runs and the worlds the selftest builds cannot drift apart.
REPEAT_MAX_COUNT = 2
REPEAT_SPAN_FRAC = 0.5
# R9 rules on the coverage ratio. The literal reading -- the longest block that occurs
# twice -- is reported beside it as a diagnostic, so an auditor can tell the two apart.
# The exact longest-repeat scan is quadratic, so it is bounded by this window; every E1
# cap used so far is far inside it.
REPEAT_SCAN_TOKENS = 512
POSITIVE_PROMPT_ID = "en-qa-03"

# The 32,768-slot gate vocabulary's pinned special ids (harness check `pinned_ids`;
# read from data/tokenizer.json, sha256 c6d5eec9..., 2026-10-09). Used only by the
# control worlds; the real run takes these from ladder_common.load_for_eval's meta.
VOCAB_EOS_ID = 1
VOCAB_IM_START_ID = 32763
VOCAB_IM_END_ID = 32764

# The known-good sample for the frozen positive control: item en-qa-03 of the
# operational prompt file, generated greedily by ckpt_v42s2.pt (sha256 4c82366b...,
# vocab_id f1f860970d15d623, step 54051) in this stage's first run, 2026-10-09,
# runs/ladder/ckpt_v42s2/e1/items.jsonl. It passes every gating check -- English
# output on an English prompt, no echo, worst 8-gram count 1, repeated span 0.00 --
# and its stop_reason is max_new, which R8 records and does not gate. Empty means
# "not recorded yet", and the world is then built from the prompt file and a
# tokenised literal instead; both go through judge() identically.
RECORDED_POSITIVE: dict = {
    "prompt_id": "en-qa-03",
    "prompt_sha256": "225f3df9577130a2101061771748a268535454e08ac0bd8dabe889aae5702d1f",
    "prompt_ids": [5603, 27, 1754, 315, 274, 939, 7413, 746, 305, 2556, 506, 11232, 2015, 32, 200, 8474, 27],
    "gen_ids": [457, 939, 7413, 746, 305, 2556, 315, 274, 4748, 506, 723, 2556, 939, 5951, 15, 3270, 11232, 2015, 13, 274, 4748, 315, 6036, 380, 4558, 22066, 329, 1400, 4558, 23281, 488, 200, 200, 1457, 749, 274, 939, 7413, 746, 305, 2556, 506, 11232, 2015, 13, 432, 517, 702, 274, 1853, 27, 200, 200, 11709, 7413, 4427, 305, 11306, 278, 24738, 506, 19651, 3881, 351],
    "response_raw": " The boiling point of water is the temperature at which water boils. At sea level, the temperature is approximately 0 degrees Celsius (32 degrees Fahrenheit).\n\nTo find the boiling point of water at sea level, we can use the formula:\n\nBoiling Point of Water = Temperature at Sea Level -",
    "max_new": 64,
    "ckpt_sha256": "4c82366ba5086adc82605b0cb9fb5ee4a3bc0c0397763295bc7f7252b292169f",
    "recorded_utc": "2026-10-09",
    "note": "ckpt_v42s2.pt first E1 run, item en-qa-03, greedy, cap 64; every gating check passes",
}

# The literal continuation the fallback positive world uses when RECORDED_POSITIVE
# names a prompt this set does not contain (an audit cross-check set, R7): the first
# is in the style of zh_web_dc (weight 0.14 in data/mix_v42_stage2.json), the second
# is English so the world's script still matches an English prompt.
FALLBACK_CONTINUATION = {
    "zh": "大脑的供氧量就会随之减少，人也就容易感到疲倦。",
    "en": "The continuation keeps going in the same language as the prompt it was given.",
}


# --------------------------------------------------------------------------- prompts

_ID_KEYS = ("id", "item_id", "qid", "name", "key")
_PROMPT_KEYS = ("prompt", "text", "question", "input", "content", "prefix")
_LANG_KEYS = ("lang", "language", "locale")
_KIND_KEYS = ("kind", "type", "category", "task")
_QA_MARKERS = ("\u95ee\uff1a", "\u7b54\uff1a", "\u95ee\u9898\uff1a", "\u7b54\u6848\uff1a",
               "Q:", "A:", "Question:", "Answer:", "Problem:", "Solution:")


def _pick(obj, keys):
    low = {str(k).lower(): k for k in obj}
    for key in keys:
        if key in low:
            return obj[low[key]]
    return None


def _infer_lang(text):
    """zh when a fifth of the non-space characters are CJK, else en."""
    body = [c for c in text if not c.isspace()]
    if not body:
        return "en"
    cjk = sum(1 for c in body if "\u4e00" <= c <= "\u9fff")
    return "zh" if cjk / len(body) > 0.2 else "en"


def _infer_kind(text):
    return "qa" if any(m in text for m in _QA_MARKERS) else "continuation"


def _parse_prompt_file(path, text):
    """(items, shape, declared) from either supported input shape.

    Shape 1, this stage's operational file (`data/eval/ladder/e1_prompts.jsonl`):
    one {"id","lang","kind","prompt"} object per line, exactly those four keys.
    Shape 2, a cross-check set: {"prompts": [...]} as one JSON object, whose entries
    may be objects under other key names or bare strings. `declared` is True only when
    every item stated its own lang and kind; an undeclared set gets them inferred from
    the prompt text and is reported as inferred rather than refused for its composition.
    """
    text = text.strip()
    obj = None
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except ValueError:
            obj = None  # not one JSON document, so it is the JSONL shape
    if isinstance(obj, dict) and "prompts" in obj:
        lc.require(isinstance(obj.get("prompts"), list),
                   f"prompt file {path}: 'prompts' must be a list, "
                   f"got {type(obj['prompts']).__name__}")
        items, declared = [], True
        for i, entry in enumerate(obj["prompts"]):
            where = f"{path}:prompts[{i}]"
            if isinstance(entry, str):
                item = {"id": f"p{i + 1:02d}", "prompt": entry}
                declared = False
            else:
                lc.require(isinstance(entry, dict), f"{where} is neither a string nor an object")
                pid = _pick(entry, _ID_KEYS)
                prompt = _pick(entry, _PROMPT_KEYS)
                lc.require(prompt is not None,
                           f"{where} has no prompt field; keys are {sorted(entry)}")
                item = {"id": str(pid) if pid is not None else f"p{i + 1:02d}", "prompt": prompt}
                lang, kind = _pick(entry, _LANG_KEYS), _pick(entry, _KIND_KEYS)
                item["lang"], item["kind"] = lang, kind
                if lang is None or kind is None:
                    declared = False
            # Leading and trailing whitespace is part of a continuation prefix and is
            # meaningful (humaneval's --rstrip_nl arm exists because of it); only a
            # whitespace-only prompt is refused, and the bytes are hashed either way.
            lc.require(isinstance(item["prompt"], str) and item["prompt"].strip(),
                       f"{where} prompt must be a string with at least one non-space character")
            items.append(item)
        lc.require(items, f"prompt file {path}: 'prompts' is empty")
        meta_extra = {"negative_controls_n": len(obj.get("negative_controls") or [])}
        return items, "json_object", declared, meta_extra

    items, declared = [], True
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError as exc:
            raise lc.Refuse(f"prompt file {path}:{i + 1} is not JSON: {exc}") from exc
        lc.require(isinstance(item, dict), f"prompt file {path}:{i + 1} is not an object")
        missing = [k for k in REQUIRED_KEYS if k not in item]
        extra = [k for k in item if k not in REQUIRED_KEYS]
        lc.require(not missing and not extra,
                   f"prompt file {path}:{i + 1} keys {sorted(item)}; "
                   f"need exactly {list(REQUIRED_KEYS)}")
        items.append(item)
    lc.require(items, f"prompt file {path} holds no items")
    return items, "jsonl", declared, {}


def load_prompts(path=None):
    """Read and validate a prompt set. Returns (items, meta) or raises Refuse.

    Both input shapes are read here and are judged by the same protocol and the same
    judge() below; only the transport differs. The file is the stage's data revision:
    order is the file's order, both the file bytes and the ordered id list are hashed
    into the seal, and the ids in every artifact are the input file's own ids. Every
    rule here is a precondition, so a violation REFUSEs (exit 2) rather than scoring a
    prompt set the stage cannot vouch for.
    """
    path = pathlib.Path(path or PROMPTS)
    lc.require(path.is_file(), f"prompt file {path} is missing")
    raw = path.read_bytes()
    try:
        items, shape, declared, meta_extra = _parse_prompt_file(path, raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise lc.Refuse(f"prompt file {path} is not UTF-8: {exc}") from exc

    ids = []
    for i, item in enumerate(items):
        where = f"{path}[{i}]"
        pid = item.get("id")
        lc.require(isinstance(pid, str) and pid.strip() == pid and pid,
                   f"{where} id must be a non-empty string with no outer whitespace")
        text = item["prompt"]
        lc.require(isinstance(text, str) and text.strip(),
                   f"{where} prompt must be a string with at least one non-space character")
        hits = [s for s in FORBIDDEN_TEXT if s in text]
        lc.require(not hits, f"{where} prompt carries a special token as text: {hits}")
        lc.require(pid not in ids, f"{where} duplicate id {pid!r}")
        ids.append(pid)

        lang, kind = item.get("lang"), item.get("kind")
        if lang is None:
            lang = _infer_lang(text)
        else:
            low = str(lang).strip().lower()
            if low.startswith("zh") or low in ("chinese", "cn", "\u4e2d\u6587"):
                lang = "zh"
            elif low.startswith("en") or low in ("english", "\u82f1\u6587"):
                lang = "en"
            else:
                raise lc.Refuse(f"{where} lang {lang!r} is neither zh nor en")
        if kind is None:
            kind = _infer_kind(text)
        else:
            low = str(kind).strip().lower()
            if low in ("qa", "q&a", "question", "question_answer", "\u95ee\u7b54"):
                kind = "qa"
            elif low in ("continuation", "cont", "prose", "\u7eed\u5199"):
                kind = "continuation"
            else:
                raise lc.Refuse(f"{where} kind {kind!r} is neither continuation nor qa")
        item["lang"], item["kind"] = lang, kind

    by_lang = collections.Counter(it["lang"] for it in items)
    by_kind = collections.Counter(it["kind"] for it in items)
    # The >= 6 per language composition is a property of THIS stage's operational set
    # (contract section 5: Chinese and English continuation are both checks). A
    # cross-check set brings its own composition and is reported with composition_ok
    # instead of being refused for a composition it never claimed -- a different set
    # must be scoreable, or the cross-check cannot happen at all.
    enforced = path.resolve() == PROMPTS.resolve()
    composition_ok = all(by_lang[lang] >= MIN_PER_LANG for lang in LANGS) and len(items) >= 2 * MIN_PER_LANG
    if enforced and not composition_ok:
        raise lc.Refuse(f"{path}: {len(items)} items {dict(by_lang)}; the operational set needs "
                        f">= {2 * MIN_PER_LANG} items and >= {MIN_PER_LANG} per language")
    meta = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        "n": len(items),
        "ids": ids,
        "shape": shape,
        "lang_kind_declared": declared,
        "composition_enforced": enforced,
        "composition_ok": composition_ok,
        "by_lang": dict(sorted(by_lang.items())),
        "by_kind": dict(sorted(by_kind.items())),
    }
    meta.update(meta_extra)
    return items, meta


# ---------------------------------------------------------------------------- judge

def contains_run(hay, needle):
    """True when `needle` occurs as a contiguous block inside `hay`."""
    n, m = len(hay), len(needle)
    if m == 0 or m > n:
        return False
    return any(hay[i] == needle[0] and hay[i:i + m] == needle for i in range(n - m + 1))


def longest_common_run(a, b):
    """Length of the longest contiguous run of equal token ids shared by a and b."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def tail_loop(ids, max_period=64, min_reps=3, min_span=24):
    """The trailing exact repetition of a period-p block, or None.

    "The output repeats itself" is a statement about the end of the generation: the
    model fell into a cycle and stayed in it. Requiring the repetition to be exact and
    to reach the end keeps the criterion off ordinary text, where a phrase recurs but
    never as r identical blocks in a row.
    """
    n = len(ids)
    for period in range(1, max_period + 1):
        if period * min_reps > n:
            break
        reps = 1
        block = ids[n - period:]
        while (reps + 1) * period <= n and ids[n - (reps + 1) * period:n - reps * period] == block:
            reps += 1
        if reps >= min_reps and period * reps >= min_span:
            return {"period": period, "reps": reps, "span": period * reps}
    return None


def longest_repeat(ids):
    """Length in tokens of the longest block that occurs at least twice.

    The two occurrences may overlap. Exact for a generation of at most
    REPEAT_SCAN_TOKENS tokens (every E1 cap used so far is far inside that); a longer
    generation is measured on its last REPEAT_SCAN_TOKENS tokens. This is the literal
    reading of "the longest repeated span", which R9 records beside the coverage ratio
    it rules on. Example: range(20) * 2 gives 20 tokens, half the generation.
    """
    window = [int(t) for t in ids][-REPEAT_SCAN_TOKENS:]
    n = len(window)
    best = 0
    prev = [0] * (n + 1)
    for i in range(1, n + 1):
        cur = [0] * (n + 1)
        token = window[i - 1]
        for j in range(1, n + 1):
            if i != j and token == window[j - 1]:
                value = prev[j - 1] + 1
                cur[j] = value
                if value > best:
                    best = value
        prev = cur
    return best


def repeat_stats(ids, ngram_n=8):
    """Token-level repetition evidence, never the decoded string.

    max_ngram_count   how many times the worst token n-gram occurs.
    repeat_span       how many positions sit inside an n-gram that occurs more than
                      once; repeat_span_frac is that over the generated length. R9
                      rules on this: a generation whose repeated material covers most
                      of itself is a loop, however the decoded text happens to read.
    loop              the trailing exact repetition, kept as a diagnostic. R9's two
                      rules above subsume it, so it no longer decides pass or fail.
    """
    ids = list(ids)
    n = len(ids)
    counts = collections.Counter(tuple(ids[i:i + ngram_n]) for i in range(max(0, n - ngram_n + 1)))
    top_gram, top_count = (None, 0)
    if counts:
        top_gram, top_count = counts.most_common(1)[0]
    covered = bytearray(n)
    for i in range(max(0, n - ngram_n + 1)):
        if counts[tuple(ids[i:i + ngram_n])] > 1:
            covered[i:i + ngram_n] = b"\x01" * ngram_n
    span = int(sum(covered))
    longest = longest_repeat(ids) if n else 0
    return {
        "ngram_n": ngram_n,
        "max_ngram_count": top_count,
        "max_ngram": list(top_gram) if top_gram else None,
        "repeat_span": span,
        "repeat_span_frac": round(span / n, 4) if n else 0.0,
        "longest_repeat_span": longest,
        "longest_repeat_span_frac": round(longest / n, 4) if n else 0.0,
        "repeat_scan_tokens": REPEAT_SCAN_TOKENS,
        "loop": tail_loop(ids),
    }


def lang_counts(text):
    """CJK and Latin letter counts, and the language they vote for.

    Digits, punctuation and whitespace vote for neither: the question is which
    script the continuation is written in, and a continuation written in no script
    at all is not the prompt's language.
    """
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    detected = "zh" if cjk > latin else "en" if latin > cjk else None
    return {"cjk": cjk, "latin": latin, "detected": detected}


def judge(rec, cfg):
    """The stage's verdict on one generation. Pure: no torch, no filesystem.

    rec: id, lang, kind, prompt, prompt_ids, gen_ids, text, n_new, max_new, stop_reason,
    and the tokenizer's special ids. cfg: echo_max_overlap, repeat_ngram,
    repeat_max_count, repeat_span_frac.

    Gating checks decide passed/fail_class. `stopped` is judged and reported but marked
    gating=False (R8): it is evidence about the prompt, not a gate on the model.
    """
    checks = []

    def add(name, ok, detail, gating=True):
        checks.append({"check": name, "ok": bool(ok), "detail": detail, "gating": gating})

    prompt = rec["prompt"]
    prompt_ids = list(rec.get("prompt_ids") or [])
    gen_ids = list(rec.get("gen_ids") or [])
    text = rec.get("text") or ""

    hits = [s for s in FORBIDDEN_TEXT if s in prompt]
    bad_ids = sorted({k: rec[k] for k in ("eos_id", "im_start_id", "im_end_id")
                      if rec.get(k) is not None and rec[k] in prompt_ids}.items())
    add("protocol", not hits and not bad_ids,
        f"special_text={hits} special_ids_in_prompt={bad_ids}")

    add("nonempty", rec["n_new"] > 0 and bool(text.strip()),
        f"n_new={rec['n_new']} text_chars={len(text.strip())}")

    full_run = contains_run(gen_ids, prompt_ids)
    run = longest_common_run(gen_ids, prompt_ids)
    ratio = round(run / len(prompt_ids), 4) if prompt_ids else 0.0
    norm_prompt = " ".join(prompt.split())
    norm_gen = " ".join(text.split())
    text_hit = len(norm_prompt) >= 16 and norm_prompt in norm_gen
    add("no_echo", not (ratio >= cfg["echo_max_overlap"] or text_hit),
        f"prompt_tokens={len(prompt_ids)} longest_run={run} ratio={ratio} "
        f"full_run={full_run} text_contains={text_hit}")

    rep = repeat_stats(gen_ids, cfg["repeat_ngram"])
    add("no_repeat",
        rep["max_ngram_count"] <= cfg["repeat_max_count"]
        and rep["repeat_span_frac"] <= cfg["repeat_span_frac"],
        f"max_ngram_count={rep['max_ngram_count']}<={cfg['repeat_max_count']} "
        f"repeat_span_frac={rep['repeat_span_frac']}<={cfg['repeat_span_frac']} "
        f"longest_repeat_span_frac={rep['longest_repeat_span_frac']} "
        + json.dumps({k: rep[k] for k in ("ngram_n", "repeat_span", "longest_repeat_span")},
                     sort_keys=True))

    counts = lang_counts(text)
    add("lang", counts["detected"] == rec["lang"],
        f"prompt_lang={rec['lang']} detected={counts['detected']} "
        f"cjk={counts['cjk']} latin={counts['latin']}")

    stop_ok = rec["stop_reason"] in ("eos", "im_end")
    add("stopped", stop_ok,
        f"stop_reason={rec['stop_reason']} n_new={rec['n_new']} max_new={rec['max_new']} "
        f"last_token={gen_ids[-1] if gen_ids else None}", gating=False)

    failed = [c["check"] for c in checks if c["gating"] and not c["ok"]]
    return {
        "id": rec["id"],
        "passed": not failed,
        "checks": {c["check"]: c["ok"] for c in checks},
        "gating": {c["check"]: c["ok"] for c in checks if c["gating"]},
        "check_rows": checks,
        "failed": failed,
        "fail_class": failed[0] if failed else None,
        "stop_reason": rec["stop_reason"],
        "stop_ok": stop_ok,
        "n_new": rec["n_new"],
        "echo": {"longest_run": run, "ratio": ratio, "full_run": full_run, "text_contains": text_hit},
        "repeat": rep,
        "lang": counts,
    }


# ------------------------------------------------------------------------- controls

def _small_tokenizer():
    """The real tokenizer, or None when this tree has no data/tokenizer.json.

    Control worlds must not require a GPU or the pod, and the criteria they exercise
    are sequence predicates; the tokenizer only makes the world's text field real.
    """
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    if not TOKENIZER.is_file():
        return None
    return Tokenizer.from_file(str(TOKENIZER))


def _world(item, gen_ids, tok, max_new, stop_reason=None, prompt_text=None, prompt_ids=None,
           text=None):
    """One control world as judge() sees it, with stop_reason derived from token ids."""
    prompt_text = item["prompt"] if prompt_text is None else prompt_text
    if prompt_ids is None:
        prompt_ids = (list(tok.encode(prompt_text).ids) if tok is not None
                      else list(range(200, 224)))
    gen_ids = [int(t) for t in gen_ids]
    if text is None:
        text = tok.decode(gen_ids) if tok is not None else ""
    n_new = len(gen_ids)
    if stop_reason is None:
        stop_reason = lc.stop_reason(gen_ids[-1] if gen_ids else None,
                                     VOCAB_EOS_ID, VOCAB_IM_END_ID, n_new, max_new)
    return {
        "id": item["id"],
        "lang": item["lang"],
        "kind": item["kind"],
        "prompt": prompt_text,
        "prompt_ids": prompt_ids,
        "gen_ids": gen_ids,
        "text": text,
        "n_new": n_new,
        "max_new": max_new,
        "stop_reason": stop_reason,
        "eos_id": VOCAB_EOS_ID,
        "im_start_id": VOCAB_IM_START_ID,
        "im_end_id": VOCAB_IM_END_ID,
    }


def control_worlds(items, max_new=DEFAULT_MAX_NEW):
    """(positive_world, negatives, evidence, provenance).

    The positive world is the recorded known-good generation when RECORDED_POSITIVE
    names a prompt in this file, and otherwise a continuation tokenised from
    FALLBACK_CONTINUATION. Each negative is that same world with exactly one field
    changed, so a negative can only die on the criterion it targets. The evidence
    worlds assert a recorded verdict that is NOT a gate (R8's stop_reason), which is a
    different kind of claim from a negative and is reported as one.
    """
    tok = _small_tokenizer()
    item = next((it for it in items if it["id"] == POSITIVE_PROMPT_ID), items[0])
    literal = FALLBACK_CONTINUATION[item["lang"]]
    recorded = dict(RECORDED_POSITIVE or {})
    if recorded.get("prompt_id") == item["id"] and recorded.get("gen_ids"):
        gen_ids = [int(t) for t in recorded["gen_ids"]]
        max_new = int(recorded.get("max_new", max_new))
        source = "recorded"
        note = recorded.get("note", "")
    else:
        if tok is not None:
            gen_ids = list(tok.encode(literal).ids) + [VOCAB_EOS_ID]
            source = "tokenized_literal"
        else:
            # No tokenizer here: ids in the vocabulary's real range, and the text field
            # is the literal itself so `nonempty` and `lang` are answered by it.
            digest = hashlib.sha256(literal.encode("utf-8")).digest()
            gen_ids = [200 + (b % 30000) for b in digest] + [VOCAB_EOS_ID]
            source = "synthetic_ids"
        note = "fallback: no recorded positive for this prompt"
    positive = _world(item, gen_ids, tok, max_new)
    if source == "recorded" and recorded.get("response_raw"):
        positive["text"] = str(recorded["response_raw"])
    elif source == "synthetic_ids":
        positive["text"] = literal
    if recorded.get("prompt_ids") and source == "recorded":
        positive["prompt_ids"] = [int(t) for t in recorded["prompt_ids"]]

    # A block said twice: every 8-gram inside it occurs exactly twice, so the count rule
    # cannot fire and only the span rule can. This is the world that says whether the
    # second half of R9 does any work. Both blocks are in the positive prompt's language
    # so that `lang` is not disturbed and no_repeat is the only check that can fire.
    span_block = {
        "zh": "\u8fd9\u6bb5\u6587\u5b57\u4f1a\u88ab\u5b8c\u6574\u91cd\u590d\u4e00\u904d\uff0c"
              "\u7528\u6765\u5236\u9020\u5360\u6bd4\u5f88\u9ad8\u7684\u91cd\u590d\u7a7a\u95f4",
        "en": "This block of text is written out twice on purpose to make a large repeated span",
    }[item["lang"]]
    if tok is not None:
        span_ids = list(tok.encode(span_block).ids) * 2
        span_text = span_block * 2
    else:
        span_ids = [1000 + i for i in range(20)] * 2
        span_text = span_block * 2
    # A continuation in the OTHER language: non-empty, non-repeating, no echo, and in
    # the wrong script for the prompt it answers.
    mismatch_text = {
        "zh": "The answer continues in English even though the prompt is Chinese.",
        "en": "\u8fd9\u6bb5\u7eed\u5199\u7528\u4e2d\u6587\uff0c\u5c3d\u7ba1\u63d0\u793a\u662f\u82f1\u6587\u7684\u3002",
    }[item["lang"]]
    mismatch_desc = ("a Latin continuation for a Chinese prompt" if item["lang"] == "zh"
                     else "a Chinese continuation for an English prompt")

    # (name, the check that must fail, world, note, extra expectations). `extra` asserts
    # the property that makes the world discriminating rather than merely failing:
    # negative_repeat_span is only evidence for R9's second rule if the count rule did
    # NOT fire on it.
    negatives = [
        ("negative_echo_forced", "no_echo",
         _world(item, list(positive["prompt_ids"]), tok, max_new,
                prompt_text=positive["prompt"], text=positive["prompt"]),
         "gen_ids := prompt_ids: the whole input handed back", {}),
        ("negative_chatml_prefix", "protocol",
         _world(item, list(positive["gen_ids"]), tok, max_new,
                prompt_text="<|im_start|>user\n" + positive["prompt"],
                prompt_ids=[VOCAB_IM_START_ID] + list(positive["prompt_ids"])),
         "ChatML prefix on a base protocol prompt", {}),
        ("negative_repeat_loop", "no_repeat",
         _world(item, list(positive["gen_ids"]) + [101, 102, 103, 104] * 6, tok, max_new),
         "a 4-token block appended 6 times: the count rule, not the span rule",
         {"max_ngram_count_above": REPEAT_MAX_COUNT}),
        ("negative_repeat_span", "no_repeat",
         _world(item, span_ids, tok, max_new, text=None if tok is None else span_text),
         "the whole generation is one block said twice: every 8-gram occurs exactly "
         "twice, so only the span rule can fire",
         {"max_ngram_count_at_most": REPEAT_MAX_COUNT,
          "repeat_span_frac_above": REPEAT_SPAN_FRAC,
          "longest_below_coverage": True}),
        ("negative_lang_mismatch", "lang",
         _world(item, (list(tok.encode(mismatch_text).ids) if tok is not None
                       else list(range(500, 530))) + [VOCAB_EOS_ID], tok, max_new,
                text=None if tok is not None else mismatch_text),
         mismatch_desc + ", in a clean world otherwise", {}),
        ("negative_empty_output", "nonempty",
         _world(item, [], tok, max_new, stop_reason="other"),
         "gen_ids := []: nothing generated", {}),
    ]
    evidence = [
        ("evidence_truncated_max_new",
         "stop_reason == max_new and no gating check fails (R8: recorded, not gated)",
         _world(item, gen_ids[:1], tok, 1,
                text=(positive["text"].split()[0] if positive["text"].split() else literal)),
         "one token with cap 1: a forced truncation, no terminator"),
    ]
    provenance = {
        "prompt_id": item["id"],
        "source": source,
        "note": note,
        "max_new": max_new,
        "gen_tokens": len(positive["gen_ids"]),
        "prompt_tokens": len(positive["prompt_ids"]),
        "text_chars": len(positive["text"]),
        "tokenizer": str(TOKENIZER) if tok is not None else None,
    }
    return positive, negatives, evidence, provenance


def run_controls(items, cfg, live=None, max_new=DEFAULT_MAX_NEW):
    """Judge the frozen worlds, plus an optional live world. Returns (controls, failures).

    A negative is only satisfied when its named check is among the failing ones: a
    world that dies on something else proves nothing about the criterion it was built
    for. The live world (a real generation on the designated prompt) asserts only that
    the pipeline produced a non-empty judged record -- full stability on that one
    prompt is what the stage's own gates measure, not what a control asserts.
    """
    positive, negatives, evidence, provenance = control_worlds(items, max_new=max_new)
    controls = {"positive_world": provenance, "live": live or {"ran": False}}
    failures = []

    res = judge(positive, cfg)
    controls["positive_known_good"] = {
        "criterion": "every gating check passes", "ok": res["passed"],
        "failed": res["failed"], "fail_class": res["fail_class"],
        "checks": res["gating"], "stop_reason": res["stop_reason"],
    }
    if not res["passed"]:
        failures.append(f"positive world failed on {res['failed']}")

    for name, expect, world, note, extra in negatives:
        res = judge(world, cfg)
        ok = expect in res["failed"]
        why = [] if ok else [f"did not fail on {expect} (failed: {res['failed']})"]
        if "max_ngram_count_above" in extra:
            bound = extra["max_ngram_count_above"]
            if not res["repeat"]["max_ngram_count"] > bound:
                ok = False
                why.append(f"max_ngram_count={res['repeat']['max_ngram_count']} is not "
                           f"above {bound}, so the count rule did not fire")
        if "max_ngram_count_at_most" in extra:
            bound = extra["max_ngram_count_at_most"]
            if res["repeat"]["max_ngram_count"] > bound:
                ok = False
                why.append(f"the count rule fired too (max_ngram_count="
                           f"{res['repeat']['max_ngram_count']} > {bound}), so this world "
                           f"proves nothing about the span rule")
        if extra.get("longest_below_coverage"):
            cov = res["repeat"]["repeat_span_frac"]
            lon = res["repeat"]["longest_repeat_span_frac"]
            if not lon < cov:
                ok = False
                why.append(f"longest_repeat_span_frac={lon} is not below the coverage "
                           f"ratio {cov}, so this world does not separate the two readings")
        if "repeat_span_frac_above" in extra:
            bound = extra["repeat_span_frac_above"]
            if not res["repeat"]["repeat_span_frac"] > bound:
                ok = False
                why.append(f"repeat_span_frac={res['repeat']['repeat_span_frac']} is not "
                           f"above {bound}")
        controls[name] = {
            "criterion": f"must fail {expect}", "ok": ok, "note": note,
            "failed": res["failed"], "fail_class": res["fail_class"],
            "checks": res["gating"],
            "repeat": {k: res["repeat"][k] for k in ("max_ngram_count", "repeat_span_frac")},
            "lang": res["lang"],
        }
        failures.extend(f"{name} {w}" for w in why)

    for name, criterion, world, note in evidence:
        res = judge(world, cfg)
        ok = res["stop_reason"] == "max_new" and not res["failed"]
        controls[name] = {
            "criterion": criterion, "ok": ok, "note": note,
            "stop_reason": res["stop_reason"], "failed": res["failed"],
            "passed": res["passed"],
        }
        if not ok:
            failures.append(f"{name}: stop_reason={res['stop_reason']} failed={res['failed']}")

    if live and live.get("ran"):
        res = judge(live["rec"], cfg)
        ok = "nonempty" not in res["failed"] and live["rec"]["stop_reason"] in (
            "eos", "im_end", "max_new", "other")
        controls["live"] = {
            "criterion": "non-empty judged record", "ok": ok, "ran": True,
            "passed": res["passed"], "failed": res["failed"], "fail_class": res["fail_class"],
            "stop_reason": live["rec"]["stop_reason"], "n_new": live["rec"]["n_new"],
            "text_head": live["rec"]["text"][:120], "checks": res["gating"],
        }
        if not ok:
            failures.append("live positive produced no judged non-empty output")

    controls["ok"] = not failures
    controls["failures"] = failures
    return controls, failures


def skip_forward_control(probe=None):
    """(ok, detail): the --skip-forward path must never report success.

    A run that only validates inputs is not a result, and the ladder releases the next
    stage on this summary's exit_code. This control calls the stage's own main() with
    --skip-forward against an existing file that is not a checkpoint (the prompt file),
    then reads back the summary it wrote. No GPU and no model are needed.
    """
    import io
    import shutil

    probe = pathlib.Path(probe or PROMPTS)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = main(["--ckpt", str(probe), "--skip-forward"])
    scratch = lc.LADDER / lc.ckpt_tag(probe)
    summary_path = scratch / "e1" / "summary.json"
    got = {}
    with contextlib.suppress(OSError, ValueError):
        got = json.loads(summary_path.read_text(encoding="utf-8"))
    inside = str(scratch.resolve()).startswith(str(lc.LADDER.resolve()) + os.sep)
    if inside:
        shutil.rmtree(scratch, ignore_errors=True)
    ok = (rc == lc.EXIT_REFUSE and got.get("evaluated") is False
          and got.get("skipped_forward") is True and bool(got.get("refused")))
    return ok, {
        "criterion": f"rc == {lc.EXIT_REFUSE} and evaluated == false in the summary",
        "rc": rc, "evaluated": got.get("evaluated"),
        "skipped_forward": got.get("skipped_forward"),
        "refused": (got.get("refused") or "")[:140],
        "scratch_removed": inside,
        "stdout_tail": buf.getvalue().strip()[-160:],
    }


# ------------------------------------------------------------------ stage plumbing

def _claim_info():
    """Which claim file this process wrote, if any. Read back, never assumed.

    The pid is compared as the JSON field it is. Searching the file text for a quoted
    pid finds nothing -- the record writes `"pid": 1121407`, not `"1121407"` -- and the
    first version of this function therefore reported no claim while the process held
    card 0 (caught by reading the summary's own claim block against
    `runs/claims/eval_ladder.0.json`).
    """
    d = ROOT / "runs" / "claims"
    out = {"dir": str(d), "matched": [], "cards": [],
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    if not d.is_dir():
        return out
    pid = os.getpid()
    for p in sorted(d.glob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict) and rec.get("pid") == pid:
            out["matched"].append(p.name)
            out["cards"] = list(rec.get("cards") or [])
            out["record"] = rec
    return out


def _gpu_info(device):
    info = {"device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    try:
        import torch
        if str(device).startswith("cuda") and torch.cuda.is_available():
            info["name"] = torch.cuda.get_device_name(device)
            info["index"] = int(torch.device(device).index or 0)
            info["capability"] = list(torch.cuda.get_device_capability(device))
    except Exception as exc:  # a machine without torch still writes a summary
        info["probe_error"] = str(exc)
    return info


def main(argv):
    ap = argparse.ArgumentParser(prog="eval_ladder.py e1")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new", type=int, default=DEFAULT_MAX_NEW,
                    help="fixed generation cap; every item is judged against it")
    ap.add_argument("--control", action="store_true",
                    help="run the frozen control worlds and a live positive; not passing exits 3")
    ap.add_argument("--prompts", default=str(PROMPTS),
                    help="prompt set: this stage's JSONL by default, or a cross-check set as a "
                         "JSON object {\"prompts\": [...]}; both are judged identically and the "
                         "ids reported are the input file's")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N items (debug)")
    ap.add_argument("--rep-stop", action="store_true",
                    help="let train.generate_batch also stop on its decoded-text 8-gram rule")
    ap.add_argument("--echo-max-overlap", type=float, default=1.0,
                    help="fail no_echo at this fraction of the prompt echoed back (1.0 = the whole input)")
    ap.add_argument("--repeat-max-count", type=int, default=REPEAT_MAX_COUNT,
                    help="R9: fail no_repeat when the worst token 8-gram occurs more than this")
    ap.add_argument("--repeat-span-frac", type=float, default=REPEAT_SPAN_FRAC,
                    help="R9: fail no_repeat when repeated material covers more than this "
                         "fraction of the generated tokens")
    ap.add_argument("--skip-forward", action="store_true",
                    help="validate the prompt set and the checkpoint path only; no model, no "
                         "generation (the dispatcher's refuse-path control calls this)")
    args = ap.parse_args(argv)

    def stage(paths, log):
        items, pmeta = load_prompts(args.prompts)
        lc.require(args.max_new >= 1, f"--max-new must be >= 1, got {args.max_new}")
        run_items = items[: args.limit] if args.limit > 0 else items
        lc.require(run_items, "--limit selected no items")
        cfg = {"echo_max_overlap": args.echo_max_overlap, "repeat_ngram": 8,
               "repeat_max_count": args.repeat_max_count,
               "repeat_span_frac": args.repeat_span_frac}

        log.write(f"prompts {pmeta['path']} shape={pmeta['shape']} sha256={pmeta['sha256']} "
                  f"n={pmeta['n']} langs={pmeta['by_lang']} kinds={pmeta['by_kind']} "
                  f"lang_kind_declared={pmeta['lang_kind_declared']} "
                  f"composition_ok={pmeta['composition_ok']}\n")
        if pmeta["shape"] != "jsonl":
            log.write("cross-check set: its own negative_controls are not used; E1's controls "
                      "are control_worlds() over this same judge\n")
        if args.skip_forward:
            # Inputs only, and NEVER a pass. The ladder releases the next stage on this
            # summary's exit_code, so a zero here would put a later stage on top of an
            # empty evaluation. The checkpoint must still exist: E1 is a statement about
            # a model, and a summary that says "no checkpoint" is the refuse path.
            lc.require(pathlib.Path(args.ckpt).is_file(), f"checkpoint {args.ckpt} does not exist")
            log.write("REFUSE: prompt set validated, nothing evaluated (--skip-forward)\n")
            lc.write_items(paths, [])
            return {
                "ckpt": args.ckpt, "protocol": PROTOCOL, "n": 0, "passed": 0,
                "skipped_forward": True, "evaluated": False,
                "refused": "prompt set validated, nothing evaluated (--skip-forward); "
                           "run the stage without --skip-forward for a verdict",
                "gates": {"evaluated": False},
                "inputs": {"prompts": pmeta["path"], "prompts_sha256": pmeta["sha256"],
                           "ids_sha256": pmeta["ids_sha256"], "ids": [it["id"] for it in run_items],
                           "n_prompts": pmeta["n"], "shape": pmeta["shape"],
                           "lang_kind_declared": pmeta["lang_kind_declared"],
                           "composition_ok": pmeta["composition_ok"],
                           "composition_enforced": pmeta["composition_enforced"],
                           "negative_controls_n": pmeta.get("negative_controls_n"),
                           "by_lang": pmeta["by_lang"], "by_kind": pmeta["by_kind"]},
                "exit_code": lc.EXIT_REFUSE,
            }
        log.write(f"scoring {len(run_items)} items, max_new={args.max_new}, greedy, "
                  f"rep_stop={bool(args.rep_stop)}\n")
        lc.seal(paths, ckpt=args.ckpt, stage="e1", protocol=PROTOCOL, device=args.device,
                max_new=args.max_new, temperature=0.0, rep_stop=bool(args.rep_stop),
                control=bool(args.control), prompts=pmeta["path"],
                prompts_sha256=pmeta["sha256"], ids_sha256=pmeta["ids_sha256"],
                ids=[it["id"] for it in run_items], shard_i=0, shard_n=1,
                stage_started_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        model, tok, meta = lc.load_for_eval(args.ckpt, device=args.device, claim=True)
        # train.generate_batch stops on its own module constant _EOS == 1; a tokenizer
        # whose terminator is not 1 would make the generator and ladder_common's token-id
        # rule disagree about what "stopped" means, and the gate would read the wrong one.
        lc.require(meta["eos_id"] == 1,
                   f"tokenizer terminator id is {meta['eos_id']}, generate_batch stops on 1")
        lc.require(not getattr(meta["cfg"], "fone", False),
                   "FoNE checkpoint: E1 has no digit head and would read every [NUM] as zero")
        prompt_ids = [list(tok.encode(it["prompt"]).ids) for it in run_items]
        for it, ids in zip(run_items, prompt_ids, strict=True):
            lc.require(len(ids) >= MIN_PROMPT_TOKENS,
                       f"prompt {it['id']} tokenized to {len(ids)} tokens")
            bad = [k for k, v in (("eos", meta["eos_id"]), ("im_start", meta["im_start_id"]),
                                  ("im_end", meta["im_end_id"]))
                   if v is not None and v in ids]
            lc.require(not bad, f"prompt {it['id']} carries special token ids {bad}")

        lc.seal(paths, ckpt=args.ckpt, stage="e1", protocol=PROTOCOL, device=args.device,
                max_new=args.max_new, temperature=0.0, rep_stop=bool(args.rep_stop),
                control=bool(args.control), prompts=pmeta["path"],
                prompts_sha256=pmeta["sha256"], ids_sha256=pmeta["ids_sha256"],
                ids=[it["id"] for it in run_items], shard_i=0, shard_n=1,
                ckpt_sha256=meta["ckpt_sha256"], step=meta["step"], vocab_id=meta["vocab_id"],
                vocab_real=meta["vocab_real"], eos_id=meta["eos_id"],
                im_start_id=meta["im_start_id"], im_end_id=meta["im_end_id"],
                gpu=_gpu_info(args.device), claim=_claim_info(),
                stage_started_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        from train import generate_batch
        t_start = time.time()
        t0 = time.time()
        generated = generate_batch(model, prompt_ids, args.max_new, args.device,
                                   tokenizer=tok, rep_stop=bool(args.rep_stop))
        decode_s = round(time.time() - t0, 2)
        log.write(f"generate_batch: {len(generated)} rows, cap {args.max_new}, {decode_s}s\n")

        results, rows, recs = [], [], []
        for item, pids, gids in zip(run_items, prompt_ids, generated, strict=True):
            gids = [int(t) for t in gids]
            text = tok.decode(gids)
            n_new = len(gids)
            stop = lc.stop_reason(gids[-1] if gids else None, meta["eos_id"],
                                  meta["im_end_id"], n_new, args.max_new)
            rec = {"id": item["id"], "lang": item["lang"], "kind": item["kind"],
                   "prompt": item["prompt"], "prompt_ids": pids, "gen_ids": gids,
                   "text": text, "n_new": n_new, "max_new": args.max_new,
                   "stop_reason": stop, "eos_id": meta["eos_id"],
                   "im_start_id": meta["im_start_id"], "im_end_id": meta["im_end_id"]}
            res = judge(rec, cfg)
            res["prompt_lang"] = item["lang"]
            res["kind"] = item["kind"]
            recs.append(rec)
            results.append(res)
            # R3/R10 field names: id, prompt (verbatim), response_raw, parsed, passed,
            # stop_reason, n_tokens, fail_class, meta; prompt_sha256 kept as an extra.
            # Summary aggregates must be recomputable from these rows, so every judged
            # number is here.
            rows.append({
                "id": item["id"],
                "prompt": item["prompt"],
                "prompt_sha256": hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest(),
                "response_raw": text,
                "parsed": {
                    "text": text,
                    "gen_ids": gids,
                    "n_tokens": n_new,
                    "last_token": gids[-1] if gids else None,
                    "detected_lang": res["lang"]["detected"],
                    "lang_counts": {"cjk": res["lang"]["cjk"], "latin": res["lang"]["latin"]},
                },
                "passed": res["passed"],
                "stop_reason": stop,
                "n_tokens": n_new,
                "fail_class": res["fail_class"],
                "failed": res["failed"],
                "meta": {
                    "stage": "e1", "protocol": PROTOCOL,
                    "lang": item["lang"], "kind": item["kind"],
                    "prompt_tokens": len(pids),
                    "max_new": args.max_new, "seed": None, "temperature": 0.0,
                    "greedy": True, "rep_stop": bool(args.rep_stop),
                    "decode_s": decode_s, "elapsed_s": round(time.time() - t_start, 1),
                    "ckpt_sha256": meta["ckpt_sha256"], "step": meta["step"],
                    "vocab_id": meta["vocab_id"],
                    "checks": res["gating"],
                    "stop_ok": res["stop_ok"],
                    "echo": res["echo"], "repeat": res["repeat"],
                    "repeat_numbers": {
                        "max_ngram_count": res["repeat"]["max_ngram_count"],
                        "repeat_span_frac": res["repeat"]["repeat_span_frac"],
                        "longest_repeat_span_frac": res["repeat"]["longest_repeat_span_frac"],
                    },
                    "repeat_criteria": {"max_ngram_count": args.repeat_max_count,
                                        "repeat_span_frac": args.repeat_span_frac},
                },
            })
            log.write(f"{'PASS' if res['passed'] else 'FAIL'} {item['id']} "
                      f"[{item['lang']}/{item['kind']}] n_new={n_new} stop={stop} "
                      f"failed={res['failed']}\n    out={text[:160]!r}\n")

        n = len(results)
        gating_names = ("protocol", "nonempty", "no_echo", "no_repeat", "lang")
        counts = collections.Counter(name for r in results for name in gating_names
                                     if not r["gating"][name])
        fail_classes = collections.Counter(r["fail_class"] for r in results if r["failed"])
        stop_counts = collections.Counter(r["stop_reason"] for r in results)
        passed = sum(1 for r in results if r["passed"])
        stop_rate = round(sum(stop_counts[k] for k in ("eos", "im_end")) / n, 4)
        gates = {name: counts[name] == 0 for name in gating_names}
        # R8: the termination rate is recorded and never gated -- a base checkpoint has
        # no EOS cue on a prose prefix, so it measures the prompt, not the model.
        gates["generation_stable"] = all(gates[name] for name in gating_names)

        controls = {"ran": False, "ok": True}
        if args.control:
            # The live positive is the designated prompt's OWN judged row when that
            # prompt is in this run: same batch, same conditions, no extra GPU work.
            # Re-generating it alone would pay a fresh torch.compile for a new batch
            # shape (measured 186 s elapsed against 16 s of decoding on the first run).
            idx = next((i for i, it in enumerate(run_items)
                        if it["id"] == POSITIVE_PROMPT_ID), None)
            reused = idx is not None
            if reused:
                live = {"ran": True, "rec": recs[idx]}
            else:
                pos_item = run_items[0]
                pos_ids = [list(tok.encode(pos_item["prompt"]).ids)]
                live_cap = max(1, min(args.max_new, CONTROL_MAX_NEW))
                live_gen = [int(t) for t in generate_batch(
                    model, pos_ids, live_cap, args.device, tokenizer=tok,
                    rep_stop=bool(args.rep_stop))[0]]
                live = {"ran": True, "rec": {
                    "id": pos_item["id"], "lang": pos_item["lang"], "kind": pos_item["kind"],
                    "prompt": pos_item["prompt"], "prompt_ids": pos_ids[0], "gen_ids": live_gen,
                    "text": tok.decode(live_gen), "n_new": len(live_gen), "max_new": live_cap,
                    "stop_reason": lc.stop_reason(live_gen[-1] if live_gen else None,
                                                  meta["eos_id"], meta["im_end_id"],
                                                  len(live_gen), live_cap),
                    "eos_id": meta["eos_id"], "im_start_id": meta["im_start_id"],
                    "im_end_id": meta["im_end_id"]}}
            controls, ctrl_failures = run_controls(run_items, cfg, live=live,
                                                   max_new=args.max_new)
            controls["ran"] = True
            controls["live"]["reused_item_row"] = reused
            sf_ok, sf_detail = skip_forward_control()
            controls["skip_forward_refuses"] = dict(sf_detail, ok=sf_ok)
            if not sf_ok:
                controls["ok"] = False
                controls["failures"] = list(controls["failures"]) + [
                    f"skip_forward_refuses: rc={sf_detail['rc']} "
                    f"evaluated={sf_detail['evaluated']}"]
            log.write(f"controls ok={controls['ok']} failures={ctrl_failures} "
                      f"(live positive {'reuses the designated item row' if reused else 'generated separately'})\n")
        exit_code = lc.EXIT_OK if (gates["generation_stable"] and controls["ok"]) else lc.EXIT_GATE

        log.write(f"E1 n={n} passed={passed} stopped_rate={stop_rate} (recorded, not gated, R8) "
                  f"fail_class_counts={dict(counts)} stop_counts={dict(stop_counts)}\n")
        log.write(f"E1 gates {json.dumps(gates, sort_keys=True)}\n")
        log.write("E1 gate PASSED\n" if exit_code == lc.EXIT_OK else "E1 gate FAILED\n")

        summary = {
            "ckpt": args.ckpt, "ckpt_sha256": meta["ckpt_sha256"], "step": meta["step"],
            "protocol": PROTOCOL, "n": n, "passed": passed, "gates": gates,
            "inputs": {"prompts": pmeta["path"], "prompts_sha256": pmeta["sha256"],
                       "ids_sha256": pmeta["ids_sha256"], "ids": [it["id"] for it in run_items],
                       "n_prompts": pmeta["n"], "shape": pmeta["shape"],
                       "lang_kind_declared": pmeta["lang_kind_declared"],
                       "composition_ok": pmeta["composition_ok"],
                       "composition_enforced": pmeta["composition_enforced"],
                       "negative_controls_n": pmeta.get("negative_controls_n"),
                       "by_lang": pmeta["by_lang"], "by_kind": pmeta["by_kind"]},
            "config": {"max_new": args.max_new, "temperature": 0.0, "seed": None,
                       "rep_stop": bool(args.rep_stop), "limit": args.limit,
                       "echo_max_overlap": args.echo_max_overlap,
                       "repeat_max_count": args.repeat_max_count,
                       "repeat_span_frac": args.repeat_span_frac,
                       "gating_checks": list(gating_names),
                       "stopped_gating": False},
            "gpu": _gpu_info(args.device), "pid": os.getpid(), "claim": _claim_info(),
            "shards": [{"shard_i": 0, "shard_n": 1, "n": n, "gpu": args.device,
                        "ids": [it["id"] for it in run_items]}],
            "decode_s": decode_s, "elapsed_s": round(time.time() - t_start, 1),
            "stopped_rate": stop_rate, "stop_reason_counts": dict(stop_counts),
            "fail_counts": dict(counts),
            "stop_ok_count": sum(1 for r in results if r["stop_ok"]),
            "stop_reason_per_item": {r["id"]: r["stop_reason"] for r in results},
            "repeat_span_frac_per_item": {r["id"]: r["repeat"]["repeat_span_frac"] for r in results},
            "max_ngram_count_per_item": {r["id"]: r["repeat"]["max_ngram_count"] for r in results},
            "repeat_stats_per_item": {r["id"]: r["repeat"] for r in results},
            "failed_ids": [r["id"] for r in results if not r["passed"]],
            "fail_classes": dict(fail_classes),
            "by_lang_passed": {lang: sum(1 for r in results
                                         if r["passed"] and r["prompt_lang"] == lang)
                               for lang in ("zh", "en")},
            "by_lang_n": {lang: sum(1 for r in results if r["prompt_lang"] == lang)
                          for lang in ("zh", "en")},
            "vocab_id": meta["vocab_id"], "eos_id": meta["eos_id"],
            "im_start_id": meta["im_start_id"], "im_end_id": meta["im_end_id"],
            "controls": controls,
            "items": {r["id"]: {"passed": r["passed"], "failed": r["failed"],
                                "fail_class": r["fail_class"],
                                "stop_reason": r["stop_reason"], "stop_ok": r["stop_ok"],
                                "n_new": r["n_new"]}
                      for r in results},
            "exit_code": exit_code,
        }
        lc.write_items(paths, rows)
        return summary

    return lc.run_stage("e1", stage, ckpt=args.ckpt)


# ------------------------------------------------------------------------- selftest

def control_selftest():
    """CPU: the frozen control worlds over the real prompt file. No checkpoint, no GPU."""
    failures = []
    try:
        items, pmeta = load_prompts(PROMPTS)
    except lc.Refuse as exc:
        print(f"FAIL e1 control: {exc}", file=sys.stderr)
        return lc.EXIT_GATE
    cfg = {"echo_max_overlap": 1.0, "repeat_ngram": 8,
           "repeat_max_count": REPEAT_MAX_COUNT, "repeat_span_frac": REPEAT_SPAN_FRAC}
    controls, failures = run_controls(items, cfg, live=None)
    sf_ok, sf_detail = skip_forward_control()
    if not sf_ok:
        failures.append(f"--skip-forward control: rc={sf_detail['rc']} "
                        f"evaluated={sf_detail['evaluated']}")
    prov = controls["positive_world"]
    print(f"e1 controls: prompts n={pmeta['n']} sha256={pmeta['sha256'][:12]} "
          f"positive={prov['prompt_id']} source={prov['source']} "
          f"gen_tokens={prov['gen_tokens']}")
    for name in ("positive_known_good", "negative_echo_forced", "negative_chatml_prefix",
                 "negative_repeat_loop", "negative_repeat_span", "negative_lang_mismatch",
                 "negative_empty_output", "evidence_truncated_max_new"):
        c = controls[name]
        print(f"  {'PASS' if c['ok'] else 'FAIL'} {name}: criterion={c['criterion']} "
              f"failed={c['failed']} stop={c.get('stop_reason')}")
    print(f"  {'PASS' if sf_ok else 'FAIL'} skip_forward_refuses: criterion="
          f"{sf_detail['criterion']} rc={sf_detail['rc']} "
          f"evaluated={sf_detail['evaluated']}")
    for f in failures:
        print(f"FAIL e1 control: {f}", file=sys.stderr)
    if failures:
        return lc.EXIT_GATE
    n_neg = sum(1 for k in controls if k.startswith("negative_"))
    print(f"PASS e1 controls: 1 positive world (source={prov['source']}), {n_neg} negative worlds, "
          f"1 evidence world; gates nonempty/protocol/no_echo/no_repeat/lang with R9 "
          f"count>{cfg['repeat_max_count']} or span>{cfg['repeat_span_frac']}")
    return lc.EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

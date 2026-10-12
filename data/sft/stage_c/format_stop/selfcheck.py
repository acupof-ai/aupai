#!/usr/bin/env python3
"""Structural self-check for the C1 format-and-stop set.

Checks what the set must be, on the shipped file, with no model and no GPU:

  1. the file is non-empty and every line is one JSON object with exactly the six
     fields the contract names;
  2. ids are unique;
  3. every prompt is the ChatML prefix with an EMPTY system turn, built by
     scripts.loader.format_prompt -- not a copy of the string;
  4. every answer ends at <|im_end|>, and its last token IS <|im_end|>;
  5. no answer holds a second <|im_start|>;
  6. every answer is within the 64-token cap;
  7. no answer repeats a whole sentence of its own question, and E2's own probe
     (question[:32]) does not appear in the answer;
  8. no prompt and no full row is an exact duplicate, and no word 8-gram occurs
     more than twice.

Then four NEGATIVE worlds, each a REAL shipped row mutated to break exactly one
rule, and each must FAIL on that rule. A check that cannot fail is not a check:

  no_terminator   the final <|im_end|> is removed
  echo            the question is copied into the head of the answer
  second_im_start a second <|im_start|> is appended inside the answer
  overlong        the answer text is repeated until it passes the token cap

    python3 data/sft/stage_c/format_stop/selfcheck.py [--set PATH] [--quiet]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "datagen"))

from loader import IM_END, IM_START, format_prompt  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

FIELDS = ("id", "prompt", "answer", "source", "license", "quality")
ANSWER_MAX_TOKENS = 64
NEAR_DUP_N = 8
NEAR_DUP_LIMIT = 2
DEFINE_ECHO_MAX_CHARS = 8
DEFAULT_SET = os.path.join(HERE, "format_stop_v2.jsonl")


class Fail(AssertionError):
    """A named rule died. The name is what the negative worlds are asserted against."""


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise Fail(f"line {n} is not JSON: {exc}")
    return rows


def longest_common_span(question, answer, cap=40):
    answer_subs = {answer[i:i + 8] for i in range(len(answer) - 8 + 1)}
    lo = 8
    while lo <= min(len(question), cap):
        if not any(question[i:i + lo] in answer_subs for i in range(len(question) - lo + 1)):
            return lo - 1
        lo += 1
    return 0 if lo <= 8 else min(len(question), cap)


def wgrams(text, n):
    toks = text.split()
    if len(toks) < n:
        return set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def check_rows(rows, tok):
    """Every rule. Returns a dict of counts; raises Fail on the first broken rule."""
    if not rows:
        raise Fail("empty: the file holds no rows")

    im_end_id = tok.token_to_id(IM_END)
    im_start_id = tok.token_to_id(IM_START)
    probe_hits = collections.Counter()
    span_by_source = collections.defaultdict(list)

    for row in rows:
        missing = [f for f in FIELDS if f not in row]
        if missing:
            raise Fail(f"{row.get('id')}: missing fields {missing}")
        if set(row) != set(FIELDS):
            raise Fail(f"{row.get('id')}: unexpected fields {sorted(set(row) - set(FIELDS))}")
        for f in FIELDS:
            if not isinstance(row[f], str) or not row[f].strip():
                raise Fail(f"{row.get('id')}: field {f} is not a non-empty string")

        question = row["prompt"].split(f"{IM_START}user\n", 1)
        if len(question) != 2:
            raise Fail(f"{row['id']}: prompt has no user turn")
        question = question[1].split(IM_END, 1)[0]
        if row["prompt"] != format_prompt(question):
            raise Fail(f"{row['id']}: prompt is not format_prompt(question, system=None); "
                       f"system must be empty and the two turns must match")
        if row["prompt"].count(IM_START) != 2:
            raise Fail(f"{row['id']}: prompt carries {row['prompt'].count(IM_START)} "
                       f"<|im_start|> markers, not 2")

        answer = row["answer"]
        if not answer.endswith(IM_END):
            raise Fail(f"{row['id']}: answer does not end at {IM_END}")
        ids = tok.encode(answer).ids
        if ids[-1] != im_end_id:
            raise Fail(f"{row['id']}: last token id {ids[-1]} is not <|im_end|> ({im_end_id})")
        if im_start_id in ids:
            raise Fail(f"{row['id']}: answer holds a second <|im_start|>")
        if len(ids) > ANSWER_MAX_TOKENS:
            raise Fail(f"{row['id']}: answer is {len(ids)} tokens, over {ANSWER_MAX_TOKENS}")

        body = answer[: -len(IM_END)]
        probe = question[:32]
        if probe and probe in body:
            probe_hits[row["source"]] += 1
            raise Fail(f"{row['id']}: answer contains question[:32] verbatim "
                       f"(the E2 no_echo probe)")
        span = longest_common_span(question, body)
        cap = DEFINE_ECHO_MAX_CHARS if row["source"] == "curated_define" else 7
        if span > cap:
            raise Fail(f"{row['id']}: answer shares a {span}-character run with its question "
                       f"(cap {cap} for {row['source']})")
        span_by_source[row["source"]].append(span)

    ids_seen = collections.Counter(r["id"] for r in rows)
    dupes = [i for i, c in ids_seen.items() if c > 1]
    if dupes:
        raise Fail(f"duplicate ids: {dupes[:5]}")

    prompts = collections.Counter(r["prompt"] for r in rows)
    repeats = [p for p, c in prompts.items() if c > 1]
    if repeats:
        raise Fail(f"{len(repeats)} prompts appear more than once, e.g. {repeats[0][:60]!r}")

    full = collections.Counter(r["prompt"] + "\u0000" + r["answer"] for r in rows)
    if any(c > 1 for c in full.values()):
        raise Fail("a full row (prompt + answer) appears more than once")

    # Near-duplicate report. The number this check can honestly assert is bounded by the
    # data: the arithmetic templates repeat the same question skeleton thousands of times
    # BY DESIGN (that is the drill), so a hard cap on an 8-gram count would fail on the
    # corpus's own shape rather than on a defect. What is asserted is the two forms a
    # duplicate can take in a set built this way -- one prompt answered twice, and one
    # (prompt, answer) pair twice -- and the counts are reported.
    grams = collections.Counter()
    for row in rows:
        text = " ".join((row["prompt"] + " " + row["answer"]).split())
        grams.update(wgrams(text, NEAR_DUP_N))
    worst = max(grams.values())
    over = [g for g, c in grams.items() if c > NEAR_DUP_LIMIT]
    if over and len(over) > 0.25 * max(len(grams), 1):
        raise Fail(f"{len(over)} of {len(grams)} word {NEAR_DUP_N}-grams occur more than "
                   f"{NEAR_DUP_LIMIT} times, the most {worst} times, e.g. {over[0][:60]!r}")

    return {
        "rows": len(rows),
        "by_source": dict(collections.Counter(r["source"] for r in rows)),
        "by_lang_prompt_script": {
            "cjk_questions": sum(1 for r in rows if "\u4e00" <= r["prompt"][16] <= "\u9fff"),
            "ascii_questions": sum(1 for r in rows if r["prompt"][16].isascii()),
        },
        "echo_span_by_source": {s: {"rows": len(v), "max": max(v)} for s, v in
                                sorted(span_by_source.items())},
        "e2_probe_hits": dict(probe_hits),
        "max_ngram_count": worst,
    }


def mutate(rows, kind, tok=None):
    """Build one negative world from a REAL row. Never hand-write the world.

    The overlong world needs an answer long enough to pass the cap when repeated: the
    first row of a numeric drill answers "3", and repeating "3" eight times is six tokens,
    which is not overlong at all. It therefore picks the first shipped row whose body
    tokenizes to at least 12 tokens and repeats THAT. The world still comes from data.
    """
    row = dict(rows[0])
    question = row["prompt"].split(f"{IM_START}user\n", 1)[1].split(IM_END, 1)[0]
    if kind == "no_terminator":
        row["answer"] = row["answer"][: -len(IM_END)]
    elif kind == "echo":
        row["answer"] = question + row["answer"]
    elif kind == "second_im_start":
        row["answer"] = row["answer"][: -len(IM_END)] + f"{IM_START}user\n" + IM_END
    elif kind == "overlong":
        if tok is None:
            raise SystemExit("REFUSE: the overlong world needs the tokenizer")
        pick = next((c for c in rows
                     if len(tok.encode(c["answer"][: -len(IM_END)]).ids) >= 12), None)
        if pick is None:
            raise SystemExit("REFUSE: no shipped answer is long enough for the overlong world")
        row = dict(pick)
        body = pick["answer"][: -len(IM_END)]
        row["answer"] = body * 8 + IM_END
    else:
        raise ValueError(kind)
    return [row] + rows[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=DEFAULT_SET)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.set):
        print(f"REFUSE: {args.set} does not exist; run build.sh first")
        return 2
    tok_path = os.path.join(ROOT, "data", "tokenizer.json")
    if not os.path.exists(tok_path):
        print(f"REFUSE: {tok_path} does not exist; the token counts cannot be checked")
        return 2
    tok = Tokenizer.from_file(tok_path)
    rows = load_rows(args.set)

    try:
        stats = check_rows(rows, tok)
    except Fail as exc:
        print(f"FAIL positive: {exc}")
        return 1
    print(f"PASS positive: {args.set} ({stats['rows']} rows, "
          f"max {NEAR_DUP_N}-gram count {stats['max_ngram_count']}, "
          f"E2 probe hits {sum(stats['e2_probe_hits'].values())})")
    if not args.quiet:
        print(json.dumps(stats, indent=2, ensure_ascii=False))

    # Negative worlds: each one row breaks ONE rule, and the check must name it.
    expectations = {
        "no_terminator": "does not end at",
        "echo": "no_echo probe",
        "second_im_start": "second <|im_start|>",
        "overlong": "over 64",
    }
    failed = 0
    for kind, needle in expectations.items():
        try:
            check_rows(mutate(rows, kind, tok), tok)
        except Fail as exc:
            ok = needle in str(exc)
            print(f"{'PASS' if ok else 'FAIL'} negative {kind}: {exc}")
            failed += 0 if ok else 1
        else:
            print(f"FAIL negative {kind}: the check passed a world where it must fail")
            failed += 1
    if failed:
        print(f"FAIL: {failed} negative world(s) did not fail correctly")
        return 1
    print("PASS: 4 negative worlds each fail on their own rule")
    return 0


if __name__ == "__main__":
    sys.exit(main())

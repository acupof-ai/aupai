#!/usr/bin/env python3
"""Is ChatML anywhere in the pretraining corpus? (de, 2026-09-01)

fb asked for the smallest measurement separating "ChatML is under-represented so the
model does not know it" from "ChatML is represented and actively steers away from
code". The answer is neither, and it needed no model:

    42 domains x 4000 rows = 168,000 rows scanned
    <|im_start|> occurrences: 0

The chat domain (wiki_chat) is 问：/答： plain text -- 4000 of 4000 rows -- and carries
no ChatML at all.

Why this matters more than the metric that prompted it. AGENTS.md states "the
pretraining chat domain renders in ChatML too, so SFT does not teach the format from
nothing". That is false of this corpus. eval/code_zh.py and eval/math_zh.py both
prompt through scripts/loader.format_prompt, which emits
<|im_start|>user...<|im_start|>assistant. So every base-checkpoint code-500 and
math-500 number was taken by handing the model a token sequence that appears NOWHERE
in its training data and scoring the continuation.

That is stronger than "the prompt suppresses code". The model is not steered away
from code; it is handed a prefix it has no distribution over, and falls back to what
an unfamiliar prefix produces -- repeating the input, or drifting to web boilerplate.
Both are visible in the generations. It also explains the measured gap without a
competition story: continuation prompts (题目：...解答：) are in-distribution, ChatML
is not.

    2586 ChatML generations: 41 fences (1.6%), 7 "def " (0.3%)
    497 continuation generations, 1-shot: 469 "def " (94.4%)

BOUNDARY, and it is a bound rather than a zero (tilerl's correction, 2026-09-01).
This reads the first `--rows` lines of the first shard per domain, so it cannot say
"zero occurrences in the corpus" -- it says the rate is below what 4000 draws could
miss. Rule of three: 0 of 4000 bounds a domain's ChatML rate at **< 0.075% (95%)**,
and 0 of 168,000 pooled bounds it at < 0.0018%. wiki_chat holds 372,827 rows, so the
per-domain bound still permits up to ~279 ChatML rows there.

That is more than enough to rule on -- a chat format present in under 0.1% of a chat
domain is not a format the model learned -- and it is the honest form. "Zero" invites
one counterexample to overturn it; a bound does not.

    ~/bin/pod "cd /work/aupai && python3 probes/chatml_in_corpus.py"
    python3 probes/chatml_in_corpus.py --rows 20000 --domain chat

# restartable: read-only over corpus shards, prints a table. Seconds. No GPU.
"""

import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The exact string scripts/loader.format_prompt emits. Not a regex and not a
#: near-match: the question is whether the model has seen THIS token, and a fuzzy
#: match would answer a different one.
CHATML = "<|im_start|>"
#: The format the chat domain actually uses, measured. Counted alongside so a zero on
#: CHATML is readable as "a different format is present" rather than "the scan is
#: broken" -- a check that can only report absence cannot distinguish the two.
QA = ("问：", "答：")
FENCE = "```"


def scan(root, rows_cap, only=None):
    out = []
    for d in sorted(glob.glob(os.path.join(root, "data", "corpus", "*/"))):
        name = os.path.basename(d.rstrip("/"))
        if only and name != only:
            continue
        files = sorted(glob.glob(os.path.join(d, "*.jsonl")))
        if not files:
            continue
        n = im = qa = fence = 0
        for f in files:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    n += 1
                    if CHATML in line:
                        im += 1
                    if any(q in line for q in QA):
                        qa += 1
                    if FENCE in line:
                        fence += 1
                    if n >= rows_cap:
                        break
            if n >= rows_cap:
                break
        out.append((name, n, im, qa, fence))
    return out


def selftest():
    """A scan that can only report zero is indistinguishable from a broken scan.

    This is the known-answer pair: a row that DOES contain ChatML must be counted.
    Without it, a scan with a typo'd needle reports 0 everywhere and reads as the
    finding it was written to look for -- the most dangerous possible failure for
    this particular script.
    """
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        dom = os.path.join(d, "data", "corpus", "fake")
        os.makedirs(dom)
        with open(os.path.join(dom, "a.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": f"{CHATML}user\nhi"}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"text": "问：x\n答：y"}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"text": "```python\npass\n```"}, ensure_ascii=False) + "\n")
        got = scan(d, 100)
        assert len(got) == 1, got
        name, n, im, qa, fence = got[0]
        assert (n, im, qa, fence) == (3, 1, 1, 1), (
            f"the scan must FIND ChatML when it is present: got {got[0]}. A scan that "
            f"only ever reports 0 cannot be told apart from a broken one.")
    print("selftest OK: the scan counts ChatML, 问答 and fences when they are there")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4000, help="rows per domain")
    ap.add_argument("--domain", default=None, help="one domain only")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    got = scan(a.root, a.rows, a.domain)
    if not got:
        print(f"no corpus domains under {a.root}/data/corpus -- run this on the pod")
        return 1
    print(f"{'domain':<22}{'rows':>8}{'ChatML':>9}{'QA':>8}{'fence':>8}")
    for name, n, im, qa, fence in got:
        print(f"{name:<22}{n:>8}{im:>9}{qa:>8}{fence:>8}")
    tot_im = sum(x[2] for x in got)
    tot_n = sum(x[1] for x in got)
    print(f"\n{tot_im} ChatML occurrences in {tot_n} rows across {len(got)} domains")
    if tot_im == 0 and tot_n:
        # Rule of three: 0 hits in n draws bounds the rate at ~3/n with 95%
        # confidence. Reported as a bound, never as "zero in the corpus" -- this
        # samples the head of one shard per domain, and a zero claim is overturned by
        # a single counterexample where a bound is not (tilerl, 2026-09-01).
        print(f"ChatML rate bounded at < {3 / tot_n:.4%} pooled (95%, rule of three); "
              f"< {3 / min(x[1] for x in got):.3%} for the smallest domain sampled.")
        print("So: ChatML is not a format the base learned. Every base-checkpoint eval "
              "prompted through loader.format_prompt handed the model a prefix it has "
              "no distribution over; those numbers measure response to that, not "
              "capability -- re-label them, do not re-read them.")
        print("BOUNDARY: first shard, first --rows lines per domain. A sample, not a census.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Generate reasoning traces from the 27B for sequence-level distillation into k6.

**Why not token-level reverse KL.** Reverse KL, D(student || teacher), is the right
objective for a small student -- it is mode-seeking, so the student concentrates on
what the teacher is confident about instead of trying to cover a distribution it has
no capacity for. It is also **undefined here**: the teacher is Qwen3 with its own
vocabulary and the student has 32,773 tokens of its own. `P(token_i)` does not refer
to the same object on both sides, so there is no KL to compute. tileRL's server also
returns `"logprobs": None`.

What is left, and what R1-distill does: the teacher writes solutions, the student
trains on the text. That is sequence-level distillation, and it is tokenizer-blind.

The mode-seeking property can be recovered later without a shared vocabulary by
sampling k solutions per problem and keeping only those that reach the right answer
-- rejection sampling is a hard-thresholded approximation of the same idea, and it
needs no logprobs at all. `--k` and `--keep_correct` do that here.

    python datagen/distil_traces.py --problems data/synthetic/math_hard_train.jsonl \\
        --n 20000 --k 4 --keep_correct --out data/synthetic/distil_v1.jsonl
"""

import argparse
import concurrent.futures as cf
import glob
import json
import random
import re
import urllib.error
import urllib.request

PROMPT = """请用中文解答下面的数学题。

思考过程也请全部用中文，不要用英文。
最后给出解答：每行一个算式，算式必须算对，最后一行写 答案是：\\boxed{{结果}}

题目：{q}"""


def split_think(txt):
    """(reasoning, answer). The reasoning is the point: this is a distillation of
    HOW the teacher solves, not of what it concludes. An earlier version of this
    file discarded the think block and would have produced ordinary SFT data
    wearing the name of reasoning distillation.

    A block that never closed means the generation was truncated mid-reasoning --
    that is not a trace and must not be used."""
    if txt.lstrip().startswith("<think>") and "</think>" not in txt:
        return None, None
    if "</think>" not in txt:
        return "", txt.strip()
    head, body = txt.split("</think>", 1)
    return head.replace("<think>", "").strip(), body.strip()


def ask(url, model, prompt, max_tokens, temperature, timeout=600):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"]


BOXED = re.compile(r"\\boxed\{([^}]*)\}")


def final_answer(text):
    m = BOXED.findall(text or "")
    return m[-1].strip() if m else None


def norm(x):
    """Compare answers as numbers where possible so 24 and 24.0 agree."""
    if x is None:
        return None
    x = x.strip().replace(" ", "").replace(",", "").rstrip("。.")
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True, help="jsonl with instruction/output or q/answer")
    ap.add_argument("--urls", default=",".join(f"http://127.0.0.1:807{g}" for g in range(6)))
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--k", type=int, default=1, help="samples per problem; >1 enables rejection")
    ap.add_argument("--keep_correct", action="store_true", help="keep only traces reaching the gold answer")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=2048, help="must cover the whole think block")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="data/synthetic/distil_v1.jsonl")
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(a.problems)):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    q = d.get("instruction") or d.get("q") or d.get("question")
                    gold = d.get("answer") or d.get("output")
                    if q:
                        rows.append((q, gold))
    random.Random(0).shuffle(rows)
    rows = rows[: a.n]
    urls = [u.strip() for u in a.urls.split(",") if u.strip()]
    print(f"{len(rows)} problems, {a.k} sample(s) each, {len(urls)} teachers", flush=True)

    jobs = [(i, j) for i in range(len(rows)) for j in range(a.k)]

    def one(job):
        i, j = job
        q, gold = rows[i]
        try:
            raw = ask(urls[(i + j) % len(urls)], a.model, PROMPT.format(q=q), a.max_tokens, a.temperature)
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
            return None
        think, body = split_think(raw or "")
        if body is None or not body:
            return None
        return {
            "instruction": q,
            "think": think,
            "output": body,
            "gold": gold,
            "pred": final_answer(body),
        }

    kept = seen = 0
    with (
        open(a.out, "w", encoding="utf-8") as o,
        cf.ThreadPoolExecutor(a.workers) as ex,
    ):
        for r in ex.map(one, jobs):
            seen += 1
            if r is None:
                continue
            if a.keep_correct:
                # Rejection sampling stands in for the mode-seeking term that reverse
                # KL would supply if the vocabularies matched.
                if r["pred"] is None or norm(r["pred"]) != norm(r["gold"]):
                    continue
            kept += 1
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
            if kept % 500 == 0:
                print(f"  {kept} kept / {seen} sampled ({100 * kept / seen:.0f}%)", flush=True)
    print(f"{kept} traces kept of {seen} sampled ({100 * kept / max(1, seen):.0f}%) -> {a.out}")


def _demo():
    assert final_answer("步骤\n答案是：\\boxed{42}") == "42"
    assert final_answer("\\boxed{1} then \\boxed{7}") == "7", "must take the LAST boxed answer"
    assert final_answer("no answer here") is None
    assert norm("24.0") == norm("24") == "24"
    assert norm("8/3") == "8/3"
    assert split_think("<think>\n算一下 3 + 4") == (None, None), "a truncated block is not a trace"
    th, bd = split_think("<think>\n先求总量\n</think>\n\n答案是：\\boxed{7}")
    assert th == "先求总量" and bd.endswith("{7}"), (th, bd)
    assert split_think("no think block at all") == ("", "no think block at all")
    print("distil_traces self-test OK")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _demo()
    else:
        main()

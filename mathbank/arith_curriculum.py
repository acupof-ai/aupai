#!/usr/bin/env python3
"""Bare arithmetic in the formats that are known to be learnable.

Measured on ckpt_k7_v3: **0/180 on two-digit addition, subtraction and
multiplication** with no word problem attached. It is not computing at all -- given
`59 + 63 = ` it emits `00\\n答案是：\\boxed`, filling the slot after the equals sign
with the most common continuation. That is why 51.2% of the equations inside its
solutions are wrong while 91% of generations still produce one: it learned the
shape of a solution, which is cheap, and not the computation, which is not.

Lee et al., *Teaching Arithmetic to Small Transformers* (arXiv 2307.03381), give
the sample counts for NanoGPT (10.6M parameters, twenty times smaller than this
model) to reach ~100% on three-digit addition:

    plain      `123+456=579`      never converges, plateaus near 85%
    reverse    least digit first  ~2,500 samples
    simplified scratchpad         ~2,000 samples
    detailed   scratchpad         ~1,000 samples

A thousand samples. This model has 0.19B scheduled math tokens and cannot add two
digits, so the problem was never volume -- it is that plain `a+b=c` is the one
format that does not work, and it is the only format the corpus contains.

Why reverse works: carries propagate from the least significant digit upward, so
emitting that digit first means every step depends only on what has already been
written. Left-to-right forces the model to know a carry before computing it.

    python mathbank/arith_curriculum.py --n 200000 --out data/synthetic/arith_v1.jsonl
    python mathbank/arith_curriculum.py --selftest
"""

import argparse
import json
import random
import sys


def _rev(n):
    """Digits least-significant first, space separated so BPE cannot merge them."""
    return " ".join(reversed(str(abs(n))))


def plain(a, b, op):
    v = {"+": a + b, "-": a - b, "*": a * b}[op]
    sym = {"+": "+", "-": "-", "*": "×"}[op]
    return f"{a} {sym} {b} = ", str(v)


def reverse(a, b, op):
    """Answer written least-significant digit first. The prompt keeps normal order
    so the model reads the question the way the corpus writes it."""
    v = {"+": a + b, "-": a - b, "*": a * b}[op]
    sym = {"+": "+", "-": "-", "*": "×"}[op]
    sign = "-" if v < 0 else ""
    return f"{a} {sym} {b} = ", f"{sign}{_rev(v)}"


def scratchpad_add(a, b):
    """Digit-by-digit with the carry written down, least significant first.

    This is the format with the best sample efficiency in the paper (~1,000
    examples for three digits) because every line is a one-digit problem whose
    inputs are all already on the page.
    """
    da, db = list(str(a))[::-1], list(str(b))[::-1]
    n = max(len(da), len(db))
    da += ["0"] * (n - len(da))
    db += ["0"] * (n - len(db))
    lines, carry = [], 0
    for i in range(n):
        x, y = int(da[i]), int(db[i])
        s = x + y + carry
        lines.append(f"{x} + {y} + 进位{carry} = {s}，写 {s % 10}，进位 {s // 10}")
        carry = s // 10
    if carry:
        lines.append(f"最高位进位 {carry}，写 {carry}")
    body = "\n".join(lines)
    return f"{a} + {b} = ", f"\n{body}\n结果 = {a + b}"


def scratchpad_sub(a, b):
    """Borrowing, written down, least significant first. a >= b by construction."""
    da, db = list(str(a))[::-1], list(str(b))[::-1]
    db += ["0"] * (len(da) - len(db))
    lines, borrow = [], 0
    for i in range(len(da)):
        x, y = int(da[i]), int(db[i])
        t = x - y - borrow
        if t < 0:
            lines.append(f"{x} - {y} - 借位{borrow} = {t}，不够减，借 10 得 {t + 10}，写 {t + 10}，借位 1")
            borrow = 1
        else:
            lines.append(f"{x} - {y} - 借位{borrow} = {t}，写 {t}，借位 0")
            borrow = 0
    body = "\n".join(lines)
    return f"{a} - {b} = ", f"\n{body}\n结果 = {a - b}"


FORMATS = ("plain", "reverse", "scratchpad")

# Three formats trained on identical prompts leave the model guessing which one
# to answer in, and it hedges: measured on ckpt_k6_arith3, only 17% of
# generations emit <eos> -- it writes the right answer and then keeps going in
# another format. The BPE control terminated at 21%, so this is not the number
# representation, it is the ambiguity. A tag in the prompt removes it.
TAGS = {"plain": "[答]", "reverse": "[逆]", "scratchpad": "[竖式]"}


def held_out(a, b, op):
    """A deterministic train/test split ON THE PROBLEM, not on the row.

    Without this the two-digit space (90x90 = 8,100 pairs) is exhausted long
    before 200,000 samples, so a probe drawing from the same space scores
    memorisation. Measured on arith_v1: 138 of 180 probe cases (77%) were in the
    training set, a median of 3 times each and one 197 times.

    Hashing the problem rather than sampling rows means the split is stable across
    regenerations and cannot leak by reshuffling."""
    import hashlib

    h = hashlib.blake2b(f"{a}{op}{b}".encode(), digest_size=4).digest()
    return int.from_bytes(h, "little") % 10 == 0  # 10% held out


def generate(n, rng, digits=3, formats=FORMATS, split="train", tag=False, strip_eq=False):
    """A curriculum: 1-digit through `digits`-digit, every format, balanced.

    split="train" emits only problems that hash to the training side, "test" only
    the held-out ones, "all" ignores the split. tag=True prefixes the prompt with
    the format marker so the answer is determined by the question.

    strip_eq=True removes the trailing `= ` from SCRATCHPAD prompts only. Measured
    on ckpt_k6_arith4: prompted `[竖式] 61 + 48 = ` the model answers
    `109，0 + 1 + 进位0 = 1，写 0...` -- the result FIRST and the working after it,
    as decoration. A scratchpad is only sample-efficient because it IS the
    computation (Lee et al. 2307.03381); answered this way the mechanism never
    runs. The cause is the prompt: a third of the training data is `a + b = `
    followed immediately by the answer, so `= ` is a shortcut slot the model can
    fill without working. Removing the slot removes the shortcut -- plain and
    reverse keep theirs, so they stay comparable to round 4 as internal controls."""
    out = []
    guard = 0
    while len(out) < n:
        guard += 1
        if guard > n * 50:
            break
        d = rng.randint(1, digits)
        lo, hi = 10 ** (d - 1) if d > 1 else 0, 10**d - 1
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        fmt = rng.choice(formats)
        op = rng.choice(["+", "-", "*"]) if d <= 2 else rng.choice(["+", "-"])
        if op == "-" and a < b:
            a, b = b, a
        if fmt == "scratchpad" and op == "*":
            fmt = "reverse"  # long multiplication needs its own scratchpad; not yet
        if fmt == "plain":
            q, ans = plain(a, b, op)
        elif fmt == "reverse":
            q, ans = reverse(a, b, op)
        else:
            q, ans = scratchpad_add(a, b) if op == "+" else scratchpad_sub(a, b)
        if split != "all" and held_out(a, b, op) != (split == "test"):
            continue
        q = q.strip()
        if strip_eq and fmt == "scratchpad":
            q = q.rstrip("= ")
        q = f"{TAGS[fmt]} {q}" if tag else q
        out.append({"instruction": q, "output": ans, "fmt": fmt, "op": op, "digits": d})
    return out


def _demo():
    """Every format must reproduce the true value, or the curriculum teaches errors."""
    rng = random.Random(0)
    rows = generate(4000, rng, digits=4, split="all")
    tr = generate(300, random.Random(1), digits=3, split="train")
    te = generate(300, random.Random(2), digits=3, split="test")
    keys = lambda rs: {r["instruction"] for r in rs}
    assert not (keys(tr) & keys(te)), "train and test share a problem"
    print(f"split check: {len(keys(tr))} train / {len(keys(te))} test problems, 0 shared")
    tg = generate(300, random.Random(3), digits=3, split="train", tag=True)
    assert all(r["instruction"].startswith(TAGS[r["fmt"]]) for r in tg), "tag does not match fmt"
    assert len({r["instruction"].split()[0] for r in tg}) == 3, "not all three tags emitted"
    print(f"tag check: {len(tg)} rows, every prompt carries its own format tag")
    se = generate(400, random.Random(4), digits=3, split="train", tag=True, strip_eq=True)
    for r in se:
        if r["fmt"] == "scratchpad":
            assert not r["instruction"].rstrip().endswith("="), r["instruction"]
        else:
            assert r["instruction"].rstrip().endswith("="), r["instruction"]
    n_sp = sum(r["fmt"] == "scratchpad" for r in se)
    assert n_sp, "no scratchpad rows drawn; strip_eq untested"
    print(f"strip_eq check: {n_sp}/{len(se)} scratchpad prompts lost `=`, others kept it")
    for r in rows:
        q, out = r["instruction"], r["output"]
        lhs, _ = q.split("=")
        a, sym, b = lhs.split()
        a, b = int(a), int(b)
        truth = {"+": a + b, "-": a - b, "×": a * b}[sym]
        if r["fmt"] == "plain":
            assert int(out) == truth, (q, out, truth)
        elif r["fmt"] == "reverse":
            neg = out.startswith("-")
            got = int(("-" if neg else "") + "".join(out.lstrip("-").split())[::-1])
            assert got == truth, (q, out, truth)
        else:
            assert out.rstrip().endswith(str(truth)), (q, out[-40:], truth)
            # every scratchpad line must itself be arithmetically true
            for ln in out.strip().split("\n")[:-1]:
                if "进位" in ln and "最高位" not in ln:
                    head = ln.split("=")[0]
                    x, y, c = (int(t) for t in head.replace("进位", " ").replace("+", " ").split())
                    assert int(ln.split("=")[1].split("，")[0]) == x + y + c, ln
    n_by = {}
    for r in rows:
        n_by[r["fmt"]] = n_by.get(r["fmt"], 0) + 1
    print(f"arith_curriculum self-test OK ({len(rows)} rows verified, {n_by})")
    print("\nsamples:")
    for f in FORMATS:
        r = next(x for x in rows if x["fmt"] == f)
        print(f"  [{f}] {r['instruction']}  ->  {r['output'][:90]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--digits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["train", "test", "all"], default="train")
    ap.add_argument("--tag", action="store_true", help="prefix the prompt with its format marker")
    ap.add_argument("--strip_eq", action="store_true", help="drop `= ` from scratchpad prompts")
    ap.add_argument("--out", default="data/synthetic/arith_v1.jsonl")
    a = ap.parse_args()
    rows = generate(
        a.n, random.Random(a.seed), digits=a.digits, split=a.split, tag=a.tag, strip_eq=a.strip_eq
    )
    with open(a.out, "w", encoding="utf-8") as o:
        for r in rows:
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
    chars = sum(len(r["instruction"]) + len(r["output"]) for r in rows)
    print(f"{len(rows)} rows -> {a.out} ({chars / 1e6:.1f}M chars, ~{chars / 1.5e6:.1f}M tokens)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _demo()
    else:
        main()

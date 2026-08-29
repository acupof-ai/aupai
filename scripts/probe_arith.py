"""Re-score both probes with 'gold anywhere in the generation'.

"First number emitted" scores format choice, not correctness: the BPE control answers a
bare `61 + 48 = ` with scratchpad prose, so its first number is a scratchpad digit.
"""

import argparse, random, re, sys, torch

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "mathbank")
import fone
from loader import load_checkpoint, load_tokenizer
from arith_curriculum import held_out

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt")
ap.add_argument("--tokenizer")
ap.add_argument("--fone", action="store_true")
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--tag", action="store_true", help="prompt with the format marker (round 4+)")
ap.add_argument("--strip_eq", action="store_true", help="scratchpad prompts drop `= ` (round 5+)")
ap.add_argument("--steps", type=int, default=48)
a = ap.parse_args()

model, cfg = load_checkpoint(a.ckpt, device="cuda:0", dtype=torch.bfloat16)
tok = load_tokenizer(a.tokenizer, cfg)


rng = random.Random(0)
cases = []


def draw(n, lo, hi, op, sym):
    got, guard = 0, 0
    while got < n and guard < n * 500:
        guard += 1
        x, y = rng.randint(lo, hi), rng.randint(lo, hi)
        if op == "-" and x < y:
            x, y = y, x
        if not held_out(x, y, op):
            continue
        cases.append((f"{x} {sym} {y} = ", {"+": x + y, "-": x - y, "*": x * y}[op], sym))
        got += 1
    assert got == n


draw(a.n, 10, 99, "+", "+")
draw(a.n, 10, 99, "-", "-")
draw(a.n, 2, 19, "*", "×")


def readings(t):
    """Both encodings of the leading answer: parsing only the reverse reading misreads
    a plain `109` as 901."""
    plain_nums = [int(x) for x in re.findall(r"-?\d+", t)]
    m = re.match(r"\s*(-?)((?:\d\s*)+)", t)
    rev = [int(m.group(1) + re.sub(r"\s", "", m.group(2))[::-1])] if m else []
    return plain_nums, rev


def score(prefix, label, strip_eq=False):
    """strip_eq must match how the checkpoint was trained, else this measures prompt
    tolerance, not computation."""
    first, anywhere, stops = {"+": [0, 0], "-": [0, 0], "×": [0, 0]}, {"+": 0, "-": 0, "×": 0}, 0
    prompts = [prefix + (p.rstrip("= ") if strip_eq else p) for p, _g, _o in cases]
    texts = fone.generate_texts(model, tok, cfg, prompts, a.steps)
    for (p, gold, op), t in zip(cases, texts, strict=True):
        # generate_texts stops at <eos> per row, so a short generation IS a terminated one.
        stopped = len(tok.encode(t, add_special_tokens=False).ids) < a.steps
        stops += stopped
        nums, rev = readings(t)
        first[op][1] += 1
        first[op][0] += (bool(nums) and nums[0] == gold) or (bool(rev) and rev[0] == gold)
        anywhere[op] += gold in nums or gold in rev
    tf = sum(c for c, _ in first.values())
    ta, tn = sum(anywhere.values()), sum(n for _, n in first.values())
    per = "  ".join(f"{op} {c}/{n}" for op, (c, n) in first.items())
    print(
        f"  {label:<12} first {tf}/{tn} = {100 * tf / tn:>4.1f}%   anywhere {ta}/{tn} = {100 * ta / tn:>4.1f}%"
        f"   <eos> {100 * stops / tn:>4.1f}%   [{per}]"
    )


print(f"{a.ckpt}  ({'FoNE' if a.fone else 'BPE'})")
if a.tag:
    # One prompt per format, so plain-worst/scratchpad-best becomes testable.
    for t, lab in (("[答] ", "plain"), ("[逆] ", "reverse"), ("[竖式] ", "scratchpad")):
        score(t, lab, strip_eq=a.strip_eq and lab == "scratchpad")
else:
    score("", "untagged")

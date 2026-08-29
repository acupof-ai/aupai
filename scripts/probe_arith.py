"""Re-score both probes with 'gold anywhere in the generation'.

"First number emitted" penalises a model for CHOOSING A DIFFERENT FORMAT, not
for being wrong: the BPE control answers a bare `61 + 48 = ` with scratchpad
prose, so the first number it emits is a scratchpad digit and never the answer.
Scoring the whole generation removes that confound. If BPE is still at zero
here, the format choice was not what was hiding the arithmetic.
"""

import argparse, random, re, sys, torch

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "mathbank")
from loader import load_checkpoint, load_tokenizer
from arith_curriculum import held_out

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt")
ap.add_argument("--tokenizer")
ap.add_argument("--fone", action="store_true")
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--tag", action="store_true", help="prompt with the format marker (round 4+)")
ap.add_argument("--steps", type=int, default=48)
a = ap.parse_args()

model, cfg = load_checkpoint(a.ckpt, device="cuda:0")
model = model.bfloat16()
for p in model.parameters():
    p.data = p.data.contiguous()
tok = load_tokenizer(a.tokenizer, cfg)
eos = tok.token_to_id("<eos>")
if a.fone:
    import fone

    num_id = cfg.num_id


@torch.no_grad()
def gen(prompt):
    if a.fone:
        (ids,), (vals,) = fone.encode_prompts([prompt], tok, num_id)
        out, outv, stopped = list(ids), list(vals), False
        for _ in range(a.steps):
            lg, hd = model(
                torch.tensor([out], device="cuda:0"),
                num_vals=torch.tensor([outv], device="cuda:0"),
                return_hidden=True,
            )
            nxt = lg[0, -1].argmax().item()
            if nxt == eos:
                stopped = True
                break
            v = float(fone.decode(model.num_logits(hd[0, -1].float()))) if nxt == num_id else 0.0
            out.append(nxt)
            outv.append(v)
        tail = out[len(ids) :]
        txt = fone.decode_text(tail, [x for t, x in zip(tail, outv[len(ids) :]) if t == num_id], tok, num_id)
        return txt, stopped
    ids = tok.encode(prompt, add_special_tokens=False).ids
    out, stopped = list(ids), False
    for _ in range(a.steps):
        lg = model(torch.tensor([out], device="cuda:0"))
        if isinstance(lg, tuple):
            lg = lg[0]
        nxt = lg[0, -1].argmax().item()
        if nxt == eos:
            stopped = True
            break
        out.append(nxt)
    return tok.decode(out[len(ids) :]), stopped


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
    """Both encodings of the leading answer, because the model does not obey the
    format tag: prompted `[逆] 61 + 48 = ` it answers `109`, not `9 0 1`. Parsing
    only the reverse reading scored that correct answer as 901 and reported 5.6%
    where the arithmetic was right. Scoring both asks "did it compute", which is
    the question, instead of "did it use the format I asked for"."""
    plain_nums = [int(x) for x in re.findall(r"-?\d+", t)]
    m = re.match(r"\s*(-?)((?:\d\s*)+)", t)
    rev = [int(m.group(1) + re.sub(r"\s", "", m.group(2))[::-1])] if m else []
    return plain_nums, rev


def score(prefix, label):
    first, anywhere, stops = {"+": [0, 0], "-": [0, 0], "×": [0, 0]}, {"+": 0, "-": 0, "×": 0}, 0
    for p, gold, op in cases:
        t, stopped = gen(prefix + p)
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
    # With one prompt per format the literature's prediction becomes testable for
    # the first time: plain should be worst, scratchpad best. Round 3's flat
    # 24/24/23% could not test it -- the prompt did not say which format to use.
    for t, lab in (("[答] ", "plain"), ("[逆] ", "reverse"), ("[竖式] ", "scratchpad")):
        score(t, lab)
else:
    score("", "untagged")

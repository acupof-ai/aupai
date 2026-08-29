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
        out, outv = list(ids), list(vals)
        for _ in range(a.steps):
            lg, hd = model(
                torch.tensor([out], device="cuda:0"),
                num_vals=torch.tensor([outv], device="cuda:0"),
                return_hidden=True,
            )
            nxt = lg[0, -1].argmax().item()
            if nxt == eos:
                break
            v = float(fone.decode(model.num_logits(hd[0, -1].float()))) if nxt == num_id else 0.0
            out.append(nxt)
            outv.append(v)
        tail = out[len(ids) :]
        return fone.decode_text(tail, [x for t, x in zip(tail, outv[len(ids) :]) if t == num_id], tok, num_id)
    ids = tok.encode(prompt, add_special_tokens=False).ids
    out = list(ids)
    for _ in range(a.steps):
        lg = model(torch.tensor([out], device="cuda:0"))
        if isinstance(lg, tuple):
            lg = lg[0]
        nxt = lg[0, -1].argmax().item()
        if nxt == eos:
            break
        out.append(nxt)
    return tok.decode(out[len(ids) :])


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

first, anywhere = {"+": [0, 0], "-": [0, 0], "×": [0, 0]}, {"+": 0, "-": 0, "×": 0}
for p, gold, op in cases:
    t = gen(p)
    nums = [int(x) for x in re.findall(r"-?\d+", t)]
    first[op][1] += 1
    first[op][0] += bool(nums) and nums[0] == gold
    anywhere[op] += gold in nums
print(f"{a.ckpt}  ({'FoNE' if a.fone else 'BPE'})")
for op in first:
    c, n = first[op]
    print(
        f"  {op}  first {c}/{n} = {100 * c / n:>3.0f}%   anywhere {anywhere[op]}/{n} = {100 * anywhere[op] / n:>3.0f}%"
    )
tf, ta, tn = sum(c for c, _ in first.values()), sum(anywhere.values()), sum(n for _, n in first.values())
print(f"  TOTAL first {tf}/{tn} = {100 * tf / tn:.1f}%   anywhere {ta}/{tn} = {100 * ta / tn:.1f}%")

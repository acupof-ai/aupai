"""The BPE half of the FoNE comparison. Same held-out cases, same scoring.

probe3.py cannot score a non-FoNE checkpoint (it calls fone.encode_prompts /
model.num_logits). Same rng, draw() and held_out filter as probe3.py, so both probes see
the SAME 180 problems in the SAME order and the comparison is paired.
"""

import argparse, random, re, sys, torch

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "mathbank")
from loader import load_checkpoint, load_tokenizer
from arith_curriculum import held_out

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ckpt_k5_arith_bpe.pt")
ap.add_argument("--tokenizer", default="data/tokenizer_k5.json")
ap.add_argument("--n", type=int, default=60)
a = ap.parse_args()

model, cfg = load_checkpoint(a.ckpt, device="cuda:0")
model = model.bfloat16()
for p in model.parameters():
    p.data = p.data.contiguous()
tok = load_tokenizer(a.tokenizer, cfg)
eos = tok.token_to_id("<eos>")


@torch.no_grad()
def gen(prompt, n=24):
    ids = tok.encode(prompt, add_special_tokens=False).ids
    out, stopped = list(ids), False
    for _ in range(n):
        lg = model(torch.tensor([out], device="cuda:0"))
        if isinstance(lg, tuple):
            lg = lg[0]
        nxt = lg[0, -1].argmax().item()
        if nxt == eos:
            stopped = True
            break
        out.append(nxt)
    return tok.decode(out[len(ids) :]), stopped


# --- identical to probe3.py from here ---
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
        v = {"+": x + y, "-": x - y, "*": x * y}[op]
        cases.append((f"{x} {sym} {y} = ", v, sym))
        got += 1
    assert got == n, f"only {got}/{n} held-out cases for {op}"


draw(a.n, 10, 99, "+", "+")
draw(a.n, 10, 99, "-", "-")
draw(a.n, 2, 19, "*", "×")
print(f"{len(cases)} probe cases, all held out by construction")

first_ok = {"+": [0, 0], "-": [0, 0], "×": [0, 0]}
stop_n, ex = 0, []
for p, gold, op in cases:
    t, stopped = gen(p)
    stop_n += stopped
    m = re.search(r"-?\d+", t)
    ok = bool(m) and int(m.group()) == gold
    first_ok[op][1] += 1
    first_ok[op][0] += ok
    if len(ex) < 5:
        ex.append(f"{p}-> {t.strip()[:34]!r} stop={stopped} gold={gold} {'OK' if ok else ''}")

print(f"\n{a.ckpt}")
print("\nCOMPUTATION (first number emitted):")
for op, (c, n) in first_ok.items():
    print(f"  {op}  {c}/{n} = {100 * c / n:.0f}%")
print(f"\nTERMINATION: emitted <eos> in {stop_n}/{len(cases)} = {100 * stop_n / len(cases):.0f}%")
print("\nsamples:")
for e in ex:
    print("  ", e)

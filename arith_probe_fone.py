"""Bare arithmetic through the FoNE channel. Directly comparable to the BPE probe
that scored ckpt_k7_v3 at 0/180.

Numbers go in as [NUM] plus a value and come out of the digit head, so nothing in
this path touches BPE.
"""
import argparse, random, re, sys, torch
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import fone
from loader import load_checkpoint, load_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ckpt_k6_arith.pt")
ap.add_argument("--tokenizer", default="data/tokenizer_k6.json")
ap.add_argument("--n", type=int, default=60)
a = ap.parse_args()

model, cfg = load_checkpoint(a.ckpt, device="cuda:0")
model = model.bfloat16()
for p in model.parameters():
    p.data = p.data.contiguous()
tok = load_tokenizer(a.tokenizer, cfg)
num_id = cfg.num_id
print(f"{a.ckpt}: fone={cfg.fone} num_id={num_id}")

rng = random.Random(0)
cases = []
for _ in range(a.n):
    x, y = rng.randint(10, 99), rng.randint(10, 99)
    cases.append((f"{x} + {y} = ", x + y, "+"))
for _ in range(a.n):
    x, y = rng.randint(10, 99), rng.randint(10, 99)
    cases.append((f"{max(x,y)} - {min(x,y)} = ", abs(x - y), "-"))
for _ in range(a.n):
    x, y = rng.randint(2, 19), rng.randint(2, 19)
    cases.append((f"{x} × {y} = ", x * y, "×"))

@torch.no_grad()
def gen(prompt, n=10):
    (ids,), (vals,) = fone.encode_prompts([prompt], tok, num_id)
    out, outv = list(ids), list(vals)
    # simple greedy loop through the full forward each step (short sequences)
    for _ in range(n):
        xx = torch.tensor([out], device="cuda:0")
        vv = torch.tensor([outv], device="cuda:0")
        lg, hd = model(xx, num_vals=vv, return_hidden=True)
        nxt = lg[0, -1].argmax().item()
        val = 0.0
        if nxt == num_id:
            val = float(fone.decode(model.num_logits(hd[0, -1].float())))
        out.append(nxt); outv.append(val)
        if nxt == tok.token_to_id("<eos>"):
            break
    txt = fone.decode_text(out[len(ids):], [x for t, x in zip(out[len(ids):], outv[len(ids):]) if t == num_id], tok, num_id)
    return txt

by = {"+": [0, 0], "-": [0, 0], "×": [0, 0]}
wrong = []
for p, gold, op in cases:
    t = gen(p)
    m = re.search(r"-?\d+", t)
    ok = m and int(m.group()) == gold
    by[op][1] += 1; by[op][0] += bool(ok)
    if not ok and len(wrong) < 6:
        wrong.append(f"{p}-> {t.strip()[:24]!r}  (gold {gold})")
print(f"\nbare arithmetic through FoNE, greedy, {a.n} each:")
for op, (c, n) in by.items():
    print(f"  {op}  {c}/{n} = {100*c/n:.0f}%")
print("  [ckpt_k7_v3 through BPE scored 0% on all three]")
if wrong:
    print("\nwrong examples:")
    for w in wrong: print("  ", w)

"""Bare arithmetic, no word problem. If the model fails here, no amount of
reasoning data fixes the math score."""
import random, sys, torch
sys.path.insert(0, "scripts")
from loader import load_checkpoint, load_tokenizer

model, cfg = load_checkpoint("ckpt_k7_v3.pt", device="cuda:0")
model = model.bfloat16()  # FlashAttention takes fp16/bf16 only
for p in model.parameters():
    p.data = p.data.contiguous()
tok = load_tokenizer("data/tokenizer.json", cfg)
rng = random.Random(0)

cases = []
for _ in range(60):
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    cases.append((f"{a} + {b} = ", a + b))
for _ in range(60):
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    cases.append((f"{max(a,b)} - {min(a,b)} = ", abs(a - b)))
for _ in range(60):
    a, b = rng.randint(2, 19), rng.randint(2, 19)
    cases.append((f"{a} × {b} = ", a * b))

@torch.no_grad()
def gen(prompt, n=8):
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], device="cuda:0")
    out = []
    for _ in range(n):
        logits, _ = model(x)
        nxt = logits[0, -1].argmax().item()
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device="cuda:0")], 1)
    return tok.decode(out)

import re
by = {"+": [0, 0], "-": [0, 0], "×": [0, 0]}
wrong_examples = []
for p, gold in cases:
    op = "+" if "+" in p else ("-" if "-" in p else "×")
    txt = gen(p)
    m = re.match(r"\s*(-?\d+)", txt)
    ok = m and int(m.group(1)) == gold
    by[op][1] += 1
    by[op][0] += bool(ok)
    if not ok and len(wrong_examples) < 6:
        wrong_examples.append(f"{p}{txt.strip()[:12]!r} (gold {gold})")
print("bare arithmetic, greedy, 60 each:")
for op, (c, n) in by.items():
    print(f"  {op}  {c}/{n} = {100*c/n:.0f}%")
print("\nwrong examples:")
for w in wrong_examples:
    print("  ", w)

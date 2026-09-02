"""Does FoNE actually fix the two defects the harness measured on BPE?

Defect 1: the same number tokenises differently depending on context (6/6 fail).
Defect 2: place value is not represented (only 3/6 numbers align).

Claiming FoNE fixes them is an argument; this measures it.
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import fone
from tokenizers import Tokenizer

tok = Tokenizer.from_file("data/tokenizer.json")
num_id = tok.token_to_id("[NUM]")
print(f"[NUM] id = {num_id}")

ctxs = ["{n}", " {n}", "= {n}", "共{n}个", "第{n}章", "({n})"]
nums = ["7", "63", "122", "1640", "2024", "10000"]

print("\n--- BPE (what the model sees today) ---")
bad_bpe = 0
for n in nums:
    seen = {tuple(t for t in tok.encode(c.format(n=n), add_special_tokens=False).tokens
                  if any(ch.isdigit() for ch in t)) for c in ctxs}
    bad_bpe += len(seen) > 1
    print(f"  {n:>6}: {len(seen)} distinct tokenisations across 6 contexts")

print("\n--- FoNE (single [NUM] + a value channel) ---")
bad_fone = 0
for n in nums:
    reps = set()
    for c in ctxs:
        ids, vals = fone.encode_text([c.format(n=n)], tok, num_id)
        seq = ids[0].tolist()
        k = seq.count(num_id)
        v = tuple(round(float(x), 6) for x in vals[:k])
        reps.add((k, v))
    bad_fone += len(reps) > 1
    k, v = next(iter(reps))
    print(f"  {n:>6}: {len(reps)} distinct representations; {k} [NUM] token(s), value {v}")

print(f"\ncontext-inconsistent:  BPE {bad_bpe}/6   FoNE {bad_fone}/6")

print("\n--- place value: does the encoding separate digits? ---")
import torch
for n in [7, 63, 122, 1640]:
    e = fone.encode(torch.tensor([float(n)]))
    d = fone.digit_targets(torch.tensor([float(n)]))
    print(f"  {n:>6}: encoding dim {tuple(e.shape)}, recovered digits {d.tolist()[0]}")

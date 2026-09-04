"""How many tracked entry points take a GPU card and load a checkpoint (de-55).

Run: python3 runs/audit_0904/card_claim_population.py

The ruling that produced claim_my_cards was written against a hand list of 11 files. This
measures it from the tracked tree instead. Both numbers it prints are FLOORS, not counts: the
predicate is textual, so a file that reaches a card by a route not written here is missed. The
first version of this scan missed eval/score_matrix.py, whose device comes back from a helper
returning an f-string -- which is why `f"cuda:` is in the pattern below and why the docstring
says floor.
"""
import os
import re
import subprocess

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
r = subprocess.run(["git", "-C", root, "ls-files"], capture_output=True, text=True)
files = [f for f in r.stdout.split() if f.endswith(".py")]

# A file TAKES a card if it puts something on cuda, not merely if the string appears.
TAKES = re.compile(r"\.cuda\(\)|"                        # explicit
                   r"""default\s*=\s*["']cuda|"""        # an argparse device default
                   r"""=\s*f?["']cuda:?[\d{]*"""         # dev = "cuda" / "cuda:0" / f"cuda:{i}"
                   r"""|return\s+f?["']cuda|"""          # a device-picking helper
                   r"""\.to\(\s*["']cuda""")             # .to("cuda")
LOADS = re.compile(r"torch\.load|load_checkpoint")
rows = []
for f in files:
    try:
        s = open(os.path.join(root, f), encoding="utf-8").read()
    except OSError:
        continue
    if not (LOADS.search(s) and TAKES.search(s)):
        continue
    has_main = bool(re.search(r"if __name__ == .__main__.", s))
    rows.append((f, has_main, "claim_my_cards" in s or "card_claim" in s))
print(f"takes a card AND loads a ckpt: {len(rows)} (floor); runnable: "
      f"{sum(1 for _, m, _ in rows if m)}; claimed: {sum(1 for *_, c in rows if c)}")
for f, m, c in sorted(rows):
    print(("  CLAIM " if c else "  ---- ") + f + ("" if m else "   (no __main__)"))

# Read-only: count low-content LLM placeholders/refusals in the full swallow_math.
import glob
import json
import re

PATS = [
    r"there (?:is|appears to be) no (?:specific )?math(?:ematical)? problem",
    r"does not contain a (?:specific )?math(?:ematical)? problem",
    r"no (?:specific )?math(?:ematical)? problem to solve",
    r"the provided text does not contain",
    r"if you could provide a (?:specific )?math(?:ematical)? problem",
]
RX = re.compile("|".join(PATS), re.I)
tot = hit = 0
byshard = {}
for p in sorted(glob.glob("/work/aupai/data/corpus/swallow_math/swallow_math_*.jsonl")):
    n = h = 0
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line).get("content") or ""
            tot += 1
            n += 1
            if RX.search(t[:400]):
                hit += 1
                h += 1
    byshard[p.split("/")[-1]] = h
print(f"total={tot} placeholder_hits={hit} ({hit / tot * 100:.3f}%)")
print("top shards:", sorted(byshard.items(), key=lambda x: -x[1])[:5])
print("DONE")

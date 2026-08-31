# restartable: deterministic from random.Random(20260831), full run ~30s -- an
# interrupt costs a rerun, and the output file is re-creatable at any time.
import json, glob, random, re

rng = random.Random(20260831)
LANGS = [
    ("python", re.compile(r"^(def |class |import |from |#!)", re.M)),
    ("javascript", re.compile(r"^(var |const |let |function |\$\(|module\.exports|require\()", re.M)),
    ("java", re.compile(r"^(package |public class |import java\.)", re.M)),
    ("c_cpp", re.compile(r"^(#include|#pragma)", re.M)),
    ("go", re.compile(r"^(package |func )", re.M)),
    ("rust", re.compile(r"^(fn |use |pub fn|#!\[)", re.M)),
    ("php", re.compile(r"<\?php")),
    ("ruby", re.compile(r"^(require |def |class )", re.M)),
]
def lang(s):
    head = s[:600]
    for name, rx in LANGS:
        if rx.search(head): return name
    return "other"

reservoir = {}
n = 0
for path in sorted(glob.glob("/work/aupai/data/corpus/code_rp1t/*.jsonl")):
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= 400: break
            n += 1
            try: d = json.loads(line)
            except: continue
            c = d.get("content","")
            L = lang(c)
            r = reservoir.setdefault(L, [])
            if len(r) < 60: r.append(c)
            else:
                j = rng.randrange(300)
                if j < 60: r[j] = c
counts = {k: len(v) for k,v in reservoir.items()}
out = []
for L, docs in reservoir.items():
    if L == "other" or len(docs) < 2: continue
    for c in docs[:8]:
        out.append({"lang": L, "content": c[:1800]})
rng.shuffle(out)
out = out[:50]
with open("/tmp/t24_sample50.jsonl","w") as f:
    for d in out: f.write(json.dumps(d, ensure_ascii=False)+"\n")
print("scanned", n, "counts", counts, "sampled", len(out))

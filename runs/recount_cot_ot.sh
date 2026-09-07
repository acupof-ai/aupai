cd /work/aupai || exit 1
python3 - <<'PY'
import glob, json, sys, time
sys.path.insert(0, "scripts")
from count_tokens import count_docs, CONVENTION
from tokenizers import Tokenizer
t0=time.time()
tok = Tokenizer.from_file("data/tokenizer.json")
ps = sorted(glob.glob("data/corpus/cot_open_thoughts/*.jsonl"))
docs=toks=stamped=0
for p in ps:
    texts=[]
    with open(p, "rb") as f:
        raw=f.read()
    for line in raw.decode("utf-8","replace").split("\n"):
        if not line.strip(): continue
        try: r=json.loads(line)
        except json.JSONDecodeError: continue
        texts.append(r["chain"]); stamped += r.get("tokens") or 0
    docs += len(texts)
    for i in range(0,len(texts),2000):
        toks += count_docs(texts[i:i+2000], tok)
print(json.dumps({"domain":"cot_open_thoughts","shards":len(ps),"docs":docs,
 "tokens":toks,"per_row_tokens_sum":stamped,"field":"chain","convention":CONVENTION,
 "seconds":round(time.time()-t0)}, indent=1))
PY
echo "=== RC3 DONE ==="

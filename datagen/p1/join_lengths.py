import json, os
from tokenizers import Tokenizer

BASE = "/work/aupai/data/corpus"
DOMAINS = ["code_rp1t_dd09", "code_rp1t_b2v2_dd", "code_dedup08"]
SAMPLE_BUDGET = 330 * 1024 * 1024
TOK = "/work/aupai/data/tokenizer.json"

tok = Tokenizer.from_file(TOK)
targets = {}
for line in open("/work/aupai/data/p1/classifier_full_100k.jsonl"):
    d = json.loads(line)
    dom, rest = d["id"].split("/", 1)
    targets.setdefault(dom, set()).add(int(rest.split(":")[1]))

out = open("/work/aupai/data/p1/classifier_lengths.jsonl", "w")
for dom in DOMAINS:
    ddir = os.path.join(BASE, dom)
    want = targets.get(dom, set())
    found, valid_idx = 0, 0
    sample_bytes, sample_toks = 0, 0
    for sf in sorted(os.listdir(ddir)):
        if not sf.endswith(".jsonl"):
            continue
        with open(os.path.join(ddir, sf)) as f:
            for line in f:
                try:
                    content = json.loads(line).get("content") or ""
                except Exception:
                    continue
                if not content.strip():
                    continue
                valid_idx += 1
                b = len(line.encode("utf-8"))
                if valid_idx in want:
                    out.write(json.dumps({"id": dom + "/" + sf + ":" + str(valid_idx), "bytes": b}) + "\n")
                    found += 1
                if sample_bytes < SAMPLE_BUDGET:
                    sample_bytes += b
                    sample_toks += len(tok.encode(content).ids) + 1
    ratio = sample_toks / sample_bytes
    print(f"{dom}: {valid_idx} valid docs, {found}/{len(want)} target ids found, "
          f"measured tok/byte {ratio:.6f} on {sample_bytes/1e6:.1f} MB", flush=True)
out.close()
print("done")

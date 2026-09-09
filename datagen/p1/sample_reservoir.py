import json, os, random, sys
# restartable: single streaming pass over one domain (minutes); rerun from scratch is the resume

domain_dir, n, seed, out_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
HEAD = 350
rng = random.Random(seed)
reservoir = []
seen = 0
for sf in sorted(os.listdir(domain_dir)):
    if not sf.endswith(".jsonl"):
        continue
    with open(os.path.join(domain_dir, sf)) as f:
        for line in f:
            try:
                text = json.loads(line).get("content") or ""
            except Exception:
                continue
            if not text.strip():
                continue
            seen += 1
            item = {"id": os.path.basename(domain_dir) + "/" + sf + ":" + str(seen), "text": text[:HEAD]}
            if len(reservoir) < n:
                reservoir.append(item)
            else:
                j = rng.randrange(seen)
                if j < n:
                    reservoir[j] = item
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    for item in reservoir:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(domain_dir + ": " + str(seen) + " valid docs, " + str(len(reservoir)) + " sampled -> " + out_path)

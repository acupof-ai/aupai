import json, os
from tokenizers import Tokenizer

DOMAIN = "/work/aupai/data/corpus/code_dedup08"
TOK = "/work/aupai/data/tokenizer.json"
SAMPLE_BYTES = 315 * 1024 * 1024

tok = Tokenizer.from_file(TOK)
eos = tok.token_to_id("<eos>")
assert eos is not None

total_bytes = 0
for sf in sorted(os.listdir(DOMAIN)):
    if sf.endswith(".jsonl"):
        total_bytes += os.path.getsize(os.path.join(DOMAIN, sf))

sample_bytes = 0
sample_tokens = 0
n_docs = 0
for sf in sorted(os.listdir(DOMAIN)):
    if not sf.endswith(".jsonl"):
        continue
    with open(os.path.join(DOMAIN, sf)) as f:
        for line in f:
            sample_bytes += len(line.encode("utf-8"))
            d = json.loads(line)
            sample_tokens += len(tok.encode(d["content"]).ids) + 1
            n_docs += 1
            if sample_bytes >= SAMPLE_BYTES:
                break
    if sample_bytes >= SAMPLE_BYTES:
        break

ratio = sample_tokens / sample_bytes
est_tokens = ratio * total_bytes
print(f"domain bytes: {total_bytes} ({total_bytes/1e9:.2f} GB)")
print(f"sample: {n_docs} docs, {sample_bytes/1e6:.1f} MB, {sample_tokens} tokens")
print(f"tok/byte: {ratio:.6f}")
print(f"ESTIMATED TOKENS: {est_tokens/1e9:.2f}B")

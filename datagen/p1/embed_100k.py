import json, os, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModel

MODEL = "/work/aupai/data/p1/arctic-embed"
SAMPLES = "/work/aupai/data/p1/classifier_full_100k.jsonl"
OUT = "/work/aupai/data/p1/arctic_embeds_100k.npy"
BATCH = 256

rows = [json.loads(l) for l in open(SAMPLES)]
ids = [r["id"] for r in rows]
texts = [r["text"] for r in rows]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda:0").eval()

pool_cfg = os.path.join(MODEL, "1_Pooling", "config.json")
pooling = "cls"
if os.path.exists(pool_cfg):
    cfg = json.load(open(pool_cfg))
    if cfg.get("pooling_mode_mean_tokens"):
        pooling = "mean"
print("pooling:", pooling, flush=True)

t0 = time.time()
embs = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
done = 0
with torch.no_grad():
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda:0")
        out = model(**enc).last_hidden_state
        if pooling == "cls":
            v = out[:, 0]
        else:
            m = enc["attention_mask"].unsqueeze(-1).half()
            v = (out * m).sum(1) / m.sum(1).clamp(min=1e-6)
        v = torch.nn.functional.normalize(v, dim=-1)
        embs[i:i+len(batch)] = v.cpu().numpy()
        done += len(batch)
        if done % (BATCH*20) == 0 or done == len(texts):
            el = time.time() - t0
            print(f"{done}/{len(texts)} {el:.0f}s eta {el/done*(len(texts)-done):.0f}s", flush=True)
np.save(OUT, embs)
with open("/work/aupai/data/p1/arctic_embeds_100k.ids.json", "w") as f:
    json.dump(ids, f)
print("saved", OUT, embs.shape, f"{time.time()-t0:.0f}s", flush=True)

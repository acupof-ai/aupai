import json, os, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModel

BASE = "/work/aupai/data/corpus"
DOMAINS = ["code_rp1t_dd09", "code_rp1t_b2v2_dd", "code_dedup08"]
OUT = "/work/aupai/data/p1/keep_set"
MODEL = "/work/aupai/data/p1/arctic-embed"
HEAD = "/work/aupai/data/p1/head_ge3_w.npy"
CUT = -0.258355
BATCH = 256
HEAD_CHARS = 350

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda:0").eval()
w = np.load(HEAD)
bias = float(open("/work/aupai/data/p1/head_ge3_bias.txt").read().splitlines()[0])
# drift guard: CUT is a hand-copied value from bias.txt line 2; a retrained head that
# overwrites bias.txt must not silently leave scoring on the old cut
assert abs(float(open("/work/aupai/data/p1/head_ge3_bias.txt").read().splitlines()[1]) - CUT) < 1e-6

os.makedirs(OUT, exist_ok=True)
t0 = time.time()
grand = {"scored": 0, "kept": 0, "bytes_in": 0, "bytes_kept": 0}
manifest = {}
for dom in DOMAINS:
    ddir = os.path.join(BASE, dom)
    odir = os.path.join(OUT, dom)
    os.makedirs(odir, exist_ok=True)
    d = {"scored": 0, "kept": 0, "bytes_in": 0, "bytes_kept": 0}
    dt0 = time.time()
    shards = sorted(f for f in os.listdir(ddir) if f.endswith(".jsonl"))
    for si, sf in enumerate(shards):
        opath = os.path.join(odir, sf)
        if os.path.exists(opath):
            continue
        docs, heads, raw_bytes = [], [], 0
        with open(os.path.join(ddir, sf)) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    text = rec.get("content") or ""
                except Exception:
                    continue
                if not text.strip():
                    continue
                docs.append((rec, line))
                heads.append(text[:HEAD_CHARS])
                raw_bytes += len(line.encode("utf-8"))
        keep_idx = []
        with torch.no_grad():
            for i in range(0, len(heads), BATCH):
                batch = heads[i:i+BATCH]
                enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda:0")
                v = model(**enc).last_hidden_state[:, 0].float().cpu().numpy()
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                s = v @ w + bias
                keep_idx.extend(np.where(s >= CUT)[0] + i)
        kept_bytes = 0
        tmp = opath + ".tmp"
        with open(tmp, "w") as f:
            for k in keep_idx:
                f.write(docs[k][1])
                kept_bytes += len(docs[k][1].encode("utf-8"))
        os.rename(tmp, opath)
        d["scored"] += len(docs); d["kept"] += len(keep_idx)
        d["bytes_in"] += raw_bytes; d["bytes_kept"] += kept_bytes
        el = time.time() - dt0
        rate = d["scored"] / max(el, 1)
        kk, ss = d["kept"], d["scored"]
        print(f"{dom} {si+1}/{len(shards)} {sf}: kept {len(keep_idx)}/{len(docs)} "
              f"| dom {kk}/{ss} ({kk/max(ss,1):.3f}) | {rate:.0f} docs/s", flush=True)
    if d["scored"] == 0:
        # all shards skipped (resume): stats from disk; scored from the join run
        join_counts = {"code_rp1t_dd09": 3434322, "code_rp1t_b2v2_dd": 2103485, "code_dedup08": 6239038}
        d["scored"] = join_counts[dom]
        d["bytes_in"] = sum(os.path.getsize(os.path.join(ddir, f)) for f in shards)
        outs = sorted(os.listdir(odir))
        d["bytes_kept"] = sum(os.path.getsize(os.path.join(odir, f)) for f in outs)
        d["kept"] = sum(sum(1 for _ in open(os.path.join(odir, f))) for f in outs)
        print(f"== {dom} stats recomputed from disk (all shards skipped)", flush=True)
    manifest[dom] = d
    for k in grand:
        grand[k] += d[k]
    dk, ds_ = d["kept"], d["scored"]
    dbk, dbi = d["bytes_kept"], d["bytes_in"]
    print(f"== {dom} DONE: keep doc {dk/ds_:.4f} byte {dbk/dbi:.4f}", flush=True)
manifest["_total"] = grand
manifest["_cut"] = CUT
manifest["_elapsed_s"] = time.time() - t0
manifest["_dedup08_status"] = ("pre_decontamination: 169561 deletable dedup08 docs not yet deleted "
    "(= 157684 dd09-intersection + 12120 b2v2-intersection - 243 already decontaminated; "
    "b0 pod count 2026-09-10: 138.6K on the 15 rp1t shards + 31.0K on the 283 starcoder shards); "
    "code_dedup08 rows above are pre-deletion and MUST be recomputed by the deletion pass")
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
gk, gs_ = grand["kept"], grand["scored"]
gbk, gbi = grand["bytes_kept"], grand["bytes_in"]
el = manifest["_elapsed_s"]
print(f"== TOTAL: keep doc {gk/gs_:.4f} byte {gbk/gbi:.4f} in {el:.0f}s", flush=True)

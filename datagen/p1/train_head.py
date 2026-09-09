import json, numpy as np, torch

X = np.load("/work/aupai/data/p1/arctic_embeds_100k.npy").astype(np.float32)
ids = json.load(open("/work/aupai/data/p1/arctic_embeds_100k.ids.json"))
lab = {d["id"]: d["score"] for d in map(json.loads, open("/work/aupai/data/p1/classifier_labels_100k.jsonl"))}
ln = {d["id"]: d["bytes"] for d in map(json.loads, open("/work/aupai/data/p1/classifier_lengths.jsonl"))}
y = np.array([lab[i] for i in ids])
B = np.array([ln.get(i, -1) for i in ids])
dom = np.array([i.split("/")[0] for i in ids])
assert (B > 0).all(), "missing lengths"

def auc(s, y):
    # rank-sum AUC, ties averaged
    order = np.argsort(s)
    ss = s[order]
    ranks = np.empty(len(s))
    start = 0
    for i in range(1, len(s) + 1):
        if i == len(s) or ss[i] != ss[start]:
            ranks[order[start:i]] = (start + 1 + i) / 2.0
            start = i
    npos, nneg = y.sum(), len(y) - y.sum()
    return (ranks[y == 1].sum() - npos*(npos+1)/2) / (npos*nneg)

def _auc_selftest():
    assert abs(auc(np.array([0.1,0.4,0.8,0.9]), np.array([0,0,1,1])) - 1.0) < 1e-9
    r = np.random.RandomState(0)
    v = [auc(r.rand(2000), (r.rand(2000) < 0.5).astype(float)) for _ in range(5)]
    assert abs(np.mean(v) - 0.5) < 0.03, v
_auc_selftest()

rng = np.random.RandomState(42)
te = np.zeros(len(y), bool)
for c in range(6):
    idx = np.where(y == c)[0]
    rng.shuffle(idx)
    te[idx[:len(idx)//5]] = True
tr = ~te

def train_head(yt):
    Xt = torch.tensor(X[tr]).cuda(0)
    tt = torch.tensor(yt[tr]).cuda(0)
    w = torch.zeros(X.shape[1]+1, device="cuda:0", requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        p = torch.sigmoid(Xt @ w[:-1] + w[-1])
        loss = -(tt*torch.log(p+1e-9) + (1-tt)*torch.log(1-p+1e-9)).mean()
        loss.backward(); opt.step()
    return (w[:-1].detach().cpu().numpy(), w[-1].item())

buckets = [("<2KB", B < 2048), ("2-10KB", (B >= 2048) & (B < 10240)), (">10KB", B >= 10240)]
for t in (2, 3, 4):
    yt = (y >= t).astype(np.float32)
    w, b0 = train_head(yt)
    s = X @ w + b0
    print(f"=== threshold >= {t} (train pos rate {yt[tr].mean():.3f}) ===", flush=True)
    print(f"AUC pooled: {auc(s[te], yt[te]):.4f}  (n={te.sum()})", flush=True)
    for d in sorted(set(dom)):
        m = te & (dom == d)
        print(f"AUC {d}: {auc(s[m], yt[m]):.4f}  (n={m.sum()}, pos {yt[m].mean():.3f})", flush=True)
    for name, m in buckets:
        m = te & m
        print(f"AUC len {name}: {auc(s[m], yt[m]):.4f}  (n={m.sum()}, pos {yt[m].mean():.3f})", flush=True)
    pred = s[te] >= 0.0
    print(f"keep@0.5 doc: {pred.mean():.4f}  byte: {B[te][pred].sum()/B[te].sum():.4f}", flush=True)
    print("  cut sweep (test set):", flush=True)
    for q in (0.03, 0.05, 0.10, 0.17, 0.25, 0.30):
        cut = np.quantile(s[te], 1.0 - q)
        pr = s[te] >= cut
        tp = int((pr & (yt[te] == 1)).sum())
        print(f"  keep~{q:.2f} doc={pr.mean():.3f} byte={B[te][pr].sum()/B[te].sum():.3f} "
              f"prec={tp/max(pr.sum(),1):.3f} recall={tp/int(yt[te].sum()):.3f}", flush=True)

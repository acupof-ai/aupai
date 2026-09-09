import json, numpy as np, torch

X = np.load("/work/aupai/data/p1/arctic_embeds_100k.npy").astype(np.float32)
ids = json.load(open("/work/aupai/data/p1/arctic_embeds_100k.ids.json"))
lab = {d["id"]: d["score"] for d in map(json.loads, open("/work/aupai/data/p1/classifier_labels_100k.jsonl"))}
y = (np.array([lab[i] for i in ids]) >= 3).astype(np.float32)

Xt = torch.tensor(X).cuda(0)
tt = torch.tensor(y).cuda(0)
w = torch.zeros(X.shape[1]+1, device="cuda:0", requires_grad=True)
opt = torch.optim.Adam([w], lr=0.05)
for _ in range(300):
    opt.zero_grad()
    p = torch.sigmoid(Xt @ w[:-1] + w[-1])
    loss = -(tt*torch.log(p+1e-9) + (1-tt)*torch.log(1-p+1e-9)).mean()
    loss.backward(); opt.step()
s = (X @ w[:-1].detach().cpu().numpy() + w[-1].item())
cut = float(np.quantile(s, 0.75))
print(f"train pos {y.mean():.4f} final loss {loss.item():.4f}", flush=True)
print(f"cut (75th pct of 100K sample scores): {cut:.6f}", flush=True)
print(f"sample keep at cut: {(s >= cut).mean():.4f}", flush=True)
np.save("/work/aupai/data/p1/head_ge3_w.npy", w[:-1].detach().cpu().numpy())
with open("/work/aupai/data/p1/head_ge3_bias.txt", "w") as f:
    f.write(repr(w[-1].item()) + "\n" + repr(cut) + "\n")
print("saved head_ge3_w.npy + bias", flush=True)

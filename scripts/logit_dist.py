#!/usr/bin/env python3
# restartable: three checkpoints, each a few seconds of forward on one card, and the JSON is
# written at the end. An interrupt costs the whole run and the whole run is under a minute --
# cheaper to rerun than to shard.
"""Does the embedding amplification reach the logits, or does something downstream absorb it?

§4.6 left this open with a bound, not a reading: the Cauchy-Schwarz upper bound on |logit|
grew 1.99x across two intervals while softcap stayed at 15.0, and `norm.g` rms was FLAT
(0.171/0.166/0.171) -- so nothing was seen shrinking to compensate. But a bound is the
fully-aligned worst case; the real logit is ||h||*||w_row||*cos(theta), and cos(theta) is
small in 1024 dims. This runs the real forward on real tokens and reads the distribution.

The reading that matters is the PRE-SOFTCAP logit spread, because softcap is a fixed 15.0:
if the pre-softcap scale grows while the cap does not, a growing fraction of positions is
being squashed, and that IS the amplification reaching the output. If instead cos(theta)
falls at the same rate the norms grow, the distribution is stable and the amplification is
absorbed by de-alignment.
"""
import json, os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("FLA_FLASH_KDA", "0")
from scripts.loader import load_checkpoint, load_tokenizer  # noqa: E402

# Step-numbered checkpoints only, so each row's x-axis is unambiguous. NOT the bare .pt:
# de measured its row_cursor over-counting 213,164 rows (a --max_steps run-end save that got
# no step), and while the WEIGHTS are identical the ambiguity is free to avoid here.
CKPTS = ["ckpt_p200m_4b_0902.pt.step2500", "ckpt_p200m_4b_0902.pt.step3000",
         "ckpt_p200m_4b_0902.pt.step3500"]
DEV = "cuda:0"          # card 2 is exposed as cuda:0 via CUDA_VISIBLE_DEVICES
N_TOK = 4096            # one sequence's worth; the distribution is over N_TOK*vocab logits

def read(path, ids):
    model, cfg = load_checkpoint(path, device=DEV, dtype=torch.bfloat16)
    # SOFTCAP is a MODULE constant in model.py:63 (env-overridable, default 15.0), NOT a cfg
    # field. The first version of this script read getattr(cfg, "logit_softcap", 0.0), got 0.0,
    # and therefore reported post-softcap numbers labelled "pre" -- absmax 14.62 sitting just
    # under 15 was tanh saturation being read as a flat distribution.
    import model as _m
    softcap = float(_m.SOFTCAP) if _m.SOFTCAP else 0.0
    assert softcap > 0, "SOFTCAP is None/0: model.py:63 changed, and the pre/post inversion below is then meaningless"
    real = int(getattr(cfg, "vocab_real", cfg.vocab))
    x = torch.tensor([ids], device=DEV)
    with torch.no_grad():
        out = model(x, return_hidden=True)
    # return_hidden gives (logits, hidden) or similar; take both defensively
    hidden = None
    if isinstance(out, tuple):
        for t in out:
            if torch.is_tensor(t) and t.shape[-1] == cfg.d:
                hidden = t
    assert hidden is not None, "return_hidden did not give a [B,T,d] tensor; cannot get pre-softcap logits"
    # PRE-softcap comes from the raw head, NOT from inverting tanh on the emitted logits:
    # atanh blows up as |logit| -> softcap, and the emitted values sit within 0.4 of the cap,
    # so the inversion would be numerically dominated by exactly the tail we care about.
    with torch.no_grad():
        pre = model.head(hidden)[0, :, :real].float()
    post = (softcap * torch.tanh(pre / softcap)) if softcap else pre

    a = pre.abs()
    # NOT quantile(): it refuses over 2^24 elements and this is 4096*32773 = 134M. kthvalue
    # on the flattened tensor is the same number without the cap.
    flat = a.flatten()
    def q(f):
        k = max(1, min(flat.numel(), int(round(f * flat.numel()))))
        return float(flat.kthvalue(k).values)
    res = {
        "ckpt": os.path.basename(path), "softcap": softcap, "n_tok": int(pre.shape[0]),
        "pre_absmax": float(a.max()), "pre_p999": q(0.999),
        "pre_p99": q(0.99), "pre_std": float(pre.std()),
        "post_absmax": float(post.abs().max()),
        # the load-bearing number: what fraction of positions is the cap actually squashing
        "frac_pre_over_cap": float((a > softcap).float().mean()) if softcap else None,
        "frac_pre_over_half_cap": float((a > softcap / 2).float().mean()) if softcap else None,
        "top1_minus_top2_mean": float((pre.topk(2, dim=-1).values.diff(dim=-1).abs()).mean()),
        "entropy_nats_mean": float(-(post.softmax(-1) * post.log_softmax(-1)).sum(-1).mean()),
    }
    if True:
        h = hidden[0].float()
        res["hidden_rms"] = float(h.pow(2).mean().sqrt())
        # cos(theta) between each position's hidden and its own argmax row: the term the
        # bound assumed was 1.0
        W = model.head.weight[:real].float().detach()
        am = pre.argmax(-1)
        hn = h / h.norm(dim=-1, keepdim=True)
        wn = W[am] / W[am].norm(dim=-1, keepdim=True)
        res["cos_to_argmax_row_mean"] = float((hn * wn).sum(-1).mean().detach())
        res["head_row_norm_med"] = float(W.norm(dim=1).median())
        res["bound_absmax"] = res["hidden_rms"] * (cfg.d ** 0.5) * res["head_row_norm_med"]
    del model
    torch.cuda.empty_cache()
    return res

def main():
    tok = load_tokenizer(os.path.join(ROOT, "data", "tokenizer.json"), None)
    # Real text, not random ids: random ids would put the model off-distribution, and the
    # question is what the logits look like on data it was trained on.
    with open(os.path.join(ROOT, "data", "eval", "lambada_en", "lambada_test_en.jsonl"), encoding="utf-8") as f:
        txt = " ".join(json.loads(l)["text"] for l in list(f)[:200])
    ids = tok.encode(txt, add_special_tokens=False).ids[:N_TOK]
    print(f"{len(ids)} tokens of real text\n", flush=True)
    rows = []
    for c in CKPTS:
        p = os.path.join(ROOT, c)
        if not os.path.exists(p):
            print(f"skip missing {c}", flush=True); continue
        r = read(p, ids)
        rows.append(r)
        print(json.dumps(r, indent=1), flush=True)
    with open(os.path.join(ROOT, "runs", "_logit_dist.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("\nwrote runs/_logit_dist.json")

main()

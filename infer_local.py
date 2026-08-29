#!/usr/bin/env python3
"""Local inference for the aupai HybridLM (K3: KDA + Gated MLA) on Mac (MPS/CPU).

Pure PyTorch — no fla / Triton / CUDA required. `fla.ops.kda.chunk_kda` is
replaced by a sequential float32 recurrence that is mathematically identical
to fla's `fused_recurrent_kda` kernel (verified against the Triton source):

    gate  = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))   # per (head, dim)
    S     = exp(gate) * S                    # forget gate, decays over the key dim
    v     = beta * (v - S @ k)               # delta rule, computed on the decayed S
    S     = S + outer(v, k)
    out   = S @ q                            # q is L2-normalized then scaled by 1/sqrt(K)

Usage:
    uv run python3 infer_local.py "中国的首都是哪里"
    uv run python3 infer_local.py                       # interactive REPL
"""

import argparse
import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(ROOT) == "scripts":  # tolerate a copy under scripts/
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import fone  # noqa: E402
from train import AttnRes, Source, remap_legacy_state_dict  # noqa: E402


def find_latest_ckpt():
    """Find the most recently modified .pt checkpoint in the project root."""
    import glob

    ckpts = glob.glob(os.path.join(ROOT, "ckpt*.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=os.path.getmtime)


# ---------------------------------------------------------------- KDA recurrence


def kda_forward(q, k, v, g, beta, A_log, dt_bias, state=None, lower_bound=-5.0):
    """Pure-PyTorch replacement for `fla.ops.kda.chunk_kda` at inference.

    Shapes:
        q, k, v: (B, T, H, K)
        g:       (B, T, H, K)   data-dependent gate input
        beta:    (B, T, H)      raw logits (sigmoid applied here)
        A_log:   (H,)
        dt_bias: (H * K,)
        state:   (B, H, V, K) float32 recurrent state (None = zeros)

    Returns (out (B, T, H, V), new_state (B, H, V, K)).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = K**-0.5

    # The Triton kernel accumulates in float32 — do the same for stability.
    q, k, v = q.float(), k.float(), v.float()
    g, beta = g.float(), beta.float()

    # QK L2 norm (use_qk_l2norm_in_kernel=True, eps=1e-6), then scale q
    q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6) * scale
    k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)

    A = torch.exp(A_log.float()).view(1, 1, H, 1)
    bias = dt_bias.float().view(1, 1, H, K)
    decay = torch.exp(lower_bound * torch.sigmoid(A * (g + bias)))  # (B, T, H, K)

    beta = torch.sigmoid(beta)  # use_beta_sigmoid_in_kernel

    S = q.new_zeros(B, H, V, K) if state is None else state.float()
    out = torch.empty(B, T, H, V, device=q.device)
    for t in range(T):
        S = S * decay[:, t].unsqueeze(2)
        kt, vt = k[:, t], v[:, t]
        vt = vt - (S @ kt.unsqueeze(-1)).squeeze(-1)
        vt = vt * beta[:, t].unsqueeze(-1)
        S = S + vt.unsqueeze(-1) * kt.unsqueeze(-2)
        out[:, t] = (S @ q[:, t].unsqueeze(-1)).squeeze(-1)
    return out, S


# ---------------------------------------------------------------- Model
# Mirrors train.py's HybridLM (same param names/shapes so checkpoints load directly):
# chunk_kda -> kda_forward, plus KV/state caches for incremental decoding.


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        # float32 for stability; free at this size.
        in_dtype = x.dtype
        x = x.float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g.float()).to(in_dtype)


class DeltaRecurrence(nn.Module):
    """KDA: bounded decay + ShortConv + QK-norm. Pure-PyTorch recurrence."""

    def __init__(self, cfg):
        super().__init__()
        self.h, self.hd = cfg.heads, cfg.d // cfg.heads
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.o = nn.Linear(cfg.d, cfg.d, bias=False)
        self.gb = nn.Linear(cfg.d, cfg.d + cfg.heads + (-cfg.heads) % 16, bias=False)  # gate|beta|pad
        self.A_log = nn.Parameter(torch.zeros(cfg.heads))
        self.dt_bias = nn.Parameter(torch.zeros(cfg.heads * self.hd))
        self.short_conv = nn.Conv1d(cfg.d, cfg.d, kernel_size=4, padding=0, groups=cfg.d)

    def forward(self, x, cache=None):
        # cache = (state (B,H,V,K), conv_state (B,3,D)) or None
        B, T, D = x.shape
        state, conv_state = cache if cache is not None else (None, None)
        ksize = self.short_conv.kernel_size[0]
        if conv_state is None:
            conv_state = x.new_zeros(B, ksize - 1, D)
        # Causal conv: rolling window of the last ksize-1 inputs; zeros on first
        # call reproduces F.pad(x, (3, 0)).
        xc = torch.cat([conv_state, x], dim=1)
        h = F.silu(self.short_conv(xc.transpose(1, 2)).transpose(1, 2))
        new_conv = xc[:, -(ksize - 1) :, :]

        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.reshape(B, T, self.h, self.hd)
        k = k.reshape(B, T, self.h, self.hd)
        v = v.reshape(B, T, self.h, self.hd)
        gb = self.gb(x)
        g = gb[..., : self.h * self.hd].reshape(B, T, self.h, self.hd)
        beta = gb[..., self.h * self.hd : self.h * self.hd + self.h]

        out, new_state = kda_forward(q, k, v, g, beta, self.A_log, self.dt_bias, state=state)
        return self.o(out.reshape(B, T, D).to(x.dtype)), (new_state, new_conv)


class SlidingWindowAttention(nn.Module):
    """K3 Gated MLA: latent KV compression + full attention (NoPE, KDA handles position)."""

    def __init__(self, cfg):
        super().__init__()
        self.h, self.hd = cfg.heads, cfg.d // cfg.heads
        self.latent = cfg.d // 4
        self.kv_down = nn.Linear(cfg.d, self.latent, bias=False)
        self.kv_up = nn.Linear(self.latent, 2 * cfg.d, bias=False)  # k|v
        self.qg = nn.Linear(cfg.d, 2 * cfg.d, bias=False)  # q|gate
        self.o = nn.Linear(cfg.d, cfg.d, bias=False)

    def forward(self, x, cache=None):
        # cache = (k (B,Tpast,H,hd), v (B,Tpast,H,hd)) or None
        B, T, D = x.shape
        latent = self.kv_down(x)
        k, v = self.kv_up(latent).chunk(2, dim=-1)
        q, gate_in = self.qg(x).chunk(2, dim=-1)
        k = k.view(B, T, self.h, self.hd)
        v = v.view(B, T, self.h, self.hd)
        q = q.view(B, T, self.h, self.hd)
        q = F.rms_norm(q, (self.hd,))
        k = F.rms_norm(k, (self.hd,))

        past_k, past_v = cache if cache is not None else (None, None)
        full_k = k if past_k is None else torch.cat([past_k, k], dim=1)
        full_v = v if past_v is None else torch.cat([past_v, v], dim=1)

        q = q.transpose(1, 2)
        k2 = full_k.transpose(1, 2)
        v2 = full_v.transpose(1, 2)
        # Prefill (no cache): causal mask. Decode (T=1, cache present): the single
        # query sees every cached key, so no mask is needed.
        y = F.scaled_dot_product_attention(q, k2, v2, is_causal=(past_k is None))
        y = y.transpose(1, 2).reshape(B, T, D)
        gate = torch.sigmoid(gate_in)
        return self.o(y * gate), (full_k, full_v)


class SwiGLU(nn.Module):
    """K3 SiTU-GLU: bounded activation, tracks SwiGLU near zero."""

    def __init__(self, cfg):
        super().__init__()
        self.w13 = nn.Linear(cfg.d, 2 * cfg.ffn_hidden, bias=False)
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.d, bias=False)
        self.beta1 = 4.0
        self.beta2 = 25.0

    def forward(self, x):
        a, b = self.w13(x).chunk(2, dim=-1)
        gate = self.beta1 * torch.tanh(a / self.beta1) * torch.sigmoid(b)
        up = self.beta2 * torch.tanh(self.w2(gate) / self.beta2)
        return up


class Block(nn.Module):
    def __init__(self, cfg, is_attn=False):
        super().__init__()
        self.n1 = RMSNorm(cfg.d)
        self.mixer = SlidingWindowAttention(cfg) if is_attn else DeltaRecurrence(cfg)
        self.n2 = RMSNorm(cfg.d)
        self.ffn = SwiGLU(cfg)
        attn_res = getattr(cfg, "attn_res", False)
        dyn_q = getattr(cfg, "attn_res_dyn_q", False)
        self.ar1 = AttnRes(cfg.d, dyn_q) if attn_res else None
        self.ar2 = AttnRes(cfg.d, dyn_q) if attn_res else None

    def forward(self, x, cache=None):
        mixer_out, new_cache = self.mixer(self.n1(x), cache=cache)
        x = x + mixer_out
        return x + self.ffn(self.n2(x)), new_cache


class HybridLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.padded_vocab = ((cfg.vocab + 63) // 64) * 64
        self.tok = nn.Embedding(self.padded_vocab, cfg.d)
        self.blocks = nn.ModuleList(
            [Block(cfg, is_attn=(i % cfg.attn_every == cfg.attn_every - 1)) for i in range(cfg.layers)]
        )
        self.norm = RMSNorm(cfg.d)
        self.head = nn.Linear(cfg.d, self.padded_vocab, bias=False)
        self.head.weight = self.tok.weight  # tied
        self.attn_res = getattr(cfg, "attn_res", False)
        n_sub = 2 * cfg.layers
        n_blocks = min(n_sub, getattr(cfg, "attn_res_blocks", 0) or n_sub)
        self.ar_block_ends = {round((j + 1) * n_sub / n_blocks) for j in range(n_blocks)}
        self.final_ar = AttnRes(cfg.d, getattr(cfg, "attn_res_dyn_q", False)) if self.attn_res else None
        # Mac-side reimplementation of train.HybridLM (fla's KDA kernel is
        # CUDA-only): every architecture change must be mirrored here. The FoNE
        # pieces come from fone.py, pure PyTorch.
        self.fone = getattr(cfg, "fone", False)
        if self.fone:
            self.num_proj = nn.Linear(fone.NUM_DIMS, cfg.d, bias=False)
            self.num_head = nn.Linear(cfg.d, fone.NUM_DIMS, bias=False)

    def _body(self, x, cache):
        """Same as train.HybridLM._body (plain residual or Block AttnRes), threading the KV/state cache."""
        new_caches = []
        if not self.attn_res:
            for i, b in enumerate(self.blocks):
                x, c = b(x, cache=cache[i] if cache is not None else None)
                new_caches.append(c)
            return x, new_caches
        blocks, partial, n = [Source.of(x)], [], 0
        for i, b in enumerate(self.blocks):
            for j, (ar, norm, f) in enumerate(((b.ar1, b.n1, b.mixer), (b.ar2, b.n2, b.ffn))):
                h = ar(blocks + partial)
                if j == 0:
                    out, c = f(norm(h), cache=cache[i] if cache is not None else None)
                    new_caches.append(c)
                else:
                    out = f(norm(h))
                partial = [Source.of(partial[0].v + out if partial else out)]
                n += 1
                if n in self.ar_block_ends:
                    blocks, partial = blocks + partial, []
        return self.final_ar(blocks + partial), new_caches

    def num_logits(self, hidden):
        """Per-digit logits at a position, for reading the number a [NUM] stands for."""
        return fone.digit_logits(self.num_head(hidden.to(self.num_head.weight.dtype)).float())

    def forward(self, idx, cache=None, num_vals=None):
        emb = self.tok(idx)
        if self.fone and num_vals is not None:
            mask = (idx == self.cfg.num_id).unsqueeze(-1)
            feat = fone.encode_tensor(num_vals.masked_fill(~mask.squeeze(-1), 0.0)).to(emb.dtype)
            emb = emb + torch.where(mask, self.num_proj(feat), emb.new_zeros(()))
        x, new_caches = self._body(emb, cache)
        hidden = self.norm(x)
        logits = self.head(hidden)[..., : self.cfg.vocab].float()
        logits = 15.0 * torch.tanh(logits / 15.0)
        return logits, new_caches, hidden


# ---------------------------------------------------------------- Inference


def load_model(ckpt_path, device, dtype):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    defaults = dict(
        d=1024,
        heads=8,
        layers=12,
        attn_every=4,
        attn_window=1024,
        ffn_hidden=3072,
        vocab=32772,
        seq=4096,
        grad_ckpt=False,
    )
    cfg = SimpleNamespace(**{**defaults, **ck.get("cfg", {})})
    cfg.grad_ckpt = False

    model = HybridLM(cfg)
    model.load_state_dict(remap_legacy_state_dict(ck["model"]))  # strict: partial loads generate garbage
    model = model.to(device)
    if dtype == "bf16":
        model = model.bfloat16()
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"loaded {ckpt_path} | {n_params / 1e6:.0f}M params | {cfg.layers} layers "
        f"(attn every {cfg.attn_every}) | device {device} | dtype {dtype}",
        file=sys.stderr,
    )
    return model, cfg


def sample_next(logits, prev_ids, temperature, top_k, repeat_penalty):
    if repeat_penalty != 1.0:
        for tid in set(prev_ids):
            logits[tid] = logits[tid] / repeat_penalty if logits[tid] > 0 else logits[tid] * repeat_penalty
    if temperature <= 0.0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return torch.multinomial(torch.softmax(logits, -1), 1)


@torch.no_grad()
def generate(model, tok, prompt, args, device):
    # SFT data is 问：.../答：... only: a bare question is a continuation prompt;
    # --raw feeds it verbatim for probing a base model.
    if not args.raw:
        prompt = f"问：{prompt}\n答："
    eos = tok.token_to_id("<eos>")
    # [NUM] carries no number of its own: the digit head reads it off the
    # predicting hidden state; without this a --fone checkpoint prints literal [NUM].
    num_id = model.cfg.num_id if model.fone else None
    if model.fone:
        (ids,), (vals,) = fone.encode_prompts([prompt], tok, num_id)
    else:
        ids, vals = tok.encode(prompt).ids, []
    x = torch.tensor([ids], dtype=torch.long, device=device)
    v = torch.tensor([vals], device=device) if model.fone else None
    logits, cache, hidden = model(x, num_vals=v)
    out, out_vals = list(ids), list(vals)

    t0 = time.time()
    for _ in range(args.max_new_tokens):
        nxt = sample_next(logits[0, -1].float(), out, args.temperature, args.top_k, args.repeat_penalty)
        nid = nxt.item()
        if nid == eos:
            break
        out.append(nid)
        val = 0.0
        if model.fone and nid == num_id:
            val = float(fone.decode(model.num_logits(hidden[0, -1].float())))
        out_vals.append(val)
        nv = torch.tensor([[val]], device=device) if model.fone else None
        logits, cache, hidden = model(
            torch.tensor([[nid]], dtype=torch.long, device=device), cache=cache, num_vals=nv
        )
    dt = time.time() - t0
    n_new = len(out) - len(ids)
    if model.fone:
        print(fone.decode_text(out, [x for t, x in zip(out, out_vals) if t == num_id], tok, num_id))
    else:
        print(tok.decode(out))
    print(f"\n[{n_new} tokens in {dt:.1f}s ({n_new / max(dt, 1e-9):.1f} tok/s)]", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Local inference for aupai HybridLM (MPS/CPU, no fla)")
    parser.add_argument("prompt", nargs="?", default=None, help="prompt text (omit for interactive REPL)")
    parser.add_argument("--ckpt", default=None, help="checkpoint path (default: auto-detect latest ckpt*.pt)")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repeat_penalty", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype", choices=["bf16", "fp32"], default=None, help="default: bf16 on MPS, fp32 on CPU"
    )
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="feed the prompt verbatim instead of wrapping it as 问：.../答：. Use this to probe "
        "a base checkpoint, which has never seen the chat format and only continues text.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="the vocabulary the checkpoint was trained on. data/tokenizer.json is rebuilt in place "
        "and ids do not survive a rebuild, so the wrong file decodes to fluent-looking nonsense "
        "without raising. Defaults to data/tokenizer.json and is checked against cfg.vocab.",
    )
    args = parser.parse_args()

    if args.ckpt is None:
        args.ckpt = find_latest_ckpt()
        if args.ckpt is None:
            print("ERROR: no checkpoint found (ckpt*.pt in project root)", file=sys.stderr)
            sys.exit(1)
        print(f"auto-detected latest checkpoint: {args.ckpt}", file=sys.stderr)

    torch.manual_seed(args.seed)
    device = "cpu" if args.cpu or not torch.backends.mps.is_available() else "mps"
    dtype = args.dtype or ("bf16" if device == "mps" else "fp32")

    model, _ = load_model(args.ckpt, device, dtype)
    tok_path = args.tokenizer or os.path.join(ROOT, "data", "tokenizer.json")
    tok = Tokenizer.from_file(tok_path)
    assert tok.get_vocab_size() == model.cfg.vocab, (
        f"{tok_path} has vocab {tok.get_vocab_size()} but {args.ckpt} was trained at "
        f"{model.cfg.vocab}. Pass --tokenizer with the matching file; decoding against the "
        "wrong vocabulary produces plausible-looking garbage and raises nothing."
    )

    if args.prompt is not None:
        generate(model, tok, args.prompt, args, device)
        return
    print("interactive mode — type a prompt, Ctrl-D to exit\n", file=sys.stderr)
    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip():
            generate(model, tok, prompt, args, device)


if __name__ == "__main__":
    main()

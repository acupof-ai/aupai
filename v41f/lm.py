"""V42LM: the v41f V41FModel with the surface train.py's loop reads off HybridLM.

Subclass rather than wrapper, so state_dict keys stay the V41FModel keys. What it adds:

  forward(x, targets=None, cu=None, num_vals=None, no_head=False) -> (hidden, hidden)
      train.py unpacks `hidden, _ = model(x, y, cu, v)` in the step and validate(), and
      `_, hidden = model(x, no_head=True)` in generate_batch; returning the normed pre-head hidden
      in both slots serves both. The loss is the caller's fused linear-CE over head.weight.
  cfg.vocab / cfg.num_id / cfg.fone_loss_w, blocks, moe_layers, memory, lm_logits,
  commit_moe_token_counts, aux_loss(), n_active()
  to(dtype): casts parameters and buffers EXCEPT the ones the reference keeps fp32 (head,
      router, mHC, attn_sink, the router bias, an m>1 compressor's projections) and the complex
      RoPE tables, whose imaginary part nn.Module.to(bf16) would drop (torch 2.12 warns
      "Casting complex values to real").
"""

from types import SimpleNamespace

import torch

from .model import V41FModel

KEEP_FP32 = ("head.weight", "ffn.gate.weight", "ffn.gate.bias", "attn.attn_sink", ".hc.")


def _keep_fp32(name):
    return name == "head.weight" or any(k in name for k in KEEP_FP32[1:])


class V42LM(V41FModel):
    def __init__(self, cfg, balance_alpha=1e-4, bias_gamma=1e-3, **kw):
        super().__init__(cfg, **kw)
        self.v41f_cfg = cfg
        # train.py's reads; fone is refused for v42 at launch, so num_id is never consulted
        self.cfg = SimpleNamespace(vocab=cfg.vocab_size, num_id=cfg.vocab_size - 1, fone_loss_w=0.0)
        self.memory = None
        self.moe_layers = list(range(cfg.n_layers))
        for layer in self.layers:
            layer.ffn.balance_alpha = balance_alpha
            layer.ffn.gamma = bias_gamma

    @property
    def blocks(self):
        return self.layers

    def forward(self, x, targets=None, cu=None, num_vals=None, no_head=False):
        if num_vals is not None:
            raise ValueError("v42 has no FoNE path")
        hidden, _ = super().forward(x, cu=cu, return_hidden=True)
        return hidden, hidden

    def lm_logits(self, hidden):
        return self.head(hidden)

    def aux_loss(self):
        terms = [b.ffn.aux_loss for b in self.layers if b.ffn.aux_loss is not None]
        if getattr(self, "indexer_loss", None) is not None:
            # indexer_train_mode "kl": its inputs are detached, so this only trains the indexer
            terms.append(self.indexer_loss)
        return torch.stack(terms).sum() if terms else None

    def commit_moe_token_counts(self):
        for b in self.layers:
            b.ffn.commit_token_counts()
        return len(self.layers)

    def n_active(self):
        """Parameters one token multiplies: total minus the routed experts it skips."""
        tot = sum(p.numel() for p in self.parameters())
        c = self.v41f_cfg
        routed = sum(p.numel() for n, p in self.named_parameters() if ".ffn.experts." in n)
        return tot - routed + routed * c.n_activated_experts // c.n_routed_experts

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is None:
            return super().to(*args, **kwargs)
        if device is not None:
            super().to(device, non_blocking=non_blocking)
        # an m>1 compressor pools in fp32 and builds wkv/wgate fp32 (reference); its norm stays
        # in the activation dtype like every other RMSNorm
        pooled = {f"{n}.{leaf}" for n, m in self.named_modules()
                  if n.endswith(".compressor") and m.compress_ratio > 1
                  for leaf in ("wkv.weight", "wgate.weight", "kv_state", "score_state")}
        for n, t in list(self.named_parameters()) + list(self.named_buffers()):
            if t.is_floating_point() and not _keep_fp32(n) and n not in pooled:
                t.data = t.data.to(dtype)
        return self

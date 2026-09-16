"""Compressor: softmax-gated pooling of `ratio` consecutive tokens into one KV latent.

Faithful 1:1 port of upstream Compressor (model_ref.py.ref class Compressor, ~:429-485).
Only the linear and RMSNorm providers differ: plain torch.nn.Linear and the local
v41f.norm_gate.RMSNorm, which are numerically identical to the upstream bf16/fp32
Linear and RMSNorm on the CPU P0 path.

Semantics (do not "simplify"):
- ratio == 1: one token per group -> no gate, no fp32, norm(wkv(x)) in bf16.
- ratio  > 1: project kv and a softmax score gate in fp32; pool each consecutive group
  with score.softmax(over the ratio within-group positions) as the weight, then RMSNorm
  and cast back to the input dtype.
- prefill (start_pos == 0): a trailing partial group does NOT pool; it is parked in
  kv_state/score_state for the decode step that completes it. The training path is
  prefill only, but the states and decode branch are ported verbatim so the port stays
  1:1 and the eventual generator reuses one module. The returned tensor is pre-RoPE
  (Attention applies compress RoPE afterwards; the indexer needs the unrotated form).
"""
import torch
from torch import nn

from v41f.norm_gate import RMSNorm


class Compressor(nn.Module):
    def __init__(self, dim: int, head_dim: int, compress_ratio: int,
                 norm_eps: float = 1e-6, max_batch_size: int = 4):
        super().__init__()
        if compress_ratio < 1:
            # ratio 0 marks a sliding-window-only layer in the config; such layers never
            # instantiate a Compressor. Guard here so a wrong layer index fails loudly.
            raise ValueError(f"Compressor requires compress_ratio >= 1, got {compress_ratio}")
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.norm = RMSNorm(head_dim, norm_eps)
        # ratio 1 stays in the checkpoint bf16; the softmax pooling above ratio 1 runs in
        # fp32, so those two projections are promoted to fp32 to match the reference.
        self.wkv = nn.Linear(dim, head_dim, bias=False,
                             dtype=torch.float32 if compress_ratio > 1 else torch.bfloat16)
        if compress_ratio > 1:
            self.wgate = nn.Linear(dim, head_dim, bias=False, dtype=torch.float32)
            # tail of an incomplete group, carried across decode steps
            state_shape = (max_batch_size, compress_ratio, head_dim)
            self.register_buffer("kv_state", torch.zeros(state_shape, dtype=torch.float32),
                                 persistent=False)
            self.register_buffer("score_state", torch.full(state_shape, -torch.inf,
                                                            dtype=torch.float32),
                                 persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        bsz, seqlen, _ = x.size()
        ratio, dtype = self.compress_ratio, x.dtype
        if ratio == 1:  # one token per group: nothing to pool, so no gate and no fp32
            return self.norm(self.wkv(x))

        x = x.float()
        kv, score = self.wkv(x), self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:  # trailing partial group waits in the state
                kv, kv_tail = kv.split([cutoff, remainder], dim=1)
                score, score_tail = score.split([cutoff, remainder], dim=1)
                self.kv_state[:bsz, :remainder] = kv_tail
                self.score_state[:bsz, :remainder] = score_tail
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:  # decode: one token per step; pool only when the group just completed
            should_compress = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            self.kv_state[:bsz, slot] = kv.squeeze(1)
            self.score_state[:bsz, slot] = score.squeeze(1)
            if should_compress:
                kv = (self.kv_state[:bsz]
                      * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return None
        return self.norm(kv.to(dtype))

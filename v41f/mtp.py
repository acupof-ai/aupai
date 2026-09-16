"""DSpark MTP stage-1 training loss (P3 first cut, n_mtp_layers=1, block_size=5).

Scope, deliberately smaller than upstream DSparkBlock
(third_party/deepseek_v41_ref/model_ref.py.ref:1100-1156): the reference implements
the INFERENCE path only — a sparse-attn draft block plus Markov/confidence heads and
autoregressive speculative sampling (forward_head/forward_spec). P3 omits all three:
no Markov-rank head (config dspark_markov_rank=0), no confidence head, no
forward_spec. This module is the TRAINING-only scaffolding the reference does not
have, namely:

1. main projection: the anchor layer's hidden is the concat of the attn INPUTS of
   `dspark_target_layer_ids` target layers, [B, T, dim] flattened to [B, dim*T],
   projected back to dim and RMSNormed — the same order as forward_embed:
       main_x = main_norm(main_proj(main_hidden))         (model_ref:1130)
2. draft input construction: a block_size query whose position 0 is the real anchor
   token and positions 1..block_size-1 are learned NOISE tokens, exactly
   forward_embed:1131-1135. One draft forward predicts all block_size next tokens in
   parallel; noise lives on the INPUT side only.
3. teacher-forced multi-token CE: draft position i is supervised against the real
   token at main-sequence position anchor+1+i. Every target is a REAL future token;
   a noise query position is never a supervision target.

Upstream provides NO training loss to match, so this is not an allclose module: the
weighting is v41f-defined (uniform mean CE over the real future tokens, optional
per-position weights) and is registered in the v41f prereg. The draft transformer
block and the LM head are injected callables — the real DSparkBlock (sparse-attn,
hyper-connections) and the tied head plug in at P3; here a deterministic mock lets
the loss/position/noise contract be checked in a known-answer world.
"""

import torch
import torch.nn.functional as F
from torch import nn

from v41f.norm_gate import RMSNorm


class DSparkMTP(nn.Module):
    """Stage-1 DSpark training pieces: main projection, draft ids, multi-token CE.

    dim: model width. n_target_layers: number of layers whose attn-input hidden is
    concatenated (len(dspark_target_layer_ids)). The embedding is standalone here
    (plain nn.Embedding); training ties it to the shared token embedding.
    """

    def __init__(
        self,
        dim: int,
        n_target_layers: int,
        vocab_size: int,
        block_size: int,
        noise_token_id: int = 0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if n_target_layers < 1:
            raise ValueError("DSpark needs at least one target layer")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.dim = dim
        self.block_size = block_size
        self.noise_token_id = noise_token_id
        # model_ref:1113 main_proj is a Linear with the default no-bias contract.
        self.main_proj = nn.Linear(dim * n_target_layers, dim, bias=False)
        self.main_norm = RMSNorm(dim, eps)
        self.embed = nn.Embedding(vocab_size, dim)

    def project_main(self, main_hidden: torch.Tensor) -> torch.Tensor:
        """[B, T, dim] target-layer attn-input hidden -> projected anchor [B, dim].

        Flatten order is target-layer order then dim: position t*dim:(t+1)*dim carries
        target layer t, matching the concat the caller feeds main_proj upstream.
        """
        if main_hidden.dim() != 3 or main_hidden.shape[-1] != self.dim:
            raise ValueError(f"expected [B,T,dim], got {tuple(main_hidden.shape)}")
        return self.main_norm(self.main_proj(main_hidden.flatten(-2)))

    def draft_input_ids(self, anchor_ids: torch.Tensor) -> torch.Tensor:
        """[B] anchor token -> [B, block_size] with the real token at 0, noise after."""
        if anchor_ids.dim() != 1:
            raise ValueError(f"anchor_ids must be [B], got {tuple(anchor_ids.shape)}")
        draft = anchor_ids.new_full((anchor_ids.size(0), self.block_size), self.noise_token_id)
        draft[:, 0] = anchor_ids
        return draft

    def draft_embed(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """[B, block_size] -> [B, block_size, dim]. The hc_mult expansion happens inside
        the real draft block's hyper-connections; this stays on plain dim."""
        return self.embed(draft_ids)

    def forward(
        self,
        main_hidden: torch.Tensor,
        anchor_ids: torch.Tensor,
        future_ids: torch.Tensor,
        draft_block,
        head,
        ignore_index: int = -100,
        pos_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-token teacher-forced loss.

        main_hidden: [B, T, dim] attn inputs of the target layers at the anchor.
        anchor_ids: [B] the real token at the anchor position p.
        future_ids: [B, block_size] REAL tokens at p+1 .. p+block_size (use ignore_index
            to mask a position); noise query positions are inputs, never these labels.
        draft_block: (x:[B,S,dim], main_x:[B,dim]) -> [B,S,dim], the future DSparkBlock.
        head: (hidden:[B,S,dim]) -> logits [B,S,vocab].
        pos_weights: optional [block_size] draft-position weights (v41f-defined).

        Returns (mean CE over non-ignored targets, logits [B,S,vocab]).
        """
        if future_ids.shape != (anchor_ids.size(0), self.block_size):
            raise ValueError(f"future_ids must be [B,{self.block_size}], got {tuple(future_ids.shape)}")
        main_x = self.project_main(main_hidden)
        draft_ids = self.draft_input_ids(anchor_ids)
        x = self.draft_embed(draft_ids)
        hidden = draft_block(x, main_x)
        logits = head(hidden)
        if logits.shape[:2] != future_ids.shape:
            raise ValueError(f"head logits {tuple(logits.shape)} must lead with [B,{self.block_size}]")

        log_probs = F.log_softmax(logits.float(), dim=-1)
        valid = future_ids.ne(ignore_index)
        # clamp the (masked) label to a legal column for gather; its contribution is zeroed
        # by the validity weight, so the value gathered there is never read.
        safe_labels = future_ids.clamp_min(0)
        picked = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        if pos_weights is None:
            pos_weights = torch.ones(self.block_size, device=future_ids.device, dtype=logits.dtype)
        w = pos_weights.to(logits.dtype) * valid.to(logits.dtype)
        denom = w.sum().clamp_min(1.0)
        loss = -(picked * w).sum() / denom
        return loss, logits

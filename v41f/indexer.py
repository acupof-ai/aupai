"""CSA2 second-level Indexer and first-level candidate-block selection.

Faithful pure-torch port of upstream model_ref.Indexer / select_candidate_blocks,
minus the training-only machinery P0 does not score:
- fp4 activation quant (quantises q/k in place on the SM100 kernel path; CPU allclose
  forces bf16 and treats it as the identity it is numerically),
- torch.distributed all-reduce (world_size=1 here),
- the shared_attn global and the owned k_cache publish path (index_k is passed in;
  RoPE freqs are given), so only the per-query scoring / visibility / top-k logic,
  and the candidate block pre-filter, are exercised.

Op order, dtype and the rectify-then-combine formula match the reference line for
line, so a same-weight, same-input comparison is bitwise-tight, not merely close.
"""
import torch
import torch.nn.functional as F
from torch import nn


def apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent pairs as complex numbers. Mirrors upstream apply_rotary_emb
    (no inverse): upcasts to float for the multiply, returns to x's dtype. Out of
    place; `freqs` is complex and already sliced to the query's positions."""
    out_dtype = x.dtype
    z = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    shape = (1, z.size(1), z.size(-1)) if z.ndim == 3 else (1, z.size(1), 1, z.size(-1))
    return torch.view_as_real(z * freqs.view(*shape)).flatten(-2).to(out_dtype)


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: "torch.Tensor | int",
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one: keep the `topk_blocks` best blocks per query as a bool mask.

    Block score is the max over its positions; -inf pads the short last block. The
    block holding each query's newest compressed position is pinned to +inf so a
    partially filled recent block is never dropped; leftover -inf top-k picks (fewer
    reachable blocks than topk_blocks) are discarded. Output is width-aligned to
    `logits` [..., width].
    """
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class Indexer(nn.Module):
    """Scores one shared index key per compressed position with fp4-style multi-head
    queries, rectifies, combines heads through `weights_proj`, applies the causal
    visibility mask (and optional level-one candidate mask), then returns the
    `index_topk` visible positions re-sorted into ascending position order;
    unreachable slots are -1, visible slots shifted by `offset`. int32.
    """

    def __init__(self, dim: int, q_lora_rank: int, n_heads: int,
                 index_head_dim: int, rope_head_dim: int, index_topk: int,
                 compress_ratio: int):
        super().__init__()
        self.n_heads = n_heads
        self.index_head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.compress_ratio = compress_ratio
        self.softmax_scale = index_head_dim ** -0.5
        # upstream: ColumnParallelLinear on each; world=1 so a plain Linear is the
        # whole output shard. weights_proj is explicitly bf16 upstream.
        self.wq_b = nn.Linear(q_lora_rank, n_heads * index_head_dim, bias=False)
        self.weights_proj = nn.Linear(dim, n_heads, bias=False)

    def score(self, x: torch.Tensor, qr: torch.Tensor, index_k: torch.Tensor,
              freqs: torch.Tensor) -> torch.Tensor:
        """[b,s,dim],[b,s,q_lora],[b,t,hkd],complex[s,rd/2] -> raw [b,s,t] scores
        pre-visibility-mask (unreachable positions still finite here)."""
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.index_head_dim))
        # only the last rope_head_dim dims rotate; the leading head dims stay linear
        q_rot = apply_rotary_emb(q[..., -self.rope_head_dim:], freqs)
        q = torch.cat([q[..., :-self.rope_head_dim], q_rot], dim=-1)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, index_k)
        return (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

    def forward(self, x: torch.Tensor, qr: torch.Tensor, index_k: torch.Tensor,
                freqs: torch.Tensor, start_pos: int, offset: int,
                candidates: "torch.Tensor | None" = None) -> torch.Tensor:
        """`freqs` is the query RoPE table already sliced to [start_pos:end_pos]'s
        length in positions (complex, index_head_dim/2). `index_k` is
        [b, end_pos//ratio, index_head_dim], already built/rotated by the owner."""
        idxs, _ = self.select(x, qr, index_k, freqs, start_pos, offset, candidates)
        return idxs

    def select(self, x: torch.Tensor, qr: torch.Tensor, index_k: torch.Tensor,
               freqs: torch.Tensor, start_pos: int, offset: int,
               candidates: "torch.Tensor | None" = None):
        """The selection seam: (hard idxs, continuous scores at those slots).

        `forward` is exactly `select(...)[0]`, so every existing caller and the bit-exact
        oracles see the same integers they saw before. The second value is what the
        training-only straight-through path needs: `score` is the SAME tensor the hard topk
        consumed (post-visibility-mask, in-place masked), and `sc = score.gather(-1, idxs)`
        is a view of it at the selected slots -- NOT a recomputation.

        Recomputing the score on a training side path is the fed-dead-weight shape: an
        allclose over fed values goes green while the real default path leaves
        `wq_b`/`weights_proj` unused. The caller must not call `score()` again.
        """
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        end_pos = start_pos + seqlen

        # THE SCORE'S INPUTS ARE DETACHED, which makes the training gradient
        # INDEXER-LOCAL (design doc §2, gate 4). The score is the only consumer of a graph
        # here -- the hard topk reads values, and ints carry no derivative -- so cutting the
        # inputs costs the faithful path nothing (measured: both indexer grads are None off,
        # with or without the detach) and stops the straight-through signal, which enters
        # through `score`, from leaking into wq_b's INPUTS. Without it the measurement is:
        # qproj.wq_a/q_norm grads move by 3.7e-09, and index_key.wk/k_norm go from None to
        # having grads -- i.e. the "indexer-local" property is false and, in a real model
        # where x is the residual stream, the STE would push gradient into the whole
        # backbone rather than into two projection weights.
        #
        # wq_b.weight / weights_proj.weight keep their gradient: a weight's gradient depends
        # on its input's VALUE, which detach does not touch.
        index_score = self.score(x.detach(), qr.detach(), index_k.detach(), freqs)

        if start_pos == 0:
            compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
            index_score.masked_fill_(
                torch.arange(seqlen // ratio, device=x.device) >= compress_lens, -torch.inf)
        else:
            compress_lens = end_pos // ratio

        if candidates is not None:
            index_score = index_score.masked_fill(~candidates, -torch.inf)

        topk = min(self.index_topk, end_pos // ratio)
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        # gather on the -inf-padded row must not read an out-of-range column; the shift is a
        # no-op in value (-inf stays -inf) and keeps every index in range.
        sc = index_score.gather(-1, idxs.clamp_max(index_score.size(-1) - 1))
        return torch.where(idxs < compress_lens, idxs + offset, -1).int(), sc

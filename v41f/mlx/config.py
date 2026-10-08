"""MLX-side config mirroring v41f.config.V41FConfig for the v42 gate stack.

Only the fields the MLX forward reads are carried. Built from a checkpoint's
``cfg.v42_cfg`` dict (see scripts/loader.load_checkpoint), never from the live
training Cfg.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MLXV42Config:
    # shape
    vocab_size: int
    dim: int
    n_layers: int
    # attention
    n_heads: int
    head_dim: int
    rope_head_dim: int
    q_lora_rank: int
    o_groups: int
    o_lora_rank: int
    window_size: int
    rope_theta: float
    compress_ratios: tuple
    kv_source_layers: tuple
    index_source_layers: tuple
    compress_rope_theta: float
    # indexer
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    # MoE
    n_routed_experts: int
    n_shared_experts: int
    n_activated_experts: int
    moe_inter_dim: int
    score_func: str
    gate_temp: float
    norm_topk_prob: bool
    route_scale: float
    swiglu_limit: float
    norm_eps: float
    # hyper-connections
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    # engram
    engram_layer_ids: tuple
    engram_max_ngram_size: int
    engram_n_heads: int
    engram_head_dim: int
    engram_vocab_size: int
    engram_pad_id: int
    engram_num_embeddings: tuple
    engram_compressed_vocab_size: int
    # numerics
    attn_logit_softcap: float = 0.0

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    @classmethod
    def from_v42_cfg(cls, v42_cfg: dict) -> "MLXV42Config":
        """Build from the dict stored at ck['cfg']['v42_cfg'].

        The checkpoint dict carries the exact field names V41FConfig uses; we
        pick them by name so a new field only needs a line here.
        """
        keys = cls.__dataclass_fields__.keys()
        kwargs = {}
        for k in keys:
            if k in v42_cfg:
                v = v42_cfg[k]
                # lists in the JSON dict become tuples for hashing
                if isinstance(v, list):
                    v = tuple(v)
                kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def from_v41f_config(cls, cfg) -> "MLXV42Config":
        """Build from a live V41FConfig dataclass (e.g. v41f_small())."""
        return cls.from_v42_cfg(asdict(cfg))

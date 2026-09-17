"""v41f configuration. Field names mirror upstream ModelArgs (inference/model.py)
wherever they describe the same quantity, so P0 can map a module onto the reference
by name. This is the single interface contract every parallel implementation imports;
do not add a field without the upstream name (or a `# v41f-only:` comment).

Two shapes:
- V41F_SMALL: same-as-upstream reference shapes, used by P0 allclose on CPU.
- v41f_s(): the trainable 8xH20 config (see docs/standards/v41_faithful_repro.md §3).

v41f_s() size, measured by scripts/v41f_param_count.py: 0.9046 B total params,
210.95 M active/token (23.32%). This is the faithful V4.1-Flash-S config, not the retired
~350M-active r3 gate line; do not retune these shapes toward that older number.
"""
from dataclasses import dataclass, fields, replace
from typing import Literal


@dataclass(frozen=True)
class V41FConfig:
    # shape
    vocab_size: int = 32768
    dim: int = 1024
    n_layers: int = 12
    # attention
    n_heads: int = 8
    head_dim: int = 128
    rope_head_dim: int = 32
    q_lora_rank: int = 256
    o_groups: int = 8
    o_lora_rank: int = 128
    window_size: int = 128
    rope_theta: float = 10000.0
    # one entry per layer: 0 = sliding window only, r = pool r:1
    compress_ratios: tuple[int, ...] = (0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1)
    kv_source_layers: tuple[int, ...] = (2,)
    index_source_layers: tuple[int, ...] = (2, 4, 8)
    # second-level candidate pre-filter; <0 turns it off (short-context v41f-S)
    candidate_source_layer: int = -1
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0
    compress_rope_theta: float = 160000.0
    # YaRN; original_seq_len<=0 disables extrapolation (v41f-S trains at 4096)
    original_seq_len: int = 0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    # indexer
    index_n_heads: int = 4
    index_head_dim: int = 64
    index_topk: int = 64
    # MoE
    n_routed_experts: int = 48
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    moe_inter_dim: int = 448
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    norm_eps: float = 1e-6
    # hyper-connections
    hc_mult: int = 2
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # engram (); () disables
    engram_layer_ids: tuple[int, ...] = (1,)
    engram_max_ngram_size: int = 4
    engram_n_heads: int = 4
    engram_head_dim: int = 128
    engram_vocab_size: int = 65536          # hash-bucket modulus, NOT compressed-vocab size
    engram_pad_id: int = 2
    # Table rows per engram layer, one entry per engram layer. The class default () is the
    # OFF path; a config with engram_layer_ids set must carry the values EngramLayout
    # derives, or Engram indexes a zero-row table (nn.Embedding(0, d) builds fine and only
    # fails on the first forward index). Use derive_engram_num_embeddings / with_derived_engram
    # rather than writing the numbers: they are the sum of a layer's bucket primes and move
    # with engram_vocab_size, engram_n_heads and engram_max_ngram_size.
    engram_num_embeddings: tuple[int, ...] = ()
    # filled in from the real tokenizer; assert-checked at build (see Engram)
    engram_compressed_vocab_size: int = 0
    # DSpark MTP draft head; 0 draft layers disables it
    n_mtp_layers: int = 1
    dspark_block_size: int = 5
    dspark_target_layer_ids: tuple[int, ...] = (8,)
    dspark_markov_rank: int = 0             # 0 = Markov head not built (P3 omits it)
    dspark_noise_token_id: int = 0
    # draft-block experts; 0 falls back to the backbone counts (ref get_moe_config `or`)
    dspark_n_routed_experts: int = 0
    dspark_n_activated_experts: int = 0
    # rank>0 builds the inference Markov/confidence heads (later PR); rank 0 omits them

    # Straight-through training signal for the CSA2 second-level indexer (v41f-defined).
    # "off" is the faithful path and constructs nothing; "ste" feeds an identity-valued,
    # softmax-differentiated slot weight into sparse_attn so wq_b/weights_proj receive the
    # main-CE gradient. NOT a reference field: the vendored model has no differentiable
    # indexer path, which is why this is the one V41FConfig field the ref ModelArgs cannot
    # consume (see the intersection gate's CFG_ONLY_TRAINING_KNOBS).
    indexer_train_mode: str = "off"

    def validate(self) -> None:
        if len(self.compress_ratios) != self.n_layers:
            raise ValueError(
                f"compress_ratios has {len(self.compress_ratios)} entries, "
                f"need one per layer ({self.n_layers})")
        if self.head_dim <= self.rope_head_dim:
            raise ValueError("head_dim must exceed rope_head_dim (nope + rope split)")
        if self.n_heads % self.o_groups:
            raise ValueError("n_heads must divide evenly into o_groups")
        for l in self.kv_source_layers:
            if not 0 <= l < self.n_layers or self.compress_ratios[l] == 0:
                raise ValueError(f"kv_source layer {l} missing or window-only")
        if self.n_activated_experts > self.n_routed_experts:
            raise ValueError("activated experts exceed routed experts")
        if self.hc_mult < 1:
            raise ValueError("hc_mult >= 1")
        if self.engram_num_embeddings and len(self.engram_num_embeddings) != len(self.engram_layer_ids):
            raise ValueError(
                f"engram_num_embeddings has {len(self.engram_num_embeddings)} entries for "
                f"{len(self.engram_layer_ids)} engram layers")
        if self.indexer_train_mode not in ("off", "ste"):
            raise ValueError(
                f"indexer_train_mode must be 'off' or 'ste', got {self.indexer_train_mode!r}")

    def derived_engram_num_embeddings(self) -> tuple[int, ...]:
        """Table rows per engram layer = that layer's sum of bucket primes.

        Derived from the SAME EngramLayout.from_args the model will build, so the number and
        the module cannot disagree. Returns () when the engram is off, which is what keeps
        the field's default honest: nothing to fill in when there is no layer to fill.

        This is not arithmetic that could be inlined: `from_args` draws the primes in order
        from `engram_vocab_size`, skipping already-used ones, so the sum moves with
        engram_vocab_size, engram_n_heads and engram_max_ngram_size together.
        """
        from .engram import EngramLayout

        layout = EngramLayout.from_args(self)
        if layout is None:
            return ()
        return tuple(sum(p for ngram in layer for p in ngram) for layer in layout.primes)

    def with_derived_engram(self) -> "V41FConfig":
        """A copy with engram_num_embeddings filled from the layout; raises if the engram is
        on and no tokenizer was supplied, because a compressed-vocab size of 0 cannot have
        been measured and Engram asserts on it."""
        return replace(self, engram_num_embeddings=self.derived_engram_num_embeddings())

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim


def v41f_s() -> V41FConfig:
    """Trainable 8xH20 config."""
    c = V41FConfig()
    c.validate()
    return c


def v41f_small(**over) -> V41FConfig:
    """Upstream-small reference shape for P0 allclose (CPU). Mirrors ModelArgs defaults:
    dim1024 / 5 layers / 8 experts / q_lora256 / head_dim128 / rope32 / hc_mult4.
    Overrides move it onto our pure-torch path (no vision, no TP)."""
    c = V41FConfig(
        vocab_size=12800,
        dim=1024,
        n_layers=5,
        n_heads=16,
        head_dim=128,
        rope_head_dim=32,
        q_lora_rank=256,
        o_groups=16,
        o_lora_rank=64,
        compress_ratios=(0, 2, 2, 1, 1),
        kv_source_layers=(1, 3),
        index_source_layers=(1, 3),
        index_n_heads=16,
        index_head_dim=64,
        index_topk=64,
        n_routed_experts=8,
        n_activated_experts=2,
        moe_inter_dim=1024,
        swiglu_limit=0.0,
        hc_mult=4,
        engram_layer_ids=(),
        n_mtp_layers=0,
        dspark_block_size=0,
        dspark_target_layer_ids=(),
        candidate_source_layer=-1,
        original_seq_len=0,
    )
    for k, v in over.items():
        c = __import__("dataclasses").replace(c, **{k: v})
    c.validate()
    return c


if __name__ == "__main__":
    for name, c in (("v41f_s", v41f_s()), ("v41f_small", v41f_small())):
        n = sum(1 for _ in fields(c))
        print(f"{name}: dim{c.dim} L{c.n_layers} h{c.n_heads}hd{c.head_dim} "
              f"E{c.n_routed_experts}top{c.n_activated_experts} hc{c.hc_mult} "
              f"ratios={c.compress_ratios} ({n} fields)")

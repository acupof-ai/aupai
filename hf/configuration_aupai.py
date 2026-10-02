"""HF PretrainedConfig for the aupai v42 stack.

Every architectural field lives under `v42_cfg`, verbatim as the checkpoint's `cfg["v42_cfg"]`
wrote it, and is handed to `v41f.config.V41FConfig` unchanged. The flat fields beside it are
the ones transformers itself reads (vocab_size, hidden_size, num_hidden_layers,
num_attention_heads, max_position_embeddings); they are DERIVED from v42_cfg at construction,
never a second source of truth -- a config whose flat field disagrees with its v42_cfg entry
would load a different model than it describes.
"""

from transformers import PretrainedConfig


class AupaiV42Config(PretrainedConfig):
    model_type = "aupai_v42"

    def __init__(self, v42_cfg=None, vocab_id=None, train_step=None, **kw):
        self.v42_cfg = dict(v42_cfg or {})
        self.vocab_id = vocab_id
        self.train_step = train_step
        c = self.v42_cfg
        kw.setdefault("vocab_size", c.get("vocab_size"))
        kw.setdefault("hidden_size", c.get("dim"))
        kw.setdefault("num_hidden_layers", c.get("n_layers"))
        kw.setdefault("num_attention_heads", c.get("n_heads"))
        kw.setdefault("max_position_embeddings", c.get("max_seq_len", 4096))
        kw.setdefault("tie_word_embeddings", False)
        # V42LM has no incremental-decoding path: every forward recomputes the whole prefix, so
        # a cache cannot be reported as present. generate() still works, at O(n^2).
        kw["use_cache"] = False
        super().__init__(**kw)

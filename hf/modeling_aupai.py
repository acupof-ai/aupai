"""HF CausalLM wrapper over v41f's V42LM.

The weights are the training weights unchanged: this file adds the transformers interface and
nothing else, so a logit from here must equal a logit from `v41f.lm.V42LM` on the same input.
`scripts/export_hf.py --verify` asserts exactly that against the source checkpoint.

V41F ON THE IMPORT PATH. The exported directory carries a copy of `v41f/` beside this file,
but `trust_remote_code` copies only the .py modules it resolves into its own cache directory,
so `os.path.dirname(__file__)` is NOT where v41f sits at load time. The model directory is
recovered from `config._name_or_path`, which transformers sets to the path or repo id it
loaded from; `_v41f_dir` tries that first and falls back to this file's directory for a plain
`import modeling_aupai` next to the weights.
"""

import os
import sys

import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_aupai import AupaiV42Config
except ImportError:
    from configuration_aupai import AupaiV42Config


def _v41f_dir(config):
    for cand in (getattr(config, "_name_or_path", None), os.path.dirname(os.path.abspath(__file__))):
        if cand and os.path.isdir(os.path.join(cand, "v41f")):
            return cand
    raise ImportError(
        "the v41f package was not found beside the model. The exported directory must keep its "
        "v41f/ copy; load with a local path (from_pretrained('/path/to/export')) or snapshot the "
        "whole repo first, because trust_remote_code fetches only .py modules it can resolve."
    )


def _import_v41f(config):
    d = _v41f_dir(config)
    if d not in sys.path:
        sys.path.insert(0, d)
    from v41f.config import V41FConfig
    from v41f.lm import V42LM

    return V41FConfig, V42LM


class AupaiV42ForCausalLM(PreTrainedModel):
    config_class = AupaiV42Config
    base_model_prefix = "model"
    _supports_cache_class = False

    def __init__(self, config):
        super().__init__(config)
        V41FConfig, V42LM = _import_v41f(config)
        fields = {f.name for f in __import__("dataclasses").fields(V41FConfig)}
        unknown = sorted(set(config.v42_cfg) - fields)
        if unknown:
            raise ValueError(
                f"config.v42_cfg carries {len(unknown)} field(s) V41FConfig does not define: "
                f"{unknown[:6]}. The export was made against a different v41f than this copy, so "
                f"the shape fields cannot be trusted; re-export rather than dropping them."
            )
        self.model = V42LM(V41FConfig(**{k: v for k, v in config.v42_cfg.items()}))
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed

    def set_input_embeddings(self, value):
        self.model.embed = value

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        """attention_mask is accepted and IGNORED: the stack is causal over whole rows and has no
        padding path, so a mask that is not all-ones would silently not take effect. Pad-free
        batching goes through v41f's cu_seqlens instead (`train.doc_cu_seqlens`), which this
        interface does not expose."""
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError(
                "AupaiV42ForCausalLM has no padding path: a non-all-ones attention_mask would be "
                "ignored. Batch equal-length rows, or call v41f.lm.V42LM with cu_seqlens."
            )
        logits, _ = self.model(input_ids)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kw):
        # No KV cache: every step re-reads the whole prefix. Correct, and O(n^2) in the prefix.
        return {"input_ids": input_ids}

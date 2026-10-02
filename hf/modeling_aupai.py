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

    @classmethod
    def from_pretrained(cls, *args, **kw):
        """Load, then restore the checkpoint's own per-tensor dtypes.

        transformers loads every tensor into the model's default fp32 unless told otherwise, so
        a bf16 export silently doubles in memory AND changes its numerics: measured on the tiny
        mixed fixture, an all-fp32 load differs from the training dtypes by max|delta| 1.7e-2 on
        the logits -- bf16 rounding, not a wrapper bug, but the export is then not the model that
        was trained.

        The target comes from the export's own `param_dtype_default` /
        `param_dtype_exceptions`, which the exporter READ off the saved tensors -- not from
        `v41f.lm.KEEP_FP32`, which is the training policy and would be a second source of truth
        that silently diverges from a file written under an older one. An all-fp32 checkpoint
        therefore stays all-fp32, and a mixed one is restored tensor by tensor.

        Every cast here is lossless given how the file was written: a tensor saved bf16 and
        loaded into fp32 returns to its exact bits, and a tensor saved fp32 is left alone.
        Passing torch_dtype=bfloat16 instead would round head.weight through bf16 and lose
        precision the training run deliberately kept.
        """
        result = super().from_pretrained(*args, **kw)
        # output_loading_info=True makes transformers return (model, info); the dtype restore
        # applies either way and the caller's shape is handed back unchanged.
        model = result[0] if isinstance(result, tuple) else result
        default = getattr(model.config, "param_dtype_default", None)
        if default is None:
            return result  # an export from before this field: leave what transformers loaded
        exceptions = getattr(model.config, "param_dtype_exceptions", {}) or {}
        for name, p in model.model.named_parameters():
            want = getattr(torch, exceptions.get(name, default))
            if p.dtype != want:
                p.data = p.data.to(want)
        model._restore_nonpersistent_buffers(args[0] if args else kw.get("pretrained_model_name_or_path"))
        return result

    def _restore_nonpersistent_buffers(self, model_dir):
        """Copy in the persistent=False buffers the export carries.

        100 of this stack's buffers (25.2 MB, almost all of it the 24 per-layer freqs_cis RoPE
        tables) are registered persistent=False: absent from the checkpoint by design, computed
        by each module's __init__. transformers materialises the module from the weight files
        alone and never runs that __init__ against a real device, so every such buffer arrives
        as uninitialised memory. Measured on the real step16000 export: 0 missing keys, 0
        unexpected keys, 0 NaN parameters, and NaN LOGITS out of 8-9 NaN buffers.
        low_cpu_mem_usage=False does not change it -- transformers 5.6 no longer honours the
        flag -- which is why the fix lives in the export rather than in a load argument.

        A missing file is a hard error, not a warning: without these buffers the model answers
        NaN, and a model that loads and then answers NaN is the failure this whole wrapper is
        built to prevent.
        """
        import os

        from safetensors.torch import load_file

        path = os.path.join(str(model_dir), "nonpersistent_buffers.safetensors")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} is missing. It carries the persistent=False buffers (freqs_cis, the "
                f"compressor states, the MoE token counters); without them this model returns "
                f"NaN. Re-export with scripts/export_hf.py, which writes it beside the shards."
            )
        live = dict(self.model.named_buffers())
        for name, value in load_file(path).items():
            if name in live:
                live[name].copy_(value.to(live[name].dtype))

    def _autocast(self):
        """The dtype context the weights were trained in, enabled only when they are mixed.

        A gate checkpoint is NOT single-dtype: `v41f.lm.KEEP_FP32` holds `head.weight`,
        `ffn.gate.*`, `attn.attn_sink` and every `.hc.` tensor in fp32 while the rest is bf16,
        and the training loop makes that work by running the step under `torch.autocast`.
        Without the same context the stack raises at the first matmul whose operands straddle
        the split -- `V41FHead.forward` calls `F.linear(x.float(), self.weight)`, so a bf16
        weight meets an fp32 activation and torch refuses ("expected m1 and m2 to have the same
        dtype", measured on the real step16000 export 2026-10-03, which wrote every file and
        then failed its own verify).

        Enabled by the WEIGHTS, not by a flag: a model loaded entirely in fp32 (what a fresh
        V42LM build and `--no-verify`-free tests produce) needs no autocast and gets none, so
        the context cannot change a single-dtype result.
        """
        mixed = any(p.dtype == torch.bfloat16 for p in self.model.parameters())
        dev = next(self.model.parameters()).device.type
        return torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=mixed)

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
        with self._autocast():
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

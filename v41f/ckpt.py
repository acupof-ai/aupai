"""v41f checkpoint save/load: model weights plus the V41FConfig that shapes them.

A checkpoint is a state_dict STRICTLY paired with its config (a V41FConfig rebuilt with a
mismatched field is a load error, not a silent reshape). Loading constructs under the
bf16 default dtype exactly as inference/training do, then copies the saved tensors; the
fp32 LM head is stored fp32. Optimizer/master-weight state is a training concern and is
not part of this model checkpoint.
"""

from dataclasses import asdict

import torch

from .config import V41FConfig
from .model import V41FModel


def save_checkpoint(path, model: V41FModel, cfg: V41FConfig) -> None:
    torch.save({"config": asdict(cfg), "state_dict": model.state_dict()}, path)


def load_checkpoint(path, max_batch_size: int = 4, eval_mode: bool = True, *, tokenizer=None):
    """Inference loader. Keyword-only `tokenizer` is required for an engram-on config (the
    model cannot build NgramHashState without it); engram-off allows None.

    Refuses a TRAINING blob (format 'v41f_train_ckpt') -- use v41f.master.load_train_checkpoint
    for that (M11: the two formats are disjoint and a training blob cannot be read here).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("format") == "v41f_train_ckpt":
        raise ValueError("this is a v41f TRAINING checkpoint; load it with v41f.master.load_train_checkpoint")
    cfg = V41FConfig(**ckpt["config"])
    if cfg.engram_layer_ids and tokenizer is None:
        raise ValueError("an engram-on checkpoint requires the injected tokenizer= for load_checkpoint")
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = V41FModel(cfg, max_batch_size=max_batch_size, tokenizer=tokenizer)
        model.load_state_dict(ckpt["state_dict"], strict=True)
    finally:
        torch.set_default_dtype(prev)
    if eval_mode:
        model.eval()
    return model, cfg

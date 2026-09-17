"""The assembled v41f transformer: embedding -> Block stack -> norm -> fp32 head.

P1 assembly following Block (#434), matching vendored model_ref.Transformer
(third_party/deepseek_v41_ref/model_ref.py.ref :1187 init / :1242 forward). The residual
stream is hc_mult copies; one SharedAttnState is created per forward and threaded through
every Block (no process global). This stage is the v41f_small whole-network forward for
allclose: it can produce logits against the reference, but there is no training loop, no
checkpoint load/save, no engram hash, no DSpark/MTP draft.

The forward keeps the production v41f_s control-flow positions even though small leaves
the features off: the per-layer engram injection runs immediately BEFORE each owning Block
and BEFORE the target-layer hidden is read (ref Transformer.forward :1258-1265), the
DSSpark target hiddens are the attention INPUT (mean over hc copies) not the block output,
and the tail is hc_pre -> RMSNorm -> head. v41f_small sets engram_hash=None, every
self.engrams slot None and target_layer_ids=(), so each is skipped, but no small-only
simplified path exists to rework when the features switch on.

Engram modules are held on the model aligned by layer id (self.engrams), not on Block:
stage (A) freezes block.py. Stage (B) registers real Engram modules in these slots and
builds the tokenizer-dependent NgramHashState; the call sites below already match ref.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .attention import SharedAttnState
from .block import Block, make_identity_pre_mix
from .norm_gate import RMSNorm


class V41FHead(nn.Module):
    """Unsharded fp32 logits head (ref ParallelHead at world_size=1).

    The weight is bf16 in the checkpoint but kept fp32 (ref ParallelHead.__init__), and
    F.linear casts the hidden to fp32, so logits come out fp32 directly. Training scores
    every position, so this always returns the full [b, s, vocab] tensor (no last-position
    inference slice).
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.float32))
        # Training-side init: upstream allocates torch.empty because inference loads the head
        # from a checkpoint (convert.py) and specifies no init, so an unloaded head is an
        # uninitialized parameter. Left as empty it happens to read zero -> constant logits
        # -> the fp32 head severs every upstream gradient. Initialise the LM head at the
        # standard projection scale; the checkpoint loader overwrites it for inference.
        nn.init.normal_(self.weight, std=dim**-0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.float(), self.weight)


class V41FModel(nn.Module):
    """Embed + Block stack + norm/head, matching ref Transformer.

    forward(input_ids) -> (logits [b,s,vocab] fp32, main_hidden | None).
    Prefill only (start_pos=0). main_hidden is the concat of the DSpark target layers'
    attention-input means; None when dspark_target_layer_ids is empty (v41f_small).
    """

    def __init__(self, cfg, max_batch_size: int = 4):
        super().__init__()
        self.cfg = cfg
        self.hc_mult = cfg.hc_mult
        self.target_layer_ids = tuple(cfg.dspark_target_layer_ids)
        # n-gram hash state needs the rebuilt tokenizer; stage (B) sets it. None disables
        # the engram path while keeping the per-layer injection site in forward.
        self.engram_hash = None
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([Block(cfg, i, max_batch_size) for i in range(cfg.n_layers)])
        # aligned with self.layers as a ModuleList of None: stage (B) assigns a real Engram
        # module to a slot and it is registered immediately (parameters/.to/.train), no
        # structural change needed when the engram feature switches on.
        self.engrams = nn.ModuleList([None for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.head = V41FHead(cfg.vocab_size, cfg.dim)

    def forward(self, input_ids: torch.Tensor):
        engram_hashes = self.engram_hash(input_ids, 0, None) if self.engram_hash is not None else None
        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        main_hiddens = []
        pre_mix = make_identity_pre_mix(h, self.hc_mult)
        state = SharedAttnState()
        for i, layer in enumerate(self.layers):
            engram = self.engrams[i]
            if engram is not None:
                h = engram(h, engram_hashes[:, :, engram.layer_hash_index, :], None)
            # MTP reads the attention INPUT of its target layer, before the block runs
            if i in self.target_layer_ids:
                main_hiddens.append(h.mean(dim=2))
            h, pre_mix, state = layer(h, 0, pre_mix, state)
        h = self.layers[-1].hc_pre(h, pre_mix)
        logits = self.head(self.norm(h))
        main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
        return logits, main_hidden

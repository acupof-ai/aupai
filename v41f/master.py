"""Training state for v41f: fp32 master weights, bf16 forward, and name-keyed resume.

The inference-only `v41f/ckpt.py` cannot open a TRAINING blob and vice versa: the formats
are disjoint strings (M11). This file owns the training path designed in
`docs/standards/v41f_train_checkpoint_design.md`.

Three structural rules (do not weaken to name lists):
  * optimizer membership == `param.requires_grad` at build (D-PRE). Hard-topk indexer leaves
    are frozen in faithful mode and live under ste; permanently-dead index_key leaves are
    always frozen.
  * fp32-native params (head, HC tables, attn_sink, the source-layer compressor pair) ALIAS
    master storage: master[name] IS the model Parameter, on save AND on load, inside one blob.
    Only bf16-native params get a distinct fp32 master. The bf16 refresh is dispatched on the
    group and never touches an alias (M13 truncation).
  * optimizer state is keyed by NAME (#487 G2), never by AdamW's integer construction
    index: a param reorder between save and load must not bind m/v onto the wrong tensor.
"""

import os
import tempfile
from dataclasses import asdict

import torch

from .config import V41FConfig
from .model import V41FModel
from .vocab import fingerprint

TRAIN_FORMAT = "v41f_train_ckpt"
TRAIN_VERSION = 1

# AdamW constructor kwargs we persist. state_dict-only keys (decoupled_weight_decay,
# capturable, foreach, fused, ...) are deliberately excluded: passing them to AdamW() raises.
_HYPER = ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize")
# kwargs AdamW actually accepts at construction (subset of _HYPER we forward).
_CTOR_HYPER = ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize")


class TrainState:
    """Holds the fp32 master map and the optimizer over a name-sorted in-group parameter list.

    Construct AFTER the model (which already applied the D-PRE requires_grad freeze). Pass it
    to `v41f.train.train_step(..., state=state)`; production calls refresh/step explicitly if
    not using that helper.
    """

    def __init__(self, model: V41FModel, lr: float = 3e-4, **adamw):
        self.model = model
        named = dict(model.named_parameters())
        # canonical identity order: NAME-sorted, independent of module construction order.
        self.in_group_names = sorted(n for n, p in named.items() if p.requires_grad)
        self.fp32_native = {n for n in self.in_group_names if named[n].dtype == torch.float32}
        self.bf16_native = [n for n in self.in_group_names if n not in self.fp32_native]
        self.dormant = sorted(n for n, p in named.items() if not p.requires_grad)

        # master map: fp32-native shares the model Parameter; bf16 gets a distinct fp32 copy.
        self.master = {}
        with torch.no_grad():
            for n in self.in_group_names:
                p = named[n]
                if n in self.fp32_native:
                    self.master[n] = p  # alias, one storage (G7)
                else:
                    self.master[n] = torch.nn.Parameter(p.detach().float().clone())  # distinct fp32 master
        # build the live bf16 run weights from master once so they start equal.
        self.refresh_bf16()

        self.hyper = {k: adamw[k] for k in _CTOR_HYPER if k in adamw}
        self.hyper.setdefault("lr", lr)
        self.optimizer = torch.optim.AdamW([self.master[n] for n in self.in_group_names], **self.hyper)

    # -- training loop ------------------------------------------------------
    def refresh_bf16(self) -> None:
        """Copy fp32 master -> bf16 run weight for bf16-native params ONLY.

        fp32-native aliases are unreachable here: copying them would round an fp32 value
        through bf16 in place (1.0000305 -> 1.0) while dtype still reads float32 (M13).
        """
        named = dict(self.model.named_parameters())
        with torch.no_grad():
            for n in self.bf16_native:
                named[n].data.copy_(self.master[n].data)

    def collect_master_grads(self) -> None:
        """Cast each in-group model grad to fp32 into its master after backward."""
        named = dict(self.model.named_parameters())
        with torch.no_grad():
            for n in self.in_group_names:
                g = named[n].grad
                if g is not None:
                    self.master[n].grad = g.float()

    def zero_model_grads(self) -> None:
        self.model.zero_grad(set_to_none=True)


# --------------------------------------------------------------------------
# name-keyed optimizer state (G2)
# --------------------------------------------------------------------------


def _save_optim_named(state: TrainState) -> dict:
    opt = state.optimizer
    pname = {p: n for n in state.in_group_names for p in (state.master[n],)}
    state_by_name = {}
    for p, st in opt.state.items():
        if not st:
            continue  # never stepped -> absent (step-0 save)
        n = pname[p]
        m = st["exp_avg"]
        state_by_name[n] = {
            "step": st["step"].detach().cpu(),  # 0-d tensor in torch 2.x
            "exp_avg": st["exp_avg"].detach().cpu(),
            "exp_avg_sq": st["exp_avg_sq"].detach().cpu(),
            "shape": list(m.shape),
            "dtype": str(m.dtype),
        }
    return {
        "param_names": list(state.in_group_names),
        "state_by_name": state_by_name,
        "hyper": {k: opt.param_groups[0][k] for k in _HYPER if k in opt.param_groups[0]},
    }


class OptimStateError(AssertionError):
    pass


def _load_optim_named(state: TrainState, blob: dict) -> None:
    live = state.in_group_names
    saved = list(blob["param_names"])
    if set(live) != set(saved):
        only_live = sorted(set(live) - set(saved))
        only_saved = sorted(set(saved) - set(live))
        raise OptimStateError(
            f"optimizer param name set differs: only_live={only_live} only_saved={only_saved}"
        )
    if live != saved:  # canonical name order must agree too
        raise OptimStateError("optimizer param order differs after name canonicalization")
    by_name = state.master
    sbn = blob["state_by_name"]
    # state_by_name must be a SUBSET of param_names (extra keys have nowhere to bind). A
    # missing key is legal (that param never stepped), but an extra/renamed key is a G2
    # mismatch -- a position-keyed loader would have hidden it, so name it loudly.
    extra = sorted(set(sbn) - set(live))
    if extra:
        raise OptimStateError(f"optimizer state names not in param_names: {extra[:4]}")
    for n, rec in sbn.items():
        p = by_name[n]
        if tuple(p.shape) != tuple(rec["shape"]):
            raise OptimStateError(f"{n}: shape {tuple(p.shape)} != saved {tuple(rec['shape'])}")
        if str(p.dtype) != rec["dtype"]:
            raise OptimStateError(f"{n}: dtype {p.dtype} != saved {rec['dtype']}")
    # construct over the canonical name-sorted master list with constructor-accepted kwargs
    hyper = {k: v for k, v in blob["hyper"].items() if k in _CTOR_HYPER}
    params = [state.master[n] for n in live]
    opt = torch.optim.AdamW(params, **hyper)
    # Inject state WITHOUT a throwaway step: a dummy step would apply weight_decay/an AdamW
    # update to every master and move the weights once, after which overwriting only m/v/step
    # would leave the polluted weights in place (observed as resume != control). Build the
    # standard state_dict directly -- AdamW lazily adopts pre-existing state on load, keyed by
    # the canonical param index; params with no saved state (never stepped) get None.
    new_state = {}
    for i, n in enumerate(live):
        rec = sbn.get(n)
        if rec is None:
            continue
        new_state[i] = {
            "step": rec["step"].clone(),
            "exp_avg": rec["exp_avg"].clone().to(params[i]),
            "exp_avg_sq": rec["exp_avg_sq"].clone().to(params[i]),
        }
    sd = opt.state_dict()
    sd["state"] = new_state
    opt.load_state_dict(sd)
    # post-load identity: state reattached under the SAME name, never a reordered neighbour.
    for n, rec in sbn.items():
        stt = opt.state[state.master[n]]
        if not torch.equal(stt["exp_avg"].cpu(), rec["exp_avg"]):
            raise OptimStateError(f"{n}: exp_avg not restored to the same name")
        if not torch.equal(stt["exp_avg_sq"].cpu(), rec["exp_avg_sq"]):
            raise OptimStateError(f"{n}: exp_avg_sq not restored to the same name")
        if not torch.equal(stt["step"].cpu(), rec["step"]):
            raise OptimStateError(f"{n}: step not restored")
    state.optimizer = opt
    state.hyper = hyper


# --------------------------------------------------------------------------
# atomic IO
# --------------------------------------------------------------------------


def _atomic_replace(path: str, write) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=d)
    os.close(fd)
    try:
        write(tmp)
        with open(tmp, "rb") as fh:
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)  # failure must be loud on the os.replace path; tmp cleanup is best-effort


def save_train_checkpoint(path, *, model, cfg, state, tokenizer, step=0) -> None:
    """Write one atomic blob. fp32-native aliases are placed as the SAME object under both
    `model` and `master_fp32` so torch.save preserves shared storage (G7)."""
    vid = fingerprint(tokenizer)
    named = dict(model.named_parameters())
    sd = model.state_dict()  # full: params + persistent gate.bias buffers (G6)
    in_group = set(state.in_group_names)
    master_blob = {}
    # fp32-native: place the SAME live Parameter object under both `model` and `master_fp32`.
    # model.state_dict() materialised fresh copies in sd; overwrite that entry with the live
    # parameter so one object sits under both keys and torch.save stores ONE storage (G7). On
    # load the model group populates the model and master aliases that same model Parameter.
    for n in state.in_group_names:
        if n in state.fp32_native:
            sd[n] = named[n]
            master_blob[n] = named[n]
        else:
            master_blob[n] = state.master[n].detach()
    param_meta = {}
    for n, p in named.items():
        param_meta[n] = {
            "dtype": str(p.dtype).replace("torch.", ""),
            "group": "fp32_native" if n in state.fp32_native else ("bf16" if n in in_group else "dormant"),
            "grad": n in in_group,
        }
    blob = {
        "format": TRAIN_FORMAT,
        "version": TRAIN_VERSION,
        "config": asdict(cfg),
        "vocab_id": vid,
        "step": int(step),
        "param_meta": param_meta,
        "model": sd,
        "master_fp32": master_blob,
        "optim_named": _save_optim_named(state),
    }
    # save-time dtype assertions (#442 truncation precondition)
    for n, t in master_blob.items():
        if t.dtype != torch.float32:
            raise AssertionError(f"master {n} saved as {t.dtype}, expected float32")

    def _write(tmp):
        torch.save(blob, tmp)

    _atomic_replace(path, _write)


def load_train_checkpoint(path, *, tokenizer, max_batch_size: int = 4, lr: float | None = None):
    """Rebuild model+TrainState+cfg+step with an injected tokenizer (G3). Returns a bundle,
    never a bare model. fp32-native master is reconstructed as the SAME model Parameter."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("format") != TRAIN_FORMAT:
        raise ValueError(f"not a {TRAIN_FORMAT} blob: got {ckpt.get('format')!r}")
    if ckpt.get("version") != TRAIN_VERSION:
        raise ValueError(f"unsupported train ckpt version {ckpt.get('version')!r}")
    cfg = V41FConfig(**ckpt["config"])
    if cfg.engram_layer_ids and tokenizer is None:
        raise ValueError("an engram-on train checkpoint requires the injected tokenizer")
    vid = ckpt.get("vocab_id")
    if not vid:
        raise ValueError("train checkpoint carries no vocab_id")
    if vid != fingerprint(tokenizer):
        raise ValueError("train checkpoint vocab_id does not match the injected tokenizer")

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = V41FModel(cfg, max_batch_size=max_batch_size, tokenizer=tokenizer)
    finally:
        torch.set_default_dtype(prev)
    model.load_state_dict(ckpt["model"], strict=True)  # buffers incl gate.bias required (G6)

    state = TrainState.__new__(TrainState)
    state.model = model
    named = dict(model.named_parameters())
    saved_names = list(ckpt["optim_named"]["param_names"])
    # guard 1 (membership as a SET): saved in-group must equal the model's requires_grad set.
    live_rg = {n for n, p in named.items() if p.requires_grad}
    if live_rg != set(saved_names):
        raise OptimStateError(
            f"checkpoint in-group {len(saved_names)} != built requires_grad {len(live_rg)}: "
            f"only_live={sorted(live_rg - set(saved_names))[:4]} "
            f"only_saved={sorted(set(saved_names) - live_rg)[:4]}"
        )
    # guard 2 (canonical ORDER): the save format promises name-sorted order; the per-name
    # resolution in _load_optim_named also refuses a non-canonical list. Reordering cannot
    # misbind m/v (content is keyed by name), but an unsorted list is rejected rather than
    # silently accepted. Normalise here after the set check so master uses canonical order.
    if saved_names != sorted(saved_names):
        raise OptimStateError("optim_named.param_names is not in canonical name-sorted order")
    state.in_group_names = sorted(saved_names)
    state.fp32_native = {n for n in state.in_group_names if named[n].dtype == torch.float32}
    state.bf16_native = [n for n in state.in_group_names if n not in state.fp32_native]
    state.dormant = sorted(n for n, p in named.items() if not p.requires_grad)
    state.master = {}
    saved_master = ckpt["master_fp32"]
    if set(saved_master) != set(state.in_group_names):
        raise OptimStateError("master_fp32 name set != in-group set")
    with torch.no_grad():
        for n in state.in_group_names:
            t = saved_master[n]
            if t.dtype != torch.float32:
                raise AssertionError(f"master {n} loaded as {t.dtype}, expected float32")
            if n in state.fp32_native:
                if t.data_ptr() != named[n].data_ptr():
                    # torch.load preserves sharing only when same object graph; verify alias.
                    state.master[n] = named[n]
                else:
                    state.master[n] = named[n]
            else:
                p = torch.nn.Parameter(t.clone())
                state.master[n] = p
    # alias identity after load (G7): fp32-native master shares model storage.
    for n in state.fp32_native:
        if state.master[n].data_ptr() != named[n].data_ptr():
            raise OptimStateError(f"{n}: fp32-native master not aliased to model after load")
    state.hyper = {}
    state.optimizer = None
    _load_optim_named(state, ckpt["optim_named"])
    if lr is not None:  # caller override applied to the restored optimizer
        for g in state.optimizer.param_groups:
            g["lr"] = lr
    return model, state, cfg, int(ckpt["step"])

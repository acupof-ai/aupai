"""A closing memory account for one training step: every byte attributed, or named as residual.

Not a list of estimates -- a sum that has to match what the allocator reports. Activation bytes
are attributed by intercepting what autograd actually saves (`saved_tensors_hooks`) and charging
each saved tensor to the innermost module executing when it was packed. Tensors saved once and
used twice are deduped by storage pointer, so the column sums to the real total by construction.

The residual line is the point of the exercise: whatever the named rows do not explain is what
we have not understood yet.

Run: CUDA_VISIBLE_DEVICES=0 python scripts/mem_account.py --batch 16 --json facts/_raw/mem.json
"""

import argparse
import collections
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train  # noqa: E402

MB = 1e6


class Attributor:
    """Charges each autograd-saved tensor to the innermost module that was running."""

    def __init__(self, model):
        self.stack = ["<outside>"]
        self.seen = set()
        self.bytes = collections.Counter()
        self.count = collections.Counter()
        self.handles = []
        for _name, mod in model.named_modules():
            label = f"{type(mod).__name__}"
            self.handles.append(mod.register_forward_pre_hook(self._push(label)))
            self.handles.append(mod.register_forward_hook(self._pop()))

    def _push(self, label):
        def hook(_mod, _inp):
            self.stack.append(label)

        return hook

    def _pop(self):
        def hook(_mod, _inp, _out):
            self.stack.pop()

        return hook

    def pack(self, t):
        key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        if key not in self.seen and key[0] != 0:
            self.seen.add(key)
            self.bytes[self.stack[-1]] += key[1]
            self.count[self.stack[-1]] += 1
        return t

    def close(self):
        for h in self.handles:
            h.remove()

    def table(self):
        return [
            {"module": k, "MB": v / MB, "tensors": self.count[k]}
            for k, v in sorted(self.bytes.items(), key=lambda kv: -kv[1])
        ]


def param_account(model, opts):
    """Weights, grads, and whatever the optimizers are holding -- the batch-independent floor."""
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    grads = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
    opt = 0
    for o in opts:
        for st in o.state.values():
            opt += sum(v.numel() * v.element_size() for v in st.values() if torch.is_tensor(v))
    return {"params_MB": params / MB, "grads_MB": grads / MB, "optimizer_state_MB": opt / MB}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument(
        "--steps", type=int, default=2, help="steps before the accounted one (fills optimizer state)"
    )
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda")
    cfg = train.Cfg
    cfg.batch, cfg.compile = args.batch, False
    if args.seq:
        cfg.seq = args.seq

    model = train.HybridLM(cfg).to(dev).to(torch.bfloat16)
    opts = train.build_optimizers(model, cfg)
    idx = torch.randint(0, cfg.vocab, (cfg.batch, cfg.seq + 1), device=dev)
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()

    run = torch.compile(model) if args.compile else model
    for _ in range(args.steps):  # fill optimizer state so its rows are real, not zero
        run(x, y)[0].sum().backward()
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()

    att = Attributor(model)
    with torch.autograd.graph.saved_tensors_hooks(att.pack, lambda t: t):
        loss = run(x, y)[0]
    att.close()
    peak_fwd = torch.cuda.max_memory_allocated()
    loss.sum().backward()
    peak = torch.cuda.max_memory_allocated()

    saved = att.table()
    saved_total = sum(r["MB"] for r in saved)
    params = param_account(model, opts)
    reserved = torch.cuda.memory_reserved()

    out = {
        "config": {
            "batch": cfg.batch,
            "seq": cfg.seq,
            "d": cfg.d,
            "layers": cfg.layers,
            "attn_every": cfg.attn_every,
            "vocab": cfg.vocab,
            "ffn_hidden": cfg.ffn_hidden,
            "attn_res": cfg.attn_res,
            "attn_res_blocks": cfg.attn_res_blocks,
            "grad_ckpt": cfg.grad_ckpt,
            "compiled": args.compile,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "dtype": "bfloat16",
            "ddp": False,
            "note": "single process; DDP buckets not included",
        },
        "saved_activations_by_module": saved,
        "saved_activations_total_MB": saved_total,
        "parameters": params,
        "allocator": {
            "baseline_before_step_MB": before / MB,
            "peak_after_forward_MB": peak_fwd / MB,
            "peak_after_backward_MB": peak / MB,
            "reserved_MB": reserved / MB,
            "fragmentation_MB": (reserved - torch.cuda.memory_allocated()) / MB,
        },
    }
    # The closing check: what the named rows do not explain.
    named = saved_total + params["params_MB"] + params["grads_MB"] + params["optimizer_state_MB"]
    out["closure"] = {
        "named_MB": named,
        "peak_MB": peak / MB,
        "residual_MB": peak / MB - named,
        "residual_frac": (peak / MB - named) / (peak / MB),
    }
    print(json.dumps(out, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

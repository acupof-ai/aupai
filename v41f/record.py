"""--v42_record: per-module activation / grad-output / parameter-grad statistics for one micro-batch.

Recorder.attach() puts a forward hook and a full backward hook on every LEAF module of the model
(Linear, RMSNorm, Gate, Embedding, the head, ...; a module with children is covered by its leaves),
and wraps HyperConn.hc_pre / hc_post and MoE._routed_grouped, which are method calls rather than
module forwards, so the mHC collapse and expansion and the routed-expert sum (the grouped_mm w2
result scattered back to tokens, before the shared expert is added) are recorded too. The stacked
MoE expert weights (w1/w3/w2 on the MoE module, no Linear) appear in the parameter-grad table only.
GroupedOProj.wo_b is a leaf Linear and is hooked like any other (key layers.N.attn.oproj.wo_b). detach() removes
everything, so a step without recording runs the untouched (compiled) model.

row(step, ...) is one JSON-able dict:
  {"step", "rank", "loss", "healthy", "mem_alloc_gib",
   "act":  {fqn: {"rms", "absmax", "finite"}},          # module output, last micro-batch
   "gout": {fqn: {"rms", "absmax", "finite"}},          # grad w.r.t. that output
   "grad": {fqn: {"rms", "absmax", "nan", "inf"}},      # parameter grads after backward
   "summary": {"act_max": [fqn, v], "gout_max": [fqn, v], "grad_max": [fqn, v]},
              # act/gout: the FIRST non-finite module in hook order (where it entered), else the max
   "nonfinite_params": [[fqn, nan, inf], ...]}          # empty on a healthy step
Statistics are fp32 on the tensors' device; a tuple output records every tensor in it. Values are
Python floats (inf/nan preserved; json.dumps writes Infinity/NaN, json.loads reads them back).
"""

import json
import math

import torch

from .hyperconn import HyperConn
from .moe import MoE

# method calls to wrap per module type: (method, recorded key suffix)
_METHODS = {HyperConn: (("hc_pre", "hc_pre"), ("hc_post", "hc_post")), MoE: (("_routed_grouped", "routed"),)}


def _stats(tensors):
    tot_sq, n, amax, finite = 0.0, 0, 0.0, True
    for t in tensors:
        if not isinstance(t, torch.Tensor) or not t.is_floating_point() or t.numel() == 0:
            continue
        f = t.detach().float()
        tot_sq += float(f.square().sum())
        n += f.numel()
        m = float(f.abs().max())
        amax = max(amax, m) if m == m else float("nan")
        finite = finite and bool(torch.isfinite(f).all())
    if n == 0:
        return None
    return {"rms": math.sqrt(tot_sq / n) if tot_sq == tot_sq else float("nan"), "absmax": amax, "finite": finite}


def _flatten(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [t for t in out if isinstance(t, torch.Tensor)]
    return []


class Recorder:
    def __init__(self, model, rank=0):
        self.model, self.rank = model, rank
        self.act, self.gout = {}, {}
        self._handles, self._patched = [], []

    def attach(self):
        self.act.clear()
        self.gout.clear()
        for name, mod in self.model.named_modules():
            for cls, meths in _METHODS.items():
                if isinstance(mod, cls):
                    self._patch_methods(name, mod, meths)
            if any(True for _ in mod.children()):
                continue
            if isinstance(mod, HyperConn):
                continue
            self._handles.append(mod.register_forward_hook(self._fwd_hook(name)))
            self._handles.append(mod.register_full_backward_hook(self._bwd_hook(name)))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for mod, attr in self._patched:
            delattr(mod, attr)  # back to the class method
        self._patched.clear()

    @staticmethod
    def _put(table, key, s):
        """Re-insert at the end: a module called twice per block (hc_pre for both sublayers, a shared
        norm) must sit at its LAST call's position, or an inf planted later reads as arriving earlier."""
        table.pop(key, None)
        table[key] = s

    def _fwd_hook(self, name):
        def hook(mod, inp, out):
            s = _stats(_flatten(out))
            if s is not None:
                self._put(self.act, name, s)
        return hook

    def _bwd_hook(self, name):
        def hook(mod, gin, gout):
            s = _stats(_flatten(gout))
            if s is not None:
                self._put(self.gout, name, s)
        return hook

    def _patch_methods(self, name, mod, meths):
        for meth, suffix in meths:
            orig = getattr(type(mod), meth)
            key = f"{name}.{suffix}"

            def wrapped(*a, _orig=orig, _key=key, _hc=mod, **kw):
                out = _orig(_hc, *a, **kw)
                s = _stats([out])
                if s is not None:
                    self._put(self.act, _key, s)
                if isinstance(out, torch.Tensor) and out.requires_grad:
                    out.register_hook(lambda g, _k=_key: self._put(self.gout, _k, _stats([g])))
                return out
            setattr(mod, meth, wrapped)
            self._patched.append((mod, meth))

    def grads(self):
        out = {}
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            fin = torch.isfinite(g)
            nan = int(torch.isnan(g).sum())
            out[n] = {"rms": float(g.square().mean().sqrt()), "absmax": float(g.abs().max()),
                      "nan": nan, "inf": int((~fin).sum()) - nan}
        return out

    @staticmethod
    def _argmax(table, causal=False):
        """[fqn, absmax] of the largest entry. With causal=True (act/gout tables, which are in hook
        call order) the FIRST non-finite entry wins: it is where the inf/nan entered, the later ones
        are its consequences. Without it (param grads, no order) any non-finite entry wins."""
        best = None
        for k, v in table.items():
            a = v["absmax"]
            bad = a != a or a in (float("inf"), float("-inf"))
            if bad:
                return [k, a] if causal or a != a else ([k, a] if best is None or best[1] == best[1] else best)
            if best is None or a > best[1]:
                best = [k, a]
        return best or ["(none)", 0.0]

    def row(self, step, loss=None, healthy=True, mem_alloc_gib=None):
        grad = self.grads()
        summary = {"act_max": self._argmax(self.act, causal=True), "gout_max": self._argmax(self.gout, causal=True),
                   "grad_max": self._argmax(grad)}
        nonfinite = [[n, v["nan"], v["inf"]] for n, v in grad.items() if v["nan"] or v["inf"]]
        return {"step": step, "rank": self.rank, "loss": loss, "healthy": healthy, "mem_alloc_gib": mem_alloc_gib,
                "act": self.act, "gout": self.gout, "grad": grad, "summary": summary, "nonfinite_params": nonfinite}

    def any_nonfinite(self, row):
        return (bool(row["nonfinite_params"]) or any(not v["finite"] for v in row["act"].values())
                or any(not v["finite"] for v in row["gout"].values()))

    @staticmethod
    def summary_line(row):
        s = row["summary"]
        return (f"step {row['step']} record: act_max {s['act_max'][0]}={s['act_max'][1]:.4g} "
                f"gout_max {s['gout_max'][0]}={s['gout_max'][1]:.4g} grad_max {s['grad_max'][0]}={s['grad_max'][1]:.4g}"
                + (f" nonfinite_params={len(row['nonfinite_params'])}" if row["nonfinite_params"] else ""))

    @staticmethod
    def write(path, row):
        with open(path, "a") as fh:
            fh.write(json.dumps(row) + "\n")

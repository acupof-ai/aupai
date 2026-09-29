"""What one v42 forward keeps for backward, per layer and per sublayer, measured on any device.

Runs the v42_s24 trunk (attn_impl fused, rope real, MoE stacked+grouped path) forward at --B x --T
under torch.autograd.graph.saved_tensors_hooks and attributes every saved tensor (deduplicated by
storage, parameters excluded) to the module that saved it. Prints per layer: attention / hc+norm /
moe bytes, the total, and the whole-trunk total plus the 4-stream residual the next layer holds.
Off CUDA the window branch is the torch fallback and the numbers are what the same graph saves there;
on CUDA the flash window branch saves q, kv, o, lse instead.

  CUDA_VISIBLE_DEVICES= python3 scripts/v42_mem_account.py --B 1 --T 4096          # CPU, 14 layers
  CUDA_VISIBLE_DEVICES=0 python3 scripts/v42_mem_account.py --B 4 --T 4096 --layers 24 --grad_ckpt
--grad_ckpt sets block_ckpt, so the print shows what each block keeps when it is recomputed.
--layers 14 (the CPU default; the laptop memguard kills a python over 6 GB and the 24-layer trunk is
6.5 GB bf16) keeps every layer kind of v42_s24: window-only 0-1, m=2 source 2/8, m=2 reuse 3-7/9-11,
m=1 decoder source 12, m=1 reuse 13; a 24-layer total is layers 12/13 repeated with sources at 16/20.
--experts 16 (the CPU default) shrinks the expert tables only: what MoE saves for backward is the
activated rows x inter (top-k, independent of E) plus router logits [tokens, E], so 16 vs 64 moves the
moe column by tokens x 48 x 4 bytes.
"""

# restartable: prints one row per cell/layer as it goes and holds no state; an interrupt costs one rerun
import argparse
import collections
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v41f.config import v42_s24  # noqa: E402
from v41f.lm import V42LM  # noqa: E402
from v41f.moe import MoE  # noqa: E402

MIB = 2**20


def category(name):
    if ".attn." in name or name.endswith(".attn"):
        return "attention"
    if ".ffn." in name or name.endswith(".ffn"):
        return "moe"
    if ".hc" in name or "_norm" in name:
        return "hc+norm"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--moe_gemm", default="grouped_mm", choices=["grouped_mm", "deepgemm"])
    ap.add_argument("--layers", type=int, default=None, help="truncate the trunk (default 14 on CPU, 24 on CUDA)")
    ap.add_argument("--experts", type=int, default=None, help="routed experts (default 16 on CPU, 64 on CUDA)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    L = a.layers or (24 if dev == "cuda" else 14)
    over = {} if L == 24 else dict(
        n_layers=L, compress_ratios=((0, 0) + (2,) * 10 + (1,) * 12)[:L],
        kv_source_layers=tuple(s for s in (2, 8, 12) if s < L),
        index_source_layers=tuple(s for s in (2, 8, 12, 16, 20) if s < L))
    over["n_routed_experts"] = a.experts or (64 if dev == "cuda" else 16)
    cfg = v42_s24(vocab_size=a.vocab, attn_impl="fused", rope_impl="real", moe_stacked=True,
                  block_ckpt=a.grad_ckpt, moe_gemm=a.moe_gemm, **over)
    cfg.validate()
    MoE.grouped_on_cpu = True
    torch.manual_seed(0)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(dev):
            m = V42LM(cfg, max_batch_size=a.B).train()
    finally:
        torch.set_default_dtype(prev)
    params = {p.untyped_storage().data_ptr() for p in m.parameters()}
    stack = ["root"]
    for n, mod in m.named_modules():
        mod.register_forward_pre_hook(lambda mod, inp, n=n: stack.append(n))
        mod.register_forward_hook(lambda mod, inp, out: (stack.pop(), None)[1])
    seen = set()
    per = collections.defaultdict(lambda: collections.Counter())

    def pack(t):
        st = t.untyped_storage()
        key = (st.data_ptr(), st.nbytes())
        if st.data_ptr() in params or key in seen:
            return t
        seen.add(key)
        name = stack[-1]
        layer = name.split("layers.")[1].split(".")[0] if "layers." in name else "trunk"
        per[layer][category(name)] += st.nbytes()
        return t

    ids = torch.randint(0, a.vocab, (a.B, a.T), device=dev)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        hidden, _ = m(ids, cu=None)
    stream = a.B * a.T * cfg.hc_mult * cfg.dim * 2
    print(f"v42_mem_account: B{a.B} T{a.T} dev={dev} layers={L} experts={over['n_routed_experts']} moe_gemm={a.moe_gemm} "
          f"block_ckpt={a.grad_ckpt}; 4-stream residual "
          f"[b,s,hc,d] bf16 = {stream / MIB:.0f} MiB per layer boundary")
    print(f"{'layer':>6} {'attention':>10} {'hc+norm':>10} {'moe':>10} {'other':>8} {'total MiB':>10}")
    tot = collections.Counter()
    for layer in sorted(per, key=lambda k: (k == "trunk", int(k) if k != "trunk" else 0)):
        c = per[layer]
        row = [c[k] / MIB for k in ("attention", "hc+norm", "moe", "other")]
        print(f"{layer:>6} {row[0]:10.0f} {row[1]:10.0f} {row[2]:10.0f} {row[3]:8.0f} {sum(row):10.0f}")
        tot.update(c)
    total = sum(tot.values())
    print(f"{'all':>6} {tot['attention'] / MIB:10.0f} {tot['hc+norm'] / MIB:10.0f} {tot['moe'] / MIB:10.0f} "
          f"{tot['other'] / MIB:8.0f} {total / MIB:10.0f}   ({total / 2**30:.2f} GiB saved for backward)")
    if dev == "cuda":
        print(f"torch.cuda.max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()

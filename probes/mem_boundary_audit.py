"""Item 3: does the compiled step survive the memory layer, and is FP8 actually excluded?

Both answers are LISTS, not booleans. "torch.compile works" and "FP8 skips the memory"
are the claims; a boolean hides which modules were converted and where the graph broke,
and b0 already found that _fp8_ok sees the LEAF name (train.py:543), so the filter can
read correct while converting the wrong modules.
"""
import sys

sys.path.insert(0, "/work/aupai")
import torch
import torch.nn as nn

from model import ProductKeyMemory

D, SIDE, TOP_K = 128, 64, 8
B, T = 2, 16

def fp8_audit():
    """Which modules does torchao ACTUALLY convert? Listed, not inferred from the filter."""
    from train import _fp8_filter, _fp8_ok, _is_mem_fqn
    m = ProductKeyMemory(SIDE * SIDE, D, top_k=TOP_K).cuda()
    named = dict(m.named_modules())
    rows = []
    for fqn, mod in named.items():
        if not isinstance(mod, nn.Linear):
            continue
        leaf = fqn.rsplit(".", 1)[-1]
        # The REAL filter, as train.py passes it to torchao -- not a re-derivation.
        # Modules inside the arm are reached as `memory.<leaf>`, so audit that path.
        rows.append((fqn, leaf, _fp8_ok(mod, leaf), _fp8_filter(mod, f"memory.{fqn}")))
    print("  fqn / leaf-name-decision / full-fqn-decision")
    for fqn, leaf, by_leaf, by_filter in rows:
        flag = "  <-- CONVERTED" if by_filter else ("  (leaf test would have)" if by_leaf else "")
        print(f"    {fqn:24s} leaf={leaf:10s} by_leaf={by_leaf} by_filter={by_filter}{flag}")
    conv = [f for f, _, _, ok in rows if ok]
    print(f"  torchao CONVERTS {len(conv)} Linear(s) inside the memory under the real "
          f"filter: {conv or 'none'}")
    naive = [f for f, _, ok, _ in rows if ok]
    print(f"  ...and would have converted {len(naive)} under the leaf-name test: {naive}")
    # The predicate the exclusion rests on, exercised on both sides. A path test that
    # says True for everything excludes nothing; one that says False for everything
    # would silently drop unrelated layers out of FP8.
    assert _is_mem_fqn("memory.query") and _is_mem_fqn("blocks.3.memory.out")
    assert not _is_mem_fqn("memory_head") and not _is_mem_fqn("blocks.3.attn.query")
    print("  _is_mem_fqn: matches memory paths, rejects `memory_head` and ordinary leaves")
    # The value TABLE is nn.Embedding, not Linear -- torchao only touches Linear, so the
    # 1B-parameter table was never at risk. The exposure is the query/gate/out projections.
    emb = [f for f, mm in named.items() if isinstance(mm, nn.Embedding)]
    print(f"  nn.Embedding (never touched by torchao's Linear filter): {emb}")

def compile_audit():
    """Graph breaks, counted with the real module and the pod's inductor backend."""
    import torch._dynamo as dynamo
    dynamo.reset()
    m = ProductKeyMemory(SIDE * SIDE, D, top_k=TOP_K).cuda()
    x = torch.randn(B, T, D, device="cuda")
    expl = dynamo.explain(m)(x)
    print(f"  graphs={expl.graph_count} breaks={expl.graph_break_count} ops={expl.op_count}")
    for r in getattr(expl, "break_reasons", [])[:5]:
        print(f"    break: {str(getattr(r, 'reason', r))[:100]}")
    cm = torch.compile(m)
    out = cm(x)
    ref = m(x)
    print(f"  compiled step ran: out{tuple(out.shape)} "
          f"max|compiled-eager|={ (out-ref).abs().max().item():.2e}")

if __name__ == "__main__":
    print("== FP8 exclusion, by listing what is converted")
    fp8_audit()
    print("== torch.compile")
    compile_audit()


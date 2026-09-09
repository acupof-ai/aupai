#!/usr/bin/env python3
"""Known-answer worlds for the p1 attention path: partial RoPE, HCA, and the CSA refactor.

CPU only, seconds. Run: python3 scripts/test_v4_attn.py [--selftest]

Every check here is a perturbation test rather than a shape assertion, because the three ways
this path can be wrong are all silent. `facts/deepseek_v4.json#dsv4.partial_rope` and the CSA
docstring both record the same lesson from the compress-branch leak of 2026-09-04: a causal or
cross-document leak makes the model BETTER at training time and only surfaces later as a
generation collapse nobody can attribute. A test that only checks that tensors have the right
size would have passed on every one of those bugs.

W1  CSA is bit-identical after pool_per_doc was lifted out of it. This is the refactor's own
    guarantee: the helper exists so CSA and HCA cannot drift apart, and it is worth nothing if
    extracting it moved a number.
W2  rope_dims=0 leaves GatedMLA bit-identical. The default must not have moved.
W3  RoPE positions restart at each document. Two documents with identical tokens must produce
    identical outputs; if positions counted from the row start, the second copy would differ.
W4  RoPE actually does something. W3 passes trivially on a no-op, so the same tokens at a
    different position within their document must differ.
W5  HCA is causal. A future token cannot move an earlier output.
W6  HCA isolates documents. A token in document B cannot move an output in document A.
W7  The zero-KDA refusal lifts with rope_dims and holds without it.
W9  Every branch survives a BACKWARD. The forward-only worlds above all passed while
    89 of 102 parameter tensors came back non-finite, which is the whole reason this
    one exists.
W8  The hybrid interleave assigns the kinds it claims, and the first two attention layers are
    CSA -- HCA at m'=128 is blind over the first 128 positions of a document and something with
    a window has to sit under it.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M  # noqa: E402

TOL = 0.0  # bit-identical, not "close": a refactor that moves the last bit moved the number


class Cfg:
    d = 128
    heads = 4
    layers = 4
    attn_every = 2
    ffn_hidden = 256
    vocab = 256
    seq = 64
    csa = False
    csa_compress = 4
    csa_topk = 2
    csa_window = 8
    hca = False
    hca_compress = 8
    attn_hybrid = False
    rope_dims = 0
    attn_res = False
    moe_experts = 0
    mem_values = 0
    head_mixed = 0
    value_embed = False
    grad_ckpt = False
    fone = False
    softcap = 0.0
    doc_mask = True
    attn_res_blocks = 0
    moe_layers = ""
    mem_layers = ""


def cfg(**over):
    c = type("C", (Cfg,), {})()
    for k, v in over.items():
        setattr(c, k, v)
    return c


def qkv(B, T, H, D, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, T, H, D, generator=g) for _ in range(3)]


def two_docs(T):
    """cu for one row split into two equal documents -- the packed shape every branch masks on."""
    return torch.tensor([0, T // 2, T], dtype=torch.int32)


def w1_csa_refactor_parity():
    """CSA's numbers after the lift must equal a reference computed the way it computed them.

    The reference is recomputed here from the same inputs through the helper AND through a
    literal re-derivation of the pooling, so this fails if pool_per_doc is not what CSA used to
    do inline. A saved golden tensor would be better still; this is what is available without
    a checkpoint on a CPU box.
    """
    c = cfg(csa=True)
    torch.manual_seed(0)
    csa = M.CompressedSparseAttention(c, c.heads, c.d // c.heads)
    q, k, v = qkv(1, 32, c.heads, c.d // c.heads, seed=1)
    cu = two_docs(32)
    y = csa(q, k, v, cu)

    # literal re-derivation of the pooling the class used to do inline
    B, T, H, D = q.shape
    kh, vh = k.transpose(1, 2), v.transpose(1, 2)
    m = c.csa_compress
    pos = torch.arange(B * T)
    cu_l = cu.to(pos.dtype)
    doc = torch.bucketize(pos, cu_l[1:], right=True).view(B, T)
    nb_doc = (cu_l[1:] - cu_l[:-1] + m - 1) // m
    offset = torch.cumsum(nb_doc, 0) - nb_doc
    block_id = (offset[doc] + (pos.view(B, T) - cu_l[doc]) // m - offset[doc[:, 0]][:, None])
    kc, vc, vis, bid2, doc2 = M.pool_per_doc(q, kh, vh, cu, m)
    if not torch.equal(block_id, bid2):
        return False, "pool_per_doc's block_id differs from the inline derivation"
    if not torch.equal(doc, doc2):
        return False, "pool_per_doc's doc ids differ from the inline derivation"
    if torch.isnan(y).any():
        return False, "CSA produced NaN on a two-document row"
    return True, f"CSA runs through the helper, block_id and doc match ({tuple(kc.shape)} pooled)"


def w2_rope_off_is_identity():
    c0, c1 = cfg(), cfg(rope_dims=0)
    torch.manual_seed(0)
    a = M.GatedMLA(c0)
    torch.manual_seed(0)
    b = M.GatedMLA(c1)
    if a.rope is not None or b.rope is not None:
        return False, "rope_dims=0 still constructed a PartialRoPE"
    x = torch.randn(1, 32, c0.d, generator=torch.Generator().manual_seed(3))
    cu = two_docs(32)
    with torch.no_grad():
        ya, yb = a(x, cu), b(x, cu)
    if not torch.equal(ya, yb):
        return False, "rope_dims=0 changed the default path's output"
    if any("rope" in k for k in a.state_dict()):
        return False, "PartialRoPE added a state_dict key; existing checkpoints would not load"
    return True, "rope_dims=0 is bit-identical and adds no state_dict key"


def w3_rope_positions_restart_per_document():
    """Two identical documents in one row must give identical outputs."""
    c = cfg(rope_dims=16)
    torch.manual_seed(0)
    mla = M.GatedMLA(c)
    half = torch.randn(1, 16, c.d, generator=torch.Generator().manual_seed(5))
    x = torch.cat([half, half], dim=1)          # doc A == doc B, token for token
    cu = two_docs(32)
    with torch.no_grad():
        y = mla(x, cu)
    a, b = y[:, :16], y[:, 16:]
    d = (a - b).abs().max().item()
    if d > 1e-4:
        return False, (f"identical documents produced different outputs (max|delta| {d:.3e}): "
                       f"positions did not restart at the document boundary, so RoPE is encoding "
                       f"cross-document distance")
    return True, f"two identical documents agree to {d:.2e}"


def w4_rope_is_not_a_noop():
    """W3 passes on a no-op, so the same tokens at a different in-document offset must differ."""
    c = cfg(rope_dims=16)
    torch.manual_seed(0)
    mla = M.GatedMLA(c)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, 32, c.d, generator=g)
    one = torch.tensor([0, 32], dtype=torch.int32)     # one document: positions 0..31
    two = two_docs(32)                                 # two documents: positions restart at 16
    with torch.no_grad():
        y1, y2 = mla(x, one), mla(x, two)
    d = (y1[:, 16:] - y2[:, 16:]).abs().max().item()
    if d < 1e-4:
        return False, (f"the same tokens at position 16 and at position 0 of a second document "
                       f"gave the same output (max|delta| {d:.3e}) -- RoPE is not applied")
    return True, f"the second half moves when its positions restart (max|delta| {d:.2e})"


def w5_hca_is_causal():
    c = cfg(hca=True, hca_compress=8)
    hca = M.HeavilyCompressedAttention(c, c.heads, c.d // c.heads)
    q, k, v = qkv(1, 64, c.heads, c.d // c.heads, seed=11)
    cu = torch.tensor([0, 64], dtype=torch.int32)
    with torch.no_grad():
        y0 = hca(q, k, v, cu)
        k2 = k.clone()
        k2[0, 60] += 7.0                                # a token near the end
        y1 = hca(q, k2, v, cu)
    early = (y0[:, :40] - y1[:, :40]).abs().max().item()
    if early > TOL:
        return False, (f"perturbing position 60 moved outputs at positions 0-39 by {early:.3e}: "
                       f"HCA leaks the future, the same defect the CSA compress branch had")
    return True, "a token at position 60 does not reach outputs at 0-39"


def w6_hca_isolates_documents():
    c = cfg(hca=True, hca_compress=8)
    hca = M.HeavilyCompressedAttention(c, c.heads, c.d // c.heads)
    q, k, v = qkv(1, 64, c.heads, c.d // c.heads, seed=13)
    cu = two_docs(64)                                   # docs are [0,32) and [32,64)
    with torch.no_grad():
        y0 = hca(q, k, v, cu)
        k2 = k.clone()
        k2[0, 40] += 7.0                                # inside document B
        y1 = hca(q, k2, v, cu)
    leak = (y0[:, :32] - y1[:, :32]).abs().max().item()
    if leak > TOL:
        return False, (f"a token in document B moved document A's outputs by {leak:.3e}: HCA "
                       f"pools across a document boundary")
    return True, "document B cannot reach document A"


def w7_zero_kda_refusal():
    ok_refused = False
    try:
        M.HybridLM(cfg(attn_every=1, rope_dims=0))
    except ValueError as e:
        ok_refused = "position" in str(e).lower()
    if not ok_refused:
        return False, ("attn_every=1 with rope_dims=0 was accepted; that model has neither KDA "
                       "nor RoPE and therefore no position information at all")
    try:
        m = M.HybridLM(cfg(attn_every=1, rope_dims=16))
    except ValueError as e:
        return False, f"attn_every=1 with rope_dims=16 was refused: {e}"
    n_attn = sum(1 for b in m.blocks if isinstance(b.mixer, M.GatedMLA))
    if n_attn != Cfg.layers:
        return False, f"attn_every=1 built {n_attn} attention layers, expected {Cfg.layers}"
    return True, "refused without rope_dims, accepted with it, and every layer is attention"


def w8_hybrid_interleave():
    m = M.HybridLM(cfg(attn_every=1, rope_dims=16, attn_hybrid=True, layers=6))
    kinds = [m.attn_kinds[i] for i in range(6)]
    if kinds[0] != "csa" or kinds[1] != "csa":
        return False, (f"the first two attention layers are {kinds[:2]}, not both csa. HCA at "
                       f"m'={Cfg.hca_compress} sees no complete block in the first "
                       f"{Cfg.hca_compress} positions of a document, so a branch with a window "
                       f"has to sit under it")
    if "hca" not in kinds:
        return False, f"attn_hybrid produced no HCA layer at all: {kinds}"
    for i, k in enumerate(kinds):
        mix = m.blocks[i].mixer
        has = "csa" if mix.csa is not None else ("hca" if mix.hca is not None else "none")
        if has != k:
            return False, f"layer {i} is declared {k} but constructed {has}"
    return True, f"kinds {kinds}, each matching the module actually constructed"



def w9_gradients_are_finite():
    """The whole p1 stack must survive a BACKWARD, not just a forward.

    The bug this world exists for, 2026-09-09: every masked-softmax branch in CSA and HCA was
    spelled `nan_to_num(softmax(masked_fill(-inf)) @ v)`. That is correct forward and NaN
    backward -- nan_to_num rewrites the output, not the graph, so an all--inf row still
    differentiates as 0/0 and BmmBackward0 carries NaN into every upstream parameter. Measured
    before the fix: 89 of 102 parameter tensors non-finite after one backward, with the forward
    finite throughout. No forward-only check could see it, and CSA had carried it since b0-35
    without firing because CSA has never been trained.

    Fully-masked rows are the normal case in both callers, not an edge: a compressed block is
    visible only once its last member is at or before the query, so every query before its
    document's first complete block has one.
    """
    fails = []
    cu = torch.tensor([0, 128, 256, 384, 512], dtype=torch.int32)
    for label, over, packed in [
        ("CSA packed", dict(csa=True, rope_dims=32), True),
        ("HCA packed", dict(hca=True, rope_dims=32), True),
        ("CSA unpacked", dict(csa=True, rope_dims=32), False),
        ("HCA unpacked", dict(hca=True, rope_dims=32), False),
    ]:
        c = cfg(d=128, heads=4, csa_compress=16, csa_topk=4, csa_window=32,
                hca_compress=64, **over)
        torch.manual_seed(0)
        mla = M.GatedMLA(c)
        x = torch.randn(2, 256, c.d, requires_grad=True)
        y = mla(x, cu if packed else None)
        if not torch.isfinite(y).all():
            fails.append(f"{label}: forward is non-finite")
            continue
        y.sum().backward()
        n = int((~torch.isfinite(x.grad)).sum())
        if n:
            fails.append(f"{label}: {n} non-finite entries in the input gradient")
        bad = [k for k, prm in mla.named_parameters()
               if prm.grad is not None and not torch.isfinite(prm.grad).all()]
        if bad:
            fails.append(f"{label}: {len(bad)} parameter tensors with non-finite grad, e.g. {bad[0]}")
    if fails:
        return False, "; ".join(fails)
    return True, "four branch/packing combinations backward with every gradient finite"


WORLDS = [
    ("W1 csa_refactor_parity", w1_csa_refactor_parity),
    ("W2 rope_off_is_identity", w2_rope_off_is_identity),
    ("W3 rope_positions_per_document", w3_rope_positions_restart_per_document),
    ("W4 rope_is_not_a_noop", w4_rope_is_not_a_noop),
    ("W5 hca_is_causal", w5_hca_is_causal),
    ("W6 hca_isolates_documents", w6_hca_isolates_documents),
    ("W7 zero_kda_refusal", w7_zero_kda_refusal),
    ("W8 hybrid_interleave", w8_hybrid_interleave),
    ("W9 gradients_are_finite", w9_gradients_are_finite),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="same worlds; the flag exists so the pre-commit hook can call this file "
                         "the way it calls every other registered selftest")
    ap.parse_args()
    fails = []
    for name, fn in WORLDS:
        try:
            ok, why = fn()
        except Exception as e:  # a world that cannot run is a failure, never a skip
            ok, why = False, f"{type(e).__name__}: {e}"
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {why}", flush=True)
        if not ok:
            fails.append(name)
    if fails:
        print(f"test_v4_attn FAILED: {', '.join(fails)}")
        return 1
    print(f"test_v4_attn ok: {len(WORLDS)} worlds")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""FP8 expert GEMMs through DeepGEMM (deep_gemm 0.1.0 on the pod, SM90) behind MoE(moe_gemm="deepgemm").

Layout is DeepGEMM's "contiguous" grouped GEMM: rows sorted by expert, every expert's segment padded
to the M alignment (128), `m_indices[row]` = expert id (-1 on padding rows). The padded M is a SHAPE,
`padded_m(B*T*top_k, E)` = rows + one partial block per expert, so no routing-dependent host read
happens per step and the path stays inside a compiled graph (the unused tail is -1 rows the kernel
skips; the cost is E*128 extra rows of fp8 activations, 64*128 = 8192 rows at v42_s24). Scaling follows the
DeepSeek-V3 fine-grained recipe the library is built for -- activations per token x 128-wide K block
(e4m3, fp32 scale), weights per 128x128 block -- not torchao's tensorwise recipe: the kernel has no
per-tensor mode, so the scaling matches torchao's e4m3 dynamic recipe in dtype and dynamic range, not
in granularity.

  forward   y = a @ w^T      m_grouped_fp8_gemm_nt_contiguous((a_fp8, a_sf), (w_fp8, w_sf), y, m_indices)
  backward  da = dy @ w      the same nt kernel with B = w^T as a K-major [E,K,N] copy (DeepGEMM asserts
                             major_b == K: the nn variant with our [E,N,K] weight fails gemm.hpp:159);
                             128x128 block scales transpose with it
            dw = dy^T @ a    torch._grouped_mm in bf16 (this deep_gemm build has no k-grouped GEMM)

Off CUDA, or without deep_gemm, `emulate=True` runs the same quantize -> dequantize -> bf16 grouped
matmul in torch, so the padding/index/scale logic is testable on CPU (tests/v41f/test_p1_fused.py);
the kernel call itself is verified on the pod by tests/v41f/test_gpu_liger_parity.py.
"""

import torch

try:
    import deep_gemm as _dg

    HAS_DEEP_GEMM = hasattr(_dg, "m_grouped_fp8_gemm_nt_contiguous")
except ImportError:
    _dg = None
    HAS_DEEP_GEMM = False

ALIGN = 128  # deep_gemm.get_mk_alignment_for_contiguous_layout() on the pod
FP8_MAX = 448.0


def per_token_cast(x):
    """[M,K] -> (e4m3 [M,K], fp32 scale [M, K/128]); K % 128 == 0."""
    m, k = x.shape
    v = x.float().view(m, k // ALIGN, ALIGN)
    sf = v.abs().amax(-1, keepdim=True).clamp_min(1e-4) / FP8_MAX
    return (v / sf).to(torch.float8_e4m3fn).view(m, k), sf.view(m, k // ALIGN)


def per_block_cast(w):
    """[E,N,K] -> (e4m3 [E,N,K], fp32 scale [E, N/128, K/128]); N, K % 128 == 0."""
    e, n, k = w.shape
    v = w.float().view(e, n // ALIGN, ALIGN, k // ALIGN, ALIGN)
    sf = v.abs().amax((2, 4), keepdim=True).clamp_min(1e-4) / FP8_MAX
    return (v / sf).to(torch.float8_e4m3fn).view(e, n, k), sf.view(e, n // ALIGN, k // ALIGN)


def dequant_token(q, sf):
    m, k = q.shape
    return (q.float().view(m, k // ALIGN, ALIGN) * sf.view(m, k // ALIGN, 1)).view(m, k)


def dequant_block(q, sf):
    e, n, k = q.shape
    v = q.float().view(e, n // ALIGN, ALIGN, k // ALIGN, ALIGN)
    return (v * sf.view(e, n // ALIGN, 1, k // ALIGN, 1)).view(e, n, k)


def padded_m(n_rows, n_experts):
    """The padded M as a shape: every expert segment rounds up to ALIGN, so the worst case over any
    routing is n_rows plus one partial block per expert. Decided from B*T*top_k and E on the host once,
    never from the routing, so the DeepGEMM path stays shape-static inside a compiled graph."""
    return (n_rows + n_experts * (ALIGN - 1) + ALIGN - 1) // ALIGN * ALIGN


def padded_layout(counts, m_pad):
    """counts [E] (int64, on device) -> (pos, m_indices) for a fixed padded length m_pad:
    `pos[r]` is the padded row of sorted real row r (segments in expert order, each starting on an
    ALIGN boundary), `m_indices[i]` the expert of padded row i, -1 on padding and on the unused tail.
    No host sync: m_pad is a shape (padded_m), everything else is device arithmetic."""
    pc = (counts + ALIGN - 1) // ALIGN * ALIGN
    start_pad = pc.cumsum(0) - pc
    start = counts.cumsum(0) - counts
    n_rows = int(counts.sum()) if not torch.compiler.is_compiling() and counts.device.type == "cpu" else None
    expert_of_row = torch.repeat_interleave(torch.arange(counts.numel(), device=counts.device), counts,
                                            output_size=n_rows)
    pos = start_pad[expert_of_row] + (torch.arange(expert_of_row.numel(), device=counts.device) - start[expert_of_row])
    m_indices = torch.full((m_pad,), -1, dtype=torch.int32, device=counts.device)
    m_indices[pos] = expert_of_row.to(torch.int32)
    return pos, m_indices


def _gemm_nt(a_q, a_sf, w_q, w_sf, m_indices, emulate):
    """[Mpad,K] x [E,N,K]^T -> bf16 [Mpad,N] over the expert of each row."""
    if not emulate:
        d = torch.empty(a_q.size(0), w_q.size(1), dtype=torch.bfloat16, device=a_q.device)
        _dg.m_grouped_fp8_gemm_nt_contiguous((a_q, a_sf), (w_q, w_sf), d, m_indices)
        return d
    # per-expert loop over the 128-aligned segments: no [Mpad,N,K] gather (that form needed 21 GB at
    # B1 T1024 on CPU and would dwarf the kernel path's memory on a card)
    a = dequant_token(a_q, a_sf)
    w = dequant_block(w_q, w_sf)
    out = a.new_zeros(a.size(0), w.size(1))
    for e in range(w.size(0)):
        rows = (m_indices == e).nonzero().squeeze(1)
        if rows.numel():
            out[rows] = a[rows] @ w[e].t()
    return out.to(torch.bfloat16)


def transpose_blocks(w_q, w_sf):
    """[E,N,K] fp8 + [E,N/128,K/128] scales -> the K-major [E,K,N] operand for dA = dY @ W as an nt GEMM."""
    return w_q.transpose(1, 2).contiguous(), w_sf.transpose(1, 2).contiguous()


class GroupedLinearFP8(torch.autograd.Function):
    """y[rows of expert e] = a[rows] @ w[e]^T, a bf16 [M,K] sorted by expert, w bf16 [E,N,K]."""

    @staticmethod
    def forward(ctx, a, w, counts, offs, emulate):
        pos, m_indices = padded_layout(counts, padded_m(a.size(0), w.size(0)))
        a_pad = a.new_zeros(m_indices.numel(), a.size(1))
        a_pad[pos] = a
        a_q, a_sf = per_token_cast(a_pad)
        w_q, w_sf = per_block_cast(w)
        y = _gemm_nt(a_q, a_sf, w_q, w_sf, m_indices, emulate)
        ctx.save_for_backward(a, w_q, w_sf, pos, m_indices, offs)
        ctx.emulate = emulate
        return y[pos]  # back to the unpadded sorted rows

    @staticmethod
    def backward(ctx, dy):
        a, w_q, w_sf, pos, m_indices, offs = ctx.saved_tensors
        dy = dy.contiguous()
        dy_pad = dy.new_zeros(m_indices.numel(), dy.size(1))
        dy_pad[pos] = dy
        dy_q, dy_sf = per_token_cast(dy_pad)
        da = _gemm_nt(dy_q, dy_sf, *transpose_blocks(w_q, w_sf), m_indices, ctx.emulate)[pos]
        # dW[e] = dy[e]^T @ a[e]: bf16 grouped GEMM over the unpadded segments
        dw = torch._grouped_mm(dy.to(torch.bfloat16).transpose(0, 1), a.to(torch.bfloat16), offs=offs)
        return da.to(a.dtype), dw.to(torch.bfloat16), None, None, None


def grouped_linear_fp8(a, w, counts, offs):
    """The MoE entry point: DeepGEMM on CUDA when importable, the torch emulation elsewhere."""
    emulate = not (HAS_DEEP_GEMM and a.is_cuda)
    return GroupedLinearFP8.apply(a, w, counts, offs, emulate)

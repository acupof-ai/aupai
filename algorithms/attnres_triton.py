"""Triton kernels for fused AttnRes: the accumulator lives in registers, not HBM.

WHY THIS EXISTS RATHER THAN THE TORCH NODE. algorithms/attnres_fused.py is correct and
2.2x SLOWER than eager, measured: the online softmax rescales the whole [B,T,D] fp32
accumulator once per source, and in torch every rescale is a full HBM round trip
(+1.68 GB against 0.84 GB saved on v reads -- net LOSS). The whole design rests on that
accumulator staying in fast memory. Here it does: one program owns R rows, streams the
n sources, and touches HBM once per source read plus once for the final store.
docs/lessons/fused_attnres_is_slower_in_torch.md has the ledger.

NO COPY OF THE SOURCES, IN ANY SPELLING. The sources are separate bf16 allocations owned
by previous layers; the kernel reads them where they live, through a device array of
their data pointers, and converts to fp32 in registers. Two earlier versions copied and
both cost 13.3 GiB against eager's 1.75 at L=12: first torch.stack into [n,B,T,D], then
a list of per-source .float() views -- the same O(L^2) bytes, differently spelled. That
is the copy model.py:268 records as dominating at L=24. fp32 is a constraint on the
ACCUMULATOR, not on the loads, and reading it as a constraint on the loads is what
reintroduced the copy.

BLOCKING. R <= 16 rows per program (docs/standards/attnres_logits_kernel.md): the fp32
accumulator out[R,D] is 64 KB at R=16 and D=1024, and it must be fp32 -- a bf16
accumulator reads 4.2e-03 against a 1e-6 bar, four orders past failing. Precision and
block size are one constraint seen twice; widening R means dropping precision.

CONTRACT (verified in algorithms/attnres_fused.py against autograd, fp64, to 1e-12):
    logit[i,r] = <v_i[r], gq> * scale_i[r]
    a[:,r]     = softmax_i(logit[:,r])
    out[r,:]   = sum_i a[i,r] * v_i[r,:]
    dV_i    = a_i.dout + (dlogit_i * scale_i).gq     <- BOTH terms; dropping the second
    dlogit  = a_i * (dA_i - sum_j a_j dA_j)             reads as relative error 1.00
    dgq     = sum_i sum_r (dlogit_i * scale_i) * v_i
    dscale  = dlogit_i * <v_i, gq>

# ponytail: one program per row block, no warp specialisation, no TMA. The measurement
# that justifies more is the roofline gap after this lands, not before.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # Mac dev box: the reference and its gates still run
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _load_rows(PTRS, i, rows, d, D: tl.constexpr, rmask, BF16: tl.constexpr):
        """Source i's [R,D] tile as fp32, read from wherever that source lives."""
        p = tl.load(PTRS + i)
        if BF16:
            base = p.to(tl.pointer_type(tl.bfloat16))
        else:
            base = p.to(tl.pointer_type(tl.float32))
        return tl.load(base + rows[:, None] * D + d[None, :],
                       mask=rmask[:, None], other=0.0).to(tl.float32)

    @triton.jit
    def _store_rows(PTRS, i, rows, d, D: tl.constexpr, rmask, val, BF16: tl.constexpr):
        p = tl.load(PTRS + i)
        if BF16:
            base = p.to(tl.pointer_type(tl.bfloat16))
            tl.store(base + rows[:, None] * D + d[None, :], val.to(tl.bfloat16),
                     mask=rmask[:, None])
        else:
            base = p.to(tl.pointer_type(tl.float32))
            tl.store(base + rows[:, None] * D + d[None, :], val, mask=rmask[:, None])

    @triton.jit
    def _fwd_kernel(VP, GQ, S, OUTP, A, n_src, n_rows,
                    R: tl.constexpr, D: tl.constexpr, BF16: tl.constexpr):
        """One program = R rows, streaming every source. A comes back as the softmax
        weights, which backward needs and which cost D=1024x less to store than v."""
        pid = tl.program_id(0)
        rows = pid * R + tl.arange(0, R)
        rmask = rows < n_rows
        d = tl.arange(0, D)
        gq = tl.load(GQ + d).to(tl.float32)

        m = tl.full((R,), float("-inf"), tl.float32)
        ell = tl.zeros((R,), tl.float32)
        acc = tl.zeros((R, D), tl.float32)
        for i in range(n_src):
            v = _load_rows(VP, i, rows, d, D, rmask, BF16)
            s = tl.load(S + i * n_rows + rows, mask=rmask, other=0.0).to(tl.float32)
            logit = tl.where(rmask, tl.sum(v * gq[None, :], axis=1) * s, float("-inf"))
            m_new = tl.maximum(m, logit)
            # exp(m_old - m_new) with m_new >= m_old: the exponent is <= 0 by
            # construction, so this cannot overflow. First source: m is -inf, factor 0.
            rescale = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_new))
            p = tl.exp(logit - m_new)
            ell = ell * rescale + p
            acc = acc * rescale[:, None] + p[:, None] * v
            m = m_new
            tl.store(A + i * n_rows + rows, logit, mask=rmask)  # raw; normalised below

        _store_rows(OUTP, 0, rows, d, D, rmask, acc / ell[:, None], BF16)
        for i in range(n_src):
            lg = tl.load(A + i * n_rows + rows, mask=rmask, other=float("-inf"))
            tl.store(A + i * n_rows + rows, tl.exp(lg - m) / ell, mask=rmask)

    @triton.jit
    def _bwd_kernel(VP, GQ, S, A, DOUTP, DVP, DGQ, DS, n_src, n_rows,
                    R: tl.constexpr, D: tl.constexpr, BF16: tl.constexpr):
        """dV, dgq and dscale. dgq is [D] summed over every row, so it is accumulated
        per-program in registers and atomically added once at the end."""
        pid = tl.program_id(0)
        rows = pid * R + tl.arange(0, R)
        rmask = rows < n_rows
        d = tl.arange(0, D)
        gq = tl.load(GQ + d).to(tl.float32)
        dout = _load_rows(DOUTP, 0, rows, d, D, rmask, BF16)

        # dlogit needs every dA at this row before any one of them is final, so the
        # sources are traversed twice. The second pass hits L2; the alternative --
        # materialising n dA tensors -- is the copy this kernel exists to avoid.
        s_dA = tl.zeros((R,), tl.float32)
        for i in range(n_src):
            v = _load_rows(VP, i, rows, d, D, rmask, BF16)
            a = tl.load(A + i * n_rows + rows, mask=rmask, other=0.0)
            s_dA += a * tl.sum(dout * v, axis=1)

        dgq_local = tl.zeros((D,), tl.float32)
        for i in range(n_src):
            v = _load_rows(VP, i, rows, d, D, rmask, BF16)
            a = tl.load(A + i * n_rows + rows, mask=rmask, other=0.0)
            sc = tl.load(S + i * n_rows + rows, mask=rmask, other=0.0).to(tl.float32)
            dlogit = a * (tl.sum(dout * v, axis=1) - s_dA)
            w = dlogit * sc
            # BOTH terms: the mixing gradient, and the one flowing through this v's own
            # logit. Dropping the second matches the forward exactly and reads as
            # relative error 1.00 on dV -- which is why the gate is on dV.
            _store_rows(DVP, i, rows, d, D, rmask,
                        a[:, None] * dout + w[:, None] * gq[None, :], BF16)
            dgq_local += tl.sum(w[:, None] * v, axis=0)
            tl.store(DS + i * n_rows + rows, dlogit * tl.sum(v * gq[None, :], axis=1),
                     mask=rmask)
        tl.atomic_add(DGQ + d, dgq_local)


def _ptrs(tensors):
    """Device int64 array of data pointers -- the alternative to stacking the sources."""
    return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64,
                        device=tensors[0].device)


class TritonAttnRes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gq, n, *tensors):
        v, scale = list(tensors[:n]), list(tensors[n:])
        B, T, D = v[0].shape
        rows, dt = B * T, v[0].dtype
        assert all(x.is_contiguous() and x.dtype == dt for x in v), "sources: contiguous, one dtype"
        bf16 = dt == torch.bfloat16
        # Only the [n,rows] scales are stacked -- 1/D of the sources' bytes, so this is
        # not the copy the module docstring is about.
        s = torch.stack([x.detach().reshape(rows) for x in scale]).contiguous()
        out = torch.empty(rows, D, dtype=dt, device=v[0].device)
        A = torch.empty(n, rows, dtype=torch.float32, device=v[0].device)
        vp = _ptrs(v)
        _fwd_kernel[(triton.cdiv(rows, 16),)](vp, gq.detach().contiguous(), s, _ptrs([out]),
                                              A, n, rows, R=16, D=D, BF16=bf16)
        ctx.save_for_backward(s, A, gq.detach().contiguous(), *v)
        ctx.meta = (n, B, T, D, rows, bf16, scale[0].dtype, gq.dtype)
        return out.reshape(B, T, D)

    @staticmethod
    def backward(ctx, dout):
        s, A, gq, *v = ctx.saved_tensors
        n, B, T, D, rows, bf16, sdt, gdt = ctx.meta
        dv = [torch.empty_like(x) for x in v]
        dS = torch.empty_like(s)
        dgq = torch.zeros(D, dtype=torch.float32, device=A.device)
        dout = dout.contiguous().reshape(rows, D)
        _bwd_kernel[(triton.cdiv(rows, 16),)](_ptrs(v), gq, s, A, _ptrs([dout]), _ptrs(dv),
                                              dgq, dS, n, rows, R=16, D=D, BF16=bf16)
        return (dgq.to(gdt), None,
                *[x.reshape(B, T, D) for x in dv],
                *[dS[i].reshape(B, T, 1).to(sdt) for i in range(n)])


def triton_attn_res(v, gq, scale):
    """Same contract as algorithms.attnres_fused.fused_attn_res; scale must be LIVE."""
    if not HAS_TRITON:
        raise RuntimeError("triton is not installed; use algorithms.attnres_fused instead")
    return TritonAttnRes.apply(gq, len(v), *v, *scale)
